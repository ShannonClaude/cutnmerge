"""基于逻辑拓扑的 Markdown 表格生成（合并单元格 Unrolling + 多子表拆分）。

【Phase 2 重构说明 —— 解决合并单元格导致的列错位问题】
标准 Markdown 表格语法本身**不支持** rowspan/colspan（合并单元格），一旦
表格中存在跨行或跨列的单元格，旧版实现（按 x_key/y_key 几何位置分组成行）
会导致某一行的列数与其他行不一致，下游渲染或 RAG 切片时极易发生"列错位"：
后面的单元格文本会被错误地对齐到相邻列。

本文件采用"展开为规整网格 + 仅起始格保留文本"策略解决该问题：
- 对每个逻辑单元格，若其 row_span > 1 或 col_span > 1（即跨行/跨列），
  仍按逻辑范围占满网格，保证矩阵严格为 N 行 x M 列；
- 【BugFix】真实 text 仅写入起始坐标 (row_start, col_start)，其余被跨越的
  子格填入空字符串，避免把同一段文本复制到每个子格导致 RAG 词频膨胀。

【四级缺陷重构 —— 单图多子表混杂】
一张图片上若上下排列多个独立子表（如 [表 1-1]、[表 1-2]），LORE 可能把它们
预测进同一套逻辑网格。展开前按物理 Y 间距 / 表头特征切分为多个子表，
各自 unroll 成独立 Markdown，中间用空行分隔。
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 相邻逻辑行物理 Y 间距超过此阈值则切分为新子表
_Y_GAP_THRESH_PX = 40.0
# 专利表题特征，例如 [表 1-2]、表1-1
_HEADER_CAPTION_RE = re.compile(r"\[?\s*表\s*[\d\-]+")
# 中段换头：再次出现「聚合物」类列名（P97/P98 上下两段）
_SECTION_HEADER_RE = re.compile(r"(聚合物|单体\s*[\[［]|单体\[)")


def _cell_y_top(cell: Dict[str, Any]) -> float:
    """单元格物理顶部 Y（优先 y_key，否则 polygon 最小 Y）。"""
    if cell.get("y_key") is not None:
        return float(cell["y_key"])
    poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
    return float(poly[:, 1].min())


def _cell_y_bottom(cell: Dict[str, Any]) -> float:
    """单元格物理底部 Y。"""
    poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
    return float(poly[:, 1].max())


def _renormalize_subtable(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将子表内 row_* / col_* 再归一到从 0 开始。

    使用浅拷贝，避免修改原始 cell dict，防止多次调用 split 时
    索引被反复改写导致不同子表行号互相污染。
    """
    if not cells:
        return []
    copied = [dict(c) for c in cells]
    min_row = min(int(c["row_start"]) for c in copied)
    min_col = min(int(c["col_start"]) for c in copied)
    for c in copied:
        c["row_start"] = int(c["row_start"]) - min_row
        c["row_end"] = int(c["row_end"]) - min_row
        c["col_start"] = int(c["col_start"]) - min_col
        c["col_end"] = int(c["col_end"]) - min_col
        c["row_span"] = int(c["row_end"]) - int(c["row_start"]) + 1
        c["col_span"] = int(c["col_end"]) - int(c["col_start"]) + 1
    return copied

def _row_has_header_caption(row_cells: List[Dict[str, Any]]) -> bool:
    """该逻辑行文本是否呈现专利子表题特征（如 [表 1-2]）。"""
    joined = " ".join(str(c.get("text") or "") for c in row_cells)
    return bool(_HEADER_CAPTION_RE.search(joined))


def _row_has_section_header(row_cells: List[Dict[str, Any]]) -> bool:
    """中段换头：再次出现聚合物/单体列名。"""
    joined = " ".join(str(c.get("text") or "") for c in row_cells)
    return bool(_SECTION_HEADER_RE.search(joined))


def extract_caption_row(
    cells: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    若第 0 行几乎只有「表 N」类表题、其余为空，则提升为前缀并从表中删除该行。

    Returns:
        (caption_text, remaining_cells_renormalized)
    """
    if not cells:
        return "", []

    by_row: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_row[int(cell["row_start"])].append(cell)
    if 0 not in by_row:
        return "", cells

    row0 = by_row[0]
    texts = [str(c.get("text") or "").strip() for c in row0]
    non_empty = [t for t in texts if t]
    if not non_empty:
        return "", cells
    # 非空单元格都应像表题，或整行拼接后是表题且非空格子很少
    joined = " ".join(non_empty)
    if not _HEADER_CAPTION_RE.search(joined):
        return "", cells
    # 允许 1~2 个非空格（表题可能被切到相邻空列边）
    if len(non_empty) > 3:
        return "", cells
    # 其它非空内容不能太长（避免误伤真正表头行）
    if any(len(t) > 24 and not _HEADER_CAPTION_RE.search(t) for t in non_empty):
        return "", cells

    caption = joined.strip()
    remaining = [c for c in cells if int(c["row_start"]) > 0]
    return caption, _renormalize_subtable(remaining)


def split_cells_into_subtables(
    cells: List[Dict[str, Any]],
    y_gap_thresh: float = _Y_GAP_THRESH_PX,
) -> List[List[Dict[str, Any]]]:
    """
    将可能混杂的多子表单元格列表，拆成多个独立子表。

    切分条件（按 row_start 升序扫描相邻逻辑行）：
    1. 物理 Y 间距：下一行顶部 Y - 上一行底部 Y > y_gap_thresh（默认 40px）；
    2. 表头特征：该行文本匹配「表 N」类标题，且不是当前子表首行；
    3. 中段换头：非首行再次出现「聚合物 / 单体」类列名（框线路径单表两段）。

    每个子表内会重新将 row_* / col_* 归一到从 0 开始，便于独立 unroll。
    """
    if not cells:
        return []

    # 按 row_start 分组
    by_row: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_row[int(cell["row_start"])].append(cell)

    sorted_rows = sorted(by_row.keys())
    if not sorted_rows:
        return [cells]

    # 每逻辑行的物理 Y 范围与单元格
    row_meta: List[Tuple[int, float, float, List[Dict[str, Any]]]] = []
    for rs in sorted_rows:
        row_cells = by_row[rs]
        y_top = min(_cell_y_top(c) for c in row_cells)
        y_bot = max(_cell_y_bottom(c) for c in row_cells)
        row_meta.append((rs, y_top, y_bot, row_cells))

    subtables: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    prev_y_bot: float | None = None
    is_first_row_of_subtable = True

    for _, y_top, y_bot, row_cells in row_meta:
        should_split = False
        if current and prev_y_bot is not None:
            # 条件 1：相邻逻辑行物理空隙过大
            if (y_top - prev_y_bot) > y_gap_thresh:
                should_split = True
            # 条件 2：非首行再次出现表题特征
            elif (not is_first_row_of_subtable) and _row_has_header_caption(row_cells):
                should_split = True
            # 条件 3：中段再次出现聚合物/单体表头
            elif (not is_first_row_of_subtable) and _row_has_section_header(row_cells):
                should_split = True

        if should_split:
            subtables.append(_renormalize_subtable(current))
            current = []
            is_first_row_of_subtable = True

        current.extend(row_cells)
        prev_y_bot = y_bot
        is_first_row_of_subtable = False

    if current:
        subtables.append(_renormalize_subtable(current))

    return subtables if subtables else [cells]


def unroll_cells_to_grid(cells: List[Dict[str, Any]]) -> List[List[str]]:
    """
    将带逻辑拓扑（row_start/row_end/col_start/col_end）的单元格列表，
    展开（Unroll）为一张规整的 N 行 x M 列字符串矩阵。

    核心逻辑："仅起始格保留文本"：
    - 单元格覆盖的逻辑范围是闭区间 [row_start, row_end] x [col_start, col_end]；
    - 【BugFix】仅在起始坐标 (row_start, col_start) 写入真实 text，
      其余被跨越的子格写入空字符串 ""，避免合并单元格语义膨胀破坏 RAG 词频；
    - 展开后矩阵仍严格 N x M，任意一行的列数都相同，消除列错位。

    矩阵尺寸：
    - 行数 = 所有单元格 row_end 的最大值 + 1；
    - 列数 = 所有单元格 col_end 的最大值 + 1；
    - （models.predict_cells / 子表归一化已把逻辑坐标归一化为从 0 开始）。

    鲁棒性：
    - 若单元格拓扑坐标越界（异常数据），跳过越界部分并记录警告，不中断整体流程；
    - 若多个单元格的逻辑范围发生重叠（理论上 LORE 输出不应重叠），后写入的
      单元格会覆盖先写入的文本；只在非空覆盖非空时记录警告，便于排查模型异常。
    """
    if not cells:
        return []

    max_row = max(int(c["row_end"]) for c in cells)
    max_col = max(int(c["col_end"]) for c in cells)
    n_rows = max_row + 1
    n_cols = max_col + 1

    grid: List[List[str]] = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    filled: List[List[bool]] = [[False for _ in range(n_cols)] for _ in range(n_rows)]

    for cell in cells:
        text = str(cell.get("text") or "").replace("\n", " ").strip()
        row_start = int(cell["row_start"])
        row_end = int(cell["row_end"])
        col_start = int(cell["col_start"])
        col_end = int(cell["col_end"])

        for r in range(row_start, row_end + 1):
            if r < 0 or r >= n_rows:
                logger.warning("单元格逻辑行坐标越界，已跳过: r=%s (0~%s)", r, n_rows - 1)
                continue
            for c in range(col_start, col_end + 1):
                if c < 0 or c >= n_cols:
                    logger.warning("单元格逻辑列坐标越界，已跳过: c=%s (0~%s)", c, n_cols - 1)
                    continue
                # 【BugFix】合并单元格仅在起始坐标填入真实文本，其余子格留空，避免 RAG 词频膨胀
                cell_text = text if (r == row_start and c == col_start) else ""
                if filled[r][c] and grid[r][c] and cell_text and grid[r][c] != cell_text:
                    logger.warning(
                        "检测到单元格逻辑范围重叠覆盖: (%s, %s) 原值=%r 新值=%r",
                        r, c, grid[r][c], cell_text,
                    )
                grid[r][c] = cell_text
                filled[r][c] = True

    return grid


def _grid_to_markdown(grid: List[List[str]]) -> str:
    """单张规整矩阵 → 标准 Markdown 表格字符串。"""
    if not grid:
        return ""

    def escape(text: str) -> str:
        return text.replace("|", "\\|").replace("\n", " ")

    def fmt_row(cols: List[str]) -> str:
        return "| " + " | ".join(escape(c) for c in cols) + " |"

    n_cols = len(grid[0])
    header = grid[0]
    sep = ["---"] * n_cols
    lines = [fmt_row(header), fmt_row(sep)]
    for data_row in grid[1:]:
        lines.append(fmt_row(data_row))

    # 只有一行时也输出表头+分隔，便于下游 RAG 解析
    return "\n".join(lines)


def _is_empty_grid(grid: List[List[str]]) -> bool:
    """整张矩阵是否全空。"""
    return all(not (cell or "").strip() for row in grid for cell in row)


def compress_empty_spanned_columns(grid: List[List[str]]) -> List[List[str]]:
    """
    压缩「整列为空」的列。

    用于抵消偶发半高竖线造成的多余空列。若某列所有行都为空字符串，则删除该列。
    若删完后没有列，返回原 grid。
    """
    if not grid or not grid[0]:
        return grid
    n_cols = len(grid[0])
    keep: List[int] = []
    for c in range(n_cols):
        if any((row[c] or "").strip() for row in grid):
            keep.append(c)
    if not keep or len(keep) == n_cols:
        return grid
    return [[row[c] for c in keep] for row in grid]


def compress_empty_rows(grid: List[List[str]]) -> List[List[str]]:
    """删除整行全空的行（对称于 compress_empty_spanned_columns）。"""
    if not grid:
        return grid
    kept = [row for row in grid if any((c or "").strip() for c in row)]
    return kept if kept else grid


def cells_to_markdown_table(
    cells: List[Dict[str, Any]],
    *,
    split_subtables: bool = True,
    compress_empty_cols: bool = True,
    lift_captions: bool = True,
) -> str:
    """
    将带逻辑拓扑的单元格展开为规整矩阵后，生成标准 Markdown 表格。

    - split_subtables=True：按物理 Y 间距 / 表题 / 中段表头拆子表；
    - split_subtables=False：直接展开单表（仍可 lift 表题行）；
    - lift_captions：第 0 行若几乎只有「表 N」则提到表格上方；
    - 丢弃全空子表；
    - compress_empty_cols：删除整列为空的列，并压缩整行全空。
    """
    if split_subtables:
        subtables = split_cells_into_subtables(cells)
    else:
        subtables = [cells] if cells else []

    md_parts: List[str] = []
    for sub in subtables:
        caption = ""
        work = sub
        if lift_captions:
            caption, work = extract_caption_row(work)
        # 子表内若所有 cell 文本为空，跳过
        if not any(str(c.get("text") or "").strip() for c in work):
            if caption:
                md_parts.append(caption)
            continue
        grid = unroll_cells_to_grid(work)
        if compress_empty_cols:
            grid = compress_empty_spanned_columns(grid)
            grid = compress_empty_rows(grid)
        if _is_empty_grid(grid):
            if caption:
                md_parts.append(caption)
            continue
        md = _grid_to_markdown(grid)
        if caption and md:
            md_parts.append(caption + "\n\n" + md)
        elif md:
            md_parts.append(md)
        elif caption:
            md_parts.append(caption)
    return "\n\n".join(md_parts)


def format_free_texts(free_texts: List[Dict[str, Any]]) -> str:
    """游离文本按阅读顺序（先 Y 后 X）拼接为前缀段落。"""
    if not free_texts:
        return ""

    def sort_key(tb: Dict[str, Any]):
        tl = tb.get("top_left")
        if tl is not None:
            return (float(tl[1]), float(tl[0]))
        return (0.0, 0.0)

    ordered = sorted(free_texts, key=sort_key)
    return "\n".join(tb["text"] for tb in ordered if tb.get("text"))


def build_markdown_output(
    cells: List[Dict[str, Any]],
    free_texts: List[Dict[str, Any]],
    *,
    split_subtables: bool = True,
    compress_empty_cols: bool = True,
) -> str:
    """
    将表格外部游离文本作为前缀，拼接在 Markdown 表格上方。

    若检测到多子表，cells_to_markdown_table 会输出多张用空行分隔的表格。
    """
    prefix = format_free_texts(free_texts)
    table = cells_to_markdown_table(
        cells,
        split_subtables=split_subtables,
        compress_empty_cols=compress_empty_cols,
    )

    if prefix and table:
        return prefix + "\n\n" + table
    return prefix or table or ""
