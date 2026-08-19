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
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from ..utils.segments import (
    _HEADER_CAPTION_RE,
    _Y_GAP_THRESH_PX,
    find_row_segments,
)

logger = logging.getLogger(__name__)

# 游离散文不应当作 caption：过长则保持为独立块
_MAX_CAPTION_CHARS = 80


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

    joined = " ".join(non_empty)
    caption_hits = [t for t in non_empty if _HEADER_CAPTION_RE.search(t)]
    # 碎片题注：一格是 [表1-2]，其余为短噪声（单字/数字/[表）
    fragmented = False
    if caption_hits and len(non_empty) > 3:
        others = [t for t in non_empty if t not in caption_hits]
        if others and all(
            len(t) <= 4 and (t.isdigit() or t in {"[", "]", "表", "[表", "1", "1 1"} or _HEADER_CAPTION_RE.search(t))
            for t in others
        ):
            fragmented = True
            # 取最完整的题注片段
            joined = max(caption_hits, key=len)

    if not _HEADER_CAPTION_RE.search(joined) and not fragmented:
        return "", cells
    if not fragmented:
        # 允许 1~3 个非空格（表题可能被切到相邻空列边）
        if len(non_empty) > 3:
            return "", cells
        # 其它非空内容不能太长（避免误伤真正表头行）
        if any(len(t) > 24 and not _HEADER_CAPTION_RE.search(t) for t in non_empty):
            return "", cells

    caption = joined.strip()
    # 过长「caption」实为结构失败后的散文粘连，勿提升为表题
    if len(caption) > _MAX_CAPTION_CHARS:
        return "", cells
    remaining = [c for c in cells if int(c["row_start"]) > 0]
    return caption, _renormalize_subtable(remaining)


def split_cells_into_subtables(
    cells: List[Dict[str, Any]],
    y_gap_thresh: float = _Y_GAP_THRESH_PX,
) -> List[List[Dict[str, Any]]]:
    """
    将可能混杂的多子表单元格列表，拆成多个独立子表。

    切分判据见 segments.find_row_segments；每个子表内 row_*/col_* 归一到 0。
    """
    if not cells:
        return []

    segments = find_row_segments(cells, y_gap_thresh=y_gap_thresh)
    if not segments:
        return [_renormalize_subtable(cells)]

    subtables: List[List[Dict[str, Any]]] = []
    for row_lo, row_hi in segments:
        group = [c for c in cells if row_lo <= int(c["row_start"]) <= row_hi]
        if not group:
            continue
        # 过滤全空子表（所有单元格无文本内容）
        if all(not str(c.get("text", "")).strip() for c in group):
            continue
        subtables.append(_renormalize_subtable(group))
    return subtables if subtables else [_renormalize_subtable(cells)]


def unroll_cells_to_grid(cells: List[Dict[str, Any]]) -> List[List[str]]:
    """
    将带逻辑拓扑（row_start/row_end/col_start/col_end）的单元格列表，
    展开（Unroll）为一张规整的 N 行 x M 列字符串矩阵。

    核心逻辑："仅起始格保留文本"：
    - 单元格覆盖的逻辑范围是闭区间 [row_start, row_end] x [col_start, col_end]；
    - 【BugFix】仅在起始坐标 (row_start, col_start) 写入真实 text，
      其余被跨越的子格写入空字符串 ""，避免合并单元格语义膨胀破坏 RAG 词频；
    - 展开后矩阵仍严格 N x M，任意一行的列数都相同，消除列错位。
    - 逻辑重叠时：后到的非空文本并入已有格，并 warning，避免静默丢字。
    """
    if not cells:
        return []

    # 先做与 HTML 侧一致的冲突消解（延迟导入避免循环）
    try:
        from .html_formatter import _resolve_logic_overlaps

        cells = _resolve_logic_overlaps([dict(c) for c in cells])
    except Exception:  # noqa: BLE001
        cells = list(cells)

    max_row = max(int(c["row_end"]) for c in cells)
    max_col = max(int(c["col_end"]) for c in cells)
    n_rows = max_row + 1
    n_cols = max_col + 1

    grid: List[List[str]] = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    filled: List[List[bool]] = [[False for _ in range(n_cols)] for _ in range(n_rows)]
    owner: List[List[Tuple[int, int] | None]] = [
        [None for _ in range(n_cols)] for _ in range(n_rows)
    ]

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
                # 仅起始格写真实文本
                cell_text = text if (r == row_start and c == col_start) else ""
                if filled[r][c] and grid[r][c] and cell_text and grid[r][c] != cell_text:
                    # 并入已有，不覆盖丢失
                    if cell_text not in grid[r][c]:
                        grid[r][c] = (grid[r][c] + " " + cell_text).strip()
                    logger.warning(
                        "检测到单元格逻辑范围重叠，文本已合并: (%s, %s) %r",
                        r, c, cell_text[:40],
                    )
                    continue
                if filled[r][c] and not cell_text:
                    continue
                # 起点被其它 span 覆盖：把文本并入 owner 起点
                if (
                    cell_text
                    and filled[r][c]
                    and owner[r][c] is not None
                    and owner[r][c] != (row_start, col_start)
                ):
                    or_, oc_ = owner[r][c]  # type: ignore[misc]
                    prev = grid[or_][oc_]
                    if cell_text not in prev:
                        grid[or_][oc_] = (prev + " " + cell_text).strip() if prev else cell_text
                        logger.warning(
                            "覆盖位文本并入起点 (%s,%s): %r",
                            or_,
                            oc_,
                            cell_text[:40],
                        )
                    continue
                if not filled[r][c]:
                    grid[r][c] = cell_text
                    filled[r][c] = True
                    owner[r][c] = (row_start, col_start)
                elif cell_text and not grid[r][c]:
                    grid[r][c] = cell_text

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
    若无有效单元格文本（结构失败），将 free_texts 作为独立散文块，避免与空表粘连。
    """
    prefix = format_free_texts(free_texts)
    has_cell_text = any(str(c.get("text") or "").strip() for c in cells)
    if not has_cell_text:
        return prefix or ""

    table = cells_to_markdown_table(
        cells,
        split_subtables=split_subtables,
        compress_empty_cols=compress_empty_cols,
    )

    if prefix and table:
        return prefix + "\n\n" + table
    return prefix or table or ""
