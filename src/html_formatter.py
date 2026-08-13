"""基于逻辑拓扑的 HTML 表格生成（保留 rowspan/colspan）。

与 Markdown unroll 不同：合并单元格在起始格输出带 span 的 <td>，
被覆盖的子格跳过，不展开为空格子。
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from .formatter import (
    extract_caption_row,
    format_free_texts,
    split_cells_into_subtables,
)
from .label_patterns import is_index_column

logger = logging.getLogger(__name__)

# 噪声行：整行仅剩孤立字母/符号（如末行 "L"）
_NOISE_CELL_RE = re.compile(r"^[A-Za-z]$|^[·•\.\,;:]$")


def _cell_key(cell: Dict[str, Any]) -> Tuple[int, int, int, int]:
    return (
        int(cell["row_start"]),
        int(cell["row_end"]),
        int(cell["col_start"]),
        int(cell["col_end"]),
    )


def _occupied_columns(cells: List[Dict[str, Any]]) -> Set[int]:
    """逻辑上被任意 cell 覆盖的列索引。"""
    cols: Set[int] = set()
    for c in cells:
        for col in range(int(c["col_start"]), int(c["col_end"]) + 1):
            cols.add(col)
    return cols


def _occupied_rows(cells: List[Dict[str, Any]]) -> Set[int]:
    rows: Set[int] = set()
    for c in cells:
        for row in range(int(c["row_start"]), int(c["row_end"]) + 1):
            rows.add(row)
    return rows


def compress_empty_logic_columns(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    删除「没有任何 cell 覆盖」的幽灵逻辑列，并重映射 col_*。

    与 MD 的整列文本为空压缩不同：这里按拓扑占用判断，避免误删
    仅作 rowspan/colspan 占位、文本在起始格的合并列。
    """
    if not cells:
        return cells
    occupied = sorted(_occupied_columns(cells))
    if not occupied:
        return cells
    max_col = max(int(c["col_end"]) for c in cells)
    all_cols = list(range(max_col + 1))
    if occupied == all_cols:
        return cells

    remap = {old: new for new, old in enumerate(occupied)}
    out: List[Dict[str, Any]] = []
    for c in cells:
        nc = dict(c)
        cs, ce = int(c["col_start"]), int(c["col_end"])
        # 覆盖区间映射到压缩后连续区间
        new_idxs = [remap[i] for i in range(cs, ce + 1) if i in remap]
        if not new_idxs:
            continue
        nc["col_start"] = min(new_idxs)
        nc["col_end"] = max(new_idxs)
        nc["col_span"] = nc["col_end"] - nc["col_start"] + 1
        out.append(nc)
    return out


def compress_empty_logic_rows(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """删除没有任何 cell 覆盖的幽灵逻辑行。"""
    if not cells:
        return cells
    occupied = sorted(_occupied_rows(cells))
    if not occupied:
        return cells
    max_row = max(int(c["row_end"]) for c in cells)
    if occupied == list(range(max_row + 1)):
        return cells

    remap = {old: new for new, old in enumerate(occupied)}
    out: List[Dict[str, Any]] = []
    for c in cells:
        nc = dict(c)
        rs, re = int(c["row_start"]), int(c["row_end"])
        new_idxs = [remap[i] for i in range(rs, re + 1) if i in remap]
        if not new_idxs:
            continue
        nc["row_start"] = min(new_idxs)
        nc["row_end"] = max(new_idxs)
        nc["row_span"] = nc["row_end"] - nc["row_start"] + 1
        out.append(nc)
    return out


def _resolve_logic_overlaps(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    渲染前消解逻辑矩形重叠：后到的非空文本并入占位格，并裁掉冲突覆盖。

    避免 `covered` 静默丢字。
    """
    if not cells:
        return cells

    # 按面积升序：先处理小格（更具体）
    ordered = sorted(cells, key=lambda c: (
        (int(c["row_end"]) - int(c["row_start"]) + 1)
        * (int(c["col_end"]) - int(c["col_start"]) + 1),
        int(c["row_start"]),
        int(c["col_start"]),
    ))

    occupancy: Dict[Tuple[int, int], Dict[str, Any]] = {}
    out: List[Dict[str, Any]] = []

    for cell in ordered:
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        text = str(cell.get("text") or "").strip()
        conflict_owners: List[Dict[str, Any]] = []
        free_positions: List[Tuple[int, int]] = []
        for r in range(rs, re + 1):
            for c in range(cs, ce + 1):
                owner = occupancy.get((r, c))
                if owner is None:
                    free_positions.append((r, c))
                elif owner is not cell:
                    conflict_owners.append(owner)

        if not conflict_owners:
            # 无冲突：登记占位
            nc = dict(cell)
            out.append(nc)
            for r in range(rs, re + 1):
                for c in range(cs, ce + 1):
                    occupancy[(r, c)] = nc
            continue

        if text:
            # 把文本并入第一个冲突占位格
            owner = conflict_owners[0]
            prev = str(owner.get("text") or "").strip()
            if text and text not in prev:
                owner["text"] = (prev + " " + text).strip() if prev else text
                logger.warning(
                    "逻辑格重叠，文本并入占位格 (%s,%s): %r",
                    owner.get("row_start"),
                    owner.get("col_start"),
                    text[:40],
                )
            # 若仍有空位，用空位重建一个缩减格（无文本，仅占位）
            if free_positions:
                frs = min(p[0] for p in free_positions)
                fre = max(p[0] for p in free_positions)
                fcs = min(p[1] for p in free_positions)
                fce = max(p[1] for p in free_positions)
                # 仅当缩减区域是矩形全覆盖时保留
                needed = {
                    (r, c)
                    for r in range(frs, fre + 1)
                    for c in range(fcs, fce + 1)
                }
                if needed.issubset(set(free_positions)):
                    nc = dict(cell)
                    nc["row_start"], nc["row_end"] = frs, fre
                    nc["col_start"], nc["col_end"] = fcs, fce
                    nc["row_span"] = fre - frs + 1
                    nc["col_span"] = fce - fcs + 1
                    nc["text"] = ""
                    out.append(nc)
                    for r, c in free_positions:
                        if (r, c) in needed:
                            occupancy[(r, c)] = nc
            continue

        # 空文本冲突格：若有空位则缩减，否则丢弃
        if free_positions:
            frs = min(p[0] for p in free_positions)
            fre = max(p[0] for p in free_positions)
            fcs = min(p[1] for p in free_positions)
            fce = max(p[1] for p in free_positions)
            needed = {
                (r, c)
                for r in range(frs, fre + 1)
                for c in range(fcs, fce + 1)
            }
            if needed.issubset(set(free_positions)):
                nc = dict(cell)
                nc["row_start"], nc["row_end"] = frs, fre
                nc["col_start"], nc["col_end"] = fcs, fce
                nc["row_span"] = fre - frs + 1
                nc["col_span"] = fce - fcs + 1
                out.append(nc)
                for r, c in needed:
                    occupancy[(r, c)] = nc
        # else: 完全被覆盖的空格，丢弃

    return out


def _merge_leading_label_gaps(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge leading empty placeholders into the first label cell of a row."""
    if not cells:
        return cells
    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for cell in cells:
        if int(cell["row_start"]) != int(cell["row_end"]):
            continue
        by_row.setdefault(int(cell["row_start"]), []).append(cell)

    for row_cells in by_row.values():
        row_cells.sort(key=lambda c: int(c["col_start"]))
        if len(row_cells) < 2:
            continue
        first = row_cells[0]
        if int(first["col_span"]) != 1 or not str(first.get("text") or "").strip():
            continue
        merge_until = int(first["col_end"])
        absorbed: List[Dict[str, Any]] = []
        for cell in row_cells[1:]:
            if int(cell["col_span"]) != 1:
                break
            if str(cell.get("text") or "").strip():
                break
            merge_until = int(cell["col_end"])
            absorbed.append(cell)
        if not absorbed:
            continue
        first["col_end"] = merge_until
        first["col_span"] = int(first["col_end"]) - int(first["col_start"]) + 1
        for cell in absorbed:
            cell["_drop_render"] = True
    return [c for c in cells if not c.get("_drop_render")]


def _cell_physical_bbox_x_y(cell: Dict[str, Any]) -> Tuple[float, float, float, float]:
    poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
    return float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max())


def drop_evidenceless_columns(cells: List[Dict[str, Any]], *, narrow_ratio: float = 0.35) -> List[Dict[str, Any]]:
    """
    丢弃“证据不足”的幽灵列（零文本、且几何上非常窄、且不参与跨列 colspan）。
    窄列若仅有短碎片文本、且不参与 colspan，同样丢弃。

    注意：此实现无法直接判定“无框线/无墨迹”（html_formatter 不接收原图），
    因此使用可观测代理：窄宽度 + 不存在跨列占用 + 覆盖该列的文本为空/碎片。
    """
    if not cells:
        return cells

    def _is_frag(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        compact = "".join(t.split())
        return len(compact) <= 2

    # 物理宽度统计：用所有 col_span==1 的 cell 估计“标准列宽”
    atomic_widths: List[float] = []
    for c in cells:
        if int(c.get("col_span") or 1) == 1:
            x1, _y1, x2, _y2 = _cell_physical_bbox_x_y(c)
            atomic_widths.append(max(0.0, x2 - x1))
    if not atomic_widths:
        return cells
    median_w = float(np.median(atomic_widths))
    if median_w <= 1e-6:
        return cells

    max_col = max(int(c["col_end"]) for c in cells)
    dropped: Set[int] = set()

    # 先统计每列被哪些 cell 覆盖，以及这些 cell 是否带文本/跨列
    col_to_cells: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(max_col + 1)}
    for c in cells:
        cs, ce = int(c["col_start"]), int(c["col_end"])
        for col in range(cs, ce + 1):
            col_to_cells[col].append(c)

    for col, covered_cells in col_to_cells.items():
        if not covered_cells:
            continue
        texts = [str(c.get("text") or "").strip() for c in covered_cells]
        any_text = any(texts)
        # 整列多为行序号（如 16–30）：窄但仍是真实列，保留
        if is_index_column(texts):
            continue
        only_frag = (not any_text) or all(_is_frag(t) for t in texts if t) or (
            any_text and all(_is_frag(t) for t in texts)
        )
        spans_cross = any(int(c.get("col_span") or 1) > 1 for c in covered_cells)
        if spans_cross and any(not _is_frag(t) for t in texts):
            continue

        # 该列的窄宽度判断：取其原子 cell 的宽度中位数
        widths: List[float] = []
        for c in covered_cells:
            if int(c.get("col_span") or 1) != 1:
                continue
            x1, _y1, x2, _y2 = _cell_physical_bbox_x_y(c)
            widths.append(max(0.0, x2 - x1))
        if not widths:
            # 无原子格宽度时，若整列只有碎片文本且无跨列，仍丢弃
            if only_frag and not spans_cross:
                dropped.add(col)
            continue
        narrow = float(np.median(widths)) < narrow_ratio * median_w
        # 窄列碎片，或非窄但整列仅孤立数字/单字且多数格为空
        empty_n = sum(1 for t in texts if not t)
        frag_only_sparse = only_frag and empty_n >= max(1, len(texts) // 2)
        if (narrow and only_frag) or (frag_only_sparse and not spans_cross):
            # 碎片文本并入左侧邻列（若存在）
            if any_text and col > 0:
                frag_bits = [t for t in texts if t]
                for c in cells:
                    if int(c["col_start"]) <= col - 1 <= int(c["col_end"]) and int(
                        c.get("row_start", 0)
                    ) == min(int(x.get("row_start", 0)) for x in covered_cells):
                        prev = str(c.get("text") or "").strip()
                        add = "".join(frag_bits)
                        if add and add not in prev:
                            c["text"] = (prev + add).strip() if prev else add
                        break
            dropped.add(col)

    if not dropped:
        return cells

    kept_cols = [i for i in range(max_col + 1) if i not in dropped]
    if not kept_cols:
        return cells

    remap = {old: new for new, old in enumerate(kept_cols)}
    out: List[Dict[str, Any]] = []
    for c in cells:
        cs, ce = int(c["col_start"]), int(c["col_end"])
        # 如果 cell 只覆盖被丢弃列，则丢弃
        new_idxs = [remap[i] for i in range(cs, ce + 1) if i in remap]
        if not new_idxs:
            continue
        nc = dict(c)
        nc["col_start"] = min(new_idxs)
        nc["col_end"] = max(new_idxs)
        nc["col_span"] = nc["col_end"] - nc["col_start"] + 1
        out.append(nc)
    return out


def drop_evidenceless_rows(cells: List[Dict[str, Any]], *, short_ratio: float = 0.35) -> List[Dict[str, Any]]:
    """与 `drop_evidenceless_columns` 对称：窄行/零文本/不参与跨行 rowspan。"""
    if not cells:
        return cells

    atomic_heights: List[float] = []
    for c in cells:
        if int(c.get("row_span") or 1) == 1:
            _x1, y1, _x2, y2 = _cell_physical_bbox_x_y(c)
            atomic_heights.append(max(0.0, y2 - y1))
    if not atomic_heights:
        return cells
    median_h = float(np.median(atomic_heights))
    if median_h <= 1e-6:
        return cells

    max_row = max(int(c["row_end"]) for c in cells)
    dropped: Set[int] = set()

    row_to_cells: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(max_row + 1)}
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        for row in range(rs, re + 1):
            row_to_cells[row].append(c)

    for row, covered_cells in row_to_cells.items():
        if not covered_cells:
            continue
        any_text = any(str(c.get("text") or "").strip() for c in covered_cells)
        if any_text:
            continue

        heights: List[float] = []
        for c in covered_cells:
            if int(c.get("row_span") or 1) != 1:
                continue
            _x1, y1, _x2, y2 = _cell_physical_bbox_x_y(c)
            heights.append(max(0.0, y2 - y1))
        if not heights:
            continue
        if float(np.median(heights)) < short_ratio * median_h:
            dropped.add(row)

    if not dropped:
        return cells

    kept_rows = [i for i in range(max_row + 1) if i not in dropped]
    if not kept_rows:
        return cells

    remap = {old: new for new, old in enumerate(kept_rows)}
    out: List[Dict[str, Any]] = []
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        new_idxs = [remap[i] for i in range(rs, re + 1) if i in remap]
        if not new_idxs:
            continue
        nc = dict(c)
        nc["row_start"] = min(new_idxs)
        nc["row_end"] = max(new_idxs)
        nc["row_span"] = nc["row_end"] - nc["row_start"] + 1
        out.append(nc)
    return out


def drop_noise_rows(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    丢弃几乎全空、仅含孤立噪声字符的逻辑行（如表末幻觉 'L'）。

    不丢弃含数字、多字中文或跨行 rowspan 起点的行。
    """
    if not cells:
        return cells

    max_row = max(int(c["row_end"]) for c in cells)
    by_origin_row: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(max_row + 1)}
    for c in cells:
        by_origin_row[int(c["row_start"])].append(c)

    dropped: Set[int] = set()
    for row, row_cells in by_origin_row.items():
        if not row_cells:
            continue
        # 有跨多行的起点格 → 保留
        if any(int(c.get("row_span") or 1) > 1 for c in row_cells):
            continue
        texts = [str(c.get("text") or "").strip() for c in row_cells]
        non_empty = [t for t in texts if t]
        if not non_empty:
            # 全空行留给 compress_empty_logic_rows
            continue
        if len(non_empty) <= 2 and all(_NOISE_CELL_RE.fullmatch(t) for t in non_empty):
            dropped.add(row)

    if not dropped:
        return cells

    kept_rows = [i for i in range(max_row + 1) if i not in dropped]
    if not kept_rows:
        return cells

    remap = {old: new for new, old in enumerate(kept_rows)}
    out: List[Dict[str, Any]] = []
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        new_idxs = [remap[i] for i in range(rs, re + 1) if i in remap]
        if not new_idxs:
            continue
        nc = dict(c)
        nc["row_start"] = min(new_idxs)
        nc["row_end"] = max(new_idxs)
        nc["row_span"] = nc["row_end"] - nc["row_start"] + 1
        out.append(nc)
    logger.info("丢弃噪声行: %s", sorted(dropped))
    return out


def _escape_cell_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # 保留格内换行
    parts = t.split("\n")
    return "<br>".join(html.escape(p) for p in parts)


def cells_to_html_table(
    cells: List[Dict[str, Any]],
    *,
    compress_empty: bool = True,
) -> str:
    """
    将带逻辑拓扑的单元格列表渲染为单个 <table>。

    占位矩阵：起始格写 td（带 rowspan/colspan），被 span 覆盖的位置跳过。
    """
    if not cells:
        return ""

    work = [dict(c) for c in cells]
    work = _resolve_logic_overlaps(work)
    # 去掉原先只处理“行首标签”的非对称美化；改为对称幽灵行/列清理
    work = drop_evidenceless_columns(work)
    work = drop_evidenceless_rows(work)
    work = drop_noise_rows(work)
    if compress_empty:
        work = compress_empty_logic_columns(work)
        work = compress_empty_logic_rows(work)
    if not work:
        return ""

    max_row = max(int(c["row_end"]) for c in work)
    max_col = max(int(c["col_end"]) for c in work)
    n_rows = max_row + 1
    n_cols = max_col + 1

    # (r,c) -> cell dict at origin; covered positions marked
    origin: Dict[Tuple[int, int], Dict[str, Any]] = {}
    covered: Set[Tuple[int, int]] = set()

    # 去重：同一逻辑矩形只保留一份（文本更长者优先）
    by_logic: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
    for cell in work:
        key = _cell_key(cell)
        prev = by_logic.get(key)
        if prev is None:
            by_logic[key] = cell
            continue
        t_new = str(cell.get("text") or "")
        t_old = str(prev.get("text") or "")
        if len(t_new) > len(t_old):
            by_logic[key] = cell

    for cell in by_logic.values():
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        if rs < 0 or cs < 0 or rs >= n_rows or cs >= n_cols:
            continue
        # 若起点已被覆盖：把文本并入覆盖者
        if (rs, cs) in covered or (rs, cs) in origin:
            text = str(cell.get("text") or "").strip()
            if text:
                owner = origin.get((rs, cs))
                if owner is None:
                    # 找任意覆盖该点的已登记格
                    for (or_, oc_), oc in origin.items():
                        ore, oce = int(oc["row_end"]), int(oc["col_end"])
                        if or_ <= rs <= ore and oc_ <= cs <= oce:
                            owner = oc
                            break
                if owner is not None:
                    prev = str(owner.get("text") or "").strip()
                    if text not in prev:
                        owner["text"] = (prev + " " + text).strip() if prev else text
                        logger.warning(
                            "渲染冲突，文本并入 (%s,%s): %r",
                            owner.get("row_start"),
                            owner.get("col_start"),
                            text[:40],
                        )
            continue
        origin[(rs, cs)] = cell
        for r in range(rs, min(re, n_rows - 1) + 1):
            for c in range(cs, min(ce, n_cols - 1) + 1):
                if (r, c) != (rs, cs):
                    covered.add((r, c))

    lines: List[str] = ['<table border="1" cellspacing="0" cellpadding="4">']
    for r in range(n_rows):
        lines.append("<tr>")
        for c in range(n_cols):
            if (r, c) in covered:
                continue
            cell = origin.get((r, c))
            if cell is None:
                lines.append("<td></td>")
                continue
            rs, re = int(cell["row_start"]), int(cell["row_end"])
            cs, ce = int(cell["col_start"]), int(cell["col_end"])
            row_span = max(re - rs + 1, 1)
            col_span = max(ce - cs + 1, 1)
            attrs = []
            if row_span > 1:
                attrs.append(f'rowspan="{row_span}"')
            if col_span > 1:
                attrs.append(f'colspan="{col_span}"')
            attr_s = (" " + " ".join(attrs)) if attrs else ""
            text = _escape_cell_text(str(cell.get("text") or ""))
            lines.append(f"<td{attr_s}>{text}</td>")
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def cells_to_html(
    cells: List[Dict[str, Any]],
    *,
    split_subtables: bool = True,
    compress_empty: bool = True,
    lift_captions: bool = True,
) -> str:
    """多子表切分后各自生成 HTML，表题提升为段落。"""
    if split_subtables:
        subtables = split_cells_into_subtables(cells)
    else:
        subtables = [cells] if cells else []

    parts: List[str] = []
    for sub in subtables:
        caption = ""
        work = sub
        if lift_captions:
            caption, work = extract_caption_row(work)
        if not any(str(c.get("text") or "").strip() for c in work):
            if caption:
                parts.append(f"<p>{html.escape(caption)}</p>")
            continue
        table_html = cells_to_html_table(work, compress_empty=compress_empty)
        if not table_html:
            if caption:
                parts.append(f"<p>{html.escape(caption)}</p>")
            continue
        if caption:
            parts.append(f"<p>{html.escape(caption)}</p>\n{table_html}")
        else:
            parts.append(table_html)
    return "\n\n".join(parts)


def build_html_output(
    cells: List[Dict[str, Any]],
    free_texts: List[Dict[str, Any]],
    *,
    split_subtables: bool = True,
    compress_empty: bool = True,
) -> str:
    """游离文本前缀 + HTML 表格。结构失败时只输出游离文本块。"""
    prefix = format_free_texts(free_texts)
    paras = [html.escape(line) for line in prefix.splitlines() if line.strip()] if prefix else []
    prefix_html = "\n".join(f"<p>{p}</p>" for p in paras)

    has_cell_text = any(str(c.get("text") or "").strip() for c in cells)
    if not has_cell_text:
        return prefix_html

    table = cells_to_html(
        cells,
        split_subtables=split_subtables,
        compress_empty=compress_empty,
    )
    if prefix_html and table:
        return prefix_html + "\n\n" + table
    return prefix_html or table or ""
