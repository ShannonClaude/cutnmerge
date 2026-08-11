"""基于逻辑拓扑的 HTML 表格生成（保留 rowspan/colspan）。

与 Markdown unroll 不同：合并单元格在起始格输出带 span 的 <td>，
被覆盖的子格跳过，不展开为空格子。
"""

from __future__ import annotations

import html
import logging
from typing import Any, Dict, List, Set, Tuple

from .formatter import (
    extract_caption_row,
    format_free_texts,
    split_cells_into_subtables,
)

logger = logging.getLogger(__name__)


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
