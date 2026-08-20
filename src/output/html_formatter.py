"""基于逻辑拓扑的 HTML 表格生成（保留 rowspan/colspan）。

与 Markdown unroll 不同：合并单元格在起始格输出带 span 的 <td>，
被覆盖的子格跳过，不展开为空格子。
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .formatter import (
    extract_caption_row,
    format_free_texts,
    split_cells_into_subtables,
)
from ..structure.row_header import is_physically_right_child
from ..utils.label_patterns import is_index_column

logger = logging.getLogger(__name__)

# 噪声行：整行仅剩孤立字母/符号（如末行 "L"）
_NOISE_CELL_RE = re.compile(r"^[A-Za-z]$|^[·•\.\,;:]$")
# 身列数据值：纯数字、比率、缺测横线、单字母等级代号（A/B/C…）。
# 不含 jER-604 / 实施例1 / 4,4'-DAE。
_DATA_VALUE_RE = re.compile(
    r"^(?:[-–—−~～]|[-–—−]?\d+(?:\.\d+)?(?:\s*[\/／]\s*\d+(?:\.\d+)?)?|[A-Za-z][+＋]?)$"
)
_LETTER_DATA_RE = re.compile(r"^[A-Za-z][+＋]?$")
_EXAMPLE_COL_HEADER_RE = re.compile(
    r"(合成例|实施例|実施例|比較例|比较例|对照例|参考例)"
)


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


def _logic_area(cell: Dict[str, Any]) -> int:
    return (
        (int(cell["row_end"]) - int(cell["row_start"]) + 1)
        * (int(cell["col_end"]) - int(cell["col_start"]) + 1)
    )


def _unique_cell_refs(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[int] = set()
    out: List[Dict[str, Any]] = []
    for cell in cells:
        cid = id(cell)
        if cid not in seen:
            seen.add(cid)
            out.append(cell)
    return out


def _evict_occupants(
    occupants: List[Dict[str, Any]],
    occupancy: Dict[Tuple[int, int], Dict[str, Any]],
    out: List[Dict[str, Any]],
) -> None:
    ids = {id(o) for o in occupants}
    for owner in occupants:
        rs, re = int(owner["row_start"]), int(owner["row_end"])
        cs, ce = int(owner["col_start"]), int(owner["col_end"])
        for r in range(rs, re + 1):
            for c in range(cs, ce + 1):
                if occupancy.get((r, c)) is owner:
                    del occupancy[(r, c)]
    out[:] = [c for c in out if id(c) not in ids]


def _try_place_rect_remainder(
    cell: Dict[str, Any],
    free_positions: List[Tuple[int, int]],
    occupancy: Dict[Tuple[int, int], Dict[str, Any]],
    out: List[Dict[str, Any]],
    *,
    clear_text: bool,
) -> None:
    """若剩余空位能拼成完整矩形则占位；L 形不拆成额外空格。"""
    if not free_positions:
        return
    frs = min(p[0] for p in free_positions)
    fre = max(p[0] for p in free_positions)
    fcs = min(p[1] for p in free_positions)
    fce = max(p[1] for p in free_positions)
    needed = {
        (r, c)
        for r in range(frs, fre + 1)
        for c in range(fcs, fce + 1)
    }
    if not needed.issubset(set(free_positions)):
        return
    nc = dict(cell)
    nc["row_start"], nc["row_end"] = frs, fre
    nc["col_start"], nc["col_end"] = fcs, fce
    nc["row_span"] = fre - frs + 1
    nc["col_span"] = fce - fcs + 1
    if clear_text:
        nc["text"] = ""
    out.append(nc)
    for r, c in needed:
        occupancy[(r, c)] = nc


def _place_cell_rect(
    cell: Dict[str, Any],
    occupancy: Dict[Tuple[int, int], Dict[str, Any]],
    out: List[Dict[str, Any]],
) -> None:
    nc = dict(cell)
    out.append(nc)
    rs, re = int(nc["row_start"]), int(nc["row_end"])
    cs, ce = int(nc["col_start"]), int(nc["col_end"])
    for r in range(rs, re + 1):
        for c in range(cs, ce + 1):
            occupancy[(r, c)] = nc


def _clip_row_header_logic_overlap(
    cell: Dict[str, Any],
    unique_owners: List[Dict[str, Any]],
    occupancy: Dict[Tuple[int, int], Dict[str, Any]],
    out: List[Dict[str, Any]],
    rs: int,
    re: int,
    cs: int,
    ce: int,
) -> bool:
    """左右相邻的行头父子格：裁过宽父格的 col_end，而不是合并文本。

    成功把 incoming 放到无冲突矩形时返回 True。
    """
    if not unique_owners:
        return False

    right_owners = [
        o for o in unique_owners if is_physically_right_child(cell, o)
    ]
    if right_owners:
        clip_at = min(int(o["col_start"]) for o in right_owners) - 1
        new_end = max(cs, min(ce, clip_at))
        if new_end < ce:
            cell["col_end"] = new_end
            cell["col_span"] = new_end - cs + 1
            ce = new_end
            still = False
            for r in range(rs, re + 1):
                for c in range(cs, ce + 1):
                    owner = occupancy.get((r, c))
                    if owner is not None and owner is not cell:
                        still = True
                        break
                if still:
                    break
            if not still:
                _place_cell_rect(cell, occupancy, out)
                return True

    left_parents = [
        o for o in unique_owners if is_physically_right_child(o, cell)
    ]
    if left_parents:
        for parent in left_parents:
            clip_at = int(cell["col_start"]) - 1
            pcs = int(parent["col_start"])
            old_ce = int(parent["col_end"])
            if clip_at < pcs or old_ce <= clip_at:
                continue
            for r in range(int(parent["row_start"]), int(parent["row_end"]) + 1):
                for c in range(clip_at + 1, old_ce + 1):
                    if occupancy.get((r, c)) is parent:
                        del occupancy[(r, c)]
            parent["col_end"] = clip_at
            parent["col_span"] = clip_at - pcs + 1
        still = False
        for r in range(rs, re + 1):
            for c in range(int(cell["col_start"]), int(cell["col_end"]) + 1):
                owner = occupancy.get((r, c))
                if owner is not None and owner is not cell:
                    still = True
                    break
            if still:
                break
        if not still:
            _place_cell_rect(cell, occupancy, out)
            return True

    return False


def _resolve_logic_overlaps(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    渲染前消解逻辑矩形重叠：后到的非空文本并入占位格，并裁掉冲突覆盖。

    空/碎片占位者遇到更大有字格时被赶走，避免大格剩余 L 形空位渲染成幽灵线。
    """
    if not cells:
        return cells

    # 按面积升序：先处理小格（更具体）
    ordered = sorted(cells, key=lambda c: (
        _logic_area(c),
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
            nc = dict(cell)
            out.append(nc)
            for r in range(rs, re + 1):
                for c in range(cs, ce + 1):
                    occupancy[(r, c)] = nc
            continue

        unique_owners = _unique_cell_refs(conflict_owners)
        incoming_area = _logic_area(cell)
        if text and not _is_cell_frag(text):
            frag_ids = {
                id(o)
                for o in unique_owners
                if _is_cell_frag(str(o.get("text") or ""))
                and incoming_area > _logic_area(o)
            }
            if frag_ids:
                _evict_occupants(
                    [o for o in unique_owners if id(o) in frag_ids],
                    occupancy,
                    out,
                )
                unique_owners = [o for o in unique_owners if id(o) not in frag_ids]
                conflict_owners = unique_owners
                free_positions = [
                    (r, c)
                    for r in range(rs, re + 1)
                    for c in range(cs, ce + 1)
                    if occupancy.get((r, c)) is None
                ]
                if not conflict_owners:
                    nc = dict(cell)
                    out.append(nc)
                    for r in range(rs, re + 1):
                        for c in range(cs, ce + 1):
                            occupancy[(r, c)] = nc
                    continue

        # 左行头父格盖住右侧子格：裁父格列，而不是把子标签拼进父格。
        placed = _clip_row_header_logic_overlap(
            cell, unique_owners, occupancy, out, rs, re, cs, ce
        )
        if placed:
            continue
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        conflict_owners = []
        free_positions = []
        for r in range(rs, re + 1):
            for c in range(cs, ce + 1):
                owner = occupancy.get((r, c))
                if owner is None:
                    free_positions.append((r, c))
                elif owner is not cell:
                    conflict_owners.append(owner)
        if not conflict_owners:
            nc = dict(cell)
            out.append(nc)
            for r in range(rs, re + 1):
                for c in range(cs, ce + 1):
                    occupancy[(r, c)] = nc
            continue
        unique_owners = _unique_cell_refs(conflict_owners)

        if text:
            phys_row_header = any(
                is_physically_right_child(cell, o) or is_physically_right_child(o, cell)
                for o in unique_owners
            )
            if phys_row_header:
                _try_place_rect_remainder(
                    cell, free_positions, occupancy, out, clear_text=False
                )
                continue
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
            # 有字冲突：不把 L 形剩余拆成额外空 <td>
            _try_place_rect_remainder(
                cell, free_positions, occupancy, out, clear_text=True
            )
            continue

        _try_place_rect_remainder(
            cell, free_positions, occupancy, out, clear_text=False
        )

    return out


def _looks_like_data_value(text: str) -> bool:
    t = "".join((text or "").split())
    return bool(t) and bool(_DATA_VALUE_RE.fullmatch(t))


def _column_has_data_values(
    cells: Sequence[Dict[str, Any]],
    col: int,
    *,
    skip_rows: Optional[Set[int]] = None,
) -> bool:
    """其他行在该列是否有原子数据格（数字 / 缺测横线）。"""
    skip = skip_rows or set()
    for cell in cells:
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        if skip and not skip.isdisjoint(range(rs, re + 1)):
            continue
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        if cs != ce or cs != col:
            continue
        if _looks_like_data_value(str(cell.get("text") or "")):
            return True
    return False


def _header_colspan_spans(cells: Sequence[Dict[str, Any]]) -> Set[Tuple[int, int]]:
    header_end = _effective_header_end(cells)
    spans: Set[Tuple[int, int]] = set()
    if header_end < 0:
        return spans
    for cell in cells:
        if int(cell["row_end"]) > header_end:
            continue
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        if ce > cs:
            spans.add((cs, ce))
    return spans


def _empty_atomic_from_bbox(
    src: Dict[str, Any],
    col: int,
    orig_cs: int,
    orig_ce: int,
    bbox: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """按原宽格比例切出一列空占位，避免幽灵列清理把空数据列删掉。"""
    x1, y1, x2, y2 = bbox
    span = max(orig_ce - orig_cs + 1, 1)
    w = (x2 - x1) / float(span)
    off = col - orig_cs
    nx1 = x1 + off * w
    nx2 = x1 + (off + 1) * w
    return {
        "row_start": int(src["row_start"]),
        "row_end": int(src["row_end"]),
        "col_start": col,
        "col_end": col,
        "row_span": int(src["row_end"]) - int(src["row_start"]) + 1,
        "col_span": 1,
        "text": "",
        "texts": [],
        "polygon": [[nx1, y1], [nx2, y1], [nx2, y2], [nx1, y2]],
    }


def _first_example_data_column(cells: Sequence[Dict[str, Any]]) -> Optional[int]:
    """表头带内实施例/比较例/参考例列头的最小 col_start（须 ≥1）。

    用于区分左侧行头区与数据区。列头落在第 0 列（行索引表）时返回 None，
    避免把空数据格当行头空隙吞掉。
    """
    if not cells:
        return None
    header_end = _effective_header_end(cells)
    first: Optional[int] = None
    for cell in cells:
        if int(cell["row_start"]) > header_end:
            continue
        cs = int(cell["col_start"])
        if cs < 1:
            continue
        if not _HEADER_HAS_EXAMPLE_RE.search(str(cell.get("text") or "")):
            continue
        if first is None or cs < first:
            first = cs
    return first


def _split_label_over_data_columns(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    身行标签 colspan 盖住其他行已有数字的数据列时，收到左缘并补空格。

    保留与表头同区间的真合并（P98 NC-7000L）；不拆表头带。
    若能定位实施例列头，则仅当 colspan 伸进该列及以右时才裁切，
    避免把行头区的 A/B/C 子列当成数据列。
    """
    if len(cells) < 2:
        return cells
    header_end = _effective_header_end(cells)
    aligned = _header_colspan_spans(cells)
    first_data = _first_example_data_column(cells)
    extras: List[Dict[str, Any]] = []
    changed = 0
    for cell in cells:
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        if re != rs or ce <= cs:
            continue
        if rs <= header_end:
            continue
        if (cs, ce) in aligned:
            continue
        if not str(cell.get("text") or "").strip():
            continue
        clip_end = cs
        hit_data = False
        for col in range(cs + 1, ce + 1):
            if first_data is not None and col < first_data:
                clip_end = col
                continue
            if _column_has_data_values(cells, col, skip_rows={rs}):
                hit_data = True
                break
            clip_end = col
        if not hit_data or clip_end >= ce:
            continue
        orig_cs, orig_ce = cs, ce
        bbox = _cell_physical_bbox_x_y(cell)
        x1, y1, x2, y2 = bbox
        span = max(orig_ce - orig_cs + 1, 1)
        w = (x2 - x1) / float(span)
        nx2 = x1 + (clip_end - orig_cs + 1) * w
        cell["col_end"] = clip_end
        cell["col_span"] = clip_end - cs + 1
        cell["polygon"] = [[x1, y1], [nx2, y1], [nx2, y2], [x1, y2]]
        for col in range(clip_end + 1, orig_ce + 1):
            extras.append(_empty_atomic_from_bbox(cell, col, orig_cs, orig_ce, bbox))
        changed += 1
        logger.info(
            "身列空格拆 colspan: text=%r col %s-%s → %s-%s",
            str(cell.get("text") or "")[:24],
            orig_cs,
            orig_ce,
            cs,
            clip_end,
        )
    if changed:
        logger.info("身列空格拆 colspan 完成: %d 格", changed)
        cells.extend(extras)
    return cells


def _drop_leading_header_only_columns(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    去掉仅被表头空格占用、表体从未覆盖的第 0 列（P26X194 左上角幽灵列）。
    """
    if not cells:
        return cells
    header_end = _effective_header_end(cells)
    max_col = max(int(c["col_end"]) for c in cells)
    drop_leading = 0
    for col in range(max_col + 1):
        covers = [
            c
            for c in cells
            if int(c["col_start"]) <= col <= int(c["col_end"])
        ]
        if not covers:
            drop_leading = col + 1
            continue
        body_hits = [
            c
            for c in covers
            if int(c["row_start"]) > header_end
            and str(c.get("text") or "").strip()
            and not _is_cell_frag(str(c.get("text") or ""))
        ]
        if body_hits:
            break
        header_only = all(int(c["row_end"]) <= header_end for c in covers)
        if header_only and all(not str(c.get("text") or "").strip() for c in covers):
            drop_leading = col + 1
            continue
        break
    if drop_leading <= 0:
        return cells
    out: List[Dict[str, Any]] = []
    for c in cells:
        nc = dict(c)
        cs, ce = int(c["col_start"]), int(c["col_end"])
        if ce < drop_leading:
            continue
        nc["col_start"] = max(0, cs - drop_leading)
        nc["col_end"] = ce - drop_leading
        nc["col_span"] = int(nc["col_end"]) - int(nc["col_start"]) + 1
        out.append(nc)
    logger.info("去掉表头幽灵列: drop_leading=%d", drop_leading)
    return out


def _left_cols_have_rowspan_body_labels(
    cells: Sequence[Dict[str, Any]],
    max_col: int,
    header_end: int,
) -> bool:
    """表体左侧列存在跨多行的行头大格（如聚酰亚胺组成），区别于 P26 索引列。"""
    for cell in cells:
        if int(cell["row_start"]) <= header_end:
            continue
        if int(cell["col_start"]) >= max_col:
            continue
        rsp = int(
            cell.get("row_span")
            or (int(cell["row_end"]) - int(cell["row_start"]) + 1)
        )
        if rsp < 2:
            continue
        t = str(cell.get("text") or "").strip()
        if t and not _is_cell_frag(t):
            return True
    return False


def _collapse_header_empty_corners(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    表头行左侧纯空角格并入右侧首个表头标签（P24/P25：项目前有 3 个空 td）。

    仅当左侧列在表体已被 rowspan 大行头占用时触发，避免 P26 索引列空角被吞。
    """
    if not cells:
        return cells
    header_end = _effective_header_end(cells)
    _CJK_RE = re.compile(r"[\u4e00-\u9fff]")
    min_label_col: int | None = None
    for cell in cells:
        rs, re_ = int(cell["row_start"]), int(cell["row_end"])
        if rs > header_end or re_ != rs:
            continue
        t = str(cell.get("text") or "").strip()
        if t and not _is_cell_frag(t) and _CJK_RE.search(t):
            cs = int(cell["col_start"])
            if min_label_col is None or cs < min_label_col:
                min_label_col = cs
    if min_label_col is None or min_label_col <= 0:
        return cells
    if not _left_cols_have_rowspan_body_labels(cells, min_label_col, header_end):
        return cells

    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for cell in cells:
        rs, re_ = int(cell["row_start"]), int(cell["row_end"])
        if rs > header_end or re_ != rs:
            continue
        by_row.setdefault(rs, []).append(cell)

    changed = 0
    for row_cells in by_row.values():
        row_cells.sort(key=lambda c: int(c["col_start"]))
        label_idx: int | None = None
        for i, cell in enumerate(row_cells):
            if int(cell["col_start"]) < min_label_col:
                continue
            t = str(cell.get("text") or "").strip()
            if t and not _is_cell_frag(t):
                label_idx = i
                break
        if label_idx is None:
            continue
        label = row_cells[label_idx]
        orig_cs = int(label["col_start"])
        absorbed: List[Dict[str, Any]] = []
        for j in range(label_idx):
            cell = row_cells[j]
            if int(cell.get("col_span") or 1) != 1:
                break
            if str(cell.get("text") or "").strip():
                break
            col = int(cell["col_start"])
            if col >= min_label_col:
                break
            if not _column_has_body_content(cells, col, header_end):
                break
            absorbed.append(cell)
        if absorbed:
            new_start = int(absorbed[0]["col_start"])
            label["col_start"] = new_start
            label["col_span"] = int(label["col_end"]) - new_start + 1
            for cell in absorbed:
                cell["_drop_render"] = True
            changed += 1
            continue
        # 表头行无占位 cell、但逻辑列 0..orig_cs-1 由表体 rowspan 占用 → 左扩标签 colspan
        rs = int(label["row_start"])
        has_left = any(
            int(c["col_start"]) < orig_cs
            and int(c["row_start"]) <= rs <= int(c["row_end"])
            for c in cells
            if not c.get("_drop_render")
        )
        if not has_left and orig_cs >= min_label_col:
            label["col_start"] = 0
            label["col_span"] = int(label["col_end"]) + 1
            changed += 1

    if changed:
        logger.info("表头空角压缩: %d 行", changed)
    return [c for c in cells if not c.get("_drop_render")]


def _split_header_over_index_column(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """表头格跨第 0 列、而身列 0 为实施例索引时，拆成空角 + 独立标签列。"""
    if not cells:
        return cells
    header_end = _effective_header_end(cells)
    if header_end < 0:
        return cells
    if not _column_has_body_content(cells, 0, header_end):
        return cells
    # P24/P25：身列 0 为 rowspan 大类行头，不是实施例索引列，勿拆表头角格
    if _left_cols_have_rowspan_body_labels(cells, 1, header_end):
        return cells

    extras: List[Dict[str, Any]] = []
    changed = 0
    for cell in cells:
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        if rs > header_end:
            continue
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        if cs > 0 or ce <= cs:
            continue
        if not str(cell.get("text") or "").strip():
            continue
        bbox = _cell_physical_bbox_x_y(cell)
        x1, y1, x2, y2 = bbox
        span = ce - cs + 1
        w = (x2 - x1) / float(span)
        label_col = ce
        cell["col_start"] = label_col
        cell["col_end"] = label_col
        cell["col_span"] = 1
        nx1 = x1 + (label_col - cs) * w
        nx2 = nx1 + w
        cell["polygon"] = [[nx1, y1], [nx2, y1], [nx2, y2], [nx1, y2]]
        for col in range(cs, label_col):
            ex = _empty_atomic_from_bbox(cell, col, cs, ce, bbox)
            ex["row_start"] = rs
            ex["row_end"] = re
            ex["row_span"] = re - rs + 1
            extras.append(ex)
        changed += 1
        logger.info(
            "表头索引列角拆分: text=%r col %s-%s → %s",
            str(cell.get("text") or "")[:24],
            cs,
            ce,
            label_col,
        )
    if changed:
        cells.extend(extras)
    return cells


def _column_has_body_content(
    cells: Sequence[Dict[str, Any]],
    col: int,
    header_end: int,
) -> bool:
    """表头带以下该列是否有实质文本（如实施例索引列）。"""
    for cell in cells:
        if int(cell["row_start"]) <= header_end:
            continue
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        if cs <= col <= ce:
            t = str(cell.get("text") or "").strip()
            if t and not _is_cell_frag(t):
                return True
    return False


def _body_anchors_between(
    cells: Sequence[Dict[str, Any]],
    lo: int,
    hi: int,
    header_end: int,
) -> bool:
    """表体是否有格子的 col_start 落在 [lo, hi) 内。"""
    for cell in cells:
        if int(cell["row_start"]) <= header_end:
            continue
        cs = int(cell["col_start"])
        if lo <= cs < hi and str(cell.get("text") or "").strip():
            return True
    return False


def _merge_leading_empty_into_label(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """行首空列并入右侧中文标签格；同行「分辨率」+(μm) 合成单格。"""
    if not cells:
        return cells
    header_end = _effective_header_end(cells)
    _CJK_RE_LOCAL = re.compile(r"[\u4e00-\u9fff]")
    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for cell in cells:
        if int(cell["row_start"]) != int(cell["row_end"]):
            continue
        by_row.setdefault(int(cell["row_start"]), []).append(cell)

    for row_cells in by_row.values():
        row_cells.sort(key=lambda c: int(c["col_start"]))
        if len(row_cells) < 2:
            continue
        label_idx: int | None = None
        for i, cell in enumerate(row_cells):
            t = str(cell.get("text") or "").strip()
            if t and _CJK_RE_LOCAL.search(t):
                label_idx = i
                break
        if label_idx is None or label_idx == 0:
            continue
        label = row_cells[label_idx]
        orig_cs = int(label["col_start"])
        new_start = orig_cs
        absorbed: List[Dict[str, Any]] = []
        for j in range(label_idx):
            cell = row_cells[j]
            if int(cell.get("col_span") or 1) != 1:
                break
            if str(cell.get("text") or "").strip():
                break
            col = int(cell["col_start"])
            if _column_has_body_content(cells, col, header_end):
                break
            if _body_anchors_between(cells, col, orig_cs, header_end):
                break
            new_start = col
            absorbed.append(cell)
        if not absorbed:
            continue
        label["col_start"] = new_start
        label["col_span"] = int(label["col_end"]) - int(label["col_start"]) + 1
        for cell in absorbed:
            cell["_drop_render"] = True

    # 分辨率 + (μm) 等同属性标签与单位格合并
    _UNIT_TAIL_RE = re.compile(r"^[\(（]?\s*μm\s*[\)）]?$", re.I)
    for row_cells in by_row.values():
        row_cells.sort(key=lambda c: int(c["col_start"]))
        for i, cell in enumerate(row_cells):
            t = str(cell.get("text") or "").strip()
            if not t or not _CJK_RE_LOCAL.search(t):
                continue
            if i + 1 >= len(row_cells):
                continue
            nxt = row_cells[i + 1]
            if int(nxt["col_start"]) != int(cell["col_end"]) + 1:
                continue
            nt = str(nxt.get("text") or "").strip()
            if not _UNIT_TAIL_RE.fullmatch(nt):
                continue
            cell["text"] = f"{t}{nt}"
            cell["col_end"] = int(nxt["col_end"])
            cell["col_span"] = int(cell["col_end"]) - int(cell["col_start"]) + 1
            nxt["_drop_render"] = True

    return [c for c in cells if not c.get("_drop_render")]


def _cover_stub_column_gaps(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """行头区内：把空原子格与未覆盖逻辑格并入左侧同行标签，不跨数据列。"""
    first_data = _first_example_data_column(cells)
    if first_data is None or first_data <= 1:
        return cells

    header_end = _effective_header_end(cells)
    occupancy: Dict[Tuple[int, int], Dict[str, Any]] = {}
    max_row = 0
    for cell in cells:
        if cell.get("_drop_render"):
            continue
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        max_row = max(max_row, re)
        for r in range(rs, re + 1):
            for c in range(cs, ce + 1):
                occupancy[(r, c)] = cell

    for row in range(header_end + 1, max_row + 1):
        col = 0
        while col < first_data:
            owner = occupancy.get((row, col))
            if (
                owner is None
                or owner.get("_drop_render")
                or int(owner["row_start"]) != row
                or int(owner["row_end"]) != row
                or int(owner["col_start"]) != col
                or not str(owner.get("text") or "").strip()
            ):
                col += 1
                continue
            end = int(owner["col_end"])
            absorbed: List[Dict[str, Any]] = []
            j = end + 1
            while j < first_data:
                nxt = occupancy.get((row, j))
                if nxt is not None and nxt.get("_drop_render"):
                    nxt = None
                if nxt is None:
                    end = j
                    j += 1
                    continue
                if nxt is owner:
                    j += 1
                    continue
                if int(nxt["row_start"]) != row:
                    break
                if str(nxt.get("text") or "").strip():
                    break
                if int(nxt.get("col_span") or 1) != 1:
                    break
                absorbed.append(nxt)
                end = int(nxt["col_end"])
                j = end + 1
            if end > int(owner["col_end"]):
                owner["col_end"] = end
                owner["col_span"] = end - int(owner["col_start"]) + 1
                for a in absorbed:
                    a["_drop_render"] = True
                for c in range(int(owner["col_start"]), end + 1):
                    occupancy[(row, c)] = owner
            col = int(owner["col_end"]) + 1
    return [c for c in cells if not c.get("_drop_render")]


def _merge_leading_label_gaps(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把同行右侧连续空原子格并进左侧标签；身列已有数字的空格不吞。

    能定位实施例列头时，只在行头区补 colspan（含未覆盖逻辑格），不跨数据列。
    """
    if not cells:
        return cells
    work = _cover_stub_column_gaps(cells)
    first_data = _first_example_data_column(work)

    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for cell in work:
        if int(cell["row_start"]) != int(cell["row_end"]):
            continue
        by_row.setdefault(int(cell["row_start"]), []).append(cell)

    for row_idx, row_cells in by_row.items():
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
            col = int(cell["col_start"])
            if first_data is not None and col >= first_data:
                break
            if _column_has_data_values(work, col, skip_rows={row_idx}):
                break
            merge_until = int(cell["col_end"])
            absorbed.append(cell)
        if not absorbed:
            continue
        first["col_end"] = merge_until
        first["col_span"] = int(first["col_end"]) - int(first["col_start"]) + 1
        for cell in absorbed:
            cell["_drop_render"] = True
    return [c for c in work if not c.get("_drop_render")]


def _cell_physical_bbox_x_y(cell: Dict[str, Any]) -> Tuple[float, float, float, float]:
    poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
    return float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max())


def _is_cell_frag(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    compact = "".join(t.split())
    # 单字母等级代号是表体数据，不是空角碎片
    if _LETTER_DATA_RE.fullmatch(compact):
        return False
    return len(compact) <= 2


def _merge_text_into_cell(cell: Dict[str, Any], text: str) -> None:
    add = (text or "").strip()
    if not add:
        return
    prev = str(cell.get("text") or "").strip()
    if add not in prev:
        cell["text"] = (prev + " " + add).strip() if prev else add


def _salvage_dropped_cell_text(
    out: List[Dict[str, Any]],
    *,
    row: int,
    old_start: int,
    text: str,
    kept: List[int],
    remap: Dict[int, int],
    axis: str,
) -> None:
    """把整段逻辑范围都被删掉的 cell 文本并入最近保留邻格。"""
    add = (text or "").strip()
    if not add or not out or not kept:
        return
    neighbor = next((k for k in reversed(kept) if k < old_start), None)
    if neighbor is None:
        logger.warning("删%s后左侧无邻格，文本已丢: %r", axis, add[:40])
        return
    target = remap[neighbor]
    for oc in out:
        if axis == "col":
            if int(oc["row_start"]) <= row <= int(oc["row_end"]) and int(
                oc["col_start"]
            ) <= target <= int(oc["col_end"]):
                _merge_text_into_cell(oc, add)
                return
        else:
            if int(oc["col_start"]) <= row <= int(oc["col_end"]) and int(
                oc["row_start"]
            ) <= target <= int(oc["row_end"]):
                _merge_text_into_cell(oc, add)
                return
    logger.warning("删%s后无法并入邻格，文本已丢: %r", axis, add[:40])


def drop_evidenceless_columns(cells: List[Dict[str, Any]], *, narrow_ratio: float = 0.35) -> List[Dict[str, Any]]:
    """
    丢弃“证据不足”的幽灵列（零文本、且几何上非常窄、且不参与跨列 colspan）。
    窄列若仅有短碎片文本、且不参与 colspan，同样丢弃。

    非空 cell 的 col_start（含纯 colspan 原点）一律保留，避免 P98 合成例 15/16
    那种「表头+表体都是 colspan、没有原子格」的列把字连格删掉。

    注意：此实现无法直接判定“无框线/无墨迹”（html_formatter 不接收原图），
    因此使用可观测代理：窄宽度 + 不存在跨列占用 + 覆盖该列的文本为空/碎片。
    """
    if not cells:
        return cells

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
        # colspan 原点列：格子从这里起步且有实质文本，不能当幽灵列删
        if any(
            int(c["col_start"]) == col
            and not _is_cell_frag(str(c.get("text") or ""))
            for c in covered_cells
        ):
            continue
        only_frag = (not any_text) or all(_is_cell_frag(t) for t in texts if t) or (
            any_text and all(_is_cell_frag(t) for t in texts)
        )
        spans_cross = any(int(c.get("col_span") or 1) > 1 for c in covered_cells)
        if spans_cross:
            atomic_texts = [
                str(c.get("text") or "").strip()
                for c in covered_cells
                if int(c.get("col_span") or 1) == 1
            ]
            if any(not _is_cell_frag(t) for t in atomic_texts):
                continue
            if not atomic_texts:
                dropped.add(col)
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
                        _merge_text_into_cell(c, "".join(frag_bits))
                        break
            dropped.add(col)

    if not dropped:
        return cells

    kept_cols = [i for i in range(max_col + 1) if i not in dropped]
    if not kept_cols:
        return cells

    remap = {old: new for new, old in enumerate(kept_cols)}
    out: List[Dict[str, Any]] = []
    orphans: List[Tuple[int, int, str]] = []
    for c in cells:
        cs, ce = int(c["col_start"]), int(c["col_end"])
        new_idxs = [remap[i] for i in range(cs, ce + 1) if i in remap]
        if not new_idxs:
            text = str(c.get("text") or "").strip()
            if text and not _is_cell_frag(text):
                orphans.append((int(c["row_start"]), cs, text))
            continue
        nc = dict(c)
        nc["col_start"] = min(new_idxs)
        nc["col_end"] = max(new_idxs)
        nc["col_span"] = nc["col_end"] - nc["col_start"] + 1
        out.append(nc)
    for row, old_cs, text in orphans:
        _salvage_dropped_cell_text(
            out,
            row=row,
            old_start=old_cs,
            text=text,
            kept=kept_cols,
            remap=remap,
            axis="col",
        )
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
    orphans: List[Tuple[int, int, str]] = []
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        new_idxs = [remap[i] for i in range(rs, re + 1) if i in remap]
        if not new_idxs:
            text = str(c.get("text") or "").strip()
            if text and not _is_cell_frag(text):
                orphans.append((int(c["col_start"]), rs, text))
            continue
        nc = dict(c)
        nc["row_start"] = min(new_idxs)
        nc["row_end"] = max(new_idxs)
        nc["row_span"] = nc["row_end"] - nc["row_start"] + 1
        out.append(nc)
    for col, old_rs, text in orphans:
        _salvage_dropped_cell_text(
            out,
            row=col,
            old_start=old_rs,
            text=text,
            kept=kept_rows,
            remap=remap,
            axis="row",
        )
    return out


def _drop_body_empty_columns(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    删除表体（表头以下）所有原子格均为空/碎片的列。

    与 drop_evidenceless_columns 不同：这里不看几何宽度，只看表体内容。
    表头 colspan 文本不算该列的证据——colspan 会在列删除后自动收缩。
    仅在表头 anchor 是该列唯一非空来源时才删。
    """
    if not cells:
        return cells
    header_end = _effective_header_end(cells)
    max_col = max(int(c["col_end"]) for c in cells)

    col_to_cells: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(max_col + 1)}
    for c in cells:
        for col in range(int(c["col_start"]), int(c["col_end"]) + 1):
            col_to_cells[col].append(c)

    dropped: Set[int] = set()
    for col in range(max_col + 1):
        covered = col_to_cells.get(col, [])
        if not covered:
            continue
        # 该列有数据值（数字/缺测横线等） → 真实数据列
        if _column_has_data_values(cells, col):
            continue
        # 表头带起点有实质标签（实施例列头、分散液、分辨率等）→ 保留
        if any(
            int(c["col_start"]) == col
            and int(c["row_start"]) <= header_end
            and not _is_cell_frag(str(c.get("text") or ""))
            for c in covered
        ):
            continue
        # 该列是某 cell 的 col_start 且有行标签等实质非数据文本 → 保留
        # （如「实施例 1」「聚酰亚胺」等，但不含短单位 "(mJ/cm2)"）
        body_anchors = [
            c for c in covered
            if int(c["row_start"]) > header_end
            and int(c["col_start"]) == col
            and not _is_cell_frag(str(c.get("text") or ""))
        ]
        # 有多个 body anchor 带文本：很可能是行标签列，保留
        if len(body_anchors) >= 2:
            continue
        # 唯一的 body anchor：若为单位/注释文本 → 幽灵列，但需保留文本
        if body_anchors:
            t = str(body_anchors[0].get("text") or "").strip()
            # 纯单位/注释：必须带括号，如 (mJ/cm2)、(℃)。勿把 jER828 / C 当单位删列。
            if not re.fullmatch(r"[\(\（][A-Za-z%°℃²³/\s\.\d\-μ]+[\)\）]", t):
                continue
        # 幽灵列：无数据值，无实质行标签
        dropped.add(col)

    # 删除前：把整段范围全在 dropped 中的 cell 的文本迁移到右侧保留列
    if dropped:
        for col in sorted(dropped):
            covered = col_to_cells.get(col, [])
            for c in covered:
                if int(c["col_start"]) != col:
                    continue
                cs, ce = int(c["col_start"]), int(c["col_end"])
                # 只迁移整段都在 dropped 中的 cell（跨保留列的 cell 会自行缩列）
                if not all(i in dropped for i in range(cs, ce + 1)):
                    continue
                t = str(c.get("text") or "").strip()
                if not t or _is_cell_frag(t):
                    continue
                row = int(c["row_start"])
                right = next((k for k in range(ce + 1, max_col + 1) if k not in dropped), None)
                if right is None:
                    continue
                for rc in col_to_cells.get(right, []):
                    if int(rc["row_start"]) <= row <= int(rc["row_end"]):
                        prev = str(rc.get("text") or "").strip()
                        if not prev:
                            rc["text"] = t
                        elif t not in prev:
                            rc["text"] = t + "\n" + prev
                        c["text"] = ""
                        break

    if not dropped:
        return cells

    kept_cols = [i for i in range(max_col + 1) if i not in dropped]
    if not kept_cols:
        return cells

    remap = {old: new for new, old in enumerate(kept_cols)}
    out: List[Dict[str, Any]] = []
    orphans: List[Tuple[int, int, str]] = []
    for c in cells:
        cs, ce = int(c["col_start"]), int(c["col_end"])
        new_idxs = [remap[i] for i in range(cs, ce + 1) if i in remap]
        if not new_idxs:
            text = str(c.get("text") or "").strip()
            if text and not _is_cell_frag(text):
                orphans.append((int(c["row_start"]), cs, text))
            continue
        nc = dict(c)
        nc["col_start"] = min(new_idxs)
        nc["col_end"] = max(new_idxs)
        nc["col_span"] = nc["col_end"] - nc["col_start"] + 1
        out.append(nc)
    for row, old_cs, text in orphans:
        _salvage_dropped_cell_text(
            out,
            row=row,
            old_start=old_cs,
            text=text,
            kept=kept_cols,
            remap=remap,
            axis="col",
        )
    if dropped:
        logger.info("删除表体空列: %s", sorted(dropped))
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


def _effective_header_end(cells: Sequence[Dict[str, Any]]) -> int:
    """顶表头带末行；无 rowspan 线索时取首个有字身行之上。"""
    he = _top_header_band_end(list(cells))
    if he >= 0:
        return he
    body_starts = [
        int(c["row_start"])
        for c in cells
        if str(c.get("text") or "").strip() and not _is_cell_frag(str(c.get("text") or ""))
    ]
    if not body_starts:
        return 0
    return max(0, min(body_starts) - 1)


_LONE_EXAMPLE_NUM_RE = re.compile(r"^\d+$")
_HEADER_HAS_EXAMPLE_RE = re.compile(r"实[施試]例|実[施試]例|比較例|比较例|参考例")


def _repair_lone_example_number_headers(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """顶栏仅 OCR 出数字时，若同行已有实施例/比较例等列头，补回前缀。"""
    header_end = _effective_header_end(cells)
    has_example_peer = any(
        _HEADER_HAS_EXAMPLE_RE.search(str(c.get("text") or ""))
        for c in cells
        if int(c["row_start"]) <= header_end
    )
    if not has_example_peer:
        return cells

    for c in cells:
        if int(c["row_start"]) > header_end:
            continue
        t = str(c.get("text") or "").strip()
        if not _LONE_EXAMPLE_NUM_RE.fullmatch(t):
            continue
        c["text"] = f"实施例 {t}"
    return cells


def _top_header_band_end(cells: List[Dict[str, Any]]) -> int:
    """顶表头带最后一行（含）。表体左侧大行头的 rowspan 不计入。

    只有 ``row_start <= 1`` 的跨行格才能把表头带向下扩；若存在表体左侧
    大行头（``row_start >= 2`` 且偏左），表头带截止在其上一行，避免把
    ``比较例 1`` / 数据 ``100`` 错误向下合并进表体空格。
    """
    header_end = -1
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        if re > rs and rs <= 1:
            header_end = max(header_end, re)

    first_body_stub: int | None = None
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        cs = int(c["col_start"])
        if re > rs and rs >= 2 and cs <= 1:
            first_body_stub = rs if first_body_stub is None else min(first_body_stub, rs)
    if first_body_stub is not None:
        cap = first_body_stub - 1
        header_end = cap if header_end < 0 else min(header_end, cap)
    return header_end


_HEADER_UNIT_RE = re.compile(
    r"^[\(（][\w%°℃²³/\s\.\-μ]+[\)）]$"
)


def _is_header_unit_line(text: str) -> bool:
    """括号包裹的纯单位文本，如 (mJ/cm2)、(mN)、(%)、(℃)。"""
    t = "".join((text or "").split())
    return bool(t) and bool(_HEADER_UNIT_RE.fullmatch(t))


def _merge_header_unit_rows(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    表头中：若某行所有非空格都是括号单位文本且上方同列有描述性表头，
    则将单位行合并到上方格（文本拼接为换行）并删除单位行。

    典型场景：row0=「最小曝光量（Eth）|密合强度」, row1=「(mJ/cm2)|(mN)」
    """
    if not cells:
        return cells

    max_row = max(int(c["row_end"]) for c in cells)
    if max_row < 1:
        return cells

    by_key: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        cs, ce = int(c["col_start"]), int(c["col_end"])
        if rs == re:
            by_key[(rs, cs, ce)] = c

    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for c in cells:
        rs = int(c["row_start"])
        re = int(c["row_end"])
        if rs == re:
            by_row.setdefault(rs, []).append(c)

    # 找候选单位行：该行所有非空格都是括号单位
    unit_rows: Set[int] = set()
    for row, row_cells in by_row.items():
        if row < 1:
            continue
        non_empty = [c for c in row_cells if str(c.get("text") or "").strip()]
        if not non_empty:
            continue
        if all(_is_header_unit_line(str(c.get("text") or "").strip()) for c in non_empty):
            # 还需上方行存在描述性表头
            has_upper = False
            for c in non_empty:
                cs, ce = int(c["col_start"]), int(c["col_end"])
                upper = by_key.get((row - 1, cs, ce))
                if upper and str(upper.get("text") or "").strip():
                    has_upper = True
                    break
            if has_upper:
                unit_rows.add(row)

    if not unit_rows:
        return cells

    dropped_ids: Set[int] = set()
    for row in unit_rows:
        for c in by_row.get(row, []):
            t = str(c.get("text") or "").strip()
            cs, ce = int(c["col_start"]), int(c["col_end"])
            upper = by_key.get((row - 1, cs, ce))
            if upper and t:
                ut = str(upper.get("text") or "").strip()
                upper["text"] = ut + "\n" + t if ut else t
                upper["row_end"] = int(c["row_end"])
                upper["row_span"] = int(upper["row_end"]) - int(upper["row_start"]) + 1
            dropped_ids.add(id(c))

    if not dropped_ids:
        return cells
    return [c for c in cells if id(c) not in dropped_ids]


def _split_example_header_rowspans(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """实施例/比较例列头若被误扩成跨两行 rowspan，收回为单行。

    P24：下层是聚酰亚胺行的 A/B/C 数据，不是二级表头。
    """
    if not cells:
        return cells
    changed = 0
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        if re <= rs:
            continue
        if rs > 1:
            continue
        text = str(c.get("text") or "")
        if not _EXAMPLE_COL_HEADER_RE.search(text):
            continue
        # 仅拆「跨一行」的假二级表头（避免误伤更复杂多级）
        if re != rs + 1:
            continue
        c["row_end"] = rs
        c["row_span"] = 1
        changed += 1
    if changed:
        logger.info("实施例列头 rowspan 收回: %d 格", changed)
    return cells


def _merge_header_empty_below(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    合并表头中：上方有“实文本”的单行格，且正下方同列范围是空/碎片文本的单行格。

    目标修复：诸如 P100X888 的“分散液”表头被切成两层的情况。
    仅在真正的顶表头带内合并，不用表体左侧大行头去扩大截止行。
    """
    if not cells:
        return cells

    header_end = _top_header_band_end(cells)
    if header_end < 1:
        return cells

    def _is_frag_or_empty(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        compact = "".join(t.split())
        # 括号包裹的单位/注释（如 (℃)、(min)、(%)）不是碎片
        if compact.startswith("(") or compact.startswith("（"):
            return False
        # 单字母等级代号（A/B/C…）是表体数据，不可当下层空碎片吞掉
        if _LETTER_DATA_RE.fullmatch(compact):
            return False
        return len(compact) <= 2 or compact in {"-", "—", "_"}

    # 用 (row_start, col_start, col_end) 精确匹配“正下方同列范围”的格
    # （注意：某些表头“空格”在 TSR/cells 中可能根本不存在，此时 renderer 会输出空 <td>。）
    by_key: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        cs, ce = int(c["col_start"]), int(c["col_end"])
        if rs == re:  # 只在 atomic row 内尝试“向下合并”
            by_key[(rs, cs, ce)] = c

    # 判断某个“(next_row, column_range)”区域是否被任何非空 cell 占用
    def _occupied_by_nonempty(next_row: int, cs: int, ce: int) -> bool:
        for c in cells:
            rs, re = int(c["row_start"]), int(c["row_end"])
            cols_s, cols_e = int(c["col_start"]), int(c["col_end"])
            if not (rs <= next_row <= re):
                continue
            # c 的列范围至少覆盖目标列范围
            if not (cols_s <= cs and cols_e >= ce):
                continue
            if str(c.get("text") or "").strip():
                return True
        return False

    dropped_ids: Set[int] = set()
    out: List[Dict[str, Any]] = []
    for upper in cells:
        if id(upper) in dropped_ids:
            continue

        ur = int(upper["row_start"])
        ue = int(upper["row_end"])
        uc_s = int(upper["col_start"])
        uc_e = int(upper["col_end"])
        upper_text = str(upper.get("text") or "").strip()

        # 只合并顶表头带内、且上方格为单行格的情况。
        # 下一行若已出表头带（进入表体），不得把列头/数据格向下吞掉。
        if ue != ur:
            out.append(upper)
            continue
        if ue + 1 > header_end:
            out.append(upper)
            continue
        # 只处理单列表头：避免把诸如“组成[质量%]”(colspan>1) 的大格也错误向下延展
        if uc_s != uc_e:
            out.append(upper)
            continue
        if not upper_text or _is_frag_or_empty(upper_text):
            out.append(upper)
            continue
        # 实施例/比较例列头不得向下吞并（下层常为 A/B/C 等数据字母）
        if _EXAMPLE_COL_HEADER_RE.search(upper_text):
            out.append(upper)
            continue

        lower = by_key.get((ue + 1, uc_s, uc_e))
        if lower is None:
            # 下方没有 cell 对象时，renderer 会输出空 td；若没有任何非空 cell 占用该区域，则直接扩展 upper rowspan
            next_row = ue + 1
            if not _occupied_by_nonempty(next_row, uc_s, uc_e):
                upper["row_end"] = next_row
                upper["row_span"] = int(upper["row_end"]) - int(upper["row_start"]) + 1
            out.append(upper)
            continue

        lower_text = str(lower.get("text") or "")
        if not _is_frag_or_empty(lower_text):
            out.append(upper)
            continue

        # 合并：把 lower 的行范围并入 upper，丢弃 lower
        upper["row_end"] = int(lower["row_end"])
        upper["row_span"] = int(upper["row_end"]) - int(upper["row_start"]) + 1
        dropped_ids.add(id(lower))
        out.append(upper)

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
    work = _split_example_header_rowspans(work)
    work = _resolve_logic_overlaps(work)
    work = _repair_lone_example_number_headers(work)
    work = _drop_leading_header_only_columns(work)
    work = _split_label_over_data_columns(work)
    work = _split_header_over_index_column(work)
    work = _collapse_header_empty_corners(work)
    work = _merge_leading_empty_into_label(work)
    work = _merge_leading_label_gaps(work)
    # 去掉原先只处理“行首标签”的非对称美化；改为对称幽灵行/列清理
    work = drop_evidenceless_columns(work)
    work = _drop_body_empty_columns(work)
    work = drop_evidenceless_rows(work)
    work = drop_noise_rows(work)
    if compress_empty:
        work = compress_empty_logic_columns(work)
        work = compress_empty_logic_rows(work)
    work = _merge_header_unit_rows(work)
    work = _merge_header_empty_below(work)
    work = [c for c in work if not c.get("_drop_render")]
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
