"""表头带假横切/竖切修复：按列检测横/竖线墨迹，局部恢复 rowspan/colspan。

- rowspan：仅处理「首个表体行之前」的行界；合并条件为横线弱 + 碎片/sliver/OCR 跨界/同列折行。
- colspan：仅处理表头行内相邻原子格；表头竖线弱且表体竖线强时合并（表体竖线穿入表头）。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .tsr_refine import (
    _cell_bbox,
    dedupe_overlapping_cells,
    _derive_seps,
    _rebuild_polygon,
    _refresh_spans,
)

logger = logging.getLogger(__name__)

_DATA_ROW_RE = re.compile(
    r"(合成例|实施例|実施例|比較例|比较例|对照例|参考例)"
)
_PARENT_HDR_RE = re.compile(r"(单体\s*[\[［]|聚合物)")
_CJK_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]"
)
_SUBHEADER_TAIL_RE = re.compile(
    r"(衍生物|化合物|单体|封端剂|硅烷|当量|比率|低聚物|\)|）)$"
)


def _tb_bbox(tb: Dict[str, Any]) -> Tuple[float, float, float, float]:
    poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )


def _median_text_h(text_boxes: Sequence[Dict[str, Any]]) -> float:
    hs: List[float] = []
    for tb in text_boxes:
        _, y1, _, y2 = _tb_bbox(tb)
        h = max(0.0, y2 - y1)
        if h > 1e-3:
            hs.append(h)
    return float(np.median(hs)) if hs else 12.0


def _compact(text: str) -> str:
    return "".join((text or "").split())


def _is_fragment_text(text: str) -> bool:
    t = _compact(text)
    return (not t) or len(t) <= 2


def _hline_coverage_ratio(
    binary: np.ndarray,
    y: float,
    x1: float,
    x2: float,
    *,
    tol: float = 4.0,
) -> float:
    """
    本列区间内、y 附近的「横线」覆盖率（经水平形态学，抑制笔画误判）。
    返回 [0,1]；失败时偏保守返回 1.0（视为有线，不合并）。
    """
    if binary is None or binary.size == 0:
        return 1.0
    h, w = binary.shape[:2]
    ya = max(0, int(round(y - tol)))
    yb = min(h, int(round(y + tol + 1)))
    xa = max(0, int(round(x1)))
    xb = min(w, int(round(x2)))
    if yb <= ya or xb <= xa + 2:
        return 1.0
    band = binary[ya:yb, xa:xb]
    bw = int(band.shape[1])
    kw = max(9, min(bw // 3, bw))
    if kw < 3:
        return float(np.mean(band > 0))
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    hmask = cv2.morphologyEx(band, cv2.MORPH_OPEN, ker, iterations=1)
    col_hit = np.any(hmask > 0, axis=0)
    if col_hit.size == 0:
        return 1.0
    return float(np.mean(col_hit))


def _vline_coverage_ratio(
    binary: np.ndarray,
    x: float,
    y1: float,
    y2: float,
    *,
    tol: float = 4.0,
) -> float:
    """
    y1..y2 区间内、x 附近的「竖线」覆盖率（经竖直形态学）。
    失败时偏保守返回 1.0（视为有线，不合并）。
    """
    if binary is None or binary.size == 0:
        return 1.0
    h, w = binary.shape[:2]
    xa = max(0, int(round(x - tol)))
    xb = min(w, int(round(x + tol + 1)))
    ya = max(0, int(round(y1)))
    yb = min(h, int(round(y2)))
    if yb <= ya or xb <= xa + 2:
        return 1.0
    band = binary[ya:yb, xa:xb]
    bh = int(band.shape[0])
    kw = max(9, min(bh // 3, bh))
    if kw < 3:
        return float(np.mean(band > 0))
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kw))
    vmask = cv2.morphologyEx(band, cv2.MORPH_OPEN, ker, iterations=1)
    row_hit = np.any(vmask > 0, axis=1)
    if row_hit.size == 0:
        return 1.0
    return float(np.mean(row_hit))


def _substantive_cjk_label(text: str) -> bool:
    """表头子标签是否像完整的一列标题（而非折行碎片）。"""
    t = _compact(text)
    if len(t) < 5 or not _CJK_RE.search(t):
        return False
    if len(t) >= 12:
        return True
    return bool(_SUBHEADER_TAIL_RE.search(t))


def _looks_like_two_independent_subheaders(left_text: str, right_text: str) -> bool:
    """两侧 OCR 都像独立子表头时，禁止 colspan 合并。"""
    return _substantive_cjk_label(left_text) and _substantive_cjk_label(right_text)


def _same_column_wrap(
    upper: Dict[str, Any],
    lower: Dict[str, Any],
    *,
    min_ratio: float = 0.75,
) -> bool:
    ux1, _uy1, ux2, _uy2 = _cell_bbox(upper)
    lx1, _ly1, lx2, _ly2 = _cell_bbox(lower)
    overlap = min(ux2, lx2) - max(ux1, lx1)
    uw = max(ux2 - ux1, 1.0)
    lw = max(lx2 - lx1, 1.0)
    return overlap / min(uw, lw) >= min_ratio


def _ocr_texts_in_bbox(
    text_boxes: Sequence[Dict[str, Any]],
    bbox: Tuple[float, float, float, float],
) -> List[str]:
    x1, y1, x2, y2 = bbox
    out: List[str] = []
    for tb in text_boxes:
        bx1, by1, bx2, by2 = _tb_bbox(tb)
        cx = 0.5 * (bx1 + bx2)
        cy = 0.5 * (by1 + by2)
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            out.append(str(tb.get("text") or "").strip())
    return out


def _ocr_crosses_y_in_x(
    text_boxes: Sequence[Dict[str, Any]],
    y: float,
    x1: float,
    x2: float,
    *,
    strip_half: float,
) -> bool:
    lo, hi = y - strip_half, y + strip_half
    for tb in text_boxes:
        bx1, by1, bx2, by2 = _tb_bbox(tb)
        cx = 0.5 * (bx1 + bx2)
        if not (x1 <= cx <= x2):
            continue
        if by1 < lo and by2 > hi:
            return True
    return False


def _first_body_row_index(
    cells: Sequence[Dict[str, Any]],
    text_boxes: Sequence[Dict[str, Any]],
    row_seps: Sequence[float],
) -> int:
    """返回首个表体逻辑行索引；找不到则退化为 min(3, n_rows)。"""
    n_rows = max(0, len(row_seps) - 1)
    if n_rows <= 1:
        return n_rows

    for r in range(n_rows):
        y0, y1 = float(row_seps[r]), float(row_seps[r + 1])
        for tb in text_boxes:
            text = str(tb.get("text") or "")
            if not _DATA_ROW_RE.search(text):
                continue
            _x1, by1, _x2, by2 = _tb_bbox(tb)
            cy = 0.5 * (by1 + by2)
            if y0 <= cy < y1:
                return r

    for c in cells:
        text = str(c.get("text") or "")
        if _DATA_ROW_RE.search(text):
            return int(c["row_start"])

    return min(3, n_rows)


def _same_col_key(cell: Dict[str, Any]) -> Tuple[int, int]:
    return int(cell["col_start"]), int(cell["col_end"])


def _merge_pair(
    upper: Dict[str, Any],
    lower: Dict[str, Any],
) -> Dict[str, Any]:
    ux1, uy1, ux2, uy2 = _cell_bbox(upper)
    lx1, ly1, lx2, ly2 = _cell_bbox(lower)
    x1, y1 = min(ux1, lx1), min(uy1, ly1)
    x2, y2 = max(ux2, lx2), max(uy2, ly2)
    merged = dict(upper)
    merged["row_start"] = min(int(upper["row_start"]), int(lower["row_start"]))
    merged["row_end"] = max(int(upper["row_end"]), int(lower["row_end"]))
    merged["col_start"] = min(int(upper["col_start"]), int(lower["col_start"]))
    merged["col_end"] = max(int(upper["col_end"]), int(lower["col_end"]))
    _refresh_spans(merged)
    merged["polygon"] = _rebuild_polygon(x1, y1, x2, y2)
    return merged


def _merge_horizontal_pair(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> Dict[str, Any]:
    lx1, ly1, lx2, ly2 = _cell_bbox(left)
    rx1, ry1, rx2, ry2 = _cell_bbox(right)
    x1, y1 = min(lx1, rx1), min(ly1, ry1)
    x2, y2 = max(lx2, rx2), max(ly2, ry2)
    merged = dict(left)
    merged["row_start"] = int(left["row_start"])
    merged["row_end"] = int(left["row_end"])
    merged["col_start"] = min(int(left["col_start"]), int(right["col_start"]))
    merged["col_end"] = max(int(left["col_end"]), int(right["col_end"]))
    _refresh_spans(merged)
    merged["polygon"] = _rebuild_polygon(x1, y1, x2, y2)
    return merged


def repair_rowspans_by_hline_gaps(
    cells: List[Dict[str, Any]],
    binary: np.ndarray,
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    tol: float = 4.0,
    min_ink_ratio: float = 0.40,
    sliver_ratio: float = 0.40,
    max_header_boundaries: int = 3,
) -> List[Dict[str, Any]]:
    """
    表头带内：若某列在行界处横线墨迹弱，且有碎片/sliver/OCR 跨界证据，
    则把该列上下原子格合并为 rowspan。不改 col_*。
    """
    if not cells or binary is None or binary.size == 0:
        return cells

    text_boxes = list(text_boxes or [])
    row_seps, _col_seps = _derive_seps(cells)
    if len(row_seps) < 3:
        return cells

    n_rows = len(row_seps) - 1
    body_row = _first_body_row_index(cells, text_boxes, row_seps)
    # 只处理表头行之间的边界（不含表体行界）。body_row=2 → 仅 boundary 0。
    n_header_boundaries = max(0, body_row - 1)
    max_boundary = min(n_header_boundaries, max_header_boundaries, n_rows - 1)
    if max_boundary <= 0:
        logger.info(
            "hline_repair: 无可修表头行界 (body_row=%d n_rows=%d)",
            body_row,
            n_rows,
        )
        return cells

    median_h = _median_text_h(text_boxes)
    strip_half = max(3.0, 0.20 * median_h)
    sliver_h = max(4.0, sliver_ratio * median_h)

    work = [dict(c) for c in cells]
    merges = 0

    for boundary in range(max_boundary - 1, -1, -1):
        y = float(row_seps[boundary + 1])
        upper_row = boundary
        lower_row = boundary + 1

        uppers: Dict[Tuple[int, int], List[int]] = {}
        lowers: Dict[Tuple[int, int], List[int]] = {}
        for idx, cell in enumerate(work):
            rs, re = int(cell["row_start"]), int(cell["row_end"])
            key = _same_col_key(cell)
            if rs == re == upper_row:
                uppers.setdefault(key, []).append(idx)
            elif rs == re == lower_row:
                lowers.setdefault(key, []).append(idx)

        to_drop: set = set()
        to_add: List[Dict[str, Any]] = []

        for key, u_idxs in uppers.items():
            l_idxs = lowers.get(key) or []
            if len(u_idxs) != 1 or len(l_idxs) != 1:
                continue
            ui, li = u_idxs[0], l_idxs[0]
            if ui in to_drop or li in to_drop:
                continue
            upper, lower = work[ui], work[li]
            x1, y1, x2, y2 = _cell_bbox(upper)
            lx1, ly1, lx2, ly2 = _cell_bbox(lower)
            xa, xb = min(x1, lx1), max(x2, lx2)

            ink = _hline_coverage_ratio(binary, y, xa, xb, tol=tol)
            if ink >= min_ink_ratio:
                continue

            u_texts = _ocr_texts_in_bbox(text_boxes, (x1, y1, x2, y2))
            l_texts = _ocr_texts_in_bbox(text_boxes, (lx1, ly1, lx2, ly2))
            u_joined = " ".join(u_texts)
            l_joined = " ".join(l_texts)
            u_h = max(0.0, y2 - y1)
            l_h = max(0.0, ly2 - ly1)
            u_frag = _is_fragment_text(u_joined)
            l_frag = _is_fragment_text(l_joined)
            u_sliver = u_h <= sliver_h and u_frag
            l_sliver = l_h <= sliver_h and l_frag
            crosses = _ocr_crosses_y_in_x(
                text_boxes, y, xa, xb, strip_half=strip_half
            )

            wrap_merge = False
            if (
                not u_frag
                and not l_frag
                and _substantive_cjk_label(u_joined)
                and _substantive_cjk_label(l_joined)
                and _same_column_wrap(upper, lower)
                and not _PARENT_HDR_RE.search(u_joined)
                and not _PARENT_HDR_RE.search(l_joined)
            ):
                wrap_merge = True

            evidence = (
                (u_frag and not l_frag)
                or (l_frag and not u_frag)
                or (u_frag and l_frag)
                or u_sliver
                or l_sliver
                or crosses
                or wrap_merge
            )
            if not evidence:
                continue

            merged = _merge_pair(upper, lower)
            to_drop.add(ui)
            to_drop.add(li)
            to_add.append(merged)
            merges += 1
            logger.debug(
                "hline_repair: merge col=%s rows=%d+%d ink=%.2f frag=(%s,%s) cross=%s wrap=%s",
                key,
                upper_row,
                lower_row,
                ink,
                u_frag,
                l_frag,
                crosses,
                wrap_merge,
            )

        if to_drop:
            work = [c for i, c in enumerate(work) if i not in to_drop]
            work.extend(to_add)

    if merges:
        logger.info(
            "hline_repair: 合并 %d 对表头格 (body_row=%d boundaries=0..%d)",
            merges,
            body_row,
            max_boundary - 1,
        )
    return work


def repair_colspans_by_vline_gaps(
    cells: List[Dict[str, Any]],
    binary: np.ndarray,
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    tol: float = 4.0,
    header_max_vline: float = 0.15,
    body_min_vline: float = 0.40,
) -> List[Dict[str, Any]]:
    """
    表头带内：若列界在表头 Y 带竖线弱、在表体 Y 带竖线强，则合并相邻表头格为 colspan。

    只改表头 col_* 与 polygon；表体列结构不动。
    """
    if not cells or binary is None or binary.size == 0:
        return cells

    text_boxes = list(text_boxes or [])
    row_seps, col_seps = _derive_seps(cells)
    if len(row_seps) < 3 or len(col_seps) < 3:
        return cells

    n_rows = len(row_seps) - 1
    body_row = _first_body_row_index(cells, text_boxes, row_seps)
    body_row = min(max(body_row, 1), n_rows - 1)
    if body_row <= 0:
        return cells

    body_y1 = float(row_seps[body_row])
    body_y2 = float(row_seps[min(body_row + 2, n_rows)])

    work = [dict(c) for c in cells]
    merges = 0

    for header_row in range(body_row):
        for col_boundary in range(len(col_seps) - 2, 0, -1):
            left_col = col_boundary - 1
            right_col = col_boundary
            left_idx = None
            right_idx = None
            for idx, cell in enumerate(work):
                rs, re = int(cell["row_start"]), int(cell["row_end"])
                cs, ce = int(cell["col_start"]), int(cell["col_end"])
                if rs != re or rs != header_row:
                    continue
                if max(int(cell.get("col_span") or 1), 1) != 1:
                    continue
                if cs == ce == left_col:
                    left_idx = idx
                elif cs == ce == right_col:
                    right_idx = idx
            if left_idx is None or right_idx is None:
                continue

            left, right = work[left_idx], work[right_idx]
            lx1, ly1, lx2, ly2 = _cell_bbox(left)
            rx1, ry1, rx2, ry2 = _cell_bbox(right)
            x = float(col_seps[col_boundary])

            header_v = _vline_coverage_ratio(binary, x, ly1, ly2, tol=tol)
            if header_v >= header_max_vline:
                continue

            body_v = _vline_coverage_ratio(binary, x, body_y1, body_y2, tol=tol)
            if body_v < body_min_vline:
                continue

            l_texts = _ocr_texts_in_bbox(text_boxes, (lx1, ly1, lx2, ly2))
            r_texts = _ocr_texts_in_bbox(text_boxes, (rx1, ry1, rx2, ry2))
            if _looks_like_two_independent_subheaders(
                " ".join(l_texts), " ".join(r_texts)
            ):
                continue

            merged = _merge_horizontal_pair(left, right)
            work[left_idx] = merged
            work.pop(right_idx)
            merges += 1
            logger.debug(
                "vline_repair: merge header row=%d cols %d+%d hv=%.2f bv=%.2f",
                header_row,
                left_col,
                right_col,
                header_v,
                body_v,
            )

    # --- Second pass: absorb empty header cells into left non-empty neighbor ---
    # Handles cases where a multi-colspan cell (e.g. colspan=5) sits next to
    # empty cells that should be part of the same parent header span.
    changed = True
    while changed:
        changed = False
        for header_row in range(body_row):
            row_cells = [
                (idx, c)
                for idx, c in enumerate(work)
                if int(c["row_start"]) == int(c["row_end"]) == header_row
            ]
            row_cells.sort(key=lambda t: int(t[1]["col_start"]))
            for ci in range(len(row_cells) - 1):
                l_idx, left = row_cells[ci]
                r_idx, right = row_cells[ci + 1]
                if int(left["col_end"]) + 1 != int(right["col_start"]):
                    continue
                r_texts = _ocr_texts_in_bbox(
                    text_boxes, _cell_bbox(right)
                )
                if any(t.strip() for t in r_texts):
                    continue
                l_texts = _ocr_texts_in_bbox(
                    text_boxes, _cell_bbox(left)
                )
                if not any(t.strip() for t in l_texts):
                    continue
                col_boundary = int(right["col_start"])
                if col_boundary < 1 or col_boundary >= len(col_seps):
                    continue
                x = float(col_seps[col_boundary])
                lx1, ly1, lx2, ly2 = _cell_bbox(left)
                header_v = _vline_coverage_ratio(
                    binary, x, ly1, ly2, tol=tol
                )
                if header_v >= header_max_vline:
                    continue
                body_v = _vline_coverage_ratio(
                    binary, x, body_y1, body_y2, tol=tol
                )
                if body_v < body_min_vline:
                    continue
                merged = _merge_horizontal_pair(left, right)
                work[l_idx] = merged
                work.pop(r_idx)
                merges += 1
                changed = True
                logger.debug(
                    "vline_repair(pass2): absorb empty into left "
                    "row=%d cols %d..%d+%d hv=%.2f bv=%.2f",
                    header_row,
                    int(left["col_start"]),
                    int(left["col_end"]),
                    int(right["col_start"]),
                    header_v,
                    body_v,
                )
                break
            if changed:
                break

    if merges:
        logger.info(
            "vline_repair: 合并 %d 对表头 colspan (body_row=%d)",
            merges,
            body_row,
        )
        work = dedupe_overlapping_cells(work)
    return work
