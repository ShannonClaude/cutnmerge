"""框线网格后处理：按 OCR 文本聚类补列、拆错误纵向合并。

在 detect_tables 之后、文本归属之前调用，修补：
1. 图上无竖线但多列文本并列的情况（如 P98 酸当量|双键当量）；
2. 假竖线剔除后变宽的双列区（如 JP B1|4g、C1|30g）；
3. 横线断口导致的错误 rowspan（如 JP 连续 50g/4g 挤一格）。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lines import DetectedTable, Separator, build_cells_from_separators

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]"
)


def _tb_bbox(tb: Dict[str, Any]) -> Tuple[float, float, float, float]:
    poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )


def _cell_bbox(cell: Dict[str, Any]) -> Tuple[float, float, float, float]:
    poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )


def _row_index(cy: float, row_seps: Sequence[float]) -> int:
    for r in range(len(row_seps) - 1):
        if row_seps[r] <= cy < row_seps[r + 1]:
            return r
    if row_seps and cy >= row_seps[-1]:
        return len(row_seps) - 2
    return -1


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _rebuild_table(
    table: DetectedTable,
    h_seps: List[Separator],
    v_seps: List[Separator],
    *,
    merge_cover_thresh: float = 0.55,
) -> DetectedTable:
    cells, conf = build_cells_from_separators(
        h_seps, v_seps, merge_cover_thresh=merge_cover_thresh
    )
    if not cells:
        return table
    return DetectedTable(
        cells=cells,
        bbox=table.bbox,
        row_seps=[s.coord for s in h_seps],
        col_seps=[s.coord for s in v_seps],
        confidence=conf if conf > 0 else table.confidence,
        h_separators=h_seps,
        v_separators=v_seps,
    )


def _find_column_split(
    col_left: float,
    col_right: float,
    row_seps: Sequence[float],
    boxes: Sequence[Dict[str, Any]],
    *,
    min_gap_frac: float = 0.30,
    min_support_rows: int = 2,
    max_box_frac: float = 0.60,
) -> Optional[Tuple[float, List[Tuple[float, float]]]]:
    """
    在 [col_left, col_right] 内找文本双簇分界。

    Returns:
        (split_x, spans) 或 None。spans 只覆盖有左右两簇证据的行区间。
    """
    width = col_right - col_left
    if width <= 8:
        return None

    inside: List[Dict[str, Any]] = []
    for tb in boxes:
        x1, y1, x2, y2 = _tb_bbox(tb)
        if x1 < col_left - 3 or x2 > col_right + 3:
            continue
        if (x2 - x1) > max_box_frac * width:
            continue  # 排除跨列表头
        inside.append(tb)
    if len(inside) < 4:
        return None

    inside.sort(key=lambda tb: (_tb_bbox(tb)[0] + _tb_bbox(tb)[2]) / 2.0)
    centers = [(_tb_bbox(tb)[0] + _tb_bbox(tb)[2]) / 2.0 for tb in inside]
    gaps = np.diff(centers)
    if len(gaps) == 0:
        return None
    k = int(np.argmax(gaps))
    if float(gaps[k]) < min_gap_frac * width:
        return None

    thr = (centers[k] + centers[k + 1]) / 2.0
    left = inside[: k + 1]
    right = inside[k + 1 :]
    if not left or not right:
        return None

    def rows_of(group: Sequence[Dict[str, Any]]) -> set:
        out = set()
        for tb in group:
            _, y1, _, y2 = _tb_bbox(tb)
            ri = _row_index((y1 + y2) / 2.0, row_seps)
            if ri >= 0:
                out.add(ri)
        return out

    both = rows_of(left) & rows_of(right)
    if len(both) < min_support_rows:
        return None

    max_r = max(_tb_bbox(tb)[2] for tb in left)
    min_l = min(_tb_bbox(tb)[0] for tb in right)
    split_x = (max_r + min_l) / 2.0 if min_l > max_r else thr

    # spans：只覆盖有证据的行（表头合并区保持不动）
    spans: List[Tuple[float, float]] = []
    for ri in sorted(both):
        if ri < 0 or ri >= len(row_seps) - 1:
            continue
        spans.append((float(row_seps[ri]), float(row_seps[ri + 1])))
    if not spans:
        return None
    # 合并相邻行区间
    spans.sort()
    merged: List[List[float]] = [[spans[0][0], spans[0][1]]]
    for a, b in spans[1:]:
        if a <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return split_x, [(float(a), float(b)) for a, b in merged]


def split_columns_by_text_clusters(
    table: DetectedTable,
    boxes: Sequence[Dict[str, Any]],
    *,
    max_iters: int = 2,
) -> DetectedTable:
    """
    对过宽原子列按 OCR 文本双簇插入竖线，重建网格。

    只在宽度 ≥ 中位列宽、且 ≥2 行同时出现左右两簇时触发。
    """
    if not table.v_separators or not table.h_separators or not boxes:
        return table

    current = table
    for _ in range(max_iters):
        cols = sorted(s.coord for s in current.v_separators)
        if len(cols) < 3:
            break
        widths = np.diff(cols)
        med_w = float(np.median(widths))
        if med_w < 8:
            break

        inserted = False
        v_seps = list(current.v_separators)
        existing = [s.coord for s in v_seps]

        for i in range(len(cols) - 1):
            a, b = cols[i], cols[i + 1]
            w = b - a
            if w < med_w:
                continue
            found = _find_column_split(a, b, current.row_seps, boxes)
            if found is None:
                continue
            split_x, spans = found
            if any(abs(split_x - x) < max(6.0, 0.15 * med_w) for x in existing):
                continue
            length = sum(e - s for s, e in spans)
            v_seps.append(Separator(coord=float(split_x), spans=spans, length=float(length)))
            existing.append(split_x)
            inserted = True
            logger.info(
                "文本聚类补列: x=%.1f in [%.1f,%.1f] spans=%d",
                split_x,
                a,
                b,
                len(spans),
            )

        if not inserted:
            break
        v_seps = sorted(v_seps, key=lambda s: s.coord)
        current = _rebuild_table(current, list(current.h_separators), v_seps)

    return current


def unmerge_row_spanned_cells(
    table: DetectedTable,
    boxes: Sequence[Dict[str, Any]],
) -> DetectedTable:
    """
    拆掉「框数 == 行数、每行恰好 1 框、文本无 CJK 或全相同」的错误纵向合并。

    专治横线断口导致的 50g×8 / 4g×8 挤一格；不动多行长表头。
    """
    if not table.cells or not boxes or len(table.row_seps) < 3:
        return table

    row_seps = table.row_seps
    col_seps = table.col_seps
    new_cells: List[Dict[str, Any]] = []
    changed = False

    for cell in table.cells:
        row_span = int(cell.get("row_span") or 1)
        col_span = int(cell.get("col_span") or 1)
        if col_span != 1 or row_span < 2:
            new_cells.append(cell)
            continue

        cx1, cy1, cx2, cy2 = _cell_bbox(cell)
        contained: List[Dict[str, Any]] = []
        for tb in boxes:
            x1, y1, x2, y2 = _tb_bbox(tb)
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if cx1 <= mx <= cx2 and cy1 <= my <= cy2:
                contained.append(tb)

        if len(contained) != row_span:
            new_cells.append(cell)
            continue

        # 每原子行恰好 1 框，且无框跨越内部行界
        row_start = int(cell["row_start"])
        row_end = int(cell["row_end"])
        by_row: Dict[int, List[Dict[str, Any]]] = {r: [] for r in range(row_start, row_end + 1)}
        ok = True
        for tb in contained:
            x1, y1, x2, y2 = _tb_bbox(tb)
            my = (y1 + y2) / 2.0
            ri = _row_index(my, row_seps)
            if ri < row_start or ri > row_end:
                ok = False
                break
            # 框不得跨越内部行界（容差 0.25×框高）
            box_h = max(y2 - y1, 1.0)
            tol = 0.25 * box_h
            for boundary in row_seps[row_start + 1 : row_end + 1]:
                if y1 + tol < boundary < y2 - tol:
                    ok = False
                    break
            if not ok:
                break
            by_row[ri].append(tb)
        if not ok or any(len(v) != 1 for v in by_row.values()):
            new_cells.append(cell)
            continue

        texts = [str(by_row[r][0].get("text") or "").strip() for r in range(row_start, row_end + 1)]
        if not texts:
            new_cells.append(cell)
            continue
        # 全部无 CJK，或全部文本相同
        all_same = all(t == texts[0] for t in texts)
        none_cjk = all(not _has_cjk(t) for t in texts)
        if not (all_same or none_cjk):
            new_cells.append(cell)
            continue

        # 拆成逐行原子格
        col_start = int(cell["col_start"])
        for r in range(row_start, row_end + 1):
            if r >= len(row_seps) - 1 or col_start >= len(col_seps) - 1:
                continue
            x1 = float(col_seps[col_start])
            x2 = float(col_seps[col_start + 1])
            y1 = float(row_seps[r])
            y2 = float(row_seps[r + 1])
            inset = 1.0
            x1i, x2i = x1 + inset, x2 - inset
            y1i, y2i = y1 + inset, y2 - inset
            if x2i <= x1i:
                x1i, x2i = x1, x2
            if y2i <= y1i:
                y1i, y2i = y1, y2
            polygon = np.array(
                [[x1i, y1i], [x2i, y1i], [x2i, y2i], [x1i, y2i]],
                dtype=np.float64,
            )
            new_cells.append(
                {
                    "polygon": polygon,
                    "x_key": x1i,
                    "y_key": y1i,
                    "row_start": r,
                    "row_end": r,
                    "col_start": col_start,
                    "col_end": col_start,
                    "row_span": 1,
                    "col_span": 1,
                    "texts": [],
                    "text": "",
                }
            )
        changed = True
        logger.info(
            "拆错误纵向合并: r%d-%d c%d texts=%s",
            row_start,
            row_end,
            col_start,
            texts[:3],
        )

    if not changed:
        return table

    return DetectedTable(
        cells=new_cells,
        bbox=table.bbox,
        row_seps=table.row_seps,
        col_seps=table.col_seps,
        confidence=table.confidence,
        h_separators=table.h_separators,
        v_separators=table.v_separators,
    )


def refine_table(
    table: DetectedTable,
    boxes: Sequence[Dict[str, Any]],
) -> DetectedTable:
    """对单表依次：文本聚类补列 → 拆错误纵向合并。"""
    table = split_columns_by_text_clusters(table, boxes)
    table = unmerge_row_spanned_cells(table, boxes)
    return table
