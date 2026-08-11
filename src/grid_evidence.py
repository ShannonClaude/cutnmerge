"""
网格证据校验层：在 TSR cells 落定后、OCR 文本归属之前修正“错分行/列边界”。

目标：防止出现类似
  - 表头折行被切成多行（span 被系统性摧毁）
  - 幽灵行/列导致 colspan/rowspan 错位

原则（证据驱动，不做专利/表格内容硬编码）：
  1. 内部行边界（row_i 与 row_{i+1} 之间）如果“没有线证据且也没有空白走廊证据”，则合并它。
  2. 内部列边界同理。

证据定义（尽量宽容，避免误杀真框线）：
  - 线证据：来自 `lines.detect_tables()` 的水平/竖分隔线（Separator）在该边界坐标附近且覆盖率足够。
  - 走廊空白证据：OCR 文本框与边界附近条带（strip）没有交叠。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .grid_fusion import _cell_bbox as _cell_bbox_physical
from .lines import DetectedTable
from .tsr_refine import _derive_seps


def _tb_bbox(tb: Dict[str, Any]) -> Tuple[float, float, float, float]:
    poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )


def _cells_bbox(cells: Sequence[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    xs1, ys1, xs2, ys2 = [], [], [], []
    for c in cells:
        x1, y1, x2, y2 = _cell_bbox_physical(c)
        xs1.append(x1)
        ys1.append(y1)
        xs2.append(x2)
        ys2.append(y2)
    return min(xs1), min(ys1), max(xs2), max(ys2)


def _median_text_box_h(text_boxes: Sequence[Dict[str, Any]]) -> float:
    hs: List[float] = []
    for tb in text_boxes:
        _, y1, _, y2 = _tb_bbox(tb)
        h = max(0.0, y2 - y1)
        if h > 1e-3:
            hs.append(h)
    return float(np.median(hs)) if hs else 10.0


def _tb_crosses_strip_y(tb: Dict[str, Any], strip_y_lo: float, strip_y_hi: float) -> bool:
    """
    严格的“跨边界”判定：
    只有当 OCR bbox 同时覆盖了 strip 的上下两侧时，才认为它跨过了该边界。
    这能显著降低“紧贴边界的轻微重叠”造成的误判合并。
    """
    _x1, y1, _x2, y2 = _tb_bbox(tb)
    return y1 < strip_y_lo and y2 > strip_y_hi


def _tb_crosses_strip_x(tb: Dict[str, Any], strip_x_lo: float, strip_x_hi: float) -> bool:
    """
    严格的“跨边界”判定：bbox 必须横跨 strip 左右两侧。
    """
    x1, _y1, x2, _y2 = _tb_bbox(tb)
    return x1 < strip_x_lo and x2 > strip_x_hi


def _line_evidence_y(
    y: float,
    line_tables: Sequence[DetectedTable],
    *,
    tol: float = 6.0,
    min_cov_ratio: float = 0.30,
) -> bool:
    for t in line_tables:
        total_w = max(float(t.bbox[2] - t.bbox[0]), 1.0)
        for sep in t.h_separators:
            if abs(float(sep.coord) - float(y)) <= tol and sep.coverage_ratio(total_w) >= min_cov_ratio:
                return True
    return False


def _line_evidence_x(
    x: float,
    line_tables: Sequence[DetectedTable],
    *,
    tol: float = 6.0,
    min_cov_ratio: float = 0.30,
) -> bool:
    for t in line_tables:
        total_h = max(float(t.bbox[3] - t.bbox[1]), 1.0)
        for sep in t.v_separators:
            if abs(float(sep.coord) - float(x)) <= tol and sep.coverage_ratio(total_h) >= min_cov_ratio:
                return True
    return False


def _merge_row_indices(
    cells: List[Dict[str, Any]],
    *,
    n_rows: int,
    merge_lower_row_indices: Sequence[int],
) -> None:
    # 从大到小合并，避免索引位移影响尚未处理的边界
    for i in sorted(set(int(x) for x in merge_lower_row_indices), reverse=True):
        if i < 0 or i >= n_rows - 1:
            continue
        for cell in cells:
            rs = int(cell["row_start"])
            re = int(cell["row_end"])
            # 边界下移：所有 > i 的行索引整体 -1
            new_rs = rs if rs <= i else rs - 1
            new_re = re if re <= i else re - 1
            if new_re < new_rs:
                new_re = new_rs
            cell["row_start"] = int(new_rs)
            cell["row_end"] = int(new_re)
            cell["row_span"] = int(new_re - new_rs + 1)


def _merge_col_indices(
    cells: List[Dict[str, Any]],
    *,
    n_cols: int,
    merge_left_col_indices: Sequence[int],
) -> None:
    for j in sorted(set(int(x) for x in merge_left_col_indices), reverse=True):
        if j < 0 or j >= n_cols - 1:
            continue
        for cell in cells:
            cs = int(cell["col_start"])
            ce = int(cell["col_end"])
            new_cs = cs if cs <= j else cs - 1
            new_ce = ce if ce <= j else ce - 1
            if new_ce < new_cs:
                new_ce = new_cs
            cell["col_start"] = int(new_cs)
            cell["col_end"] = int(new_ce)
            cell["col_span"] = int(new_ce - new_cs + 1)


def apply_grid_evidence_merge(
    cells: List[Dict[str, Any]],
    text_boxes: Sequence[Dict[str, Any]],
    *,
    line_tables: Optional[Sequence[DetectedTable]] = None,
    tol_coord: float = 6.0,
    # strip 取 OCR 文本高度尺度，避免把折行/换行小间隙当边界
    strip_half_ratio: float = 0.20,
    strip_min_half: float = 4.0,
    min_line_cov_ratio: float = 0.30,
) -> List[Dict[str, Any]]:
    """
    返回“逻辑拓扑合并后”的 cells（原地修改并返回同一引用）。

    该层只改 row_start/row_end / col_start/col_end，不强制重建 polygon。
    """
    if not cells:
        return cells

    line_tables = list(line_tables) if line_tables else []
    row_seps, col_seps = _derive_seps(cells)
    if len(row_seps) < 3 or len(col_seps) < 3:
        return cells

    n_rows = len(row_seps) - 1
    n_cols = len(col_seps) - 1
    if n_rows < 2 and n_cols < 2:
        return cells

    x1, y1, x2, y2 = _cells_bbox(cells)
    # 只用表格区域内的 OCR 证据，避免表外文字干扰
    relevant_tbs: List[Dict[str, Any]] = []
    for tb in text_boxes:
        bx1, by1, bx2, by2 = _tb_bbox(tb)
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            relevant_tbs.append(tb)

    median_h = _median_text_box_h(relevant_tbs)
    strip_half = max(strip_min_half, strip_half_ratio * median_h)

    # ---------- Decide row boundaries to merge ----------
    merge_rows: List[int] = []
    for i in range(n_rows - 1):
        y = float(row_seps[i + 1])
        strip_lo, strip_hi = y - strip_half, y + strip_half
        crosses = False
        for tb in relevant_tbs:
            if _tb_crosses_strip_y(tb, strip_lo, strip_hi):
                crosses = True
                break

        line_ok = False
        if line_tables:
            line_ok = _line_evidence_y(
                y,
                line_tables,
                tol=tol_coord,
                min_cov_ratio=min_line_cov_ratio,
            )

        # 两种证据都没有：倾向于“错切边界”，合并
        blank_corridor = not crosses
        if (not line_ok) and (not blank_corridor):
            merge_rows.append(i)

    # ---------- Decide col boundaries to merge ----------
    merge_cols: List[int] = []
    for j in range(n_cols - 1):
        x = float(col_seps[j + 1])
        strip_lo, strip_hi = x - strip_half, x + strip_half
        crosses = False
        for tb in relevant_tbs:
            if _tb_crosses_strip_x(tb, strip_lo, strip_hi):
                crosses = True
                break

        line_ok = False
        if line_tables:
            line_ok = _line_evidence_x(
                x,
                line_tables,
                tol=tol_coord,
                min_cov_ratio=min_line_cov_ratio,
            )

        blank_corridor = not crosses
        if (not line_ok) and (not blank_corridor):
            merge_cols.append(j)

    if merge_rows:
        _merge_row_indices(cells, n_rows=n_rows, merge_lower_row_indices=merge_rows)
    if merge_cols:
        _merge_col_indices(cells, n_cols=n_cols, merge_left_col_indices=merge_cols)

    return cells

