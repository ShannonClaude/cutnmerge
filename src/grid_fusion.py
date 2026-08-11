"""Fuse TSR topology with line-derived grid separators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lines import DetectedTable, Separator
from .tsr_refine import _derive_seps


@dataclass
class _UnionFind:
    parent: List[int]
    rank: List[int]

    @classmethod
    def create(cls, n: int) -> "_UnionFind":
        return cls(parent=list(range(n)), rank=[0] * n)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _cell_bbox(cell: Dict[str, Any]) -> Tuple[float, float, float, float]:
    poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _make_cell(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> Dict[str, Any]:
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
    return {
        "polygon": polygon,
        "x_key": x1i,
        "y_key": y1i,
        "row_start": int(row_start),
        "row_end": int(row_end),
        "col_start": int(col_start),
        "col_end": int(col_end),
        "row_span": int(row_end - row_start + 1),
        "col_span": int(col_end - col_start + 1),
        "texts": [],
        "text": "",
    }


def _shape(cells: Sequence[Dict[str, Any]]) -> tuple[int, int]:
    if not cells:
        return 0, 0
    return (
        max(int(c["row_end"]) for c in cells) + 1,
        max(int(c["col_end"]) for c in cells) + 1,
    )


def _full_span_separators(seps: Sequence[float], total: float) -> List[Separator]:
    return [
        # 这里生成的是「TSR 推导出的假分隔线」，不应作为强证据阻止合并。
        # 将 length 置 0，使得 coverage_ratio≈0，从而 _separator_strong 永远为 False。
        Separator(coord=float(s), spans=[(0.0, float(total))], length=0.0)
        for s in seps
    ]


def _best_seps(tsr_cells: Sequence[Dict[str, Any]], table: DetectedTable) -> tuple[List[float], List[float], List[Separator], List[Separator]]:
    tsr_row_seps, tsr_col_seps = _derive_seps(list(tsr_cells)) if tsr_cells else ([], [])
    line_row_seps = [float(s) for s in table.row_seps]
    line_col_seps = [float(s) for s in table.col_seps]
    total_h = max(float(table.bbox[3] - table.bbox[1]), 1.0)
    total_w = max(float(table.bbox[2] - table.bbox[0]), 1.0)

    row_seps = line_row_seps if len(line_row_seps) >= max(len(tsr_row_seps), 3) else tsr_row_seps
    row_sps = (
        list(table.h_separators)
        if row_seps is line_row_seps
        else _full_span_separators(row_seps, total_w)
    )
    col_seps = line_col_seps if len(line_col_seps) >= len(tsr_col_seps) else tsr_col_seps
    col_sps = (
        list(table.v_separators)
        if col_seps is line_col_seps
        else _full_span_separators(col_seps, total_h)
    )
    return row_seps, col_seps, row_sps, col_sps


def _pick_table(tsr_cells: Sequence[Dict[str, Any]], tables: Sequence[DetectedTable]) -> Optional[DetectedTable]:
    if not tables:
        return None
    if not tsr_cells:
        return max(tables, key=lambda t: (t.confidence, len(t.cells)))
    xs: List[float] = []
    ys: List[float] = []
    for cell in tsr_cells:
        x1, y1, x2, y2 = _cell_bbox(cell)
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    tsr_bbox = (min(xs), min(ys), max(xs), max(ys))
    return max(
        tables,
        key=lambda t: (_bbox_iou(tsr_bbox, t.bbox), t.confidence, len(t.cells)),
    )


def _separator_strong(sep: Separator, total: float, min_ratio: float = 0.55) -> bool:
    return sep.coverage_ratio(total) >= min_ratio


def fuse_tsr_with_lines(
    tsr_cells: List[Dict[str, Any]],
    tables: Sequence[DetectedTable],
    *,
    min_table_conf: float = 0.45,
    min_gain_cols: int = 1,
    merge_cover_thresh: float = 0.85,
) -> List[Dict[str, Any]]:
    """Use line grid as atomic separators and TSR as merge prior."""
    table = _pick_table(tsr_cells, tables)
    if table is None:
        return tsr_cells

    row_seps, col_seps, h_separators, v_separators = _best_seps(tsr_cells, table)
    n_rows = max(len(row_seps) - 1, 0)
    n_cols = max(len(col_seps) - 1, 0)
    tsr_rows, tsr_cols = _shape(tsr_cells)
    if n_rows < 1 or n_cols < 2:
        return tsr_cells
    if table.confidence < min_table_conf:
        return tsr_cells
    if n_cols < tsr_cols + min_gain_cols and n_rows < tsr_rows:
        return tsr_cells

    total_h = max(float(table.bbox[3] - table.bbox[1]), 1.0)
    total_w = max(float(table.bbox[2] - table.bbox[0]), 1.0)
    uf = _UnionFind.create(n_rows * n_cols)
    owner_grid: List[Optional[int]] = [None] * (n_rows * n_cols)
    line_cells = list(table.cells or [])
    line_owner_grid: List[Optional[int]] = [None] * (n_rows * n_cols)

    def idx(r: int, c: int) -> int:
        return r * n_cols + c

    def best_owner(r: int, c: int) -> Optional[int]:
        x1 = float(col_seps[c])
        x2 = float(col_seps[c + 1])
        y1 = float(row_seps[r])
        y2 = float(row_seps[r + 1])
        best_i = None
        best_score = 0.0
        for i, cell in enumerate(tsr_cells):
            cx1, cy1, cx2, cy2 = _cell_bbox(cell)
            ix1 = max(x1, cx1)
            iy1 = max(y1, cy1)
            ix2 = min(x2, cx2)
            iy2 = min(y2, cy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            atom_area = max((x2 - x1) * (y2 - y1), 1.0)
            score = inter / atom_area
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is not None and best_score >= 0.35:
            return best_i
        for i, cell in enumerate(line_cells):
            rs, re = int(cell["row_start"]), int(cell["row_end"])
            cs, ce = int(cell["col_start"]), int(cell["col_end"])
            if rs <= r <= re and cs <= c <= ce:
                return None
        return None

    for r in range(n_rows):
        for c in range(n_cols):
            owner_grid[idx(r, c)] = best_owner(r, c)
            cx = (float(col_seps[c]) + float(col_seps[c + 1])) / 2.0
            cy = (float(row_seps[r]) + float(row_seps[r + 1])) / 2.0
            for li, cell in enumerate(line_cells):
                rs, re = int(cell["row_start"]), int(cell["row_end"])
                cs, ce = int(cell["col_start"]), int(cell["col_end"])
                if rs >= len(table.row_seps) - 1 or cs >= len(table.col_seps) - 1:
                    continue
                x1, y1, x2, y2 = _cell_bbox(cell)
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    line_owner_grid[idx(r, c)] = li
                    break

    for r in range(n_rows):
        y0 = float(row_seps[r])
        y1 = float(row_seps[r + 1])
        for c in range(n_cols - 1):
            a = idx(r, c)
            b = idx(r, c + 1)
            owner_a = owner_grid[a]
            owner_b = owner_grid[b]
            if owner_a is None or owner_a != owner_b:
                continue
            sep = v_separators[c + 1]
            if _separator_strong(sep, total_h) and sep.covers(y0, y1, thresh=merge_cover_thresh):
                continue
            uf.union(a, b)

    for c in range(n_cols):
        x0 = float(col_seps[c])
        x1 = float(col_seps[c + 1])
        for r in range(n_rows - 1):
            a = idx(r, c)
            b = idx(r + 1, c)
            owner_a = owner_grid[a]
            owner_b = owner_grid[b]
            # 纵向合并优先使用 TSR 的同一逻辑 cell 归属作为原子证据；
            # 若 TSR 归属不稳定，则允许用 line_owner_grid 作为兜底证据。
            tsr_ok = owner_a is not None and owner_a == owner_b
            if not tsr_ok:
                la = line_owner_grid[a]
                lb = line_owner_grid[b]
                if la is None or la != lb:
                    continue
            sep = h_separators[r + 1]
            if _separator_strong(sep, total_w) and sep.covers(x0, x1, thresh=merge_cover_thresh):
                continue
            uf.union(a, b)

    groups: Dict[int, List[Tuple[int, int]]] = {}
    for r in range(n_rows):
        for c in range(n_cols):
            groups.setdefault(uf.find(idx(r, c)), []).append((r, c))

    out: List[Dict[str, Any]] = []
    for atoms in groups.values():
        rs = [a[0] for a in atoms]
        cs = [a[1] for a in atoms]
        row_start, row_end = min(rs), max(rs)
        col_start, col_end = min(cs), max(cs)
        expected = (row_end - row_start + 1) * (col_end - col_start + 1)
        if len(atoms) != expected:
            for rr, cc in atoms:
                out.append(
                    _make_cell(
                        float(col_seps[cc]),
                        float(row_seps[rr]),
                        float(col_seps[cc + 1]),
                        float(row_seps[rr + 1]),
                        rr,
                        rr,
                        cc,
                        cc,
                    )
                )
            continue
        out.append(
            _make_cell(
                float(col_seps[col_start]),
                float(row_seps[row_start]),
                float(col_seps[col_end + 1]),
                float(row_seps[row_end + 1]),
                row_start,
                row_end,
                col_start,
                col_end,
            )
        )
    return out or tsr_cells
