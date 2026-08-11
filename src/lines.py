"""基于 OpenCV 框线检测重建表格网格。

核心策略（针对有框线专利表）：
1. Otsu 二值化 + 形态学 OPEN 提取横/竖线掩膜；
2. 投影峰值找分隔线（阈值相对表宽/高）；
3. 覆盖率 + 交点投票过滤文字笔画；
4. 原子网格 + 缺失内部线段 → 并查集合并单元格。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MERGE_COVER_THRESH = 0.55
_MIN_TABLE_AREA_RATIO = 0.015
_CONFIDENCE_THRESH = 0.35
_HLINE_MIN_COVER = 0.30
_VLINE_MIN_COVER = 0.35
_MIN_INTERSECTIONS = 2


@dataclass
class Separator:
    coord: float
    spans: List[Tuple[float, float]] = field(default_factory=list)
    length: float = 0.0

    def coverage_ratio(self, total: float) -> float:
        if total <= 1e-6:
            return 0.0
        return float(self.length) / float(total)

    def covers(self, a: float, b: float, thresh: float = _MERGE_COVER_THRESH) -> bool:
        lo, hi = (a, b) if a <= b else (b, a)
        span = hi - lo
        if span <= 1e-6:
            return True
        covered = 0.0
        for s, e in self.spans:
            covered += max(0.0, min(e, hi) - max(s, lo))
        return (covered / span) >= thresh


@dataclass
class DetectedTable:
    cells: List[Dict[str, Any]]
    bbox: Tuple[int, int, int, int]
    row_seps: List[float]
    col_seps: List[float]
    confidence: float
    h_separators: List[Separator] = field(default_factory=list)
    v_separators: List[Separator] = field(default_factory=list)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

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


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def binarize_otsu(image: np.ndarray) -> np.ndarray:
    gray = _to_gray(image)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return otsu


def binarize_adaptive(image: np.ndarray) -> np.ndarray:
    gray = _to_gray(image)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, -2
    )


def binarize(image: np.ndarray) -> np.ndarray:
    """兼容旧接口：返回 Otsu（主路径）。"""
    return binarize_otsu(image)


def _line_masks(binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """MORPH_OPEN 提取横/竖线（不做 CLOSE）。"""
    h, w = binary.shape[:2]
    hk = max(25, w // 40)
    vk = max(15, min(h // 35, 30))
    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    hmask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hker, iterations=1)
    vmask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vker, iterations=1)
    return hmask, vmask


def _vline_mask(binary: np.ndarray, vk: int) -> np.ndarray:
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(vk, 5)))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, vker, iterations=1)


def _merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    sorted_iv = sorted((min(a, b), max(a, b)) for a, b in intervals)
    merged: List[List[float]] = [[sorted_iv[0][0], sorted_iv[0][1]]]
    for a, b in sorted_iv[1:]:
        if a <= merged[-1][1] + 3:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(float(a), float(b)) for a, b in merged]


def _runs_from_binary_row(row: np.ndarray, min_run: int = 3) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    n = len(row)
    i = 0
    while i < n:
        if row[i]:
            j = i + 1
            while j < n and row[j]:
                j += 1
            if j - i >= min_run:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _group_peaks(peaks: List[int], tol: int = 3) -> List[List[int]]:
    if not peaks:
        return []
    groups: List[List[int]] = [[peaks[0]]]
    for i in peaks[1:]:
        if i - groups[-1][-1] <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _separators_from_projection(
    mask: np.ndarray,
    *,
    horizontal: bool,
    min_proj_ratio: float,
    cluster_tol: int = 3,
    min_seg_len: int = 8,
) -> List[Separator]:
    h, w = mask.shape[:2]
    fg = (mask > 0).astype(np.uint8)

    if horizontal:
        proj = fg.sum(axis=1).astype(np.float64)
        total = float(w)
        axis_len = h
    else:
        proj = fg.sum(axis=0).astype(np.float64)
        total = float(h)
        axis_len = w

    thresh = max(total * min_proj_ratio, 10.0)
    peaks = [i for i, v in enumerate(proj) if v >= thresh]
    if not peaks:
        return []

    groups = _group_peaks(peaks, tol=cluster_tol)
    separators: List[Separator] = []
    for g in groups:
        weights = proj[g]
        coord = float(np.average(g, weights=weights))
        band = fg[min(g) : max(g) + 1] if horizontal else fg[:, min(g) : max(g) + 1]
        union_1d = band.any(axis=0) if horizontal else band.any(axis=1)
        runs = _runs_from_binary_row(union_1d, min_run=min_seg_len)
        if not runs:
            continue
        spans = _merge_intervals([(float(a), float(b)) for a, b in runs])
        length = sum(e - s for s, e in spans)
        separators.append(Separator(coord=coord, spans=spans, length=float(length)))
    return separators


def _count_intersections(
    sep: Separator,
    others: Sequence[Separator],
    *,
    sep_is_horizontal: bool,
    hit_tol: float = 4.0,
) -> int:
    count = 0
    for other in others:
        if sep_is_horizontal:
            x_ok = any(s - hit_tol <= other.coord <= e + hit_tol for s, e in sep.spans)
            y_ok = other.covers(sep.coord - hit_tol, sep.coord + hit_tol, thresh=0.01)
        else:
            x_ok = any(s - hit_tol <= sep.coord <= e + hit_tol for s, e in other.spans)
            y_ok = sep.covers(other.coord - hit_tol, other.coord + hit_tol, thresh=0.01)
        if x_ok and y_ok:
            count += 1
    return count


def _dedupe_close(seps: List[Separator], min_gap: float = 5.0) -> List[Separator]:
    if not seps:
        return []
    ordered = sorted(seps, key=lambda s: s.coord)
    out: List[Separator] = [ordered[0]]
    for sep in ordered[1:]:
        if sep.coord - out[-1].coord < min_gap:
            if sep.length > out[-1].length:
                out[-1] = sep
        else:
            out.append(sep)
    return out


def _adaptive_min_gap(seps: List[Separator], floor: float = 5.0, ratio: float = 0.25) -> float:
    """按分隔线间距中位数估计去重最小间距，消除粗线双峰；ratio 宜偏小以免合并真分区线。"""
    if len(seps) < 3:
        return floor
    coords = sorted(s.coord for s in seps)
    diffs = np.diff(coords)
    big = diffs[diffs >= floor]
    if len(big) == 0:
        return floor
    med = float(np.median(big))
    return max(floor, med * ratio)


def _crossing_ratio(v: Separator, h_seps: Sequence[Separator]) -> float:
    """
    竖线 spans 在横线坐标处的连续性。

    真框线穿过横线（spans 连通），文字竖笔堆叠在横线处断开。
    证据不足（内部横线 < 4）时返回 1.0，避免误杀短真线。
    """
    if not v.spans or not h_seps:
        return 1.0
    lo = min(a for a, b in v.spans)
    hi = max(b for a, b in v.spans)
    inner = [s.coord for s in h_seps if lo + 4.0 < s.coord < hi - 4.0]
    if len(inner) < 4:
        return 1.0
    hit = 0
    for y in inner:
        if any(a - 3.0 <= y <= b + 3.0 for a, b in v.spans):
            hit += 1
    return float(hit) / float(len(inner))


def _filter_separators(
    h_seps: List[Separator],
    v_seps: List[Separator],
    table_w: float,
    table_h: float,
) -> Tuple[List[Separator], List[Separator]]:
    h_anchor = [s for s in h_seps if s.coverage_ratio(table_w) >= _HLINE_MIN_COVER]
    v_anchor = [s for s in v_seps if s.coverage_ratio(table_h) >= 0.50]
    if len(h_anchor) < 2:
        h_anchor = sorted(h_seps, key=lambda s: -s.length)[: max(3, len(h_seps))]
    if len(v_anchor) < 2:
        v_anchor = sorted(v_seps, key=lambda s: -s.length)[: max(3, len(v_seps))]

    def keep_h(sep: Separator, v_ref: Sequence[Separator]) -> bool:
        if sep.coverage_ratio(table_w) >= _HLINE_MIN_COVER:
            return True
        return _count_intersections(sep, v_ref, sep_is_horizontal=True) >= _MIN_INTERSECTIONS

    def keep_v(sep: Separator, h_ref: Sequence[Separator]) -> bool:
        cover = sep.coverage_ratio(table_h)
        if cover < 0.15:
            return False
        # 通高真线直接保留；其余须跨行连续（挡掉文字竖笔堆叠）
        if cover >= 0.85:
            return True
        if _crossing_ratio(sep, h_ref) < 0.35:
            return False
        # 弱/中等竖线：仍要求与足够多横线相交
        if cover >= _VLINE_MIN_COVER:
            return True
        need = 4 if cover < 0.45 else max(_MIN_INTERSECTIONS, min(3, len(h_ref) // 3))
        return _count_intersections(sep, h_ref, sep_is_horizontal=False) >= need

    h_keep = [s for s in h_seps if keep_h(s, v_anchor)]
    v_keep = [s for s in v_seps if keep_v(s, h_keep or h_anchor)]

    if len(h_keep) >= 2 and len(v_keep) >= 2:
        h_keep = [s for s in h_keep if keep_h(s, v_keep)]
        v_keep = [s for s in v_keep if keep_v(s, h_keep)]

    h_gap = _adaptive_min_gap(h_keep, floor=5.0, ratio=0.35)
    v_gap = _adaptive_min_gap(v_keep, floor=5.0, ratio=0.12)
    h_keep = _dedupe_close(h_keep, h_gap)
    v_keep = _dedupe_close(v_keep, v_gap)
    v_keep = _suppress_shadow_vlines(v_keep, table_h)
    return h_keep, v_keep


def _suppress_shadow_vlines(
    v_seps: List[Separator],
    table_h: float,
    *,
    strong_cover: float = 0.50,
    shadow_cover_max: float = 0.45,
    near_frac: float = 0.30,
) -> List[Separator]:
    """
    丢弃紧贴通高竖线的影子短线（cover 中等但贴边），避免把 B1/4g、0.12 切碎。
    """
    if len(v_seps) < 3:
        return v_seps
    ordered = sorted(v_seps, key=lambda s: s.coord)
    gaps = np.diff([s.coord for s in ordered])
    med_gap = float(np.median(gaps)) if len(gaps) else 40.0
    near = max(8.0, med_gap * near_frac)

    strong = [s for s in ordered if s.coverage_ratio(table_h) >= strong_cover]
    if not strong:
        return ordered

    kept: List[Separator] = []
    for sep in ordered:
        cover = sep.coverage_ratio(table_h)
        if cover >= strong_cover:
            kept.append(sep)
            continue
        if cover <= shadow_cover_max:
            if any(abs(sep.coord - s.coord) < near for s in strong):
                continue
        kept.append(sep)
    return kept


def _recover_columns_by_ink_gutters(
    binary: np.ndarray,
    h_seps: List[Separator],
    v_seps: List[Separator],
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    gap_ratio: float = 1.8,
    min_depth_ratio: float = 0.35,
) -> List[Separator]:
    """
    在相邻强竖线之间的过大间隔内，用表身墨水投影找“沟”补弱列线。

    只补大间隔，避免在已正确的密列表（如 CN114）上乱加列。
    """
    if binary is None or len(v_seps) < 2 or len(h_seps) < 3:
        return v_seps

    ordered = sorted(v_seps, key=lambda s: s.coord)
    gaps = np.diff([s.coord for s in ordered])
    if len(gaps) == 0:
        return v_seps
    med_gap = float(np.median(gaps))
    if med_gap < 8:
        return v_seps

    # 表身：跳过前两条横线（表头），去掉底边
    ys = sorted(s.coord for s in h_seps)
    body_top = int(max(y1, ys[min(2, len(ys) - 1)]))
    body_bot = int(min(y2, ys[-1]))
    if body_bot - body_top < 20:
        return v_seps

    # 强竖线：高覆盖，用作大间隔端点
    th = max(y2 - y1, 1.0)
    strong = [s for s in ordered if s.coverage_ratio(th) >= 0.45] or ordered
    strong = sorted(strong, key=lambda s: s.coord)

    h_img, w_img = binary.shape[:2]
    body = binary[max(0, body_top) : min(h_img, body_bot), :]
    if body.size == 0:
        return v_seps
    # 前景墨水投影（二值已是白前景）
    ink = (body > 0).sum(axis=0).astype(np.float64)

    new_seps: List[Separator] = []
    min_gap = med_gap * gap_ratio
    min_dist = max(6.0, med_gap * 0.35)
    body_h = float(body_bot - body_top)

    for i in range(len(strong) - 1):
        left, right = strong[i].coord, strong[i + 1].coord
        width = right - left
        if width < min_gap:
            continue
        lo = int(max(0, left + min_dist))
        hi = int(min(w_img - 1, right - min_dist))
        if hi - lo < 8:
            continue
        segment = ink[lo : hi + 1]
        if segment.size < 8:
            continue
        # 平滑
        k = max(3, int(med_gap // 8) | 1)
        kernel = np.ones(k, dtype=np.float64) / k
        smooth = np.convolve(segment, kernel, mode="same")
        peak = float(smooth.max())
        if peak < 3:
            continue
        # 找局部极小
        candidates: List[Tuple[float, float]] = []  # (depth, x)
        for j in range(2, len(smooth) - 2):
            v = smooth[j]
            if v <= smooth[j - 1] and v <= smooth[j + 1] and v <= smooth[j - 2] and v <= smooth[j + 2]:
                depth = (peak - v) / peak
                if depth >= min_depth_ratio:
                    x = float(lo + j)
                    candidates.append((depth, x))
        if not candidates:
            continue
        # 在大间隔内按期望列数取沟：间隔能放下几列
        n_extra = max(1, int(round(width / med_gap)) - 1)
        n_extra = min(n_extra, 6)
        # 贪心：按深度选，且彼此间距够
        candidates.sort(reverse=True)
        chosen: List[float] = []
        for _, x in candidates:
            if any(abs(x - c) < min_dist for c in chosen):
                continue
            if any(abs(x - s.coord) < min_dist for s in ordered):
                continue
            chosen.append(x)
            if len(chosen) >= n_extra:
                break
        for x in chosen:
            new_seps.append(
                Separator(
                    coord=x,
                    spans=[(float(body_top), float(body_bot))],
                    length=body_h,
                )
            )

    if not new_seps:
        return v_seps
    return sorted(list(v_seps) + new_seps, key=lambda s: s.coord)


def _recover_columns_by_ocr_corridors(
    text_boxes: Sequence[Dict[str, Any]],
    h_seps: Sequence[Separator],
    v_seps: Sequence[Separator],
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    corridor_hold_ratio: float = 0.70,
    min_corridor_px: float = 5.5,
    near_tol: float = 6.0,
    max_new: int = 10,
    min_col_text_rows: int = 2,
) -> List[Separator]:
    """
    用 OCR 文本框的“x 覆盖空洞”来补回缺失竖分隔线（用于 lineless 表）。

    思路：
      1) 只看表体区域（跳过前两条横线，和 _recover_columns_by_ink_gutters 保持一致）。
      2) 在 ROI 内对 x 做 1D coverage：若某 x 落在任意 OCR bbox 内 → coverage=1。
      3) 找 coverage=0 的长空洞 run，取 run 中心作为候选竖线。
      4) 候选竖线必须在大多数行带上都保持空洞（corridor_hold_ratio）。
      5) 候选竖线加入后，新列也要在足够多行带出现 OCR 文本（min_col_text_rows）。
    """
    if not text_boxes or len(v_seps) < 2 or len(h_seps) < 3:
        return list(v_seps)

    h_coords_sorted = sorted(float(s.coord) for s in h_seps)
    body_top = float(max(y1, h_coords_sorted[min(2, len(h_coords_sorted) - 1)]))
    body_bot = float(min(y2, h_coords_sorted[-1]))
    if body_bot - body_top < 25:
        return list(v_seps)

    # relevant text：限制在表体区域
    relevant: List[Dict[str, Any]] = []
    for tb in text_boxes:
        poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
        bx1 = float(poly[:, 0].min())
        by1 = float(poly[:, 1].min())
        bx2 = float(poly[:, 0].max())
        by2 = float(poly[:, 1].max())
        # y 方向落入表体区域（用于筛选相关 OCR bbox）
        if bx2 < x1 or bx1 > x2:
            continue
        if by2 < body_top or by1 > body_bot:
            continue
        relevant.append(tb)
    if not relevant:
        return list(v_seps)

    v_sorted = sorted(list(v_seps), key=lambda s: s.coord)
    coords = [float(s.coord) for s in v_sorted]
    if len(coords) < 2:
        return list(v_seps)
    gaps = np.diff(coords)
    med_gap = float(np.median(gaps)) if len(gaps) else (x2 - x1) / 4.0

    width = float(x2 - x1)
    step = max(2, int(round(width / 220.0)))  # 控制 1D 数组长度
    if step <= 0:
        step = 2
    n = max(1, int(round(width / step)) + 1)
    coverage = np.zeros(n, dtype=np.uint8)
    # 将 OCR bbox 在 x 方向投影成 coverage=1
    for tb in relevant:
        poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
        bx1 = float(poly[:, 0].min())
        bx2 = float(poly[:, 0].max())
        lo = int(max(0, np.floor((bx1 - x1) / step)))
        hi = int(min(n - 1, np.ceil((bx2 - x1) / step)))
        coverage[lo : hi + 1] = 1

    # 找覆盖为 0 的 run
    candidates: List[Tuple[float, float]] = []  # (run_len, x_center)
    i = 0
    while i < n:
        if coverage[i] == 0:
            j = i
            while j < n and coverage[j] == 0:
                j += 1
            run_len = (j - i) * step
            if run_len >= min_corridor_px:
                x_center = x1 + (i + j - 1) / 2.0 * step
                candidates.append((float(run_len), float(x_center)))
            i = j
        else:
            i += 1
    if not candidates:
        return list(v_seps)

    # 候选必须落在“大间隔”区域（否则在密表上乱补）
    big_gap_thr = med_gap * 1.2
    big_gaps: List[Tuple[float, float]] = []
    for a, b in zip(coords[:-1], coords[1:]):
        if b - a >= big_gap_thr:
            big_gaps.append((a, b))
    if not big_gaps:
        return list(v_seps)

    def in_big_gap(x: float) -> bool:
        return any(lo - near_tol <= x <= hi + near_tol for lo, hi in big_gaps)

    filtered: List[Tuple[float, float]] = []
    for run_len, x_center in candidates:
        if not in_big_gap(x_center):
            continue
        if any(abs(x_center - c) <= near_tol for c in coords):
            continue
        filtered.append((run_len, x_center))

    # 备用候选：用 OCR bbox 的中心点 x 的“大间隙”直接推断分隔线位置
    # 当空洞 run 候选偏少时更有效（lineless + 文字宽度变化较大时）。
    if len(filtered) < 4:
        centroids: List[float] = []
        for tb in relevant:
            poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
            bx1 = float(poly[:, 0].min())
            bx2 = float(poly[:, 0].max())
            centroids.append((bx1 + bx2) / 2.0)
        centroids.sort()
        for a, b in zip(centroids[:-1], centroids[1:]):
            if b - a < med_gap * 0.55:
                continue
            x_center = (a + b) / 2.0
            if not in_big_gap(x_center):
                continue
            if any(abs(x_center - c) <= near_tol for c in coords):
                continue
            filtered.append((float(b - a), float(x_center)))
    if not filtered:
        return list(v_seps)

    filtered.sort(reverse=True, key=lambda t: t[0])

    # 行带边界：从 h_seps 坐标取与 body_top/body_bot 重叠的区间
    h_sorted = sorted(float(s.coord) for s in h_seps)
    row_bands: List[Tuple[float, float]] = []
    for y_lo, y_hi in zip(h_sorted[:-1], h_sorted[1:]):
        lo = max(body_top, y_lo)
        hi = min(body_bot, y_hi)
        if hi - lo >= 8:
            row_bands.append((lo, hi))
    if not row_bands:
        return list(v_seps)

    # 预先算每行带里哪些 tb 在 y 上有效
    tb_bboxes: List[Tuple[float, float, float, float]] = []
    for tb in relevant:
        poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
        bx1 = float(poly[:, 0].min())
        by1 = float(poly[:, 1].min())
        bx2 = float(poly[:, 0].max())
        by2 = float(poly[:, 1].max())
        tb_bboxes.append((bx1, by1, bx2, by2))

    def corridor_holds_at_x(xc: float) -> float:
        holds = 0
        for lo, hi in row_bands:
            band_has_text_at_x = False
            for (bx1, by1, bx2, by2), tb in zip(tb_bboxes, relevant):
                if by2 < lo or by1 > hi:
                    continue
                if bx1 <= xc <= bx2:
                    band_has_text_at_x = True
                    break
            if not band_has_text_at_x:
                holds += 1
        return holds / max(1, len(row_bands))

    def columns_have_text_at_coords(col_coords: Sequence[float]) -> bool:
        # 计算每列“出现文本的行带数”
        cols = list(col_coords)
        if len(cols) < 2:
            return False
        n_cols_local = len(cols) - 1
        col_good = 0
        for ci in range(n_cols_local):
            c_lo, c_hi = cols[ci], cols[ci + 1]
            rows_with_text = 0
            for lo, hi in row_bands:
                has = False
                for bx1, by1, bx2, by2 in tb_bboxes:
                    if by2 < lo or by1 > hi:
                        continue
                    cx = (bx1 + bx2) / 2.0
                    if c_lo <= cx <= c_hi:
                        has = True
                        break
                if has:
                    rows_with_text += 1
            if rows_with_text >= min_col_text_rows:
                col_good += 1
        return (col_good / max(1, n_cols_local)) >= 0.50

    existing_coords = sorted(set(float(s.coord) for s in v_seps))
    accepted: List[float] = []

    for _run_len, x_center in filtered[: max_new * 4]:
        if len(accepted) >= max_new:
            break
        hold_ratio = corridor_holds_at_x(x_center)
        if hold_ratio < corridor_hold_ratio:
            continue
        proposed = sorted(existing_coords + accepted + [x_center])
        if columns_have_text_at_coords(proposed):
            accepted.append(x_center)

    if not accepted:
        return list(v_seps)

    body_span = float(body_bot - body_top)
    new_seps: List[Separator] = []
    for xc in accepted:
        new_seps.append(
            Separator(
                coord=float(xc),
                spans=[(float(body_top), float(body_bot))],
                length=float(body_span),
            )
        )

    merged = _merge_separator_lists(
        list(v_seps),
        new_seps,
        near_tol=near_tol,
        min_cover_total=body_span,
        min_cover_ratio=0.25,
    )
    return merged


def _find_table_rois(
    hmask: np.ndarray,
    vmask: np.ndarray,
    min_area_ratio: float = _MIN_TABLE_AREA_RATIO,
) -> List[Tuple[int, int, int, int]]:
    combined = cv2.bitwise_or(hmask, vmask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.dilate(combined, kernel, iterations=2)
    hh, ww = combined.shape[:2]
    min_area = max(int(hh * ww * min_area_ratio), 4000)
    n, _, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    rois: List[Tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or w < 50 or h < 50:
            continue
        rois.append((max(0, x - 2), max(0, y - 2), min(ww, x + w + 2), min(hh, y + h + 2)))
    rois.sort(key=lambda r: (r[1], r[0]))
    return rois or [(0, 0, ww, hh)]


def _clip_separators_to_roi(
    seps: Sequence[Separator],
    *,
    axis_min: float,
    axis_max: float,
    span_min: float,
    span_max: float,
    coord_pad: float = 3.0,
) -> List[Separator]:
    out: List[Separator] = []
    for sep in seps:
        if sep.coord < axis_min - coord_pad or sep.coord > axis_max + coord_pad:
            continue
        clipped = []
        for a, b in sep.spans:
            lo, hi = max(a, span_min), min(b, span_max)
            if hi - lo >= 3:
                clipped.append((lo, hi))
        if not clipped:
            continue
        merged = _merge_intervals(clipped)
        length = sum(e - s for s, e in merged)
        out.append(Separator(coord=sep.coord, spans=merged, length=float(length)))
    return out


def _ensure_border_seps(
    h_seps: List[Separator],
    v_seps: List[Separator],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Tuple[List[Separator], List[Separator]]:
    tw, th = x2 - x1, y2 - y1
    # 相对表尺寸的容差，避免首条横线距 ROI 稍远时插入幽灵边框行
    h_tol = max(8.0, 0.012 * th)
    v_tol = max(8.0, 0.012 * tw)
    h_seps = list(h_seps)
    v_seps = list(v_seps)
    if not h_seps or abs(h_seps[0].coord - y1) > h_tol:
        h_seps.insert(0, Separator(coord=y1, spans=[(x1, x2)], length=tw))
    if not h_seps or abs(h_seps[-1].coord - y2) > h_tol:
        h_seps.append(Separator(coord=y2, spans=[(x1, x2)], length=tw))
    if not v_seps or abs(v_seps[0].coord - x1) > v_tol:
        v_seps.insert(0, Separator(coord=x1, spans=[(y1, y2)], length=th))
    if not v_seps or abs(v_seps[-1].coord - x2) > v_tol:
        v_seps.append(Separator(coord=x2, spans=[(y1, y2)], length=th))
    return _dedupe_close(h_seps, 4.0), _dedupe_close(v_seps, 4.0)


def _tight_bbox_from_masks(
    hmask: np.ndarray,
    vmask: np.ndarray,
    pad: int = 2,
) -> Tuple[int, int, int, int]:
    """从线掩膜估计表格紧致外接框。"""
    combined = cv2.bitwise_or(hmask, vmask)
    ys, xs = np.where(combined > 0)
    if len(xs) == 0:
        h, w = combined.shape[:2]
        return 0, 0, w, h
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(combined.shape[1], int(xs.max()) + pad + 1)
    y2 = min(combined.shape[0], int(ys.max()) + pad + 1)
    return x1, y1, x2, y2


def build_cells_from_separators(
    h_seps: List[Separator],
    v_seps: List[Separator],
    merge_cover_thresh: float = _MERGE_COVER_THRESH,
) -> Tuple[List[Dict[str, Any]], float]:
    if len(h_seps) < 2 or len(v_seps) < 2:
        return [], 0.0

    row_ys = [s.coord for s in h_seps]
    col_xs = [s.coord for s in v_seps]
    n_rows = len(row_ys) - 1
    n_cols = len(col_xs) - 1
    if n_rows < 1 or n_cols < 1:
        return [], 0.0

    uf = _UnionFind(n_rows * n_cols)

    for r in range(n_rows):
        y0, y1 = row_ys[r], row_ys[r + 1]
        for c in range(n_cols - 1):
            if not v_seps[c + 1].covers(y0, y1, thresh=merge_cover_thresh):
                uf.union(r * n_cols + c, r * n_cols + c + 1)

    for c in range(n_cols):
        x0, x1 = col_xs[c], col_xs[c + 1]
        for r in range(n_rows - 1):
            if not h_seps[r + 1].covers(x0, x1, thresh=merge_cover_thresh):
                uf.union(r * n_cols + c, (r + 1) * n_cols + c)

    groups: Dict[int, List[Tuple[int, int]]] = {}
    for r in range(n_rows):
        for c in range(n_cols):
            root = uf.find(r * n_cols + c)
            groups.setdefault(root, []).append((r, c))

    cells: List[Dict[str, Any]] = []
    for atoms in groups.values():
        rs = [a[0] for a in atoms]
        cs = [a[1] for a in atoms]
        row_start, row_end = min(rs), max(rs)
        col_start, col_end = min(cs), max(cs)
        # 连通组必须填满外接矩形，否则拆成逐原子格（避免错误大合并）
        expected = (row_end - row_start + 1) * (col_end - col_start + 1)
        if len(atoms) != expected:
            # 非矩形：逐原子输出
            for rr, cc in atoms:
                x1 = float(col_xs[cc])
                x2 = float(col_xs[cc + 1])
                y1 = float(row_ys[rr])
                y2 = float(row_ys[rr + 1])
                cells.append(_make_cell(x1, y1, x2, y2, rr, rr, cc, cc))
            continue

        x1 = float(col_xs[col_start])
        x2 = float(col_xs[col_end + 1])
        y1 = float(row_ys[row_start])
        y2 = float(row_ys[row_end + 1])
        cells.append(
            _make_cell(x1, y1, x2, y2, row_start, row_end, col_start, col_end)
        )

    return cells, _grid_confidence(h_seps, v_seps)


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


def _normalize_full_spans(
    h_seps: List[Separator],
    v_seps: List[Separator],
) -> Tuple[List[Separator], List[Separator]]:
    """
    高覆盖率分隔线视为通栏：把 spans 扩展到对向线的全局范围。
    修复 adaptive 横线断裂导致交点置信度偏低、合并误判。
    """
    if not h_seps or not v_seps:
        return h_seps, v_seps
    x_min = min(s.coord for s in v_seps)
    x_max = max(s.coord for s in v_seps)
    y_min = min(s.coord for s in h_seps)
    y_max = max(s.coord for s in h_seps)
    table_w = max(x_max - x_min, 1.0)
    table_h = max(y_max - y_min, 1.0)

    h_out: List[Separator] = []
    for sep in h_seps:
        if sep.coverage_ratio(table_w) >= 0.50:
            h_out.append(
                Separator(coord=sep.coord, spans=[(x_min, x_max)], length=table_w)
            )
        else:
            h_out.append(sep)

    v_out: List[Separator] = []
    for sep in v_seps:
        if sep.coverage_ratio(table_h) >= 0.50:
            v_out.append(
                Separator(coord=sep.coord, spans=[(y_min, y_max)], length=table_h)
            )
        else:
            v_out.append(sep)
    return h_out, v_out


def _grid_confidence(h_seps: List[Separator], v_seps: List[Separator]) -> float:
    """
    置信度：优先用高覆盖率的通栏线计算交点命中率；
    短分区线不拉低分数（它们本来就不与全部对向线相交）。
    """
    if len(h_seps) < 2 or len(v_seps) < 2:
        return 0.0

    def span_len(sep: Separator) -> float:
        return float(sep.length)

    table_w = max(span_len(s) for s in h_seps) if h_seps else 1.0
    table_h = max(span_len(s) for s in v_seps) if v_seps else 1.0
    h_strong = [s for s in h_seps if s.coverage_ratio(table_w) >= 0.45] or h_seps
    v_strong = [s for s in v_seps if s.coverage_ratio(table_h) >= 0.45] or v_seps

    total = len(h_strong) * len(v_strong)
    hit = 0
    for hs in h_strong:
        for vs in v_strong:
            x_ok = any(s - 3 <= vs.coord <= e + 3 for s, e in hs.spans)
            y_ok = vs.covers(hs.coord - 3, hs.coord + 3, thresh=0.01)
            if x_ok and y_ok:
                hit += 1
    return float(hit) / float(total) if total else 0.0


def _merge_separator_lists(    primary: List[Separator],
    extra: List[Separator],
    *,
    near_tol: float = 6.0,
    min_cover_total: float,
    min_cover_ratio: float,
) -> List[Separator]:
    """把 extra 中离 primary 足够远且覆盖达标的线补进结果。"""
    merged = list(primary)
    for sep in extra:
        if sep.coverage_ratio(min_cover_total) < min_cover_ratio:
            continue
        near_idx = None
        for i, p in enumerate(merged):
            if abs(sep.coord - p.coord) <= near_tol:
                near_idx = i
                break
        if near_idx is not None:
            if sep.length > merged[near_idx].length:
                merged[near_idx] = sep
            continue
        merged.append(sep)
    return sorted(merged, key=lambda s: s.coord)


def detect_tables(
    image: np.ndarray,
    *,
    confidence_thresh: float = _CONFIDENCE_THRESH,
    merge_cover_thresh: float = _MERGE_COVER_THRESH,
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[DetectedTable]:
    """检测图像中所有有框线表格。"""
    if image is None or getattr(image, "size", 0) == 0:
        return []

    h, w = image.shape[:2]
    otsu = binarize_otsu(image)
    adapt = binarize_adaptive(image)

    hmask_o, vmask_o = _line_masks(otsu)
    hmask_a, vmask_a = _line_masks(adapt)

    # ---- 横线：Otsu 为主；太少时用 adaptive 补充（JP 淡扫描）----
    h_otsu = _separators_from_projection(
        hmask_o, horizontal=True, min_proj_ratio=0.30, cluster_tol=3, min_seg_len=max(10, w // 80)
    )
    h_adapt = _separators_from_projection(
        hmask_a, horizontal=True, min_proj_ratio=0.25, cluster_tol=3, min_seg_len=max(10, w // 80)
    )
    adapt_h_fg = float((hmask_a > 0).mean())
    if len(h_otsu) >= 8:
        h_all = h_otsu
        if adapt_h_fg <= 0.12:
            h_all = _merge_separator_lists(
                h_otsu, h_adapt, near_tol=6.0, min_cover_total=float(w), min_cover_ratio=0.40
            )
    else:
        h_all = _merge_separator_lists(
            h_adapt, h_otsu, near_tol=6.0, min_cover_total=float(w), min_cover_ratio=0.20
        )

    # ---- 竖线：Otsu 主核 + 稍短核补分区线；adaptive 仅在掩膜干净时补漏 ----
    vk_main = max(15, min(h // 45, 25))
    vk_short = max(10, min(12, vk_main - 5))
    vmask_o = _vline_mask(otsu, vk_main)
    vmask_o_short = _vline_mask(otsu, vk_short)
    v_otsu = _separators_from_projection(
        vmask_o, horizontal=False, min_proj_ratio=0.18, cluster_tol=3, min_seg_len=max(8, h // 100)
    )
    v_otsu_short = _separators_from_projection(
        vmask_o_short, horizontal=False, min_proj_ratio=0.15, cluster_tol=3, min_seg_len=max(8, h // 100)
    )
    # 短核线：覆盖率达到 35% 才并入，挡住文字竖笔
    v_otsu = _merge_separator_lists(
        v_otsu, v_otsu_short, near_tol=6.0, min_cover_total=float(h), min_cover_ratio=0.35
    )

    vmask_a = _vline_mask(adapt, vk_main)
    adapt_v_fg = float((vmask_a > 0).mean())
    if adapt_v_fg <= 0.12:
        v_adapt = _separators_from_projection(
            vmask_a, horizontal=False, min_proj_ratio=0.18, cluster_tol=3, min_seg_len=max(8, h // 100)
        )
        v_all = _merge_separator_lists(
            v_otsu, v_adapt, near_tol=6.0, min_cover_total=float(h), min_cover_ratio=0.35
        )
    else:
        v_all = v_otsu

    if len(v_all) < 3:
        v_loose = _separators_from_projection(
            vmask_o, horizontal=False, min_proj_ratio=0.10, cluster_tol=3, min_seg_len=max(6, h // 120)
        )
        if len(v_loose) > len(v_all):
            v_all = v_loose

    if len(h_all) < 3:
        h_loose = _separators_from_projection(
            hmask_o, horizontal=True, min_proj_ratio=0.15, cluster_tol=3, min_seg_len=max(8, w // 100)
        )
        if len(h_loose) > len(h_all):
            h_all = h_loose

    rois = _find_table_rois(hmask_o, vmask_o)
    if len(rois) == 1 and rois[0] == (0, 0, w, h):
        tight = _tight_bbox_from_masks(hmask_o, vmask_o)
        if (tight[2] - tight[0]) * (tight[3] - tight[1]) > w * h * 0.05:
            rois = [tight]

    tables: List[DetectedTable] = []
    for x1, y1, x2, y2 in rois:
        tw, th = float(x2 - x1), float(y2 - y1)
        h_roi = _clip_separators_to_roi(
            h_all, axis_min=y1, axis_max=y2, span_min=x1, span_max=x2
        )
        v_roi = _clip_separators_to_roi(
            v_all, axis_min=x1, axis_max=x2, span_min=y1, span_max=y2
        )
        h_roi, v_roi = _filter_separators(h_roi, v_roi, tw, th)
        # 大间隔内用表身墨水沟补弱竖线（JP 的 A 区等）
        v_roi = _recover_columns_by_ink_gutters(
            otsu, h_roi, v_roi, x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)
        )
        # 对 lineless 表：用 OCR 空白走廊补回缺失竖线
        if text_boxes:
            v_roi = _recover_columns_by_ocr_corridors(
                text_boxes,
                h_roi,
                v_roi,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            )
        v_roi = _suppress_shadow_vlines(v_roi, th)
        h_roi, v_roi = _ensure_border_seps(h_roi, v_roi, float(x1), float(y1), float(x2), float(y2))

        # 合并判定必须用原始 spans（局部缺失的线才能检出 rowspan/colspan）；
        # 通栏扩展仅用于置信度，避免把合并单元格拆碎。
        cells, _ = build_cells_from_separators(
            h_roi, v_roi, merge_cover_thresh=merge_cover_thresh
        )
        h_norm, v_norm = _normalize_full_spans(h_roi, v_roi)
        conf = _grid_confidence(h_norm, v_norm)
        n_rows, n_cols = max(0, len(h_roi) - 1), max(0, len(v_roi) - 1)
        if not cells or n_cols < 2 or n_rows < 1:
            continue

        tables.append(
            DetectedTable(
                cells=cells,
                bbox=(x1, y1, x2, y2),
                row_seps=[s.coord for s in h_roi],
                col_seps=[s.coord for s in v_roi],
                confidence=conf,
                h_separators=h_roi,
                v_separators=v_roi,
            )
        )
        logger.info(
            "检测到表格 ROI=%s rows=%d cols=%d cells=%d conf=%.3f",
            (x1, y1, x2, y2),
            n_rows,
            n_cols,
            len(cells),
            conf,
        )

    good = [t for t in tables if t.confidence >= confidence_thresh]
    if good:
        return good
    if tables:
        tables.sort(key=lambda t: t.confidence, reverse=True)
        return tables[:1]
    return []


def render_debug_overlay(
    image: np.ndarray,
    tables: Sequence[DetectedTable],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> np.ndarray:
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    colors = [(0, 180, 0), (180, 0, 0), (0, 0, 180), (180, 120, 0), (120, 0, 180)]

    for ti, table in enumerate(tables):
        color = colors[ti % len(colors)]
        x1, y1, x2, y2 = table.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        for ys in table.row_seps:
            y = int(round(ys))
            cv2.line(canvas, (x1, y), (x2, y), (0, 200, 255), 1)
        for xs in table.col_seps:
            x = int(round(xs))
            cv2.line(canvas, (x, y1), (x, y2), (255, 150, 0), 1)

        for cell in table.cells:
            poly = np.asarray(cell["polygon"], dtype=np.int32)
            cv2.polylines(canvas, [poly], True, color, 1)
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            label = f"{cell['row_start']},{cell['col_start']}"
            if cell.get("row_span", 1) > 1 or cell.get("col_span", 1) > 1:
                label += f"({cell.get('row_span')}x{cell.get('col_span')})"
            cv2.putText(
                canvas,
                label,
                (cx - 10, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            f"T{ti} conf={table.confidence:.2f} r={len(table.row_seps)-1} c={len(table.col_seps)-1}",
            (x1 + 4, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    if text_boxes:
        for tb in text_boxes:
            poly = np.asarray(tb.get("polygon"), dtype=np.int32).reshape(-1, 2)
            if poly.shape[0] >= 2:
                cv2.polylines(canvas, [poly], True, (200, 0, 200), 1)

    return canvas


def imwrite_unicode(path: str, image: np.ndarray) -> bool:
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(path)
    return True
