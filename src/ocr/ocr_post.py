"""OCR 文本后处理：符号规范化与基于墨迹证据的幻觉清理。

不做化学名幻觉修复；只做规则化，避免破坏正确识别。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..utils.label_patterns import fix_iii_ocr, split_value_grade

logger = logging.getLogger(__name__)

# 各类破折号 / 否定符 → ASCII '-'（几何归一会再覆盖单字符横线误识）
_DASH_MAP = str.maketrans(
    {
        "\u2014": "-",  # —
        "\u2013": "-",  # –
        "\u2212": "-",  # −
        "\u00ac": "-",  # ¬
        "\u30fc": "-",  # ー
        "\u2015": "-",  # ―
        "\u2500": "-",  # ─
        "\uff0d": "-",  # －
    }
)

_CM_TILDE_RE = re.compile(r"cm\s*[~～]\s*1", re.IGNORECASE)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# 单字符且可能是横线的候选（含「一」等 OCR 常见误识）。
# 不含 ASCII「1」：正常组合物编号/序号由 _maybe_tiny_one_to_dash 按极小扁框处理，
# 避免有二值图时仅因横带墨迹把高置信「1」改成「-」。
_DASH_CANDIDATE_RE = re.compile(
    r"^[\-\u4e00\u4e28\u2014\u2013\u2212\u30fc\u2015\u2500\uff0d~～_—–−ー―─－|丨／/]$"
)


def normalize_ocr_text(text: str) -> str:
    """单段文本规范化（不含几何判据）。"""
    if not text:
        return text
    t = _CTRL_RE.sub("", text)
    t = t.translate(_DASH_MAP)
    t = _CM_TILDE_RE.sub("cm-1", t)
    # 整段仅为破折号变体时统一为 '-'
    if re.fullmatch(r"[\-\s]+", t):
        return "-"
    t = fix_iii_ocr(t)
    # 叠放等级常见误序：A+/0 → "+0"
    if t in {"+0", "0+"}:
        return "0\nA+"
    grade = split_value_grade(t)
    if grade is not None:
        return f"{grade[0]}\n{grade[1]}"
    # jER 系列常见丢首字母 j（如 ER828 / 化合物ER828）
    if re.search(r"(?<![jJ])ER828", t):
        t = re.sub(r"(?<![jJ])ER828", "jER828", t)
    # 「无法评价40」留给 IoA 按列切开，此处不再整段丢掉尾数
    return t


def _tb_wh(tb: Dict[str, Any]) -> tuple[float, float]:
    poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
    if poly.size < 4:
        return 0.0, 0.0
    w = float(poly[:, 0].max() - poly[:, 0].min())
    h = float(poly[:, 1].max() - poly[:, 1].min())
    return w, h


def _tb_xyxy_nms(tb: Dict[str, Any]) -> tuple[float, float, float, float]:
    poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
    if poly.size < 4:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 1e-9:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = max(area_a + area_b - inter, 1e-9)
    return float(inter / denom)


def _contain_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 1e-9:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = max(min(area_a, area_b), 1e-9)
    return float(inter / denom)


def _nms_text_boxes(
    text_boxes: List[Dict[str, Any]],
    *,
    # 更保守：避免误删“同一单元格内的不同文字片段”
    iou_thresh: float = 0.78,
    contain_ratio_thresh: float = 0.97,
) -> List[Dict[str, Any]]:
    """
    对高重叠 OCR 框做去重（贪心 NMS）。

    - 重叠判据：IoU 或包含率
    - 保留策略：score 更高优先；score 相近则保留面积更大的（通常包含更多墨迹）
    """
    if not text_boxes:
        return []

    # 预计算 bbox/score/area
    scored: List[Tuple[float, float, int, tuple[float, float, float, float]]] = []
    for i, tb in enumerate(text_boxes):
        score = float(tb.get("score") if tb.get("score") is not None else 1.0)
        x1, y1, x2, y2 = _tb_xyxy_nms(tb)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        scored.append((score, area, i, (x1, y1, x2, y2)))

    # 按 score desc, area desc
    order = sorted(range(len(scored)), key=lambda k: (scored[k][0], scored[k][1]), reverse=True)
    keep: List[int] = []
    suppressed: set[int] = set()

    for oi in order:
        if oi in suppressed:
            continue
        _score, _area, idx_i, box_i = scored[oi]
        keep.append(idx_i)
        for oj in order:
            if oj == oi or oj in suppressed:
                continue
            _score_j, _area_j, idx_j, box_j = scored[oj]
            iou = _iou_xyxy(box_i, box_j)
            contain = _contain_ratio(box_i, box_j)
            if iou >= iou_thresh or contain >= contain_ratio_thresh:
                suppressed.add(oj)

    kept_set = set(keep)
    # 保持原始相对顺序，便于 debug 对照
    return [tb for i, tb in enumerate(text_boxes) if i in kept_set]


def _tb_xyxy(tb: Dict[str, Any], shape: Sequence[int]) -> tuple[int, int, int, int]:
    poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
    h, w = int(shape[0]), int(shape[1])
    x1 = int(max(0, np.floor(poly[:, 0].min())))
    y1 = int(max(0, np.floor(poly[:, 1].min())))
    x2 = int(min(w, np.ceil(poly[:, 0].max())))
    y2 = int(min(h, np.ceil(poly[:, 1].max())))
    return x1, y1, x2, y2


def _tb_roi(binary: np.ndarray, tb: Dict[str, Any]) -> np.ndarray:
    x1, y1, x2, y2 = _tb_xyxy(tb, binary.shape)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0), dtype=np.uint8)
    return binary[y1:y2, x1:x2]


def _has_horizontal_ink_band(
    binary: np.ndarray,
    tb: Dict[str, Any],
) -> bool:
    """框内墨迹是否像居中横带（破折号特征）。"""
    roi = _tb_roi(binary, tb)
    if roi.size == 0:
        return False
    fg = (roi > 0).astype(np.uint8)
    if fg.size == 0 or int(fg.sum()) == 0:
        return False
    rows = fg.sum(axis=1).astype(np.float64)
    cols = fg.sum(axis=0).astype(np.float64)
    row_thresh = max(1.0, 0.18 * fg.shape[1])
    hot_rows = np.flatnonzero(rows >= row_thresh)
    if hot_rows.size == 0:
        return False
    band_top = int(hot_rows[0])
    band_bottom = int(hot_rows[-1])
    band_h = band_bottom - band_top + 1
    band_center = (band_top + band_bottom) / 2.0
    roi_center = (fg.shape[0] - 1) / 2.0
    col_thresh = max(1.0, 0.10 * fg.shape[0])
    hot_cols = np.flatnonzero(cols >= col_thresh)
    if hot_cols.size == 0:
        return False
    span_w = int(hot_cols[-1] - hot_cols[0] + 1)
    return (
        band_h <= max(1, int(round(fg.shape[0] * 0.35)))
        and abs(band_center - roi_center) <= max(1.5, 0.22 * fg.shape[0])
        and span_w >= max(6, int(round(fg.shape[1] * 0.45)))
    )


def _has_vertical_ink_band(
    binary: np.ndarray,
    tb: Dict[str, Any],
) -> bool:
    """框内墨迹是否像居中竖带（数字 1 / 竖笔特征）。"""
    roi = _tb_roi(binary, tb)
    if roi.size == 0:
        return False
    fg = (roi > 0).astype(np.uint8)
    if fg.size == 0 or int(fg.sum()) == 0:
        return False
    rows = fg.sum(axis=1).astype(np.float64)
    cols = fg.sum(axis=0).astype(np.float64)
    col_thresh = max(1.0, 0.18 * fg.shape[0])
    hot_cols = np.flatnonzero(cols >= col_thresh)
    if hot_cols.size == 0:
        return False
    band_left = int(hot_cols[0])
    band_right = int(hot_cols[-1])
    band_w = band_right - band_left + 1
    band_center = (band_left + band_right) / 2.0
    roi_center = (fg.shape[1] - 1) / 2.0
    row_thresh = max(1.0, 0.10 * fg.shape[1])
    hot_rows = np.flatnonzero(rows >= row_thresh)
    if hot_rows.size == 0:
        return False
    span_h = int(hot_rows[-1] - hot_rows[0] + 1)
    return (
        band_w <= max(1, int(round(fg.shape[1] * 0.35)))
        and abs(band_center - roi_center) <= max(1.5, 0.22 * fg.shape[1])
        and span_h >= max(6, int(round(fg.shape[0] * 0.45)))
    )


def roi_looks_like_short_dash(roi: np.ndarray) -> bool:
    """
    单元格内 ROI 是否像短缺测横线（非通栏框线）。

    要求：居中细横带，且水平跨度明显短于单元格宽，避免把格线当 '-'。
    """
    if roi is None or roi.size == 0:
        return False
    h, w = int(roi.shape[0]), int(roi.shape[1])
    if h < 6 or w < 8:
        return False
    # 内缩避开四边框线
    iy = max(2, h // 6)
    ix = max(2, w // 6)
    if h - 2 * iy < 4 or w - 2 * ix < 6:
        return False
    inner = roi[iy : h - iy, ix : w - ix]
    fg = (inner > 0).astype(np.uint8)
    if int(fg.sum()) < 4:
        return False
    rows = fg.sum(axis=1).astype(np.float64)
    cols = fg.sum(axis=0).astype(np.float64)
    row_thresh = max(1.0, 0.12 * fg.shape[1])
    hot_rows = np.flatnonzero(rows >= row_thresh)
    if hot_rows.size == 0:
        return False
    band_h = int(hot_rows[-1] - hot_rows[0] + 1)
    band_center = (float(hot_rows[0]) + float(hot_rows[-1])) / 2.0
    roi_center = (fg.shape[0] - 1) / 2.0
    # 空单元格 ROI 往往远高于笔画厚度，列阈值按 band 高度而非整格高
    col_thresh = max(1.0, 0.35 * max(band_h, 1))
    hot_cols = np.flatnonzero(cols >= col_thresh)
    if hot_cols.size == 0:
        return False
    span_w = int(hot_cols[-1] - hot_cols[0] + 1)
    # 短横线：相对内框宽度不超过约 60%；通栏格线会被拒
    max_span = max(8, int(round(fg.shape[1] * 0.60)))
    min_span = max(5, int(round(fg.shape[1] * 0.08)))
    return (
        band_h <= max(2, int(round(fg.shape[0] * 0.40)))
        and abs(band_center - roi_center) <= max(2.0, 0.28 * fg.shape[0])
        and min_span <= span_w <= max_span
    )


def _maybe_tiny_one_to_dash(
    tb: Dict[str, Any],
    text: str,
    *,
    median_box_h: float,
    binary: Optional[np.ndarray] = None,
) -> str:
    """
    「—」被 OCR 成 1/|/l 时改回 '-'。

    优先信墨迹：框内居中横带 → '-'（覆盖高置信误识，如 P93 酸当量）。
    其次：极小扁框 / 低置信扁框兜底。
    """
    t = (text or "").strip()
    if t not in {"1", "l", "I", "|", "丨"}:
        return text
    if median_box_h <= 0:
        return text
    w, h = _tb_wh(tb)
    if h <= 0:
        return text
    score = float(tb.get("score") if tb.get("score") is not None else 1.0)
    aspect = w / h
    # 墨迹优先：横带明确时，短框或略扁框都收成 '-'（避免被高 score 挡住）
    if binary is not None and _has_horizontal_ink_band(binary, tb):
        if h <= 0.70 * median_box_h or aspect >= 1.05:
            return "-"
    # 原有路径：极小框（阈值略放宽到 0.50×中位高）
    if h <= 0.50 * median_box_h and aspect >= 1.10:
        if max(w, h) >= 10.0 and binary is not None:
            if not _has_horizontal_ink_band(binary, tb):
                return text
        return "-"
    # 补充路径：低置信且宽扁，按墨迹判定
    if score < 0.70 and aspect >= 1.10 and binary is not None:
        if _has_horizontal_ink_band(binary, tb):
            return "-"
    return text


def _maybe_dash_to_one(
    tb: Dict[str, Any],
    text: str,
    *,
    median_box_h: float,
    binary: Optional[np.ndarray] = None,
) -> str:
    """
    缺测横线 '-' / 「一」被 OCR 成破折号、但墨迹实为竖笔时改回 '1'。

    仅在「有竖带且无横带」且偏瘦高时触发，避免误伤真破折号。
    """
    t = (text or "").strip()
    if t not in {"-", "—", "–", "−", "ー", "―", "─", "－", "一", "~", "～"}:
        return text
    if binary is None or median_box_h <= 0:
        return text
    w, h = _tb_wh(tb)
    if h <= 0 or w <= 0:
        return text
    aspect = w / h
    # 真破折号通常扁；瘦高才可能是竖笔 1
    if aspect > 0.95:
        return text
    # 超高框多为竖排多字碎片，不在此翻成单位数字
    if h > 1.80 * median_box_h:
        return text
    if _has_horizontal_ink_band(binary, tb):
        return text
    if _has_vertical_ink_band(binary, tb):
        return "1"
    return text


def _strip_leading_pipe_digits(text: str) -> str:
    """去掉表格竖线噪点：'|93'→'93'；'比较例 8|93'→'比较例 8 93'；'37|Bk-2'→'37 Bk-2'。"""
    t = (text or "").strip()
    if not t:
        return t
    m = re.fullmatch(r"\|+(.+)", t)
    if m:
        return m.group(1)
    t2 = re.sub(r"\|+", " ", t)
    return re.sub(r"\s+", " ", t2).strip()


def _maybe_geometric_dash(
    tb: Dict[str, Any],
    text: str,
    *,
    binary: Optional[np.ndarray] = None,
) -> str:
    """
    单字符横笔画 → '-'。

    覆盖「一」等 OCR 常见误识；绝不把 ASCII 字母（如等级 A/B）改成破折号。
    """
    t = (text or "").strip()
    if len(t) != 1:
        return text
    if t in {"-", "—", "–", "−", "ー", "―", "─", "－", "~", "～", "¬"}:
        return "-"
    if t.isascii() and t.isalpha():
        return text

    is_cjk_dash_like = t == "一"
    is_candidate = bool(_DASH_CANDIDATE_RE.match(t)) or is_cjk_dash_like
    if not is_candidate:
        return text

    w, h = _tb_wh(tb)
    if h <= 0:
        return text
    aspect = w / h

    if binary is None:
        if t == "一" and aspect >= 1.25:
            return "-"
        if aspect >= 2.0 and w >= 6:
            return "-"
        if _DASH_CANDIDATE_RE.match(t) and aspect >= 1.6:
            return "-"
        return text

    # 有二值图时仍要求扁框：仅横带墨迹不足以把竖笔「丨/|」等收成 dash
    if t == "一":
        min_aspect = 1.25
    else:
        min_aspect = 1.6
    if aspect < min_aspect:
        return text
    if _has_horizontal_ink_band(binary, tb):
        return "-"
    return text


def _ink_density(
    binary: np.ndarray,
    tb: Dict[str, Any],
) -> float:
    """文本框内前景像素占比（binary 为白底黑字的 INV 结果：前景=255）。"""
    roi = _tb_roi(binary, tb)
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / float(roi.size)


def _ink_metrics(binary: np.ndarray, tb: Dict[str, Any]) -> tuple[float, int, int, int]:
    roi = _tb_roi(binary, tb)
    if roi.size == 0:
        return 0.0, 0, 0, 0
    fg = (roi > 0).astype(np.uint8)
    area = int(fg.size)
    fg_pixels = int(fg.sum())
    if fg_pixels <= 0:
        return 0.0, 0, 0, area
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    largest = 0
    for i in range(1, n_labels):
        largest = max(largest, int(stats[i, cv2.CC_STAT_AREA]))
    return float(fg_pixels) / float(area), fg_pixels, largest, area


def _count_text_line_bands(binary: np.ndarray, tb: Dict[str, Any]) -> int:
    roi = _tb_roi(binary, tb)
    if roi.size == 0:
        return 0
    fg = (roi > 0).astype(np.uint8)
    if fg.size == 0 or int(fg.sum()) == 0:
        return 0
    rows = fg.sum(axis=1).astype(np.float64)
    thresh = max(1.0, 0.10 * fg.shape[1])
    hot = rows >= thresh
    count = 0
    i = 0
    while i < len(hot):
        if hot[i]:
            j = i + 1
            while j < len(hot) and hot[j]:
                j += 1
            if (j - i) >= 2:
                count += 1
            i = j
        else:
            i += 1
    return count


def _looks_like_garbage_ink(
    text: str,
    score: float,
    ink: Optional[float],
    median_ink: Optional[float],
    *,
    fg_pixels: Optional[int] = None,
    largest_cc: Optional[int] = None,
    area: Optional[int] = None,
) -> bool:
    """
    低置信 + 墨迹显著低于全图中位 → 空白区幻觉框。
    """
    t = (text or "").strip()
    if not t:
        return False
    if t in {"×", "x", "X"}:
        return False
    if _DASH_CANDIDATE_RE.match(t):
        return False
    if score >= 0.85:
        return False
    is_single_cjk = len(t) == 1 and bool(
        re.match(r"[\u3400-\u9fff\uf900-\ufaff]", t)
    )
    if ink is None or median_ink is None:
        # 无二值图时：极低置信短串，或单字 CJK 且置信一般
        if score < 0.40 and len(t) <= 4 and not re.search(r"[0-9A-Za-z]", t):
            return True
        return bool(is_single_cjk and score < 0.70)
    if fg_pixels is not None and largest_cc is not None and area is not None and area > 0:
        min_fg = max(6, int(round(area * 0.006)))
        min_cc = max(4, int(round(area * 0.004)))
        if score < 0.60 and len(t) <= 8 and fg_pixels < min_fg and largest_cc < min_cc:
            return True
        if score < 0.48 and fg_pixels < max(10, int(round(area * 0.010))):
            return True

    if median_ink is not None and median_ink > 1e-6 and ink is not None:
        if ink < 0.012 and score < 0.50:
            return True
    elif ink is not None and ink < 0.008 and score < 0.50:
        return True
    # 单字 CJK：空角格常见幻觉，墨迹低于中位且置信一般时清空
    if (
        is_single_cjk
        and score < 0.75
        and ink is not None
        and median_ink is not None
        and ink < 0.45 * median_ink
    ):
        return True
    return False


def postprocess_text_boxes(
    text_boxes: List[Dict[str, Any]],
    *,
    binary: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    规范化 OCR 框文本；低置信且墨迹稀薄时清空（保留几何供 debug）。

    Args:
        binary: 可选二值图（前景=255）。提供时启用墨迹密度幻觉检测与
                更稳的几何破折号归一。
    """
    # ---- 前置：NMS 去重，减少“重影框”污染后续切分/回退 ----
    text_boxes = _nms_text_boxes(text_boxes)

    cleaned = 0
    inks: List[float] = []
    median_box_h = 0.0
    box_heights = []
    for tb in text_boxes:
        _w, h = _tb_wh(tb)
        if h > 0:
            box_heights.append(h)
    if box_heights:
        median_box_h = float(np.median(box_heights))
    if binary is not None:
        for tb in text_boxes:
            inks.append(_ink_density(binary, tb))
        median_ink = float(np.median(inks)) if inks else 0.0
    else:
        median_ink = None

    for i, tb in enumerate(text_boxes):
        raw = str(tb.get("text") or "")
        score = float(tb.get("score") if tb.get("score") is not None else 1.0)
        norm = normalize_ocr_text(raw)
        norm = _maybe_geometric_dash(tb, norm, binary=binary)
        norm = _maybe_tiny_one_to_dash(
            tb, norm, median_box_h=median_box_h, binary=binary
        )
        norm = _maybe_dash_to_one(
            tb, norm, median_box_h=median_box_h, binary=binary
        )
        norm = _strip_leading_pipe_digits(norm)
        ink = inks[i] if binary is not None else None
        fg_pixels = None
        largest_cc = None
        area = None
        if binary is not None:
            _ink_ratio, fg_pixels, largest_cc, area = _ink_metrics(binary, tb)
            if median_box_h > 0:
                _w, box_h = _tb_wh(tb)
                line_bands = _count_text_line_bands(binary, tb)
                if box_h >= 1.8 * median_box_h and line_bands >= 2:
                    tb["needs_reocr"] = True
                # 竖排候选：细长框
                if box_h >= 2.2 * max(_w, 1.0) and box_h >= 1.5 * median_box_h:
                    tb["needs_reocr"] = True
                    tb["maybe_vertical"] = True
        if _looks_like_garbage_ink(
            norm,
            score,
            ink,
            median_ink,
            fg_pixels=fg_pixels,
            largest_cc=largest_cc,
            area=area,
        ):
            logger.debug(
                "OCR 幻觉清空: %r score=%.3f ink=%s",
                norm,
                score,
                f"{ink:.3f}" if ink is not None else "-",
            )
            tb["text"] = ""
            tb["ocr_garbage"] = True
            cleaned += 1
        else:
            tb["text"] = norm
    if cleaned:
        logger.info("OCR 后处理清空疑似幻觉框 %d 个", cleaned)
    return [tb for tb in text_boxes if str(tb.get("text") or "").strip()]
