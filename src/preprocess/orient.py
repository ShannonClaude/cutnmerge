"""表格图像方向归正：投影剖面定轴向 + OCR 置信度消歧 180°。

必须在 OCR 之前完成轴向旋转；180° 消歧可在首次 OCR 之后按需再跑一次。
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ROTATE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """按顺时针角度旋转（0/90/180/270）。"""
    angle = int(angle) % 360
    if angle == 0:
        return image
    code = _ROTATE.get(angle)
    if code is None:
        raise ValueError(f"不支持的旋转角: {angle}")
    return cv2.rotate(image, code)


def _projection_periodicity(profile: np.ndarray) -> float:
    """投影剖面周期性：相邻差分能量 + 零值间隙数。"""
    p = np.asarray(profile, dtype=np.float64)
    if p.size < 8:
        return 0.0
    k = max(3, min(9, p.size // 40 * 2 + 1))
    kernel = np.ones(k, dtype=np.float64) / k
    sm = np.convolve(p, kernel, mode="same")
    mx = float(sm.max())
    if mx <= 1e-6:
        return 0.0
    sm = sm / mx
    diff = np.diff(sm)
    energy = float(np.sum(diff * diff))
    thresh = 0.08
    low = sm < thresh
    gaps = 0
    in_gap = False
    min_gap = max(2, p.size // 80)
    run = 0
    for v in low:
        if v:
            run += 1
            if not in_gap and run >= min_gap:
                in_gap = True
                gaps += 1
        else:
            run = 0
            in_gap = False
    return energy * (1.0 + 0.15 * gaps)


def estimate_axis_rotation(image: np.ndarray) -> int:
    """
    弱几何先验：仅在「去线后字块明显偏高」时建议 90，否则 0。

    真正的轴向以 OCR 框宽高比 / 置信度为准（见 ensure_upright_axis）。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 30, 15), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 30, 15)))
    text_mask = cv2.subtract(
        binary,
        cv2.bitwise_or(
            cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk),
            cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk),
        ),
    )
    row_score = _projection_periodicity(text_mask.sum(axis=1).astype(np.float64))
    col_score = _projection_periodicity(text_mask.sum(axis=0).astype(np.float64))
    h_line = float(cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk).sum()) / 255.0 / max(w, 1)
    v_line = float(cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk).sum()) / 255.0 / max(h, 1)
    logger.info(
        "方向轴向先验: textR=%.3f textC=%.3f hLine=%.1f vLine=%.1f",
        row_score,
        col_score,
        h_line,
        v_line,
    )
    # 侧躺：文字呈竖向条带（row 周期强）且竖线墨迹更重
    if row_score > col_score * 1.35 and v_line > h_line * 1.2:
        return 90
    return 0


def _orientation_quality(text_boxes: list) -> float:
    """综合分：置信度 + 横长框占比。"""
    if not text_boxes:
        return 0.0
    return _mean_ocr_score(text_boxes) + 0.25 * _box_aspect_upright_ratio(text_boxes)


def looks_sideways(text_boxes: list, *, score_thresh: float = 0.55) -> bool:
    """OCR 结果是否像侧躺（竖长框多或置信度差）。"""
    if len(text_boxes) < 5:
        return True
    aspect = _box_aspect_upright_ratio(text_boxes)
    score = _mean_ocr_score(text_boxes)
    
    # 【核心修复】放宽侧躺检测的触发阈值。
    # 原本的 0.40 太严苛，导致全是短字符（数字/单字母）的侧躺表被漏判。
    # 提高到 0.75 后，短字符侧躺表会进入 try90 测试，由 q90 > q0 做最终正确裁决。
    if aspect < 0.75:
        return True
    if aspect < 0.85 and score < score_thresh:
        return True
    return False

def ensure_upright_axis(
    image: np.ndarray,
    text_boxes: list,
    *,
    ocr_engine=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    score_thresh: float = 0.55,
    cache_extra_base: str = "deskew=1|orient=0",
) -> Tuple[np.ndarray, int, list]:
    """
    若当前 OCR 结果像侧躺，则试转 90° 并取质更好者。
    """
    from ..core.models import predict_texts

    # 恢复为纯 OCR 框特征门控，彻底抛弃会误杀无横线表格的 estimate_axis_rotation
    if not looks_sideways(text_boxes, score_thresh=score_thresh):
        return image, 0, text_boxes

    rotated = rotate_image(image, 90)
    boxes90 = predict_texts(
        rotated,
        ocr_engine=ocr_engine,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        cache_extra=f"{cache_extra_base}|try90",
    )
    q0 = _orientation_quality(text_boxes)
    q90 = _orientation_quality(boxes90)
    logger.info("方向轴向 OCR 比对: q0=%.3f q90=%.3f", q0, q90)
    
    if q90 > q0:
        return rotated, 90, boxes90
    return image, 0, text_boxes


def _mean_ocr_score(text_boxes: list) -> float:
    if not text_boxes:
        return 0.0
    scores = [float(tb.get("score") or 0.0) for tb in text_boxes]
    return float(np.mean(scores)) if scores else 0.0


def _box_aspect_upright_ratio(text_boxes: list) -> float:
    if not text_boxes:
        return 0.0
    upright = 0
    for tb in text_boxes:
        poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
        if poly.size < 4:
            continue
        w = float(poly[:, 0].max() - poly[:, 0].min())
        h = float(poly[:, 1].max() - poly[:, 1].min())
        if w >= h * 0.9:
            upright += 1
    return upright / max(len(text_boxes), 1)


def parse_orientation_mode(mode: Union[str, int]) -> Union[str, int]:
    """归一化 CLI/API 的 orientation 参数。"""
    mode_s = str(mode).strip().lower()
    if mode_s in {"auto", "none", "off", "disable"}:
        return "none" if mode_s in {"none", "off", "disable"} else "auto"
    forced = int(mode_s) % 360
    if forced not in {0, 90, 180, 270}:
        raise ValueError(f"orientation 仅支持 auto/none/0/90/180/270，收到 {mode!r}")
    return forced


def apply_orientation_axis(
    image: np.ndarray,
    *,
    mode: Union[str, int] = "auto",
) -> Tuple[np.ndarray, int, str]:
    """
    强制角度立即旋转；auto 不做几何预转（交给 OCR 轴向校验）。

    Returns:
        (image, axis_angle, mode_kind)  mode_kind in {auto, none, forced}
    """
    if image is None or getattr(image, "size", 0) == 0:
        return image, 0, "none"

    parsed = parse_orientation_mode(mode)
    if parsed == "none":
        return image, 0, "none"
    if parsed != "auto":
        out = rotate_image(image, int(parsed))
        logger.info("方向强制旋转 %d°", int(parsed))
        return out, int(parsed), "forced"

    # auto：保留原图，由 ensure_upright_axis 在 OCR 后决定是否转 90
    return image, 0, "auto"


def maybe_flip_180_by_ocr(
    image: np.ndarray,
    text_boxes: list,
    *,
    ocr_engine=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    score_thresh: float = 0.55,
    cache_extra_base: str = "deskew=1|orient",
) -> Tuple[np.ndarray, int, list]:
    """
    根据已有 OCR 结果决定是否再转 180°；需要时重跑 OCR。

    Returns:
        (image, added_180, text_boxes)
    """
    from ..core.models import predict_texts

    s0 = _mean_ocr_score(text_boxes)
    a0 = _box_aspect_upright_ratio(text_boxes)
    if s0 >= score_thresh and a0 >= 0.55 and len(text_boxes) >= 8:
        logger.info("方向 180 消歧: keep score=%.3f aspect=%.3f", s0, a0)
        return image, 0, text_boxes

    flipped = rotate_image(image, 180)
    # 翻转后角度 = 原缓存角 + 180；这里用独立 key
    boxes1 = predict_texts(
        flipped,
        ocr_engine=ocr_engine,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        cache_extra=f"{cache_extra_base}_flip180",
    )
    s1 = _mean_ocr_score(boxes1)
    a1 = _box_aspect_upright_ratio(boxes1)
    score0 = s0 + 0.15 * a0
    score1 = s1 + 0.15 * a1
    logger.info(
        "方向 180 消歧: s0=%.3f a0=%.3f vs s1=%.3f a1=%.3f", s0, a0, s1, a1
    )
    if score1 > score0:
        return flipped, 180, boxes1
    return image, 0, text_boxes


def correct_orientation(
    image: np.ndarray,
    *,
    mode: Union[str, int] = "auto",
    ocr_engine=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    score_thresh: float = 0.55,
) -> Tuple[np.ndarray, int, Optional[list]]:
    """
    兼容入口：OCR 轴向校验 + 180 消歧。
    """
    image, angle, kind = apply_orientation_axis(image, mode=mode)
    if kind != "auto":
        return image, angle, None

    from ..core.models import predict_texts

    boxes = predict_texts(
        image,
        ocr_engine=ocr_engine,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        cache_extra=f"orient_ax{angle}=0",
    )
    image, axis_delta, boxes = ensure_upright_axis(
        image,
        boxes,
        ocr_engine=ocr_engine,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        score_thresh=score_thresh,
        cache_extra_base=f"orient_ax{angle}",
    )
    angle = (angle + axis_delta) % 360
    image, flip, boxes = maybe_flip_180_by_ocr(
        image,
        boxes,
        ocr_engine=ocr_engine,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        score_thresh=score_thresh,
        cache_extra_base=f"orient_ax{angle}",
    )
    total = (angle + flip) % 360
    return image, total, boxes
