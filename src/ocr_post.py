"""OCR 文本后处理：符号规范化与基于墨迹密度的幻觉清理。

不做化学名幻觉修复；只做规则化，避免破坏正确识别。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

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

# 单字符且可能是横线的候选（含「一」等 OCR 常见误识）
_DASH_CANDIDATE_RE = re.compile(
    r"^[\-\u4e00\u4e28\u2014\u2013\u2212\u30fc\u2015\u2500\uff0d~～_—–−ー―─－|丨lI1／/]$"
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
    return t


def _tb_wh(tb: Dict[str, Any]) -> tuple[float, float]:
    poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
    if poly.size < 4:
        return 0.0, 0.0
    w = float(poly[:, 0].max() - poly[:, 0].min())
    h = float(poly[:, 1].max() - poly[:, 1].min())
    return w, h


def _maybe_geometric_dash(tb: Dict[str, Any], text: str) -> str:
    """
    单字符横笔画 → '-'。

    覆盖「一」等 OCR 常见误识；对明确破折号候选放宽宽高比。
    """
    t = (text or "").strip()
    if len(t) != 1:
        return text
    w, h = _tb_wh(tb)
    if h <= 0:
        return text
    aspect = w / h
    # 已是破折号类
    if t in {"-", "—", "–", "−", "ー", "―", "─", "－", "~", "～", "¬"}:
        return "-"
    # 「一」：表格空单元格最常见误识，略放宽
    if t == "一" and aspect >= 1.35:
        return "-"
    # 其它单字符：显著横长才归一
    if aspect >= 2.2 and w >= 6:
        return "-"
    if _DASH_CANDIDATE_RE.match(t) and aspect >= 1.8:
        return "-"
    return text


def _ink_density(
    binary: np.ndarray,
    tb: Dict[str, Any],
) -> float:
    """文本框内前景像素占比（binary 为白底黑字的 INV 结果：前景=255）。"""
    h, w = binary.shape[:2]
    poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
    x1 = int(max(0, np.floor(poly[:, 0].min())))
    y1 = int(max(0, np.floor(poly[:, 1].min())))
    x2 = int(min(w, np.ceil(poly[:, 0].max())))
    y2 = int(min(h, np.ceil(poly[:, 1].max())))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = binary[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / float(roi.size)


def _looks_like_garbage_ink(
    text: str,
    score: float,
    ink: Optional[float],
    median_ink: Optional[float],
) -> bool:
    """
    低置信 + 墨迹显著低于全图中位 → 空白区幻觉框。
    """
    t = (text or "").strip()
    if not t:
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
    # 墨迹极低（相对中位）
    if median_ink <= 1e-6:
        return ink < 0.01 and score < 0.55
    if ink < 0.35 * median_ink and score < 0.60 and len(t) <= 8:
        return True
    if ink < 0.02 and score < 0.55:
        return True
    # 单字 CJK：空角格常见幻觉，墨迹低于中位且置信一般时清空
    if is_single_cjk and score < 0.75 and ink < 0.85 * median_ink:
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
    cleaned = 0
    inks: List[float] = []
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
        norm = _maybe_geometric_dash(tb, norm)
        ink = inks[i] if binary is not None else None
        if _looks_like_garbage_ink(norm, score, ink, median_ink):
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
