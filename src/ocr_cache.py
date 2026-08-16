"""OCR 结果本地缓存：按图片内容 hash + 模型配置落盘，避免重复烧云端额度。"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import ROOT, get_settings

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = ROOT / "data" / "cache"


def _image_bytes(image: Any) -> bytes:
    """把路径或 ndarray 统一成可哈希的字节。"""
    if isinstance(image, (str, Path)):
        return Path(image).read_bytes()
    if isinstance(image, np.ndarray):
        # 用 png 编码保证确定性
        import cv2

        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("无法编码 ndarray 为 PNG 以计算缓存 key")
        return buf.tobytes()
    raise TypeError("image 须为文件路径或 numpy.ndarray")


def cache_key_for_image(image: Any, *, extra: Optional[str] = None) -> str:
    """sha1(图片字节 + 任务/模型配置 [+ extra])。"""
    settings = get_settings()
    meta = (
        f"api=http_v2|task={settings.task}|ocr={settings.ocr_model}"
        f"|vl={settings.vl_model}"
        f"|layout={settings.use_layout_detection}"
        f"|orient={settings.use_doc_orientation_classify}"
        f"|unwarp={settings.use_doc_unwarping}"
        f"|textline={settings.use_textline_orientation}"
        f"|det_limit={settings.text_det_limit_type}:{settings.text_det_limit_side_len}"
        f"|det_thresh={settings.text_det_thresh}"
        f"|det_box={settings.text_det_box_thresh}"
        f"|det_unclip={settings.text_det_unclip_ratio}"
        f"|rec_score={settings.text_rec_score_thresh}"
        f"|md_ignore={','.join(settings.markdown_ignore_labels)}"
        f"|pre_max={settings.preprocess_max_long_side}"
        f"|pre_min_short={settings.preprocess_min_short_side}"
        f"|pre_q={settings.preprocess_jpeg_quality}"
    )
    if extra:
        meta += f"|{extra}"
    h = hashlib.sha1()
    h.update(_image_bytes(image))
    h.update(meta.encode("utf-8"))
    return h.hexdigest()


def _serialize_text_boxes(text_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tb in text_boxes:
        poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
        tl = tb.get("top_left")
        out.append(
            {
                "polygon": poly.tolist(),
                "text": tb.get("text", ""),
                "score": float(tb.get("score", 1.0)),
                "top_left": [float(tl[0]), float(tl[1])] if tl is not None else None,
            }
        )
    return out


def _deserialize_text_boxes(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in raw:
        poly = np.asarray(item["polygon"], dtype=np.float64)
        tl = item.get("top_left")
        out.append(
            {
                "polygon": poly,
                "text": item.get("text", ""),
                "score": float(item.get("score", 1.0)),
                "top_left": (float(tl[0]), float(tl[1])) if tl else None,
            }
        )
    return out


def load_ocr_cache(
    image: Any,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    extra: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """命中缓存则返回 text_boxes，否则 None。"""
    key = cache_key_for_image(image, extra=extra)
    path = cache_dir / f"{key}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        boxes = _deserialize_text_boxes(data.get("text_boxes") or [])
        logger.info("OCR 缓存命中: %s (%d boxes)", path.name, len(boxes))
        return boxes
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR 缓存读取失败 %s: %s", path, exc)
        return None


def save_ocr_cache(
    image: Any,
    text_boxes: List[Dict[str, Any]],
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    extra: Optional[str] = None,
) -> Path:
    """写入 OCR 缓存，返回缓存文件路径。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key_for_image(image, extra=extra)
    path = cache_dir / f"{key}.json"
    payload = {
        "key": key,
        "text_boxes": _serialize_text_boxes(text_boxes),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info("OCR 缓存已写入: %s (%d boxes)", path.name, len(text_boxes))
    return path
