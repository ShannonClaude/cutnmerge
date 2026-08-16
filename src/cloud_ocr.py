"""云端 PaddleOCR：HTTP 提交 / 轮询 / 下载（对齐 source/ppocr_batch.py）。

结构识别仍使用原图像素；本模块仅对上传图做本地预处理，并将 OCR 坐标
按缩放比例映回原图坐标系。
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import requests

from .config import ROOT, get_settings

logger = logging.getLogger(__name__)

DEFAULT_JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
DEFAULT_ARTIFACT_DIR = ROOT / "data" / "ocr"
MAX_RETRY = 3
RETRY_WAIT = 6.0
POLL_INTERVAL = 5.0
# 串行质量优先：两次云端 OCR 之间的间隔（秒）
INTER_REQUEST_WAIT = 1.5
_last_ocr_finish_monotonic: float = 0.0


def _imread_any(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """加载为 BGR（或 BGRA）；路径支持非 ASCII。"""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image.copy()
    path = Path(image)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def preprocess_for_cloud(
    image: Union[str, Path, np.ndarray],
    *,
    max_long_side: int = 2200,
    min_short_side: int = 720,
    jpeg_quality: int = 90,
) -> Tuple[bytes, Tuple[int, int], Tuple[int, int], np.ndarray]:
    """
    本地预处理：去透明 + 矮图放大短边 + 长边上限 + JPEG。

    Returns:
        (jpeg_bytes, (upload_w, upload_h), (orig_w, orig_h), original_bgr)
    """
    original = _imread_any(image)
    if original.shape[2] == 4:
        b, g, r, a = cv2.split(original)
        alpha = a.astype(np.float32) / 255.0
        white = np.ones_like(b, dtype=np.float32) * 255.0
        b = (b.astype(np.float32) * alpha + white * (1.0 - alpha)).astype(np.uint8)
        g = (g.astype(np.float32) * alpha + white * (1.0 - alpha)).astype(np.uint8)
        r = (r.astype(np.float32) * alpha + white * (1.0 - alpha)).astype(np.uint8)
        bgr = cv2.merge([b, g, r])
        original = bgr
    else:
        bgr = original

    orig_h, orig_w = bgr.shape[:2]
    upload = bgr
    short0 = min(orig_w, orig_h)
    # 矮图先放大短边，利于小字召回（坐标稍后映回原图）
    if min_short_side > 0 and short0 > 0 and short0 < min_short_side:
        scale_up = float(min_short_side) / float(short0)
        new_w = max(1, int(round(orig_w * scale_up)))
        new_h = max(1, int(round(orig_h * scale_up)))
        upload = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.info(
            "本地放大短边: %dx%d → %dx%d (min_short=%d)",
            orig_w,
            orig_h,
            new_w,
            new_h,
            min_short_side,
        )
    uh, uw = upload.shape[:2]
    if max(uw, uh) > max_long_side:
        scale = max_long_side / float(max(uw, uh))
        new_w = max(1, int(uw * scale))
        new_h = max(1, int(uh * scale))
        upload = cv2.resize(upload, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(
        ".jpg", upload, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    if not ok:
        raise RuntimeError("无法将预处理图像编码为 JPEG")
    uh, uw = upload.shape[:2]
    return buf.tobytes(), (uw, uh), (orig_w, orig_h), original


def _jobs_url() -> str:
    settings = get_settings()
    if settings.jobs_url:
        return settings.jobs_url.rstrip("/")
    if settings.base_url:
        base = settings.base_url.rstrip("/")
        if base.endswith("/ocr/jobs"):
            return base
        return f"{base}/api/v2/ocr/jobs"
    return DEFAULT_JOB_URL


def _auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"bearer {token}"}


def _build_optional_payload(
    *,
    image_wh: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    camelCase optionalPayload；值来自 .env。

    若配置为 min，且按 limitSideLen 放大后长边将顶云端上限(~4000)，
    钳制 limitSideLen 并继续 min（不再切 max，以免矮宽表漏检）。
    """
    s = get_settings()
    payload: Dict[str, Any] = {
        "markdownIgnoreLabels": list(s.markdown_ignore_labels or []),
        "useDocOrientationClassify": bool(s.use_doc_orientation_classify),
        "useDocUnwarping": bool(s.use_doc_unwarping),
        "useTextlineOrientation": bool(s.use_textline_orientation),
    }
    limit_type = (s.text_det_limit_type or "").strip().lower() or None
    limit_side = s.text_det_limit_side_len
    cloud_max_side = 4000
    if (
        limit_type == "min"
        and limit_side is not None
        and image_wh is not None
        and min(image_wh) > 0
        and min(image_wh) < int(limit_side)
    ):
        short_s, long_s = min(image_wh), max(image_wh)
        projected_long = long_s * (float(limit_side) / float(short_s))
        safe_long = cloud_max_side * 0.95
        if projected_long >= safe_long:
            clamped = int(safe_long * float(short_s) / float(long_s))
            clamped = max(960, min(int(limit_side), clamped))
            logger.info(
                "钳制 textDetLimitSideLen: %s→%d（保持 min；短边=%d 原投影长边≈%.0f）",
                limit_side,
                clamped,
                short_s,
                projected_long,
            )
            limit_side = clamped
    if limit_type:
        payload["textDetLimitType"] = limit_type
    if limit_side is not None:
        payload["textDetLimitSideLen"] = int(limit_side)
    if s.text_det_thresh is not None:
        payload["textDetThresh"] = float(s.text_det_thresh)
    if s.text_det_box_thresh is not None:
        payload["textDetBoxThresh"] = float(s.text_det_box_thresh)
    if s.text_det_unclip_ratio is not None:
        payload["textDetUnclipRatio"] = float(s.text_det_unclip_ratio)
    if s.text_rec_score_thresh is not None:
        payload["textRecScoreThresh"] = float(s.text_rec_score_thresh)
    return payload


def _wait_inter_request() -> None:
    """两次 OCR 之间稍等，质量优先、降低连打压力。"""
    global _last_ocr_finish_monotonic
    if _last_ocr_finish_monotonic <= 0:
        return
    elapsed = time.monotonic() - _last_ocr_finish_monotonic
    remain = INTER_REQUEST_WAIT - elapsed
    if remain > 0:
        time.sleep(remain)


def _mark_ocr_finished() -> None:
    global _last_ocr_finish_monotonic
    _last_ocr_finish_monotonic = time.monotonic()


def _format_api_error(prefix: str, *, detail: Any) -> str:
    """把云端错误整理成可读多行文本。"""
    if isinstance(detail, (dict, list)):
        body = json.dumps(detail, ensure_ascii=False, indent=2)
    else:
        body = str(detail)
    if len(body) > 4000:
        body = body[:4000] + "\n…(truncated)"
    return f"{prefix}\n{body}"


def _submit_job(
    jpeg_bytes: bytes,
    *,
    filename: str,
    token: str,
    model: str,
    timeout: float,
    optional: Optional[Dict[str, Any]] = None,
) -> str:
    url = _jobs_url()
    headers = _auth_headers(token)
    optional = optional if optional is not None else _build_optional_payload()
    files = {"file": (filename, io.BytesIO(jpeg_bytes), "image/jpeg")}
    data = {
        "model": model,
        "optionalPayload": json.dumps(optional, ensure_ascii=False),
    }
    resp = requests.post(url, headers=headers, data=data, files=files, timeout=timeout)
    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise RuntimeError(
            _format_api_error(
                f"OCR 任务提交失败 HTTP {resp.status_code} | url={url} | "
                f"file={filename} | bytes={len(jpeg_bytes)} | model={model} | "
                f"optionalPayload={json.dumps(optional, ensure_ascii=False)}",
                detail=detail,
            )
        )
    body = resp.json()
    job_id = (body.get("data") or {}).get("jobId")
    if not job_id:
        raise RuntimeError(
            _format_api_error("OCR 提交响应缺少 jobId", detail=body)
        )
    return str(job_id)


def _poll_job(
    job_id: str,
    *,
    token: str,
    poll_timeout: float,
    request_timeout: float,
) -> str:
    """轮询至 done，返回 jsonUrl。"""
    url = f"{_jobs_url()}/{job_id}"
    headers = _auth_headers(token)
    deadline = time.monotonic() + float(poll_timeout)
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"OCR 轮询超时 ({poll_timeout}s)，jobId={job_id}")
        time.sleep(POLL_INTERVAL)
        resp = requests.get(url, headers=headers, timeout=request_timeout)
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:  # noqa: BLE001
                detail = resp.text
            raise RuntimeError(
                _format_api_error(
                    f"OCR 轮询失败 HTTP {resp.status_code} | jobId={job_id} | url={url}",
                    detail=detail,
                )
            )
        payload = resp.json()
        data = payload.get("data") or {}
        state = data.get("state")
        if state == "done":
            result_url = data.get("resultUrl") or {}
            json_url = result_url.get("jsonUrl") or result_url.get("json_url")
            if not json_url:
                raise RuntimeError(
                    _format_api_error(
                        f"OCR 完成但缺少 jsonUrl | jobId={job_id}",
                        detail=payload,
                    )
                )
            return str(json_url)
        if state == "failed":
            err = (
                data.get("errorMsg")
                or data.get("error_msg")
                or data.get("message")
                or data.get("msg")
                or ""
            )
            raise RuntimeError(
                _format_api_error(
                    f"OCR 任务失败 | jobId={job_id} | errorMsg={err!r}",
                    detail=payload,
                )
            )
        logger.info("OCR job %s state=%s", job_id, state)


def _download_jsonl(json_url: str, *, timeout: float) -> List[dict]:
    resp = requests.get(json_url, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"下载 OCR 结果失败 ({resp.status_code}): {resp.text[:500]}")
    lines: List[dict] = []
    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        lines.append(json.loads(line))
    if not lines:
        raise RuntimeError("OCR 结果 JSONL 为空")
    return lines


def _to_quad(pts: Any) -> Optional[np.ndarray]:
    arr = np.asarray(pts, dtype=np.float64).reshape(-1)
    if arr.size == 4:
        x1, y1, x2, y2 = arr.tolist()
        return np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float64,
        )
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] < 4:
        return None
    return arr[:4].copy()


def _top_left_from_quad(quad: np.ndarray) -> Tuple[float, float]:
    sums = quad.sum(axis=1)
    tl_idx = int(np.argmin(sums))
    return float(quad[tl_idx, 0]), float(quad[tl_idx, 1])


def _append_text_box(
    out: List[Dict[str, Any]],
    text: Any,
    box: Any,
    score: float = 1.0,
) -> None:
    text_s = str(text).strip() if text is not None else ""
    if not text_s:
        return
    quad = _to_quad(box)
    if quad is None:
        return
    out.append(
        {
            "polygon": quad,
            "text": text_s,
            "score": float(score),
            "top_left": _top_left_from_quad(quad),
        }
    )


def _parse_pruned_result(pruned: Any) -> List[Dict[str, Any]]:
    text_boxes: List[Dict[str, Any]] = []
    if not isinstance(pruned, dict):
        return text_boxes
    texts = pruned.get("rec_texts")
    polys = (
        pruned.get("rec_polys")
        or pruned.get("dt_polys")
        or pruned.get("rec_boxes")
        or pruned.get("dt_boxes")
    )
    scores = pruned.get("rec_scores") or []
    if isinstance(texts, (list, tuple)) and isinstance(polys, (list, tuple, np.ndarray)):
        poly_list = list(polys)
        for i, text in enumerate(texts):
            box = poly_list[i] if i < len(poly_list) else None
            score = float(scores[i]) if i < len(scores) else 1.0
            if box is not None:
                _append_text_box(text_boxes, text, box, score)
    return text_boxes


def _normalize_aabb(coords: Any, width: float, height: float) -> Optional[np.ndarray]:
    """source 风格：归一化 [x1,y1,x2,y2] → 像素四点。"""
    try:
        c = list(coords)
        if len(c) < 4:
            return None
        # 已是四点
        if isinstance(c[0], (list, tuple)) or (
            hasattr(c[0], "__len__") and not isinstance(c[0], (str, bytes))
        ):
            return _to_quad(c)
        x1, y1, x2, y2 = float(c[0]), float(c[1]), float(c[2]), float(c[3])
        # 归一化坐标
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        return _to_quad([x1, y1, x2, y2])
    except Exception:  # noqa: BLE001
        return None


def parse_page_result(page_line: dict) -> Tuple[List[Dict[str, Any]], dict]:
    """
    解析 JSONL 单行 → (upload 坐标系 text_boxes, raw_result_dict)。

    优先 prunedResult（绝对像素四点）；回退 ocrResults[].coordinates。
    """
    raw = page_line.get("result") if isinstance(page_line.get("result"), dict) else page_line
    if not isinstance(raw, dict):
        return [], {}

    boxes: List[Dict[str, Any]] = []
    ocr_results = raw.get("ocrResults") or []
    data_info = raw.get("dataInfo") or {}
    width = float(data_info.get("width") or raw.get("width") or 0) or 1.0
    height = float(data_info.get("height") or raw.get("height") or 0) or 1.0

    for item in ocr_results:
        if not isinstance(item, dict):
            continue
        pruned = item.get("prunedResult") or item.get("pruned_result")
        page_boxes = _parse_pruned_result(pruned)
        if page_boxes:
            boxes.extend(page_boxes)
            continue
        coords = item.get("coordinates")
        if coords is None:
            continue
        w = float(item.get("width") or width)
        h = float(item.get("height") or height)
        quad = _normalize_aabb(coords, w, h)
        if quad is None:
            continue
        _append_text_box(
            boxes,
            item.get("text", ""),
            quad,
            float(item.get("score", 0.0) or 0.0),
        )

    if not boxes:
        pruned = raw.get("prunedResult") or raw.get("pruned_result")
        boxes = _parse_pruned_result(pruned)

    return boxes, raw


def scale_boxes_to_original(
    text_boxes: List[Dict[str, Any]],
    *,
    upload_wh: Tuple[int, int],
    orig_wh: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """把上传图坐标系映回原图。"""
    uw, uh = upload_wh
    ow, oh = orig_wh
    if uw <= 0 or uh <= 0:
        return text_boxes
    sx = float(ow) / float(uw)
    sy = float(oh) / float(uh)
    if abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9:
        return text_boxes
    out: List[Dict[str, Any]] = []
    for tb in text_boxes:
        poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2).copy()
        poly[:, 0] *= sx
        poly[:, 1] *= sy
        tl = _top_left_from_quad(poly)
        out.append(
            {
                "polygon": poly,
                "text": tb.get("text", ""),
                "score": float(tb.get("score", 1.0)),
                "top_left": tl,
            }
        )
    return out


def _serialize_page_json(text_boxes: List[Dict[str, Any]]) -> List[dict]:
    rows: List[dict] = []
    for tb in text_boxes:
        poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
        tl = tb.get("top_left") or _top_left_from_quad(poly)
        rows.append(
            {
                "polygon": [[int(round(x)), int(round(y))] for x, y in poly.tolist()],
                "text": tb.get("text", ""),
                "score": round(float(tb.get("score", 0.0)), 4),
                "top_left": {"x": int(round(tl[0])), "y": int(round(tl[1]))},
            }
        )
    return rows


def _write_csv(path: Path, text_boxes: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["序号", "文字", "置信度", "左上角X", "左上角Y", "四角坐标"])
        for i, tb in enumerate(text_boxes, 1):
            poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
            tl = tb.get("top_left") or _top_left_from_quad(poly)
            poly_i = [[int(round(x)), int(round(y))] for x, y in poly.tolist()]
            writer.writerow(
                [
                    i,
                    tb.get("text", ""),
                    round(float(tb.get("score", 0.0)), 4),
                    int(round(tl[0])),
                    int(round(tl[1])),
                    json.dumps(poly_i, ensure_ascii=False),
                ]
            )


def _draw_annotated(image_bgr: np.ndarray, text_boxes: List[Dict[str, Any]]) -> np.ndarray:
    canvas = image_bgr.copy()
    for tb in text_boxes:
        poly = np.asarray(tb["polygon"], dtype=np.int32).reshape(-1, 2)
        cv2.polylines(canvas, [poly], True, (0, 180, 0), 1, lineType=cv2.LINE_AA)
    return canvas


def _imwrite(path: Path, image_bgr: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext, image_bgr)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def _download_api_image(raw: dict, dest: Path, *, timeout: float) -> bool:
    ocr_results = raw.get("ocrResults") or []
    url = None
    for item in ocr_results:
        if isinstance(item, dict):
            url = item.get("ocrImage") or item.get("inputImage")
            if url:
                break
    if not url:
        imgs = raw.get("preprocessedImages") or []
        if imgs:
            url = imgs[0]
    if not url:
        return False
    try:
        resp = requests.get(str(url), timeout=timeout)
        if resp.status_code != 200 or not resp.content:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("下载 API 图失败: %s", exc)
        return False


def save_ocr_artifacts(
    *,
    stem: str,
    text_boxes: List[Dict[str, Any]],
    raw_pages: List[dict],
    original_bgr: np.ndarray,
    artifact_dir: Optional[Path] = None,
    request_timeout: float = 300.0,
) -> List[str]:
    """写出 page json / raw / csv / annotated / api 图。返回路径列表。"""
    out_dir = Path(artifact_dir) if artifact_dir else DEFAULT_ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    # 多页时按页拆；主流程通常一页
    n_pages = max(len(raw_pages), 1)
    for page_i in range(n_pages):
        suffix = f"_page_{page_i + 1:03d}"
        page_boxes = text_boxes if n_pages == 1 else text_boxes  # 整图一次 OCR 通常单页
        raw = raw_pages[page_i] if page_i < len(raw_pages) else {}
        raw_result = raw.get("result") if isinstance(raw.get("result"), dict) else raw

        json_path = out_dir / f"{stem}{suffix}.json"
        json_path.write_text(
            json.dumps(_serialize_page_json(page_boxes), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.append(str(json_path))

        raw_path = out_dir / f"{stem}{suffix}_raw.json"
        raw_path.write_text(
            json.dumps(raw_result if raw_result else raw, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.append(str(raw_path))

        csv_path = out_dir / f"{stem}{suffix}.csv"
        _write_csv(csv_path, page_boxes)
        written.append(str(csv_path))

        ann = _draw_annotated(original_bgr, page_boxes)
        ann_path = out_dir / f"{stem}{suffix}_annotated.jpg"
        if _imwrite(ann_path, ann):
            written.append(str(ann_path))

        api_path = out_dir / f"{stem}{suffix}_api.jpg"
        if isinstance(raw_result, dict) and _download_api_image(
            raw_result, api_path, timeout=request_timeout
        ):
            written.append(str(api_path))

    for p in written:
        logger.info("OCR 产物: %s", p)
    return written


def run_cloud_ocr(
    image: Union[str, Path, np.ndarray],
    *,
    filename: str = "image.jpg",
    save_artifacts: bool = False,
    artifact_stem: Optional[str] = None,
    artifact_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    预处理 → 提交 → 轮询 → 下载 → 解析 → 坐标映回原图。

    Returns:
        原图坐标系下的 text_boxes
    """
    settings = get_settings()
    if settings.task in {"doc_parsing", "vl", "document_parsing"}:
        raise RuntimeError(
            "HTTP OCR 路径仅支持 PADDLEOCR_TASK=ocr；"
            "doc_parsing 请改回 ocr 或另行接入 VL jobs API。"
        )
    token = settings.require_token()
    jpeg_bytes, upload_wh, orig_wh, original_bgr = preprocess_for_cloud(
        image,
        max_long_side=settings.preprocess_max_long_side,
        min_short_side=settings.preprocess_min_short_side,
        jpeg_quality=settings.preprocess_jpeg_quality,
    )

    # 用上传图尺寸决定 min 的 side 钳制（与云端实际输入一致）
    optional = _build_optional_payload(image_wh=upload_wh)
    cfg_side = settings.text_det_limit_side_len
    used_side = optional.get("textDetLimitSideLen")
    if (
        (settings.text_det_limit_type or "").lower() == "min"
        and cfg_side is not None
        and used_side is not None
        and int(used_side) < int(cfg_side)
    ):
        msg = (
            f"OCR 钳制 limitSideLen: {cfg_side}→{used_side}（保持 min；"
            f"upload={upload_wh[0]}x{upload_wh[1]}）"
        )
        logger.info(msg)
        print(f"[ocr] {msg}", flush=True)

    ctx = (
        f"file={filename} orig={orig_wh[0]}x{orig_wh[1]} "
        f"upload={upload_wh[0]}x{upload_wh[1]} jpeg_bytes={len(jpeg_bytes)} "
        f"model={settings.ocr_model} optionalPayload={json.dumps(optional, ensure_ascii=False)}"
    )

    _wait_inter_request()
    last_err: Exception | None = None
    json_url: Optional[str] = None
    try:
        for try_cnt in range(1, MAX_RETRY + 1):
            if try_cnt > 1:
                msg = f"OCR 第 {try_cnt} 次重试，等待 {RETRY_WAIT:.0f}s… | {ctx}"
                logger.info(msg)
                print(f"[ocr] {msg}", flush=True)
                time.sleep(RETRY_WAIT)
            try:
                job_id = _submit_job(
                    jpeg_bytes,
                    filename=filename,
                    token=token,
                    model=settings.ocr_model,
                    timeout=settings.request_timeout,
                    optional=optional,
                )
                logger.info("OCR 已提交 jobId=%s | %s", job_id, ctx)
                print(f"[ocr] 已提交 jobId={job_id} | {ctx}", flush=True)
                json_url = _poll_job(
                    job_id,
                    token=token,
                    poll_timeout=settings.poll_timeout,
                    request_timeout=settings.request_timeout,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                fail_msg = f"OCR 尝试 {try_cnt}/{MAX_RETRY} 失败 | {ctx}\n{exc}"
                logger.warning(fail_msg)
                print(f"[ocr] {fail_msg}", flush=True)
                json_url = None

        if not json_url:
            raise RuntimeError(
                f"OCR 重试 {MAX_RETRY} 次仍失败 | {ctx}\n最后错误: {last_err}"
            ) from last_err
    finally:
        _mark_ocr_finished()

    pages = _download_jsonl(json_url, timeout=settings.request_timeout)
    all_boxes: List[Dict[str, Any]] = []
    raw_pages: List[dict] = []
    for page in pages:
        boxes, _raw = parse_page_result(page)
        all_boxes.extend(boxes)
        raw_pages.append(page)

    text_boxes = scale_boxes_to_original(
        all_boxes, upload_wh=upload_wh, orig_wh=orig_wh
    )

    if save_artifacts and artifact_stem:
        try:
            save_ocr_artifacts(
                stem=artifact_stem,
                text_boxes=text_boxes,
                raw_pages=raw_pages,
                original_bgr=original_bgr,
                artifact_dir=artifact_dir or settings.ocr_artifact_dir,
                request_timeout=settings.request_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("写入 OCR 产物失败: %s", exc)

    return text_boxes
