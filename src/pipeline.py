"""解耦表格提取主流程：结构识别与 OCR 文本分离后再融合。

优先用 OpenCV 框线重建网格（有框线专利表）；置信度不足时回退 LORE。
OCR 结果可本地缓存，便于反复调结构参数而不重复烧云端额度。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from .formatter import build_markdown_output, format_free_texts
from .lines import (
    DetectedTable,
    binarize_otsu,
    detect_tables,
    imwrite_unicode,
    render_debug_overlay,
)
from .matching import assign_texts_to_cells
from .models import load_lore_model, load_ocr, predict_cells, predict_texts
from .refine import refine_table

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEBUG_DIR = ROOT / "data" / "debug"

# auto 模式下框线网格最低置信度
_LINES_CONF_THRESH = 0.35


def _imread_unicode(path: str) -> np.ndarray:
    """
    Windows 下 cv2.imread 无法正确处理含中文等非 ASCII 路径，
    改用 np.fromfile + imdecode 读取。
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"无法读取图像（空文件或不存在）: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法解码图像: {path}")
    return img


def _load_image(image: Union[str, np.ndarray]) -> np.ndarray:
    """加载为 BGR ndarray，避免中文路径传给 OpenCV/下游模型。"""
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, str):
        p = Path(image)
        if not p.is_file():
            raise FileNotFoundError(f"无法读取图像: {image}")
        return _imread_unicode(str(p))
    raise TypeError("image 须为文件路径或 numpy.ndarray")


def _hough_skew_angle(gray: np.ndarray, max_angle: float) -> Optional[float]:
    """基于霍夫变换的倾斜角检测。"""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_line_length = max(gray.shape[1] // 4, 30)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=20,
    )
    if lines is None or len(lines) == 0:
        return None

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1 and y2 == y1:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle > 90:
            angle -= 180
        elif angle <= -90:
            angle += 180
        if abs(angle) <= max_angle:
            angles.append(angle)

    if not angles:
        return None
    return float(np.median(angles))


def deskew_image(
    image: np.ndarray,
    max_angle: float = 15.0,
    min_angle_threshold: float = 0.1,
) -> np.ndarray:
    """图像倾斜校正（Deskew）。仅信任霍夫长直线角度。"""
    if image is None or getattr(image, "size", 0) == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    angle = _hough_skew_angle(gray, max_angle)

    if angle is None or abs(angle) < min_angle_threshold:
        return image

    logger.info("检测到图像倾斜角 %.3f°，执行 Deskew 校正", angle)

    (h, w) = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rot_mat[0, 2] += (new_w / 2.0) - center[0]
    rot_mat[1, 2] += (new_h / 2.0) - center[1]

    border_value = (255, 255, 255) if image.ndim == 3 else 255
    rotated = cv2.warpAffine(
        image,
        rot_mat,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rotated


def _extract_via_lines(
    image: np.ndarray,
    text_boxes: list,
    *,
    ioa_threshold: float,
    compress_empty_cols: bool,
) -> tuple[str, List[DetectedTable], list, list]:
    """框线路径：检测多表 → refine 补列/拆合并 → 逐表归属 → 拼接 Markdown。"""
    tables = detect_tables(image, confidence_thresh=_LINES_CONF_THRESH)
    if not tables:
        return "", [], text_boxes, []

    binary = binarize_otsu(image)
    # detect → refine（文本聚类补列 + 拆错误纵向合并）
    tables = [refine_table(t, text_boxes) for t in tables]

    bboxes = [t.bbox for t in tables]
    md_parts: List[str] = []
    # 游离文本只算一次：对所有表外文本
    remaining = list(text_boxes)

    for table in tables:
        cells, free = assign_texts_to_cells(
            table.cells,
            remaining,
            ioa_threshold=ioa_threshold,
            split_cross_cell=True,
            table_bboxes=[table.bbox],
            binary=binary,
            col_seps=table.col_seps,
            v_separators=table.v_separators,
        )
        table.cells = cells
        # 已归属进本表的文本不再参与后续表；free 里可能含表外 + 其它表内
        # 用「中心是否在本表 bbox」过滤：本表外的留下给下一张 / 最终 free
        still: list = []
        for tb in free:
            poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
            cx, cy = float(poly[:, 0].mean()), float(poly[:, 1].mean())
            x1, y1, x2, y2 = table.bbox
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                # 表内但未进格——assign 已尽量塞进最近格，这里不应再出现；丢弃防泄漏
                continue
            still.append(tb)
        remaining = still

        # 框线路径也启用子表切分：仅当中段再次出现表题/聚合物表头时才会切开
        md = build_markdown_output(
            cells,
            [],
            split_subtables=True,
            compress_empty_cols=compress_empty_cols,
        )
        if md:
            md_parts.append(md)

    all_free = remaining
    # 再过滤：真正在所有表 bbox 之外的才做前缀
    outside = []
    for tb in all_free:
        poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
        cx, cy = float(poly[:, 0].mean()), float(poly[:, 1].mean())
        if any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in bboxes):
            continue
        outside.append(tb)

    prefix = format_free_texts(outside)
    body = "\n\n".join(md_parts)
    if prefix and body:
        markdown = prefix + "\n\n" + body
    else:
        markdown = prefix or body or ""
    return markdown, tables, outside, text_boxes


def _extract_via_lore(
    image: np.ndarray,
    text_boxes: list,
    *,
    ioa_threshold: float,
    compress_empty_cols: bool,
    lore_pipe=None,
) -> str:
    """LORE 兜底路径。"""
    lore = lore_pipe or load_lore_model()
    cells = predict_cells(image, lore_pipe=lore)
    if not cells:
        return format_free_texts(text_boxes)
    cells, free_texts = assign_texts_to_cells(
        cells,
        text_boxes,
        ioa_threshold=ioa_threshold,
        split_cross_cell=True,
        table_bboxes=None,
        binary=binarize_otsu(image),
    )
    return build_markdown_output(
        cells,
        free_texts,
        split_subtables=True,
        compress_empty_cols=compress_empty_cols,
    )


def extract_table_markdown(
    image_path: Union[str, np.ndarray],
    ioa_threshold: float = 0.5,
    deskew: bool = True,
    max_skew_angle: float = 15.0,
    lore_pipe=None,
    ocr_engine=None,
    *,
    structure: str = "auto",
    use_cache: bool = True,
    refresh_cache: bool = False,
    compress_empty_cols: bool = True,
    debug: bool = False,
    debug_dir: Optional[Union[str, Path]] = None,
    debug_stem: Optional[str] = None,
) -> str:
    """
    复杂表格解耦提取 Pipeline。

    Args:
        structure: "auto" | "lines" | "lore"
            - auto：框线置信度足够走 lines，否则 lore
            - lines：强制框线网格
            - lore：强制 LORE
        use_cache / refresh_cache: OCR 本地缓存控制
        compress_empty_cols: 删除整列为空的幽灵列
        debug: 写出网格叠加图到 data/debug/
    """
    image = _load_image(image_path)
    stem = debug_stem
    if stem is None:
        if isinstance(image_path, str):
            stem = Path(image_path).stem
        else:
            stem = "image"

    # ---------- 1. Deskew ----------
    if deskew:
        image = deskew_image(image, max_angle=max_skew_angle)

    # ---------- 2. OCR（可缓存）----------
    ocr = ocr_engine or load_ocr()
    # 缓存 key 基于 deskew 后的图，避免原图/校正图混用
    text_boxes = predict_texts(
        image,
        ocr_engine=ocr,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        cache_extra=f"deskew={int(deskew)}",
    )

    structure = (structure or "auto").strip().lower()
    tables: List[DetectedTable] = []
    markdown = ""

    # ---------- 3. 结构 ----------
    if structure in {"lines", "auto"}:
        markdown, tables, _, _ = _extract_via_lines(
            image,
            text_boxes,
            ioa_threshold=ioa_threshold,
            compress_empty_cols=compress_empty_cols,
        )
        avg_conf = (
            float(np.mean([t.confidence for t in tables])) if tables else 0.0
        )
        logger.info(
            "框线路径: tables=%d avg_conf=%.3f",
            len(tables),
            avg_conf,
        )
        if structure == "lines":
            pass
        elif not tables or avg_conf < _LINES_CONF_THRESH:
            logger.info("框线置信度不足，回退 LORE")
            markdown = _extract_via_lore(
                image,
                text_boxes,
                ioa_threshold=ioa_threshold,
                compress_empty_cols=compress_empty_cols,
                lore_pipe=lore_pipe,
            )
            tables = []

    elif structure == "lore":
        markdown = _extract_via_lore(
            image,
            text_boxes,
            ioa_threshold=ioa_threshold,
            compress_empty_cols=compress_empty_cols,
            lore_pipe=lore_pipe,
        )
    else:
        raise ValueError(f"未知 structure 模式: {structure!r}，可选 auto/lines/lore")

    # ---------- 4. Debug 叠加图 ----------
    if debug:
        out_dir = Path(debug_dir) if debug_dir else DEFAULT_DEBUG_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        if not tables and structure != "lore":
            # 再跑一次检测以便可视化（即使 conf 低）
            tables = detect_tables(image, confidence_thresh=0.0)
        overlay = render_debug_overlay(image, tables, text_boxes)
        out_path = out_dir / f"{stem}_grid.png"
        imwrite_unicode(str(out_path), overlay)
        logger.info("debug 叠加图已写入: %s", out_path)
        print(f"[info] debug 叠加图: {out_path}")

    if not markdown:
        return format_free_texts(text_boxes)
    return markdown
