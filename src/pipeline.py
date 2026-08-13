"""解耦表格提取主流程：结构识别与 OCR 文本分离后再融合。

默认 TableStructureRec（tsr）+ 云端 OCR + IoA；输出 HTML（保留 rowspan/colspan），
可选 Markdown。OCR 结果可本地缓存。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from .formatter import build_markdown_output, format_free_texts
from .grid_fusion import fuse_tsr_with_lines
from .html_formatter import build_html_output
from .lines import (
    DetectedTable,
    binarize_otsu,
    detect_tables,
    imwrite_unicode,
    render_debug_overlay,
)
from .matching import assign_texts_to_cells
from .models import load_lore_model, load_ocr, predict_cells, predict_texts
from .ocr_post import postprocess_text_boxes
from .orient import apply_orientation_axis, ensure_upright_axis, maybe_flip_180_by_ocr
from .reocr import apply_reocr_to_cells
from .refine import refine_table
from .tsr import (
    cells_to_debug_table,
    predict_cells_tsr,
    render_table_vis,
    render_table_vis_logic,
)
from .tsr_refine import (
    coverage_score,
    reconstruct_header_cells,
    refine_tsr_cells,
    refine_tsr_cells_light,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEBUG_DIR = ROOT / "data" / "debug"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "output"

# auto 模式下框线网格最低置信度（仅 --structure lines/auto 的旧路径）
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


def _render_outputs(
    cells: list,
    free_texts: list,
    *,
    compress_empty_cols: bool,
) -> Dict[str, str]:
    """同时生成 html / md。"""
    html = build_html_output(
        cells,
        free_texts,
        split_subtables=True,
        compress_empty=compress_empty_cols,
    )
    md = build_markdown_output(
        cells,
        free_texts,
        split_subtables=True,
        compress_empty_cols=compress_empty_cols,
    )
    return {"html": html or "", "md": md or ""}


def _extract_via_lines(
    image: np.ndarray,
    text_boxes: list,
    *,
    ioa_threshold: float,
    compress_empty_cols: bool,
    reocr: bool = False,
    reocr_max_cells: int = 24,
    ocr_engine=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> tuple[Dict[str, str], List[DetectedTable], list, list]:
    """框线路径：检测多表 → refine 补列/拆合并 → 逐表归属 → 拼接输出。"""
    tables = detect_tables(image, confidence_thresh=_LINES_CONF_THRESH)
    if not tables:
        free = format_free_texts(text_boxes)
        return {"html": free, "md": free}, [], text_boxes, []

    binary = binarize_otsu(image)
    tables = [refine_table(t, text_boxes) for t in tables]

    bboxes = [t.bbox for t in tables]
    html_parts: List[str] = []
    md_parts: List[str] = []
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
        if reocr:
            cells = apply_reocr_to_cells(
                image,
                cells,
                binary=binary,
                ocr_engine=ocr_engine,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
                max_cells=reocr_max_cells,
            )
        table.cells = cells
        still: list = []
        for tb in free:
            poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
            cx, cy = float(poly[:, 0].mean()), float(poly[:, 1].mean())
            x1, y1, x2, y2 = table.bbox
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                continue
            still.append(tb)
        remaining = still

        outs = _render_outputs(
            cells, [], compress_empty_cols=compress_empty_cols
        )
        if outs["html"]:
            html_parts.append(outs["html"])
        if outs["md"]:
            md_parts.append(outs["md"])

    outside = []
    for tb in remaining:
        poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
        cx, cy = float(poly[:, 0].mean()), float(poly[:, 1].mean())
        if any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in bboxes):
            continue
        outside.append(tb)

    prefix = format_free_texts(outside)
    html_body = "\n\n".join(html_parts)
    md_body = "\n\n".join(md_parts)
    # lines 路径的游离前缀也包成 HTML 段落，避免裸文本与 <table> 混排
    import html as html_lib

    prefix_html = ""
    if prefix:
        paras = [html_lib.escape(line) for line in prefix.splitlines() if line.strip()]
        prefix_html = "\n".join(f"<p>{p}</p>" for p in paras)

    if prefix_html and html_body:
        html = prefix_html + "\n\n" + html_body
    else:
        html = prefix_html or html_body or ""
    if prefix and md_body:
        md = prefix + "\n\n" + md_body
    else:
        md = prefix or md_body or ""
    return {"html": html, "md": md}, tables, outside, text_boxes


def _extract_via_lore(
    image: np.ndarray,
    text_boxes: list,
    *,
    ioa_threshold: float,
    compress_empty_cols: bool,
    reocr: bool = False,
    reocr_max_cells: int = 24,
    ocr_engine=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    lore_pipe=None,
) -> tuple[Dict[str, str], List[DetectedTable]]:
    """LORE 兜底路径。"""
    lore = lore_pipe or load_lore_model()
    cells = predict_cells(image, lore_pipe=lore)
    if not cells:
        free = format_free_texts(text_boxes)
        return {"html": free, "md": free}, []
    cells, free_texts = assign_texts_to_cells(
        cells,
        text_boxes,
        ioa_threshold=ioa_threshold,
        split_cross_cell=True,
        table_bboxes=None,
        binary=binarize_otsu(image),
    )
    if reocr:
        cells = apply_reocr_to_cells(
            image,
            cells,
            binary=binarize_otsu(image),
            ocr_engine=ocr_engine,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            max_cells=reocr_max_cells,
        )
    outs = _render_outputs(
        cells, free_texts, compress_empty_cols=compress_empty_cols
    )
    return outs, [cells_to_debug_table(cells)]


def _extract_via_tsr(
    image: np.ndarray,
    text_boxes: list,
    *,
    ioa_threshold: float,
    compress_empty_cols: bool,
    fallback_lines: bool = False,
    reocr: bool = False,
    reocr_max_cells: int = 24,
    ocr_engine=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    tsr_kind: Optional[str] = None,
    tsr_aggressive: bool = False,
) -> tuple[Dict[str, str], List[DetectedTable]]:
    """TableStructureRec 路径：默认信任库拓扑，仅填格；激进后处理可选。"""
    force_kind = None
    if tsr_kind:
        k = str(tsr_kind).strip().lower()
        if k in {"wired", "lineless"}:
            force_kind = k
    cells = predict_cells_tsr(
        image, text_boxes=text_boxes, force_kind=force_kind
    )
    line_tables: list = []
    if cells:
        if tsr_aggressive:
            cells = refine_tsr_cells(cells, text_boxes)
            line_tables = detect_tables(
                image, confidence_thresh=0.0, text_boxes=text_boxes
            )
            if line_tables:
                cells = fuse_tsr_with_lines(cells, line_tables)
        else:
            cells = refine_tsr_cells_light(cells)
            logger.info("TSR-first 轻量路径：跳过激进 refine / 线融合")

    cov = coverage_score(cells, text_boxes) if cells else 0.0
    # 覆盖率偏低或格子过少时回退（竖排表头等场景常出现「有格但几乎填不进」）
    n_boxes = max(len(text_boxes), 1)
    too_few_cells = bool(cells) and len(cells) < max(8, n_boxes // 8)
    if (not cells or cov < 0.55 or too_few_cells) and fallback_lines:
        logger.info(
            "TSR 质量不足(cov=%.3f cells=%d boxes=%d)，--fallback-lines 回退框线路径",
            cov,
            len(cells or []),
            len(text_boxes),
        )
        outs, tables, _, _ = _extract_via_lines(
            image,
            text_boxes,
            ioa_threshold=ioa_threshold,
            compress_empty_cols=compress_empty_cols,
            reocr=reocr,
            reocr_max_cells=reocr_max_cells,
            ocr_engine=ocr_engine,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
        )
        return outs, tables

    if not cells:
        free = format_free_texts(text_boxes)
        return {"html": free, "md": free}, []

    binary = binarize_otsu(image)
    # 轻量路径也开跨列切分：只切 OCR 文本到已有原子列，不插线/不补列。
    # col_seps / v_separators 仅激进路径提供，避免 _explode_colspan 改合并表头拓扑。
    col_seps = None
    v_separators = None
    split_cross = True

    if tsr_aggressive:
        from .grid_evidence import apply_grid_evidence_merge

        cells = apply_grid_evidence_merge(
            cells,
            text_boxes,
            line_tables=line_tables,
        )
        cells = reconstruct_header_cells(cells, text_boxes)
        from .tsr_refine import _derive_seps as _derive_seps_local

        _row_seps, col_seps_list = _derive_seps_local(cells)
        if len(col_seps_list) >= 3:
            col_seps = col_seps_list
        if line_tables:
            best = max(
                line_tables,
                key=lambda t: float(getattr(t, "confidence", 0.0) or 0.0),
            )
            v_separators = getattr(best, "v_separators", None)
    else:
        # 默认路径：仅表头带按列横线连通性局部恢复 rowspan（不改列、不开激进融合）
        from .hline_repair import repair_rowspans_by_hline_gaps

        cells = repair_rowspans_by_hline_gaps(cells, binary, text_boxes)

    cells, free_texts = assign_texts_to_cells(
        cells,
        text_boxes,
        ioa_threshold=ioa_threshold,
        split_cross_cell=split_cross,
        table_bboxes=None,
        binary=binary,
        col_seps=col_seps,
        v_separators=v_separators,
    )
    if reocr:
        cells = apply_reocr_to_cells(
            image,
            cells,
            binary=binary,
            ocr_engine=ocr_engine,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            max_cells=reocr_max_cells,
        )
    outs = _render_outputs(
        cells, free_texts, compress_empty_cols=compress_empty_cols
    )
    return outs, [cells_to_debug_table(cells)]


def extract_table_output(
    image_path: Union[str, np.ndarray],
    ioa_threshold: float = 0.5,
    deskew: bool = True,
    max_skew_angle: float = 15.0,
    lore_pipe=None,
    ocr_engine=None,
    *,
    structure: str = "tsr",
    use_cache: bool = True,
    refresh_cache: bool = False,
    compress_empty_cols: bool = True,
    fallback_lines: bool = False,
    orientation: Union[str, int] = "auto",
    debug: bool = False,
    debug_dir: Optional[Union[str, Path]] = None,
    debug_stem: Optional[str] = None,
    reocr: bool = False,
    reocr_max_cells: int = 24,
    tsr_kind: Optional[str] = None,
    tsr_aggressive: bool = False,
    save_vis: bool = True,
    vis_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    复杂表格解耦提取 Pipeline。

    Returns:
        {"html": str, "md": str, "structure": str, "orientation": int,
         "vis_paths": list[str]}
    """
    image = _load_image(image_path)
    stem = debug_stem
    if stem is None:
        if isinstance(image_path, str):
            stem = Path(image_path).stem
        else:
            stem = "image"

    ocr = ocr_engine or load_ocr()
    image, axis_angle, orient_kind = apply_orientation_axis(
        image, mode=orientation
    )

    if deskew:
        image = deskew_image(image, max_angle=max_skew_angle)

    provisional_orient = int(axis_angle)
    cache_extra = f"deskew={int(deskew)}|orient={provisional_orient}"
    text_boxes = predict_texts(
        image,
        ocr_engine=ocr,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        cache_extra=cache_extra,
    )

    orient_angle = provisional_orient
    if orient_kind == "auto":
        image, axis_delta, text_boxes = ensure_upright_axis(
            image,
            text_boxes,
            ocr_engine=ocr,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_extra_base=cache_extra,
        )
        orient_angle = (provisional_orient + axis_delta) % 360
        if axis_delta:
            cache_extra = f"deskew={int(deskew)}|orient={orient_angle}"
        image, flip, text_boxes = maybe_flip_180_by_ocr(
            image,
            text_boxes,
            ocr_engine=ocr,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_extra_base=cache_extra,
        )
        orient_angle = (orient_angle + flip) % 360
        if axis_delta or flip:
            try:
                from .ocr_cache import save_ocr_cache

                save_ocr_cache(
                    image,
                    text_boxes,
                    extra=f"deskew={int(deskew)}|orient={orient_angle}",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("写入 OCR 缓存失败: %s", exc)

    binary = binarize_otsu(image)
    text_boxes = postprocess_text_boxes(text_boxes, binary=binary)

    structure = (structure or "tsr").strip().lower()
    # auto 已废弃：直接走 tsr，避免误导
    if structure == "auto":
        logger.info("structure=auto 已废弃，改走 tsr")
        structure = "tsr"

    tables: List[DetectedTable] = []
    outputs: Dict[str, str] = {"html": "", "md": ""}

    if structure == "lines":
        outputs, tables, _, _ = _extract_via_lines(
            image,
            text_boxes,
            ioa_threshold=ioa_threshold,
            compress_empty_cols=compress_empty_cols,
            reocr=reocr,
            reocr_max_cells=reocr_max_cells,
            ocr_engine=ocr,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
        )
        avg_conf = (
            float(np.mean([t.confidence for t in tables])) if tables else 0.0
        )
        logger.info("框线路径: tables=%d avg_conf=%.3f", len(tables), avg_conf)
    elif structure == "lore":
        outputs, tables = _extract_via_lore(
            image,
            text_boxes,
            ioa_threshold=ioa_threshold,
            compress_empty_cols=compress_empty_cols,
            reocr=reocr,
            reocr_max_cells=reocr_max_cells,
            ocr_engine=ocr,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            lore_pipe=lore_pipe,
        )
    elif structure == "tsr":
        outputs, tables = _extract_via_tsr(
            image,
            text_boxes,
            ioa_threshold=ioa_threshold,
            compress_empty_cols=compress_empty_cols,
            fallback_lines=fallback_lines,
            reocr=reocr,
            reocr_max_cells=reocr_max_cells,
            ocr_engine=ocr,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            tsr_kind=tsr_kind,
            tsr_aggressive=tsr_aggressive,
        )
    else:
        raise ValueError(
            f"未知 structure 模式: {structure!r}，可选 tsr/lines/lore/auto"
        )

    vis_paths: List[str] = []
    if save_vis and tables:
        out_dir = Path(vis_dir) if vis_dir else DEFAULT_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        vis_img = render_table_vis(image, tables=tables, colorful=True)
        vis_path = out_dir / f"{stem}_table_vis.png"
        if imwrite_unicode(str(vis_path), vis_img):
            vis_paths.append(str(vis_path))
            logger.info("表格划线可视化已写入: %s", vis_path)
            print(f"[info] 表格划线图: {vis_path}")
        logic_img = render_table_vis_logic(image, tables=tables)
        logic_path = out_dir / f"{stem}_table_vis_logic.png"
        if imwrite_unicode(str(logic_path), logic_img):
            vis_paths.append(str(logic_path))
            logger.info("表格逻辑标注图已写入: %s", logic_path)
            print(f"[info] 表格逻辑图: {logic_path}")

    if debug:
        out_dir = Path(debug_dir) if debug_dir else DEFAULT_DEBUG_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        if not tables and structure == "lines":
            tables = detect_tables(image, confidence_thresh=0.0, text_boxes=text_boxes)
        overlay = render_debug_overlay(image, tables, text_boxes)
        out_path = out_dir / f"{stem}_grid.png"
        imwrite_unicode(str(out_path), overlay)
        logger.info("debug 叠加图已写入: %s", out_path)
        print(f"[info] debug 叠加图: {out_path}")

    if not outputs.get("html") and not outputs.get("md"):
        free = format_free_texts(text_boxes)
        outputs = {"html": free, "md": free}

    return {
        "html": outputs.get("html") or "",
        "md": outputs.get("md") or "",
        "structure": structure,
        "orientation": int(orient_angle),
        "vis_paths": vis_paths,
    }


def extract_table_markdown(
    image_path: Union[str, np.ndarray],
    ioa_threshold: float = 0.5,
    deskew: bool = True,
    max_skew_angle: float = 15.0,
    lore_pipe=None,
    ocr_engine=None,
    *,
    structure: str = "tsr",
    use_cache: bool = True,
    refresh_cache: bool = False,
    compress_empty_cols: bool = True,
    fallback_lines: bool = False,
    orientation: Union[str, int] = "auto",
    debug: bool = False,
    debug_dir: Optional[Union[str, Path]] = None,
    debug_stem: Optional[str] = None,
    reocr: bool = False,
    reocr_max_cells: int = 24,
    tsr_aggressive: bool = False,
) -> str:
    """兼容旧接口：返回 Markdown 字符串。"""
    out = extract_table_output(
        image_path,
        ioa_threshold=ioa_threshold,
        deskew=deskew,
        max_skew_angle=max_skew_angle,
        lore_pipe=lore_pipe,
        ocr_engine=ocr_engine,
        structure=structure,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        compress_empty_cols=compress_empty_cols,
        fallback_lines=fallback_lines,
        orientation=orientation,
        debug=debug,
        debug_dir=debug_dir,
        debug_stem=debug_stem,
        reocr=reocr,
        reocr_max_cells=reocr_max_cells,
        tsr_aggressive=tsr_aggressive,
    )
    return out["md"] or out["html"]
