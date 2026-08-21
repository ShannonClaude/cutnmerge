"""解耦表格提取主流程：结构识别与 OCR 文本分离后再融合。

默认 TableStructureRec（tsr）+ 云端 OCR + IoA；输出 HTML（保留 rowspan/colspan），
末尾经 html2md 转 Markdown，并写出两张彩色可视化图。OCR 结果可本地缓存。
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from .config import ROOT
from .models import load_lore_model, load_ocr, predict_cells, predict_texts
from ..matching.matching import (
    assign_texts_to_cells,
    _fix_dash_column_consistency,
    detect_eval_symbols_in_empty_cells,
    upgrade_o_to_double_circle,
)
from ..ocr.ocr_post import postprocess_text_boxes
from ..ocr.reocr import apply_reocr_to_cells, recover_empty_vertical_headers
from ..output.formatter import format_free_texts
from ..output.html2md import html_to_markdown
from ..output.html_formatter import build_html_output, html_output_looks_broken
from ..preprocess.orient import apply_orientation_axis, ensure_upright_axis, maybe_flip_180_by_ocr, parse_orientation_mode
from ..structure.grid_fusion import fuse_tsr_with_lines
from ..structure.lines import (
    DetectedTable,
    binarize_otsu,
    detect_tables,
    imwrite_unicode,
    render_debug_overlay,
)
from ..structure.refine import refine_table
from ..structure.tsr import (
    cells_to_debug_table,
    predict_cells_tsr,
    render_table_vis,
    render_table_vis_logic,
)
from ..structure.tsr_refine import (
    cell_grid_stats,
    coverage_score,
    dedupe_overlapping_cells,
    logic_conflict_ratio,
    looks_oversegmented,
    merge_ghost_columns,
    merge_ghost_rows,
    merge_stacked_chem_amount_cells,
    normalize_oversegmented_table_rows,
    reconstruct_header_cells,
    refine_tsr_cells,
    refine_tsr_cells_light,
    repair_monomer_parent_spans,
    lift_misplaced_header_labels,
    promote_side_header_rowspans,
    needs_monomer_header_reconstruct,
    structure_quality_score,
)

logger = logging.getLogger(__name__)
DEFAULT_DEBUG_DIR = ROOT / "data" / "debug"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "output"


def _is_degenerate_grid(
    cells: List[Dict[str, Any]],
    text_boxes: list,
) -> bool:
    """检测退化网格——仅在极端灾难性场景触发，避免误判正常复杂表格。

    触发条件（需同时满足多个强信号）：
    - 格子数极少而文本框极多（结构基本丢失）
    - 或逻辑冲突比极高（>0.25）且格子数远少于文本框数

    已有大量文字归属时不触发：兜底融合常生成无字空壳，会把整表抹掉。
    """
    if not cells:
        return len(text_boxes) > 20
    n_cells = len(cells)
    n_boxes = max(len(text_boxes), 1)
    n_text = sum(1 for c in cells if str(c.get("text") or "").strip())
    # 文字覆盖尚可 → 宁可保留当前结果，勿用空结构替换
    if n_text >= max(8, int(n_boxes * 0.25)):
        return False
    # 信号1：格子极少、文本框极多——结构几乎完全丢失
    if n_cells < 4 and n_boxes > 20:
        return True
    # 信号2：逻辑冲突比极高 + 格子远少于文本框（双重确认）
    cr = logic_conflict_ratio(cells)
    if cr > 0.25 and n_cells < n_boxes * 0.3:
        return True
    return False

# auto 模式下框线网格最低置信度（仅 --structure lines/auto 的旧路径）
_LINES_CONF_THRESH = 0.35
# 轻量路径逻辑格重叠比例过高时，自动升级到 aggressive + 线融合
_LIGHT_ESCALATE_CONFLICT_RATIO = 0.02


def _light_has_monomer_subheaders(cells: List[Dict[str, Any]]) -> bool:
    """轻量结果已有「单体」父格 + 下一行列向子表头 → 勿为微小冲突升级毁掉表头。"""
    if not cells:
        return False
    monomer_re = re.compile(r"单体\s*[\[［]")
    for parent in cells:
        if not monomer_re.search(str(parent.get("text") or "")):
            # 归属前无字：用宽格 + 下一行列数判断
            span = int(parent.get("col_span") or 1)
            if span < 3:
                continue
        else:
            span = int(parent.get("col_span") or 1)
            if span < 2:
                continue
        pcs, pce = int(parent["col_start"]), int(parent["col_end"])
        pre = int(parent["row_end"])
        kids = [
            c
            for c in cells
            if int(c["row_start"]) == pre + 1
            and int(c["col_end"]) >= pcs
            and int(c["col_start"]) <= pce
            and int(c.get("col_span") or 1) >= 1
        ]
        # 子表头行应切成 ≥2 格，且覆盖父带大半列
        if len(kids) < 2:
            continue
        covered = set()
        for k in kids:
            for col in range(int(k["col_start"]), int(k["col_end"]) + 1):
                if pcs <= col <= pce:
                    covered.add(col)
        if len(covered) >= max(2, int(0.5 * (pce - pcs + 1) + 0.5)):
            return True
    # 无字宽格：仅靠拓扑（P97 归属前）
    for parent in cells:
        span = int(
            parent.get("col_span")
            or (int(parent["col_end"]) - int(parent["col_start"]) + 1)
        )
        if span < 3:
            continue
        pcs, pce = int(parent["col_start"]), int(parent["col_end"])
        pre = int(parent["row_end"])
        kids = [
            c
            for c in cells
            if int(c["row_start"]) == pre + 1
            and int(c["col_end"]) >= pcs
            and int(c["col_start"]) <= pce
        ]
        if len(kids) >= 2:
            return True
    return False


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
        pts = np.asarray(line).reshape(-1)
        if pts.size < 4:
            continue
        x1, y1, x2, y2 = (int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3]))
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
    """生成 HTML；Markdown 在 pipeline 末尾由 html2md 统一转换。"""
    html = build_html_output(
        cells,
        free_texts,
        split_subtables=True,
        compress_empty=compress_empty_cols,
    )
    return {"html": html or "", "md": ""}


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
        outs = _render_outputs(
            [], text_boxes, compress_empty_cols=compress_empty_cols
        )
        return outs, [], text_boxes, []

    binary = binarize_otsu(image)
    tables = [refine_table(t, text_boxes) for t in tables]

    bboxes = [t.bbox for t in tables]
    html_parts: List[str] = []
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
        cells = _fix_dash_column_consistency(cells, binary=binary)
        cells = detect_eval_symbols_in_empty_cells(cells, binary)
        cells = upgrade_o_to_double_circle(cells, binary)
        cells = recover_empty_vertical_headers(
            image,
            cells,
            binary=binary,
            ocr_engine=ocr_engine,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
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

    outside = []
    for tb in remaining:
        poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
        cx, cy = float(poly[:, 0].mean()), float(poly[:, 1].mean())
        if any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in bboxes):
            continue
        outside.append(tb)

    prefix = format_free_texts(outside)
    html_body = "\n\n".join(html_parts)
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
    return {"html": html, "md": ""}, tables, outside, text_boxes


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
        outs = _render_outputs(
            [], text_boxes, compress_empty_cols=compress_empty_cols
        )
        return outs, []
    lore_binary = binarize_otsu(image)
    cells, free_texts = assign_texts_to_cells(
        cells,
        text_boxes,
        ioa_threshold=ioa_threshold,
        split_cross_cell=True,
        table_bboxes=None,
        binary=lore_binary,
    )
    cells = _fix_dash_column_consistency(cells, binary=lore_binary)
    cells = detect_eval_symbols_in_empty_cells(cells, lore_binary)
    cells = upgrade_o_to_double_circle(cells, lore_binary)
    cells = recover_empty_vertical_headers(
        image,
        cells,
        binary=lore_binary,
        ocr_engine=ocr_engine,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
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
    escalated_from_light = False
    if cells:
        if tsr_aggressive:
            cells = refine_tsr_cells(cells, text_boxes)
            line_tables = detect_tables(
                image, confidence_thresh=0.0, text_boxes=text_boxes
            )
            if line_tables:
                cells = fuse_tsr_with_lines(cells, line_tables)
        else:
            light_cells = refine_tsr_cells_light(cells)
            light_cells = merge_ghost_columns(light_cells, text_boxes)
            light_cells = merge_ghost_rows(light_cells, text_boxes)
            light_cells = dedupe_overlapping_cells(light_cells)
            light_cells = repair_monomer_parent_spans(light_cells, text_boxes)
            conflict_ratio = logic_conflict_ratio(light_cells)
            keep_light_headers = _light_has_monomer_subheaders(light_cells)
            if (
                conflict_ratio > _LIGHT_ESCALATE_CONFLICT_RATIO
                and not keep_light_headers
            ):
                logger.info(
                    "TSR-light 逻辑格重叠比例过高(%.3f)，自动升级 aggressive 融合",
                    conflict_ratio,
                )
                agg_cells = refine_tsr_cells(cells, text_boxes)
                probe_lines = detect_tables(
                    image, confidence_thresh=0.0, text_boxes=text_boxes
                )
                fused = (
                    fuse_tsr_with_lines(agg_cells, probe_lines)
                    if probe_lines
                    else agg_cells
                )
                fused_ratio = logic_conflict_ratio(fused)
                if fused_ratio < conflict_ratio:
                    cells = fused
                    line_tables = probe_lines
                    escalated_from_light = True
                    logger.info(
                        "TSR-light 已升级融合: conflict %.3f → %.3f",
                        conflict_ratio,
                        fused_ratio,
                    )
                else:
                    cells = light_cells
            else:
                cells = light_cells
                if keep_light_headers and conflict_ratio > _LIGHT_ESCALATE_CONFLICT_RATIO:
                    logger.info(
                        "TSR-first 轻量路径：保留单体子表头拓扑"
                        "（conflict=%.3f，跳过升级）",
                        conflict_ratio,
                    )
                else:
                    logger.info(
                        "TSR-first 轻量路径：跳过激进 refine / 线融合；已尝试单体父格对齐"
                    )

    cov = coverage_score(cells, text_boxes) if cells else 0.0
    # 覆盖率偏低或格子过少时回退（竖排表头等场景常出现「有格但几乎填不进」）
    n_boxes = max(len(text_boxes), 1)
    too_few_cells = bool(cells) and len(cells) < max(8, n_boxes // 8)
    # TSR 把多行单元格切成碎逻辑行时 cov 仍可能很高，须单独识别过切
    overseg = bool(cells) and looks_oversegmented(cells, text_boxes)
    need_lines_fallback = bool(
        (not cells or cov < 0.55 or too_few_cells) and fallback_lines
    ) or overseg
    if need_lines_fallback:
        n_cols, n_rows, n_cells = (
            cell_grid_stats(cells) if cells else (0, 0, 0)
        )
        if overseg:
            logger.info(
                "TSR 过切(cols=%d rows=%d cells=%d boxes=%d cov=%.3f)，尝试框线回退",
                n_cols,
                n_rows,
                n_cells,
                len(text_boxes),
                cov,
            )
        else:
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
        line_cells: List[Dict[str, Any]] = []
        for t in tables or []:
            line_cells.extend(list(getattr(t, "cells", None) or []))
        if not overseg:
            return outs, tables
        # 过切场景：框线不过切，或行数明显收敛时采用框线
        if line_cells:
            lc, lr, ln = cell_grid_stats(line_cells)
            line_ok = not looks_oversegmented(line_cells, text_boxes)
            rows_improved = lr < n_rows * 0.65 and lr >= 4
            qual_ok = (
                structure_quality_score(line_cells, text_boxes)
                >= structure_quality_score(cells, text_boxes) - 0.05
            )
            if (line_ok or rows_improved) and qual_ok:
                logger.info(
                    "TSR 过切已回退框线: cells %d→%d rows %d→%d cols %d→%d",
                    n_cells,
                    ln,
                    n_rows,
                    lr,
                    n_cols,
                    lc,
                )
                return outs, tables
        logger.info("框线回退未改善过切，保留 TSR 结果")

    if not cells:
        outs = _render_outputs(
            [], text_boxes, compress_empty_cols=compress_empty_cols
        )
        return outs, []

    binary = binarize_otsu(image)
    # 轻量路径也开跨列切分：只切 OCR 文本到已有原子列，不插线/不补列。
    # col_seps：light/aggressive 均可用于按列界切开粘连 OCR；v_separators 仅激进路径。
    # _explode_colspan 只在目标格 colspan>1 时触发，比率类 rowspan 表头不受影响。
    col_seps = None
    v_separators = None
    split_cross = True

    if tsr_aggressive:
        from ..structure.grid_evidence import apply_grid_evidence_merge

        cells = apply_grid_evidence_merge(
            cells,
            text_boxes,
            line_tables=line_tables,
        )
        cells = reconstruct_header_cells(cells, text_boxes)
        from ..structure.hline_repair import repair_colspans_by_vline_gaps

        cells = repair_colspans_by_vline_gaps(cells, binary, text_boxes)
        from ..structure.tsr_refine import _derive_seps as _derive_seps_local

        _row_seps, col_seps_list = _derive_seps_local(cells)
        if len(col_seps_list) >= 3:
            col_seps = col_seps_list
        if line_tables:
            best = max(
                line_tables,
                key=lambda t: float(getattr(t, "confidence", 0.0) or 0.0),
            )
            v_separators = getattr(best, "v_separators", None)
    elif escalated_from_light:
        # 融合后的网格已无逻辑重叠；不再重建表头（会把多级表头压成过少列）。
        from ..structure.tsr_refine import _derive_seps as _derive_seps_local

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
        from ..structure.hline_repair import (
            repair_colspans_by_vline_gaps,
            repair_rowspans_by_hline_gaps,
        )
        from ..structure.tsr_refine import _derive_seps as _derive_seps_local

        cells = repair_rowspans_by_hline_gaps(cells, binary, text_boxes)
        cells = repair_colspans_by_vline_gaps(cells, binary, text_boxes)
        _row_seps, col_seps_list = _derive_seps_local(cells)
        if len(col_seps_list) >= 3:
            col_seps = col_seps_list

    from ..structure.transpose_fix import (
        maybe_fix_transposed_table,
        maybe_lines_fallback_after_transpose,
        strip_caption_cells,
    )
    from ..structure.row_header import (
        clip_narrow_label_colspans,
        clip_row_header_child_overlaps,
        extend_section_rowspan_over_metric_rows,
        peel_row_header_text,
    )

    transposed = False
    cells, transposed = maybe_fix_transposed_table(cells, text_boxes)
    if transposed:
        cells = maybe_lines_fallback_after_transpose(
            image, cells, text_boxes
        )

    cells = clip_row_header_child_overlaps(cells)
    cells = clip_narrow_label_colspans(cells)

    # 轻量/轻量升级路径：归属前拆「单体」父子表头（激进路径已在上文重建）
    if (not tsr_aggressive) and needs_monomer_header_reconstruct(cells, text_boxes):
        before_n = len(cells)
        cells = reconstruct_header_cells(cells, text_boxes)
        logger.info(
            "轻量路径表头重建(归属前%s): cells %d→%d",
            "+escalated" if escalated_from_light else "",
            before_n,
            len(cells),
        )

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
    cells = merge_stacked_chem_amount_cells(cells)
    cells = _fix_dash_column_consistency(cells, binary=binary)
    cells = detect_eval_symbols_in_empty_cells(cells, binary)
    cells = upgrade_o_to_double_circle(cells, binary)
    cells = peel_row_header_text(cells, text_boxes)
    cells = extend_section_rowspan_over_metric_rows(cells)
    cells, caption_texts = strip_caption_cells(cells)
    if caption_texts:
        free_texts.extend(caption_texts)
    # 表题剥离后再压行，避免 [表1-2] 落在子表头行触发分段
    cells = normalize_oversegmented_table_rows(cells)
    # 压行之后再上提误落表头 / 侧栏 rowspan，避免 normalize 把表头拽回合成例行
    cells = lift_misplaced_header_labels(cells)
    cells = promote_side_header_rowspans(cells)
    cells = recover_empty_vertical_headers(
        image,
        cells,
        binary=binary,
        ocr_engine=ocr_engine,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
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
    # 退化网格检测：结构严重损坏时尝试升级或 LORE 兜底
    if _is_degenerate_grid(cells, text_boxes):
        logger.warning("检测到退化网格(cells=%d boxes=%d)，尝试兜底", len(cells), len(text_boxes))
        if not tsr_aggressive:
            prev_ne = sum(1 for c in cells if str(c.get("text") or "").strip())
            try:
                agg_cells = refine_tsr_cells(cells, text_boxes)
                probe_lines = detect_tables(image, confidence_thresh=0.0, text_boxes=text_boxes)
                fused = fuse_tsr_with_lines(agg_cells, probe_lines) if probe_lines else agg_cells
                # 融合后结构无字，必须重新归属；否则会输出空壳表
                fused, _fused_free = assign_texts_to_cells(
                    fused,
                    text_boxes,
                    ioa_threshold=ioa_threshold,
                    split_cross_cell=split_cross,
                    table_bboxes=None,
                    binary=binary,
                    col_seps=col_seps,
                    v_separators=v_separators,
                )
                fused_ne = sum(1 for c in fused if str(c.get("text") or "").strip())
                if fused_ne > prev_ne and not _is_degenerate_grid(fused, text_boxes):
                    cells = fused
                    logger.info(
                        "退化网格升级 aggressive+fused 成功 (ne %d→%d)",
                        prev_ne,
                        fused_ne,
                    )
                else:
                    raise ValueError(
                        f"aggressive 未改善文字覆盖 (ne {prev_ne}→{fused_ne})"
                    )
            except Exception:
                logger.info("aggressive 路径未改善，尝试 LORE 兜底")
                try:
                    lore_outs, lore_tables = _extract_via_lore(
                        image, text_boxes,
                        ioa_threshold=ioa_threshold,
                        compress_empty_cols=compress_empty_cols,
                        reocr=reocr, reocr_max_cells=reocr_max_cells,
                        ocr_engine=ocr_engine, use_cache=use_cache,
                        refresh_cache=refresh_cache,
                    )
                    if lore_outs.get("html"):
                        return lore_outs, lore_tables
                except Exception:
                    logger.warning("LORE 兜底失败，使用当前退化结果")

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
    _allow_orient_retry: bool = True,
) -> Dict[str, Any]:
    """
    复杂表格解耦提取 Pipeline。

    Returns:
        {"html": str, "md": str, "structure": str, "orientation": int,
         "vis_paths": list[str]}
        md 由最终 HTML 经 html2md 转换；vis_paths 为两张彩色可视化图路径。
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
        artifact_stem=stem,
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
                from ..ocr.ocr_cache import save_ocr_cache

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

    if not outputs.get("html"):
        outputs = _render_outputs(
            [], text_boxes, compress_empty_cols=compress_empty_cols
        )

    # Pipeline 末尾：HTML → Markdown（html2md）
    html = outputs.get("html") or ""
    md = html_to_markdown(html) if html.strip() else ""

    result = {
        "html": html,
        "md": md,
        "structure": structure,
        "orientation": int(orient_angle),
        "vis_paths": vis_paths,
    }

    # 误转 90°/270° 时 HTML 会大量空行 + 异常 rowspan；用 0° 重试一次（不递归）。
    parsed_orient = parse_orientation_mode(orientation)
    can_retry_upright = (
        parsed_orient in {90, 270}
        or (parsed_orient == "auto" and int(orient_angle) % 180 == 90)
    )
    if can_retry_upright and _allow_orient_retry and html_output_looks_broken(html):
        logger.warning(
            "输出疑似侧躺损坏(orient=%s)，改用 orientation=0 重试",
            orient_angle,
        )
        retry = extract_table_output(
            image_path,
            ioa_threshold=ioa_threshold,
            deskew=deskew,
            max_skew_angle=max_skew_angle,
            lore_pipe=lore_pipe,
            ocr_engine=ocr,
            structure=structure,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            compress_empty_cols=compress_empty_cols,
            fallback_lines=fallback_lines,
            orientation=0,
            debug=debug,
            debug_dir=debug_dir,
            debug_stem=debug_stem,
            reocr=reocr,
            reocr_max_cells=reocr_max_cells,
            tsr_kind=tsr_kind,
            tsr_aggressive=tsr_aggressive,
            save_vis=save_vis,
            vis_dir=vis_dir,
            _allow_orient_retry=False,
        )
        if not html_output_looks_broken(retry.get("html") or ""):
            return retry
        logger.warning("orientation=0 重试仍异常，保留原结果")

    return result


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
