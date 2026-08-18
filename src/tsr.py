"""TableStructureRec 结构识别：有线/无线分流，只取单元格框 + 逻辑拓扑。

不使用库内 RapidOCR / pred_html；文本仍由云端 OCR + IoA 填格。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .models import (
    _correct_columns_by_physical_x,
    _correct_rows_by_physical_y,
    _to_quad,
    _top_left_from_quad,
)

logger = logging.getLogger(__name__)

# 文档建议：wired v2 对约 1500px 内效果最好；超过此长边等比缩小再推理
_TSR_MAX_LONG_SIDE = 1500

_table_cls = None
_wired_engine = None
_lineless_engine = None


def load_tsr_models() -> Tuple[Any, Any, Any]:
    """懒加载 table_cls / wired / lineless 单例。"""
    global _table_cls, _wired_engine, _lineless_engine
    if _table_cls is not None and _wired_engine is not None and _lineless_engine is not None:
        return _table_cls, _wired_engine, _lineless_engine

    from table_cls import TableCls
    from wired_table_rec.main import WiredTableInput, WiredTableRecognition
    from lineless_table_rec.main import LinelessTableInput, LinelessTableRecognition

    logger.info("加载 TableStructureRec（table_cls + wired/unet + lineless/lore）…")
    _table_cls = TableCls()
    _wired_engine = WiredTableRecognition(WiredTableInput(model_type="unet"))
    _lineless_engine = LinelessTableRecognition(LinelessTableInput())
    return _table_cls, _wired_engine, _lineless_engine


def _maybe_downscale(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """长边超过阈值则缩小；返回 (推理图, 相对原图的缩放比)。"""
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= _TSR_MAX_LONG_SIDE:
        return image, 1.0
    scale = _TSR_MAX_LONG_SIDE / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.info(
        "TSR 输入缩放: %dx%d -> %dx%d (scale=%.4f)",
        w,
        h,
        new_w,
        new_h,
        scale,
    )
    return resized, scale


def _bbox_to_quad(box: Any) -> Optional[np.ndarray]:
    """兼容 [xmin,ymin,xmax,ymax] / 8 点 / (4,2) → (4,2) quad。"""
    arr = np.asarray(box, dtype=np.float64)
    if arr.size == 4:
        return _to_quad(arr.reshape(4))
    if arr.size == 8:
        return _to_quad(arr.reshape(4, 2))
    if arr.ndim == 2 and arr.shape[0] >= 4 and arr.shape[1] == 2:
        return _to_quad(arr[:4])
    return _to_quad(arr)


def _parse_logic(row: Any) -> Optional[Tuple[int, int, int, int]]:
    arr = np.asarray(row, dtype=np.float64).reshape(-1)
    if arr.size < 4:
        return None
    vals = [int(round(float(x))) for x in arr[:4].tolist()]
    row_start, row_end, col_start, col_end = vals
    if row_end < row_start:
        row_start, row_end = row_end, row_start
    if col_end < col_start:
        col_start, col_end = col_end, col_start
    return row_start, row_end, col_start, col_end


def _result_to_cells(
    cell_bboxes: Any,
    logic_points: Any,
    *,
    scale: float,
) -> List[Dict[str, Any]]:
    """将 TSR 输出转为与 LORE 一致的 cell_dict 列表，并还原到原图坐标。"""
    if cell_bboxes is None or logic_points is None:
        return []

    boxes = list(cell_bboxes) if not isinstance(cell_bboxes, np.ndarray) else [
        cell_bboxes[i] for i in range(len(cell_bboxes))
    ]
    logics = list(logic_points) if not isinstance(logic_points, np.ndarray) else [
        logic_points[i] for i in range(len(logic_points))
    ]
    if len(boxes) != len(logics):
        raise RuntimeError(
            f"TSR cell_bboxes 数量（{len(boxes)}）与 logic_points（{len(logics)}）不一致"
        )
    if not boxes:
        return []

    inv = 1.0 / scale if scale > 1e-9 else 1.0
    parsed: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
    for box, logic in zip(boxes, logics):
        quad = _bbox_to_quad(box)
        logi = _parse_logic(logic)
        if quad is None or logi is None:
            continue
        if inv != 1.0:
            quad = quad * inv
        parsed.append((quad, logi))

    if not parsed:
        return []

    min_row = min(r for _, (r, _, _, _) in parsed)
    min_col = min(c for _, (_, _, c, _) in parsed)

    cells: List[Dict[str, Any]] = []
    for quad, (row_start, row_end, col_start, col_end) in parsed:
        ns, ne = row_start - min_row, row_end - min_row
        cs, ce = col_start - min_col, col_end - min_col
        x_key, y_key = _top_left_from_quad(quad)
        cells.append(
            {
                "polygon": quad,
                "x_key": x_key,
                "y_key": y_key,
                "row_start": ns,
                "row_end": ne,
                "col_start": cs,
                "col_end": ce,
                "row_span": ne - ns + 1,
                "col_span": ce - cs + 1,
                "texts": [],
                "text": "",
            }
        )

    cells = _correct_columns_by_physical_x(cells, cluster_thresh=15.0)
    return _correct_rows_by_physical_y(cells)


def _run_engine(engine: Any, image: np.ndarray) -> Any:
    """调用 TSR 引擎；优先 need_ocr=False，兼容空 ocr_result。"""
    try:
        return engine(image, need_ocr=False)
    except TypeError:
        pass
    try:
        return engine(image, ocr_result=[], need_ocr=False)
    except Exception:
        return engine(image, ocr_result=[])


def _classify_table(table_cls: Any, image: np.ndarray) -> str:
    """返回 'wired' 或 'lineless'。"""
    cls_out, _elapse = table_cls(image)
    label = str(cls_out).strip().lower()
    if label in {"wired", "wire", "有线"}:
        return "wired"
    # wireless / lineless / 其它 → 无线
    return "lineless"


def _predict_kind(
    engine: Any,
    infer_img: np.ndarray,
    scale: float,
    *,
    kind_label: str,
) -> List[Dict[str, Any]]:
    result = _run_engine(engine, infer_img)
    cell_bboxes = getattr(result, "cell_bboxes", None)
    logic_points = getattr(result, "logic_points", None)
    elapse = getattr(result, "elapse", None)
    n = 0 if cell_bboxes is None else len(cell_bboxes)
    if elapse is not None:
        logger.info("TSR %s 耗时 %.3fs，cells=%s", kind_label, float(elapse), n)
    else:
        logger.info("TSR %s cells=%s", kind_label, n)
    return _result_to_cells(cell_bboxes, logic_points, scale=scale)


def _pick_finer_grid(
    primary: List[Dict[str, Any]],
    alt: List[Dict[str, Any]],
    text_boxes: Optional[List[Dict[str, Any]]],
    *,
    primary_label: str,
    alt_label: str,
    hybrid: bool,
) -> List[Dict[str, Any]]:
    """按结构质量（列/行/格数相对 OCR）选型，避免大格假覆盖或过切毁掉后处理。"""
    from .tsr_refine import (
        cell_grid_stats,
        coverage_score,
        looks_oversegmented,
        looks_undersegmented,
        structure_quality_score,
    )

    boxes = text_boxes or []
    p_cols, p_rows, p_n = cell_grid_stats(primary)
    a_cols, a_rows, a_n = cell_grid_stats(alt)
    score_p = coverage_score(primary, boxes)
    score_a = coverage_score(alt, boxes)
    q_p = structure_quality_score(primary, boxes)
    q_a = structure_quality_score(alt, boxes)
    alt_under = bool(boxes) and looks_undersegmented(alt, boxes)
    pri_under = bool(boxes) and looks_undersegmented(primary, boxes)
    alt_over = bool(boxes) and looks_oversegmented(alt, boxes)
    pri_over = bool(boxes) and looks_oversegmented(primary, boxes)

    logger.info(
        "TSR 对照 %s vs %s: cols=%d/%d rows=%d/%d cells=%d/%d cov=%.3f/%.3f "
        "qual=%.3f/%.3f hybrid=%s under=%s/%s over=%s/%s",
        primary_label,
        alt_label,
        p_cols,
        a_cols,
        p_rows,
        a_rows,
        p_n,
        a_n,
        score_p,
        score_a,
        q_p,
        q_a,
        hybrid,
        pri_under,
        alt_under,
        pri_over,
        alt_over,
    )

    # 过切候选：除非覆盖率塌了，否则不换过去
    if alt_over and not pri_over and score_p >= 0.7:
        logger.info(
            "TSR 保留 %s（拒绝过切 %s: cols=%d rows=%d cells=%d）",
            primary_label,
            alt_label,
            a_cols,
            a_rows,
            a_n,
        )
        return primary

    # 主路径过切、对照不过切 → 换到对照（即使列更少）
    if pri_over and not alt_over and score_a >= score_p - 0.08:
        logger.info(
            "TSR 选用 %s（主路径过切，对照更稳: qual %.3f→%.3f）",
            alt_label,
            q_p,
            q_a,
        )
        return alt

    # 拒绝明显更粗的网格（大格假高覆盖是挤格主因）
    if a_cols + 2 <= p_cols and not pri_under and not pri_over:
        logger.info(
            "TSR 保留 %s（拒绝更粗 %s: cols %d→%d）",
            primary_label,
            alt_label,
            p_cols,
            a_cols,
        )
        return primary
    if alt_under and not pri_under and a_n < p_n * 0.7 and not pri_over:
        logger.info(
            "TSR 保留 %s（拒绝欠切 %s: cells %d→%d）",
            primary_label,
            alt_label,
            p_n,
            a_n,
        )
        return primary

    # 结构质量分明显更高
    if q_a > q_p + 0.04 and score_a >= score_p - 0.08:
        logger.info(
            "TSR 选用 %s（结构质量更优: %.3f→%.3f）",
            alt_label,
            q_p,
            q_a,
        )
        return alt

    # 覆盖率明显更优，且不是拿欠切换细切 / 过切
    if score_a > score_p + 0.05 and not alt_under and not alt_over:
        logger.info("TSR 选用 %s（覆盖率更优）", alt_label)
        return alt

    # 列更细且未过切：允许覆盖率略低，但结构质量不能明显更差
    cov_slack = 0.10 if (hybrid or pri_under) else 0.05
    if (
        a_cols >= p_cols + 2
        and not alt_over
        and score_a >= score_p - cov_slack
        and q_a >= q_p - 0.04
    ):
        logger.info(
            "TSR 选用 %s（列更细: %d→%d, cov %.3f→%.3f）",
            alt_label,
            p_cols,
            a_cols,
            score_p,
            score_a,
        )
        return alt
    if (
        (hybrid or pri_under)
        and a_cols > p_cols
        and not alt_over
        and score_a >= score_p - cov_slack
        and q_a >= q_p - 0.04
    ):
        logger.info(
            "TSR 选用 %s（hybrid/欠切 列粒度: %d→%d）",
            alt_label,
            p_cols,
            a_cols,
        )
        return alt
    return primary


def predict_cells_tsr(
    image: np.ndarray,
    *,
    table_cls=None,
    wired_engine=None,
    lineless_engine=None,
    force_kind: Optional[str] = None,
    text_boxes: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    用 TableStructureRec 预测单元格物理框 + 逻辑拓扑。

    Args:
        force_kind: None 时用 table_cls；也可强制 "wired" / "lineless"
        text_boxes: 若提供，则按覆盖率/列粒度在 wired↔lineless 间纠偏。
            混合表（竖多横少）或欠切时会对称探测，避免锁死有线大格。
    """
    from .tsr_refine import (
        coverage_score,
        is_hybrid_line_table,
        line_density_axes,
        looks_fully_wired,
        looks_undersegmented,
    )

    cls_eng, wired, lineless = load_tsr_models()
    table_cls = table_cls or cls_eng
    wired_engine = wired_engine or wired
    lineless_engine = lineless_engine or lineless

    infer_img, scale = _maybe_downscale(image)
    kind = (force_kind or _classify_table(table_cls, infer_img)).strip().lower()
    if kind not in {"wired", "lineless"}:
        kind = "lineless"
    h_dens, v_dens = line_density_axes(image)
    dens = h_dens + v_dens
    hybrid = is_hybrid_line_table(h_dens, v_dens)
    fully_wired = looks_fully_wired(h_dens, v_dens)

    if force_kind:
        engine = wired_engine if kind == "wired" else lineless_engine
        logger.info(
            "TSR 路径: %s (forced), h=%.4f v=%.4f dens=%.4f hybrid=%s",
            kind,
            h_dens,
            v_dens,
            dens,
            hybrid,
        )
        return _predict_kind(engine, infer_img, scale, kind_label=kind)

    primary_kind = kind
    # 仅当横竖都够密且非 hybrid 时，才额外探测 wired；最终仍按列粒度选型
    prefer_wired_probe = primary_kind != "wired" and fully_wired and not hybrid

    if primary_kind == "wired":
        logger.info(
            "TSR 路径: wired (unet) primary, h=%.4f v=%.4f dens=%.4f hybrid=%s",
            h_dens,
            v_dens,
            dens,
            hybrid,
        )
        cells = _predict_kind(wired_engine, infer_img, scale, kind_label="wired")
        boxes = text_boxes or []
        cov = coverage_score(cells, boxes) if boxes else 1.0
        underseg = bool(boxes) and looks_undersegmented(cells, boxes)
        should_probe = bool(boxes) and (hybrid or underseg or cov < 0.45)
        if should_probe:
            alt = _predict_kind(
                lineless_engine, infer_img, scale, kind_label="lineless"
            )
            chosen = _pick_finer_grid(
                cells,
                alt,
                text_boxes,
                primary_label="wired",
                alt_label="lineless",
                hybrid=hybrid or underseg,
            )
            logger.info(
                "TSR 最终: %s (primary=wired hybrid=%s underseg=%s)",
                "lineless" if chosen is alt else "wired",
                hybrid,
                underseg,
            )
            return chosen
        logger.info(
            "TSR 最终: wired (cells=%d cov=%.3f, 未探测 lineless)",
            len(cells),
            cov,
        )
        return cells

    logger.info(
        "TSR 路径: lineless (lore) primary, h=%.4f v=%.4f dens=%.4f hybrid=%s fully_wired=%s",
        h_dens,
        v_dens,
        dens,
        hybrid,
        fully_wired,
    )
    cells = _predict_kind(lineless_engine, infer_img, scale, kind_label="lineless")
    boxes = text_boxes or []
    cov = coverage_score(cells, boxes) if boxes else 1.0
    underseg = bool(boxes) and looks_undersegmented(cells, boxes)
    if prefer_wired_probe or (boxes and (cov < 0.55 or underseg)):
        alt = _predict_kind(wired_engine, infer_img, scale, kind_label="wired")
        # 一律走列粒度选型；禁止仅凭覆盖率把细切换成欠切大格
        chosen = _pick_finer_grid(
            cells,
            alt,
            text_boxes,
            primary_label="lineless",
            alt_label="wired",
            hybrid=hybrid or underseg,
        )
        logger.info(
            "TSR 最终: %s (primary=lineless hybrid=%s underseg=%s)",
            "wired" if chosen is alt else "lineless",
            hybrid,
            underseg,
        )
        return chosen
    logger.info("TSR 最终: lineless (cells=%d cov=%.3f)", len(cells), cov)
    return cells


def cells_to_debug_table(cells: List[Dict[str, Any]]) -> Any:
    """将 cell 列表包装为 DetectedTable，供 debug 叠加图使用。"""
    from .lines import DetectedTable

    if not cells:
        return DetectedTable(
            cells=[],
            bbox=(0, 0, 1, 1),
            row_seps=[],
            col_seps=[],
            confidence=1.0,
        )

    xs: List[float] = []
    ys: List[float] = []
    for cell in cells:
        poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
        xs.extend(poly[:, 0].tolist())
        ys.extend(poly[:, 1].tolist())
    x1, y1 = int(np.floor(min(xs))), int(np.floor(min(ys)))
    x2, y2 = int(np.ceil(max(xs))), int(np.ceil(max(ys)))
    return DetectedTable(
        cells=list(cells),
        bbox=(x1, y1, x2, y2),
        row_seps=[],
        col_seps=[],
        confidence=1.0,
    )


def _cells_to_draw_list(
    cells: Optional[List[Dict[str, Any]]] = None,
    tables: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    if cells:
        return list(cells)
    out: List[Dict[str, Any]] = []
    for table in tables or []:
        out.extend(getattr(table, "cells", None) or [])
    return out


def render_table_vis(
    image: np.ndarray,
    cells: Optional[List[Dict[str, Any]]] = None,
    tables: Optional[List[Any]] = None,
    *,
    colorful: bool = True,
) -> np.ndarray:
    """TableStructureRec VisTable / vis_table：在原图上描单元格框线。

    colorful=True 时每个单元格用随机颜色描边（库内 vis_table）；
    False 时统一蓝色（VisTable.draw_polylines）。
    """
    import random

    draw_cells = _cells_to_draw_list(cells, tables)
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    for i, cell in enumerate(draw_cells):
        poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
        poly_i = np.round(poly).astype(np.int32)
        if colorful:
            rng = random.Random(i * 9973 + 17)
            color = (rng.randint(32, 255), rng.randint(32, 255), rng.randint(32, 255))
        else:
            color = (255, 0, 0)  # BGR 蓝，对齐 VisTable
        cv2.polylines(canvas, [poly_i], True, color, 2)
        if colorful:
            cv2.putText(
                canvas,
                str(i),
                (int(poly_i[0, 0]), int(poly_i[0, 1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
    return canvas


def render_table_vis_logic(
    image: np.ndarray,
    cells: Optional[List[Dict[str, Any]]] = None,
    tables: Optional[List[Any]] = None,
) -> np.ndarray:
    """对齐 TableStructureRec VisTable.plot_rec_box_with_logic_info：框 + row/col 标注。"""
    draw_cells = _cells_to_draw_list(cells, tables)
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()
    canvas = cv2.copyMakeBorder(
        canvas, 0, 0, 0, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    for cell in draw_cells:
        poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
        x0 = int(round(float(poly[:, 0].min())))
        y0 = int(round(float(poly[:, 1].min())))
        x1 = int(round(float(poly[:, 0].max())))
        y1 = int(round(float(poly[:, 1].max())))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 255), 1)
        rs = cell.get("row_start", 0)
        re = cell.get("row_end", rs)
        cs = cell.get("col_start", 0)
        ce = cell.get("col_end", cs)
        cv2.putText(
            canvas,
            f"row: {rs}-{re}",
            (x0 + 3, y0 + 12),
            cv2.FONT_HERSHEY_PLAIN,
            0.9,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"col: {cs}-{ce}",
            (x0 + 3, y0 + 24),
            cv2.FONT_HERSHEY_PLAIN,
            0.9,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas
