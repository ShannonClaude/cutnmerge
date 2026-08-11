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
        text_boxes: 若提供，则在 cls=lineless 且线密度高、或覆盖率差时
            额外跑 wired，按 OCR 覆盖率择优。
    """
    from .tsr_refine import coverage_score, line_density_score

    cls_eng, wired, lineless = load_tsr_models()
    table_cls = table_cls or cls_eng
    wired_engine = wired_engine or wired
    lineless_engine = lineless_engine or lineless

    infer_img, scale = _maybe_downscale(image)
    kind = (force_kind or _classify_table(table_cls, infer_img)).strip().lower()
    dens = line_density_score(image)

    if force_kind:
        engine = wired_engine if kind == "wired" else lineless_engine
        logger.info("TSR 路径: %s (forced)", kind)
        return _predict_kind(engine, infer_img, scale, kind_label=kind)

    primary_kind = kind
    # 分类为无线但线密度偏高 → 更可能是有线表被误判
    prefer_wired_probe = primary_kind != "wired" and dens >= 0.012
    if primary_kind == "wired":
        logger.info("TSR 路径: wired (unet), dens=%.4f", dens)
        cells = _predict_kind(wired_engine, infer_img, scale, kind_label="wired")
        if text_boxes and coverage_score(cells, text_boxes) < 0.45:
            alt = _predict_kind(lineless_engine, infer_img, scale, kind_label="lineless")
            if coverage_score(alt, text_boxes) > coverage_score(cells, text_boxes) + 0.05:
                logger.info("TSR 选用 lineless（覆盖率更优）")
                return alt
        return cells

    logger.info("TSR 路径: lineless (lore), dens=%.4f", dens)
    cells = _predict_kind(lineless_engine, infer_img, scale, kind_label="lineless")
    if prefer_wired_probe or (text_boxes and coverage_score(cells, text_boxes) < 0.55):
        alt = _predict_kind(wired_engine, infer_img, scale, kind_label="wired")
        score_p = coverage_score(cells, text_boxes or [])
        score_a = coverage_score(alt, text_boxes or [])
        # 线密度高时偏 wired；否则看覆盖率
        if prefer_wired_probe and (score_a >= score_p - 0.02 or len(alt) > len(cells) * 0.6):
            if score_a >= score_p - 0.05:
                logger.info(
                    "TSR 选用 wired（线密度+覆盖: lineless=%.3f wired=%.3f）",
                    score_p,
                    score_a,
                )
                return alt
        elif score_a > score_p + 0.05:
            logger.info("TSR 选用 wired（覆盖率更优）")
            return alt
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
