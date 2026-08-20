"""行列转置检测与纠正（P33 类：实施例横排为列头）。"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lines import binarize_otsu, detect_tables
from .refine import refine_table
from .tsr_refine import coverage_score

logger = logging.getLogger(__name__)

_EXAMPLE_LABEL_RE = re.compile(r"实[施試]例|比[较較]例|参考例")
_COMPOSITION_HEADER_RE = re.compile(r"清漆|树脂|感光剂|溶剂|显影|判定|组成")
_CAPTION_CELL_RE = re.compile(r"^\[?\s*表\s*[\d\-ー－]+\s*\]?$")
_MIN_ROW0_EXAMPLES = 8
_COL_ROW_RATIO = 1.5


def _tb_center(tb: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
    if poly.size < 4:
        return None
    return float(poly[:, 0].mean()), float(poly[:, 1].mean())


def _cell_bbox(cell: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    poly = np.asarray(cell.get("polygon"), dtype=np.float64).reshape(-1, 2)
    if poly.size < 4:
        return None
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )


def _cell_containing_point(
    cells: Sequence[Dict[str, Any]],
    x: float,
    y: float,
) -> Optional[Dict[str, Any]]:
    """点落入的最小面积单元格（用于把 OCR 框映射到逻辑格）。"""
    best = None
    best_area = float("inf")
    for c in cells:
        bb = _cell_bbox(c)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            continue
        area = max(x2 - x1, 1.0) * max(y2 - y1, 1.0)
        if area < best_area:
            best_area = area
            best = c
    return best


def _example_label_count(
    cells: Sequence[Dict[str, Any]],
    *,
    row: int | None = None,
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> int:
    """统计实施例/比较例：优先 cell.text；无文本时用 OCR 框中心映射到逻辑行。"""
    n = 0
    for c in cells:
        rs, re_ = int(c["row_start"]), int(c["row_end"])
        if row is not None and not (rs <= row <= re_):
            continue
        t = str(c.get("text") or "")
        if _EXAMPLE_LABEL_RE.search(t):
            n += 1
    if n > 0 or not text_boxes:
        return n
    # IoA 前：用 OCR 文本框映射
    hit_cells: set[int] = set()
    for tb in text_boxes:
        if not _EXAMPLE_LABEL_RE.search(str(tb.get("text") or "")):
            continue
        ctr = _tb_center(tb)
        if ctr is None:
            continue
        cell = _cell_containing_point(cells, ctr[0], ctr[1])
        if cell is None:
            continue
        rs, re_ = int(cell["row_start"]), int(cell["row_end"])
        if row is not None and not (rs <= row <= re_):
            continue
        hit_cells.add(id(cell))
    return len(hit_cells)


def _composition_labels_along_rows(
    cells: Sequence[Dict[str, Any]],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> int:
    """列头词落在「行方向」（转置后本应列头却在多行）。"""
    hits = 0
    for c in cells:
        if int(c.get("col_span") or 1) > 2:
            continue
        t = str(c.get("text") or "")
        if _COMPOSITION_HEADER_RE.search(t) and int(c["row_start"]) > 0:
            hits += 1
    if hits > 0 or not text_boxes:
        return hits
    for tb in text_boxes:
        if not _COMPOSITION_HEADER_RE.search(str(tb.get("text") or "")):
            continue
        ctr = _tb_center(tb)
        if ctr is None:
            continue
        cell = _cell_containing_point(cells, ctr[0], ctr[1])
        if cell is None:
            continue
        if int(cell.get("col_span") or 1) > 2:
            continue
        if int(cell["row_start"]) > 0:
            hits += 1
    return hits


def _example_labels_in_col0(
    cells: Sequence[Dict[str, Any]],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> int:
    """首列（col_start==0）上的实施例/比较例行标签数。"""
    n = 0
    for c in cells:
        if int(c["col_start"]) != 0:
            continue
        rsp = int(c.get("row_span") or (int(c["row_end"]) - int(c["row_start"]) + 1))
        if rsp > 4 and int(c["row_start"]) == 0:
            continue
        if _EXAMPLE_LABEL_RE.search(str(c.get("text") or "")):
            n += 1
    if n > 0 or not text_boxes:
        return n
    hit_cells: set[int] = set()
    for tb in text_boxes:
        if not _EXAMPLE_LABEL_RE.search(str(tb.get("text") or "")):
            continue
        ctr = _tb_center(tb)
        if ctr is None:
            continue
        cell = _cell_containing_point(cells, ctr[0], ctr[1])
        if cell is None or int(cell["col_start"]) != 0:
            continue
        rsp = int(cell.get("row_span") or (int(cell["row_end"]) - int(cell["row_start"]) + 1))
        if rsp > 4 and int(cell["row_start"]) == 0:
            continue
        hit_cells.add(id(cell))
    return len(hit_cells)


def detect_sideways_row_labels(
    cells: Sequence[Dict[str, Any]],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> bool:
    """
    实施例/比较例应在首列纵排；若首行横排 ≥6 且首列几乎无同类标签，判定侧躺/行列颠倒。

    覆盖 P57X445 等：误转 90° 后 TSR 碎成 60+ 行，旧版 detect 因 n_cols < n_rows*1.5 漏检。
    """
    if not cells or len(cells) < 10:
        return False
    row0_examples = _example_label_count(cells, row=0, text_boxes=text_boxes)
    if row0_examples < 6:
        return False
    col0_examples = _example_labels_in_col0(cells, text_boxes=text_boxes)
    if col0_examples >= 3:
        return False
    return row0_examples >= col0_examples + 5


def detect_transposed_table(
    cells: Sequence[Dict[str, Any]],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> bool:
    """
    多信号门控：首行大量实施例/比较例、首列无同类标签、列数明显大于行数。

    注意：TSR 路径在 IoA 填字前调用本函数，必须能用 text_boxes 完成判定。
    """
    if not cells or len(cells) < 12:
        return False

    if detect_sideways_row_labels(cells, text_boxes):
        return True

    max_row = max(int(c["row_end"]) for c in cells)
    max_col = max(int(c["col_end"]) for c in cells)
    n_rows = max_row + 1
    n_cols = max_col + 1

    row0_examples = _example_label_count(cells, row=0, text_boxes=text_boxes)
    col0_examples = _example_labels_in_col0(cells, text_boxes=text_boxes)
    # 碎网格侧躺：行数膨胀但首行仍横排实施例
    if n_rows >= 30 and row0_examples >= 4 and row0_examples > col0_examples + 2:
        return True

    if n_cols < _MIN_ROW0_EXAMPLES:
        return False
    if n_cols < n_rows * _COL_ROW_RATIO:
        return False

    if row0_examples < _MIN_ROW0_EXAMPLES:
        return False

    if col0_examples >= 3:
        return False

    comp_rows = _composition_labels_along_rows(cells, text_boxes)
    if comp_rows >= 2:
        return True
    return row0_examples >= _MIN_ROW0_EXAMPLES and n_cols > n_rows * _COL_ROW_RATIO


def _swap_polygon_xy(cell: Dict[str, Any]) -> np.ndarray:
    poly = np.asarray(cell.get("polygon"), dtype=np.float64).reshape(-1, 2)
    return np.column_stack([poly[:, 1], poly[:, 0]])


def swap_table_transpose(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """交换逻辑行列索引与 polygon 的 x/y 轴。"""
    out: List[Dict[str, Any]] = []
    for c in cells:
        nc = dict(c)
        rs, re_ = int(c["row_start"]), int(c["row_end"])
        cs, ce = int(c["col_start"]), int(c["col_end"])
        nc["row_start"] = cs
        nc["row_end"] = ce
        nc["col_start"] = rs
        nc["col_end"] = re_
        nc["row_span"] = ce - cs + 1
        nc["col_span"] = re_ - rs + 1
        if c.get("polygon") is not None:
            nc["polygon"] = _swap_polygon_xy(c)
        nc.pop("text", None)
        nc.pop("texts", None)
        out.append(nc)
    return out


def strip_caption_cells(cells: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """把 [表n] 从网格格移到 free_texts。"""
    captions: List[Dict[str, Any]] = []
    for c in cells:
        t = str(c.get("text") or "").strip()
        if not _CAPTION_CELL_RE.fullmatch(t):
            continue
        captions.append({"text": t, "polygon": c.get("polygon"), "score": 1.0})
        c["text"] = ""
        c["texts"] = []
        c["_drop_render"] = True
    return cells, captions


def maybe_fix_transposed_table(
    cells: List[Dict[str, Any]],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """检测命中则交换行列；返回 (cells, was_transposed)。"""
    if not detect_transposed_table(cells, text_boxes):
        return cells, False
    logger.info("检测到行列转置，执行交换纠正")
    return swap_table_transpose(cells), True


def cells_from_lines_structure(
    image: np.ndarray,
    text_boxes: Sequence[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """单表 lines 结构（无 IoA），供转置纠正后质量仍差时回退。"""
    tables = detect_tables(image, confidence_thresh=0.0, text_boxes=text_boxes)
    if not tables:
        return None
    binary = binarize_otsu(image)
    refined = [refine_table(t, text_boxes) for t in tables]
    best = max(refined, key=lambda t: len(t.cells or []))
    return list(best.cells or [])


def maybe_lines_fallback_after_transpose(
    image: np.ndarray,
    cells: List[Dict[str, Any]],
    text_boxes: Sequence[Dict[str, Any]],
    *,
    cov_threshold: float = 0.55,
) -> List[Dict[str, Any]]:
    """转置纠正后覆盖率仍低时，尝试 lines 拓扑。"""
    cov = coverage_score(cells, text_boxes) if cells else 0.0
    if cov >= cov_threshold:
        return cells
    line_cells = cells_from_lines_structure(image, text_boxes)
    if not line_cells:
        return cells
    line_cov = coverage_score(line_cells, text_boxes)
    if line_cov > cov + 0.03:
        logger.info(
            "转置纠正后仍差(cov=%.3f)，回退 lines 结构(cov=%.3f)",
            cov,
            line_cov,
        )
        return line_cells
    return cells
