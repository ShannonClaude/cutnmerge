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
_CAPTION_CELL_RE = re.compile(r"^\[表\s*\d+\]$")
_MIN_ROW0_EXAMPLES = 8
_COL_ROW_RATIO = 1.5


def _example_label_count(cells: Sequence[Dict[str, Any]], *, row: int | None = None) -> int:
    n = 0
    for c in cells:
        rs, re_ = int(c["row_start"]), int(c["row_end"])
        if row is not None and not (rs <= row <= re_):
            continue
        t = str(c.get("text") or "")
        if _EXAMPLE_LABEL_RE.search(t):
            n += 1
    return n


def _composition_labels_along_rows(cells: Sequence[Dict[str, Any]]) -> int:
    """列头词落在「行方向」（转置后本应列头却在多行）。"""
    hits = 0
    for c in cells:
        if int(c.get("col_span") or 1) > 2:
            continue
        t = str(c.get("text") or "")
        if _COMPOSITION_HEADER_RE.search(t) and int(c["row_start"]) > 0:
            hits += 1
    return hits


def detect_transposed_table(
    cells: Sequence[Dict[str, Any]],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> bool:
    """
    多信号门控：首行大量实施例/比较例、首列无同类标签、列数明显大于行数。
    """
    if not cells or len(cells) < 12:
        return False
    max_row = max(int(c["row_end"]) for c in cells)
    max_col = max(int(c["col_end"]) for c in cells)
    n_rows = max_row + 1
    n_cols = max_col + 1
    if n_cols < _MIN_ROW0_EXAMPLES:
        return False
    if n_cols < n_rows * _COL_ROW_RATIO:
        return False

    row0_examples = _example_label_count(cells, row=0)
    if row0_examples < _MIN_ROW0_EXAMPLES:
        return False

    col0_examples = 0
    for c in cells:
        if int(c["col_start"]) != 0:
            continue
        rsp = int(c.get("row_span") or (int(c["row_end"]) - int(c["row_start"]) + 1))
        if rsp > 4 and int(c["row_start"]) == 0:
            continue
        if _EXAMPLE_LABEL_RE.search(str(c.get("text") or "")):
            col0_examples += 1
    if col0_examples >= 3:
        return False

    comp_rows = _composition_labels_along_rows(cells)
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
