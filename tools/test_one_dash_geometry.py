# -*- coding: utf-8 -*-
"""1 ↔ '-' 墨迹几何纠错与空格 dash 补全。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.matching.matching import detect_dashes_in_empty_cells
from src.ocr.ocr_post import (
    _maybe_dash_to_one,
    _maybe_tiny_one_to_dash,
    postprocess_text_boxes,
    roi_looks_like_short_dash,
)


def _tb(text: str, x1, y1, x2, y2, score=0.95):
    return {
        "text": text,
        "score": score,
        "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
    }


def _paint_h_dash(binary: np.ndarray, x1, y1, x2, y2, thickness=2):
    cy = (y1 + y2) // 2
    binary[cy : cy + thickness, x1:x2] = 255


def _paint_v_one(binary: np.ndarray, x1, y1, x2, y2, thickness=2):
    cx = (x1 + x2) // 2
    binary[y1:y2, cx : cx + thickness] = 255


def test_one_with_horizontal_band_becomes_dash():
    binary = np.zeros((40, 40), dtype=np.uint8)
    _paint_h_dash(binary, 8, 14, 28, 26, thickness=2)
    tb = _tb("1", 8, 14, 28, 26, score=0.92)
    out = _maybe_tiny_one_to_dash(tb, "1", median_box_h=23.0, binary=binary)
    assert out == "-", out


def test_real_tall_one_keeps_one():
    binary = np.zeros((40, 40), dtype=np.uint8)
    _paint_v_one(binary, 16, 5, 24, 35, thickness=2)
    tb = _tb("1", 16, 5, 24, 35, score=0.95)
    out = _maybe_tiny_one_to_dash(tb, "1", median_box_h=23.0, binary=binary)
    assert out == "1", out


def test_dash_with_vertical_band_becomes_one():
    binary = np.zeros((40, 40), dtype=np.uint8)
    _paint_v_one(binary, 16, 8, 24, 32, thickness=2)
    tb = _tb("-", 16, 8, 24, 32, score=0.90)
    out = _maybe_dash_to_one(tb, "-", median_box_h=23.0, binary=binary)
    assert out == "1", out


def test_true_dash_stays_dash():
    binary = np.zeros((40, 40), dtype=np.uint8)
    _paint_h_dash(binary, 8, 16, 30, 24, thickness=2)
    tb = _tb("-", 8, 16, 30, 24, score=0.90)
    out = _maybe_dash_to_one(tb, "-", median_box_h=23.0, binary=binary)
    assert out == "-", out


def test_postprocess_flips_both_directions():
    binary = np.zeros((80, 80), dtype=np.uint8)
    # 框内居中横带 / 竖带（与真实 OCR 紧框一致）
    binary[24:26, 10:30] = 255
    binary[18:38, 53:55] = 255
    binary[55:65, 8:38] = 255
    boxes = [
        _tb("1", 10, 20, 30, 30, score=0.91),
        _tb("-", 50, 15, 58, 40, score=0.88),
        _tb("330", 5, 50, 40, 70, score=0.99),
    ]
    out = postprocess_text_boxes(boxes, binary=binary)
    by_score = {round(float(b["score"]), 2): str(b.get("text")) for b in out}
    assert by_score.get(0.91) == "-", by_score
    assert by_score.get(0.88) == "1", by_score
    assert "330" in {str(b.get("text")) for b in out}


def test_roi_short_dash_and_empty_cell_fill():
    binary = np.zeros((100, 120), dtype=np.uint8)
    # cell region with short centered dash (not full-width grid)
    _paint_h_dash(binary, 45, 40, 75, 60, thickness=2)
    assert roi_looks_like_short_dash(binary[30:70, 20:100])
    # full-width line should not count
    full = np.zeros((40, 100), dtype=np.uint8)
    full[18:21, :] = 255
    assert not roi_looks_like_short_dash(full)

    cells = [
        {
            "row_start": 3,
            "row_end": 3,
            "col_start": 5,
            "col_end": 5,
            "row_span": 1,
            "col_span": 1,
            "text": "",
            "polygon": [[20, 30], [100, 30], [100, 70], [20, 70]],
        }
    ]
    out = detect_dashes_in_empty_cells(cells, binary)
    assert out[0]["text"] == "-"


if __name__ == "__main__":
    test_one_with_horizontal_band_becomes_dash()
    test_real_tall_one_keeps_one()
    test_dash_with_vertical_band_becomes_one()
    test_true_dash_stays_dash()
    test_postprocess_flips_both_directions()
    test_roi_short_dash_and_empty_cell_fill()
    print("ok")
