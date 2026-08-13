"""hline_repair 表头带局部 rowspan 修复单测。

    python tools/test_hline_repair.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hline_repair import (  # noqa: E402
    _hline_coverage_ratio,
    repair_rowspans_by_hline_gaps,
)


def _cell(x1, y1, x2, y2, rs, re, cs, ce=None):
    ce = cs if ce is None else ce
    return {
        "polygon": np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64
        ),
        "row_start": rs,
        "row_end": re,
        "col_start": cs,
        "col_end": ce,
        "row_span": re - rs + 1,
        "col_span": ce - cs + 1,
        "texts": [],
        "text": "",
    }


def _tb(text, x1, y1, x2, y2):
    return {
        "text": text,
        "polygon": np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64
        ),
        "score": 1.0,
    }


def _blank(h=200, w=300):
    return np.zeros((h, w), dtype=np.uint8)


def _draw_hline(binary, y, x1, x2, thickness=2):
    y0 = max(0, y - thickness // 2)
    y1 = min(binary.shape[0], y0 + thickness)
    binary[y0:y1, x1:x2] = 255


def test_gap_merges_header_pair():
    """表头列断线 + 上空碎片 → 合并 rowspan。"""
    binary = _blank()
    # 旁列有完整横线，目标列无线
    _draw_hline(binary, 50, 0, 100, thickness=2)
    cells = [
        _cell(100, 10, 180, 48, 0, 0, 1),  # 上空
        _cell(100, 52, 180, 120, 1, 1, 1),  # 下有字区
        _cell(0, 10, 100, 48, 0, 0, 0),
        _cell(0, 52, 100, 120, 1, 1, 0),
        _cell(0, 130, 180, 180, 2, 2, 0),  # 表体行占位
    ]
    boxes = [
        _tb("mol%", 110, 60, 170, 110),
        _tb("合成例1", 10, 140, 80, 170),
    ]
    out = repair_rowspans_by_hline_gaps(cells, binary, boxes)
    merged = [
        c
        for c in out
        if int(c["col_start"]) == 1 and int(c["row_span"]) >= 2
    ]
    assert merged, f"expected rowspan merge, got {[ (c['row_start'],c['row_end'],c['col_start']) for c in out ]}"
    # 旁列有线，不应合并
    side = [
        c
        for c in out
        if int(c["col_start"]) == 0 and int(c["row_start"]) <= 1
    ]
    assert all(int(c["row_span"]) == 1 for c in side), side


def test_strong_hline_keeps_split():
    """真横线贯穿该列 → 不合并。"""
    binary = _blank()
    _draw_hline(binary, 50, 0, 200, thickness=3)
    cells = [
        _cell(20, 10, 100, 48, 0, 0, 0),
        _cell(20, 52, 100, 120, 1, 1, 0),
        _cell(20, 130, 100, 180, 2, 2, 0),
    ]
    boxes = [
        _tb("1", 30, 20, 50, 40),
        _tb("2", 30, 70, 50, 90),
        _tb("合成例1", 30, 140, 90, 170),
    ]
    out = repair_rowspans_by_hline_gaps(cells, binary, boxes)
    assert all(
        not (int(c["col_start"]) == 0 and int(c["row_span"]) > 1 and int(c["row_end"]) <= 1)
        for c in out
    ), out
    assert len([c for c in out if int(c["row_start"]) == 0 and int(c["col_start"]) == 0]) == 1
    assert len([c for c in out if int(c["row_start"]) == 1 and int(c["col_start"]) == 0]) == 1


def test_body_boundary_not_touched():
    """表体行界即使无线也不合并。"""
    binary = _blank()
    cells = [
        _cell(10, 10, 100, 40, 0, 0, 0),
        _cell(10, 50, 100, 90, 1, 1, 0),
        _cell(10, 100, 100, 140, 2, 2, 0),
        _cell(10, 150, 100, 190, 3, 3, 0),
    ]
    boxes = [
        _tb("聚合物", 20, 15, 80, 35),
        _tb("子列", 20, 55, 80, 85),
        _tb("合成例1", 20, 110, 90, 130),
        _tb("合成例2", 20, 160, 90, 180),
    ]
    out = repair_rowspans_by_hline_gaps(cells, binary, boxes)
    bodyish = [
        c
        for c in out
        if int(c["row_start"]) >= 2 or (int(c["row_start"]) <= 2 and int(c["row_end"]) >= 3)
    ]
    # 不应出现跨 合成例1/2 的 rowspan
    assert not any(
        int(c["row_start"]) <= 2 and int(c["row_end"]) >= 3 for c in out
    ), out


def test_hline_coverage_detects_line():
    binary = _blank(80, 120)
    _draw_hline(binary, 40, 10, 110, thickness=2)
    assert _hline_coverage_ratio(binary, 40, 10, 110, tol=3) >= 0.4
    assert _hline_coverage_ratio(binary, 40, 10, 50, tol=3) >= 0.4
    # 无墨迹列
    assert _hline_coverage_ratio(binary, 20, 10, 110, tol=2) < 0.4


def main():
    test_hline_coverage_detects_line()
    test_gap_merges_header_pair()
    test_strong_hline_keeps_split()
    test_body_boundary_not_touched()
    print("ok")


if __name__ == "__main__":
    main()
