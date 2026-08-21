# -*- coding: utf-8 -*-
"""单元测试：聚合物孤列表头迁到左侧树脂名列。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.output.html_formatter import (
    _align_orphan_left_anchor_headers,
    cells_to_html_table,
)


def _cell(x1, y1, x2, y2, rs, re, cs, ce, text=""):
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
        "text": text,
    }


def test_align_polymer_left_to_resin_body():
    """聚合物在无表体列、左侧有树脂溶液 → 表头迁左，正对树脂名。"""
    cells = [
        _cell(40, 0, 80, 40, 0, 1, 2, 2, "聚合物"),
        _cell(80, 0, 240, 20, 0, 0, 4, 7, "单体[mol比]"),
        _cell(80, 20, 120, 40, 1, 1, 4, 4, "具有酸性基团的共聚成分"),
        _cell(120, 20, 160, 40, 1, 1, 5, 5, "具有芳香族基团的共聚成分"),
        _cell(160, 20, 200, 40, 1, 1, 6, 6, "具有脂环式基团的共聚成分"),
        _cell(200, 20, 240, 40, 1, 1, 7, 7, "具有烯键式不饱和化合物"),
        _cell(0, 40, 40, 60, 2, 2, 0, 0, "合成例6"),
        _cell(40, 40, 80, 60, 2, 2, 1, 1, "丙烯酸树脂溶液\n(AC-1)"),
        _cell(80, 40, 120, 60, 2, 2, 4, 4, "MAA\n(50)"),
        _cell(120, 40, 160, 60, 2, 2, 5, 5, "STR\n(30)"),
        _cell(160, 40, 200, 60, 2, 2, 6, 6, "TCDM\n(20)"),
        _cell(200, 40, 240, 60, 2, 2, 7, 7, "GMA\n(20)"),
    ]
    out = _align_orphan_left_anchor_headers(cells)
    poly = [c for c in out if "聚合物" in str(c.get("text") or "")][0]
    assert int(poly["col_start"]) == 1
    resin = [c for c in out if "丙烯酸" in str(c.get("text") or "")][0]
    assert int(resin["col_start"]) == int(poly["col_start"])
    print("ok test_align_polymer_left_to_resin_body")


def test_align_no_move_when_body_under_polymer():
    """聚合物列已有表体 → 不迁。"""
    cells = [
        _cell(40, 0, 80, 40, 0, 1, 1, 1, "聚合物"),
        _cell(80, 0, 160, 20, 0, 0, 2, 3, "单体[mol比]"),
        _cell(0, 40, 40, 60, 2, 2, 0, 0, "合成例1"),
        _cell(40, 40, 80, 60, 2, 2, 1, 1, "聚酰亚胺\n(PI-1)"),
        _cell(80, 40, 120, 60, 2, 2, 2, 2, "ODPA\n(100)"),
    ]
    out = _align_orphan_left_anchor_headers([dict(c) for c in cells])
    poly = [c for c in out if "聚合物" in str(c.get("text") or "")][0]
    assert int(poly["col_start"]) == 1
    print("ok test_align_no_move_when_body_under_polymer")


def test_html_polymer_over_resin():
    """整表渲染：丙烯酸树脂溶液应在聚合物正下方。"""
    cells = [
        _cell(40, 0, 80, 40, 0, 1, 2, 2, "聚合物"),
        _cell(80, 0, 240, 20, 0, 0, 4, 7, "单体[mol比]"),
        _cell(80, 20, 120, 40, 1, 1, 4, 4, "酸性"),
        _cell(120, 20, 160, 40, 1, 1, 5, 5, "芳香"),
        _cell(160, 20, 200, 40, 1, 1, 6, 6, "脂环"),
        _cell(200, 20, 240, 40, 1, 1, 7, 7, "烯键"),
        _cell(0, 40, 40, 60, 2, 2, 0, 0, "合成例6"),
        _cell(40, 40, 80, 60, 2, 2, 1, 1, "丙烯酸树脂溶液\n(AC-1)"),
        _cell(80, 40, 120, 60, 2, 2, 4, 4, "MAA"),
        _cell(120, 40, 160, 60, 2, 2, 5, 5, "STR"),
        _cell(160, 40, 200, 60, 2, 2, 6, 6, "TCDM"),
        _cell(200, 40, 240, 60, 2, 2, 7, 7, "GMA"),
    ]
    html = cells_to_html_table(cells)
    # 聚合物所在列应对齐到含 AC-1 的数据格
    rows = re.findall(r"<tr>([\s\S]*?)</tr>", html)
    assert len(rows) >= 3
    hdr0 = re.findall(r"<td([^>]*)>([\s\S]*?)</td>", rows[0])
    body = re.findall(r"<td([^>]*)>([\s\S]*?)</td>", rows[2])
    # 找聚合物在第几「逻辑格」输出
    poly_i = None
    for i, (attrs, body_t) in enumerate(hdr0):
        if "聚合物" in body_t:
            poly_i = i
            break
    assert poly_i is not None
    # 数据行同序：合成例 | 树脂 | …
    assert "合成例" in body[0][1]
    assert "丙烯酸" in body[1][1] or "AC-1" in body[1][1]
    # 聚合物应是 header 中合成例空角之后的下一格（与树脂同列）
    assert poly_i >= 1
    print("ok test_html_polymer_over_resin")


if __name__ == "__main__":
    test_align_polymer_left_to_resin_body()
    test_align_no_move_when_body_under_polymer()
    test_html_polymer_over_resin()
    print("ALL PASS")
