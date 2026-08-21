# -*- coding: utf-8 -*-
"""嵌套中层表头不得被当成表体 stub 而拆掉 colspan。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.output.html_formatter import (  # noqa: E402
    _effective_header_end,
    _split_label_over_data_columns,
)


def _cell(
    rs: int,
    re: int,
    cs: int,
    ce: int,
    text: str,
) -> Dict[str, Any]:
    return {
        "row_start": rs,
        "row_end": re,
        "col_start": cs,
        "col_end": ce,
        "row_span": re - rs + 1,
        "col_span": ce - cs + 1,
        "text": text,
        "texts": [text] if text else [],
        "polygon": [
            [cs * 40.0, rs * 20.0],
            [(ce + 1) * 40.0, rs * 20.0],
            [(ce + 1) * 40.0, (re + 1) * 20.0],
            [cs * 40.0, (re + 1) * 20.0],
        ],
    }


def _p33_like_header_cells() -> list:
    """简化版 P33X234 三级表头（含空顶行）。"""
    return [
        _cell(0, 0, 0, 15, ""),
        _cell(1, 3, 0, 0, ""),
        _cell(1, 1, 1, 12, "树脂"),
        _cell(1, 3, 13, 13, "具有氟的单体率"),
        _cell(1, 3, 14, 14, "其他树脂"),
        _cell(1, 3, 15, 15, "感光剂"),
        _cell(2, 3, 1, 1, "树脂"),
        _cell(2, 2, 2, 6, "酸成分(摩尔比)"),
        _cell(2, 2, 7, 12, "二胺成分(摩尔比)"),
        _cell(3, 3, 2, 2, "PMDA-HH"),
        _cell(3, 3, 3, 3, "TDA100"),
        _cell(3, 3, 4, 4, "CBDA"),
        _cell(3, 3, 5, 5, "6FDA"),
        _cell(3, 3, 6, 6, "ODPA"),
        _cell(3, 3, 7, 7, "BAHF"),
        _cell(3, 3, 8, 8, "(a)"),
        _cell(3, 3, 9, 9, "APBS"),
        _cell(3, 3, 10, 10, "DAE"),
        _cell(3, 3, 11, 11, "ED600"),
        _cell(3, 3, 12, 12, "SiDA"),
        _cell(4, 4, 0, 0, "实施例1"),
        _cell(4, 4, 1, 1, "A"),
        _cell(4, 4, 2, 2, "10"),
        _cell(4, 4, 5, 5, "90"),
        _cell(4, 4, 7, 7, "85"),
        _cell(4, 4, 12, 12, "5"),
        _cell(4, 4, 13, 13, "92"),
        _cell(4, 4, 14, 14, "e"),
        _cell(4, 4, 15, 15, "b"),
    ]


def test_nested_mid_header_not_body_stub() -> None:
    cells = _p33_like_header_cells()
    he = _effective_header_end(cells)
    assert he >= 3, f"header_end should cover mid/bottom header rows, got {he}"


def test_split_label_keeps_mid_header_colspans() -> None:
    """header_end 正确后，身列拆分不得动中层酸/二胺 colspan。"""
    cells = _p33_like_header_cells()
    before = {
        str(c.get("text") or ""): (int(c["col_start"]), int(c["col_end"]))
        for c in cells
        if "酸成分" in str(c.get("text") or "")
        or "二胺成分" in str(c.get("text") or "")
    }
    after_cells = _split_label_over_data_columns([dict(c) for c in cells])
    after = {
        str(c.get("text") or ""): (int(c["col_start"]), int(c["col_end"]))
        for c in after_cells
        if "酸成分" in str(c.get("text") or "")
        or "二胺成分" in str(c.get("text") or "")
    }
    assert before == after, (before, after)
    assert before["酸成分(摩尔比)"] == (2, 6)
    assert before["二胺成分(摩尔比)"] == (7, 12)


def test_true_body_stub_still_caps_header() -> None:
    """真表体左侧比较例 rowspan 仍应截断表头带。"""
    cells = [
        _cell(0, 1, 0, 0, ""),
        _cell(0, 0, 1, 2, "项目"),
        _cell(1, 1, 1, 1, "A"),
        _cell(1, 1, 2, 2, "B"),
        _cell(2, 4, 0, 0, "比较例1"),
        _cell(2, 2, 1, 1, "10"),
        _cell(2, 2, 2, 2, "20"),
        _cell(3, 3, 1, 1, "11"),
        _cell(3, 3, 2, 2, "21"),
    ]
    he = _effective_header_end(cells)
    assert he == 1, f"expected header_end=1, got {he}"


if __name__ == "__main__":
    test_nested_mid_header_not_body_stub()
    test_split_label_keeps_mid_header_colspans()
    test_true_body_stub_still_caps_header()
    print("ok")
