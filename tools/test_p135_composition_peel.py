# -*- coding: utf-8 -*-
"""P135-style: 比较例186|Bk-1 → 比较例 1|86|Bk-1；组成父头归位；末行不吞 Bk 列。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.matching.matching import _parse_sticky_row_parts
from src.output.html_formatter import (
    _merge_leading_label_gaps,
    _peel_glued_example_index_columns,
    _restore_composition_parent_header,
    cells_to_html_table,
)
from src.utils.label_patterns import (
    split_example_local_and_composition_id,
    repair_glued_example_label_index,
)


def _cell(rs, cs, text, ce=None, re=None):
    ce = cs if ce is None else ce
    re = rs if re is None else re
    return {
        "row_start": rs,
        "row_end": re,
        "col_start": cs,
        "col_end": ce,
        "row_span": re - rs + 1,
        "col_span": ce - cs + 1,
        "text": text,
        "polygon": [
            [cs * 10, rs * 10],
            [ce * 10 + 8, rs * 10],
            [ce * 10 + 8, re * 10 + 8],
            [cs * 10, re * 10 + 8],
        ],
        "texts": [],
    }


def test_split_helpers():
    assert split_example_local_and_composition_id("比较例186") == ("比较例 1", "86")
    assert split_example_local_and_composition_id("比较例489") == ("比较例 4", "89")
    assert split_example_local_and_composition_id("比较例 3 88") == ("比较例 3", "88")
    assert split_example_local_and_composition_id("比较例 1") is None
    assert split_example_local_and_composition_id("比较例 86") is None
    # 序号列路径仍工作；Bk 不走 repair
    assert repair_glued_example_label_index("实施例 32 32", "") == ("实施例 32", "32")
    assert repair_glued_example_label_index("比较例186", "Bk-1") is None


def test_sticky_two_col_splits_digits():
    assert _parse_sticky_row_parts("比较例287Bk-2", n_cols=2) == [
        "比较例 2 87",
        "Bk-2",
    ]
    assert _parse_sticky_row_parts("比较例287Bk-2", n_cols=3) == [
        "比较例 2",
        "87",
        "Bk-2",
    ]
    # 仅 2 位数字 + 码：不误拆成 8|6
    assert _parse_sticky_row_parts("比较例86Bk-1", n_cols=3) == [
        "比较例 86",
        "Bk-1",
    ]


def test_peel_inserts_composition_column():
    cells = [
        _cell(0, 0, "", re=1),
        _cell(0, 1, "颜料分散液", re=1),
        _cell(0, 2, "来自调配液的(A1)", re=1),
        _cell(2, 0, "比较例186"),
        _cell(2, 1, "Bk-1"),
        _cell(2, 2, "AC-1"),
        _cell(3, 0, "比较例287"),
        _cell(3, 1, "Bk-2"),
        _cell(3, 2, "PI-1"),
        _cell(4, 0, "比较例 3 88"),
        _cell(4, 1, "Bk-2"),
        _cell(4, 2, "PI-1"),
        _cell(5, 0, "比较例489"),
        _cell(5, 1, "Bk-2"),
        _cell(5, 2, "PI-1"),
    ]
    out = _peel_glued_example_index_columns(cells)
    by = {(c["row_start"], c["col_start"]): c for c in out if not c.get("_drop_render")}
    assert by[(2, 0)]["text"] == "比较例 1"
    assert by[(2, 1)]["text"] == "86"
    assert by[(2, 2)]["text"] == "Bk-1"
    assert by[(3, 0)]["text"] == "比较例 2"
    assert by[(3, 1)]["text"] == "87"
    assert by[(4, 0)]["text"] == "比较例 3"
    assert by[(4, 1)]["text"] == "88"
    assert by[(5, 0)]["text"] == "比较例 4"
    assert by[(5, 1)]["text"] == "89"


def test_p136_index_column_untouched():
    """已有干净 比较例 N | N 序号列时不误插列。"""
    cells = [
        _cell(0, 0, "", re=1),
        _cell(0, 1, "组合物", re=1),
        _cell(2, 0, "比较例 1"),
        _cell(2, 1, "86"),
        _cell(2, 2, "35"),
        _cell(3, 0, "比较例 2"),
        _cell(3, 1, "87"),
        _cell(3, 2, "25"),
        _cell(4, 0, "比较例 3"),
        _cell(4, 1, "88"),
        _cell(4, 2, "45"),
        _cell(5, 0, "比较例 4"),
        _cell(5, 1, "89"),
        _cell(5, 2, "55"),
    ]
    out = _peel_glued_example_index_columns(cells)
    by = {(c["row_start"], c["col_start"]): c["text"] for c in out}
    assert by[(2, 0)] == "比较例 1"
    assert by[(2, 1)] == "86"
    assert by[(2, 2)] == "35"
    # 列数不应增加
    assert max(c["col_end"] for c in out) == 2


def test_merge_gaps_keeps_bk_column():
    cells = [
        _cell(0, 0, "组合物", re=0),
        _cell(0, 1, "颜料分散液", re=0),
        _cell(0, 2, "树脂", re=0),
        _cell(1, 0, "比较例 1"),
        _cell(1, 1, "Bk-1"),
        _cell(1, 2, "PI-1"),
        _cell(2, 0, "比较例 2"),
        _cell(2, 1, "Bk-2"),
        _cell(2, 2, "PI-1"),
        # 末行：Bk 列空，不得吞进标签
        _cell(3, 0, "比较例 8"),
        _cell(3, 1, ""),
        _cell(3, 2, "PI-1"),
    ]
    out = _merge_leading_label_gaps(cells)
    by = {(c["row_start"], c["col_start"]): c for c in out if not c.get("_drop_render")}
    assert by[(3, 0)]["col_span"] == 1
    assert by[(3, 0)]["text"] == "比较例 8"


def test_restore_composition_header():
    cells = [
        _cell(0, 0, "", re=1),
        _cell(0, 1, "颜料分散液 组成[质量份]", re=1),
        _cell(0, 2, "", ce=5),  # empty parent colspan
        _cell(1, 2, "来自颜料的(A1)"),
        _cell(1, 3, "来自调配液的(A1)"),
        _cell(1, 4, "溶剂"),
        _cell(1, 5, "比率"),
        _cell(2, 0, "比较例 1"),
        _cell(2, 1, "Bk-1"),
        _cell(2, 2, ""),
        _cell(2, 3, "PI-1"),
        _cell(2, 4, "MBA"),
        _cell(2, 5, "100"),
    ]
    out = _restore_composition_parent_header(cells)
    by = {(c["row_start"], c["col_start"]): c["text"] for c in out}
    assert "组成" in by[(0, 2)]
    assert "颜料" in by[(0, 1)]
    assert "组成" not in by[(0, 1)]


def test_existing_peel_path_a_still_works():
    cells = [
        _cell(0, 0, "组合物"),
        _cell(0, 1, ""),
        _cell(2, 0, "实施例 32 32"),
        _cell(2, 1, ""),
        _cell(2, 2, "Bk-2"),
    ]
    for i, n in enumerate(range(33, 42), start=3):
        cells.append(_cell(i, 0, f"实施例 {n}"))
        cells.append(_cell(i, 1, str(n)))
        cells.append(_cell(i, 2, "Bk-2"))
    out = _peel_glued_example_index_columns(cells)
    by = {(c["row_start"], c["col_start"]): c["text"] for c in out}
    assert by[(2, 0)] == "实施例 32"
    assert by[(2, 1)] == "32"


if __name__ == "__main__":
    test_split_helpers()
    test_sticky_two_col_splits_digits()
    test_peel_inserts_composition_column()
    test_p136_index_column_untouched()
    test_merge_gaps_keeps_bk_column()
    test_restore_composition_header()
    test_existing_peel_path_a_still_works()
    print("ok")
