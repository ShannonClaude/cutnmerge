# -*- coding: utf-8 -*-
"""标签列与右邻序号列串位：实施例 32 32|∅、实施例 2 3|23 等。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.output.html_formatter import _peel_glued_example_index_columns
from src.utils.label_patterns import repair_glued_example_label_index


def _cell(rs, cs, text):
    return {
        "row_start": rs,
        "row_end": rs,
        "col_start": cs,
        "col_end": cs,
        "row_span": 1,
        "col_span": 1,
        "text": text,
        "polygon": [
            [cs * 10, rs * 10],
            [cs * 10 + 8, rs * 10],
            [cs * 10 + 8, rs * 10 + 8],
            [cs * 10, rs * 10 + 8],
        ],
        "texts": [],
    }


def test_pair_helpers():
    assert repair_glued_example_label_index("实施例 32 32", "") == (
        "实施例 32",
        "32",
    )
    assert repair_glued_example_label_index("实施例 32 32", "32") == (
        "实施例 32",
        "32",
    )
    assert repair_glued_example_label_index("实施例 2 3", "23") == (
        "实施例 23",
        "23",
    )
    assert repair_glued_example_label_index("实施例 3 2", "32") == (
        "实施例 32",
        "32",
    )
    # 不误伤：序号与碎片拼接不一致
    assert repair_glued_example_label_index("实施例 2 3", "24") is None
    # 不误伤：已干净
    assert repair_glued_example_label_index("实施例 33", "33") is None
    # 不误伤：右侧不是序号列内容
    assert repair_glued_example_label_index("实施例 32 32", "Bk-2") is None


def test_column_gate_peels_p117_style():
    # header row 0–1; body from 2
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
    assert by[(3, 0)] == "实施例 33"
    assert by[(3, 1)] == "33"


def test_column_gate_repairs_split_digits():
    cells = [
        _cell(0, 0, "组合物"),
        _cell(2, 0, "实施例 2 3"),
        _cell(2, 1, "23"),
        _cell(3, 0, "实施例 24"),
        _cell(3, 1, "24"),
        _cell(4, 0, "实施例 25"),
        _cell(4, 1, "25"),
        _cell(5, 0, "实施例 3 2"),
        _cell(5, 1, "32"),
        _cell(6, 0, "实施例 33"),
        _cell(6, 1, "33"),
    ]
    out = _peel_glued_example_index_columns(cells)
    by = {(c["row_start"], c["col_start"]): c["text"] for c in out}
    assert by[(2, 0)] == "实施例 23"
    assert by[(2, 1)] == "23"
    assert by[(5, 0)] == "实施例 32"
    assert by[(5, 1)] == "32"


def test_no_index_column_untouched():
    # 右侧是数据列，不应触发
    cells = [
        _cell(0, 0, "项目"),
        _cell(2, 0, "实施例 1"),
        _cell(2, 1, "Bk-2"),
        _cell(3, 0, "实施例 2"),
        _cell(3, 1, "Bk-3"),
        _cell(4, 0, "实施例 3 3"),
        _cell(4, 1, "Bk-4"),
    ]
    out = _peel_glued_example_index_columns(cells)
    by = {(c["row_start"], c["col_start"]): c["text"] for c in out}
    assert by[(4, 0)] == "实施例 3 3"
    assert by[(4, 1)] == "Bk-4"


if __name__ == "__main__":
    test_pair_helpers()
    test_column_gate_peels_p117_style()
    test_column_gate_repairs_split_digits()
    test_no_index_column_untouched()
    print("ok")
