# -*- coding: utf-8 -*-
"""列级 dash 一致性：无 dash 证据时不把索引列「1」改成「-」。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.matching.matching import (
    _column_should_coerce_dash_misreads,
    _fix_dash_column_consistency,
)


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


def test_no_dash_index_column_keeps_one():
    assert not _column_should_coerce_dash_misreads(0, 1, 3)
    assert not _column_should_coerce_dash_misreads(0, 1, 11)
    cells = [_cell(0, 1, "")] + [_cell(i, 1, str(i)) for i in range(1, 13)]
    out = _fix_dash_column_consistency(cells)
    assert out[1]["text"] == "1"
    assert out[0]["text"] == ""


def test_dash_majority_still_coerces():
    assert _column_should_coerce_dash_misreads(5, 2, 0)
    cells = [_cell(i, 0, "-") for i in range(5)] + [_cell(5, 0, "1"), _cell(6, 0, "1")]
    out = _fix_dash_column_consistency(cells)
    assert all(c["text"] == "-" for c in out)


if __name__ == "__main__":
    test_no_dash_index_column_keeps_one()
    test_dash_majority_still_coerces()
    print("ok")
