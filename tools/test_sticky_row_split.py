"""sticky 行标粘连切分回归。

    python tools/test_sticky_row_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.matching import _parse_sticky_row_parts, _split_sticky_row_label  # noqa: E402
import numpy as np  # noqa: E402


def _cell(x1, y1, x2, y2, cs, ce=None):
    ce = cs if ce is None else ce
    return {
        "polygon": np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64),
        "row_start": 2,
        "row_end": 2,
        "col_start": cs,
        "col_end": ce,
        "row_span": 1,
        "col_span": ce - cs + 1,
        "texts": [],
    }


def test_parse_three_way():
    parts = _parse_sticky_row_parts("比较例 186Bk-1", n_cols=3, cut_fracs=[0.45, 0.7])
    assert parts == ["比较例 1", "86", "Bk-1"], parts
    parts = _parse_sticky_row_parts("比较例287Bk-2", n_cols=3, cut_fracs=[0.45, 0.7])
    assert parts == ["比较例 2", "87", "Bk-2"], parts


def test_parse_two_way_no_code():
    parts = _parse_sticky_row_parts("比较例893", n_cols=2, cut_fracs=[0.55])
    assert parts == ["比较例 8", "93"], parts


def test_no_bare_peel_single_intent():
    # 无几何列数时不应把「实施例10」当粘连（n_cols 门槛由调用方保证；
    # 即便 n_cols=2 无尾缀且只有「实施例10」这种短数字，也不该无证据乱切成 1|0——
    # 这里 digits 长度=2，默认 peel 1 → 实施例 1 | 0；因此必须靠「无多列重叠」拦截。
    assert _parse_sticky_row_parts("实施例10", n_cols=1) is None


def test_caption_digits_not_labels():
    from src.label_patterns import extract_independent_labels_from_joined

    assert extract_independent_labels_from_joined("[表1-2]\n聚合物") == []
    assert extract_independent_labels_from_joined("[表1-2]聚合物") == []
    assert extract_independent_labels_from_joined("实施例1\n实施例2") == [
        "实施例1",
        "实施例2",
    ]

def test_sticky_requires_multi_col():
    tb = {
        "text": "比较例 186Bk-1",
        "polygon": np.array([[10, 10], [170, 10], [170, 30], [10, 30]], dtype=np.float64),
        "score": 1.0,
    }
    # 仅一列重叠 → 不切
    cells = [_cell(10, 8, 80, 32, 0)]
    assert _split_sticky_row_label(tb, cells) is None

    cells3 = [
        _cell(10, 8, 60, 32, 0),
        _cell(60, 8, 100, 32, 1),
        _cell(100, 8, 170, 32, 2),
    ]
    pieces = _split_sticky_row_label(tb, cells3)
    assert pieces is not None and len(pieces) == 3, pieces
    assert [p["text"] for p in pieces] == ["比较例 1", "86", "Bk-1"]


def test_no_split_实施例10_single_col():
    tb = {
        "text": "实施例10",
        "polygon": np.array([[10, 10], [90, 10], [90, 28], [10, 28]], dtype=np.float64),
        "score": 1.0,
    }
    cells = [_cell(10, 8, 95, 30, 0), _cell(200, 8, 260, 30, 1)]
    assert _split_sticky_row_label(tb, cells) is None


if __name__ == "__main__":
    test_parse_three_way()
    test_parse_two_way_no_code()
    test_no_bare_peel_single_intent()
    test_sticky_requires_multi_col()
    test_no_split_实施例10_single_col()
    print("OK: sticky row split regressions passed")
