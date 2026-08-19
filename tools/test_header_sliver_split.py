"""P125：贴边 OCR 框越出竖线时，不得把单字切进邻格。

用法:
    python tools/test_header_sliver_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.matching.matching import (  # noqa: E402
    _cuts_have_cjk_singleton,
    _geometric_multi_col_split,
    _snap_cjk_cuts_to_ink_mass,
    _split_sticky_row_label,
    assign_texts_to_cells,
    polygon_to_shapely,
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


def test_cjk_singleton_cuts_detected():
    assert _cuts_have_cjk_singleton("来自颜料分散", [0, 1, 6]) is True
    assert _cuts_have_cjk_singleton("来自颜", [0, 1, 3]) is True
    assert _cuts_have_cjk_singleton("比较例186Bk-1", [0, 4, 7, 12]) is False


def test_sliver_cjk_header_not_split():
    """「来自颜料分散」左缘略越界 → 整框归右格，左格不得吃到「来」。"""
    cells = [
        _cell(0, 0, 80, 100, 1, 1, 0, 0),
        _cell(80, 0, 220, 100, 1, 1, 1, 1),
    ]
    boxes = [
        _tb("（C1）光聚合引发剂", 8, 10, 72, 90),
        # 大部分在右列，左缘越过 80 约 30px，足以触发旧的列重叠门槛
        _tb("来自颜料分散", 50, 12, 210, 40),
    ]
    # 旧路径会在无 binary 时走 _geometric_multi_col_split
    geom = _geometric_multi_col_split(
        boxes[1],
        [dict(c) for c in cells],
        polygon_to_shapely(boxes[1]["polygon"]),
        0.5,
    )
    assert geom is False, "CJK 表头不应被几何硬切"

    out, _free = assign_texts_to_cells(
        [dict(c) for c in cells],
        boxes,
        ioa_threshold=0.5,
        split_cross_cell=True,
        binary=None,
        col_seps=None,
    )
    left = str(out[0].get("text") or "")
    right = str(out[1].get("text") or "")
    assert "光来" not in left, left
    assert "来" not in left.replace("来自", ""), left
    assert "来自颜料分散" in right, right
    assert "光聚合引发剂" in left, left


def test_short_cjk_sliver_singleton_veto():
    """长度 <4 的中文框绕过表头禁切时，单字碎片仍应否决几何切分。"""
    cells = [
        _cell(0, 0, 80, 40, 1, 1, 0, 0),
        _cell(80, 0, 200, 40, 1, 1, 1, 1),
    ]
    tb = _tb("来自颜", 20, 8, 220, 32)
    geom = _geometric_multi_col_split(
        tb,
        [dict(c) for c in cells],
        polygon_to_shapely(tb["polygon"]),
        0.5,
    )
    assert geom is False, "单 CJK 碎片切分应被否决"

    out, _free = assign_texts_to_cells(
        [dict(c) for c in cells],
        [tb],
        ioa_threshold=0.5,
        split_cross_cell=True,
        binary=None,
        col_seps=None,
    )
    left = str(out[0].get("text") or "")
    right = str(out[1].get("text") or "")
    assert left != "来", left
    assert "来自颜" in right, right


def test_ink_mass_snaps_lai_to_right():
    """胶连两列表头「（C1）光来自颜料分散」：沟右侧墨水把「来」拉回右段。"""
    text = "（C1）光来自颜料分散"
    # 字宽模型切在「来」之后；沟在「光|来」之间
    cuts = [0, 6, len(text)]
    assert text[:6] == "（C1）光来"
    ink = np.zeros(200, dtype=np.float64)
    ink[:70] = 8.0  # （C1）光
    ink[90:] = 8.0  # 来自颜料分散
    snapped = [80.0]
    tb_box = (0.0, 0.0, 200.0, 20.0)
    out = _snap_cjk_cuts_to_ink_mass(text, cuts, snapped, tb_box, ink, 0.0)
    assert text[: out[1]] == "（C1）光", text[: out[1]]
    assert text[out[1] :] == "来自颜料分散", text[out[1] :]


def test_sticky_row_label_still_splits():
    tb = {
        "text": "比较例 186Bk-1",
        "polygon": np.array(
            [[10, 10], [170, 10], [170, 30], [10, 30]], dtype=np.float64
        ),
        "score": 1.0,
    }
    cells = [
        _cell(10, 8, 60, 32, 2, 2, 0),
        _cell(60, 8, 100, 32, 2, 2, 1),
        _cell(100, 8, 170, 32, 2, 2, 2),
    ]
    pieces = _split_sticky_row_label(tb, cells)
    assert pieces is not None and len(pieces) == 3, pieces
    assert [p["text"] for p in pieces] == ["比较例 1", "86", "Bk-1"]


def main() -> int:
    test_cjk_singleton_cuts_detected()
    test_sliver_cjk_header_not_split()
    test_short_cjk_sliver_singleton_veto()
    test_ink_mass_snaps_lai_to_right()
    test_sticky_row_label_still_splits()
    print("OK: header sliver split tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
