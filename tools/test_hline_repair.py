"""hline_repair 表头带局部 rowspan 修复单测。

    python tools/test_hline_repair.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.structure.hline_repair import (  # noqa: E402
    _hline_coverage_ratio,
    _vline_coverage_ratio,
    repair_colspans_by_vline_gaps,
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


def _draw_vline(binary, x, y1, y2, thickness=2):
    x0 = max(0, x - thickness // 2)
    x1 = min(binary.shape[1], x0 + thickness)
    binary[y1:y2, x0:x1] = 255


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


def test_vline_colspan_merges_when_body_has_line():
    """表头无竖线、表体有竖线 → 合并相邻表头 colspan。"""
    binary = _blank(h=220, w=320)
    # 竖线须落在列界 col_seps[1]≈150 处
    _draw_vline(binary, 150, 120, 200, thickness=3)
    cells = [
        _cell(10, 10, 145, 55, 0, 0, 0),
        _cell(155, 10, 290, 55, 0, 0, 1),
        _cell(10, 60, 145, 110, 1, 1, 0, ce=0),
        _cell(155, 60, 290, 110, 1, 1, 1, ce=1),
        _cell(10, 120, 145, 170, 2, 2, 0),
        _cell(155, 120, 290, 170, 2, 2, 1),
    ]
    boxes = [
        _tb("二羟基", 20, 65, 80, 105),
        _tb("二胺及其衍生物", 170, 65, 280, 105),
        _tb("BAHF(95)", 20, 125, 120, 165),
        _tb("SiDA(5)", 170, 125, 260, 165),
        _tb("合成例8", 20, 125, 100, 165),
    ]
    out = repair_colspans_by_vline_gaps(cells, binary, boxes)
    merged = [
        c
        for c in out
        if int(c["row_start"]) == 1 and int(c.get("col_span") or 1) >= 2
    ]
    assert merged, f"expected colspan merge, got {[(c['row_start'], c['col_start'], c['col_end']) for c in out]}"


def test_vline_keeps_true_dual_subheaders():
    """真·双子表头（两侧都完整）→ 不合并。"""
    binary = _blank(h=220, w=320)
    _draw_vline(binary, 150, 120, 200, thickness=3)
    cells = [
        _cell(10, 60, 145, 110, 1, 1, 0),
        _cell(155, 60, 290, 110, 1, 1, 1),
        _cell(10, 120, 145, 170, 2, 2, 0),
        _cell(155, 120, 290, 170, 2, 2, 1),
    ]
    boxes = [
        _tb("四羧酸及其衍生物", 15, 65, 135, 105),
        _tb("二胺及其衍生物", 165, 65, 285, 105),
        _tb("合成例8", 20, 125, 100, 165),
    ]
    out = repair_colspans_by_vline_gaps(cells, binary, boxes)
    row1 = [c for c in out if int(c["row_start"]) == 1]
    assert len(row1) == 2, row1
    assert all(int(c.get("col_span") or 1) == 1 for c in row1)


def test_vline_keeps_short_standalone_alkenyl_header():
    """P98：长卤素列头 + 短「烯基」→ 不得 colspan 杂糅。"""
    binary = _blank(h=220, w=320)
    _draw_vline(binary, 150, 120, 200, thickness=3)
    cells = [
        _cell(10, 60, 145, 110, 1, 1, 0),
        _cell(155, 60, 290, 110, 1, 1, 1),
        _cell(10, 120, 145, 170, 2, 2, 0),
        _cell(155, 120, 290, 170, 2, 2, 1),
    ]
    boxes = [
        _tb("以卤素", 20, 62, 90, 78),
        _tb("作为取代", 18, 78, 100, 94),
        _tb("基的基团", 18, 94, 100, 108),
        _tb("烯基", 175, 75, 230, 100),
        _tb("合成例8", 20, 125, 100, 165),
    ]
    out = repair_colspans_by_vline_gaps(cells, binary, boxes)
    row1 = [c for c in out if int(c["row_start"]) == 1]
    assert len(row1) == 2, [
        (c["col_start"], c["col_end"], c.get("col_span")) for c in row1
    ]
    assert all(int(c.get("col_span") or 1) == 1 for c in row1)


def test_wrap_rowspan_merge_substantive_cjk():
    """同列折行长中文表头 → rowspan 合并。"""
    binary = _blank(h=200, w=200)
    cells = [
        _cell(20, 10, 180, 48, 0, 0, 1),
        _cell(20, 52, 180, 95, 1, 1, 1),
        _cell(20, 110, 180, 150, 2, 2, 1),
    ]
    boxes = [
        _tb("双氨基酚化合物及其衍生物", 30, 15, 170, 45),
        _tb("二羟基二胺及其衍生物", 30, 55, 170, 90),
        _tb("合成例8", 30, 115, 120, 145),
    ]
    out = repair_rowspans_by_hline_gaps(cells, binary, boxes)
    merged = [
        c
        for c in out
        if int(c["col_start"]) == 1 and int(c.get("row_span") or 1) >= 2
    ]
    assert merged, out


def test_vline_coverage_detects_line():
    binary = _blank(80, 120)
    _draw_vline(binary, 60, 10, 70, thickness=2)
    assert _vline_coverage_ratio(binary, 60, 10, 70, tol=3) >= 0.4
    assert _vline_coverage_ratio(binary, 20, 10, 25, tol=3) < 0.4


def test_vline_absorbs_empty_peers_into_right_group_header():
    """P123：左空块 + 右有字组表头，表头无竖线、表体有竖线 → 合成一块。"""
    binary = _blank(h=360, w=960)
    # 表体列界（col3 左≈175、col6 左≈412），表头带 40-62 保持空白
    _draw_vline(binary, 175, 245, 340, thickness=3)
    _draw_vline(binary, 412, 245, 340, thickness=3)
    cells = [
        _cell(7, 40, 94, 241, 1, 2, 0, 0),
        _cell(98, 40, 123, 241, 1, 2, 1, 1),
        _cell(127, 40, 171, 62, 1, 1, 2, 2),
        _cell(175, 40, 408, 62, 1, 1, 3, 5),
        _cell(412, 40, 921, 62, 1, 1, 6, 10),
        _cell(127, 65, 171, 241, 2, 2, 2, 2),
        _cell(175, 65, 252, 241, 2, 2, 3, 3),
        _cell(256, 65, 332, 241, 2, 2, 4, 4),
        _cell(335, 65, 408, 241, 2, 2, 5, 5),
        _cell(412, 65, 543, 241, 2, 2, 6, 6),
        _cell(547, 65, 627, 241, 2, 2, 7, 7),
        _cell(631, 65, 752, 241, 2, 2, 8, 8),
        _cell(756, 65, 848, 241, 2, 2, 9, 9),
        _cell(852, 65, 921, 241, 2, 2, 10, 10),
        _cell(7, 245, 94, 293, 3, 3, 0, 0),
        _cell(98, 245, 123, 293, 3, 3, 1, 1),
        _cell(127, 245, 171, 293, 3, 3, 2, 2),
        _cell(175, 245, 252, 293, 3, 3, 3, 3),
        _cell(256, 245, 332, 293, 3, 3, 4, 4),
        _cell(335, 245, 408, 293, 3, 3, 5, 5),
        _cell(412, 245, 543, 293, 3, 3, 6, 6),
        _cell(547, 245, 627, 293, 3, 3, 7, 7),
        _cell(631, 245, 752, 293, 3, 3, 8, 8),
        _cell(756, 245, 848, 293, 3, 3, 9, 9),
        _cell(852, 245, 921, 293, 3, 3, 10, 10),
    ]
    cells[1]["text"] = "组合物"
    cells[4]["text"] = "组成[质量份]"
    boxes = [
        _tb("组合物", 100, 80, 120, 200),
        _tb("组成[质量份]", 464, 40, 585, 61),
        _tb("实施例 37", 10, 250, 80, 280),
    ]
    out = repair_colspans_by_vline_gaps(cells, binary, boxes)
    header = [
        c
        for c in out
        if int(c["row_start"]) == int(c["row_end"]) == 1
    ]
    assert len(header) == 1, [
        (c["col_start"], c["col_end"], c.get("text")) for c in header
    ]
    merged = header[0]
    assert (int(merged["col_start"]), int(merged["col_end"])) == (2, 10)
    assert "组成[质量份]" in str(merged.get("text") or "")

    from src.output.html_formatter import cells_to_html_table

    html = cells_to_html_table(out)
    assert "组成[质量份]" in html
    assert not re.search(
        r"组合物</td>\s*<td[^>]*>\s*</td>",
        html,
    ), html
    assert re.search(
        r"组合物</td>\s*<td colspan=\"9\">组成\[质量份\]",
        html,
    ), html


def test_vline_still_absorbs_empty_on_right_of_label():
    """原方向仍有效：左有字、右空 → 空块并入左侧标签。"""
    binary = _blank(h=220, w=400)
    _draw_vline(binary, 210, 120, 200, thickness=3)
    cells = [
        _cell(10, 10, 190, 50, 0, 0, 0, 2),
        _cell(210, 10, 380, 50, 0, 0, 3, 4),
        _cell(10, 55, 80, 110, 1, 1, 0, 0),
        _cell(90, 55, 190, 110, 1, 1, 1, 2),
        _cell(210, 55, 290, 110, 1, 1, 3, 3),
        _cell(300, 55, 380, 110, 1, 1, 4, 4),
        _cell(10, 120, 80, 170, 2, 2, 0, 0),
        _cell(90, 120, 190, 170, 2, 2, 1, 2),
        _cell(210, 120, 290, 170, 2, 2, 3, 3),
        _cell(300, 120, 380, 170, 2, 2, 4, 4),
    ]
    cells[0]["text"] = "组成[质量份]"
    boxes = [
        _tb("组成[质量份]", 40, 15, 160, 45),
        _tb("合成例8", 20, 125, 70, 160),
    ]
    out = repair_colspans_by_vline_gaps(cells, binary, boxes)
    row0 = [c for c in out if int(c["row_start"]) == 0]
    assert len(row0) == 1, [(c["col_start"], c["col_end"]) for c in row0]
    assert (int(row0[0]["col_start"]), int(row0[0]["col_end"])) == (0, 4)


def main():
    test_hline_coverage_detects_line()
    test_vline_coverage_detects_line()
    test_gap_merges_header_pair()
    test_strong_hline_keeps_split()
    test_body_boundary_not_touched()
    test_vline_colspan_merges_when_body_has_line()
    test_vline_keeps_true_dual_subheaders()
    test_vline_keeps_short_standalone_alkenyl_header()
    test_wrap_rowspan_merge_substantive_cjk()
    test_vline_absorbs_empty_peers_into_right_group_header()
    test_vline_still_absorbs_empty_on_right_of_label()
    print("ok")


if __name__ == "__main__":
    main()
