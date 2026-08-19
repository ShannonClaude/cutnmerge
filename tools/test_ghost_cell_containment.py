"""P100：被大格包住的空幽灵格不得画出夹在序号与「分散液」之间的空 <td>。

用法:
    python tools/test_ghost_cell_containment.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.output.html_formatter import cells_to_html_table, _resolve_logic_overlaps
from src.structure.tsr_refine import dedupe_overlapping_cells


def _cell(rs, re_, cs, ce, text, x1, x2, y1=0.0, y2=30.0):
    return {
        "row_start": rs,
        "row_end": re_,
        "col_start": cs,
        "col_end": ce,
        "row_span": re_ - rs + 1,
        "col_span": ce - cs + 1,
        "text": text,
        "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
    }


def _p100_header_cells():
    """TSR 实测几何：大格 rows1-2 cols1-3 + 右上角 24x18 幽灵格。"""
    return [
        _cell(1, 2, 0, 0, "", 0, 118, 35, 99),
        _cell(1, 2, 1, 3, "分散液", 118, 260, 35, 99),
        _cell(1, 1, 3, 3, "", 236, 260, 35, 53),
        _cell(1, 1, 4, 8, "组成[质量%]", 260, 700, 35, 53),
        _cell(2, 2, 4, 6, "着色剂", 260, 520, 53, 99),
        _cell(2, 2, 7, 7, "（A1）第一树脂", 520, 610, 53, 99),
        _cell(2, 2, 8, 8, "（E）分散剂", 610, 700, 53, 99),
        _cell(1, 2, 9, 9, "颜料分散液中的颜料的数均粒径 [nm]", 700, 820, 35, 99),
        _cell(3, 3, 0, 0, "制备例1", 0, 118, 99, 130),
        _cell(3, 3, 1, 3, "颜料分散液 (Bk-1)", 118, 260, 99, 130),
        _cell(3, 3, 4, 4, "Bk-S0100CF (75)", 260, 380, 99, 130),
        _cell(3, 3, 7, 7, "", 520, 610, 99, 130),
        _cell(3, 3, 8, 8, "S-20000 (25)", 610, 700, 99, 130),
        _cell(3, 3, 9, 9, "100", 700, 820, 99, 130),
    ]


def test_dedupe_drops_p100_ghost():
    large = _cell(1, 2, 1, 3, "分散液", 118, 260, 35, 99)
    ghost = _cell(1, 1, 3, 3, "", 236, 260, 35, 53)
    out = dedupe_overlapping_cells([large, ghost])
    assert len(out) == 1, out
    kept = out[0]
    assert kept["text"] == "分散液"
    assert (kept["row_start"], kept["row_end"]) == (1, 2)
    assert (kept["col_start"], kept["col_end"]) == (1, 3)


def test_dedupe_keeps_text_subheader_on_next_row():
    parent = _cell(0, 0, 2, 7, "单体[摩尔比]", 120, 520, 0, 20)
    child = _cell(1, 1, 2, 5, "具有芳香族基团及环氧基的化合物", 120, 360, 20, 40)
    out = dedupe_overlapping_cells([parent, child])
    texts = {str(c.get("text") or "") for c in out}
    assert "单体[摩尔比]" in texts
    assert "具有芳香族基团及环氧基的化合物" in texts


def test_dedupe_keeps_text_child_inside_parent():
    parent = _cell(0, 1, 2, 5, "芳香族环氧", 120, 360, 0, 40)
    child = _cell(1, 1, 2, 5, "NC-7000L", 120, 360, 20, 40)
    out = dedupe_overlapping_cells([parent, child])
    texts = {str(c.get("text") or "") for c in out}
    assert "芳香族环氧" in texts
    assert "NC-7000L" in texts


def test_resolve_evicts_empty_fragment():
    large = _cell(1, 2, 1, 3, "分散液", 118, 260, 35, 99)
    ghost = _cell(1, 1, 3, 3, "", 236, 260, 35, 53)
    out = _resolve_logic_overlaps([large, ghost])
    assert len(out) == 1, out
    assert out[0]["text"] == "分散液"
    assert (out[0]["row_start"], out[0]["row_end"]) == (1, 2)
    assert (out[0]["col_start"], out[0]["col_end"]) == (1, 3)


def test_html_no_bare_td_between_index_and_fensan():
    html = cells_to_html_table(_p100_header_cells())
    assert "分散液" in html
    assert "制备例1" in html
    assert "颜料分散液" in html
    ghost_between = re.search(
        r'<td rowspan="2"></td>\s*<td></td>\s*<td[^>]*>分散液',
        html,
    )
    assert ghost_between is None, html
    assert re.search(r'rowspan="2"[^>]*>分散液', html), html
    first_tr = re.search(r"<tr>(.*?)</tr>", html, re.S)
    assert first_tr is not None
    tds = re.findall(r"<td[^>]*>(.*?)</td>", first_tr.group(1), re.S)
    texts = [re.sub(r"<[^>]+>", "", t).strip() for t in tds]
    assert "分散液" in texts
    idx = texts.index("分散液")
    assert idx == 1, texts


def main() -> int:
    test_dedupe_drops_p100_ghost()
    test_dedupe_keeps_text_subheader_on_next_row()
    test_dedupe_keeps_text_child_inside_parent()
    test_resolve_evicts_empty_fragment()
    test_html_no_bare_td_between_index_and_fensan()
    print("OK: ghost cell containment tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
