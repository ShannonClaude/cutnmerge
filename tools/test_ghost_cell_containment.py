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
from src.structure.tsr_refine import (
    dedupe_overlapping_cells,
    logic_conflict_ratio,
    merge_ghost_columns,
)


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


def _tb(text, x1, x2, y1, y2):
    return {
        "text": text,
        "score": 0.99,
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


def _p100_after_light_like_cells():
    """light 去幽灵格后的拓扑：分散液仍跨到空列 3，组成在 4-8。"""
    return [
        _cell(1, 2, 0, 0, "", 0, 100, 35, 99),
        _cell(1, 2, 1, 3, "分散液", 100, 250, 35, 99),
        _cell(1, 1, 4, 8, "组成[质量%]", 250, 700, 35, 53),
        _cell(2, 2, 4, 6, "着色剂", 250, 520, 53, 99),
        _cell(2, 2, 7, 7, "（A1）第一树脂", 520, 610, 53, 99),
        _cell(2, 2, 8, 8, "（E）分散剂", 610, 700, 53, 99),
        _cell(1, 2, 9, 9, "颜料分散液中的颜料的数均粒径 [nm]", 700, 820, 35, 99),
        _cell(3, 3, 0, 0, "制备例1", 0, 100, 99, 130),
        _cell(3, 3, 1, 1, "颜料分散液 (Bk-1)", 100, 200, 99, 130),
        _cell(3, 3, 4, 4, "Bk-S0100CF (75)", 250, 380, 99, 130),
        _cell(3, 3, 7, 7, "聚酰亚胺", 520, 610, 99, 130),
        _cell(3, 3, 8, 8, "S-20000", 610, 700, 99, 130),
        _cell(3, 3, 9, 9, "100", 700, 820, 99, 130),
    ]


def _p100_boxes_for_ghost_merge():
    """列 3 无 OCR 命中（hits=0），会被标成幽灵列。"""
    return [
        _tb("分散液", 120, 200, 40, 90),
        _tb("组成[质量%]", 300, 500, 35, 50),
        _tb("着色剂", 280, 400, 55, 95),
        _tb("（A1）第一树脂", 530, 600, 55, 95),
        _tb("（E）分散剂", 620, 690, 55, 95),
        _tb("粒径", 710, 800, 40, 90),
        _tb("制备例1", 10, 90, 100, 125),
        _tb("颜料分散液", 110, 190, 100, 125),
        _tb("Bk-S0100CF", 260, 370, 100, 125),
        _tb("聚酰亚胺", 530, 600, 100, 125),
        _tb("S-20000", 620, 690, 100, 125),
        _tb("100", 710, 800, 100, 125),
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


def test_merge_ghost_does_not_crush_p100_subheader():
    """空列 3 并入右侧时，不得把「分散液」扩进「组成」列。"""
    cells = _p100_after_light_like_cells()
    boxes = _p100_boxes_for_ghost_merge()
    before = logic_conflict_ratio(cells)
    out = merge_ghost_columns([dict(c) for c in cells], boxes)
    after = logic_conflict_ratio(out)
    assert after <= before + 1e-9, (before, after, out)

    compose = next(c for c in out if "组成" in str(c.get("text") or ""))
    fensan = next(c for c in out if str(c.get("text") or "") == "分散液")
    assert int(compose["row_start"]) == int(compose["row_end"]) == 1, compose
    assert int(fensan["col_end"]) < int(compose["col_start"]), (fensan, compose)

    sub_texts = {str(c.get("text") or "") for c in out if int(c["row_start"]) == 2}
    assert "着色剂" in sub_texts
    assert any("第一树脂" in t for t in sub_texts)
    assert any("分散剂" in t for t in sub_texts)

    html = cells_to_html_table(out)
    assert re.search(r">组成\[质量%\]<", html), html
    assert "组成[质量%]<br>" not in html
    assert re.search(r">着色剂<", html), html
    assert re.search(r'rowspan="2"[^>]*>分散液', html), html


def main() -> int:
    test_dedupe_drops_p100_ghost()
    test_dedupe_keeps_text_subheader_on_next_row()
    test_dedupe_keeps_text_child_inside_parent()
    test_resolve_evicts_empty_fragment()
    test_html_no_bare_td_between_index_and_fensan()
    test_merge_ghost_does_not_crush_p100_subheader()
    print("OK: ghost cell containment tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
