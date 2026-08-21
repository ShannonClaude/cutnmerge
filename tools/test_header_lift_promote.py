# -*- coding: utf-8 -*-
"""单元测试：侧栏上延、烯键式子列、来源于右锚、粘连拆分。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.structure.tsr_refine import (
    demote_ene_inside_monomer_band,
    lift_misplaced_header_labels,
    promote_side_header_rowspans,
    refill_short_side_headers_from_ocr,
    refill_truncated_monomer_child_headers_from_ocr,
    repair_monomer_parent_spans,
    split_glued_side_headers,
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


def test_no_promote_ene_child_row():
    """子表头行烯键式（右硬锚左侧）不得上延成外侧整列。"""
    cells = [
        _cell(0, 0, 40, 40, 1, 2, 0, 0, ""),
        _cell(40, 0, 80, 40, 1, 2, 1, 1, "聚合物"),
        _cell(80, 0, 240, 20, 1, 1, 2, 6, "单体[摩尔比]"),
        _cell(200, 20, 240, 40, 2, 2, 6, 6, "具有烯键式不饱和双键基团的化合物"),
        _cell(240, 0, 280, 40, 1, 2, 7, 7, "酸当量[g/mol]"),
        _cell(80, 20, 120, 40, 2, 2, 2, 3, "四羧酸及其衍生物"),
        _cell(0, 40, 40, 60, 3, 3, 0, 0, "合成例 1"),
    ]
    out = promote_side_header_rowspans(cells)
    ene = [c for c in out if "烯键式" in str(c.get("text") or "")]
    assert len(ene) == 1
    assert int(ene[0]["row_start"]) == 2 and int(ene[0]["row_end"]) == 2
    print("ok test_no_promote_ene_child_row")


def test_demote_ene_into_monomer():
    """已跨行的烯键式在右锚左侧 → 收回子行并扩单体。"""
    cells = [
        _cell(40, 0, 80, 40, 0, 1, 1, 1, "聚合物"),
        _cell(80, 0, 200, 20, 0, 0, 2, 5, "单体[摩尔比]"),
        _cell(80, 20, 120, 40, 1, 1, 2, 3, "四羧酸"),
        _cell(120, 20, 160, 40, 1, 1, 4, 4, "二胺"),
        _cell(160, 20, 180, 40, 1, 1, 5, 5, "封端剂"),
        _cell(200, 0, 240, 40, 0, 1, 6, 6, "具有烯键式不饱和双键基团的化合物"),
        _cell(240, 0, 280, 40, 0, 1, 7, 7, "来自具有氟原子的含有比率[mol%]"),
    ]
    out = demote_ene_inside_monomer_band(cells)
    ene = [c for c in out if "烯键式" in str(c.get("text") or "")][0]
    parent = [c for c in out if "单体" in str(c.get("text") or "")][0]
    assert int(ene["row_start"]) == 1 and int(ene["row_end"]) == 1
    assert int(parent["col_end"]) >= 6
    print("ok test_demote_ene_into_monomer")


def test_no_promote_ene_inside_monomer():
    cells = [
        _cell(40, 0, 80, 40, 0, 1, 1, 1, "聚合物"),
        _cell(80, 0, 280, 20, 0, 0, 2, 7, "单体[摩尔比]"),
        _cell(80, 20, 120, 40, 1, 1, 2, 2, "羟基化合物"),
        _cell(120, 20, 160, 40, 1, 1, 3, 3, "环氧化合物"),
        _cell(160, 20, 200, 40, 1, 1, 4, 4, "四羧酸"),
        _cell(200, 20, 240, 40, 1, 1, 5, 5, "封端剂"),
        _cell(240, 20, 280, 40, 1, 1, 6, 6, "具有烯键式不饱和双键基团及环氧基的不饱和化合物"),
        _cell(280, 20, 320, 40, 1, 1, 7, 7, "具有烯键式不饱和双键基团的不饱和羧酸"),
        _cell(320, 0, 360, 40, 0, 1, 8, 8, "含有比率[mol%]"),
        _cell(0, 40, 40, 60, 2, 2, 0, 0, "合成例 13"),
    ]
    out = promote_side_header_rowspans(cells)
    enes = [c for c in out if "烯键式" in str(c.get("text") or "")]
    assert len(enes) == 2
    assert all(int(c["row_start"]) == 1 and int(c["row_end"]) == 1 for c in enes)
    print("ok test_no_promote_ene_inside_monomer")


def test_lift_acid_equiv_from_body():
    cells = [
        _cell(40, 0, 80, 40, 0, 1, 1, 1, "聚合物"),
        _cell(80, 0, 200, 20, 0, 0, 2, 5, "单体[摩尔比]"),
        _cell(200, 20, 240, 40, 1, 1, 6, 6, "来自具有芳香族基团的比率[mol%]"),
        _cell(0, 40, 40, 60, 2, 2, 0, 0, "合成例 13"),
        _cell(40, 40, 80, 60, 2, 2, 1, 1, "CR-1"),
        _cell(80, 40, 120, 60, 2, 2, 2, 2, "BHPF(100)"),
        _cell(240, 40, 280, 60, 2, 2, 7, 7, "酸当量[g/mol]"),
        _cell(280, 40, 320, 60, 2, 2, 8, 8, "双键当量[g/mol]"),
        _cell(120, 40, 160, 60, 2, 2, 3, 3, "四羧酸二酐四羧酸"),
    ]
    out = lift_misplaced_header_labels(cells)
    body = [c for c in out if int(c["row_start"]) == 2]
    assert all(
        "酸当量" not in str(c.get("text") or "")
        and "双键当量" not in str(c.get("text") or "")
        and "四羧酸二酐" not in str(c.get("text") or "")
        for c in body
    )
    hdr = " ".join(str(c.get("text") or "") for c in out if int(c["row_start"]) <= 1)
    assert "酸当量" in hdr and "双键当量" in hdr and "四羧酸二酐" in hdr
    assert any("BHPF" in str(c.get("text") or "") for c in body)
    print("ok lift_acid_equiv_from_body")


def test_right_anchor_laiyuan_not_swallowed():
    cells = [
        _cell(0, 0, 20, 40, 0, 1, 0, 0, ""),
        _cell(20, 0, 40, 40, 0, 1, 1, 1, "聚合物"),
        _cell(40, 0, 160, 20, 0, 0, 2, 10, "单体[mol比]"),
        _cell(40, 20, 60, 40, 1, 1, 2, 3, "四羧酸及其衍生物"),
        _cell(60, 20, 100, 40, 1, 1, 4, 6, "二胺及其衍生物"),
        _cell(100, 20, 120, 40, 1, 1, 7, 7, "封端剂"),
        _cell(120, 20, 140, 40, 1, 1, 8, 8, "来源于具有氟原子的单体的结构单元在全部结构单元中所占的比率[mol%]"),
        _cell(140, 20, 160, 40, 1, 1, 9, 9, "来源于具有氟原子的单体的结构单元在来源于全部羧酸衍生物的结构单元中所占的比率[mol%]"),
        _cell(160, 20, 180, 40, 1, 1, 10, 10, "来源于具有氟原子的单体的结构单元在来源于全部胺衍生物的结构单元中所占的比率[mol%]"),
        _cell(180, 0, 200, 40, 0, 1, 11, 11, "酸当量[g/mol]"),
        _cell(0, 40, 20, 60, 2, 2, 0, 0, "合成例1"),
        _cell(20, 40, 40, 60, 2, 2, 1, 1, "PI-1"),
        _cell(40, 40, 50, 60, 2, 2, 2, 2, "ODPA(100)"),
        _cell(50, 40, 60, 60, 2, 2, 3, 3, "-"),
        _cell(60, 40, 70, 60, 2, 2, 4, 4, "BAHF(85)"),
        _cell(70, 40, 80, 60, 2, 2, 5, 5, "-"),
        _cell(80, 40, 90, 60, 2, 2, 6, 6, "SiDA(5)"),
        _cell(90, 40, 100, 60, 2, 2, 7, 7, "MAP(20)"),
        _cell(100, 40, 120, 60, 2, 2, 8, 8, "40.5"),
        _cell(120, 40, 140, 60, 2, 2, 9, 9, "0.0"),
        _cell(140, 40, 160, 60, 2, 2, 10, 10, "77.3"),
        _cell(160, 40, 180, 60, 2, 2, 11, 11, "350"),
    ]
    out = repair_monomer_parent_spans(cells)
    parent = [c for c in out if "单体" in str(c.get("text") or "")][0]
    assert int(parent["col_start"]) == 2 and int(parent["col_end"]) == 7, (
        parent["col_start"],
        parent["col_end"],
    )
    print("ok test_right_anchor_laiyuan_not_swallowed")


def test_split_glued_acid_equiv():
    cells = [
        _cell(0, 0, 40, 40, 0, 1, 0, 0, "聚合物"),
        _cell(40, 0, 120, 20, 0, 0, 1, 3, "单体[mol比]"),
        _cell(
            120,
            0,
            160,
            40,
            0,
            1,
            4,
            4,
            "来源于具有氟原子的结构单元中所占的比率[mol%]酸当量[g/mol]",
        ),
        _cell(160, 0, 200, 40, 0, 1, 5, 5, ""),
        _cell(200, 0, 240, 40, 0, 1, 6, 6, "双键当量[g/mol]"),
    ]
    out = split_glued_side_headers(cells)
    glued = [c for c in out if int(c["col_start"]) == 4][0]
    empty = [c for c in out if int(c["col_start"]) == 5][0]
    assert "酸当量" not in re.sub(r"\s+", "", str(glued.get("text") or ""))
    assert "酸当量" in str(empty.get("text") or "")
    assert "来源于" in str(glued.get("text") or "")
    print("ok test_split_glued_acid_equiv")


def test_split_glued_acid_equiv_hole():
    """右侧无空格对象、与双键当量之间有占位洞 → 插入列并拆出酸当量。"""
    cells = [
        _cell(0, 0, 40, 40, 0, 1, 0, 0, "聚合物"),
        _cell(40, 0, 120, 20, 0, 0, 1, 3, "单体[mol比]"),
        _cell(
            120,
            0,
            160,
            40,
            0,
            1,
            4,
            4,
            "来源于具有氟原子的结构单元中所占的比率[mol%]酸当量[g/mol]",
        ),
        # col 5 空洞：无 cell
        _cell(200, 0, 240, 40, 0, 1, 6, 6, "双键当量[g/mol]"),
    ]
    out = split_glued_side_headers(cells)
    acid = [c for c in out if "酸当量" in str(c.get("text") or "")]
    assert len(acid) == 1
    assert int(acid[0]["col_start"]) == 5
    glued = [c for c in out if int(c["col_start"]) == 4][0]
    assert "酸当量" not in re.sub(r"\s+", "", str(glued.get("text") or ""))
    print("ok test_split_glued_acid_equiv_hole")


def test_refill_short_side_mo1():
    """短「来源于具有」应并入同列 [mo1%]（OCR 常把 mol 识成 mo1）。"""
    cells = [
        _cell(100, 0, 160, 80, 0, 1, 4, 4, "来源于具有"),
        _cell(160, 0, 200, 80, 0, 1, 5, 5, "酸当量[g/mol]"),
    ]
    boxes = [
        {"text": "来源于具有", "polygon": np.array([[110, 10], [150, 10], [150, 30], [110, 30]], dtype=np.float64), "score": 1.0},
        {"text": "[mo1%]", "polygon": np.array([[120, 50], [150, 50], [150, 70], [120, 70]], dtype=np.float64), "score": 1.0},
        {"text": "酸当量", "polygon": np.array([[170, 20], [195, 20], [195, 40], [170, 40]], dtype=np.float64), "score": 1.0},
    ]
    out = refill_short_side_headers_from_ocr(cells, boxes)
    t = re.sub(r"\s+", "", str(out[0].get("text") or ""))
    assert "来源于具有" in t
    assert "mo1" in t.lower() or "mol" in t.lower()
    assert "酸当量" not in t
    print("ok test_refill_short_side_mo1")


def test_refill_truncated_copoly_children():
    """过窄子头格：邻列中点墙内 OCR 补全共聚成分/化合物。"""
    cells = [
        _cell(0, 0, 40, 40, 0, 1, 0, 0, "聚合物"),
        _cell(40, 0, 400, 20, 0, 0, 1, 4, "单体[mol比]"),
        _cell(100, 20, 110, 40, 1, 1, 1, 1, "具有酸性基团"),
        _cell(200, 20, 210, 40, 1, 1, 2, 2, "具有芳香族基团的共聚成分"),
        _cell(300, 20, 310, 40, 1, 1, 3, 3, "具有脂环式基团"),
        _cell(380, 20, 400, 40, 1, 1, 4, 4, "具有烯键式不饱和"),
    ]
    boxes = [
        {"text": "具有酸性基团", "polygon": np.array([[50, 22], [90, 22], [90, 30], [50, 30]], dtype=np.float64), "score": 1.0},
        {"text": "的共聚成分", "polygon": np.array([[55, 32], [95, 32], [95, 38], [55, 38]], dtype=np.float64), "score": 1.0},
        {"text": "具有脂环式基团", "polygon": np.array([[250, 22], [290, 22], [290, 30], [250, 30]], dtype=np.float64), "score": 1.0},
        {"text": "的共聚成分", "polygon": np.array([[255, 32], [295, 32], [295, 38], [255, 38]], dtype=np.float64), "score": 1.0},
        {"text": "具有烯键式不饱和", "polygon": np.array([[330, 22], [370, 22], [370, 28], [330, 28]], dtype=np.float64), "score": 1.0},
        {"text": "双键基团及环氧基", "polygon": np.array([[335, 29], [375, 29], [375, 34], [335, 34]], dtype=np.float64), "score": 1.0},
        {"text": "的不饱和化合物", "polygon": np.array([[340, 35], [380, 35], [380, 39], [340, 39]], dtype=np.float64), "score": 1.0},
    ]
    out = refill_truncated_monomer_child_headers_from_ocr(cells, boxes)
    by_col = {int(c["col_start"]): re.sub(r"\s+", "", str(c.get("text") or "")) for c in out}
    assert "共聚成分" in by_col[1]
    assert "共聚成分" in by_col[3]
    assert "化合物" in by_col[4]
    assert "脂环" in by_col[3] and "烯键" not in by_col[3]
    print("ok test_refill_truncated_copoly_children")


if __name__ == "__main__":
    import re

    test_no_promote_ene_child_row()
    test_demote_ene_into_monomer()
    test_no_promote_ene_inside_monomer()
    test_lift_acid_equiv_from_body()
    test_right_anchor_laiyuan_not_swallowed()
    test_split_glued_acid_equiv()
    test_split_glued_acid_equiv_hole()
    test_refill_short_side_mo1()
    test_refill_truncated_copoly_children()
    print("ALL PASS")
