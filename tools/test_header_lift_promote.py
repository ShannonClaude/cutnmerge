# -*- coding: utf-8 -*-
"""单元测试：烯键式上延 rowspan + 合成例行误落表头上提。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.structure.tsr_refine import (
    lift_misplaced_header_labels,
    promote_side_header_rowspans,
    repair_monomer_parent_spans,
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


def test_promote_ene_rowspan():
    # r1 单体父；r2 烯键式仅占子行且在单体右侧；同列 r1 空 → P96 外侧上延
    cells = [
        _cell(0, 0, 40, 40, 1, 2, 0, 0, ""),
        _cell(40, 0, 80, 40, 1, 2, 1, 1, "聚合物"),
        _cell(80, 0, 200, 20, 1, 1, 2, 5, "单体[摩尔比]"),
        _cell(200, 20, 240, 40, 2, 2, 6, 6, "具有烯键式不饱和双键基团的化合物"),
        _cell(240, 0, 280, 40, 1, 2, 7, 7, "酸当量[g/mol]"),
        _cell(80, 20, 120, 40, 2, 2, 2, 3, "四羧酸及其衍生物"),
        _cell(0, 40, 40, 60, 3, 3, 0, 0, "合成例 1"),
    ]
    out = promote_side_header_rowspans(cells)
    ene = [c for c in out if "烯键式" in str(c.get("text") or "")]
    assert len(ene) == 1
    assert int(ene[0]["row_start"]) == 1 and int(ene[0]["row_end"]) == 2
    print("ok promote_ene_rowspan")


def test_no_promote_ene_inside_monomer():
    # P98：烯键式是单体子列（col 在单体 colspan 内）→ 不得上延成外侧整列
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
    # 化学代号保留在表体
    assert any("BHPF" in str(c.get("text") or "") for c in body)
    print("ok lift_acid_equiv_from_body")


def test_right_anchor_laiyuan_not_swallowed():
    """CN111「来源于具有…」须作单体右硬锚，不可被扩进单体 colspan（P92）。"""
    # 列：0空 1聚合物 2-7单体子列 8-10来源于 11酸当量
    cells = [
        _cell(0, 0, 20, 40, 0, 1, 0, 0, ""),
        _cell(20, 0, 40, 40, 0, 1, 1, 1, "聚合物"),
        _cell(40, 0, 160, 20, 0, 0, 2, 10, "单体[mol比]"),  # 错误过宽
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


if __name__ == "__main__":
    test_promote_ene_rowspan()
    test_no_promote_ene_inside_monomer()
    test_lift_acid_equiv_from_body()
    test_right_anchor_laiyuan_not_swallowed()
    print("ALL PASS")
