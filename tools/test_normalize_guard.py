# -*- coding: utf-8 -*-
"""单元测试：normalize 压行守卫（跨合成例 peer / 换行表头 / 回退）。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.structure.tsr_refine import (
    lift_misplaced_header_labels,
    normalize_oversegmented_table_rows,
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


def test_no_cross_synth_peer_collapse():
    """邻行合成例数据不得互为 peer；侧栏换行表头也不得被拉进数据行。"""
    cells = [
        _cell(0, 0, 40, 20, 0, 0, 0, 0, "聚合物"),
        _cell(40, 0, 200, 20, 0, 0, 1, 4, "单体[摩尔比]"),
        _cell(200, 0, 240, 40, 0, 1, 5, 5, "酸当量[g/mol]"),
        _cell(240, 0, 280, 40, 0, 1, 6, 6, "双键当\n量\n[g/mol]"),
        _cell(0, 40, 40, 60, 2, 2, 0, 0, "合成例 13"),
        _cell(40, 40, 80, 60, 2, 2, 1, 1, "CR-1"),
        _cell(80, 40, 120, 60, 2, 2, 2, 2, "BHPF(100)"),
        _cell(200, 40, 240, 60, 2, 2, 5, 5, "810"),
        _cell(240, 40, 280, 60, 2, 2, 6, 6, "810"),
        _cell(0, 60, 40, 80, 3, 3, 0, 0, "合成例 14"),
        _cell(40, 60, 80, 80, 3, 3, 1, 1, "CR-2"),
        _cell(80, 60, 120, 80, 3, 3, 2, 2, "BGPF(100)"),
        _cell(200, 60, 240, 80, 3, 3, 5, 5, "470"),
        _cell(240, 60, 280, 80, 3, 3, 6, 6, "470"),
        # 中段再出现换行侧栏表头（多段表）
        _cell(200, 80, 240, 120, 4, 5, 5, 5, "酸当量\n[g/mol]"),
        _cell(240, 80, 280, 120, 4, 5, 6, 6, "双键当\n量\n[g/mol]"),
        _cell(0, 120, 40, 140, 6, 6, 0, 0, "合成例 15"),
        _cell(40, 120, 80, 140, 6, 6, 1, 1, "AE-1"),
        _cell(200, 120, 240, 140, 6, 6, 5, 5, "540"),
        _cell(240, 120, 280, 140, 6, 6, 6, 6, "430"),
    ]
    out = normalize_oversegmented_table_rows(cells)
    synth = [
        str(c.get("text") or "")
        for c in out
        if "合成例" in str(c.get("text") or "")
    ]
    assert len(synth) == 3, synth
    nums = {
        re.sub(r"\s+", "", str(c.get("text") or ""))
        for c in out
        if re.fullmatch(r"\d+(?:\.\d+)?", re.sub(r"\s+", "", str(c.get("text") or "")))
    }
    for need in ("810", "470", "540", "430"):
        assert need in nums, (need, nums)
    # 右列不得被「双键当量」盖掉
    for c in out:
        t = re.sub(r"\s+", "", str(c.get("text") or ""))
        if t in ("810", "470", "430"):
            assert "双键" not in t
    print("ok test_no_cross_synth_peer_collapse")


def test_lift_keeps_numeric_body():
    cells = [
        _cell(40, 0, 80, 40, 0, 1, 1, 1, "聚合物"),
        _cell(80, 0, 200, 20, 0, 0, 2, 5, "单体[摩尔比]"),
        _cell(0, 40, 40, 60, 2, 2, 0, 0, "合成例 13"),
        _cell(40, 40, 80, 60, 2, 2, 1, 1, "CR-1"),
        _cell(240, 40, 280, 60, 2, 2, 7, 7, "810"),
        _cell(280, 40, 320, 60, 2, 2, 8, 8, "双键当量[g/mol]"),
    ]
    out = lift_misplaced_header_labels(cells)
    assert any(
        re.fullmatch(r"810", re.sub(r"\s+", "", str(c.get("text") or "")))
        for c in out
    )
    assert any("双键当量" in re.sub(r"\s+", "", str(c.get("text") or "")) for c in out)
    # 810 仍在合成例行
    body_nums = [
        re.sub(r"\s+", "", str(c.get("text") or ""))
        for c in out
        if int(c["row_start"]) >= 2
        and re.fullmatch(r"\d+", re.sub(r"\s+", "", str(c.get("text") or "")))
    ]
    assert "810" in body_nums
    print("ok test_lift_keeps_numeric_body")


def test_align_diamine_vs_endcap_left_wall():
    """左对齐跨列表头：下一标签左缘为墙；SiDA 归二胺，封端剂仅 MAP 一列。

    模拟 TSR 误成二胺2+封端剂2；对齐后应纠正为二胺3+封端剂1。
    """
    from src.structure.tsr_refine import _align_monomer_children_to_body

    # col2-3 酸 | col4-6 胺 | col7 封端
    col_seps = [0, 40, 80, 120, 200, 280, 360, 420, 480]
    kids = [
        _cell(80, 20, 200, 40, 1, 1, 2, 3, "四羧酸及其衍生物"),
        # 故意错误的初始 colspan（2+2），多边形仅盖住左对齐文案区
        _cell(200, 20, 360, 40, 1, 1, 4, 5, "二胺及其衍生物"),
        _cell(360, 20, 480, 40, 1, 1, 6, 7, "封端剂"),
    ]
    kids[0]["polygon"] = _cell(80, 20, 140, 40, 1, 1, 2, 3)["polygon"]
    kids[1]["polygon"] = _cell(200, 20, 280, 40, 1, 1, 4, 5)["polygon"]
    kids[2]["polygon"] = _cell(420, 20, 460, 40, 1, 1, 7, 7)["polygon"]
    bodies = [
        _cell(80, 40, 120, 60, 2, 2, 2, 2, "ODPA(100)"),
        _cell(120, 40, 200, 60, 2, 2, 3, 3, "-"),
        _cell(200, 40, 280, 60, 2, 2, 4, 4, "BAHF(85)"),
        _cell(280, 40, 360, 60, 2, 2, 5, 5, "-"),
        _cell(360, 40, 420, 60, 2, 2, 6, 6, "SiDA(5)"),
        _cell(420, 40, 480, 60, 2, 2, 7, 7, "MAP(20)"),
    ]
    n = _align_monomer_children_to_body(
        kids, bodies, col_seps, band_lo=2, band_hi=7
    )
    acid = next(c for c in kids if "四羧酸" in str(c.get("text") or ""))
    diam = next(c for c in kids if "二胺" in str(c.get("text") or ""))
    cap = next(c for c in kids if "封端剂" in str(c.get("text") or ""))
    assert (int(acid["col_start"]), int(acid["col_end"])) == (2, 3), acid
    assert (int(diam["col_start"]), int(diam["col_end"])) == (4, 6), diam
    assert (int(cap["col_start"]), int(cap["col_end"])) == (7, 7), cap
    assert n >= 1
    print("ok test_align_diamine_vs_endcap_left_wall", "changed=", n)


def test_align_trifunctional_silane_trim_dash():
    """P93：左墙会把四官能前短横列划给三官能；收尾应让出 → 3+2+1。"""
    from src.structure.tsr_refine import _align_monomer_children_to_body

    # c2 MeTMS | c3 PhTMS | c4 cyEpo | c5 - | c6 TMOS | c7 -
    col_seps = [0, 40, 80, 200, 320, 440, 520, 640, 760]
    kids = [
        # TSR 误把三官能扩到含短横列
        _cell(80, 20, 520, 40, 1, 1, 2, 5, "三官能有机硅烷"),
        _cell(520, 20, 640, 40, 1, 1, 6, 6, "四官能有机硅烷"),
        _cell(640, 20, 760, 40, 1, 1, 7, 7, "二官能有机硅烷"),
    ]
    # 文案左对齐：三官能只印在前三列区域，四官能文案从 TMOS 列起
    kids[0]["polygon"] = _cell(80, 20, 280, 40, 1, 1, 2, 4)["polygon"]
    kids[1]["polygon"] = _cell(560, 20, 620, 40, 1, 1, 6, 6)["polygon"]
    kids[2]["polygon"] = _cell(680, 20, 740, 40, 1, 1, 7, 7)["polygon"]
    bodies = [
        _cell(80, 40, 200, 60, 2, 2, 2, 2, "MeTMS(30)"),
        _cell(200, 40, 320, 60, 2, 2, 3, 3, "PhTMS(50)"),
        _cell(320, 40, 440, 60, 2, 2, 4, 4, "cyEpoTMS(10)"),
        _cell(440, 40, 520, 60, 2, 2, 5, 5, "-"),
        _cell(520, 40, 640, 60, 2, 2, 6, 6, "TMOS(10)"),
        _cell(640, 40, 760, 60, 2, 2, 7, 7, "-"),
    ]
    n = _align_monomer_children_to_body(
        kids, bodies, col_seps, band_lo=2, band_hi=7
    )
    tri = next(c for c in kids if "三官能" in str(c.get("text") or ""))
    tetra = next(c for c in kids if "四官能" in str(c.get("text") or ""))
    di = next(c for c in kids if "二官能" in str(c.get("text") or ""))
    assert (int(tri["col_start"]), int(tri["col_end"])) == (2, 4), tri
    assert (int(tetra["col_start"]), int(tetra["col_end"])) == (5, 6), tetra
    assert (int(di["col_start"]), int(di["col_end"])) == (7, 7), di
    assert n >= 1
    print("ok test_align_trifunctional_silane_trim_dash", "changed=", n)


def test_align_trifunctional_keeps_chem_fourth_col():
    """P97：第四列有 AcrTMS 实质内容时，不得因空行短横被让给四官能。"""
    from src.structure.tsr_refine import _align_monomer_children_to_body

    col_seps = [0, 40, 80, 200, 320, 440, 560, 680, 800]
    kids = [
        _cell(80, 20, 560, 40, 1, 1, 2, 5, "三官能有机硅烷"),
        _cell(560, 20, 680, 40, 1, 1, 6, 6, "四官能有机硅烷"),
        _cell(680, 20, 800, 40, 1, 1, 7, 7, "二官能有机硅烷"),
    ]
    kids[0]["polygon"] = _cell(80, 20, 300, 40, 1, 1, 2, 4)["polygon"]
    kids[1]["polygon"] = _cell(600, 20, 660, 40, 1, 1, 6, 6)["polygon"]
    kids[2]["polygon"] = _cell(720, 20, 780, 40, 1, 1, 7, 7)["polygon"]
    bodies = [
        _cell(80, 40, 200, 60, 2, 2, 2, 2, "MeTMS(35)"),
        _cell(200, 40, 320, 60, 2, 2, 3, 3, "PhTMS(50)"),
        _cell(320, 40, 440, 60, 2, 2, 4, 4, "TMSSucA(10)"),
        _cell(440, 40, 560, 60, 2, 2, 5, 5, "AcrTMS(20)"),
        _cell(560, 40, 680, 60, 2, 2, 6, 6, "TMOS(5)"),
        _cell(680, 40, 800, 60, 2, 2, 7, 7, "-"),
        # 另一行同列为空，不得据此把 AcrTMS 列让走
        _cell(440, 60, 560, 80, 3, 3, 5, 5, ""),
    ]
    _align_monomer_children_to_body(
        kids, bodies, col_seps, band_lo=2, band_hi=7
    )
    tri = next(c for c in kids if "三官能" in str(c.get("text") or ""))
    tetra = next(c for c in kids if "四官能" in str(c.get("text") or ""))
    assert (int(tri["col_start"]), int(tri["col_end"])) == (2, 5), tri
    assert (int(tetra["col_start"]), int(tetra["col_end"])) == (6, 6), tetra
    print("ok test_align_trifunctional_keeps_chem_fourth_col")


if __name__ == "__main__":
    test_no_cross_synth_peer_collapse()
    test_lift_keeps_numeric_body()
    test_align_diamine_vs_endcap_left_wall()
    test_align_trifunctional_silane_trim_dash()
    test_align_trifunctional_keeps_chem_fourth_col()
    print("ALL PASS")
