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


def test_align_diamine_vs_endcap_prefers_covering():
    """P96：二胺已覆盖 SiDA 列时，不得被物理中心对齐偷给封端剂。"""
    from src.structure.tsr_refine import _align_monomer_children_to_body

    kids = [
        _cell(40, 20, 120, 40, 1, 1, 2, 3, "四羧酸及其衍生物"),
        _cell(120, 20, 200, 40, 1, 1, 4, 6, "二胺及其衍生物"),  # 逻辑已盖住 SiDA 列
        _cell(240, 20, 280, 40, 1, 1, 7, 7, "封端剂"),  # polygon 偏右
    ]
    # 故意把二胺 polygon 收窄到只盖 BAHF，模拟 TSR 物理框偏窄
    kids[1]["polygon"] = _cell(120, 20, 160, 40, 1, 1, 4, 6)["polygon"]
    bodies = [
        _cell(40, 40, 80, 60, 2, 2, 2, 2, "ODPA(100)"),
        _cell(120, 40, 160, 60, 2, 2, 4, 4, "BAHF(85)"),
        _cell(160, 40, 200, 60, 2, 2, 5, 5, "-"),
        _cell(200, 40, 240, 60, 2, 2, 6, 6, "SiDA(5)"),
        _cell(240, 40, 280, 60, 2, 2, 7, 7, "MAP(20)"),
    ]
    col_seps = [0, 40, 80, 120, 160, 200, 240, 280, 320]
    n = _align_monomer_children_to_body(
        kids, bodies, col_seps, band_lo=2, band_hi=7
    )
    diam = next(c for c in kids if "二胺" in str(c.get("text") or ""))
    cap = next(c for c in kids if "封端剂" in str(c.get("text") or ""))
    assert int(diam["col_start"]) == 4 and int(diam["col_end"]) == 6, diam
    assert int(cap["col_start"]) == 7 and int(cap["col_end"]) == 7, cap
    print("ok test_align_diamine_vs_endcap_prefers_covering", "changed=", n)


if __name__ == "__main__":
    test_no_cross_synth_peer_collapse()
    test_lift_keeps_numeric_body()
    test_align_diamine_vs_endcap_prefers_covering()
    print("ALL PASS")
