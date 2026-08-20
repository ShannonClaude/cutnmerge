"""Unit tests for monomer parent span repair + chem amount merge.

用法:
    python tools/test_monomer_repair.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.structure.tsr_refine import (  # noqa: E402
    merge_stacked_chem_amount_cells,
    normalize_oversegmented_table_rows,
    repair_monomer_parent_spans,
)


def _cell(
    rs: int,
    re: int,
    cs: int,
    ce: int,
    text: str,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> Dict[str, Any]:
    return {
        "row_start": rs,
        "row_end": re,
        "col_start": cs,
        "col_end": ce,
        "row_span": re - rs + 1,
        "col_span": ce - cs + 1,
        "text": text,
        "texts": [],
        "polygon": np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64
        ),
    }


def _tb(text: str, x0: float, x1: float, y0: float, y1: float) -> Dict[str, Any]:
    return {
        "text": text,
        "score": 1.0,
        "polygon": np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64
        ),
    }


def test_monomer_without_right_metric_anchor() -> None:
    """P47 类：无含有比率锚点时，单体父格仍应扩到聚合物右侧全带。"""
    cells = [
        _cell(0, 1, 0, 0, "", x0=0, x1=40, y0=0, y1=40),
        _cell(0, 1, 1, 1, "聚合物", x0=40, x1=100, y0=0, y1=40),
        # 错位：只占中间一列
        _cell(0, 0, 3, 3, "单体 [mol比]", x0=160, x1=220, y0=0, y1=20),
        _cell(1, 1, 2, 2, "具有酸性基团的共聚成分", x0=100, x1=160, y0=20, y1=40),
        _cell(1, 1, 3, 3, "具有芳香族基团的共聚成分", x0=160, x1=220, y0=20, y1=40),
        _cell(1, 1, 4, 4, "具有脂环式基团的共聚成分", x0=220, x1=280, y0=20, y1=40),
        _cell(1, 1, 5, 5, "具有烯键式不饱和双键基团及环氧基的不饱和化合物", x0=280, x1=360, y0=20, y1=40),
        _cell(2, 2, 0, 0, "合成例6", x0=0, x1=40, y0=40, y1=60),
        _cell(2, 2, 1, 1, "丙烯酸树脂溶液(AC-1)", x0=40, x1=100, y0=40, y1=60),
        _cell(2, 2, 2, 2, "MAA(50)", x0=100, x1=160, y0=40, y1=60),
        _cell(2, 2, 3, 3, "STR(30)", x0=160, x1=220, y0=40, y1=60),
        _cell(2, 2, 4, 4, "TCDM(20)", x0=220, x1=280, y0=40, y1=60),
        _cell(2, 2, 5, 5, "GMA(20)", x0=280, x1=360, y0=40, y1=60),
    ]
    boxes = [
        _tb("聚合物", 40, 100, 0, 40),
        _tb("单体 [mol比]", 160, 220, 0, 20),
        _tb("具有酸性基团的共聚成分", 100, 160, 20, 40),
        _tb("具有芳香族基团的共聚成分", 160, 220, 20, 40),
        _tb("具有脂环式基团的共聚成分", 220, 280, 20, 40),
        _tb("具有烯键式不饱和双键基团及环氧基的不饱和化合物", 280, 360, 20, 40),
        _tb("合成例6", 0, 40, 40, 60),
        _tb("MAA(50)", 100, 160, 40, 60),
        _tb("STR(30)", 160, 220, 40, 60),
        _tb("TCDM(20)", 220, 280, 40, 60),
        _tb("GMA(20)", 280, 360, 40, 60),
    ]
    out = repair_monomer_parent_spans(cells, boxes)
    parent = next(c for c in out if "单体" in str(c.get("text") or ""))
    assert int(parent["col_start"]) == 2, parent
    assert int(parent["col_end"]) == 5, parent


def test_merge_stacked_chem_amount() -> None:
    cells = [
        _cell(0, 0, 0, 0, "STR", x0=0, x1=40, y0=0, y1=15),
        _cell(2, 2, 0, 0, "(30)", x0=0, x1=40, y0=30, y1=45),
        _cell(0, 2, 1, 1, "MAA\n(50)", x0=40, x1=80, y0=0, y1=45),
    ]
    out = merge_stacked_chem_amount_cells(cells)
    texts = {str(c.get("text") or "").replace("\n", "/") for c in out}
    assert "STR/(30)" in texts or "STR\n(30)" in {str(c.get("text") or "") for c in out}
    assert not any(str(c.get("text") or "").strip() == "(30)" for c in out)
    assert any("MAA" in str(c.get("text") or "") for c in out)


def test_normalize_oversegmented_header_and_body() -> None:
    """过切表头空行 + 合成例错位 → 压成 2 级表头 + 单行表体。"""
    cells = [
        _cell(2, 5, 1, 1, "聚合物", x0=40, x1=100, y0=20, y1=80),
        _cell(2, 2, 2, 5, "单体 [mol比]", x0=100, x1=360, y0=20, y1=40),
        _cell(5, 6, 2, 2, "具有酸性基团的共聚成分", x0=100, x1=160, y0=60, y1=80),
        _cell(5, 6, 3, 3, "具有芳香族基团的共聚成分", x0=160, x1=220, y0=60, y1=80),
        _cell(5, 6, 4, 4, "具有脂环式基团的共聚成分", x0=220, x1=280, y0=60, y1=80),
        _cell(4, 7, 5, 5, "具有烯键式不饱和双键基团及环氧基的不饱和化合物", x0=280, x1=360, y0=50, y1=90),
        # 数据在 r9，合成例错在 r10
        _cell(9, 12, 1, 1, "丙烯酸树脂溶液\n(AC-1)", x0=40, x1=100, y0=100, y1=140),
        _cell(9, 12, 2, 2, "MAA\n(50)", x0=100, x1=160, y0=100, y1=140),
        _cell(9, 12, 3, 3, "STR\n(30)", x0=160, x1=220, y0=100, y1=140),
        _cell(10, 11, 0, 0, "合成例6", x0=0, x1=40, y0=110, y1=130),
    ]
    out = normalize_oversegmented_table_rows(cells)
    by_text = {str(c.get("text") or "").split("\n")[0]: c for c in out}
    poly = by_text["聚合物"]
    mono = by_text["单体 [mol比]"]
    assert int(poly["row_start"]) == 0 and int(poly["row_end"]) == 1
    assert int(mono["row_start"]) == 0 and int(mono["row_end"]) == 0
    kids = [c for c in out if "共聚成分" in str(c.get("text") or "") or "不饱和化合物" in str(c.get("text") or "")]
    assert kids and all(int(c["row_start"]) == 1 and int(c["row_end"]) == 1 for c in kids)
    lab = by_text["合成例6"]
    ac = by_text["丙烯酸树脂溶液"]
    assert int(lab["row_start"]) == int(ac["row_start"]) == int(lab["row_end"]) == int(ac["row_end"])
    assert int(lab["row_start"]) == 2


def main() -> int:
    test_monomer_without_right_metric_anchor()
    test_merge_stacked_chem_amount()
    test_normalize_oversegmented_header_and_body()
    print("OK: monomer repair + chem merge + normalize rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
