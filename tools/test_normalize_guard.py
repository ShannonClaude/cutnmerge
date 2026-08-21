# -*- coding: utf-8 -*-
"""normalize 过压回退 + 退化兜底不得抹字。"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.structure.tsr_refine import normalize_oversegmented_table_rows
from src.core.pipeline import _is_degenerate_grid


def _cell(rs, re, cs, ce, text=""):
    return {
        "polygon": __import__("numpy").array(
            [[cs * 10, rs * 10], [ce * 10 + 9, rs * 10], [ce * 10 + 9, re * 10 + 9], [cs * 10, re * 10 + 9]],
            dtype=float,
        ),
        "row_start": rs,
        "row_end": re,
        "col_start": cs,
        "col_end": ce,
        "row_span": re - rs + 1,
        "col_span": ce - cs + 1,
        "texts": [text] if text else [],
        "text": text,
    }


def test_degenerate_skips_when_text_rich():
    cells = [_cell(i, i, 0, 0, f"合成例 {i}") for i in range(20)]
    boxes = [{"text": f"x{i}", "polygon": cells[0]["polygon"], "score": 1.0} for i in range(40)]
    assert _is_degenerate_grid(cells, boxes) is False


def test_normalize_keeps_when_would_crush():
    # 构造「聚合物/单体/合成例」触发条件，但行结构复杂到压完会丢字
    cells = []
    cells.append(_cell(0, 0, 1, 1, "聚合物"))
    cells.append(_cell(0, 0, 2, 5, "单体[摩尔比]"))
    for i in range(1, 16):
        cells.append(_cell(i, i, 0, 0, f"合成例 {i}"))
        cells.append(_cell(i, i, 1, 1, f"树脂-{i}"))
        cells.append(_cell(i, i, 2, 2, f"A({i})"))
        cells.append(_cell(i, i, 3, 3, f"B({i})"))
    before_ne = sum(1 for c in cells if c["text"])
    out = normalize_oversegmented_table_rows(cells)
    after_ne = sum(1 for c in out if str(c.get("text") or "").strip())
    # 要么改善/持平，要么回退到原文；不得大幅丢字
    assert after_ne >= int(before_ne * 0.55), (before_ne, after_ne, len(cells), len(out))


if __name__ == "__main__":
    test_degenerate_skips_when_text_rich()
    test_normalize_keeps_when_would_crush()
    print("ok")
