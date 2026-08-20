"""Regression: empty stub beside rowspan must not invent a mid-line."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.output.html_formatter import cells_to_html_table  # noqa: E402


def _cell(
    rs: int,
    re: int,
    cs: int,
    ce: int,
    text: str,
) -> Dict[str, Any]:
    return {
        "row_start": rs,
        "row_end": re,
        "col_start": cs,
        "col_end": ce,
        "row_span": re - rs + 1,
        "col_span": ce - cs + 1,
        "text": text,
        "polygon": [
            [cs * 40, rs * 20],
            [(ce + 1) * 40, rs * 20],
            [(ce + 1) * 40, (re + 1) * 20],
            [cs * 40, (re + 1) * 20],
        ],
    }


def test_empty_stub_matches_polymer_rowspan() -> None:
    """P46：聚合物 rowspan=2 左侧空角应是一个 rowspan=2 空 td，而非两行空 td 中线。"""
    cells = [
        _cell(0, 1, 1, 1, "聚合物"),
        _cell(0, 0, 2, 5, "单体 [mol比]"),
        _cell(1, 1, 2, 2, "二羧酸"),
        _cell(1, 1, 3, 4, "双氨基酚"),
        _cell(1, 1, 5, 5, "封端剂"),
        _cell(2, 2, 0, 0, "合成例3"),
        _cell(2, 2, 1, 1, "PBO-1"),
        _cell(2, 2, 2, 2, "BFE"),
        _cell(2, 2, 3, 3, "BAHF"),
        _cell(2, 2, 4, 4, "SiDA"),
        _cell(2, 2, 5, 5, "NA"),
    ]
    html = cells_to_html_table(cells)
    # first data cell of header row: empty stub with rowspan=2, then 聚合物
    assert re.search(
        r'<tr>\s*<td rowspan="2"></td>\s*<td rowspan="2">聚合物</td>',
        html,
        re.DOTALL,
    ), html
    # second header row must NOT start with another empty td (would invent mid-line)
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)
    assert len(rows) >= 2
    assert not re.match(r"\s*<td></td>", rows[1]), rows[1]


def test_split_empty_when_right_neighbor_not_rowspan() -> None:
    """右侧是两行独立格时，左侧空位仍保持两行空 td（保留应有的横线）。"""
    cells = [
        _cell(0, 0, 1, 1, "上"),
        _cell(1, 1, 1, 1, "下"),
        _cell(0, 0, 2, 2, "A"),
        _cell(1, 1, 2, 2, "B"),
        _cell(2, 2, 0, 0, "合成例1"),
        _cell(2, 2, 1, 1, "x"),
        _cell(2, 2, 2, 2, "y"),
    ]
    html = cells_to_html_table(cells)
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)
    assert re.match(r"\s*<td></td>", rows[0]), rows[0]
    assert re.match(r"\s*<td></td>", rows[1]), rows[1]


if __name__ == "__main__":
    test_empty_stub_matches_polymer_rowspan()
    test_split_empty_when_right_neighbor_not_rowspan()
    print("ok")
