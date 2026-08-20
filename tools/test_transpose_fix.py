# -*- coding: utf-8 -*-
"""侧躺/行列转置检测与损坏 HTML 判定。

用法:
    python tools/test_transpose_fix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.output.html_formatter import html_output_looks_broken  # noqa: E402
from src.structure.transpose_fix import (  # noqa: E402
    detect_sideways_row_labels,
    detect_transposed_table,
)


def _cell(rs, re_, cs, ce, text="", x1=0.0, x2=40.0, y1=0.0, y2=20.0):
    return {
        "row_start": rs,
        "row_end": re_,
        "col_start": cs,
        "col_end": ce,
        "row_span": re_ - rs + 1,
        "col_span": ce - cs + 1,
        "text": text,
        "texts": [],
        "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
    }


def _tb(text, x1, y1, x2, y2):
    return {
        "text": text,
        "score": 0.99,
        "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
    }


def test_sideways_via_text_boxes():
    """IoA 前 cell.text 为空，须靠 OCR 框判定实施例横排。"""
    # 12 列 x 若干行：首行单元格空文本，OCR「实施例N」落在 row0
    cells = []
    boxes = []
    for i in range(12):
        cells.append(_cell(0, 0, i, i, "", x1=i * 50, x2=i * 50 + 45, y1=0, y2=25))
        boxes.append(_tb(f"实施例{i+1}", i * 50 + 5, 5, i * 50 + 40, 20))
    for r in range(1, 8):
        for i in range(12):
            cells.append(
                _cell(r, r, i, i, "", x1=i * 50, x2=i * 50 + 45, y1=r * 30, y2=r * 30 + 25)
            )
    assert detect_sideways_row_labels(cells, boxes), "应检出侧躺"
    assert detect_transposed_table(cells, boxes), "应检出转置"
    # 无 OCR 时不得误报
    assert not detect_sideways_row_labels(cells, None)


def test_upright_example_column_not_flagged():
    """正常：实施例在首列纵排，不得当转置。"""
    cells = []
    boxes = []
    cells.append(_cell(0, 0, 0, 0, "", 0, 80, 0, 25))
    cells.append(_cell(0, 0, 1, 1, "", 80, 160, 0, 25))
    boxes.append(_tb("组合物", 90, 5, 150, 20))
    for r in range(1, 13):
        cells.append(_cell(r, r, 0, 0, "", 0, 80, r * 30, r * 30 + 25))
        cells.append(_cell(r, r, 1, 1, "", 80, 160, r * 30, r * 30 + 25))
        boxes.append(_tb(f"实施例{r}", 5, r * 30 + 5, 70, r * 30 + 20))
        boxes.append(_tb(str(r), 90, r * 30 + 5, 140, r * 30 + 20))
    assert not detect_sideways_row_labels(cells, boxes)
    assert not detect_transposed_table(cells, boxes)


def test_html_broken_empty_rows():
    bad = (
        '<p>[表3-2]</p>\n<table border="1">\n'
        + "".join(f'<tr>\n<td rowspan="7">实施例 {i}</td>\n</tr>\n<tr>\n</tr>\n' for i in range(6))
        + "</table>"
    )
    assert html_output_looks_broken(bad)


def test_html_good_p57_like():
    good = """<p>[表3-2]</p>
<table border="1" cellspacing="0" cellpadding="4">
<tr>
<td rowspan="2"></td>
<td rowspan="2">组合物</td>
<td colspan="6">感光特性/固化膜的特性</td>
</tr>
<tr>
<td>灵敏度<br>[mJ/cm²]</td>
<td>显影残渣<br>[%]</td>
</tr>
<tr>
<td>实施例1</td>
<td>1</td>
<td>30<br>A+</td>
<td>10<br>B</td>
</tr>
</table>"""
    assert not html_output_looks_broken(good)


def test_html_broken_example_as_columns():
    """侧躺：首行横排多个实施例（即使下文有组合物也判坏）。"""
    bad = (
        "".join(f"<p>{x}</p>\n" for x in ["100"] * 10)
        + '<p>[表3-2]</p>\n<table border="1">\n<tr>\n'
        + "".join(f"<td>实施例{i}</td>\n" for i in range(1, 10))
        + "</tr>\n<tr>\n<td>组合物</td>\n</tr>\n</table>"
    )
    assert html_output_looks_broken(bad)


def main() -> int:
    tests = [
        test_sideways_via_text_boxes,
        test_upright_example_column_not_flagged,
        test_html_broken_empty_rows,
        test_html_good_p57_like,
        test_html_broken_example_as_columns,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"OK  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
