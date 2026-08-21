# -*- coding: utf-8 -*-
"""html_output_looks_broken / html_structure_health：侧躺误判与配方倾倒。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.output.html_formatter import html_output_looks_broken, html_structure_health


def _upright_experiment_table() -> str:
    """P26X199 类：碱溶性树脂列头 + 实施例行标签（旧逻辑会误判 broken）。"""
    rows = [
        "<tr>"
        '<td rowspan="2"></td>'
        '<td rowspan="2">碱溶性树脂</td>'
        '<td rowspan="2">醌二叠氮化合物</td>'
        '<td rowspan="2">通式(1)表示的<br>化合物</td>'
        '<td rowspan="2">交联剂</td>'
        '<td rowspan="2">溶剂</td>'
        '<td colspan="2">密合性<br>(剥离个数)</td>'
        '<td rowspan="2">敏感度<br>mJ/cm2</td>'
        '<td rowspan="2">耐化学<br>药品性</td>'
        "</tr>",
        "<tr><td>PCT处理前</td><td>PCT处理400小时后</td></tr>",
    ]
    for i in range(1, 18):
        rows.append(
            f"<tr><td>实施例{i}</td><td>A-1<br>5g</td><td>B-1<br>0.7g</td>"
            f"<td>C-1<br>0.07g</td><td>HMOM<br>1.0g</td><td>GBL</td>"
            f"<td>0</td><td>0</td><td>A</td><td>B</td></tr>"
        )
    for i in range(1, 9):
        rows.append(
            f"<tr><td>比较例{i}</td><td>A-1<br>5g</td><td>B-1<br>0.7g</td>"
            f"<td>无</td><td>HMOM<br>1.0g</td><td>GBL</td>"
            f"<td>100</td><td>100</td><td>C</td><td>C</td></tr>"
        )
    return '<table border="1">\n' + "\n".join(rows) + "\n</table>"


def _sideways_recipe_dump() -> str:
    """误转 90° 后的宽表：少行多列、等级字母头、GBL/HMOM 竖叠。"""
    grades = "".join(
        f'<td rowspan="2">{g}</td>'
        for g in [
            "B",
            "B",
            "A<br>A",
            "A<br>A",
            "B<br>A",
            "B<br>A",
            "B<br>A",
            "B<br>A",
            "B<br>A",
            "B<br>A",
            "B<br>A",
            "B<br>A",
            "B<br>B",
            "A<br>A",
            "A<br>A",
            "A<br>B",
            "A<br>A",
            "B<br>A",
            "B<br>A",
            "B<br>A",
            "B<br>A",
            "B<br>C",
            "A<br>A",
            "C B<br>A A",
        ]
    )
    gbl = "".join(
        f"<td>0<br>0<br>GBL</td>" for _ in range(12)
    ) + "".join(f"<td>100<br>0<br>GBL</td>" for _ in range(12))
    recipe = "".join(
        "<td>HMOM<br>1.0g<br>C-5 0.07<br>B-1 0.7g<br>A-1 5g</td>"
        for _ in range(24)
    )
    return (
        '<table border="1">'
        f"<tr><td>品</td>{grades}</tr>"
        "<tr><td>2 mJ/cm</td></tr>"
        f"<tr><td></td>{gbl}</tr>"
        f"<tr><td></td>{recipe}</tr>"
        "</table>"
    )


def _sideways_example_as_columns() -> str:
    """实施例横排成首行列头（经典侧躺）。"""
    heads = "".join(f"<td>实施例{i}</td>" for i in range(1, 13))
    return (
        f'<table border="1"><tr><td>组成</td>{heads}</tr>'
        "<tr><td>树脂</td>"
        + "".join("<td>A</td>" for _ in range(12))
        + "</tr></table>"
    )


def test_upright_experiment_table_not_broken():
    html = _upright_experiment_table()
    assert html_output_looks_broken(html) is False
    assert html_structure_health(html) > 20


def test_sideways_recipe_dump_is_broken():
    html = _sideways_recipe_dump()
    assert html_output_looks_broken(html) is True
    assert html_structure_health(html) < html_structure_health(
        _upright_experiment_table()
    )


def test_sideways_example_columns_is_broken():
    html = _sideways_example_as_columns()
    assert html_output_looks_broken(html) is True


def test_health_prefers_upright_over_dump():
    good = _upright_experiment_table()
    bad = _sideways_recipe_dump()
    assert html_structure_health(good) > html_structure_health(bad) + 10


if __name__ == "__main__":
    test_upright_experiment_table_not_broken()
    test_sideways_recipe_dump_is_broken()
    test_sideways_example_columns_is_broken()
    test_health_prefers_upright_over_dump()
    print("ok")
