"""P98：纯 colspan 列（无原子格）不得把有字的格子删掉。

用法:
    python tools/test_drop_evidenceless.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.output.html_formatter import cells_to_html_table, drop_evidenceless_columns


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


def test_example15_colspan4_keeps_nc7000l():
    """合成例 15：芳香族环氧列 colspan=4，旁边才是原子列。"""
    # cols: 0 序号 | 1 聚合物 | 2-5 芳香族环氧 | 6 二羧酸酐 | 7 不饱和羧酸 | 8 酸当量
    cells = [
        _cell(0, 1, 0, 0, "", 0, 40, 0, 40),
        _cell(0, 1, 1, 1, "聚合物", 40, 120, 0, 40),
        _cell(0, 0, 2, 7, "单体[摩尔比]", 120, 520, 0, 20),
        _cell(0, 1, 8, 8, "酸当量 [g/mol]", 520, 600, 0, 40),
        _cell(1, 1, 2, 5, "具有芳香族基团及环氧基的化合物", 120, 360, 20, 40),
        _cell(1, 1, 6, 6, "二羧酸酐 二羧酸", 360, 440, 20, 40),
        _cell(1, 1, 7, 7, "不饱和羧酸", 440, 520, 20, 40),
        _cell(2, 2, 0, 0, "合成例 15", 0, 40, 40, 80),
        _cell(2, 2, 1, 1, "酸改性环氧树脂溶液 (AE-1)", 40, 120, 40, 80),
        _cell(2, 2, 2, 5, "NC-7000L (环氧基准摩尔比：100)", 120, 360, 40, 80),
        _cell(2, 2, 6, 6, "THPHA (摩尔比: 80)", 360, 440, 40, 80),
        _cell(2, 2, 7, 7, "MAA (摩尔比: 100)", 440, 520, 40, 80),
        _cell(2, 2, 8, 8, "540", 520, 600, 40, 80),
    ]
    out = drop_evidenceless_columns(cells)
    texts = " ".join(str(c.get("text") or "") for c in out)
    assert "NC-7000L" in texts, texts
    assert "THPHA" in texts
    assert "MAA" in texts
    html = cells_to_html_table(cells)
    assert "NC-7000L" in html, html
    assert "THPHA" in html
    assert "具有芳香族基团及环氧基的化合物" in html


def test_example16_two_colspan2_keeps_maa_str():
    """合成例 16：酸性/芳香族各 colspan=2，脂环与环氧是原子列。"""
    cells = [
        _cell(0, 1, 0, 0, "", 0, 40, 0, 40),
        _cell(0, 1, 1, 1, "聚合物", 40, 120, 0, 40),
        _cell(0, 0, 2, 7, "单体[摩尔比]", 120, 520, 0, 20),
        _cell(0, 1, 8, 8, "酸当量 [g/mol]", 520, 600, 0, 40),
        _cell(1, 1, 2, 3, "具有酸性基团的共聚成分", 120, 220, 20, 40),
        _cell(1, 1, 4, 5, "具有芳香族基团的共聚成分", 220, 320, 20, 40),
        _cell(1, 1, 6, 6, "具有脂环式基团的共聚成分", 320, 420, 20, 40),
        _cell(1, 1, 7, 7, "不饱和化合物", 420, 520, 20, 40),
        _cell(2, 2, 0, 0, "合成例 16", 0, 40, 40, 70),
        _cell(2, 2, 1, 1, "丙烯酸树脂溶液 (AC-1)", 40, 120, 40, 70),
        _cell(2, 2, 2, 3, "MAA (50)", 120, 220, 40, 70),
        _cell(2, 2, 4, 5, "STR (30)", 220, 320, 40, 70),
        _cell(2, 2, 6, 6, "TCDM (20)", 320, 420, 40, 70),
        _cell(2, 2, 7, 7, "GMA (20)", 420, 520, 40, 70),
        _cell(2, 2, 8, 8, "490", 520, 600, 40, 70),
    ]
    out = drop_evidenceless_columns(cells)
    texts = " ".join(str(c.get("text") or "") for c in out)
    assert "MAA (50)" in texts, texts
    assert "STR (30)" in texts, texts
    assert "TCDM (20)" in texts
    assert "GMA (20)" in texts
    html = cells_to_html_table(cells)
    assert "MAA (50)" in html, html
    assert "STR (30)" in html, html


def test_salvage_when_origin_cols_all_dropped():
    """兜底：若原点列仍被删，文本并入左侧保留格而不是静默丢失。"""
    cells = [
        _cell(0, 0, 0, 0, "聚合物", 0, 200),
        # 窄碎片列 + 纯 colspan 且文本极短（会被当成 frag），触发整段 col 被删
        _cell(0, 0, 1, 1, "α", 200, 210),
        _cell(1, 1, 0, 0, "A-1", 0, 200, 30, 60),
        _cell(1, 1, 1, 2, "xy", 200, 250, 30, 60),
    ]
    out = drop_evidenceless_columns(cells)
    texts = " ".join(str(c.get("text") or "") for c in out)
    assert "聚合物" in texts
    assert "A-1" in texts
    # 碎片可并入邻格，但不能整段蒸发成空表
    assert out, "expected surviving cells"


def test_colspan_ghost_columns_p28_pattern():
    """P28X229：TSR 5 列（label|ghost|data|ghost|data）→ 3 列。"""
    # 5 cols x 4 rows: header x2 + 2 data rows
    cells = [
        _cell(0, 0, 0, 0, "", 0, 80, 0, 30),
        _cell(0, 0, 1, 2, "最小曝光量（Eth）", 80, 280, 0, 30),
        _cell(0, 0, 3, 4, "密合强度", 280, 480, 0, 30),
        _cell(1, 1, 0, 0, "", 0, 80, 30, 60),
        _cell(1, 1, 1, 1, "", 80, 130, 30, 60),
        _cell(1, 1, 2, 2, "(mJ/cm2)", 130, 230, 30, 60),
        _cell(1, 1, 3, 3, "(mN)", 230, 330, 30, 60),
        _cell(1, 1, 4, 4, "", 330, 480, 30, 60),
        _cell(2, 2, 0, 0, "实施例 1", 0, 80, 60, 90),
        _cell(2, 2, 1, 1, "", 80, 130, 60, 90),
        _cell(2, 2, 2, 2, "200", 130, 230, 60, 90),
        _cell(2, 2, 3, 3, "", 230, 330, 60, 90),
        _cell(2, 2, 4, 4, "358", 330, 480, 60, 90),
        _cell(3, 3, 0, 0, "实施例 2", 0, 80, 90, 120),
        _cell(3, 3, 1, 1, "", 80, 130, 90, 120),
        _cell(3, 3, 2, 2, "180", 130, 230, 90, 120),
        _cell(3, 3, 3, 3, "", 230, 330, 90, 120),
        _cell(3, 3, 4, 4, "309", 330, 480, 90, 120),
    ]
    html = cells_to_html_table(cells)
    assert "200" in html and "358" in html
    assert "最小曝光量" in html and "密合强度" in html
    assert "<td></td>" not in html.split("<tr>")[3]  # 数据行无幽灵空 td 串


def test_p32_solubility_viscosity_columns_separate():
    """P32X266：溶解性 / 聚合物溶液粘度 应为并列子列，不得合并。"""
    cells = [
        _cell(0, 1, 0, 0, "", 0, 60, 0, 40),
        _cell(0, 1, 1, 1, "树脂", 60, 140, 0, 40),
        _cell(0, 0, 2, 3, "溶解性评价\n30wt% PGMEA 溶液", 140, 340, 0, 20),
        _cell(0, 1, 4, 4, "判定", 340, 420, 0, 40),
        _cell(1, 1, 2, 2, "溶解性", 140, 220, 20, 40),
        _cell(1, 1, 3, 3, "聚合物溶液粘度", 220, 340, 20, 40),
        _cell(2, 2, 0, 0, "树脂(A)", 0, 60, 40, 70),
        _cell(2, 2, 1, 1, "聚酰胺酸酯", 60, 140, 40, 70),
        _cell(2, 2, 2, 2, "溶解", 140, 220, 40, 70),
        _cell(2, 2, 3, 3, "119 mPa·s", 220, 340, 40, 70),
        _cell(2, 2, 4, 4, "A", 340, 420, 40, 70),
    ]
    html = cells_to_html_table(cells)
    assert "溶解性<br>聚合物溶液粘度" not in html, html
    assert "<td>溶解性</td>" in html and "<td>聚合物溶液粘度</td>" in html
    assert "溶解" in html and "119" in html


def test_colspan_ghost_columns_p29_pattern():
    """P29X231：比较例表，5 列幽灵 padding → 3 列，单位并入表头。"""
    cells = [
        _cell(0, 0, 0, 0, "", 0, 80, 0, 30),
        _cell(0, 0, 1, 2, "最小曝光量（Eth）", 80, 280, 0, 30),
        _cell(0, 0, 3, 4, "密合强度", 280, 480, 0, 30),
        _cell(1, 1, 0, 0, "", 0, 80, 30, 60),
        _cell(1, 1, 1, 1, "", 80, 130, 30, 60),
        _cell(1, 1, 2, 2, "(mJ/cm²)", 130, 230, 30, 60),
        _cell(1, 1, 3, 3, "(mN)", 230, 330, 30, 60),
        _cell(1, 1, 4, 4, "", 330, 480, 30, 60),
        _cell(2, 2, 0, 0, "比较例1", 0, 80, 60, 90),
        _cell(2, 2, 1, 1, "", 80, 130, 60, 90),
        _cell(2, 2, 2, 2, "350", 130, 230, 60, 90),
        _cell(2, 2, 3, 3, "", 230, 330, 60, 90),
        _cell(2, 2, 4, 4, "31", 330, 480, 60, 90),
    ]
    html = cells_to_html_table(cells)
    assert "350" in html and "31" in html
    assert "最小曝光量" in html and "密合强度" in html
    assert "(mJ/cm" in html and "(mN)" in html
    assert html.count("<td") <= 12  # 3 cols * ~4 rows
    data_row = html.split("<tr>")[3]
    assert data_row.count("<td") == 3, data_row
    assert "<td></td>" not in data_row


def main() -> int:
    test_example15_colspan4_keeps_nc7000l()
    test_example16_two_colspan2_keeps_maa_str()
    test_salvage_when_origin_cols_all_dropped()
    test_colspan_ghost_columns_p28_pattern()
    test_colspan_ghost_columns_p29_pattern()
    test_p32_solubility_viscosity_columns_separate()
    print("OK: drop_evidenceless colspan origin tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
