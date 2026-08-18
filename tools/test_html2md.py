"""html2md：colspan 填充列压缩 vs 表体独立列保留。

用法:
    python tools/test_html2md.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.output.html2md import html_to_markdown  # noqa: E402


def _header_cells(md: str, table_index: int = 0) -> list[str]:
    tables = [blk for blk in md.strip().split("\n\n") if blk.startswith("|")]
    header = tables[table_index].splitlines()[0]
    return [c.strip() for c in header.strip("|").split("|")]


def _data_row(md: str, table_index: int = 0, row_index: int = 0) -> list[str]:
    tables = [blk for blk in md.strip().split("\n\n") if blk.startswith("|")]
    # skip header + separator
    rows = tables[table_index].splitlines()[2:]
    return [c.strip() for c in rows[row_index].strip("|").split("|")]


def test_example15_colspan4_collapses() -> None:
    """合成例 15：表头+表体同为 colspan=4 → 收成 1 列。"""
    html = """
    <p>[表 1-3]</p>
    <table>
    <tr>
      <td rowspan="2"></td>
      <td rowspan="2">聚合物</td>
      <td colspan="6">单体[摩尔比]</td>
      <td rowspan="2">酸当量 [g/mol]</td>
    </tr>
    <tr>
      <td colspan="4">具有芳香族基团及环氧基的化合物</td>
      <td>二羧酸酐 二羧酸</td>
      <td>具有烯键式不饱和双键基团的不饱和羧酸</td>
    </tr>
    <tr>
      <td>合成例 15</td>
      <td>酸改性环氧树脂溶液 (AE-1)</td>
      <td colspan="4">NC-7000L (环氧基准摩尔比：100)</td>
      <td>THPHA (摩尔比: 80)</td>
      <td>MAA (摩尔比: 100)</td>
      <td>540</td>
    </tr>
    </table>
    """
    md = html_to_markdown(html)
    headers = _header_cells(md)
    aromatic = [h for h in headers if "具有芳香族基团及环氧基的化合物" in h]
    assert len(aromatic) == 1, f"expected 1 aromatic col, got {aromatic} in {headers}"
    assert any("二羧酸酐" in h for h in headers)
    assert any("不饱和羧酸" in h or "不饱和" in h for h in headers)
    row = _data_row(md)
    assert "NC-7000L" in " | ".join(row)
    assert row.count("") < 3  # padding empties gone


def test_example16_two_colspan2_collapse() -> None:
    """合成例 16：两处 colspan=2 → 各收成 1 列。"""
    html = """
    <table>
    <tr>
      <td rowspan="2"></td>
      <td rowspan="2">聚合物</td>
      <td colspan="6">单体[摩尔比]</td>
    </tr>
    <tr>
      <td colspan="2">具有酸性基团的共聚成分</td>
      <td colspan="2">具有芳香族基团的共聚成分</td>
      <td>具有脂环式基团的共聚成分</td>
      <td>具有烯键式不饱和双键基团及环氧基的不饱和化合物</td>
    </tr>
    <tr>
      <td>合成例 16</td>
      <td>丙烯酸树脂溶液 (AC-1)</td>
      <td colspan="2">MAA (50)</td>
      <td colspan="2">STR (30)</td>
      <td>TCDM (20)</td>
      <td>GMA (20)</td>
    </tr>
    </table>
    """
    md = html_to_markdown(html)
    headers = _header_cells(md)
    acid = [h for h in headers if "酸性基团" in h]
    aroma = [h for h in headers if "芳香族基团" in h and "共聚成分" in h]
    assert len(acid) == 1, f"acid headers={acid} all={headers}"
    assert len(aroma) == 1, f"aroma headers={aroma} all={headers}"
    row = _data_row(md)
    joined = " | ".join(row)
    assert "MAA (50)" in joined
    assert "STR (30)" in joined
    assert "TCDM (20)" in joined
    assert "GMA (20)" in joined


def test_p96_header_span_body_split_keeps_columns() -> None:
    """P96 型：表头 colspan=2，表体两独立格 → 仍为 2 列。"""
    html = """
    <table>
    <tr>
      <td rowspan="2">聚合物</td>
      <td colspan="7">单体[摩尔比]</td>
    </tr>
    <tr>
      <td colspan="2">四羧酸及其衍生物</td>
      <td colspan="3">二胺及其衍生物</td>
      <td>封端剂</td>
      <td>不饱和化合物</td>
    </tr>
    <tr>
      <td>聚酰亚胺 (PI-1)</td>
      <td>ODPA (100)</td>
      <td></td>
      <td>BAHF (85)</td>
      <td>-</td>
      <td>SiDA (5)</td>
      <td>MAP (20)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>聚酰亚胺 (PI-3)</td>
      <td>ODPA (60)</td>
      <td>6FDA (40)</td>
      <td>BAHF (85)</td>
      <td></td>
      <td>SiDA (5)</td>
      <td>MAP (20)</td>
      <td>-</td>
    </tr>
    </table>
    """
    md = html_to_markdown(html)
    headers = _header_cells(md)
    tetra = [h for h in headers if "四羧酸" in h]
    diamine = [h for h in headers if "二胺" in h]
    assert len(tetra) == 2, f"keep 2 tetra cols, got {tetra} in {headers}"
    assert len(diamine) == 3, f"keep 3 diamine cols, got {diamine} in {headers}"
    row0 = _data_row(md, row_index=0)
    row1 = _data_row(md, row_index=1)
    assert "ODPA (100)" in row0
    assert "6FDA (40)" in row1


def test_p97_header_span_four_body_cells_kept() -> None:
    """P97 型：表头 colspan=4，表体 4 个独立格 → 仍为 4 列。"""
    html = """
    <table>
    <tr>
      <td rowspan="2">聚合物</td>
      <td colspan="6">单体[mol%]</td>
    </tr>
    <tr>
      <td colspan="4">三官能有机硅烷</td>
      <td>四官能有机硅烷</td>
      <td>双官能有机硅烷</td>
    </tr>
    <tr>
      <td>聚硅氧烷溶液 (PS-1)</td>
      <td>MeTMS (35)</td>
      <td>PhTMS (50)</td>
      <td>TMSSucA (10)</td>
      <td></td>
      <td>TMOS (5)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>聚硅氧烷溶液 (PS-2)</td>
      <td>MeTMS (20)</td>
      <td>PhTMS (50)</td>
      <td>TMSSucA (10)</td>
      <td>AcrTMS (20)</td>
      <td>-</td>
      <td>-</td>
    </tr>
    </table>
    """
    md = html_to_markdown(html)
    headers = _header_cells(md)
    trifunc = [h for h in headers if "三官能" in h]
    assert len(trifunc) == 4, f"keep 4 trifunc cols, got {trifunc} in {headers}"
    row1 = _data_row(md, row_index=1)
    assert "AcrTMS (20)" in row1


def test_example13_distinct_subheaders_not_merged() -> None:
    """合成例 13 型：6 个不同子表头、部分表体为空 → 不合并。"""
    html = """
    <table>
    <tr>
      <td rowspan="2"></td>
      <td rowspan="2">聚合物</td>
      <td colspan="6">单体[摩尔比]</td>
    </tr>
    <tr>
      <td>具有羟基及两个以上芳香族基团的化合物</td>
      <td>具有环氧基及两个以上芳香族基团的化合物</td>
      <td>四羧酸二酐 四羧酸</td>
      <td>封端剂</td>
      <td>具有烯键式不饱和双键基团及环氧基的不饱和化合物</td>
      <td>具有烯键式不饱和双键基团的不饱和羧酸</td>
    </tr>
    <tr>
      <td>合成例 13</td>
      <td>含多环侧链的树脂溶液 (CR-1)</td>
      <td>BHPF (100)</td>
      <td></td>
      <td>ODPA (90)</td>
      <td>PHA (20)</td>
      <td>GMA (100)</td>
      <td></td>
    </tr>
    <tr>
      <td>合成例 14</td>
      <td>含多环侧链的树脂溶液 (CR-2)</td>
      <td></td>
      <td>BGPF (100)</td>
      <td>ODPA (90)</td>
      <td>PHA (20)</td>
      <td></td>
      <td>MAA (200)</td>
    </tr>
    </table>
    """
    md = html_to_markdown(html)
    headers = _header_cells(md)
    monomer = [h for h in headers if "单体[摩尔比]" in h]
    assert len(monomer) == 6, f"keep 6 monomer cols, got {len(monomer)}: {headers}"
    row0 = _data_row(md, row_index=0)
    row1 = _data_row(md, row_index=1)
    assert "BHPF (100)" in row0
    assert "BGPF (100)" in row1
    assert "MAA (200)" in row1


if __name__ == "__main__":
    test_example15_colspan4_collapses()
    test_example16_two_colspan2_collapse()
    test_p96_header_span_body_split_keeps_columns()
    test_p97_header_span_four_body_cells_kept()
    test_example13_distinct_subheaders_not_merged()
    print("ok")
