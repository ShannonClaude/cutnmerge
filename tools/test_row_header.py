"""左侧大行头：顶表头带截止 + 父格过宽 colspan 裁剪。

用法:
    python tools/test_row_header.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.matching.matching import (  # noqa: E402
    _parse_sticky_column_parts,
    _split_sticky_column_header,
    _split_wufapingjia_glued,
    detect_eval_symbols_in_empty_cells,
)
from src.output.html_formatter import (  # noqa: E402
    _collapse_header_empty_corners,
    _merge_header_empty_below,
    _merge_leading_empty_into_label,
    _repair_lone_example_number_headers,
    _resolve_logic_overlaps,
    _split_example_header_rowspans,
    cells_to_html_table,
)
from src.ocr.ocr_post import normalize_ocr_text  # noqa: E402
from src.structure.row_header import (  # noqa: E402
    clip_narrow_label_colspans,
    clip_row_header_child_overlaps,
    extend_section_rowspan_over_metric_rows,
    peel_row_header_text,
    relocate_misplaced_category_labels,
)
from src.structure.tsr_refine import (  # noqa: E402
    dedupe_overlapping_cells,
    refine_tsr_cells_light,
)
from src.matching.matching import unmerge_filled_label_rowspans  # noqa: E402
from src.utils.label_patterns import (  # noqa: E402
    are_independent_row_labels,
    extract_independent_labels_from_joined,
    is_independent_row_label,
)


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


def _p25x193_like_cells():
    """比较例表：左侧 (A)成分 rowspan=2，比较例1 下方数据格为空。"""
    return [
        _cell(0, 1, 0, 1, "", 0, 160, 0, 60),
        _cell(0, 0, 2, 5, "感光性树脂组合物(重量份)", 160, 480, 0, 30),
        _cell(1, 1, 2, 2, "比较例 1", 160, 240, 30, 60),
        _cell(1, 1, 3, 3, "比较例 2", 240, 320, 30, 60),
        _cell(1, 1, 4, 4, "比较例 3", 320, 400, 30, 60),
        _cell(1, 1, 5, 5, "比较例4", 400, 480, 30, 60),
        _cell(2, 3, 0, 0, "(A)成分", 0, 80, 60, 120),
        _cell(2, 2, 1, 1, "合成例1的聚酰亚胺", 80, 160, 60, 90),
        _cell(2, 2, 3, 3, "100", 240, 320, 60, 90),
        _cell(2, 2, 4, 4, "100", 320, 400, 60, 90),
        _cell(3, 3, 1, 1, "合成例2的聚酰亚胺", 80, 160, 90, 120),
        _cell(3, 3, 5, 5, "100", 400, 480, 90, 120),
        _cell(4, 5, 0, 0, "(B)成分", 0, 80, 120, 180),
        _cell(4, 4, 1, 1, "jER-630", 80, 160, 120, 150),
        _cell(5, 5, 1, 1, "jER-604", 80, 160, 150, 180),
    ]


def _nested_row_header_cells():
    """表1 形态：外层组成 colspan 过宽，盖住酸酐 / 品名。"""
    names = [
        "jER828",
        "OXT-191",
        "EP4003S",
        "jER152",
        "jER630",
        "NC3000",
        "EPICLON850",
    ]
    cells = [
        _cell(0, 1, 0, 2, "项目", 0, 240, 0, 40),
        _cell(0, 1, 3, 3, "实施例 1", 240, 320, 0, 40),
        _cell(2, 10, 0, 1, "聚酰亚胺组成 (摩尔比)", 0, 80, 40, 220),
        _cell(2, 3, 1, 1, "酸酐", 80, 160, 40, 80),
        _cell(2, 2, 2, 2, "ODPA", 160, 240, 40, 60),
        _cell(3, 3, 2, 2, "BPDA", 160, 240, 60, 80),
        _cell(14, 20, 1, 3, "化合物(b)", 80, 160, 280, 420),
    ]
    for i, name in enumerate(names):
        y1 = 280.0 + i * 20.0
        cells.append(_cell(14 + i, 14 + i, 2, 2, name, 160, 280, y1, y1 + 20))
    return cells


def _p100_fensan_split_cells():
    """顶表头带内：单列「分散液」被切成两层，粒径 rowspan 把表头带扩到 row 2。"""
    return [
        _cell(1, 1, 1, 1, "分散液", 118, 200, 35, 53),
        _cell(2, 2, 1, 1, "", 118, 200, 53, 99),
        _cell(1, 2, 2, 2, "颜料分散液中的颜料的数均粒径 [nm]", 200, 320, 35, 99),
        _cell(3, 3, 0, 0, "制备例1", 0, 118, 99, 130),
    ]


def test_body_left_stub_does_not_merge_column_headers():
    html = cells_to_html_table(_p25x193_like_cells())
    assert re.search(r'rowspan="2"[^>]*>\(A\)成分', html), html
    assert re.search(r'rowspan="2"[^>]*>\(B\)成分', html), html
    assert not re.search(r'rowspan="2"[^>]*>比较例 1', html), html
    assert not re.search(r'rowspan="2"[^>]*>比较例4', html), html
    assert not re.search(r'rowspan="2"[^>]*>100', html), html
    assert "<td>比较例 1</td>" in html or re.search(
        r"<td>比较例 1</td>", html
    )


def test_p100_fensan_still_merges_in_header_band():
    merged = _merge_header_empty_below(_p100_fensan_split_cells())
    fensan = next(
        c for c in merged if str(c.get("text") or "") == "分散液"
    )
    assert int(fensan["row_start"]) == 1
    assert int(fensan["row_end"]) == 2, fensan
    html = cells_to_html_table(_p100_fensan_split_cells())
    assert re.search(r'rowspan="2"[^>]*>分散液', html), html


def test_p40_stacked_chemical_header():
    """P40：同列「二苯基醚/二甲酰氯」应纵向合并为一格。"""
    cells = [
        _cell(0, 0, 0, 0, "树脂", 0, 80, 0, 20),
        _cell(0, 0, 1, 5, "单体、封端剂组成", 80, 400, 0, 20),
        _cell(1, 1, 1, 3, "酸二酐 (摩尔比率)", 80, 240, 20, 40),
        _cell(1, 1, 4, 4, "二苯基醚", 240, 320, 20, 40),
        _cell(1, 1, 5, 9, "二胺(摩尔比率)", 320, 560, 20, 40),
        _cell(2, 2, 1, 1, "6FDA", 80, 160, 40, 60),
        _cell(2, 2, 2, 2, "ODPA", 160, 240, 40, 60),
        _cell(2, 2, 4, 4, "二甲酰氯", 240, 320, 40, 60),
        _cell(2, 2, 5, 5, "α", 320, 400, 40, 60),
        _cell(3, 3, 0, 0, "树脂（AA）", 0, 80, 60, 80),
        _cell(3, 3, 1, 1, "100", 80, 160, 60, 80),
    ]
    html = cells_to_html_table(cells)
    assert re.search(r'rowspan="2"[^>]*>二苯基醚<br>二甲酰氯', html), html
    assert not re.search(
        r">二苯基醚</td>\s*</tr>\s*<tr>[^<]*<td>6FDA",
        html,
    ), html
    assert re.search(r">6FDA</td>", html), html
    assert re.search(r">ODPA</td>", html), html


def test_clip_nested_row_header_parent_colspan():
    cells = _nested_row_header_cells()
    out = clip_row_header_child_overlaps([dict(c) for c in cells])
    parent = next(c for c in out if "聚酰亚胺组成" in str(c.get("text") or ""))
    assert int(parent["col_start"]) == 0
    assert int(parent["col_end"]) == 0, parent
    suan = next(c for c in out if str(c.get("text") or "") == "酸酐")
    assert (int(suan["row_start"]), int(suan["row_end"])) == (2, 3)
    b_cell = next(c for c in out if "化合物" in str(c.get("text") or ""))
    assert int(b_cell["col_end"]) == 1, b_cell
    names = {str(c.get("text") or "") for c in out}
    assert "jER828" in names and "EPICLON850" in names


def test_html_keeps_nested_labels_separate():
    cells = clip_row_header_child_overlaps(_nested_row_header_cells())
    html = cells_to_html_table(cells)
    assert "聚酰亚胺组成" in html
    assert "酸酐" in html
    assert "ODPA" in html
    assert "化合物(b)" in html
    assert "jER828" in html
    assert "OXT-191" in html
    blob = re.search(
        r'rowspan="7"[^>]*>[^<]*化合物[^<]*jER828',
        html,
    )
    assert blob is None, html
    assert re.search(r'rowspan="9"[^>]*>聚酰亚胺组成', html), html
    assert re.search(r'rowspan="2"[^>]*>酸酐', html), html


def test_resolve_does_not_swallow_right_child_text():
    parent = _cell(14, 20, 1, 3, "化合物(b)", 80, 160, 280, 420)
    child = _cell(14, 14, 2, 2, "jER828", 160, 280, 280, 300)
    out = _resolve_logic_overlaps([parent, child])
    texts = {str(c.get("text") or "") for c in out}
    assert "化合物(b)" in texts
    assert "jER828" in texts
    assert not any("化合物(b)" in str(c.get("text")) and "jER828" in str(c.get("text")) for c in out)


def test_light_refine_clips_before_dedupe():
    cells = _nested_row_header_cells()
    out = refine_tsr_cells_light([dict(c) for c in cells])
    texts = {str(c.get("text") or "") for c in out}
    assert "酸酐" in texts
    assert "jER828" in texts
    parent = next(c for c in out if "聚酰亚胺组成" in str(c.get("text") or ""))
    assert int(parent["col_end"]) == 0, parent


def test_clip_same_col_start_name_cell():
    """品名格与 (b) 分类格同一 col_start、物理上在右 → 父格单列、子格右移。"""
    parent = _cell(14, 20, 1, 2, "化合物(b)", 80, 160, 280, 420)
    child = _cell(14, 14, 1, 1, "jER828", 160, 280, 280, 300)
    out = clip_row_header_child_overlaps([parent, child])
    b_cell = next(c for c in out if "化合物" in str(c.get("text") or ""))
    name = next(c for c in out if str(c.get("text") or "") == "jER828")
    assert int(b_cell["col_start"]) == 1
    assert int(b_cell["col_end"]) == 1, b_cell
    assert int(name["col_start"]) == 2, name
    html = cells_to_html_table(out)
    assert "化合物(b)" in html
    assert "jER828" in html
    assert not re.search(r"化合物\(b\)[^<]*jER828|jER828[^<]*化合物", html), html


def test_p100_ghost_not_clipped_by_row_header():
    """分散液右上角碎片不得把父格 colspan 裁窄。"""
    large = _cell(1, 2, 1, 3, "分散液", 118, 260, 35, 99)
    ghost = _cell(1, 1, 3, 3, "", 236, 260, 35, 53)
    out = clip_row_header_child_overlaps([large, ghost])
    kept = next(c for c in out if str(c.get("text") or "") == "分散液")
    assert (int(kept["col_start"]), int(kept["col_end"])) == (1, 3)
    # 去重叠仍应丢掉幽灵格
    deduped = dedupe_overlapping_cells(out)
    assert any(str(c.get("text") or "") == "分散液" for c in deduped)


def _p24x176_peel_cells():
    """酸酐空兄弟 + 父格误含「…酸酐…」。"""
    parent = _cell(2, 10, 0, 0, "聚酰亚胺酸酐组成(摩尔比)", 0, 80, 40, 220)
    sibling = _cell(2, 3, 1, 1, "", 80, 160, 40, 80)
    return [parent, sibling]


def test_peel_sublabel_from_parent():
    cells = _p24x176_peel_cells()
    cells[0]["text"] = "聚酰亚胺组成(摩尔比)酸酐"
    out = peel_row_header_text(cells)
    suan = next(c for c in out if str(c.get("text") or "") == "酸酐")
    parent = next(c for c in out if "聚酰亚胺" in str(c.get("text") or ""))
    assert "酸酐" not in str(parent.get("text") or "")
    assert int(suan["col_start"]) == 1


def test_clip_narrow_odpa_colspan():
    """ODPA 物理窄、逻辑 colspan=2 → 收到单列。"""
    odpa = _cell(2, 2, 2, 3, "ODPA", 160, 200, 40, 60)
    # 造若干正常宽列供中位宽
    peers = [
        _cell(2, 2, 4, 4, "100", 320, 400, 40, 60),
        _cell(2, 2, 5, 5, "100", 400, 480, 40, 60),
        _cell(2, 2, 6, 6, "100", 480, 560, 40, 60),
    ]
    out = clip_narrow_label_colspans([odpa, *peers])
    clipped = next(c for c in out if str(c.get("text") or "") == "ODPA")
    assert int(clipped["col_end"]) == int(clipped["col_start"]) == 2


def test_peel_jer828_from_compound_parent():
    parent = _cell(14, 20, 1, 1, "化合物jER828(b)", 80, 160, 280, 420)
    sibling = _cell(14, 14, 2, 2, "", 160, 280, 280, 300)
    out = peel_row_header_text([parent, sibling])
    name = next(c for c in out if str(c.get("text") or "") == "jER828")
    bcell = next(c for c in out if "化合物" in str(c.get("text") or ""))
    assert "jER828" not in str(bcell.get("text") or "")
    assert int(name["col_start"]) == 2


def test_sticky_column_header_parse():
    parts = _parse_sticky_column_parts("实施例8实施例9实施例10")
    assert parts == ["实施例 8", "实施例 9", "实施例 10"]
    parts2 = _parse_sticky_column_parts("8实施例9实施例10")
    assert parts2 == ["实施例 8", "实施例 9", "实施例 10"]


def test_sticky_column_header_split_geom():
    tb = {
        "text": "实施例8实施例9",
        "polygon": [[240, 0], [400, 0], [400, 30], [240, 30]],
        "top_left": (240, 0),
    }
    cells = [
        _cell(0, 0, 4, 4, "", 240, 320, 0, 30),
        _cell(0, 0, 5, 5, "", 320, 400, 0, 30),
    ]
    pieces = _split_sticky_column_header(tb, cells)
    assert pieces is not None
    assert len(pieces) == 2
    assert "8" in pieces[0]["text"]
    assert "9" in pieces[1]["text"]


def test_merge_leading_empty_into_label():
    cells = [
        _cell(20, 20, 0, 0, "", 0, 80, 400, 430),
        _cell(20, 20, 1, 3, "酰亚胺化率", 80, 240, 400, 430),
        _cell(20, 20, 4, 4, "100", 240, 320, 400, 430),
    ]
    out = _merge_leading_empty_into_label([dict(c) for c in cells])
    label = next(c for c in out if "酰亚胺化率" in str(c.get("text") or ""))
    assert int(label["col_start"]) == 0
    assert int(label["col_end"]) == 3
    html = cells_to_html_table(out)
    assert not re.search(r"<tr>\s*<td></td>\s*<td[^>]*>酰亚胺化率", html), html


def test_merge_resolution_unit():
    cells = [
        _cell(30, 30, 0, 2, "分辨率", 0, 160, 500, 530),
        _cell(30, 30, 3, 3, "(μm)", 160, 200, 500, 530),
    ]
    out = _merge_leading_empty_into_label([dict(c) for c in cells])
    merged = next(c for c in out if "分辨率" in str(c.get("text") or ""))
    assert "(μm)" in str(merged.get("text") or "")
    assert int(merged["col_end"]) == 3


def test_resolution_um_merged_on_metric_data_row():
    """P24：测量汇总行（同行有实施例数值）应合并 分辨率+(μm) 为一格。"""
    cells = [
        _cell(0, 0, 0, 1, "其他", 0, 160, 0, 30),
        _cell(0, 0, 2, 2, "合成例10的丙烯酸树脂溶液", 160, 320, 0, 30),
        _cell(1, 1, 0, 0, "分辨率", 0, 80, 30, 60),
        _cell(1, 1, 1, 3, "(μm)", 80, 320, 30, 60),
        _cell(1, 1, 4, 4, "25", 320, 400, 30, 60),
        _cell(1, 1, 5, 5, "30", 400, 480, 30, 60),
    ]
    html = cells_to_html_table(cells)
    assert "分辨率(μm)" in html.replace(" ", "")
    assert not re.search(r">分辨率</td>\s*<td[^>]*>\(μm\)</td>", html), html


def test_resolution_um_separate_when_unit_is_column():
    """嵌套表头：单位列上方有独立列头、同行无数值时保持分列。"""
    cells = [
        _cell(0, 0, 0, 0, "其他", 0, 80, 0, 30),
        _cell(0, 0, 1, 1, "烯酸树脂溶液", 80, 160, 0, 30),
        _cell(1, 1, 0, 0, "分辨率", 0, 80, 30, 60),
        _cell(1, 1, 1, 1, "(μm)", 80, 160, 30, 60),
    ]
    html = cells_to_html_table(cells)
    assert "分辨率(μm)" not in html.replace(" ", "")
    assert re.search(r">分辨率</td>\s*<td[^>]*>\(μm\)</td>", html), html


def test_no_merge_other_with_um_unit():
    """P33：「其他」与下方数据列的 (μm) 单位格不得横向合并。"""
    cells = [
        _cell(0, 0, 0, 0, "", 0, 80, 0, 30),
        _cell(0, 0, 1, 4, "清漆组成", 80, 400, 0, 15),
        _cell(0, 0, 5, 5, "其他", 400, 480, 0, 15),
        _cell(0, 0, 6, 6, "显影膜损失量", 480, 560, 0, 15),
        _cell(0, 0, 7, 7, "判定", 560, 640, 0, 30),
        _cell(1, 1, 1, 1, "清漆", 80, 160, 15, 30),
        _cell(1, 1, 2, 2, "树脂", 160, 240, 15, 30),
        _cell(1, 1, 3, 3, "感光剂", 240, 320, 15, 30),
        _cell(1, 1, 4, 4, "溶剂", 320, 400, 15, 30),
        _cell(1, 1, 5, 5, "其他", 400, 480, 15, 30),
        _cell(1, 1, 6, 6, "(μm)", 480, 560, 15, 30),
        _cell(2, 2, 0, 0, "实施例1", 0, 80, 30, 50),
        _cell(2, 2, 5, 5, "e-1", 400, 480, 30, 50),
        _cell(2, 2, 6, 6, "0.41", 480, 560, 30, 50),
    ]
    out = _merge_leading_empty_into_label([dict(c) for c in cells])
    other = next(c for c in out if str(c.get("text") or "").strip() == "其他" and int(c["row_start"]) == 1)
    unit = next(c for c in out if str(c.get("text") or "").strip() == "(μm)")
    assert int(other["col_end"]) == 5
    assert int(unit["col_start"]) == 6
    html = cells_to_html_table(cells)
    assert "其他(μm)" not in html
    assert re.search(r">其他</td>\s*<td[^>]*>\(μm\)</td>", html), html


def _p25x192_body_cells():
    """标签右侧为空、其他行同列有数字：不得 colspan 吞空格。"""
    cells = [
        _cell(0, 1, 0, 1, "", 0, 160, 0, 40),
        _cell(0, 0, 2, 8, "感光性树脂组合物(重量份)", 160, 720, 0, 20),
        _cell(1, 1, 2, 2, "实施例 1", 160, 240, 20, 40),
        _cell(1, 1, 3, 3, "实施例 2", 240, 320, 20, 40),
        _cell(1, 1, 4, 4, "实施例 3", 320, 400, 20, 40),
        _cell(1, 1, 5, 5, "实施例 4", 400, 480, 20, 40),
        _cell(1, 1, 6, 6, "实施例 5", 480, 560, 20, 40),
        _cell(1, 1, 7, 7, "实施例 6", 560, 640, 20, 40),
        _cell(1, 1, 8, 8, "实施例 7", 640, 720, 20, 40),
        _cell(2, 2, 0, 0, "(A)成分", 0, 80, 40, 60),
        _cell(2, 2, 1, 1, "合成例 1 的聚酰亚胺", 80, 160, 40, 60),
        _cell(2, 2, 2, 2, "100", 160, 240, 40, 60),
        _cell(2, 2, 3, 3, "100", 240, 320, 40, 60),
        _cell(2, 2, 4, 4, "100", 320, 400, 40, 60),
        _cell(2, 2, 5, 5, "100", 400, 480, 40, 60),
        _cell(2, 2, 6, 6, "100", 480, 560, 40, 60),
        _cell(2, 2, 7, 7, "100", 560, 640, 40, 60),
        _cell(2, 2, 8, 8, "100", 640, 720, 40, 60),
        _cell(4, 5, 0, 0, "(B)成分", 0, 80, 80, 120),
        _cell(4, 4, 1, 1, "jER-630", 80, 160, 80, 100),
        _cell(4, 4, 2, 2, "60", 160, 240, 80, 100),
        _cell(4, 4, 3, 3, "", 240, 320, 80, 100),
        _cell(4, 4, 4, 4, "30", 320, 400, 80, 100),
        _cell(5, 5, 1, 1, "jER-604", 80, 160, 100, 120),
        _cell(5, 5, 2, 2, "", 160, 240, 100, 120),
        _cell(5, 5, 3, 3, "60", 240, 320, 100, 120),
        _cell(5, 5, 4, 4, "", 320, 400, 100, 120),
        _cell(8, 9, 0, 0, "(E)成分", 0, 80, 160, 200),
        _cell(8, 8, 1, 1, "850S", 80, 160, 160, 180),
        _cell(8, 8, 2, 2, "", 160, 240, 160, 180),
        _cell(8, 8, 3, 3, "", 240, 320, 160, 180),
        _cell(8, 8, 4, 4, "30", 320, 400, 160, 180),
        _cell(9, 9, 1, 7, "EP4003S", 80, 640, 180, 200),
        _cell(9, 9, 8, 8, "30", 640, 720, 180, 200),
    ]
    return cells


def test_p25x192_label_does_not_swallow_empty_data_cols():
    html = cells_to_html_table(_p25x192_body_cells())
    assert not re.search(r'colspan="\d+"[^>]*>jER-604', html), html
    assert "<td>jER-604</td>" in html
    assert not re.search(r'colspan="\d+"[^>]*>850S', html), html
    assert "<td>850S</td>" in html
    assert not re.search(r'colspan="[3-9]"[^>]*>EP4003S', html), html
    assert "EP4003S" in html
    # 右侧空格应留下独立 td，而不是被标签盖住
    assert re.search(r"<td>jER-604</td>\s*<td></td>\s*<td>60</td>", html), html


def test_p25x193_synthesis2_keeps_empty_cols():
    cells = [
        _cell(0, 1, 0, 1, "", 0, 160, 0, 40),
        _cell(0, 0, 2, 5, "感光性树脂组合物(重量份)", 160, 480, 0, 20),
        _cell(1, 1, 2, 2, "比较例 1", 160, 240, 20, 40),
        _cell(1, 1, 3, 3, "比较例 2", 240, 320, 20, 40),
        _cell(1, 1, 4, 4, "比较例 3", 320, 400, 20, 40),
        _cell(1, 1, 5, 5, "比较例 4", 400, 480, 20, 40),
        _cell(2, 3, 0, 0, "(A)成分", 0, 80, 40, 80),
        _cell(2, 2, 1, 1, "合成例1的聚酰亚胺", 80, 160, 40, 60),
        _cell(2, 2, 2, 2, "100", 160, 240, 40, 60),
        _cell(2, 2, 3, 3, "100", 240, 320, 40, 60),
        _cell(2, 2, 4, 4, "100", 320, 400, 40, 60),
        _cell(2, 2, 5, 5, "", 400, 480, 40, 60),
        _cell(3, 3, 1, 4, "合成例2的聚酰亚胺", 80, 400, 60, 80),
        _cell(3, 3, 5, 5, "100", 400, 480, 60, 80),
        _cell(5, 5, 1, 5, "jER-604", 80, 480, 100, 120),
    ]
    html = cells_to_html_table(cells)
    assert "合成例2的聚酰亚胺" in html
    assert not re.search(r'colspan="4"[^>]*>合成例2', html), html
    assert not re.search(r'colspan="5"[^>]*>jER-604', html), html


def test_p26x194_ghost_col0_dropped():
    """TSR 在第 0 列留表头-only 幽灵列时，渲染前左移列号。"""
    cells = [
        _cell(0, 0, 0, 0, "", 0, 80, 0, 30),
        _cell(0, 0, 1, 1, "", 80, 160, 0, 30),
        _cell(0, 0, 2, 2, "低温热压接性", 160, 240, 0, 30),
        _cell(0, 0, 3, 3, "分辨率", 240, 320, 0, 30),
        _cell(1, 1, 1, 1, "实施例 1", 80, 160, 30, 60),
        _cell(1, 1, 2, 2, "G", 160, 240, 30, 60),
    ]
    html = cells_to_html_table(cells)
    assert re.search(r"<td></td>\s*<td>低温热压接性</td>", html), html
    assert re.search(r"<td>实施例 1</td>\s*<td>G</td>", html), html


def test_p26x194_header_corner_not_merged():
    """表头左上角空格保留；低温热压接性不得 colspan 吞掉索引列。"""
    cells = [
        _cell(0, 0, 0, 0, "", 0, 80, 0, 30),
        _cell(0, 0, 1, 1, "低温热压接性", 80, 160, 0, 30),
        _cell(0, 0, 2, 2, "分辨率", 160, 240, 0, 30),
        _cell(0, 0, 3, 3, "残膜率", 240, 320, 0, 30),
        _cell(0, 0, 4, 4, "高温时粘合强度(Mpa)", 320, 480, 0, 30),
        _cell(1, 1, 0, 0, "实施例 1", 0, 80, 30, 60),
        _cell(1, 1, 1, 1, "G", 80, 160, 30, 60),
        _cell(1, 1, 2, 2, "30/30", 160, 240, 30, 60),
        _cell(1, 1, 3, 3, "88", 240, 320, 30, 60),
        _cell(1, 1, 4, 4, "7", 320, 480, 30, 60),
    ]
    html = cells_to_html_table(cells)
    assert not re.search(r'colspan="2"[^>]*>低温热压接性', html), html
    assert re.search(r"<td></td>\s*<td>低温热压接性</td>", html), html


def test_p26x194_tsr_colspan2_header_split():
    """TSR 已输出 colspan=2 表头时，渲染阶段仍拆出索引列空角。"""
    cells = [
        _cell(0, 0, 0, 1, "低温热压接性", 0, 160, 0, 30),
        _cell(0, 0, 2, 2, "分辨率", 160, 240, 0, 30),
        _cell(0, 0, 3, 3, "残膜率", 240, 320, 0, 30),
        _cell(0, 0, 4, 4, "高温时粘合强度(Mpa)", 320, 480, 0, 30),
        _cell(1, 1, 0, 0, "实施例 1", 0, 80, 30, 60),
        _cell(1, 1, 1, 1, "G", 80, 160, 30, 60),
        _cell(1, 1, 2, 2, "30/30", 160, 240, 30, 60),
        _cell(1, 1, 3, 3, "88", 240, 320, 30, 60),
        _cell(1, 1, 4, 4, "7", 320, 480, 30, 60),
    ]
    html = cells_to_html_table(cells)
    assert not re.search(r'colspan="2"[^>]*>低温热压接性', html), html
    assert re.search(r"<td></td>\s*<td>低温热压接性</td>", html), html


def test_p98_aligned_body_colspan_kept():
    cells = [
        _cell(0, 1, 0, 0, "", 0, 40, 0, 40),
        _cell(0, 1, 1, 1, "聚合物", 40, 120, 0, 40),
        _cell(0, 0, 2, 7, "单体[摩尔比]", 120, 520, 0, 20),
        _cell(1, 1, 2, 5, "具有芳香族基团及环氧基的化合物", 120, 360, 20, 40),
        _cell(1, 1, 6, 6, "二羧酸酐", 360, 440, 20, 40),
        _cell(1, 1, 7, 7, "不饱和羧酸", 440, 520, 20, 40),
        _cell(2, 2, 0, 0, "合成例 15", 0, 40, 40, 80),
        _cell(2, 2, 1, 1, "酸改性环氧树脂溶液 (AE-1)", 40, 120, 40, 80),
        _cell(2, 2, 2, 5, "NC-7000L (环氧基准摩尔比：100)", 120, 360, 40, 80),
        _cell(2, 2, 6, 6, "THPHA (摩尔比: 80)", 360, 440, 40, 80),
        _cell(2, 2, 7, 7, "MAA (摩尔比: 100)", 440, 520, 40, 80),
    ]
    html = cells_to_html_table(cells)
    assert "NC-7000L" in html
    assert "THPHA" in html
    assert "MAA" in html


def test_relocate_fengduanji_from_data_column():
    """P24X176：封剂误入数据列时归位到空 rowspan 父格。"""
    cells = [
        _cell(2, 10, 0, 0, "聚酰亚胺组成", 0, 80, 40, 220),
        _cell(2, 3, 1, 1, "酸酐", 80, 160, 40, 80),
        _cell(4, 6, 1, 1, "二胺", 80, 160, 80, 140),
        _cell(7, 10, 1, 1, "", 80, 160, 140, 200),
        _cell(7, 7, 2, 3, "端MAP", 160, 280, 140, 160),
        _cell(10, 10, 2, 3, "", 160, 280, 190, 200),
        _cell(10, 10, 4, 4, "封剂", 280, 360, 190, 200),
    ]
    out = relocate_misplaced_category_labels([dict(c) for c in cells])
    parent = next(c for c in out if int(c["row_start"]) == 7 and int(c["col_start"]) == 1)
    wrong = next(c for c in out if int(c["row_start"]) == 10 and int(c["col_start"]) == 4)
    assert str(parent.get("text") or "") == "封端剂"
    assert not str(wrong.get("text") or "").strip()


def test_sticky_column_header_split_wide_merged():
    """P25X177：宽 colspan 顶栏粘连时按 OCR 框均分。"""
    blob = "实施例11实施例12比较例1参考例1"
    tb = {
        "text": blob,
        "polygon": [[400, 0], [800, 0], [800, 30], [400, 30]],
        "top_left": (400, 0),
    }
    cells = [
        _cell(0, 1, 8, 12, blob, 400, 800, 0, 30),
    ]
    pieces = _split_sticky_column_header(tb, cells)
    assert pieces is not None
    assert len(pieces) == 4
    texts = [p["text"] for p in pieces]
    assert "11" in texts[0]
    assert "比较例" in texts[2]


def test_sticky_mixed_example_compare_ref_parse():
    parts = _parse_sticky_column_parts(
        "实施例11实施例12实施例13比较例1比较例2参考例1"
    )
    assert parts is not None
    assert len(parts) >= 5
    assert any("比较例" in p for p in parts)
    assert any("参考例" in p for p in parts)


def test_repair_lone_example_number_header():
    cells = [
        _cell(0, 0, 4, 4, "实施例 8", 320, 400, 0, 30),
        _cell(0, 0, 5, 5, "实施例 9", 400, 480, 0, 30),
        _cell(0, 0, 6, 6, "10", 480, 560, 0, 30),
    ]
    out = _repair_lone_example_number_headers([dict(c) for c in cells])
    last = next(c for c in out if int(c["col_start"]) == 6)
    assert str(last.get("text") or "") == "实施例 10"
    html = cells_to_html_table(out)
    assert "实施例 10" in html


def test_sticky_strip_bare_example_prefix():
    parts = _parse_sticky_column_parts(
        "实施例实施例实施例比较例1比较例2参考例1"
    )
    assert parts is not None
    assert len(parts) >= 3
    assert any("比较例" in p for p in parts)


def test_p24_stub_ghost_column_covered():
    """P24：行头区多出的 A/B/C 列不应在 ODPA/其他/分辨率行切出空 td。"""
    cells = [
        _cell(0, 1, 4, 4, "实施例 1", 320, 400, 0, 40),
        _cell(0, 1, 5, 5, "实施例 2", 400, 480, 0, 40),
        _cell(0, 0, 0, 3, "项目", 0, 320, 0, 20),
        _cell(1, 1, 0, 3, "聚酰亚胺", 0, 320, 20, 40),
        _cell(2, 5, 0, 0, "聚酰亚胺组成", 0, 80, 40, 160),
        _cell(2, 3, 1, 1, "酸酐", 80, 160, 40, 80),
        _cell(2, 2, 2, 2, "ODPA", 160, 240, 40, 60),
        _cell(2, 2, 3, 3, "", 240, 320, 40, 60),
        _cell(2, 2, 4, 4, "100", 320, 400, 40, 60),
        _cell(2, 2, 5, 5, "100", 400, 480, 40, 60),
        _cell(3, 3, 2, 2, "BPDA", 160, 240, 60, 80),
        _cell(3, 3, 3, 3, "", 240, 320, 60, 80),
        _cell(3, 3, 4, 4, "-", 320, 400, 60, 80),
        _cell(4, 4, 1, 2, "(c)醌二叠氮化合物", 80, 240, 80, 100),
        _cell(4, 4, 3, 3, "C", 240, 320, 80, 100),
        _cell(4, 4, 4, 4, "-", 320, 400, 80, 100),
        _cell(5, 5, 1, 1, "其他", 80, 160, 100, 120),
        _cell(5, 5, 2, 2, "", 160, 240, 100, 120),
        _cell(5, 5, 3, 3, "合成例 10 的丙烯酸树脂溶液", 240, 320, 100, 120),
        _cell(5, 5, 4, 4, "5", 320, 400, 100, 120),
        _cell(6, 6, 0, 2, "分辨率(μm)", 0, 240, 120, 140),
        _cell(6, 6, 4, 4, "25", 320, 400, 120, 140),
        _cell(7, 7, 1, 1, "(b)化合物", 80, 160, 140, 160),
        _cell(7, 7, 3, 3, "OXT-191", 240, 320, 140, 160),
        _cell(7, 7, 4, 4, "20", 320, 400, 140, 160),
    ]
    html = cells_to_html_table(cells)
    assert not re.search(r"<td>ODPA</td>\s*<td></td>", html), html
    assert re.search(r'colspan="2"[^>]*>ODPA', html), html
    assert not re.search(r"<td>其他</td>\s*<td></td>", html), html
    assert "合成例 10" in html
    assert not re.search(r"分辨率\(μm\)</td>\s*<td></td>", html), html
    assert re.search(r'colspan="4"[^>]*>分辨率', html), html
    assert re.search(r"<td>C</td>", html), html


def test_p24_header_empty_corners_collapsed():
    """P24/P25：表头角格并入项目/聚酰亚胺，不出现 3 连空 td + 独立项目。"""
    cells = [
        _cell(1, 1, 0, 3, "项目", 0, 320, 0, 30),
        _cell(1, 1, 4, 4, "实施例 1", 320, 400, 0, 30),
        _cell(2, 2, 0, 3, "聚酰亚胺组成(摩尔比)", 0, 320, 30, 60),
        _cell(3, 10, 0, 0, "聚酰亚胺组成", 0, 80, 60, 220),
        _cell(3, 3, 1, 1, "酸酐", 80, 160, 60, 100),
    ]
    html = cells_to_html_table(cells)
    assert not re.search(r"<td></td>\s*<td></td>\s*<td></td>\s*<td>项目", html), html
    assert "项目" in html


def test_peel_b_compound_colspan2_parent():
    """(b)化合物 父格 colspan=2 粘连 jER828 时剥到子行。"""
    cells = [
        _cell(14, 20, 0, 1, "化合物jER828\n(b)", 0, 160, 280, 420),
        _cell(14, 14, 2, 2, "", 160, 280, 280, 300),
    ]
    out = peel_row_header_text([dict(c) for c in cells])
    parent = next(c for c in out if int(c["row_start"]) == 14 and int(c["col_start"]) == 0)
    child = next(c for c in out if int(c["row_start"]) == 14 and int(c["col_start"]) == 2)
    assert "jER828" in str(child.get("text") or "")
    assert "(b)化合物" in re.sub(r"\s+", "", str(parent.get("text") or ""))


def test_peel_c_abc_sublabels():
    cells = [
        _cell(21, 23, 0, 1, "(c)醌二A\nA\nB\nC", 0, 160, 400, 460),
        _cell(21, 21, 2, 2, "", 160, 280, 400, 420),
        _cell(22, 22, 2, 2, "", 160, 280, 420, 440),
        _cell(23, 23, 2, 2, "", 160, 280, 440, 460),
    ]
    out = peel_row_header_text([dict(c) for c in cells])
    texts = {
        (int(c["row_start"]), int(c["col_start"])): str(c.get("text") or "")
        for c in out
    }
    assert texts.get((21, 2), "").strip() == "A"
    assert texts.get((22, 2), "").strip() == "B"
    assert texts.get((23, 2), "").strip() == "C"


def test_relocate_aniline_subrow_label():
    cells = [
        _cell(10, 10, 1, 1, "", 80, 160, 190, 210),
        _cell(10, 10, 4, 4, "苯胺", 280, 360, 190, 210),
    ]
    out = relocate_misplaced_category_labels([dict(c) for c in cells])
    texts = {int(c["col_start"]): str(c.get("text") or "") for c in out if int(c["row_start"]) == 10}
    assert texts.get(1, "").strip() == "苯胺"
    assert texts.get(4, "").strip() == ""


def test_wufapingjia_tail_to_right_on_adhesion_stress_rows():
    """粘合/应力行：无法评价+尾数或独立数字应写入右邻空列。"""
    from src.matching.matching import assign_texts_to_cells

    import numpy as np

    cells = [
        _cell(0, 0, 0, 3, "粘合强度（MPa）", 0, 320, 0, 30),
        _cell(0, 0, 4, 4, "", 320, 400, 0, 30),
        _cell(0, 0, 5, 5, "", 400, 480, 0, 30),
        _cell(1, 1, 0, 3, "应力（MPa）", 0, 320, 30, 60),
        _cell(1, 1, 4, 4, "", 320, 400, 30, 60),
        _cell(1, 1, 5, 5, "", 400, 480, 30, 60),
    ]
    boxes = [
        {
            "text": "无法评价",
            "polygon": np.array(
                [[330.0, 5.0], [390.0, 5.0], [390.0, 25.0], [330.0, 25.0]],
                dtype=np.float64,
            ),
            "score": 0.9,
        },
        {
            "text": "21",
            "polygon": np.array(
                [[410.0, 5.0], [470.0, 5.0], [470.0, 25.0], [410.0, 25.0]],
                dtype=np.float64,
            ),
            "score": 0.9,
        },
        {
            "text": "无法评价25",
            "polygon": np.array(
                [[330.0, 35.0], [470.0, 35.0], [470.0, 55.0], [330.0, 55.0]],
                dtype=np.float64,
            ),
            "score": 0.9,
        },
    ]
    out, _ = assign_texts_to_cells([dict(c) for c in cells], boxes)
    texts = {
        (int(c["row_start"]), int(c["col_start"])): str(c.get("text") or "").strip()
        for c in out
    }
    assert texts.get((0, 4)) == "无法评价", texts
    assert texts.get((0, 5)) == "21", texts
    assert texts.get((1, 4)) == "无法评价", texts
    assert texts.get((1, 5)) == "25", texts


def test_wufapingjia_tail_to_right_on_resolution_row():
    """IoA 只落一格时，分辨率行尾数写入右邻空列。"""
    from src.matching.matching import assign_texts_to_cells

    cells = [
        _cell(0, 0, 0, 3, "分辨率（μm）", 0, 320, 0, 30),
        _cell(0, 0, 4, 4, "", 320, 400, 0, 30),
        _cell(0, 0, 5, 5, "", 400, 480, 0, 30),
    ]
    import numpy as np

    tb = {
        "text": "无法评价40",
        "polygon": np.array(
            [[330.0, 5.0], [470.0, 5.0], [470.0, 25.0], [330.0, 25.0]],
            dtype=np.float64,
        ),
        "score": 0.9,
    }
    out, _ = assign_texts_to_cells([dict(c) for c in cells], [tb])
    texts = {int(c["col_start"]): str(c.get("text") or "").strip() for c in out}
    assert texts.get(4) == "无法评价", texts
    assert texts.get(5) == "40", texts


def test_wufapingjia_glued_number_split():
    """无法评价+数字跨两列切开；单格丢掉尾数；normalize 不再整段删数字。"""
    import numpy as np

    assert normalize_ocr_text("无法评价40") == "无法评价40"
    assert normalize_ocr_text("无法评价21") == "无法评价21"

    two_cols = [
        _cell(0, 0, 0, 0, "", 0, 100, 0, 30),
        _cell(0, 0, 1, 1, "", 100, 200, 0, 30),
    ]
    tb = {
        "text": "无法评价40",
        "polygon": np.array(
            [[10.0, 5.0], [190.0, 5.0], [190.0, 25.0], [10.0, 25.0]],
            dtype=np.float64,
        ),
        "score": 0.9,
    }
    pieces = _split_wufapingjia_glued(tb, two_cols)
    assert pieces is not None
    assert [p["text"] for p in pieces] == ["无法评价", "40"]

    one_col = [_cell(0, 0, 0, 0, "", 0, 200, 0, 30)]
    pieces = _split_wufapingjia_glued(tb, one_col)
    assert pieces is not None
    assert len(pieces) == 1
    assert pieces[0]["text"] == "无法评价"


def test_eval_symbols_detect_cross_mark():
    """空单元格中的 × 应被 CV 符号检测补全。"""
    import cv2
    import numpy as np

    img = np.zeros((36, 120), dtype=np.uint8)
    cx, cy, r = 90, 18, 10
    cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), 255, 2)
    cv2.line(img, (cx - r, cy + r), (cx + r, cy - r), 255, 2)
    cells = [
        _cell(0, 0, 0, 0, "比较例5", 0, 80, 0, 36),
        _cell(0, 0, 1, 1, "", 80, 120, 0, 36),
    ]
    out = detect_eval_symbols_in_empty_cells([dict(c) for c in cells], img)
    empty = next(c for c in out if int(c["col_start"]) == 1)
    assert str(empty.get("text") or "").strip() == "×"


def test_eval_symbols_skip_numeric_metric_row():
    """分辨率行已有数字时，空格不得被轮廓填成 ◎。"""
    import cv2
    import numpy as np

    img = np.zeros((40, 300), dtype=np.uint8)
    cv2.circle(img, (250, 20), 10, 255, 2)
    cv2.circle(img, (250, 20), 5, 255, 1)
    cells = [
        _cell(0, 0, 0, 0, "分辨率", 0, 80, 0, 40),
        _cell(0, 0, 1, 1, "21", 80, 160, 0, 40),
        _cell(0, 0, 2, 2, "40", 160, 220, 0, 40),
        _cell(0, 0, 3, 3, "", 220, 300, 0, 40),
    ]
    out = detect_eval_symbols_in_empty_cells([dict(c) for c in cells], img)
    empty = next(c for c in out if int(c["col_start"]) == 3)
    assert str(empty.get("text") or "").strip() == ""


def test_peel_c_fills_missing_b():
    """(c) 已有 A/C、中间空时补 B，父格规范为醌二叠氮化合物。"""
    cells = [
        _cell(21, 23, 0, 1, "(c)醌二A", 0, 160, 400, 460),
        _cell(21, 21, 2, 2, "A", 160, 280, 400, 420),
        _cell(22, 22, 2, 2, "", 160, 280, 420, 440),
        _cell(23, 23, 2, 2, "C", 160, 280, 440, 460),
    ]
    out = peel_row_header_text([dict(c) for c in cells])
    texts = {
        (int(c["row_start"]), int(c["col_start"])): str(c.get("text") or "").strip()
        for c in out
    }
    assert texts.get((21, 2)) == "A"
    assert texts.get((22, 2)) == "B"
    assert texts.get((23, 2)) == "C"
    parent = next(c for c in out if int(c["col_start"]) == 0)
    ptxt = re.sub(r"\s+", "", str(parent.get("text") or ""))
    assert ptxt == "(c)醌二叠氮化合物"


def test_relocate_photosensitive_section_and_map():
    """感光性组前缀归位大类行头；端MAP → MAP。"""
    cells = [
        _cell(2, 13, 0, 0, "合物组成(重量份）", 0, 80, 60, 400),
        _cell(2, 2, 1, 3, "感光性组（a）聚酰亚胺", 80, 280, 60, 90),
        _cell(8, 8, 1, 1, "端MAP", 80, 160, 200, 230),
    ]
    out = peel_row_header_text([dict(c) for c in cells])
    section = next(c for c in out if int(c["col_start"]) == 0)
    assert "感光性组合物组成" in re.sub(r"\s+", "", str(section.get("text") or ""))
    a_cell = next(c for c in out if int(c["row_start"]) == 2 and int(c["col_start"]) == 1)
    assert not str(a_cell.get("text") or "").startswith("感光性组")
    assert "聚酰亚胺" in str(a_cell.get("text") or "")
    map_cell = next(c for c in out if int(c["row_start"]) == 8)
    assert str(map_cell.get("text") or "").strip() == "MAP"


def test_explode_sticky_header_wide_cell():
    from src.matching.matching import explode_sticky_header_wide_cells

    cells = [
        _cell(
            0,
            1,
            8,
            12,
            "实施例实施例比较例1比较例2参考例1",
            400,
            800,
            0,
            30,
        ),
    ]
    out = explode_sticky_header_wide_cells([dict(c) for c in cells])
    assert len(out) == 3
    texts = {str(c.get("text") or "") for c in out}
    assert any("比较例" in t for t in texts)
    assert all(int(c["col_span"]) == 1 for c in out)


def test_extend_section_rowspan_over_metric_rows():
    """P24/P25：大类行头应收缩排除测量行，测量行独立全宽标签。"""
    cells = [
        _cell(2, 12, 0, 0, "聚酰亚胺组成（摩尔比)", 0, 80, 40, 260),
        _cell(2, 3, 1, 1, "酸酐", 80, 160, 40, 80),
        _cell(11, 11, 0, 0, "", 0, 80, 220, 240),
        _cell(11, 11, 1, 3, "聚酰亚胺的酰亚胺化率(%)", 80, 320, 220, 240),
        _cell(11, 11, 4, 4, "95", 320, 400, 220, 240),
        _cell(12, 12, 0, 3, "聚酰亚胺重均分子量", 0, 320, 240, 260),
        _cell(12, 12, 4, 4, "19900", 320, 400, 240, 260),
    ]
    out = extend_section_rowspan_over_metric_rows([dict(c) for c in cells])
    parent = next(c for c in out if "聚酰亚胺组成" in str(c.get("text") or ""))
    # parent 收缩到 row 10，不再覆盖测量行 11/12
    assert int(parent["row_end"]) == 10
    assert int(parent["row_span"]) == 9
    # 酰亚胺化率标签从 col0 开始，独立全宽
    metric1 = next(c for c in out if "酰亚胺化率" in str(c.get("text") or ""))
    assert int(metric1["col_start"]) == 0
    # 重均分子量标签从 col0 开始
    metric2 = next(c for c in out if "重均分子量" in str(c.get("text") or ""))
    assert int(metric2["col_start"]) == 0
    # 空占位 col0 单元格应被丢弃
    assert not any(
        int(c["row_start"]) == 11 and int(c["col_start"]) == 0
        and not str(c.get("text") or "").strip()
        for c in out
    )
    html = cells_to_html_table(out)
    assert "聚酰亚胺的酰亚胺化率" in html
    assert "聚酰亚胺重均分子量" in html


def test_p40_aa_independent_label_parse():
    assert is_independent_row_label("(AA)")
    assert is_independent_row_label("(AZ)")
    labels = extract_independent_labels_from_joined("(AA)(AB)(AC)")
    assert labels == ["(AA)", "(AB)", "(AC)"]
    assert are_independent_row_labels(["(AA)", "(AB)", "(AC)"])


def test_p40_unmerge_aa_rowspan():
    """P40：左列大 rowspan 粘连 (AA)(AB)(AC) 应拆成逐行。"""
    cells = [
        _cell(0, 2, 0, 0, "(AA)(AB)(AC)", 0, 80, 0, 60),
    ]
    out = unmerge_filled_label_rowspans([dict(c) for c in cells])
    assert len(out) == 3
    texts = [str(c.get("text") or "") for c in out]
    assert texts == ["(AA)", "(AB)", "(AC)"]
    assert all(int(c["row_span"]) == 1 for c in out)


def test_example_header_does_not_swallow_letter_row():
    """P24：实施例不得 rowspan 吞掉下层 A/B/C 数据字母。"""
    cells = [
        _cell(0, 0, 0, 3, "项目", 0, 320, 0, 30),
        _cell(0, 1, 4, 4, "实施例 1", 320, 400, 0, 60),
        _cell(0, 1, 5, 5, "实施例 2", 400, 480, 0, 60),
        _cell(1, 1, 0, 3, "聚酰亚胺", 0, 320, 30, 60),
        _cell(1, 1, 4, 4, "A", 320, 400, 30, 60),
        _cell(1, 1, 5, 5, "B", 400, 480, 30, 60),
    ]
    split = _split_example_header_rowspans([dict(c) for c in cells])
    assert all(
        int(c["row_span"]) == 1
        for c in split
        if "实施例" in str(c.get("text") or "")
    )
    html = cells_to_html_table([dict(c) for c in cells])
    assert not re.search(r'rowspan="2"[^>]*>实施例', html), html
    assert re.search(r"<td>A</td>", html), html
    assert re.search(r"<td>B</td>", html), html
    assert "聚酰亚胺" in html


def test_peel_aniline_not_self_cleared():
    """苯胺 colspan=2 不得被 peel 清空。"""
    cells = [
        _cell(10, 10, 1, 2, "苯胺", 80, 240, 190, 210),
        _cell(10, 10, 3, 3, "-", 240, 320, 190, 210),
    ]
    out = peel_row_header_text([dict(c) for c in cells])
    texts = {str(c.get("text") or "").strip() for c in out}
    assert "苯胺" in texts


def test_peel_b_inserts_sibling_for_jer828():
    """(b) 粘连 jER828 且无兄弟格时插入后剥离。"""
    parent = _cell(14, 20, 0, 1, "化合物jER828\n(b)", 0, 160, 280, 420)
    # 仅有后续品名行，首行无标签格
    child = _cell(15, 15, 2, 2, "OXT-191", 160, 280, 300, 320)
    out = peel_row_header_text([dict(parent), dict(child)])
    texts = {str(c.get("text") or "").strip() for c in out}
    assert "jER828" in texts
    parent_out = next(c for c in out if int(c["row_start"]) == 14 and int(c["col_start"]) == 0)
    assert "jER828" not in str(parent_out.get("text") or "")
    assert "(b)化合物" in re.sub(r"\s+", "", str(parent_out.get("text") or ""))


def test_peel_c_glued_ab_letters():
    """(c)醌二A / 叠氮化合B 粘连字母应剥到子行。"""
    cells = [
        _cell(21, 23, 0, 1, "(c)醌二A\n叠氮化合B\n物", 0, 160, 400, 460),
        _cell(23, 23, 2, 2, "C", 160, 280, 440, 460),
    ]
    out = peel_row_header_text([dict(c) for c in cells])
    by_row = {
        int(c["row_start"]): str(c.get("text") or "").strip()
        for c in out
        if int(c["col_start"]) >= 2
    }
    assert by_row.get(21) == "A", by_row
    assert by_row.get(22) == "B", by_row
    assert by_row.get(23) == "C", by_row
    parent = next(c for c in out if int(c["col_start"]) == 0)
    ptxt = re.sub(r"\s+", "", str(parent.get("text") or ""))
    assert not ptxt.endswith("A")
    assert "B" not in ptxt or "叠氮" in ptxt


def main() -> int:
    test_body_left_stub_does_not_merge_column_headers()
    test_p100_fensan_still_merges_in_header_band()
    test_p40_stacked_chemical_header()
    test_clip_nested_row_header_parent_colspan()
    test_html_keeps_nested_labels_separate()
    test_resolve_does_not_swallow_right_child_text()
    test_light_refine_clips_before_dedupe()
    test_clip_same_col_start_name_cell()
    test_p100_ghost_not_clipped_by_row_header()
    test_peel_sublabel_from_parent()
    test_clip_narrow_odpa_colspan()
    test_peel_jer828_from_compound_parent()
    test_sticky_column_header_parse()
    test_sticky_column_header_split_geom()
    test_merge_leading_empty_into_label()
    test_merge_resolution_unit()
    test_resolution_um_merged_on_metric_data_row()
    test_resolution_um_separate_when_unit_is_column()
    test_no_merge_other_with_um_unit()
    test_p25x192_label_does_not_swallow_empty_data_cols()
    test_p25x193_synthesis2_keeps_empty_cols()
    test_p26x194_ghost_col0_dropped()
    test_p26x194_header_corner_not_merged()
    test_p26x194_tsr_colspan2_header_split()
    test_p24_stub_ghost_column_covered()
    test_p24_header_empty_corners_collapsed()
    test_peel_b_compound_colspan2_parent()
    test_peel_c_abc_sublabels()
    test_relocate_aniline_subrow_label()
    test_wufapingjia_glued_number_split()
    test_wufapingjia_tail_to_right_on_resolution_row()
    test_wufapingjia_tail_to_right_on_adhesion_stress_rows()
    test_eval_symbols_detect_cross_mark()
    test_eval_symbols_skip_numeric_metric_row()
    test_peel_c_fills_missing_b()
    test_relocate_photosensitive_section_and_map()
    test_p98_aligned_body_colspan_kept()
    test_relocate_fengduanji_from_data_column()
    test_sticky_column_header_split_wide_merged()
    test_sticky_mixed_example_compare_ref_parse()
    test_repair_lone_example_number_header()
    test_sticky_strip_bare_example_prefix()
    test_explode_sticky_header_wide_cell()
    test_extend_section_rowspan_over_metric_rows()
    test_p40_aa_independent_label_parse()
    test_p40_unmerge_aa_rowspan()
    test_example_header_does_not_swallow_letter_row()
    test_peel_aniline_not_self_cleared()
    test_peel_b_inserts_sibling_for_jer828()
    test_peel_c_glued_ab_letters()
    print("OK: row header tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
