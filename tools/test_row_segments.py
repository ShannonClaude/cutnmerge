"""find_row_segments 回归：多级表头不误切；表体后重复表头仍切；段不重叠。

用法:
    python tools/test_row_segments.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.segments import find_row_segments  # noqa: E402


def _cell(
    rs: int,
    re: int,
    cs: int,
    ce: int,
    text: str,
    *,
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
        "polygon": [[0.0, y0], [10.0, y0], [10.0, y1], [0.0, y1]],
        "y_key": y0,
    }


def _assert_no_overlap(segs: List[tuple[int, int]]) -> None:
    for i in range(len(segs) - 1):
        assert segs[i][1] < segs[i + 1][0], f"segments overlap: {segs}"


def test_multilevel_header_not_split() -> None:
    """P96 类：子表头与父表头共享长「氟」文，表体前不得切。"""
    parent = (
        "聚合物 单体[摩尔比] "
        "来自具有氟原子的单体的结构单元在来自全部羧酸衍生物的结构单元中所占的含有比率 "
        "来自具有氟原子的单体的结构单元在来自全部胺衍生物的结构单元中所占的含有比率"
    )
    sub = (
        "四羧酸及其衍生物 二胺及其衍生物 封端剂 "
        "来自具有氟原子的单体的结构单元在全部结构单元中所占的含有比率"
    )
    cells = [
        _cell(0, 1, 0, 0, "聚合物", y0=0, y1=40),
        _cell(0, 0, 1, 6, parent, y0=0, y1=20),
        _cell(1, 1, 1, 2, sub, y0=20, y1=40),
        _cell(2, 2, 0, 0, "合成例1", y0=40, y1=55),
        _cell(2, 2, 1, 1, "ODPA(100)", y0=40, y1=55),
        _cell(3, 3, 0, 0, "合成例2", y0=55, y1=70),
        _cell(3, 3, 1, 1, "ODPA(100)", y0=55, y1=70),
    ]
    segs = find_row_segments(cells)
    _assert_no_overlap(segs)
    assert len(segs) == 1, f"expected 1 segment, got {segs}"
    assert segs[0][0] == 0 and segs[0][1] >= 3, segs


def test_repeat_header_after_body_still_splits() -> None:
    """P98 类：表体后再出现聚合物/单体[，仍应拆成多段。"""
    cells = [
        _cell(0, 0, 0, 0, "聚合物", y0=0, y1=15),
        _cell(0, 0, 1, 3, "单体[摩尔比]", y0=0, y1=15),
        _cell(1, 1, 0, 0, "合成例13", y0=15, y1=30),
        _cell(1, 1, 1, 1, "BHPF(100)", y0=15, y1=30),
        _cell(2, 2, 0, 0, "合成例14", y0=30, y1=45),
        _cell(2, 2, 1, 1, "BGPF(100)", y0=30, y1=45),
        # 新段表头（与上一段紧邻，靠 section/Jaccard 切，不靠 Y 间距）
        _cell(3, 3, 0, 0, "聚合物", y0=45, y1=60),
        _cell(3, 3, 1, 3, "单体[摩尔比]", y0=45, y1=60),
        _cell(4, 4, 0, 0, "合成例15", y0=60, y1=75),
        _cell(4, 4, 1, 1, "NC-7000L", y0=60, y1=75),
    ]
    segs = find_row_segments(cells)
    _assert_no_overlap(segs)
    assert len(segs) >= 2, f"expected >=2 segments after body, got {segs}"
    assert segs[0][1] < segs[1][0]
    assert segs[1][0] == 3, segs


def test_split_clamp_no_row_overlap() -> None:
    """切分时上一段终点不得包含新段起始行（rowspan 扩展场景）。"""
    cells = [
        _cell(0, 1, 0, 0, "聚合物", y0=0, y1=40),
        _cell(0, 0, 1, 2, "单体[摩尔比]", y0=0, y1=20),
        _cell(1, 1, 1, 1, "封端剂", y0=20, y1=40),
        _cell(2, 2, 0, 0, "合成例1", y0=40, y1=55),
        _cell(3, 3, 0, 0, "聚合物", y0=55, y1=70),
        _cell(3, 3, 1, 2, "单体[摩尔比]", y0=55, y1=70),
        _cell(4, 4, 0, 0, "合成例10", y0=70, y1=85),
    ]
    segs = find_row_segments(cells)
    _assert_no_overlap(segs)
    assert len(segs) >= 2, segs


def test_data_rows_not_split_by_jaccard_or_small_gap() -> None:
    """P46 类：相邻合成例数据行不得因 Jaccard/小间距被切成多段。"""
    cells = [
        _cell(0, 0, 0, 0, "聚合物", y0=0, y1=20),
        _cell(0, 0, 1, 4, "单体[mol比]", y0=0, y1=20),
        _cell(1, 1, 1, 1, "二羧酸及其衍生物", y0=20, y1=40),
        _cell(1, 1, 2, 3, "双氨基酚化合物及其衍生物", y0=20, y1=40),
        _cell(1, 1, 4, 4, "封端剂", y0=20, y1=40),
        # 表体：两行相似单体组成，物理间距略超默认 48 但仍属同表
        _cell(2, 2, 0, 0, "合成例3", y0=50, y1=90),
        _cell(2, 2, 1, 1, "聚苯并噁唑(PBO-1)", y0=50, y1=90),
        _cell(2, 2, 2, 2, "BFE(80)", y0=50, y1=90),
        _cell(2, 2, 3, 3, "BAHF(95)", y0=50, y1=90),
        _cell(2, 2, 4, 4, "NA(40)", y0=50, y1=90),
        _cell(3, 3, 0, 0, "合成例4", y0=140, y1=180),
        _cell(3, 3, 1, 1, "聚苯并噁唑前体(PBOP-1)", y0=140, y1=180),
        _cell(3, 3, 2, 2, "BFE(80)", y0=140, y1=180),
        _cell(3, 3, 3, 3, "BAHF(95)", y0=140, y1=180),
        _cell(3, 3, 4, 4, "NA(40)", y0=140, y1=180),
        # 段后重复表头：仍应切开
        _cell(4, 4, 0, 0, "聚合物", y0=190, y1=210),
        _cell(4, 4, 1, 4, "单体[mol%]", y0=190, y1=210),
        _cell(5, 5, 0, 0, "合成例5", y0=210, y1=230),
        _cell(5, 5, 1, 1, "聚硅氧烷溶液(PS-1)", y0=210, y1=230),
    ]
    segs = find_row_segments(cells)
    _assert_no_overlap(segs)
    assert len(segs) == 2, f"expected 2 segments (mid header), got {segs}"
    assert segs[0][0] == 0 and segs[0][1] >= 3, segs
    assert segs[1][0] == 4, segs


def test_multilevel_header_monomer_polymer_not_split_by_gap() -> None:
    """P47 类：多级表头「单体」与「聚合物」夹缝 >48px 时不得互切。"""
    cells = [
        _cell(0, 0, 2, 5, "单体 [mol比]", y0=0, y1=18),
        # 间隙 60px：旧逻辑会把「聚合物」当段首表头切开
        _cell(1, 2, 1, 1, "聚合物", y0=78, y1=110),
        _cell(2, 2, 2, 2, "具有酸性基团的共聚成分", y0=90, y1=110),
        _cell(2, 2, 3, 3, "具有芳香族基团的共聚成分", y0=90, y1=110),
        _cell(3, 3, 0, 0, "合成例6", y0=110, y1=130),
        _cell(3, 3, 1, 1, "丙烯酸树脂溶液(AC-1)", y0=110, y1=130),
        _cell(3, 3, 2, 2, "MAA(50)", y0=110, y1=130),
    ]
    segs = find_row_segments(cells)
    _assert_no_overlap(segs)
    assert len(segs) == 1, f"expected single segment, got {segs}"


def main() -> int:
    test_multilevel_header_not_split()
    test_repeat_header_after_body_still_splits()
    test_split_clamp_no_row_overlap()
    test_data_rows_not_split_by_jaccard_or_small_gap()
    test_multilevel_header_monomer_polymer_not_split_by_gap()
    print("OK: all find_row_segments regressions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
