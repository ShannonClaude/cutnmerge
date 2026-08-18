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


def main() -> int:
    test_multilevel_header_not_split()
    test_repeat_header_after_body_still_splits()
    test_split_clamp_no_row_overlap()
    print("OK: all find_row_segments regressions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
