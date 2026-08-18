"""OCR 文本补全：等级粘连、截断表头、叠放顺序。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.label_patterns import (
    complete_truncated_component_header,
    split_value_grade,
)
from src.matching.matching import _order_value_grade_parts, join_cell_texts
from src.ocr.reocr import _looks_truncated_header


def test_split_value_grade():
    assert split_value_grade("40A") == ("40", "A")
    assert split_value_grade("40 A") == ("40", "A")
    assert split_value_grade("A40") == ("40", "A")
    assert split_value_grade("25 +") == ("25", "A+")
    assert split_value_grade("30+") == ("30", "A+")
    assert split_value_grade("A2") is None  # 树脂代号，勿拆
    assert split_value_grade("Bk-2") is None


def test_complete_component_header():
    donors = ["(A1)\n第1树脂", "(A2)", "(B)\n自由基"]
    assert complete_truncated_component_header("(A2)", donors) == "(A2)\n第2树脂"
    assert complete_truncated_component_header("(A2)\n第2", donors) == "(A2)\n第2树脂"
    assert complete_truncated_component_header("(A2)\n第2\n树脂", donors) == "(A2)\n第2树脂"
    assert complete_truncated_component_header("(A1)\n第1树脂", donors) == "(A1)\n第1树脂"
    assert complete_truncated_component_header("(B)", donors) == "(B)"


def test_truncated_header_flag():
    assert _looks_truncated_header("(A2)")
    assert _looks_truncated_header("(A2)\n第2")
    assert not _looks_truncated_header("(A2)\n第2树脂")
    assert not _looks_truncated_header("PI-1")


def test_grade_order():
    assert _order_value_grade_parts(["A", "35"]) == ["35", "A"]
    assert _order_value_grade_parts(["35", "A"]) == ["35", "A"]
    boxes = [
        {
            "text": "A",
            "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
        },
        {
            "text": "35",
            "polygon": [[0, 12], [20, 12], [20, 24], [0, 24]],
        },
    ]
    assert join_cell_texts(boxes) == "35\nA"


def main():
    test_split_value_grade()
    test_complete_component_header()
    test_truncated_header_flag()
    test_grade_order()
    print("ok")


if __name__ == "__main__":
    main()
