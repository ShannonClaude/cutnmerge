"""窄序号列不应被 drop_evidenceless / merge_ghost 吃掉。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.html_formatter import cells_to_html, drop_evidenceless_columns
from src.label_patterns import is_index_column
from src.tsr_refine import merge_ghost_columns


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


def test_is_index_column():
    assert is_index_column(["16", "17", "18", "19", "20"])
    assert is_index_column(["", "16", "17", "18", ""])  # empties ignored
    assert not is_index_column(["16", "17"])  # <3 nonempty
    assert not is_index_column(["3", "α", "L"])  # sparse crumbs
    assert not is_index_column(["16", "OXL-37", "18", "19"])  # mixed


def test_drop_keeps_index_after_caption_lift():
    cells = []
    # row0 caption spanning all cols (will be lifted)
    cells.append(_cell(0, 0, 0, 2, "表2-3", 0, 300, 0, 20))
    # header
    cells.append(_cell(1, 1, 0, 0, "", 0, 35, 20, 50))
    cells.append(_cell(1, 1, 1, 1, "肟酯系光聚合引发剂", 35, 130, 20, 50))
    cells.append(_cell(1, 1, 2, 2, "物性", 130, 300, 20, 50))
    for i, n in enumerate(range(16, 21)):
        y = 50 + i * 30
        cells.append(_cell(2 + i, 2 + i, 0, 0, str(n), 0, 35, y, y + 28))
        cells.append(_cell(2 + i, 2 + i, 1, 1, f"OXL-{n}", 35, 130, y, y + 28))
        cells.append(_cell(2 + i, 2 + i, 2, 2, "-", 130, 300, y, y + 28))

    html = cells_to_html(cells, compress_empty=True)
    assert re.search(r"<p>表2-3</p>", html)
    assert re.search(r">16<", html), html[:500]
    assert "OXL-16" in html
    # first data row should keep index then OXL
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)
    data = None
    for row in rows:
        if re.search(r">16<", row):
            data = row
            break
    assert data is not None
    cells_txt = [
        re.sub(r"<[^>]+>", "", c).strip()
        for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", data, re.S)
    ]
    assert cells_txt[0] == "16"
    assert cells_txt[1] == "OXL-16"


def test_drop_still_removes_sparse_digit_crumbs():
    cells = [
        _cell(0, 0, 0, 0, "3", 0, 20),
        _cell(0, 0, 1, 1, "聚合物", 20, 200),
        _cell(1, 1, 0, 0, "", 0, 20, 30, 60),
        _cell(1, 1, 1, 1, "单体", 20, 200, 30, 60),
        _cell(2, 2, 0, 0, "α", 0, 20, 60, 90),
        _cell(2, 2, 1, 1, "A-1", 20, 200, 60, 90),
    ]
    out = drop_evidenceless_columns(cells)
    max_col = max(int(c["col_end"]) for c in out)
    assert max_col == 0  # crumb col dropped, only substance col remains
    texts = {str(c.get("text") or "") for c in out}
    assert "聚合物" in texts
    assert "3" not in texts or any("3" in str(c.get("text")) for c in out if "聚合物" in str(c.get("text")))


def test_merge_ghost_keeps_index_column():
    cells = []
    boxes = []
    for i, n in enumerate(range(16, 21)):
        y = 50.0 + i * 30
        cells.append(_cell(i, i, 0, 0, "", 0, 35, y, y + 28))
        cells.append(_cell(i, i, 1, 1, "", 35, 130, y, y + 28))
        cells.append(_cell(i, i, 2, 2, "", 130, 300, y, y + 28))
        boxes.append(
            {
                "text": str(n),
                "polygon": [[5, y + 2], [30, y + 2], [30, y + 26], [5, y + 26]],
            }
        )
        boxes.append(
            {
                "text": f"OXL-{n}",
                "polygon": [[50, y + 2], [120, y + 2], [120, y + 26], [50, y + 26]],
            }
        )
        boxes.append(
            {
                "text": "通式",
                "polygon": [[150, y + 2], [280, y + 2], [280, y + 26], [150, y + 26]],
            }
        )
    out = merge_ghost_columns(cells, boxes)
    max_col = max(int(c["col_end"]) for c in out)
    assert max_col == 2, f"expected 3 cols kept, got max_col={max_col}"


if __name__ == "__main__":
    test_is_index_column()
    test_drop_keeps_index_after_caption_lift()
    test_drop_still_removes_sparse_digit_crumbs()
    test_merge_ghost_keeps_index_column()
    print("ok")
