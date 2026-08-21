"""验证双线框幽灵行列合并，并回归跑 P26X194。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_env

load_env()

from src.structure.tsr_refine import (
    _derive_seps,
    _robust_main_size,
    cell_grid_stats,
    merge_ghost_columns,
    merge_ghost_rows,
)


def _make_cell(x1, y1, x2, y2, rs, re, cs, ce):
    import numpy as np

    return {
        "polygon": np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        ),
        "row_start": rs,
        "row_end": re,
        "col_start": cs,
        "col_end": ce,
        "row_span": re - rs + 1,
        "col_span": ce - cs + 1,
        "texts": [],
        "text": "",
    }


def test_robust_main_size_ignores_seams():
    widths = [10.0, 400.0, 10.0, 390.0, 10.0, 380.0, 10.0, 370.0]
    main = _robust_main_size(widths)
    assert main > 300, main


def test_merge_double_line_seams():
    # 5 主列被双线缝拆成 5 主 + 4 缝；3 主行 + 缝
    # 列: 0缝,1主,2缝,3主,4缝,5主,6缝,7主,8缝,9主
    xs = [0, 10, 210, 220, 420, 430, 630, 640, 840, 850, 1050]
    ys = [0, 10, 90, 100, 180, 190, 270]
    cells = []
    # 仅放主格（奇数逻辑列 / 偶数逻辑行间隔）
    logical_cols = [1, 3, 5, 7, 9]
    logical_rows = [1, 3, 5]
    for ri, r in enumerate(logical_rows):
        for ci, c in enumerate(logical_cols):
            cells.append(
                _make_cell(xs[c], ys[r], xs[c + 1], ys[r + 1], r, r, c, c)
            )
    # 再塞一些纯缝格（模拟 TSR 双线产物）
    for r in (0, 2, 4):
        for c in (0, 2, 4, 6, 8):
            cells.append(
                _make_cell(xs[c], ys[r], xs[c + 1], ys[r + 1], r, r, c, c)
            )

    boxes = []
    # 文本中心故意落到缝列上（复现误命中）
    for ri, r in enumerate(logical_rows):
        for ci, c in enumerate(logical_cols):
            # 放到缝列中心 x
            seam_c = c - 1
            mx = 0.5 * (xs[seam_c] + xs[seam_c + 1])
            my = 0.5 * (ys[r] + ys[r + 1])
            import numpy as np

            boxes.append(
                {
                    "text": f"T{ri}{ci}",
                    "score": 0.99,
                    "polygon": np.array(
                        [
                            [mx - 2, my - 2],
                            [mx + 2, my - 2],
                            [mx + 2, my + 2],
                            [mx - 2, my + 2],
                        ],
                        dtype=np.float32,
                    ),
                }
            )

    out = merge_ghost_columns([dict(c) for c in cells], boxes)
    out = merge_ghost_rows(out, boxes)
    nc, nr, n = cell_grid_stats(out)
    assert nc == 5, (nc, nr, n)
    assert nr == 3, (nc, nr, n)


def test_p26_html_shape():
    from src.core.pipeline import extract_table_output

    img = next((ROOT / "data" / "input").glob("*P26X194*"))
    result = extract_table_output(
        str(img),
        structure="tsr",
        use_cache=True,
        refresh_cache=False,
        save_vis=False,
        debug=False,
    )
    html = result.get("html") or ""
    out = ROOT / "data" / "debug" / "p26_after_fix.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    assert html.count("<table") == 1, html.count("<table")
    assert "实施例 88" not in html
    assert "<td>88</td>" in html and "<td>7</td>" in html
    assert "<td>G</td>" in html and "<td>F</td>" in html
    for needle in ("30/30", "75/75", "实施例 1", "比较例 2", "低温热压接性"):
        assert needle in html, needle
    # 比较例2：G / F / 空 / 8
    assert "G F" not in html
    print("P26 OK")
    print(html)


if __name__ == "__main__":
    test_robust_main_size_ignores_seams()
    print("OK robust_main_size")
    test_merge_double_line_seams()
    print("OK merge_double_line_seams")
    test_p26_html_shape()
    print("ALL OK")
