# -*- coding: utf-8 -*-
"""调试用：把指定 stem 的 HTML 输出转成行列文本网格，便于肉眼比对。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HTML_DIR = ROOT / "data" / "output" / "html"


def find(stem_key: str) -> list[Path]:
    return sorted(p for p in HTML_DIR.glob("*.html") if stem_key in p.name)


def grid_of(html: str) -> list[list[str]]:
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.S)
    occ: dict[tuple[int, int], str] = {}
    for r, row in enumerate(rows):
        cells = re.findall(r"<t[dh]([^>]*)>(.*?)</t[dh]>", row, re.S)
        c = 0
        for attrs, inner in cells:
            while (r, c) in occ:
                c += 1
            rs = int((re.search(r'rowspan="(\d+)"', attrs) or [0, 1])[1]) if "rowspan" in attrs else 1
            cs = int((re.search(r'colspan="(\d+)"', attrs) or [0, 1])[1]) if "colspan" in attrs else 1
            txt = re.sub(r"<[^>]+>", "", inner).strip()
            for dr in range(rs):
                for dc in range(cs):
                    occ[(r + dr, c + dc)] = txt if (dr == 0 and dc == 0) else "\u2191"
            c += cs
    if not occ:
        return []
    maxr = max(k[0] for k in occ)
    maxc = max(k[1] for k in occ)
    return [[occ.get((r, c), "") for c in range(maxc + 1)] for r in range(maxr + 1)]


def main() -> None:
    for key in sys.argv[1:]:
        for p in find(key):
            html = p.read_text(encoding="utf-8")
            print("=" * 100)
            print(p.name)
            tables = re.findall(r"<table.*?</table>", html, re.S)
            print(f"tables={len(tables)}")
            for ti, t in enumerate(tables):
                print(f"--- table {ti} ---")
                for r, row in enumerate(grid_of(t)):
                    print(f"r{r:02d} | " + " | ".join(x.replace("\n", "/") for x in row))


if __name__ == "__main__":
    main()
