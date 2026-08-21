# -*- coding: utf-8 -*-
"""调试用：对指定样张跑完整 pipeline（用 OCR 缓存），把逻辑网格打印成文本。

    venv/Scripts/python tools/_dbg_run.py OUT.txt P96X874 P97X876 P98X878
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_env  # noqa: E402

load_env()

from src.core.pipeline import extract_table_output  # noqa: E402

INPUT_DIR = ROOT / "data" / "input"


def grid_of(html: str) -> list[list[str]]:
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.S)
    occ: dict[tuple[int, int], str] = {}
    for r, row in enumerate(rows):
        cells = re.findall(r"<t[dh]([^>]*)>(.*?)</t[dh]>", row, re.S)
        c = 0
        for attrs, inner in cells:
            while (r, c) in occ:
                c += 1
            m = re.search(r'rowspan="(\d+)"', attrs)
            rs = int(m.group(1)) if m else 1
            m = re.search(r'colspan="(\d+)"', attrs)
            cs = int(m.group(1)) if m else 1
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


def dump(html: str, out) -> None:
    tables = re.findall(r"<table.*?</table>", html, re.S)
    print(f"tables={len(tables)}", file=out)
    for ti, t in enumerate(tables):
        print(f"--- table {ti} ---", file=out)
        for r, row in enumerate(grid_of(t)):
            print(f"r{r:02d} | " + " | ".join(x.replace("\n", "/") for x in row), file=out)


def main() -> None:
    out_path = ROOT / sys.argv[1]
    keys = sys.argv[2:]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    imgs: list[Path] = []
    for k in keys:
        hits = [p for p in INPUT_DIR.iterdir() if k in p.name]
        if not hits:
            raise SystemExit(f"no input matches {k}")
        imgs.extend(sorted(hits))

    with out_path.open("w", encoding="utf-8") as out:
        for p in imgs:
            print("=" * 100, file=out)
            print(p.name, file=out)
            res = extract_table_output(
                str(p),
                ioa_threshold=0.5,
                structure="tsr",
                use_cache=True,
                refresh_cache=False,
                save_vis=False,
            )
            dump(res.get("html") or "", out)
            out.flush()


if __name__ == "__main__":
    main()
