# -*- coding: utf-8 -*-
"""调试用：对 data/input 全部样张跑 pipeline（用 OCR 缓存），输出逻辑网格快照。

用于「改动前 / 改动后」整库比对，找出回归。

    venv/Scripts/python tools/_dbg_all.py tools/_snap_head.txt
"""
from __future__ import annotations

import logging
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_env  # noqa: E402

load_env()

from src.core.pipeline import extract_table_output  # noqa: E402

INPUT_DIR = ROOT / "data" / "input"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


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


def main() -> None:
    out_path = ROOT / sys.argv[1]
    only = sys.argv[2:]
    logging.disable(logging.WARNING)
    imgs = sorted(
        p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if only:
        imgs = [p for p in imgs if any(k in p.name for k in only)]

    with out_path.open("w", encoding="utf-8") as out:
        for i, p in enumerate(imgs, 1):
            print(f"[{i}/{len(imgs)}] {p.name}", file=sys.stderr, flush=True)
            print("=" * 100, file=out)
            print(p.name, file=out)
            try:
                res = extract_table_output(
                    str(p),
                    ioa_threshold=0.5,
                    structure="tsr",
                    use_cache=True,
                    refresh_cache=False,
                    save_vis=False,
                )
                html = res.get("html") or ""
            except Exception:
                print("EXCEPTION", file=out)
                print(traceback.format_exc(limit=3), file=out)
                out.flush()
                continue
            tables = re.findall(r"<table.*?</table>", html, re.S)
            print(f"tables={len(tables)}", file=out)
            for ti, t in enumerate(tables):
                print(f"--- table {ti} ---", file=out)
                for r, row in enumerate(grid_of(t)):
                    print(
                        f"r{r:02d} | " + " | ".join(x.replace("\n", "/") for x in row),
                        file=out,
                    )
            out.flush()


if __name__ == "__main__":
    main()
