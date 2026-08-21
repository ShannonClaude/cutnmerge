# -*- coding: utf-8 -*-
"""Dump cell topology for remaining issue samples."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.config import load_env

load_env()
import src.core.pipeline as pipe

orig_render = pipe._render_outputs
dumps: list = []


def wrap_render(cells, free_texts, **kw):
    rows = []
    for c in cells:
        t = re.sub(r"\s+", "", str(c.get("text") or ""))[:50]
        rows.append(
            {
                "rs": int(c["row_start"]),
                "re": int(c["row_end"]),
                "cs": int(c["col_start"]),
                "ce": int(c["col_end"]),
                "text": t,
            }
        )
    dumps.append(rows)
    return orig_render(cells, free_texts, **kw)


pipe._render_outputs = wrap_render
out_lines = []
for key in ["P96X874", "P93X1068", "P47X435"]:
    dumps.clear()
    imgs = list(Path("data/input").glob(f"*{key}*"))
    if not imgs:
        out_lines.append(f"## MISSING {key}")
        continue
    pipe.extract_table_output(
        str(imgs[0]),
        structure="tsr",
        use_cache=True,
        refresh_cache=False,
        save_vis=False,
    )
    cells = dumps[-1] if dumps else []
    out_lines.append(f"## {key} n={len(cells)}")
    for c in sorted(cells, key=lambda x: (x["rs"], x["cs"])):
        if c["rs"] <= 3:
            out_lines.append(
                f"  r{c['rs']}-{c['re']} c{c['cs']}-{c['ce']}: {c['text']}"
            )
    if key == "P93X1068":
        out_lines.append("  -- acid/ratio --")
        for c in cells:
            if any(k in c["text"] for k in ("酸当量", "双键", "来源于", "来自")):
                out_lines.append(
                    f"  r{c['rs']}-{c['re']} c{c['cs']}-{c['ce']}: {c['text']}"
                )
    out_lines.append("")

Path("tools/_dump_cells.txt").write_text("\n".join(out_lines), encoding="utf-8")
print("\n".join(out_lines))
