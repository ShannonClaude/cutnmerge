# -*- coding: utf-8 -*-
from pathlib import Path
import sys
sys.path.insert(0, ".")
from src.core.config import load_env
load_env()
import src.structure.tsr_refine as refine
import src.core.pipeline as pipe

stages = {}

def snap(tag, cells):
    lines = []
    for c in cells:
        t = str(c.get("text") or "").replace("\n", "/")[:40]
        rs, re = int(c["row_start"]), int(c["row_end"])
        cs, ce = int(c["col_start"]), int(c["col_end"])
        x1, y1, x2, y2 = refine._cell_bbox(c)
        if any(k in t for k in ("单体", "烯键", "封端", "四羧", "二胺", "含有", "聚合物")) or (
            not t and ce - cs >= 2 and rs <= 2
        ):
            lines.append(
                f"r{rs}-{re} c{cs}-{ce} y={y1:.0f}-{y2:.0f} h={y2-y1:.0f} {t!r}"
            )
    stages[tag] = lines

def wrap(name, fn):
    def inner(cells, *a, **kw):
        snap("before_" + name, cells)
        r = fn(cells, *a, **kw)
        out = r[0] if isinstance(r, tuple) else r
        snap("after_" + name, out)
        return r
    return inner

refine.repair_monomer_parent_spans = wrap("repair", refine.repair_monomer_parent_spans)
pipe.repair_monomer_parent_spans = refine.repair_monomer_parent_spans

img = next(Path("data/input").glob("*P96X874*"))
pipe.extract_table_output(str(img), structure="tsr", use_cache=True, refresh_cache=False, save_vis=False)
Path("tools/_p96_ene.txt").write_text(
    "\n".join(f"## {k}\n" + "\n".join(v) for k, v in stages.items()),
    encoding="utf-8",
)
print("done")
