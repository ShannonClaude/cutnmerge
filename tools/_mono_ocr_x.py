# -*- coding: utf-8 -*-
from pathlib import Path
import sys
sys.path.insert(0, ".")
from src.core.config import load_env
load_env()
import src.core.pipeline as pipe
import src.matching.matching as m
from src.structure.tsr_refine import _tb_bbox, _MONOMER_PARENT_RE, _ENE_HEADER_RE

saved = {}
_real = m.assign_texts_to_cells

def capture(cells, boxes, **kw):
    saved["boxes"] = boxes
    return _real(cells, boxes, **kw)

m.assign_texts_to_cells = capture
pipe.assign_texts_to_cells = capture

for key in ("P96X874", "P98X878"):
    img = next(Path("data/input").glob(f"*{key}*"))
    pipe.extract_table_output(str(img), structure="tsr", use_cache=True, refresh_cache=False, save_vis=False)
    boxes = saved["boxes"]
    print("====", key, "====")
    for tb in boxes:
        t = str(tb.get("text") or "")
        if _MONOMER_PARENT_RE.search(t) or _ENE_HEADER_RE.search(t):
            x1, y1, x2, y2 = _tb_bbox(tb)
            print(f"  {t[:40]!r} x={x1:.0f}-{x2:.0f} y={y1:.0f}-{y2:.0f}")
