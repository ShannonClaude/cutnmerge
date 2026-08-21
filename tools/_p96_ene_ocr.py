# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, ".")
from src.core.config import load_env
load_env()
from src.core.pipeline import extract_table_output
from src.structure.tsr_refine import _ENE_HEADER_RE, _tb_bbox, _ene_outside_col_starts

# Hook OCR boxes
import src.core.pipeline as pipe

saved = {}
_real = pipe.assign_texts_to_cells

def capture(cells, boxes, **kw):
    saved["boxes"] = boxes
    saved["cells"] = cells
    return _real(cells, boxes, **kw)

pipe.assign_texts_to_cells = capture
# also need matching module
import src.matching.matching as m
m.assign_texts_to_cells = capture

img = next(Path("data/input").glob("*P96X874*"))
pipe.extract_table_output(str(img), structure="tsr", use_cache=True, refresh_cache=False, save_vis=False)
boxes = saved["boxes"]
enes = []
for tb in boxes:
    t = str(tb.get("text") or "")
    if _ENE_HEADER_RE.search(t) or "烯" in t:
        x1, y1, x2, y2 = _tb_bbox(tb)
        enes.append((t[:30], y1, y2, 0.5 * (y1 + y2), x1, x2))
print("ene boxes:", len(enes))
for e in enes:
    print(e)
print("outside cols parent_y 40-66:", _ene_outside_col_starts(
    boxes, parent_y1=40, parent_y2=66,
    col_seps=list(range(0, 800, 50)), band_lo=2, band_hi=12
))
# also try wider parent band (row0+row1)
print("outside cols parent_y 0-66:", _ene_outside_col_starts(
    boxes, parent_y1=0, parent_y2=66,
    col_seps=list(range(0, 800, 50)), band_lo=2, band_hi=12
))
