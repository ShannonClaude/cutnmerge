# -*- coding: utf-8 -*-
"""Debug P96 child align anchors."""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(message)s")

from src.core.config import load_env

load_env()

import src.structure.tsr_refine as refine

_orig = refine._align_monomer_children_to_body


def _wrap(children, body_cells, col_seps, *, band_lo, band_hi, boxes=None):
    boxes = list(boxes) if boxes else []
    print(f"ALIGN band={band_lo}-{band_hi} kids={len(children)} bodies={len(body_cells)} boxes={len(boxes)}")
    for k in children:
        tbs = refine._texts_in_cell(k, boxes)
        xs = []
        for tb in tbs:
            x1, y1, x2, y2 = refine._tb_bbox(tb)
            xs.append((0.5 * (x1 + x2), str(tb.get("text") or "")[:20]))
        t = re.sub(r"\s+", "", str(k.get("text") or ""))[:20]
        print(
            f"  kid c{k['col_start']}-{k['col_end']} text={t!r} "
            f"cat={refine._child_header_category(t)} ocr={xs}"
        )
    for b in body_cells:
        t = re.sub(r"\s+", "", str(b.get("text") or ""))[:15]
        print(f"  body c{b['col_start']}-{b['col_end']} {t!r}")
    n = _orig(children, body_cells, col_seps, band_lo=band_lo, band_hi=band_hi, boxes=boxes)
    print(f"  → changed={n}")
    for k in children:
        t = re.sub(r"\s+", "", str(k.get("text") or ""))[:20]
        print(f"  after c{k['col_start']}-{k['col_end']} {t!r}")
    return n


refine._align_monomer_children_to_body = _wrap

import src.core.pipeline as pipe

img = list(Path("data/input").glob("*P96X874*"))[0]
pipe.extract_table_output(
    str(img), structure="tsr", use_cache=True, refresh_cache=False, save_vis=False
)
