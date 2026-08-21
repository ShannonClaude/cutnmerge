# -*- coding: utf-8 -*-
"""Dump header-related OCR boxes and simulate join near 组成/颜料."""
import json, re
from pathlib import Path

p = Path(r"D:/Download/cutnmerge/data/cache/71e21788d61431ea68fb6b0d74f88b487ddc5e3e.json")
boxes = json.loads(p.read_text(encoding="utf-8"))["text_boxes"]
# all boxes with y < 220 (header band)
for i, b in enumerate(boxes):
    poly = b.get("polygon") or []
    ys = [pt[1] for pt in poly]
    xs = [pt[0] for pt in poly]
    if min(ys) > 220:
        continue
    t = b.get("text", "")
    print(f"{i:3d} x=[{min(xs):.0f},{max(xs):.0f}] y=[{min(ys):.0f},{max(ys):.0f}] text={t!r}")
