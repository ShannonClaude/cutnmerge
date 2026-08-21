# -*- coding: utf-8 -*-
"""Simulate sticky split on cache OCR texts for P135."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, r"D:/Download/cutnmerge")
from src.matching.matching import _parse_sticky_row_parts, _split_generic_row_label
from src.ocr.ocr_post import _strip_leading_pipe_digits

p = Path(r"D:/Download/cutnmerge/data/cache/71e21788d61431ea68fb6b0d74f88b487ddc5e3e.json")
data = json.loads(p.read_text(encoding="utf-8"))
boxes = data["text_boxes"]
for b in boxes:
    t = b.get("text", "")
    if "比较" not in t and "组成" not in t and "颜料" not in t and "组合物" not in t:
        continue
    stripped = _strip_leading_pipe_digits(t)
    print("=" * 60)
    print("RAW:", repr(t))
    print("STRIPPED:", repr(stripped))
    for nc in (2, 3):
        parts = _parse_sticky_row_parts(stripped, n_cols=nc, cut_fracs=None)
        print(f"  n_cols={nc}:", parts)
    # also with pipes replaced already in text
    t2 = re.sub(r"\|+", " ", t)
    t2 = re.sub(r"\s+", " ", t2).strip()
    print("PIPE->SPACE:", repr(t2))
    for nc in (2, 3):
        parts = _parse_sticky_row_parts(t2, n_cols=nc, cut_fracs=None)
        print(f"  n_cols={nc}:", parts)
