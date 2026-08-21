# -*- coding: utf-8 -*-
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.config import load_env
load_env()
import src.core.pipeline as pipe
from src.structure.tsr_refine import sanitize_side_header_texts

orig = pipe._render_outputs

def wrap(cells, free_texts, **kw):
    for c in cells:
        t = str(c.get("text") or "")
        if "脂环" in t or "烯键" in t:
            print("BEFORE RENDER", int(c["col_start"]), repr(t[:80]))
    cells2 = sanitize_side_header_texts(cells)
    for c in cells2:
        t = str(c.get("text") or "")
        if "脂环" in t or "烯键" in t:
            print("AFTER SANITIZE", int(c["col_start"]), repr(t[:80]))
    return orig(cells2, free_texts, **kw)

pipe._render_outputs = wrap
img = list(Path("data/input").glob("*P47X435*"))[0]
pipe.extract_table_output(str(img), structure="tsr", use_cache=True, refresh_cache=False, save_vis=False)
