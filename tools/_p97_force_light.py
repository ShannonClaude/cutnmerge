# -*- coding: utf-8 -*-
"""强制 P97 走 light 路径，对比 escalate 前后表头。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_env

load_env()

import src.core.pipeline as pipe

captured: dict = {}
_real_ratio = pipe.logic_conflict_ratio


def _topo(cells):
    by = {}
    for c in cells:
        by.setdefault(int(c["row_start"]), []).append(
            (int(c["col_start"]), int(c["col_end"]), int(c["row_end"]))
        )
    lines = []
    for r, items in sorted(by.items()):
        parts = []
        for a, b, re_ in sorted(items):
            parts.append(f"c{a}-{b}" + ("" if re_ == r else f"rs{re_}"))
        lines.append(f"r{r}: " + " ".join(parts))
    return lines


def zero_ratio(cells):
    captured["real"] = _real_ratio(cells)
    captured["light_topo"] = _topo(cells)
    return 0.0


pipe.logic_conflict_ratio = zero_ratio

img = next(Path("data/input").glob("*P97X876*"))
res = pipe.extract_table_output(
    str(img), structure="tsr", use_cache=True, refresh_cache=False, save_vis=False
)
html = res["html"] if isinstance(res, dict) else ""
if not html and isinstance(res, (list, tuple)):
    outs = res[0]
    html = outs[0].get("html", "") if outs and isinstance(outs[0], dict) else str(outs[0] if outs else "")

Path("tools/_p97_light.html").write_text(html or "", encoding="utf-8")
print("real_conflict", captured.get("real"))
print("LIGHT TOPO:")
print("\n".join(captured.get("light_topo") or []))
print("--- HTML headers ---")
for i, m in enumerate(re.finditer(r"<table[\s\S]*?</table>", html or "")):
    t = m.group(0)
    # first two rows only
    rows = re.findall(r"<tr>([\s\S]*?)</tr>", t)
    print(f"=== table {i} ===")
    for ri, row in enumerate(rows[:3]):
        cells = re.findall(r"<td([^>]*)>([\s\S]*?)</td>", row)
        bits = []
        for attrs, body in cells:
            body = re.sub(r"<br\s*/?>", "/", body)
            body = re.sub(r"\s+", "", body)[:40]
            bits.append(f"[{attrs.strip()}]{body}")
        print(f"r{ri}: " + " | ".join(bits))
