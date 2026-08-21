# -*- coding: utf-8 -*-
"""重跑 P93/P96 并写回 HTML，摘要三官能 colspan。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.config import load_env

load_env()
from src.core.pipeline import extract_table_output

out_dir = Path("data/output/html")
md_dir = Path("data/output/md")
lines = []
for key in ("P93X1068", "P96X874"):
    imgs = list(Path("data/input").glob(f"*{key}*"))
    if not imgs:
        lines.append(f"MISSING {key}")
        continue
    img = imgs[0]
    res = extract_table_output(
        str(img), structure="tsr", use_cache=True, refresh_cache=False, save_vis=False
    )
    html = res["html"]
    (out_dir / f"{img.stem}.html").write_text(html, encoding="utf-8")
    (md_dir / f"{img.stem}.md").write_text(res.get("md") or "", encoding="utf-8")
    lines.append(f"## {key}")
    for i, m in enumerate(re.finditer(r"<table[\s\S]*?</table>", html)):
        rows = re.findall(r"<tr>([\s\S]*?)</tr>", m.group(0))
        lines.append(f"  table{i}: {len(rows)} rows")
        for ri, row in enumerate(rows[:3]):
            cells = re.findall(r"<td([^>]*)>([\s\S]*?)</td>", row)
            bits = []
            for a, b in cells:
                b = re.sub(r"<br\s*/?>", "/", b)
                b = re.sub(r"<[^>]+>", "", b)
                b = re.sub(r"\s+", "", b)[:18]
                pref = ""
                cm = re.search(r'colspan="(\d+)"', a)
                rm = re.search(r'rowspan="(\d+)"', a)
                if rm:
                    pref += f"[r{rm.group(1)}]"
                if cm:
                    pref += f"[c{cm.group(1)}]"
                bits.append(f"{pref}{b}" if b or pref else "(empty)")
            lines.append(f"    r{ri}: " + " | ".join(bits))

Path("tools/_fix_p93_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
