# -*- coding: utf-8 -*-
"""九张表联合回归：结构摘要（验收门禁，规则本身不绑页码）。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.config import load_env

load_env()
from src.core.pipeline import extract_table_output

KEYS = [
    ("CN110392864B", "P96X874"),
    ("CN110392864B", "P97X876"),
    ("CN110392864B", "P98X878"),
    ("CN111263917A", "P92X1066"),
    ("CN111263917A", "P93X1068"),
    ("CN111263917A", "P94X1070"),
    ("CN111656277A", "P46X433"),
    ("CN111656277A", "P47X435"),
    ("CN111656277A", "P56X443"),
]


def summarize(html: str) -> list[str]:
    lines = []
    for i, m in enumerate(re.finditer(r"<table[\s\S]*?</table>", html or "")):
        rows = re.findall(r"<tr>([\s\S]*?)</tr>", m.group(0))
        lines.append(f"  table{i}: {len(rows)} rows")
        for ri, row in enumerate(rows[:3]):
            cells = re.findall(r"<td([^>]*)>([\s\S]*?)</td>", row)
            bits = []
            for attrs, body in cells:
                body = re.sub(r"<br\s*/?>", "/", body)
                body = re.sub(r"\s+", "", body)[:28]
                sp = ""
                cm = re.search(r'colspan="(\d+)"', attrs)
                rm = re.search(r'rowspan="(\d+)"', attrs)
                if cm:
                    sp += f"c{cm.group(1)}"
                if rm:
                    sp += f"r{rm.group(1)}"
                bits.append(f"[{sp}]{body}" if sp else body)
            lines.append(f"    r{ri}: " + " | ".join(bits))
    return lines


Path("data/output/html").mkdir(parents=True, exist_ok=True)
out_lines = []
for patent, key in KEYS:
    imgs = list(Path("data/input").glob(f"*{patent}*{key}*"))
    if not imgs:
        out_lines.append(f"## MISSING {patent} {key}")
        continue
    img = imgs[0]
    res = extract_table_output(
        str(img), structure="tsr", use_cache=True, refresh_cache=False, save_vis=False
    )
    html = res["html"] if isinstance(res, dict) else ""
    dest = Path("data/output/html") / (img.stem + ".html")
    dest.write_text(html, encoding="utf-8")
    out_lines.append(f"## {patent} {key}")
    out_lines.extend(summarize(html))
    out_lines.append("")

Path("tools/_nine_regress.txt").write_text("\n".join(out_lines), encoding="utf-8")
print("\n".join(out_lines))
