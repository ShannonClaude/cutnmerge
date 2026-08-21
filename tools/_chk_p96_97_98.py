# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.config import load_env

load_env()
from src.core.pipeline import extract_table_output

for key in ("P96X874", "P97X876", "P98X878"):
    img = next(Path("data/input").glob(f"*{key}*"))
    res = extract_table_output(
        str(img), structure="tsr", use_cache=True, refresh_cache=False, save_vis=False
    )
    html = res["html"] if isinstance(res, dict) else ""
    out = Path("data/output/html") / (img.stem + ".html")
    out.write_text(html, encoding="utf-8")
    print("====", key, "====")
    for i, m in enumerate(re.finditer(r"<table[\s\S]*?</table>", html)):
        rows = re.findall(r"<tr>([\s\S]*?)</tr>", m.group(0))
        print(f"-- table {i} --")
        for ri, row in enumerate(rows[:3]):
            cells = re.findall(r"<td([^>]*)>([\s\S]*?)</td>", row)
            bits = []
            for attrs, body in cells:
                body = re.sub(r"<br\s*/?>", "/", body)
                body = re.sub(r"\s+", "", body)[:36]
                bits.append(f"{{{attrs.strip()}}}{body}")
            print(f"r{ri}: " + " | ".join(bits))
