# -*- coding: utf-8 -*-
import json, re
from pathlib import Path

for p in [
    r"D:/Download/cutnmerge/data/cache/71e21788d61431ea68fb6b0d74f88b487ddc5e3e.json",
    r"D:/Download/cutnmerge/data/cache/e89ccbc6af6c66b61abd77550cc0982f8c05a700.json",
]:
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    print("====", p, type(data))
    if isinstance(data, dict):
        print("keys", list(data.keys())[:30])
        blob = json.dumps(data, ensure_ascii=False)
        print("比较例 count", blob.count("比较例"))
        for m in re.finditer(r"比较例[^\"\\]{0,30}", blob):
            print(" ", m.group(0))
        for m in re.finditer(r"组成[^\"\\]{0,20}", blob):
            print(" COMP", m.group(0))
            break
    elif isinstance(data, list):
        print("len", len(data))
        for b in data:
            if not isinstance(b, dict):
                continue
            t = str(b.get("text", ""))
            if "比较例" in t or "组成" in t or "颜料" in t or t.startswith("Bk") or re.fullmatch(r"8[6-9]|9[0-3]", t.strip() or "x"):
                print(repr(t))
