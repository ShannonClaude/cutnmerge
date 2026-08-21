# -*- coding: utf-8 -*-
import json, re
from pathlib import Path

p = Path(r"D:/Download/cutnmerge/data/cache/71e21788d61431ea68fb6b0d74f88b487ddc5e3e.json")
data = json.loads(p.read_text(encoding="utf-8"))
print("key", data.get("key"))
boxes = data["text_boxes"]
print("n", len(boxes))
out = []
for i, b in enumerate(boxes):
    t = b.get("text", "")
    poly = b.get("polygon") or []
    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    out.append(
        f"{i:3d} x=[{min(xs):.0f},{max(xs):.0f}] y=[{min(ys):.0f},{max(ys):.0f}] "
        f"w={max(xs)-min(xs):.0f} text={t!r}"
    )
Path(r"D:/Download/cutnmerge/tmp_p135_cache_boxes.txt").write_text(
    "\n".join(out), encoding="utf-8"
)
# highlight interesting
for line in out:
    if any(
        k in line
        for k in [
            "比较",
            "组成",
            "颜料",
            "组合物",
            "Bk",
            "质量",
            "|",
        ]
    ) or re.search(r"text='(?:8[6-9]|9[0-3])'", line):
        print(line)
