# -*- coding: utf-8 -*-
import json, glob, re, sys
from pathlib import Path

def dump_ocr(tag):
    paths = glob.glob(rf"D:/Download/cutnmerge/data/ocr/*{tag}*_page_001.json")
    print("===", tag, paths)
    if not paths:
        return
    boxes = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    print("nboxes", len(boxes))
    cjk = [b for b in boxes if re.search(r"[\u4e00-\u9fff]", b.get("text", ""))]
    print("cjk", len(cjk))
    for b in boxes:
        t = b.get("text", "")
        if (
            "比较例" in t
            or "实施例" in t
            or "组成" in t
            or "颜料" in t
            or "组合物" in t
            or re.fullmatch(r"\d{2,3}", t.strip())
            or t.startswith("Bk")
        ):
            poly = b.get("polygon") or []
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            print(
                f"  text={t!r} x=[{min(xs):.0f},{max(xs):.0f}] y=[{min(ys):.0f},{max(ys):.0f}]"
            )

    # also search raw
    raws = glob.glob(rf"D:/Download/cutnmerge/data/ocr/*{tag}*_page_001_raw.json")
    if raws:
        raw = json.loads(Path(raws[0]).read_text(encoding="utf-8"))
        blob = json.dumps(raw, ensure_ascii=False)
        print("raw has 比较例", "比较例" in blob, "组成", "组成" in blob)
        for m in re.finditer(r".{0,5}比较例.{0,25}", blob):
            print("  RAW:", m.group(0))
            if m.start() > 200000:
                break


dump_ocr("P135X957")
dump_ocr("P136X959")

# cache hits for 比较例 186
for p in glob.glob(r"D:/Download/cutnmerge/data/cache/*.json"):
    try:
        blob = Path(p).read_text(encoding="utf-8")
    except Exception:
        continue
    if "比较例 186" in blob or "比较例186" in blob:
        print("CACHE HIT", p)
        data = json.loads(blob)
        if isinstance(data, list):
            for b in data:
                t = str(b.get("text", ""))
                if "比较例" in t or t.startswith("Bk") or re.fullmatch(r"\d{2,3}", t):
                    print(" ", repr(t))
