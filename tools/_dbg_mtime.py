# -*- coding: utf-8 -*-
"""调试用：列出输入图与 HTML 输出的修改时间。"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def show(d: Path, pat: str) -> None:
    print(f"### {d}")
    for p in sorted(d.glob(pat)):
        ts = dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(" ", "seconds")
        print(f"  {ts}  {p.stat().st_size:>8}  {p.name}")


show(ROOT / "data" / "input", "*")
show(ROOT / "data" / "output" / "html", "*P9[4678]*")
for f in ["src/structure/tsr_refine.py", "src/core/pipeline.py", "src/output/html_formatter.py"]:
    p = ROOT / f
    print(dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(" ", "seconds"), f)
