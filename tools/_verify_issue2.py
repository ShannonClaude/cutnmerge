"""临时验证 P26X194 表头错列修复。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_env

load_env()

from src.core.pipeline import extract_table_output  # noqa: E402


def main() -> int:
    inp = ROOT / "data" / "input"
    img = next(p for p in inp.iterdir() if "P26X194" in p.name)
    r = extract_table_output(str(img), save_vis=False)
    html = r["html"]
    out = ROOT / "data" / "output" / "html" / f"{img.stem}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    assert not re.search(r'colspan="2"[^>]*>低温热压接性', html), html
    assert "低温热压接性" in html
    print("OK issue2 verify", out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
