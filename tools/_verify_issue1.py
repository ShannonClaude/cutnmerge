"""临时验证 P25X192/193 空格修复。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_env

load_env()

from src.core.pipeline import extract_table_output  # noqa: E402


def check(html: str, name: str) -> None:
    bad = [
        m.group(0)
        for m in re.finditer(r'colspan="\d+"[^>]*>(jER-604|850S|EP4003S|合成例2)', html)
    ]
    print(name, "bad_colspan=", bad or "none")
    assert not bad, bad


def main() -> int:
    inp = ROOT / "data" / "input"
    for key in ("P25X192", "P25X193"):
        img = next(p for p in inp.iterdir() if key in p.name)
        r = extract_table_output(str(img), save_vis=False)
        html = r["html"]
        out = ROOT / "data" / "output" / "html" / f"{img.stem}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        check(html, key)
        print("wrote", out.name)
    print("OK issue1 verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
