"""验证 P24X176 / P25X177 嵌套行头修复。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_env

load_env()

from src.core.pipeline import extract_table_output  # noqa: E402

CHECKS = {
    "P24X176": [
        ("酸酐", True),
        ("二胺", True),
        ("封端剂", True),
        ("jER828", True),
        ("OXT-191", True),
        ("实施例 10", True),
        ("实施例8实施例9", False),
    ],
    "P25X177": [
        ("酸酐", True),
        ("二胺", True),
        ("封端剂", True),
        ("jER828", True),
        ("实施例 11", True),
        ("比较例 1", True),
        ("参考例 1", True),
        ("实施例实施例", False),
    ],
}


def main() -> int:
    inp = ROOT / "data" / "input"
    for key, checks in CHECKS.items():
        img = next(p for p in inp.iterdir() if key in p.name)
        r = extract_table_output(str(img), save_vis=False)
        html = r["html"]
        out = ROOT / "data" / "output" / "html" / f"{img.stem}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print("===", key, "===")
        for text, should in checks:
            hit = text in html
            ok = hit == should
            print(f"  {text!r}: {'OK' if ok else 'FAIL'} (found={hit})")
            if not ok:
                return 1
    print("OK issue3 verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
