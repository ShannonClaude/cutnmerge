"""验证 P40X268 左列 (AA)… 标签与整表 rowspan 对齐。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_env

load_env()

from src.core.pipeline import extract_table_output  # noqa: E402

KEY = "P40X268"
REQUIRED = ["聚酰胺酸酯", "6FDA", "ODPA"]


def main() -> int:
    inp = ROOT / "data" / "input"
    img = next(p for p in inp.iterdir() if KEY in p.name)
    r = extract_table_output(str(img), save_vis=False)
    html = r["html"]
    out = ROOT / "data" / "output" / "html" / f"{img.stem}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print("===", KEY, "===")

    bad_rowspan = re.findall(r'rowspan="(\d{2,})"', html)
    print("  huge_rowspan:", bad_rowspan or "none")
    if bad_rowspan:
        return 1

    for code in ("AA", "AS"):
        hit = bool(re.search(rf"[\(（]{code}[\)）]", html))
        print(f"  label {code}: {'OK' if hit else 'FAIL'}")
        if not hit:
            return 1

    for text in REQUIRED:
        hit = text in html
        print(f"  {text!r}: {'OK' if hit else 'FAIL'}")
        if not hit:
            return 1

    empty_tr = len(re.findall(r"<tr>\s*</tr>", html))
    print(f"  empty_tr_count: {empty_tr}")
    if empty_tr > 5:
        return 1

    print("OK p40 verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
