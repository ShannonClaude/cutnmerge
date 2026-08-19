"""调试 P26X194 cells 拓扑。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_env

load_env()

from src.core.pipeline import extract_table_output  # noqa: E402
from src.output.html_formatter import _top_header_band_end, _column_has_body_content


def main() -> None:
    inp = ROOT / "data" / "input"
    img = next(p for p in inp.iterdir() if "P26X194" in p.name)
    _, tables = extract_table_output(str(img), save_vis=False)
    cells = tables[0] if tables else []
    he = _top_header_band_end(cells)
    print("header_end", he, "col0 body", _column_has_body_content(cells, 0, he))
    for c in sorted(cells, key=lambda x: (x["row_start"], x["col_start"])):
        if int(c["row_start"]) <= 1:
            print(
                c["row_start"],
                c["col_end"],
                c["col_start"],
                repr(str(c.get("text") or "")[:30]),
                "cs-ce",
                c["col_start"],
                c["col_end"],
            )


if __name__ == "__main__":
    main()
