"""Re-run P46/P47 via extract_table_output and print HTML summary."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_env

load_env()

from src.core.pipeline import extract_table_output
from src.structure.tsr import load_tsr_models
from src.utils.segments import find_row_segments


def main() -> int:
    load_tsr_models()
    out_html = ROOT / "data" / "output" / "html"
    out_html.mkdir(parents=True, exist_ok=True)
    for pat in ("*P46*", "*P47*"):
        for p in sorted((ROOT / "data" / "input").glob(pat)):
            print("=" * 72)
            print(p.name)
            result = extract_table_output(
                str(p),
                structure="tsr",
                use_cache=True,
                compress_empty_cols=True,
                fallback_lines=False,
                save_vis=True,
                vis_dir=ROOT / "data" / "output" / "images",
                debug_stem=p.stem,
            )
            html = result.get("html") or ""
            dest = out_html / f"{p.stem}.html"
            dest.write_text(html, encoding="utf-8")
            n_tables = html.count("<table")
            print(f"tables={n_tables} orient={result.get('orientation')}")
            print(html)
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
