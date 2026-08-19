"""Run pipeline on P96/P97/P98/P109/P135 and print short HTML checks."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_env

load_env()

from src.core.pipeline import extract_table_output  # noqa: E402

KEYS = ["P96X874", "P97X876", "P98X878", "P100X888", "P109X1086", "P123X933", "P135X957"]


def main() -> None:
    inp = ROOT / "data" / "input"
    out = ROOT / "data" / "output"
    images = [
        p
        for p in sorted(inp.iterdir())
        if p.is_file()
        and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and any(k in p.name for k in KEYS)
    ]
    print(f"images={len(images)}")
    for img in images:
        key = next(k for k in KEYS if k in img.name)
        print(f"\n=== RUN {key} ===")
        result = extract_table_output(
            str(img),
            structure="tsr",
            tsr_aggressive=False,
            use_cache=True,
            save_vis=True,
            vis_dir=out,
            debug_stem=img.stem,
        )
        html = result.get("html") or ""
        html_path = out / f"{img.stem}.html"
        html_path.write_text(html, encoding="utf-8")
        html_dir = out / "html"
        if html_dir.is_dir():
            (html_dir / f"{img.stem}.html").write_text(html, encoding="utf-8")
        print(f"wrote {html_path.name}")

        if key == "P96X874":
            hit = "来自具有氟" in html and "全部结构" in html
            rowspan = bool(
                re.search(
                    r'rowspan="2"[^>]*>来自具有氟|来自具有氟[^<]*rowspan',
                    html,
                )
            ) or ('rowspan="2"' in html and "全部结构" in html)
            # check the specific cell has rowspan nearby
            idx = html.find("全部结构")
            snippet = html[max(0, idx - 200) : idx + 80] if idx >= 0 else ""
            print("P96 has text", hit, "snippet_has_rowspan", 'rowspan="2"' in snippet)
            print(snippet.replace("\n", " ")[:180])
        elif key == "P97X876":
            has_biamino = "双氨基酚" in html
            bad_split = bool(re.search(r"<td>二羟基</td>\s*<td>二胺", html))
            colspan2 = bool(re.search(r'colspan="2"[^>]*>[^<]*双氨基酚', html)) or (
                "双氨基酚" in html and 'colspan="2"' in html
            )
            print(
                "P97 biamino",
                has_biamino,
                "bad_split",
                bad_split,
                "colspan2",
                colspan2,
            )
        elif key == "P98X878":
            n = html.count("单体[摩尔比]")
            print("P98 monomer_headers", n)
        elif key == "P100X888":
            ghost = bool(
                re.search(
                    r'<td rowspan="2"></td>\s*<td></td>\s*<td[^>]*>分散液',
                    html,
                )
            )
            print(
                "P100 ghost_line",
                ghost,
                "has_fensan",
                "分散液" in html,
                "n_zhizao",
                html.count("制备例"),
            )
        elif key == "P123X933":
            ghost_split = bool(
                re.search(
                    r"组合物</td>\s*<td[^>]*>\s*</td>",
                    html,
                )
            ) or bool(
                re.search(
                    r'<td colspan="3"></td>\s*<td colspan="5">组成',
                    html,
                )
            )
            merged = bool(
                re.search(r'colspan="9"[^>]*>组成\[质量份\]', html)
            ) or bool(
                re.search(r"组合物</td>\s*<td colspan=\"\d+\">组成\[质量份\]", html)
            )
            print(
                "P123 split_header",
                ghost_split,
                "merged_group",
                merged,
                "n_shishi",
                html.count("实施例"),
            )
        elif key == "P109X1086":
            print(
                "P109 B_ratio rowspan",
                bool(re.search(r'rowspan="2"[^>]*>\(B\)|\(B\)[^<]*', html)),
                "has_1_2",
                ">1<" in html and ">2<" in html,
            )
        elif key == "P135X957":
            bad = "比较例 186Bk-1" in html or "比较例186Bk-1" in html
            good = "比较例 1" in html and ">86<" in html and "Bk-1" in html
            print("P135 sticky_ok", good and not bad, "bad_glued", bad)


if __name__ == "__main__":
    main()
