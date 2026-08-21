# -*- coding: utf-8 -*-
"""P98 全链路钩子：reconstruct / assign / normalize / lift。"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.config import load_env

load_env()

import src.core.pipeline as pipe
import src.structure.tsr_refine as refine
from src.core.pipeline import extract_table_output
from src.matching import matching as matching_mod


def snap(cells, tag, out):
    max_col = max((int(c["col_end"]) for c in cells), default=-1)
    synth = sorted(
        {
            (int(c["row_start"]), str(c.get("text") or "").replace("\n", " ")[:24])
            for c in cells
            if re.search(r"合成例", str(c.get("text") or ""))
        }
    )
    # 末两列文本
    tail = []
    for c in sorted(cells, key=lambda x: (x["row_start"], x["col_start"])):
        if int(c["col_end"]) < max_col - 1:
            continue
        t = str(c.get("text") or "").replace("\n", "/").strip()
        if not t:
            continue
        tail.append(
            f"  r{c['row_start']}-{c['row_end']} c{c['col_start']}-{c['col_end']}: {t[:40]}"
        )
    out.append(f"## {tag} n={len(cells)} maxc={max_col} synth={synth}")
    out.extend(tail[:40])
    # 关键 810/470 是否在任意格
    nums = [
        (c["row_start"], c["col_start"], str(c.get("text") or "").replace("\n", "/")[:20])
        for c in cells
        if re.search(r"\b(810|470|430|740)\b", str(c.get("text") or ""))
    ]
    out.append(f"  nums_in_cells={nums}")


def main():
    stages = []
    out: list[str] = []

    real_recon = refine.reconstruct_header_cells
    real_assign = matching_mod.assign_texts_to_cells
    real_norm = refine.normalize_oversegmented_table_rows
    real_lift = refine.lift_misplaced_header_labels
    real_promo = refine.promote_side_header_rowspans
    real_needs = refine.needs_monomer_header_reconstruct

    def wrap_needs(cells, boxes=None):
        v = real_needs(cells, boxes)
        out.append(f"## needs_monomer_header_reconstruct -> {v}")
        return v

    def wrap_recon(cells, boxes=None):
        snap(cells, "before_reconstruct", out)
        r = real_recon(cells, boxes)
        snap(r, "after_reconstruct", out)
        return r

    def wrap_assign(cells, boxes, **kw):
        snap(cells, "before_assign", out)
        r, free = real_assign(cells, boxes, **kw)
        snap(r, "after_assign", out)
        # free 里有没有目标数字
        free_hit = [
            str(t.get("text") or "")
            for t in (free or [])
            if re.search(r"810|470|430|740|双键", str(t.get("text") or ""))
        ]
        out.append(f"  free_hits={free_hit[:20]} free_n={len(free or [])}")
        return r, free

    def wrap_norm(cells, *a, **k):
        snap(cells, "before_norm", out)
        r = real_norm(cells, *a, **k)
        snap(r, "after_norm", out)
        return r

    def wrap_lift(cells, *a, **k):
        snap(cells, "before_lift", out)
        r = real_lift(cells, *a, **k)
        snap(r, "after_lift", out)
        return r

    def wrap_promo(cells, *a, **k):
        r = real_promo(cells, *a, **k)
        snap(r, "after_promo", out)
        return r

    refine.needs_monomer_header_reconstruct = wrap_needs
    refine.reconstruct_header_cells = wrap_recon
    refine.normalize_oversegmented_table_rows = wrap_norm
    refine.lift_misplaced_header_labels = wrap_lift
    refine.promote_side_header_rowspans = wrap_promo
    matching_mod.assign_texts_to_cells = wrap_assign
    pipe.needs_monomer_header_reconstruct = wrap_needs
    pipe.reconstruct_header_cells = wrap_recon
    pipe.normalize_oversegmented_table_rows = wrap_norm
    pipe.lift_misplaced_header_labels = wrap_lift
    pipe.promote_side_header_rowspans = wrap_promo
    pipe.assign_texts_to_cells = wrap_assign

    img = next((ROOT / "data" / "input").glob("*P98X878*"))
    extract_table_output(
        str(img), structure="tsr", use_cache=True, refresh_cache=False, save_vis=False
    )
    Path(ROOT / "tools/_lift_dbg.txt").write_text("\n".join(out), encoding="utf-8")
    print("ok", len(out))


if __name__ == "__main__":
    main()
