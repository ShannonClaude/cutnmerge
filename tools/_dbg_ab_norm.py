# -*- coding: utf-8 -*-
"""A/B：同一 after_assign 细胞，分别用 HEAD / WIP normalize。"""
from __future__ import annotations

import copy
import importlib.util
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


def load_head_normalize():
    path = ROOT / "tools" / "_head_tsr_refine.py"
    spec = importlib.util.spec_from_file_location("head_tsr", path)
    mod = importlib.util.module_from_spec(spec)
    # head file imports relative - won't work as standalone.
    # Instead exec just the function by importing current module's helpers
    return None


def summarize(cells):
    synth = sorted(
        str(c.get("text") or "").replace("\n", " ")[:20]
        for c in cells
        if re.search(r"合成例", str(c.get("text") or ""))
    )
    nums = sorted(
        {
            re.sub(r"\s+", "", str(c.get("text") or ""))
            for c in cells
            if re.fullmatch(r"\d+(?:\.\d+)?", re.sub(r"\s+", "", str(c.get("text") or "")))
        }
    )
    return f"n={len(cells)} synth={synth} nums={nums}"


def main():
    held = {}

    real_assign = matching_mod.assign_texts_to_cells
    real_norm = refine.normalize_oversegmented_table_rows

    def wrap_assign(cells, boxes, **kw):
        r, free = real_assign(cells, boxes, **kw)
        held["after_assign"] = copy.deepcopy(r)
        return r, free

    def wrap_norm(cells, *a, **k):
        held["before_norm"] = copy.deepcopy(cells)
        # skip actual norm during extract; we'll apply manually
        return cells

    matching_mod.assign_texts_to_cells = wrap_assign
    pipe.assign_texts_to_cells = wrap_assign
    refine.normalize_oversegmented_table_rows = wrap_norm
    pipe.normalize_oversegmented_table_rows = wrap_norm
    # also skip lift/promo for clean test
    refine.lift_misplaced_header_labels = lambda c: c
    refine.promote_side_header_rowspans = lambda c: c
    pipe.lift_misplaced_header_labels = lambda c: c
    pipe.promote_side_header_rowspans = lambda c: c

    img = next((ROOT / "data" / "input").glob("*P98X878*"))
    extract_table_output(
        str(img), structure="tsr", use_cache=True, refresh_cache=False, save_vis=False
    )
    cells = held["after_assign"]
    lines = [f"after_assign {summarize(cells)}"]

    # WIP normalize
    out_wip = real_norm(copy.deepcopy(cells))
    lines.append(f"wip_norm {summarize(out_wip)}")

    # Simulate HEAD peer-skip (narrower) by patching temporarily
    import src.structure.tsr_refine as tr

    src = Path(tr.__file__).read_text(encoding="utf-8")
    # run HEAD version of peer filter only: monkeypatch the regex path
    # Easiest: temporarily edit the function's peer skip by wrapping

    original = tr.normalize_oversegmented_table_rows

    # Build HEAD-like normalize by copying WIP and restoring old peer regex via exec patch
    peer_pat_wip = (
        r"(衍生物|化合物|共聚成分|有机硅烷|封端剂|低聚物|"
        r"酸当量|双键当量|四羧酸|二羧酸酐|含有比率|含有率|"
        r"来自具有|来源于具有|烯键式|烯属不饱和)"
    )
    peer_pat_head = r"(衍生物|化合物|共聚成分|有机硅烷|封端剂|低聚物)"

    # Patch source of peer check by replacing in a local copy of the function code — 
    # simpler approach: temporarily replace the regex string in the function via bytecode? no.
    # Just call with a monkeypatched re.search wrapper for that pattern.

    calls = {"head_mode": False}
    real_re_search = re.search

    def hooked_search(pat, string, flags=0):
        if calls["head_mode"] and isinstance(pat, str) and "酸当量" in pat:
            pat = peer_pat_head
        return real_re_search(pat, string, flags)

    # Also need headerish guard off for head
    # Instead duplicate logic: run wip with patched peer only
    tr_re = tr.re
    old_search = tr_re.search

    def head_search(pat, string, flags=0):
        if isinstance(pat, str) and "酸当量" in pat and "衍生物" in pat:
            return old_search(peer_pat_head, string, flags)
        return old_search(pat, string, flags)

    tr_re.search = head_search
    # Also disable headerish by making MONOMER checks fail inside headerish? 
    # Keep headerish - HEAD didn't have it. Disable by clearing anchors temporarily.
    old_mon = tr._MONOMER_PARENT_RE
    old_left = tr._MONOMER_LEFT_ANCHOR_RE
    # For headerish only — can't easily. Run with peer patch only first.
    out_peer = original(copy.deepcopy(cells))
    lines.append(f"peer_head_only {summarize(out_peer)}")
    tr_re.search = old_search

    # Stricter rollback test: apply wip then force check
    lines.append("--- rollback would fire? ---")
    synth_in = sum(1 for c in cells if re.search(r"合成例", str(c.get("text") or "")))
    synth_out = sum(1 for c in out_wip if re.search(r"合成例", str(c.get("text") or "")))
    lines.append(f"synth {synth_in}->{synth_out} strict_rollback={synth_out < synth_in}")

    Path(ROOT / "tools/_chk.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
