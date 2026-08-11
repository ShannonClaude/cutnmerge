"""回归统计：cells / coverage / 丢字数（html.unescape 后再比对）。

用法:
    python tools/regress.py
    python tools/regress.py --refresh-cache
"""

from __future__ import annotations

import argparse
import html as html_lib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()

from src.lines import binarize_otsu  # noqa: E402
from src.matching import assign_texts_to_cells  # noqa: E402
from src.models import predict_texts  # noqa: E402
from src.ocr_post import postprocess_text_boxes  # noqa: E402
from src.orient import (  # noqa: E402
    apply_orientation_axis,
    ensure_upright_axis,
    maybe_flip_180_by_ocr,
)
from src.pipeline import _load_image, deskew_image, extract_table_output  # noqa: E402
from src.tsr import predict_cells_tsr  # noqa: E402
from src.tsr_refine import coverage_score, refine_tsr_cells  # noqa: E402

INPUT_DIR = ROOT / "data" / "input"
DEBUG_DIR = ROOT / "data" / "debug"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _count_lost(html: str, sample_texts: list[str]) -> tuple[int, list[str]]:
    body = _norm(html_lib.unescape(re.sub(r"<[^>]+>", "", html or "")))
    lost = []
    for t in sample_texts:
        nt = _norm(t)
        if nt and nt not in body:
            lost.append(t)
    return len(lost), lost[:8]


def run_one(
    path: Path,
    *,
    use_cache: bool,
    refresh_cache: bool,
    orientation: str,
) -> dict:
    result = extract_table_output(
        str(path),
        structure="tsr",
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        orientation=orientation,
        debug=False,
    )
    html = result.get("html") or ""

    img = _load_image(str(path))
    img, axis, kind = apply_orientation_axis(img, mode=orientation)
    img = deskew_image(img, max_angle=15.0)
    orient = int(axis)
    cache_extra = f"deskew=1|orient={orient}"
    boxes = predict_texts(
        img, use_cache=True, refresh_cache=False, cache_extra=cache_extra
    )
    if kind == "auto":
        img, axis_delta, boxes = ensure_upright_axis(
            img, boxes, use_cache=True, cache_extra_base=cache_extra
        )
        orient = (orient + axis_delta) % 360
        if axis_delta:
            cache_extra = f"deskew=1|orient={orient}"
        img, flip, boxes = maybe_flip_180_by_ocr(
            img, boxes, use_cache=True, cache_extra_base=cache_extra
        )
        orient = (orient + flip) % 360
    binary = binarize_otsu(img)
    tb = postprocess_text_boxes([dict(t) for t in boxes], binary=binary)
    cells = predict_cells_tsr(img, text_boxes=tb)
    if cells:
        cells = refine_tsr_cells(cells, tb)
    cov = coverage_score(cells, tb) if cells else 0.0
    if cells:
        filled, _ = assign_texts_to_cells(
            [dict(c) for c in cells],
            tb,
            ioa_threshold=0.5,
            split_cross_cell=True,
            binary=binary,
        )
    else:
        filled = []
    max_row = max((int(c["row_end"]) for c in filled), default=-1)
    max_col = max((int(c["col_end"]) for c in filled), default=-1)
    nonempty = sum(1 for c in filled if str(c.get("text") or "").strip())
    lost_n, lost_s = _count_lost(html, [str(t.get("text") or "") for t in tb])
    return {
        "name": path.name,
        "orient": result.get("orientation", orient),
        "boxes": len(tb),
        "cells": len(filled),
        "nonempty": nonempty,
        "maxrow": max_row,
        "maxcol": max_col,
        "cov": round(cov, 3),
        "lost": lost_n,
        "lost_samples": [t[:20] for t in lost_s],
        "html_len": len(html),
    }


def format_report(rows: list[dict]) -> str:
    lines = [
        "name\torient\tboxes\tcells\tnonempty\tmaxrow\tmaxcol\tcov\tlost\thtml_len",
    ]
    for r in rows:
        lines.append(
            f"{r['name']}\t{r['orient']}\t{r['boxes']}\t{r['cells']}\t"
            f"{r['nonempty']}\t{r['maxrow']}\t{r['maxcol']}\t{r['cov']}\t"
            f"{r['lost']}\t{r['html_len']}"
        )
        if r["lost_samples"]:
            lines.append(f"  lost_samples: {r['lost_samples']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="表格提取回归统计")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--orientation",
        default="auto",
        choices=["auto", "none", "0", "90", "180", "270"],
    )
    parser.add_argument("--out", default=str(DEBUG_DIR / "_regress_stats.txt"))
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args(argv)

    images = sorted(
        p
        for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"无输入图片: {INPUT_DIR}", file=sys.stderr)
        return 1

    rows = []
    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {path.name}")
        try:
            rows.append(
                run_one(
                    path,
                    use_cache=not args.no_cache,
                    refresh_cache=args.refresh_cache,
                    orientation=args.orientation,
                )
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            rows.append(
                {
                    "name": path.name,
                    "orient": -1,
                    "boxes": 0,
                    "cells": 0,
                    "nonempty": 0,
                    "maxrow": -1,
                    "maxcol": -1,
                    "cov": 0.0,
                    "lost": -1,
                    "lost_samples": [str(exc)[:40]],
                    "html_len": 0,
                }
            )

    report = format_report(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"written: {out_path}")
    if args.baseline:
        bp = Path(args.baseline)
        bp.parent.mkdir(parents=True, exist_ok=True)
        bp.write_text(report, encoding="utf-8")
        print(f"baseline: {bp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
