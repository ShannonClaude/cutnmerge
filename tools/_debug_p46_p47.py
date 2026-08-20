"""Dump post-fix topology for P46/P47."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_env

load_env()

from src.core.models import load_ocr, predict_texts
from src.core.pipeline import _extract_via_tsr, _load_image, apply_orientation_axis
from src.ocr.ocr_post import postprocess_text_boxes
from src.output.html_formatter import cells_to_html
from src.preprocess.orient import ensure_upright_axis, maybe_flip_180_by_ocr
from src.structure.lines import binarize_otsu
from src.structure.tsr import load_tsr_models
from src.utils.segments import find_row_segments


def dump(path: Path) -> None:
    print("=" * 72)
    print(path.name)
    image = _load_image(str(path))
    ocr = load_ocr()
    image, axis_angle, orient_kind = apply_orientation_axis(image, mode="auto")
    cache_extra = f"deskew=1|orient={int(axis_angle)}"
    boxes = predict_texts(
        image, ocr_engine=ocr, use_cache=True, cache_extra=cache_extra, artifact_stem=path.stem
    )
    if orient_kind == "auto":
        image, axis_delta, boxes = ensure_upright_axis(
            image, boxes, ocr_engine=ocr, use_cache=True, cache_extra_base=cache_extra
        )
        orient_angle = (int(axis_angle) + axis_delta) % 360
        cache_extra = f"deskew=1|orient={orient_angle}"
        image, flip, boxes = maybe_flip_180_by_ocr(
            image, boxes, ocr_engine=ocr, use_cache=True, cache_extra_base=cache_extra
        )
    binary = binarize_otsu(image)
    boxes = postprocess_text_boxes(boxes, binary=binary)

    outs, tables = _extract_via_tsr(
        image,
        boxes,
        ioa_threshold=0.5,
        compress_empty_cols=True,
        fallback_lines=False,
        tsr_aggressive=False,
    )
    cells = tables[0].cells if tables else []
    print(f"cells={len(cells)}")
    for c in sorted(cells, key=lambda x: (int(x["row_start"]), int(x["col_start"]))):
        t = str(c.get("text") or "").replace("\n", "/")[:56]
        print(
            "  r%s-%s c%s-%s | %s"
            % (c["row_start"], c["row_end"], c["col_start"], c["col_end"], t)
        )
    print("segments:", find_row_segments(cells, text_boxes=boxes))
    print("--- html ---")
    print(outs.get("html") or cells_to_html(cells))
    print()


def main() -> int:
    load_tsr_models()
    for pat in ("*P46*", "*P47*"):
        for p in sorted((ROOT / "data" / "input").glob(pat)):
            dump(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
