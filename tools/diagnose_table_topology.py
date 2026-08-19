"""单图表格拓扑诊断：orientation、TSR vs lines、行列统计。

用法:
    python tools/diagnose_table_topology.py --image data/input/foo.png
    python tools/diagnose_table_topology.py --image foo.png --structure lines --debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.core.pipeline import _imread_unicode
from src.core.models import predict_texts
from src.preprocess.orient import ensure_upright_axis
from src.structure.lines import binarize_otsu, detect_tables
from src.structure.refine import refine_table
from src.structure.transpose_fix import detect_transposed_table
from src.structure.tsr import predict_cells_tsr
from src.structure.tsr_refine import coverage_score, refine_tsr_cells_light


def _grid_shape(cells) -> tuple[int, int]:
    if not cells:
        return 0, 0
    max_r = max(int(c["row_end"]) for c in cells)
    max_c = max(int(c["col_end"]) for c in cells)
    return max_r + 1, max_c + 1


def _sample_labels(cells, *, axis: str, index: int, limit: int = 6) -> list[str]:
    out: list[str] = []
    for c in cells:
        if axis == "row":
            if int(c["row_start"]) != index:
                continue
        else:
            if int(c["col_start"]) != index:
                continue
        t = str(c.get("text") or "").strip()
        if t:
            out.append(t[:24])
        if len(out) >= limit:
            break
    return out


def diagnose(image_path: Path, *, structure: str, debug: bool) -> int:
    image = _imread_unicode(str(image_path))
    if image is None:
        print(f"无法读取: {image_path}")
        return 1

    text_boxes = predict_texts(image)
    upright, axis = ensure_upright_axis(image, text_boxes)
    print(f"image: {image_path.name}")
    print(f"orientation_axis: {axis}")
    print(f"ocr_boxes: {len(text_boxes)}")

    if structure in ("tsr", "both"):
        cells = predict_cells_tsr(upright, text_boxes=text_boxes)
        if cells:
            cells = refine_tsr_cells_light(cells)
        n_rows, n_cols = _grid_shape(cells)
        cov = coverage_score(cells, text_boxes) if cells else 0.0
        transposed = detect_transposed_table(cells, text_boxes) if cells else False
        print(f"tsr: cells={len(cells or [])} grid={n_rows}x{n_cols} cov={cov:.3f} transposed={transposed}")
        if cells:
            print(f"  row0: {_sample_labels(cells, axis='row', index=0)}")
            print(f"  col0: {_sample_labels(cells, axis='col', index=0)}")

    if structure in ("lines", "both"):
        tables = detect_tables(upright, confidence_thresh=0.0, text_boxes=text_boxes)
        if tables:
            binary = binarize_otsu(upright)
            tables = [refine_table(t, text_boxes) for t in tables]
            cells = max(tables, key=lambda t: len(t.cells or [])).cells
        else:
            cells = []
        n_rows, n_cols = _grid_shape(cells)
        cov = coverage_score(cells, text_boxes) if cells else 0.0
        transposed = detect_transposed_table(cells, text_boxes) if cells else False
        print(f"lines: cells={len(cells or [])} grid={n_rows}x{n_cols} cov={cov:.3f} transposed={transposed}")
        if cells:
            print(f"  row0: {_sample_labels(cells, axis='row', index=0)}")
            print(f"  col0: {_sample_labels(cells, axis='col', index=0)}")

    if debug:
        from src.structure.tsr import render_table_vis_logic

        out_dir = ROOT / "data" / "debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        if structure in ("tsr", "both"):
            cells = predict_cells_tsr(upright, text_boxes=text_boxes)
            if cells:
                vis = render_table_vis_logic(upright, cells)
                p = out_dir / f"{stem}_diag_tsr.png"
                cv2.imencode(".png", vis)[1].tofile(str(p))
                print(f"debug: {p}")
        if structure in ("lines", "both"):
            tables = detect_tables(upright, confidence_thresh=0.0, text_boxes=text_boxes)
            if tables:
                from src.structure.lines import render_debug_overlay

                vis = render_debug_overlay(upright, tables[0])
                p = out_dir / f"{stem}_diag_lines.png"
                cv2.imencode(".png", vis)[1].tofile(str(p))
                print(f"debug: {p}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="表格拓扑诊断")
    ap.add_argument("--image", required=True, help="输入图片路径")
    ap.add_argument(
        "--structure",
        choices=("tsr", "lines", "both"),
        default="both",
    )
    ap.add_argument("--debug", action="store_true", help="写出网格可视化")
    args = ap.parse_args()
    return diagnose(Path(args.image), structure=args.structure, debug=args.debug)


if __name__ == "__main__":
    raise SystemExit(main())
