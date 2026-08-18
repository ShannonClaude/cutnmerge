"""
只读诊断工具：用于定位复杂表格在「结构识别→TSR后处理→融合→网格证据→OCR填格」中的卡点。

典型用法：
    python tools/inspect_grid.py --image "data/input/xxx.png" --out "data/debug/inspect_xxx"

该脚本会尽量复用 pipeline 内部同一套函数，输出：
  - OCR 文本框数量
  - TSR 原始 cells 数、各 refine 阶段 cells 数
  - 每阶段推导的 row_seps / col_seps（用于画线对照）
  - lines 检测出的表格数 + separator count
  - fuse 前后 cells 数
  - 多张叠加图（cells 多边形、row/col 边界线、lines网格）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.config import load_env
from src.structure.lines import binarize_otsu, detect_tables, imwrite_unicode, render_debug_overlay
from src.matching.matching import assign_texts_to_cells
from src.core.models import load_ocr, predict_texts
from src.preprocess.orient import apply_orientation_axis, ensure_upright_axis, maybe_flip_180_by_ocr
from src.core.pipeline import _load_image, deskew_image
from src.ocr.reocr import apply_reocr_to_cells
from src.structure.tsr import load_tsr_models, predict_cells_tsr
from src.structure.tsr_refine import (
    _derive_seps,
    dedupe_overlapping_cells,
    merge_ghost_columns,
    split_bad_colspans,
    split_underspanned_rows,
    unmerge_bad_rowspans,
)
from src.structure.grid_fusion import fuse_tsr_with_lines
from src.ocr.ocr_post import postprocess_text_boxes


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _cell_bbox(cell: Dict[str, Any]) -> Tuple[float, float, float, float]:
    poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
    return float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max())


def _draw_cells(
    image_bgr: np.ndarray,
    cells: List[Dict[str, Any]],
    *,
    color: Tuple[int, int, int] = (0, 200, 0),
    thickness: int = 2,
    draw_labels: bool = False,
) -> np.ndarray:
    out = image_bgr.copy()
    for i, cell in enumerate(cells):
        poly = np.asarray(cell["polygon"], dtype=np.int32).reshape(-1, 2)
        cv2.polylines(out, [poly], True, color, thickness)
        if draw_labels:
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            cv2.putText(out, f"{cell.get('row_start')},{cell.get('col_start')}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return out


def _draw_seps(
    image_bgr: np.ndarray,
    row_seps: List[float],
    col_seps: List[float],
    *,
    color_h: Tuple[int, int, int] = (255, 150, 0),
    color_v: Tuple[int, int, int] = (0, 120, 255),
    thickness: int = 1,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    # row_seps: y
    for y in row_seps:
        yi = int(round(y))
        if 0 <= yi < h:
            cv2.line(out, (0, yi), (w, yi), color_h, thickness)
    # col_seps: x
    for x in col_seps:
        xi = int(round(x))
        if 0 <= xi < w:
            cv2.line(out, (xi, 0), (xi, h), color_v, thickness)
    return out


def inspect_image(
    image_path: Path,
    out_dir: Path,
    *,
    use_cache: bool,
    refresh_cache: bool,
    deskew: bool,
    max_skew_angle: float,
    orientation: str,
    ioa_threshold: float,
    reocr: bool,
    reocr_max_cells: int,
    fallback_lines: bool,
    debug_tag: str = "",
) -> None:
    _ensure_dir(out_dir)
    stem = image_path.stem

    meta: Dict[str, Any] = {"image": image_path.name, "stem": stem, "out_dir": str(out_dir), "tag": debug_tag}

    image = _load_image(str(image_path))

    # ---------- Orientation + OCR (复现 pipeline 的次序) ----------
    ocr = load_ocr()
    image, axis_angle, orient_kind = apply_orientation_axis(image, mode=orientation)
    orient_angle = int(axis_angle)

    if deskew:
        image = deskew_image(image, max_angle=max_skew_angle)

    meta["orientation"] = {
        "mode": orientation,
        "orient_angle": orient_angle,
        "orient_kind": orient_kind,
        "deskew": int(deskew),
    }

    cache_extra = f"inspect|deskew={int(deskew)}|orient={orient_angle}"
    text_boxes = predict_texts(
        image,
        ocr_engine=ocr,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        cache_extra=cache_extra,
    )

    if orient_kind == "auto":
        image, axis_delta, text_boxes = ensure_upright_axis(
            image,
            text_boxes,
            ocr_engine=ocr,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_extra_base=cache_extra,
        )
        orient_angle = (orient_angle + axis_delta) % 360
        if axis_delta:
            cache_extra = f"inspect|deskew={int(deskew)}|orient={orient_angle}"

        image, flip, text_boxes = maybe_flip_180_by_ocr(
            image,
            text_boxes,
            ocr_engine=ocr,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_extra_base=cache_extra,
        )
        orient_angle = (orient_angle + flip) % 360
        if axis_delta or flip:
            cache_extra = f"inspect|deskew={int(deskew)}|orient={orient_angle}"

    meta["orientation_after_ocr"] = {"orient_angle": orient_angle}
    meta["ocr"] = {"text_boxes": len(text_boxes)}

    binary = binarize_otsu(image)
    text_boxes = postprocess_text_boxes(text_boxes, binary=binary)
    meta["ocr_post"] = {"text_boxes_after_postprocess": len(text_boxes)}

    # ---------- TSR + refine ----------
    load_tsr_models()
    tsr_cells = predict_cells_tsr(image, text_boxes=text_boxes)
    meta["tsr"] = {"cells_raw": len(tsr_cells)}

    # Stage overlays
    stage_images: List[Tuple[str, str]] = []
    vis_raw = _draw_cells(image, tsr_cells, color=(0, 200, 0), thickness=2, draw_labels=False)
    imwrite_unicode(str(out_dir / f"{stem}{debug_tag}_tsr_cells.png"), vis_raw)

    stages: List[Tuple[str, List[Dict[str, Any]]]] = [
        ("raw", tsr_cells),
    ]

    c1 = dedupe_overlapping_cells(tsr_cells)
    stages.append(("dedupe1", c1))

    c2 = merge_ghost_columns(c1, text_boxes)
    stages.append(("ghost_merge", c2))

    c3 = dedupe_overlapping_cells(c2)
    stages.append(("dedupe2", c3))

    c4 = unmerge_bad_rowspans(c3, text_boxes)
    stages.append(("unmerge_rowspan", c4))

    c5 = split_underspanned_rows(c4, text_boxes)
    stages.append(("split_underspanned_rows", c5))

    c6 = split_bad_colspans(c5, text_boxes)
    stages.append(("split_bad_colspans", c6))

    c7 = dedupe_overlapping_cells(c6)
    stages.append(("dedupe3", c7))

    for name, cells in stages:
        meta[name] = {"cells": len(cells)}
        if cells:
            row_seps, col_seps = _derive_seps(cells)
            meta[name]["row_seps"] = row_seps[:]
            meta[name]["col_seps"] = col_seps[:]

            vis = _draw_cells(image, cells, color=(0, 200, 0), thickness=2, draw_labels=False)
            vis = _draw_seps(vis, row_seps, col_seps)
            imwrite_unicode(str(out_dir / f"{stem}{debug_tag}_{name}_seps.png"), vis)

    # ---------- Fusion with lines ----------
    tables = detect_tables(image, confidence_thresh=0.0)
    meta["lines_detected_tables"] = {"count": len(tables), "top_conf": (max([t.confidence for t in tables]) if tables else None)}
    if tables:
        fused = fuse_tsr_with_lines(c7, tables)
    else:
        fused = c7
    meta["fusion"] = {"cells_fused": len(fused)}

    vis_fused = _draw_cells(image, fused, color=(0, 0, 255), thickness=2, draw_labels=False)
    imwrite_unicode(str(out_dir / f"{stem}{debug_tag}_fused_cells.png"), vis_fused)

    # ---------- Optional: match texts ----------
    if ioa_threshold is not None:
        # 注意：匹配本身会改变 cell["texts"]/cell["text"]，但用于 debug 更直观。
        matched_cells, free_texts = assign_texts_to_cells(
            fused,
            text_boxes,
            ioa_threshold=ioa_threshold,
            split_cross_cell=True,
            table_bboxes=None,
            binary=binary,
            col_seps=None,
            v_separators=None,
        )
        if reocr and matched_cells:
            matched_cells = apply_reocr_to_cells(
                image,
                matched_cells,
                binary=binary,
                ocr_engine=ocr,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
                max_cells=reocr_max_cells,
            )
        nonempty = sum(1 for c in matched_cells if str(c.get("text") or "").strip())
        meta["match"] = {"matched_cells": len(matched_cells), "nonempty_cells": nonempty, "free_texts": len(free_texts)}

    # ---------- Lines overlay ----------
    if tables:
        overlay = render_debug_overlay(image, tables, text_boxes=text_boxes)
        imwrite_unicode(str(out_dir / f"{stem}{debug_tag}_lines_overlay.png"), overlay)

    (out_dir / f"{stem}{debug_tag}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="复杂表格处理器诊断工具（只读）。")
    p.add_argument("--image", required=True, type=str)
    p.add_argument("--out", required=False, type=str, default=None)
    p.add_argument("--use-cache", action="store_true", default=True)
    p.add_argument("--refresh-cache", action="store_true", default=False)
    p.add_argument("--no-deskew", action="store_true", default=False)
    p.add_argument("--max-skew-angle", type=float, default=15.0)
    p.add_argument("--orientation", type=str, default="auto", choices=["auto", "none", "0", "90", "180", "270"])
    p.add_argument("--ioa-threshold", type=float, default=0.5)
    p.add_argument("--no-match", action="store_true", default=False)
    p.add_argument("--reocr", action="store_true", default=False)
    p.add_argument("--reocr-max-cells", type=int, default=24)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    load_env()
    args = parse_args(argv)
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到图片: {image_path}")

    out_dir = Path(args.out) if args.out else image_path.parent / "debug_inspect"
    debug_tag = ""

    inspect_image(
        image_path,
        out_dir,
        use_cache=args.use_cache,
        refresh_cache=args.refresh_cache,
        deskew=not args.no_deskew,
        max_skew_angle=args.max_skew_angle,
        orientation=args.orientation,
        ioa_threshold=None if args.no_match else float(args.ioa_threshold),
        reocr=args.reocr and not args.no_match,
        reocr_max_cells=args.reocr_max_cells,
        fallback_lines=False,
        debug_tag=debug_tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

