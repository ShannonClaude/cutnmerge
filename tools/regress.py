"""回归统计：cells / coverage / 丢字数（html.unescape 后再比对）。

用法:
    python tools/regress.py
    python tools/regress.py --refresh-cache
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
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
EXPECTED_PATH = ROOT / "data" / "expected.json"
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


def _load_expected(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected 文件格式错误: {path}")
    out: dict[str, dict] = {}
    for name, spec in raw.items():
        if isinstance(name, str) and isinstance(spec, dict):
            out[name] = spec
    return out


def _count_expected_lost(html: str, expected_texts: list[str]) -> tuple[int, list[str]]:
    body = _norm(html_lib.unescape(re.sub(r"<[^>]+>", "", html or "")))
    lost = []
    for t in expected_texts:
        nt = _norm(t)
        if nt and nt not in body:
            lost.append(t)
    return len(lost), lost[:8]


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")


def _parse_html_table_stats(html: str) -> dict[str, int | float]:
    """
    从真实输出 html 解析统计，用于“回归护栏”，避免 TSR 独立算出来的 rows/cols 与最终 html 不一致。
    """
    if not html:
        return {"rows": 0, "cols": 0, "empty_ratio": 0.0, "text_chars": 0}

    hu = html_lib.unescape(html)
    tables = re.findall(
        r"<table\b[^>]*>(.*?)</table>",
        hu,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not tables:
        tables = [hu]

    best_rows = 0
    best_cols = 0
    total_tds = 0
    empty_tds = 0

    for t_html in tables:
        row_segs = re.findall(
            r"<tr\b[^>]*>(.*?)</tr>",
            t_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        best_rows = max(best_rows, len(row_segs))
        for row_seg in row_segs:
            col_sum = 0
            tds = re.findall(
                r"<td\b([^>]*)>(.*?)</td>",
                row_seg,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not tds:
                continue
            for td_attr, td_inner in tds:
                total_tds += 1
                m = re.search(r'colspan\s*=\s*["\']?(\d+)', td_attr or "", flags=re.IGNORECASE)
                span = int(m.group(1)) if m else 1
                col_sum += span
                inner_txt = _strip_tags(td_inner).strip()
                inner_txt = re.sub(r"\s+", "", inner_txt)
                if inner_txt == "":
                    empty_tds += 1
            best_cols = max(best_cols, col_sum)

    all_text = _strip_tags(hu)
    all_text = re.sub(r"\s+", "", all_text)
    text_chars = len(all_text)
    empty_ratio = empty_tds / max(1, total_tds)
    return {
        "rows": int(best_rows),
        "cols": int(best_cols),
        "empty_ratio": float(empty_ratio),
        "text_chars": int(text_chars),
    }


def _parse_regress_stats_report(report_text: str) -> dict[str, dict[str, float | int]]:
    """
    解析 format_report 输出的 tab-separated 报告，用于与历史基线做护栏对比。
    只提取 empty_ratio/text_chars。
    """
    if not report_text:
        return {}
    lines = [ln for ln in report_text.splitlines() if ln.strip()]
    if not lines:
        return {}
    header = lines[0].split("\t")
    if "name" not in header or "empty_ratio" not in header or "text_chars" not in header:
        return {}
    idx_name = header.index("name")
    idx_empty = header.index("empty_ratio")
    idx_text = header.index("text_chars")

    out: dict[str, dict[str, float | int]] = {}
    for ln in lines[1:]:
        if ln.startswith("  "):
            continue
        parts = ln.split("\t")
        if len(parts) <= max(idx_name, idx_empty, idx_text):
            continue
        name = parts[idx_name]
        try:
            out[name] = {
                "empty_ratio": float(parts[idx_empty]),
                "text_chars": int(float(parts[idx_text])),
            }
        except Exception:
            continue
    return out


def run_one(
    path: Path,
    *,
    use_cache: bool,
    refresh_cache: bool,
    orientation: str,
    expected: dict | None = None,
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
    table_stats = _parse_html_table_stats(html)

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
    # rows/cols/empty_ratio/text_chars 直接来自真实 html，避免“另跑 TSR”造成的指标失真
    rows = int(table_stats["rows"])
    cols = int(table_stats["cols"])
    empty_ratio = float(table_stats["empty_ratio"])
    text_chars = int(table_stats["text_chars"])
    nonempty = sum(1 for c in filled if str(c.get("text") or "").strip())
    lost_n, lost_s = _count_lost(html, [str(t.get("text") or "") for t in tb])
    expected = expected or {}
    expected_rows = int(expected.get("rows", -1))
    expected_cols = int(expected.get("cols", -1))
    gt_lost, gt_lost_s = _count_expected_lost(
        html,
        [str(t) for t in expected.get("must_have", []) if str(t).strip()],
    )
    return {
        "name": path.name,
        "orient": result.get("orientation", orient),
        "boxes": len(tb),
        "cells": len(filled),
        "nonempty": nonempty,
        "rows": rows,
        "cols": cols,
        "maxrow": max_row,
        "maxcol": max_col,
        "expected_rows": expected_rows,
        "expected_cols": expected_cols,
        "cov": round(cov, 3),
        "empty_ratio": round(empty_ratio, 4),
        "text_chars": text_chars,
        "lost": lost_n,
        "lost_samples": [t[:20] for t in lost_s],
        "gt_lost": gt_lost,
        "gt_lost_samples": [t[:30] for t in gt_lost_s],
        "html_len": len(html),
    }


def format_report(rows: list[dict]) -> str:
    lines = [
        (
            "name\torient\tboxes\tcells\tnonempty\trows\tcols\texpected_rows"
            "\texpected_cols\tmaxrow\tmaxcol\tcov\tempty_ratio\ttext_chars\tguard_flags\tlost\tgt_lost\thtml_len"
        ),
    ]
    for r in rows:
        lines.append(
            f"{r['name']}\t{r['orient']}\t{r['boxes']}\t{r['cells']}\t"
            f"{r['nonempty']}\t{r['rows']}\t{r['cols']}\t{r['expected_rows']}\t"
            f"{r['expected_cols']}\t{r['maxrow']}\t{r['maxcol']}\t{r['cov']}\t"
            f"{r.get('empty_ratio', 0.0)}\t{r.get('text_chars', 0)}\t{r.get('guard_flags','')}\t"
            f"{r['lost']}\t{r['gt_lost']}\t{r['html_len']}"
        )
        if r["lost_samples"]:
            lines.append(f"  lost_samples: {r['lost_samples']}")
        if r["gt_lost_samples"]:
            lines.append(f"  gt_lost_samples: {r['gt_lost_samples']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="表格提取回归统计")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--expected-only",
        action="store_true",
        help="只运行 data/expected.json 中列出的图片（用于快速验证）",
    )
    parser.add_argument(
        "--orientation",
        default="auto",
        choices=["auto", "none", "0", "90", "180", "270"],
    )
    parser.add_argument("--out", default=str(DEBUG_DIR / "_regress_stats.txt"))
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--expected", default=str(EXPECTED_PATH))
    args = parser.parse_args(argv)

    images = sorted(
        p
        for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"无输入图片: {INPUT_DIR}", file=sys.stderr)
        return 1

    expected = _load_expected(Path(args.expected))
    if args.expected_only and expected:
        images = [p for p in images if p.name in expected]
    baseline_stats = {}
    if args.baseline and Path(args.baseline).is_file():
        try:
            baseline_text = Path(args.baseline).read_text(encoding="utf-8")
            baseline_stats = _parse_regress_stats_report(baseline_text)
        except Exception:  # noqa: BLE001
            baseline_stats = {}

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
                    expected=expected.get(path.name),
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
                    "rows": 0,
                    "cols": 0,
                    "maxrow": -1,
                    "maxcol": -1,
                    "expected_rows": -1,
                    "expected_cols": -1,
                    "cov": 0.0,
                    "lost": -1,
                    "lost_samples": [str(exc)[:40]],
                    "gt_lost": -1,
                    "gt_lost_samples": [],
                    "html_len": 0,
                }
            )

    # 护栏：empty_ratio 不升高；text_chars 不下降
    if baseline_stats:
        for r in rows:
            b = baseline_stats.get(r.get("name", ""))
            if not b:
                r["guard_flags"] = ""
                continue
            flags = []
            cur_empty = float(r.get("empty_ratio", 0.0))
            cur_text = int(r.get("text_chars", 0))
            if cur_empty > float(b["empty_ratio"]) + 1e-6:
                flags.append("empty_up")
            if cur_text < int(b["text_chars"]):
                flags.append("text_down")
            r["guard_flags"] = ",".join(flags)

        violations = [r.get("name", "") for r in rows if r.get("guard_flags")]
        if violations:
            print("guard violation (empty_ratio↑ or text_chars↓):")
            for n in violations[:20]:
                print(f" - {n}")

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
        if baseline_stats:
            if any(r.get("guard_flags") for r in rows):
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
