"""
复杂表格解耦提取 Pipeline 入口。

用法:
    python main.py
        # 默认：tsr 结构，写出 data/output/<同名>.html
    python main.py --image data/input/demo.png
    python main.py --structure lines --debug
    python main.py --format both
    python main.py --no-cache
    python main.py --refresh-cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from src.config import load_env

load_env()

from src.pipeline import extract_table_output

# 项目根目录
ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "data" / "input"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "output"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_input_images(input_dir: Path = DEFAULT_INPUT_DIR) -> List[Path]:
    """扫描 data/input/，返回全部图片路径（按文件名排序）。"""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"默认输入目录不存在: {input_dir}")

    images = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise FileNotFoundError(
            f"未在 {input_dir} 找到测试图片，请放入 png/jpg 等表格截图后重试。"
        )
    return images


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="结构与文本解耦的复杂表格提取（默认 TSR + 云端 OCR + IoA → HTML）",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help=f"输入表格图片路径；默认扫描 {DEFAULT_INPUT_DIR} 下全部图片并批量处理",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出路径（单图模式）；扩展名可省略，将按 --format 补全。"
             "批量模式默认写入 data/output/<图片名>.html|.md",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="html",
        choices=["html", "md", "both"],
        help="输出格式：html（默认，保留 rowspan/colspan）/ md / both",
    )
    parser.add_argument(
        "--ioa-threshold",
        type=float,
        default=0.5,
        help="文本归属单元格的 IoA 阈值，默认 0.5",
    )
    parser.add_argument(
        "--structure",
        type=str,
        default="tsr",
        choices=["tsr", "lines", "lore", "auto"],
        help="结构来源：tsr（默认）/ lines / lore；auto 已废弃，等价于 tsr",
    )
    parser.add_argument(
        "--fallback-lines",
        action="store_true",
        help="仅 tsr：覆盖率过低时回退框线路径（默认关闭）",
    )
    parser.add_argument(
        "--no-deskew",
        action="store_true",
        help="关闭前置图像倾斜校正（Deskew），默认开启",
    )
    parser.add_argument(
        "--max-skew-angle",
        type=float,
        default=15.0,
        help="Deskew 允许校正的最大倾斜角（度），默认 15",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        default="auto",
        choices=["auto", "none", "0", "90", "180", "270"],
        help="方向归正：auto（默认，投影定轴+OCR 消歧180）/ none / 强制角度",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="禁用 OCR 本地缓存，每次都请求云端",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="强制重新请求云端 OCR 并覆盖本地缓存",
    )
    parser.add_argument(
        "--keep-empty-cols",
        action="store_true",
        help="保留整列为空的列（默认会压缩删除）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="写出网格叠加图到 data/debug/<stem>_grid.png",
    )
    return parser.parse_args(argv)


def _resolve_out_paths(
    image_path: Path,
    output_arg: str | None,
    fmt: str,
    *,
    single: bool,
) -> list[tuple[str, Path]]:
    """返回 [(format_key, path), ...]。"""
    stem = image_path.stem
    if single and output_arg:
        out = Path(output_arg)
        if out.suffix.lower() in {".html", ".htm", ".md", ".markdown"}:
            # 用户指定了明确扩展名
            key = "html" if out.suffix.lower() in {".html", ".htm"} else "md"
            if fmt == "both":
                other = "md" if key == "html" else "html"
                other_path = out.with_suffix(".md" if other == "md" else ".html")
                return [(key, out), (other, other_path)]
            return [(key if fmt == key else fmt, out if fmt == key else out.with_suffix(
                ".html" if fmt == "html" else ".md"
            ))]
        # 无扩展名：按 format 生成
        base = out
        if fmt == "html":
            return [("html", base.with_suffix(".html") if base.suffix == "" else Path(str(base) + ".html"))]
        if fmt == "md":
            return [("md", base.with_suffix(".md") if base.suffix == "" else Path(str(base) + ".md"))]
        return [
            ("html", Path(str(base) + ".html") if base.suffix == "" else base.with_suffix(".html")),
            ("md", Path(str(base) + ".md") if base.suffix == "" else base.with_suffix(".md")),
        ]

    if fmt == "html":
        return [("html", DEFAULT_OUTPUT_DIR / f"{stem}.html")]
    if fmt == "md":
        return [("md", DEFAULT_OUTPUT_DIR / f"{stem}.md")]
    return [
        ("html", DEFAULT_OUTPUT_DIR / f"{stem}.html"),
        ("md", DEFAULT_OUTPUT_DIR / f"{stem}.md"),
    ]


def process_one(
    image_path: Path,
    out_specs: list[tuple[str, Path]],
    *,
    ioa_threshold: float,
    deskew: bool,
    max_skew_angle: float,
    structure: str,
    use_cache: bool,
    refresh_cache: bool,
    compress_empty_cols: bool,
    fallback_lines: bool,
    orientation: str,
    debug: bool,
    lore_pipe=None,
    ocr_engine=None,
) -> None:
    """处理单张图片并按 format 写入。"""
    print(f"[info] 处理图像: {image_path}")
    result = extract_table_output(
        str(image_path),
        ioa_threshold=ioa_threshold,
        deskew=deskew,
        max_skew_angle=max_skew_angle,
        lore_pipe=lore_pipe,
        ocr_engine=ocr_engine,
        structure=structure,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        compress_empty_cols=compress_empty_cols,
        fallback_lines=fallback_lines,
        orientation=orientation,
        debug=debug,
        debug_stem=image_path.stem,
    )

    for key, out_path in out_specs:
        content = result.get(key) or ""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        try:
            print(content)
        except UnicodeEncodeError:
            print(content.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
                sys.stdout.encoding or "utf-8", errors="replace"
            ))
        print(f"[info] 已写入: {out_path}\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.image:
        image_path = Path(args.image)
        if not image_path.is_file():
            alt = ROOT / args.image
            if alt.is_file():
                image_path = alt
            else:
                print(f"错误: 找不到图片 {args.image}", file=sys.stderr)
                return 1
        images = [image_path]
        single = True
    else:
        try:
            images = list_input_images()
        except FileNotFoundError as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 1
        single = False

    print(
        f"[info] 共 {len(images)} 张图片待处理 "
        f"(structure={args.structure}, format={args.format})"
    )

    from src.models import load_ocr

    ocr = load_ocr()
    lore = None
    if args.structure == "lore":
        from src.models import load_lore_model

        lore = load_lore_model()
    if args.structure in {"tsr", "auto"}:
        from src.tsr import load_tsr_models

        print("[info] 预加载 TableStructureRec…")
        load_tsr_models()

    ok, fail = 0, 0
    for i, image_path in enumerate(images, start=1):
        out_specs = _resolve_out_paths(
            image_path, args.output, args.format, single=single and len(images) == 1
        )
        print(f"[info] ({i}/{len(images)}) {image_path.name}")
        try:
            process_one(
                image_path,
                out_specs,
                ioa_threshold=args.ioa_threshold,
                deskew=not args.no_deskew,
                max_skew_angle=args.max_skew_angle,
                structure=args.structure,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
                compress_empty_cols=not args.keep_empty_cols,
                fallback_lines=args.fallback_lines,
                orientation=args.orientation,
                debug=args.debug,
                lore_pipe=lore,
                ocr_engine=ocr,
            )
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[error] 处理失败 {image_path.name}: {exc}", file=sys.stderr)
            import traceback

            traceback.print_exc()

    print(f"[info] 完成: 成功 {ok}，失败 {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
