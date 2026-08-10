"""
复杂表格解耦提取 Pipeline 入口。

用法:
    python main.py
        # 默认：扫描 data/input/ 下全部图片，逐张生成 data/output/<同名>.md
    python main.py --image data/input/demo.png
    python main.py --structure lines --debug
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

from src.pipeline import extract_table_markdown

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
        description="结构与文本解耦的复杂表格提取（框线网格 / LORE + 云端 OCR + IoA）",
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
        help="Markdown 输出路径；仅在指定 --image 单图模式时生效。"
             "批量模式默认写入 data/output/<图片名>.md",
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
        default="auto",
        choices=["auto", "lines", "lore"],
        help="结构来源：auto（默认，框线优先）/ lines（强制框线）/ lore（强制 LORE）",
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


def process_one(
    image_path: Path,
    out_path: Path,
    *,
    ioa_threshold: float,
    deskew: bool,
    max_skew_angle: float,
    structure: str,
    use_cache: bool,
    refresh_cache: bool,
    compress_empty_cols: bool,
    debug: bool,
    lore_pipe=None,
    ocr_engine=None,
) -> None:
    """处理单张图片并写入对应 Markdown。"""
    print(f"[info] 处理图像: {image_path}")
    markdown = extract_table_markdown(
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
        debug=debug,
        debug_stem=image_path.stem,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    try:
        print(markdown)
    except UnicodeEncodeError:
        # Windows 控制台常见 GBK，无法打印部分 OCR 字符；文件已写 UTF-8
        print(markdown.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8", errors="replace"
        ))
    print(f"[info] 已写入: {out_path}\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ---------- 解析待处理图片列表 ----------
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
    else:
        try:
            images = list_input_images()
        except FileNotFoundError as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 1

    print(f"[info] 共 {len(images)} 张图片待处理 (structure={args.structure})")

    # ---------- OCR 客户端预加载；LORE 仅在需要时加载 ----------
    from src.models import load_lore_model, load_ocr

    ocr = load_ocr()
    lore = None
    if args.structure in {"lore", "auto"}:
        # auto 可能回退；预先加载避免中途失败难排查。lines 强制时跳过。
        if args.structure == "lore":
            lore = load_lore_model()

    ok, fail = 0, 0
    for i, image_path in enumerate(images, start=1):
        if args.output and len(images) == 1:
            out_path = Path(args.output)
        else:
            out_path = DEFAULT_OUTPUT_DIR / f"{image_path.stem}.md"

        print(f"[info] ({i}/{len(images)}) {image_path.name}")
        try:
            # auto 且框线失败时再懒加载 LORE
            lore_pipe = lore
            if args.structure == "auto" and lore_pipe is None:
                # 不预加载；pipeline 内部需要时再 load
                lore_pipe = None

            process_one(
                image_path,
                out_path,
                ioa_threshold=args.ioa_threshold,
                deskew=not args.no_deskew,
                max_skew_angle=args.max_skew_angle,
                structure=args.structure,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
                compress_empty_cols=not args.keep_empty_cols,
                debug=args.debug,
                lore_pipe=lore_pipe,
                ocr_engine=ocr,
            )
            ok += 1
        except Exception as exc:  # noqa: BLE001 — 单张失败不阻断整批
            fail += 1
            print(f"[error] 处理失败 {image_path.name}: {exc}", file=sys.stderr)
            import traceback

            traceback.print_exc()

    print(f"[info] 完成: 成功 {ok}，失败 {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
