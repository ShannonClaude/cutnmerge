"""子表行段切分：供结构修复与渲染共用。

在文本归属前，cell.text 可能为空，因此可选传入 OCR text_boxes，
用落入该逻辑行物理带的 OCR 文本做「换头 / 表题」判定。
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 相邻逻辑行物理 Y 间距超过此阈值则切分为新子表
_Y_GAP_THRESH_PX = 48.0
# 通用表题：表 / Table / Tab. / 図 / Fig + 编号
_HEADER_CAPTION_RE = re.compile(
    r"(?:\[?\s*(?:表|図)\s*[\d\-ー－]+|\b(?:Table|Tab\.?|Fig\.?|Figure)\s*[\d\-]+)",
    re.IGNORECASE,
)
# 重复表头 Jaccard 阈值
_HEADER_JACCARD_THRESH = 0.28
# 分段异形表：中段再出现「聚合物 / 单体[」等强表头信号时切开
# 注意：不要用单独的「有机硅烷/酸当量」——子表头行也会命中，会把同一子表切碎
_SECTION_HEADER_RE = re.compile(
    r"(聚合物|单体\s*[\[［])"
)
_DATA_ROW_RE = re.compile(r"(合成例|实施例|実施例|比較例|比较例)\s*\d*")


def _tokenize_row_text(joined: str) -> set:
    """把行文本切成可比对的 token 集合（中文按字、拉丁按词）。"""
    s = (joined or "").strip().lower()
    if not s:
        return set()
    latin = set(re.findall(r"[a-z0-9][a-z0-9\.\-/%]{0,24}", s))
    cjk_chunks = re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]+", s)
    cjk: set = set()
    for ch in cjk_chunks:
        if len(ch) <= 2:
            cjk.add(ch)
        else:
            cjk.update(ch[i : i + 2] for i in range(len(ch) - 1))
    return latin | cjk


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _cell_y_top(cell: Dict[str, Any]) -> float:
    if cell.get("y_key") is not None:
        return float(cell["y_key"])
    poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
    return float(poly[:, 1].min())


def _cell_y_bottom(cell: Dict[str, Any]) -> float:
    poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
    return float(poly[:, 1].max())


def _tb_bbox(tb: Dict[str, Any]) -> Tuple[float, float, float, float]:
    poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )


def _row_joined_text(
    row_cells: List[Dict[str, Any]],
    *,
    y_top: float,
    y_bot: float,
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """优先 cell.text；为空时用落入该行 Y 带的 OCR 文本。"""
    cell_parts = [str(c.get("text") or "").strip() for c in row_cells]
    cell_joined = " ".join(t for t in cell_parts if t)
    if cell_joined.strip():
        return cell_joined
    if not text_boxes:
        return ""
    mid_lo, mid_hi = y_top, y_bot
    if mid_hi <= mid_lo:
        mid_hi = mid_lo + 1.0
    parts: List[str] = []
    for tb in text_boxes:
        x1, y1, x2, y2 = _tb_bbox(tb)
        my = (y1 + y2) / 2.0
        if mid_lo - 2 <= my <= mid_hi + 2:
            t = str(tb.get("text") or "").strip()
            if t:
                parts.append(t)
    return " ".join(parts)


def find_row_segments(
    cells: List[Dict[str, Any]],
    *,
    y_gap_thresh: float = _Y_GAP_THRESH_PX,
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Tuple[int, int]]:
    """
    按与 split_cells_into_subtables 相同的判据，返回全局逻辑行闭区间列表。

    Returns:
        [(row_lo, row_hi), ...]，每个区间覆盖一个子表的 row_start 范围。
    """
    if not cells:
        return []

    by_row: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_row[int(cell["row_start"])].append(cell)

    sorted_rows = sorted(by_row.keys())
    if not sorted_rows:
        return []

    row_meta: List[Tuple[int, float, float, List[Dict[str, Any]]]] = []
    for rs in sorted_rows:
        row_cells = by_row[rs]
        y_top = min(_cell_y_top(c) for c in row_cells)
        y_bot = max(_cell_y_bottom(c) for c in row_cells)
        row_meta.append((rs, y_top, y_bot, row_cells))

    segments: List[Tuple[int, int]] = []
    seg_start: Optional[int] = None
    seg_end: Optional[int] = None
    prev_y_bot: Optional[float] = None
    is_first_row = True
    header_tokens: set = set()
    # 本段是否已出现表体行：未出现前，禁止用「像表头」切开（避免多级子表头误切）
    segment_seen_data = False

    for rs, y_top, y_bot, row_cells in row_meta:
        joined = _row_joined_text(
            row_cells, y_top=y_top, y_bot=y_bot, text_boxes=text_boxes
        )
        toks = _tokenize_row_text(joined)
        should_split = False
        if seg_start is not None and prev_y_bot is not None:
            if (y_top - prev_y_bot) > y_gap_thresh:
                should_split = True
            elif (not is_first_row) and _HEADER_CAPTION_RE.search(joined):
                should_split = True
            elif (
                segment_seen_data
                and (not is_first_row)
                and _SECTION_HEADER_RE.search(joined)
            ):
                # 表体后再遇「聚合物/单体[」才切；表头带内的子表头不切
                if not _DATA_ROW_RE.search(joined):
                    should_split = True
            elif (
                segment_seen_data
                and (not is_first_row)
                and header_tokens
                and len(toks) >= 2
                and _jaccard(toks, header_tokens) >= _HEADER_JACCARD_THRESH
            ):
                # 同上：Jaccard 重复表头仅在已见过表体后生效
                should_split = True

        if should_split and seg_start is not None and seg_end is not None:
            # 钳制终点，避免与新段起始行重叠（父格 rowspan 会把 seg_end 扩到 rs）
            clamped_end = min(seg_end, rs - 1)
            if clamped_end >= seg_start:
                segments.append((seg_start, clamped_end))
            seg_start = None
            seg_end = None
            is_first_row = True
            header_tokens = set()
            segment_seen_data = False

        if is_first_row:
            header_tokens = toks
            seg_start = rs
        seg_end = max(int(c["row_end"]) for c in row_cells)
        # 也覆盖本行起点（row_end 可能小于后续行）
        if seg_end < rs:
            seg_end = rs
        if _DATA_ROW_RE.search(joined):
            segment_seen_data = True
        prev_y_bot = y_bot
        is_first_row = False

    if seg_start is not None and seg_end is not None:
        # 扩展到该段内所有 cell 的最大 row_end
        max_end = seg_end
        for cell in cells:
            rs = int(cell["row_start"])
            re_ = int(cell["row_end"])
            if seg_start <= rs <= max_end or seg_start <= re_ <= max_end:
                max_end = max(max_end, re_)
        segments.append((seg_start, max_end))

    if not segments:
        max_row = max(int(c["row_end"]) for c in cells)
        min_row = min(int(c["row_start"]) for c in cells)
        return [(min_row, max_row)]
    return segments
