"""IoA 文本归属匹配与单元格内部文本排序拼接。

【框线网格路径】
单元格为严格矩形，优先用文本框中心点落入判定；弱重叠时再 IoA 兜底。
跨多列的 OCR 框（如「实施例19实施例20实施例21」）经几何门槛 + 墨水间隙校验后切分。

【格内拼接】
相邻两段若边界都是 CJK 字符则无空格拼接，避免中文换行被插入空格；
同行远距框强制补空格；孤立「一」类横线归一为「-」。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Point, box

from .geometry import compute_ioa, polygon_to_shapely

# 格内动态行分组：与上一组平均 Y 的容差
_ROW_Y_TOL_PX = 8.0
_CJK_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]"
)
_DASH_ONLY_RE = re.compile(r"^[一ー—─\-]+$")
# 跨格切分：每格覆盖文本框宽度比例 / 贴边容差
_CELL_COVER_FRAC = 0.55
_EDGE_ALIGN_FRAC = 0.30
_GAP_ABSORB_FRAC = 0.60  # 宽间隙吸附容差（相对字距）
_MIN_CELL_COVER_FRAC = 0.55
# 无空格重复标签：允许窄间隙 + 列宽比例校验
def _looks_like_repeated_header_units(text: str) -> bool:
    """
    通用“重复单元”检测：不依赖领域关键词。

    提取形如「(字母/中文段)+数字」的片段；若至少两段且前缀骨架一致，则认为是重复标签拼接。
    """
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    ms = list(re.finditer(r"([A-Za-z\u3400-\u9fff]+)\d+", compact))
    if len(ms) < 2:
        return False

    def _sk_prefix(prefix: str) -> str:
        out: List[str] = []
        for ch in prefix:
            if bool(_CJK_RE.fullmatch(ch)):
                out.append("C")
            elif ch.isalpha():
                out.append("A")
            else:
                out.append(ch)
        if not out:
            return ""
        s = "".join(out)
        comp: List[str] = [s[0]]
        for ch in s[1:]:
            if ch == comp[-1] and ch in {"A", "C"}:
                continue
            comp.append(ch)
        return "".join(comp)

    first = _sk_prefix(ms[0].group(1))
    if not first:
        return False
    for m in ms[1:]:
        if _sk_prefix(m.group(1)) != first:
            return False
    return True
# 行标签粘连：比较例1 86 Bk-1 → 切成多段
def _split_generic_row_label(tb_text: str) -> Optional[List[str]]:
    """
    通用数字粘连切分（不依赖比较例/实施例等词表）。

    - 若至少两个数字块：按第一个数字块结束切成两段
    - 若只有一个数字块且长度足够：尝试把数字拆成两段（适配无空格粘连）
    """
    t = (tb_text or "").strip()
    if not t:
        return None
    t = re.sub(r"\s+", " ", t)
    digits = list(re.finditer(r"\d+", t))
    if len(digits) >= 2:
        part1 = t[: digits[0].end()].strip()
        part2 = t[digits[0].end() :].strip()
        return [part1, part2] if part1 and part2 else None

    if len(digits) == 1:
        only = digits[0]
        prefix = t[: only.start()].strip()
        digits_str = only.group(0)
        rest = t[only.end() :].strip()
        if prefix and len(digits_str) >= 3:
            for split_at in (1, 2):
                if split_at >= len(digits_str):
                    continue
                part1 = (prefix + digits_str[:split_at]).strip()
                part2 = (digits_str[split_at:] + ((" " + rest) if rest else "")).strip()
                if part1 and part2:
                    return [part1, part2]
    return None


def _text_top_left(tb: Dict[str, Any]) -> Tuple[float, float]:
    """取文本框左上角 (x, y)，优先用 top_left，否则用 polygon 上 x+y 最小点。"""
    tl = tb.get("top_left")
    if tl is not None:
        return float(tl[0]), float(tl[1])
    poly = tb["polygon"]
    arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    idx = int(arr.sum(axis=1).argmin())
    return float(arr[idx, 0]), float(arr[idx, 1])


def _text_bbox(tb: Dict[str, Any]) -> Tuple[float, float, float, float]:
    poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )


def _cell_bbox(cell: Dict[str, Any]) -> Tuple[float, float, float, float]:
    poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )


def _x_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


def _is_cjk_char(ch: str) -> bool:
    return bool(ch) and bool(_CJK_RE.fullmatch(ch))


def _should_join_with_space(left: str, right: str) -> bool:
    """两段文本之间是否需要空格。CJK 接 CJK 不加空格。"""
    if not left or not right:
        return False
    if _is_cjk_char(left[-1]) and _is_cjk_char(right[0]):
        return False
    # 纯标点边界也不加
    if left[-1] in "([{（【「『" or right[0] in ")]}）】」』，。、；：,.!?;:":
        return False
    return True


def _char_width_weight(ch: str) -> float:
    """字宽模型：CJK 1.0 / ASCII 0.55 / 空格 0.3。"""
    if ch.isspace():
        return 0.3
    if _is_cjk_char(ch):
        return 1.0
    return 0.55


def _char_cumulative_widths(text: str) -> List[float]:
    """返回长度 n+1 的前缀宽：cum[i] = 前 i 个字符的总宽。"""
    cum = [0.0]
    for ch in text:
        cum.append(cum[-1] + _char_width_weight(ch))
    return cum


def _x_to_char_index(text: str, x_frac: float) -> int:
    """把文本框内相对位置 x_frac∈[0,1] 映射到字符切点索引。"""
    if not text:
        return 0
    cum = _char_cumulative_widths(text)
    total = cum[-1]
    if total <= 1e-9:
        return 0
    target = max(0.0, min(1.0, x_frac)) * total
    best_i = 0
    best_d = abs(cum[0] - target)
    for i, w in enumerate(cum):
        d = abs(w - target)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _ink_profile(
    binary: np.ndarray,
    tb_box: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, float, float, float]:
    """
    文本框内墨水列投影（框线列清零）。

    Returns:
        (ink_1d, abs_x0, box_h, pitch_est)
    """
    x1, y1, x2, y2 = tb_box
    h_img, w_img = binary.shape[:2]
    ix1 = max(0, int(np.floor(x1)))
    ix2 = min(w_img, int(np.ceil(x2)))
    iy1 = max(0, int(np.floor(y1)))
    iy2 = min(h_img, int(np.ceil(y2)))
    if ix2 - ix1 < 4 or iy2 - iy1 < 2:
        return np.zeros(0, dtype=np.float64), float(ix1), 1.0, 8.0

    roi = binary[iy1:iy2, ix1:ix2] > 0
    box_h = float(max(roi.shape[0], 1))
    col = roi.sum(axis=0).astype(np.float64)
    line_thresh = 0.85 * box_h
    ink = np.array([0.0 if c >= line_thresh else float(c) for c in col], dtype=np.float64)
    # 字距：用文本框宽度粗估
    pitch = max(4.0, (x2 - x1) / 12.0)
    return ink, float(ix1), box_h, pitch


def _boundary_has_ink_gutter(
    boundary_x: float,
    ink: np.ndarray,
    abs_x0: float,
    box_h: float,
    pitch: float,
    *,
    min_run: float = 0.0,
) -> Optional[Tuple[float, float]]:
    """
    格边界附近是否存在墨水沟。

    Returns:
        (snap_x, run_width) 或 None
    """
    if ink.size == 0:
        return None
    tol = max(2.0, _GAP_ABSORB_FRAC * pitch)
    lo = max(0, int(np.floor(boundary_x - tol - abs_x0)))
    hi = min(len(ink) - 1, int(np.ceil(boundary_x + tol - abs_x0)))
    if hi < lo:
        return None
    empty_thresh = max(1.0, 0.15 * box_h)
    # 在窗口内找连续空白 run
    best = None  # (width, snap)
    i = lo
    while i <= hi:
        if ink[i] <= empty_thresh:
            j = i
            while j <= hi and ink[j] <= empty_thresh:
                j += 1
            # 向窗外延伸以得到真实 run 宽
            a = i
            while a > 0 and ink[a - 1] <= empty_thresh:
                a -= 1
            b = j
            while b < len(ink) and ink[b] <= empty_thresh:
                b += 1
            width = float(b - a)
            snap = float(abs_x0 + (a + b) / 2.0)
            if width >= min_run and (best is None or width > best[0]):
                best = (width, snap)
            i = j
        else:
            i += 1
    if best is None:
        return None
    return best[1], best[0]


def _empty_run_median(
    ink: np.ndarray,
    box_h: float,
) -> float:
    if ink.size == 0:
        return 2.0
    empty_thresh = max(1.0, 0.15 * box_h)
    widths: List[float] = []
    i = 0
    n = len(ink)
    while i < n:
        if ink[i] <= empty_thresh:
            j = i + 1
            while j < n and ink[j] <= empty_thresh:
                j += 1
            if i > 1 and j < n - 1:
                widths.append(float(j - i))
            i = j
        else:
            i += 1
    if not widths:
        return 2.0
    return float(np.median(widths))


def _pieces_match_slot_widths(
    text: str,
    cuts: Sequence[int],
    slots: Sequence[Tuple[float, float]],
    *,
    max_rel_err: float = 0.28,
) -> bool:
    """切分后各段字符宽度是否与列宽大致成比例。"""
    if len(cuts) != len(slots) + 1:
        return False
    slot_ws = [max(b - a, 1.0) for a, b in slots]
    total_sw = sum(slot_ws)
    cum = _char_cumulative_widths(text)
    total_cw = max(cum[-1], 1e-6)
    for i, (a, b) in enumerate(slots):
        piece_w = cum[cuts[i + 1]] - cum[cuts[i]]
        expected = total_cw * (slot_ws[i] / total_sw)
        if expected < 1e-6:
            continue
        if abs(piece_w - expected) / expected > max_rel_err:
            return False
    return True


def _atomic_slots_from_col_seps(
    tb_box: Tuple[float, float, float, float],
    col_seps: Sequence[float],
    v_separators: Optional[Sequence[Any]] = None,
) -> List[Tuple[float, float]]:
    """
    文本框覆盖到的原子列区间。

    内部竖线必须在文本框 y 范围内有覆盖（spans），避免把仅存在于表身的
    补列线拿去切开表头 OCR 框。
    """
    if len(col_seps) < 2:
        return []
    xs = sorted(float(x) for x in col_seps)
    y_mid = (tb_box[1] + tb_box[3]) / 2.0

    if v_separators:
        active_internal: List[float] = []
        for sep in v_separators:
            x = float(sep.coord)
            # 跳过左右边框（用整段 xs 的首尾）
            if abs(x - xs[0]) <= 2.0 or abs(x - xs[-1]) <= 2.0:
                continue
            if not sep.spans:
                active_internal.append(x)
                continue
            if any(a - 3 <= y_mid <= b + 3 for a, b in sep.spans):
                active_internal.append(x)
        cut_xs = [xs[0]] + sorted(active_internal) + [xs[-1]]
    else:
        cut_xs = list(xs)

    # 去重
    cleaned: List[float] = []
    for x in sorted(cut_xs):
        if not cleaned or abs(x - cleaned[-1]) > 2.0:
            cleaned.append(x)
    cut_xs = cleaned
    if len(cut_xs) < 2:
        return []

    slots: List[Tuple[float, float]] = []
    for i in range(len(cut_xs) - 1):
        a, b = cut_xs[i], cut_xs[i + 1]
        w = b - a
        if w <= 1:
            continue
        xo = max(0.0, min(tb_box[2], b) - max(tb_box[0], a))
        if xo / w >= _MIN_CELL_COVER_FRAC:
            slots.append((a, b))
    return slots


def _make_atomic_cell(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> Dict[str, Any]:
    inset = 1.0
    x1i, x2i = x1 + inset, x2 - inset
    y1i, y2i = y1 + inset, y2 - inset
    if x2i <= x1i:
        x1i, x2i = x1, x2
    if y2i <= y1i:
        y1i, y2i = y1, y2
    polygon = np.array(
        [[x1i, y1i], [x2i, y1i], [x2i, y2i], [x1i, y2i]],
        dtype=np.float64,
    )
    return {
        "polygon": polygon,
        "x_key": x1i,
        "y_key": y1i,
        "row_start": int(row_start),
        "row_end": int(row_end),
        "col_start": int(col_start),
        "col_end": int(col_end),
        "row_span": int(row_end - row_start + 1),
        "col_span": int(col_end - col_start + 1),
        "texts": [],
        "text": "",
    }


def _explode_colspan_cell(
    cells: List[Dict[str, Any]],
    cell_idx: int,
    col_seps: Sequence[float],
) -> Dict[int, int]:
    """
    将 col_span>1 的单元格拆成逐列原子格，返回 {col_start: new_cell_idx}。

    用于跨列表头切分后能把片段写进不同逻辑列。
    """
    cell = cells[cell_idx]
    cs = int(cell["col_start"])
    ce = int(cell["col_end"])
    if ce <= cs:
        return {cs: cell_idx}
    if len(col_seps) <= ce + 1:
        return {cs: cell_idx}

    rs = int(cell["row_start"])
    re = int(cell["row_end"])
    cb = _cell_bbox(cell)
    y1, y2 = cb[1], cb[3]

    mapping: Dict[int, int] = {}
    first = True
    for c in range(cs, ce + 1):
        x1 = float(col_seps[c])
        x2 = float(col_seps[c + 1])
        new_cell = _make_atomic_cell(x1, y1, x2, y2, rs, re, c, c)
        if first:
            cells[cell_idx] = new_cell
            mapping[c] = cell_idx
            first = False
        else:
            cells.append(new_cell)
            mapping[c] = len(cells) - 1
    return mapping


def _col_index_for_slot(slot: Tuple[float, float], col_seps: Sequence[float]) -> int:
    mid = (slot[0] + slot[1]) / 2.0
    xs = list(col_seps)
    for i in range(len(xs) - 1):
        if xs[i] - 1 <= mid <= xs[i + 1] + 1:
            return i
    return -1


def _split_sticky_row_label(
    tb: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """
    将「比较例1 86 Bk-1」类粘连框按空白/紧凑模式切成多段，并按宽度比例切分 polygon。
    """
    text = str(tb.get("text") or "").strip()
    if not text:
        return None

    parts: List[str] = []
    parts = _split_generic_row_label(text) or []
    if not parts or len(parts) < 2:
        return None

    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None

    x1, y1, x2, y2 = _text_bbox(tb)
    width = max(x2 - x1, 1.0)
    weights = [max(len(p), 1) for p in parts]
    total_w = float(sum(weights))
    pieces: List[Dict[str, Any]] = []
    cursor = x1
    for i, part in enumerate(parts):
        seg_w = width * (weights[i] / total_w)
        right = x2 if i == len(parts) - 1 else cursor + seg_w
        piece = dict(tb)
        piece["text"] = part
        piece["polygon"] = np.array(
            [[cursor, y1], [right, y1], [right, y2], [cursor, y2]],
            dtype=np.float64,
        )
        piece["top_left"] = (cursor, y1)
        pieces.append(piece)
        cursor = right
    return pieces


def _geometric_multi_col_split(
    tb: Dict[str, Any],
    cells: List[Dict[str, Any]],
    text_shape: Any,
    ioa_threshold: float,
) -> bool:
    """
    无 binary / col_seps 时的跨列切分：文本框与同一行多个原子列 IoA 分散则按列切。
    """
    text = str(tb.get("text") or "")
    if len(text.strip()) < 2:
        return False

    tb_box = _text_bbox(tb)
    text_w = max(tb_box[2] - tb_box[0], 1.0)
    cy = (tb_box[1] + tb_box[3]) / 2.0

    candidates: List[Tuple[int, float, Tuple[float, float, float, float]]] = []
    for i, cell in enumerate(cells):
        if max(int(cell.get("row_span") or 1), 1) > 1:
            continue
        if max(int(cell.get("col_span") or 1), 1) > 1:
            continue
        cb = _cell_bbox(cell)
        cell_cy = (cb[1] + cb[3]) / 2.0
        row_tol = max(12.0, (cb[3] - cb[1]) * 0.55)
        if abs(cell_cy - cy) > row_tol:
            continue
        xo = _x_overlap(tb_box, cb)
        if xo / text_w < 0.12:
            continue
        cell_w = max(cb[2] - cb[0], 1.0)
        if xo / cell_w < 0.35:
            continue
        candidates.append((i, xo, cb))

    if len(candidates) < 2:
        return False

    candidates.sort(key=lambda t: t[2][0])
    # 文本须明显宽于单列均值
    mean_cw = float(np.mean([c[2][2] - c[2][0] for c in candidates]))
    if text_w < 1.35 * mean_cw:
        return False

    # 按列边界比例切字符
    cuts = [0]
    for j in range(len(candidates) - 1):
        boundary = candidates[j][2][2]
        frac = (boundary - tb_box[0]) / text_w
        cuts.append(_x_to_char_index(text, frac))
    cuts.append(len(text))
    for i in range(1, len(cuts)):
        if cuts[i] < cuts[i - 1]:
            cuts[i] = cuts[i - 1]

    assigned = False
    for si, (ci, _, cb) in enumerate(candidates):
        part = text[cuts[si] : cuts[si + 1]].strip()
        if not part:
            continue
        piece = dict(tb)
        piece["text"] = part
        xa = max(tb_box[0], cb[0])
        xb = min(tb_box[2], cb[2])
        if xb <= xa:
            xa, xb = cb[0], cb[2]
        piece["polygon"] = np.array(
            [[xa, tb_box[1]], [xb, tb_box[1]], [xb, tb_box[3]], [xa, tb_box[3]]],
            dtype=np.float64,
        )
        piece["top_left"] = (xa, tb_box[1])
        cells[ci]["texts"].append(piece)
        assigned = True
    return assigned


def _try_split_across_cells(
    tb: Dict[str, Any],
    cells: List[Dict[str, Any]],
    cell_shapes: List[Any],
    text_shape: Any,
    ioa_threshold: float,
    binary: Optional[np.ndarray] = None,
    col_seps: Optional[Sequence[float]] = None,
    v_separators: Optional[Sequence[Any]] = None,
) -> bool:
    """
    几何门槛 + 墨水间隙校验后，按字宽模型切分跨多列 OCR 框。

    优先用 col_seps 定义原子列（可切开表头合并格）；否则回退到非合并单元格。
    内部补列线仅在其 spans 覆盖文本 y 时才参与切分。

    Returns:
        True 表示已切分并归属，False 表示不满足切分条件。
    """
    if binary is None:
        return False

    text = str(tb.get("text") or "")
    if len(text.strip()) < 2:
        return False

    tb_box = _text_bbox(tb)
    text_w = max(tb_box[2] - tb_box[0], 1.0)
    cy = (tb_box[1] + tb_box[3]) / 2.0

    # ---- 候选列区间 ----
    slots: List[Tuple[float, float]] = []
    if col_seps is not None and len(col_seps) >= 3:
        slots = _atomic_slots_from_col_seps(tb_box, col_seps, v_separators=v_separators)

    # 表头合并格特殊路径：竖线只通表身时，仅对重复标签按合并格内部 col_seps 切
    if len(slots) < 2 and col_seps is not None:
        compact0 = re.sub(r"\s+", "", text)
        if _looks_like_repeated_header_units(text):
            for cell in cells:
                if max(int(cell.get("col_span") or 1), 1) < 2:
                    continue
                if max(int(cell.get("row_span") or 1), 1) > 1:
                    continue
                cb = _cell_bbox(cell)
                cell_cy = (cb[1] + cb[3]) / 2.0
                row_tol = max(12.0, (cb[3] - cb[1]) * 0.6)
                if abs(cell_cy - cy) > row_tol:
                    continue
                xo = _x_overlap(tb_box, cb)
                if xo < 0.8 * text_w:
                    continue
                cs, ce = int(cell["col_start"]), int(cell["col_end"])
                if ce <= cs or len(col_seps) <= ce + 1:
                    continue
                sub = [float(col_seps[i]) for i in range(cs, ce + 2)]
                cand = _atomic_slots_from_col_seps(tb_box, sub, v_separators=None)
                if len(cand) >= 2:
                    slots = cand
                    break

    cell_targets: List[Optional[int]] = []
    if len(slots) < 2:
        slots = []
        cell_targets = []
        for i, cell in enumerate(cells):
            if max(int(cell.get("row_span") or 1), 1) > 1:
                continue
            if max(int(cell.get("col_span") or 1), 1) > 1:
                continue
            cb = _cell_bbox(cell)
            cell_cy = (cb[1] + cb[3]) / 2.0
            row_tol = max(12.0, (cb[3] - cb[1]) * 0.6)
            if abs(cell_cy - cy) > row_tol:
                continue
            cell_w = max(cb[2] - cb[0], 1.0)
            xo = _x_overlap(tb_box, cb)
            if xo / cell_w < _MIN_CELL_COVER_FRAC:
                continue
            slots.append((cb[0], cb[2]))
            cell_targets.append(i)
        order = sorted(range(len(slots)), key=lambda k: slots[k][0])
        slots = [slots[k] for k in order]
        cell_targets = [cell_targets[k] for k in order]
    else:
        cell_targets = [None] * len(slots)
        for si, (a, b) in enumerate(slots):
            mid = (a + b) / 2.0
            best_i = None
            best_score = -1.0
            for i, cell in enumerate(cells):
                cb = _cell_bbox(cell)
                cell_cy = (cb[1] + cb[3]) / 2.0
                row_tol = max(12.0, (cb[3] - cb[1]) * 0.6)
                if abs(cell_cy - cy) > row_tol:
                    continue
                if cb[0] - 2 <= mid <= cb[2] + 2:
                    area = (cb[2] - cb[0]) * (cb[3] - cb[1])
                    score = 1.0 / max(area, 1.0)
                    if score > best_score:
                        best_score = score
                        best_i = i
            cell_targets[si] = best_i

    if len(slots) < 2:
        return False

    slot_widths = [b - a for a, b in slots]
    min_cw = min(slot_widths)
    mean_cw = float(np.mean(slot_widths))
    # 文本须明显宽于单列，避免把「(B) 成分」这类短标签切开
    if text_w < 1.5 * mean_cw:
        return False

    edge_tol = _EDGE_ALIGN_FRAC * min_cw
    if tb_box[0] - slots[0][0] > edge_tol * 2:
        return False
    if slots[-1][1] - tb_box[2] > edge_tol * 2:
        return False

    ink, abs_x0, box_h, pitch = _ink_profile(binary, tb_box)
    if ink.size == 0:
        return False
    char_pitch = max(4.0, text_w / max(len(text), 1))
    med_gap = _empty_run_median(ink, box_h)
    # 宽间隙：约 0.55 字宽；重复标签可走窄间隙例外
    wide_need = max(4.0, 0.55 * char_pitch)

    snapped: List[float] = []
    run_widths: List[float] = []
    for j in range(len(slots) - 1):
        boundary = slots[j][1]
        hit = _boundary_has_ink_gutter(boundary, ink, abs_x0, box_h, char_pitch)
        if hit is None:
            hit = _boundary_has_ink_gutter(slots[j + 1][0], ink, abs_x0, box_h, char_pitch)
        if hit is None:
            return False
        snap, run_w = hit
        snapped.append(snap)
        run_widths.append(run_w)

    cut_indices = [_x_to_char_index(text, (sx - tb_box[0]) / text_w) for sx in snapped]
    cuts = [0] + cut_indices + [len(text)]
    for i in range(1, len(cuts)):
        if cuts[i] < cuts[i - 1]:
            cuts[i] = cuts[i - 1]

    all_wide = all(w >= wide_need for w in run_widths)
    compact = re.sub(r"\s+", "", text)
    is_repeat = _looks_like_repeated_header_units(text)
    fits = _pieces_match_slot_widths(text, cuts, slots)
    if all_wide:
        # 非重复标签还须列宽比例匹配，避免「基团的|化合物」这类误切
        if not is_repeat and not fits:
            return False
    else:
        if not (is_repeat and fits):
            return False
    for i in range(len(slots)):
        part = text[cuts[i] : cuts[i + 1]].strip()
        slot_w = slots[i][1] - slots[i][0]
        if not part and slot_w > 0.5 * min_cw:
            return False

    # 若多个 slot 落到同一合并格，按 col_seps 拆开该格
    if col_seps is not None:
        unique_targets = {t for t in cell_targets if t is not None}
        for t in list(unique_targets):
            if max(int(cells[t].get("col_span") or 1), 1) <= 1:
                continue
            mapping = _explode_colspan_cell(cells, t, col_seps)
            for si, slot in enumerate(slots):
                if cell_targets[si] != t:
                    continue
                ci = _col_index_for_slot(slot, col_seps)
                if ci in mapping:
                    cell_targets[si] = mapping[ci]

    assigned_any = False
    for si, (a, b) in enumerate(slots):
        part = text[cuts[si] : cuts[si + 1]].strip()
        if not part:
            continue
        piece = dict(tb)
        piece["text"] = part
        x1 = max(tb_box[0], a)
        x2 = min(tb_box[2], b)
        y1, y2 = tb_box[1], tb_box[3]
        if x2 > x1:
            piece["polygon"] = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64
            )
            piece["top_left"] = (x1, y1)

        target = cell_targets[si]
        if target is None:
            mx = (a + b) / 2.0
            best_i, best_d = -1, 1e18
            for i, cell in enumerate(cells):
                cb = _cell_bbox(cell)
                if not (cb[1] - 5 <= cy <= cb[3] + 5):
                    continue
                if cb[0] <= mx <= cb[2]:
                    target = i
                    break
                d = min(abs(mx - cb[0]), abs(mx - cb[2]))
                if d < best_d:
                    best_d = d
                    best_i = i
            if target is None:
                target = best_i if best_i >= 0 else 0
        cells[int(target)]["texts"].append(piece)
        assigned_any = True
    return assigned_any


def assign_texts_to_cells(
    cells: List[Dict[str, Any]],
    text_boxes: List[Dict[str, Any]],
    ioa_threshold: float = 0.5,
    *,
    split_cross_cell: bool = True,
    table_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    binary: Optional[np.ndarray] = None,
    col_seps: Optional[Sequence[float]] = None,
    v_separators: Optional[Sequence[Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    将 OCR 文本框归属到单元格。

    匹配顺序：
    1. 可选：跨格切分（几何门槛 + 墨水间隙校验；可用 col_seps 切开合并表头）；
    2. 文本框中心落入某单元格 → 归属；
    3. best IoA > ioa_threshold → 归属；
    4. best IoA > 0 → 强制归属最高 IoA 格；
    5. 无交集：若提供 table_bboxes，仅在所有表格 bbox 之外才进 free_texts；
       否则一律进 free_texts。
    """
    free_texts: List[Dict[str, Any]] = []

    cell_shapes = [polygon_to_shapely(c["polygon"]) for c in cells]
    for cell in cells:
        cell["texts"] = []

    def _rebuild_rects():
        nonlocal cell_shapes, cell_rects
        cell_shapes = [polygon_to_shapely(c["polygon"]) for c in cells]
        cell_rects = []
        for cell in cells:
            x1, y1, x2, y2 = _cell_bbox(cell)
            cell_rects.append(box(x1, y1, x2, y2))

    # 预构建矩形 shapely（中心点命中更快）
    cell_rects: List[Any] = []
    _rebuild_rects()

    for tb in text_boxes:
        # 预切分已关闭：避免“无证据即切”把如表题/数字拆碎。
        # 跨格切分仅通过 _try_split_across_cells()（需要 binary/墨迹沟/几何校验）进行。
        for piece_tb in [tb]:
            text_shape = polygon_to_shapely(piece_tb["polygon"])
            centroid = text_shape.centroid
            center_pt = Point(centroid.x, centroid.y)

            n_before = len(cells)
            # ---- 跨格切分 ----
            split_ok = False
            if split_cross_cell:
                split_ok = _try_split_across_cells(
                    piece_tb,
                    cells,
                    cell_shapes,
                    text_shape,
                    ioa_threshold,
                    binary=binary,
                    col_seps=col_seps,
                    v_separators=v_separators,
                )
                if not split_ok:
                    split_ok = _geometric_multi_col_split(
                        piece_tb, cells, text_shape, ioa_threshold
                    )
            if split_ok:
                if len(cells) != n_before:
                    _rebuild_rects()
                continue

            # ---- 中心点落入：多格命中时取面积最小者（避免容器格吞并）----
            hits: List[int] = []
            for i, rect in enumerate(cell_rects):
                if rect.contains(center_pt) or rect.covers(center_pt):
                    hits.append(i)
            if hits:
                best_i = min(
                    hits,
                    key=lambda i: max(
                        float(cell_rects[i].area)
                        if hasattr(cell_rects[i], "area")
                        else 1.0,
                        1.0,
                    ),
                )
                cells[best_i]["texts"].append(piece_tb)
                continue

            # ---- IoA：相等时取面积最小格 ----
            best_idx = -1
            best_ioa = 0.0
            best_area = float("inf")
            for i, cell_shape in enumerate(cell_shapes):
                ioa = compute_ioa(text_shape, cell_shape)
                area = float(getattr(cell_shape, "area", 0.0) or 0.0)
                if ioa > best_ioa + 1e-9 or (
                    abs(ioa - best_ioa) <= 1e-9 and ioa > 0 and area < best_area
                ):
                    best_ioa = ioa
                    best_idx = i
                    best_area = area

            if best_idx >= 0 and best_ioa > ioa_threshold:
                cells[best_idx]["texts"].append(piece_tb)
                continue

            if best_idx >= 0 and best_ioa > 0:
                cells[best_idx]["texts"].append(piece_tb)
                continue

            # ---- 游离文本：仅表外 ----
            if table_bboxes:
                cx, cy = centroid.x, centroid.y
                inside_any = False
                for x1, y1, x2, y2 in table_bboxes:
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        inside_any = True
                        break
                if inside_any:
                    if cells:
                        dists = []
                        for cell in cells:
                            x1, y1, x2, y2 = _cell_bbox(cell)
                            dx = 0.0 if x1 <= cx <= x2 else min(abs(cx - x1), abs(cx - x2))
                            dy = 0.0 if y1 <= cy <= y2 else min(abs(cy - y1), abs(cy - y2))
                            dists.append(dx * dx + dy * dy)
                        cells[int(np.argmin(dists))]["texts"].append(piece_tb)
                    continue

            free_texts.append(piece_tb)

    for cell in cells:
        cell["text"] = join_cell_texts(cell.get("texts") or [])

    return cells, free_texts


def _normalize_dash_text(text: str) -> str:
    """整格仅为「一/ー/—/─/-」时归一为「-」。"""
    t = (text or "").strip()
    if t and _DASH_ONLY_RE.fullmatch(t):
        return "-"
    return text


def join_cell_texts(
    text_boxes: List[Dict[str, Any]],
    row_y_tol: float = _ROW_Y_TOL_PX,
) -> str:
    """
    同一单元格内多个 OCR 块按阅读顺序拼接。

    - 动态行分组（Y 容差）；
    - 行内按 X 排序；
    - CJK 相邻不加空格，其余加空格；
    - 同行相邻框 x 间距 > 0.8×框高时强制补空格。
    """
    if not text_boxes:
        return ""

    items: List[Tuple[float, float, Dict[str, Any]]] = []
    for tb in text_boxes:
        x, y = _text_top_left(tb)
        items.append((y, x, tb))
    items.sort(key=lambda t: (t[0], t[1]))

    rows: List[Tuple[float, int, List[Tuple[float, Dict[str, Any]]]]] = []
    for y, x, tb in items:
        if not rows:
            rows.append((y, 1, [(x, tb)]))
            continue
        y_sum, count, members = rows[-1]
        avg_y = y_sum / count
        if abs(y - avg_y) < row_y_tol:
            members.append((x, tb))
            rows[-1] = (y_sum + y, count + 1, members)
        else:
            rows.append((y, 1, [(x, tb)]))

    parts: List[str] = []
    row_parts: List[str] = []
    for _, _, members in rows:
        members.sort(key=lambda m: m[0])
        prev_right: Optional[float] = None
        prev_h: float = 10.0
        row_buf: List[str] = []
        for x, tb in members:
            text = str(tb.get("text") or "").strip()
            if not text:
                continue
            bx1, by1, bx2, by2 = _text_bbox(tb)
            box_h = max(by2 - by1, 1.0)
            force_space = False
            if prev_right is not None and (bx1 - prev_right) > 0.8 * max(prev_h, box_h):
                force_space = True
            if not row_buf:
                row_buf.append(text)
            elif force_space:
                row_buf.append(" " + text)
            elif _should_join_with_space(row_buf[-1], text):
                row_buf.append(" " + text)
            else:
                row_buf.append(text)
            prev_right = bx2
            prev_h = box_h
        row_text = "".join(row_buf).strip()
        if row_text:
            row_parts.append(row_text)
    if not row_parts:
        return ""
    return _normalize_dash_text("\n".join(row_parts))
