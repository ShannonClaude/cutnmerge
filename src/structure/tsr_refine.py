"""TSR 结构后处理：去重叠、幽灵列合并、错误 row/colspan 拆分。

TSR 不提供 row_seps/col_seps，从 cell 物理框推导网格边界后做拓扑修正。
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..utils.label_patterns import are_independent_row_labels, is_index_column

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]"
)
# 【新增】支持 2-3 个大写字母或字母加数字的合法代码识别
_LETTER_DATA_RE = re.compile(r"^(?:[A-Za-z][+＋]?|[A-Z]{2,3}|[A-Za-z]\d{1,2})$")


def _tb_bbox(tb: Dict[str, Any]) -> Tuple[float, float, float, float]:
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


def _cell_area(cell: Dict[str, Any]) -> float:
    x1, y1, x2, y2 = _cell_bbox(cell)
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def _bbox_inter_area(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _logic_strict_subset(child: Dict[str, Any], parent: Dict[str, Any]) -> bool:
    """True if child's logic rect is strictly inside parent's (not equal)."""
    crs, cre = int(child["row_start"]), int(child["row_end"])
    ccs, cce = int(child["col_start"]), int(child["col_end"])
    prs, pre = int(parent["row_start"]), int(parent["row_end"])
    pcs, pce = int(parent["col_start"]), int(parent["col_end"])
    contained = crs >= prs and cre <= pre and ccs >= pcs and cce <= pce
    equal = crs == prs and cre == pre and ccs == pcs and cce == pce
    return contained and not equal


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _logic_key(cell: Dict[str, Any]) -> Tuple[int, int, int, int]:
    return (
        int(cell["row_start"]),
        int(cell["row_end"]),
        int(cell["col_start"]),
        int(cell["col_end"]),
    )


def _refresh_spans(cell: Dict[str, Any]) -> None:
    cell["row_span"] = int(cell["row_end"]) - int(cell["row_start"]) + 1
    cell["col_span"] = int(cell["col_end"]) - int(cell["col_start"]) + 1


def _rebuild_polygon(
    x1: float, y1: float, x2: float, y2: float
) -> np.ndarray:
    inset = 0.5
    xa, xb = x1 + inset, x2 - inset
    ya, yb = y1 + inset, y2 - inset
    if xb <= xa:
        xa, xb = x1, x2
    if yb <= ya:
        ya, yb = y1, y2
    return np.array(
        [[xa, ya], [xb, ya], [xb, yb], [xa, yb]], dtype=np.float64
    )


def _derive_seps(cells: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    """从单元格物理框推导 row_seps / col_seps（边界坐标升序）。"""
    if not cells:
        return [], []

    max_row = max(int(c["row_end"]) for c in cells)
    max_col = max(int(c["col_end"]) for c in cells)
    n_rows, n_cols = max_row + 1, max_col + 1

    # 每行顶/底、每列左/右中位数
    row_tops: Dict[int, List[float]] = defaultdict(list)
    row_bots: Dict[int, List[float]] = defaultdict(list)
    col_lefts: Dict[int, List[float]] = defaultdict(list)
    col_rights: Dict[int, List[float]] = defaultdict(list)

    for c in cells:
        x1, y1, x2, y2 = _cell_bbox(c)
        rs, re = int(c["row_start"]), int(c["row_end"])
        cs, ce = int(c["col_start"]), int(c["col_end"])
        # 只对原子行/列贡献边界，合并格用起止
        row_tops[rs].append(y1)
        row_bots[re].append(y2)
        col_lefts[cs].append(x1)
        col_rights[ce].append(x2)

    row_seps: List[float] = []
    for r in range(n_rows):
        if r in row_tops and row_tops[r]:
            row_seps.append(float(np.median(row_tops[r])))
        elif row_seps:
            row_seps.append(row_seps[-1] + 10.0)
        else:
            row_seps.append(0.0)
    # 最后一条底边
    last_bots = row_bots.get(n_rows - 1) or []
    if last_bots:
        row_seps.append(float(np.median(last_bots)))
    else:
        row_seps.append(row_seps[-1] + 10.0)

    col_seps: List[float] = []
    for c in range(n_cols):
        if c in col_lefts and col_lefts[c]:
            col_seps.append(float(np.median(col_lefts[c])))
        elif col_seps:
            col_seps.append(col_seps[-1] + 10.0)
        else:
            col_seps.append(0.0)
    last_rights = col_rights.get(n_cols - 1) or []
    if last_rights:
        col_seps.append(float(np.median(last_rights)))
    else:
        col_seps.append(col_seps[-1] + 10.0)

    # 单调化
    for i in range(1, len(row_seps)):
        if row_seps[i] <= row_seps[i - 1]:
            row_seps[i] = row_seps[i - 1] + 1.0
    for i in range(1, len(col_seps)):
        if col_seps[i] <= col_seps[i - 1]:
            col_seps[i] = col_seps[i - 1] + 1.0
    return row_seps, col_seps


def dedupe_overlapping_cells(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """同一逻辑矩形去重；严重物理重叠 / 容器包含时丢冗余格。"""
    if not cells:
        return cells

    by_logic: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
    for cell in cells:
        key = _logic_key(cell)
        prev = by_logic.get(key)
        if prev is None or _cell_area(cell) > _cell_area(prev):
            by_logic[key] = dict(cell)

    kept = list(by_logic.values())
    # 物理 IoU 过高且逻辑不同：丢较小者
    drop: set = set()
    for i in range(len(kept)):
        if i in drop:
            continue
        bi = _cell_bbox(kept[i])
        ai = _cell_area(kept[i])
        for j in range(i + 1, len(kept)):
            if j in drop:
                continue
            bj = _cell_bbox(kept[j])
            # intersection
            ix1 = max(bi[0], bj[0])
            iy1 = max(bi[1], bj[1])
            ix2 = min(bi[2], bj[2])
            iy2 = min(bi[3], bj[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            aj = _cell_area(kept[j])
            union = ai + aj - inter
            if union <= 0:
                continue
            iou = inter / union
            if iou < 0.85:
                continue
            # 丢面积小的
            if ai >= aj:
                drop.add(j)
            else:
                drop.add(i)
                break

    # 包含抑制：大格内若有 ≥2 个子格（各自 ≥90% 面积落入），且子格合计覆盖大格
    # 大部分面积，则丢掉冗余父格（保留更细划分）。
    # 必须先于空碎片剪除：否则丢掉空子格后可能凑不够 2 个孩子，父格会漏网。
    survivors = [c for k, c in enumerate(kept) if k not in drop]
    if len(survivors) >= 3:
        bboxes = [_cell_bbox(c) for c in survivors]
        areas = [_cell_area(c) for c in survivors]
        drop2: set = set()
        for i, (bi, ai) in enumerate(zip(bboxes, areas)):
            if ai <= 0 or i in drop2:
                continue
            children: List[int] = []
            child_area_sum = 0.0
            for j, (bj, aj) in enumerate(zip(bboxes, areas)):
                if i == j or j in drop2 or aj <= 0:
                    continue
                # j 落入 i 的比例
                ix1 = max(bi[0], bj[0])
                iy1 = max(bi[1], bj[1])
                ix2 = min(bi[2], bj[2])
                iy2 = min(bi[3], bj[3])
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter / aj >= 0.90 and aj < ai * 0.95:
                    children.append(j)
                    child_area_sum += aj
            if len(children) >= 2 and child_area_sum >= 0.55 * ai:
                drop2.add(i)
                logger.info(
                    "TSR 容器格抑制: drop parent area=%.0f children=%d cover=%.2f",
                    ai,
                    len(children),
                    child_area_sum / ai,
                )
        survivors = [c for k, c in enumerate(survivors) if k not in drop2]
    return _drop_contained_empty_fragments(survivors)


def _drop_contained_empty_fragments(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """丢弃逻辑上被另一格严格包含、物理上也几乎完全落入、且无实质文本的碎片格。

    TSR 偶发会在合并格角落吐出微型空格；IoU 很低，容器抑制又要求 ≥2 个子格，
    两条旧规则都接不住。有字的细划分（子表头）不删。
    """
    if len(cells) < 2:
        return cells

    drop: set = set()
    bboxes = [_cell_bbox(c) for c in cells]
    areas = [_cell_area(c) for c in cells]
    for j, child in enumerate(cells):
        if j in drop:
            continue
        if not _looks_like_ocr_fragment(str(child.get("text") or "")) :
            continue
        aj = areas[j]
        if aj <= 0:
            drop.add(j)
            continue
        for i, parent in enumerate(cells):
            if i == j or i in drop:
                continue
            if not _logic_strict_subset(child, parent):
                continue
            if _bbox_inter_area(bboxes[j], bboxes[i]) / aj < 0.90:
                continue
            drop.add(j)
            logger.info(
                "TSR 包含碎片抑制: drop child logic=%s area=%.0f inside parent %s",
                _logic_key(child),
                aj,
                _logic_key(parent),
            )
            break
    if not drop:
        return cells
    return [c for k, c in enumerate(cells) if k not in drop]


def _texts_in_cell(
    cell: Dict[str, Any], boxes: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    cx1, cy1, cx2, cy2 = _cell_bbox(cell)
    out = []
    for tb in boxes:
        x1, y1, x2, y2 = _tb_bbox(tb)
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if cx1 <= mx <= cx2 and cy1 <= my <= cy2:
            out.append(tb)
    return out


def _looks_like_ocr_fragment(text: str) -> bool:
    """短碎片 / 孤立数字 / 单字截断，常见于假列吃到邻格尾巴。"""
    t = (text or "").strip()
    if not t:
        return True
    compact = re.sub(r"\s+", "", t)
    
    # 【新增】如果完全匹配合法的字母代号（如 AA, A1, B+），则绝不是碎片
    if _LETTER_DATA_RE.fullmatch(compact):
        return False
        
    if len(compact) <= 2:
        # 两汉字表体词（溶解/不溶/判定等）不是 OCR 碎片
        if re.fullmatch(r"[\u4e00-\u9fff]{2}", compact):
            return False
        return True
    if re.fullmatch(r"[0-9一二三四五六七八九十]+", compact):
        return True
    return False


def _robust_main_size(sizes: Sequence[float]) -> float:
    """
    估计真实主列/主行尺寸。

    双线框会产出大量 ~线宽缝隙（常约 8~20px）。若直接对全部宽度取中位数，
    缝会主导 median，导致剩余缝再也判不成「窄」。只取明显大于缝阈的主尺寸。
    """
    vals = [float(s) for s in sizes if float(s) > 0]
    if not vals:
        return 1.0
    vmax = max(vals)
    # 缝阈：绝对下限 + 相对最大主尺寸的一小部分
    seam_cap = max(20.0, 0.12 * vmax)
    main = [v for v in vals if v >= seam_cap]
    if len(main) >= 2:
        return float(np.median(main))
    if main:
        return float(main[0])
    return float(np.median(vals))


def _is_double_line_seam(size: float, main_size: float) -> bool:
    """双线框缝：远小于主尺寸（或绝对像素极窄）。"""
    if main_size <= 1e-6:
        return False
    return float(size) < max(18.0, 0.10 * float(main_size))


def merge_ghost_columns(
    cells: List[Dict[str, Any]],
    boxes: Sequence[Dict[str, Any]],
    *,
    min_width_ratio: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    几乎无文本落入且物理宽度异常窄的逻辑列 → 合并到右侧邻列。
    窄列若仅有短碎片 OCR（而邻列有实质文本）也视为幽灵列。
    双线框缝（极窄列）即使误落入 OCR 中心也并入邻列。
    """
    if not cells:
        return cells
    row_seps, col_seps = _derive_seps(cells)
    if len(col_seps) < 3:
        return cells

    n_cols = len(col_seps) - 1
    widths = [col_seps[i + 1] - col_seps[i] for i in range(n_cols)]
    median_w = _robust_main_size(widths)

    # 每列文本命中数与样本文本
    hits = [0] * n_cols
    col_texts: List[List[str]] = [[] for _ in range(n_cols)]
    for tb in boxes:
        x1, y1, x2, y2 = _tb_bbox(tb)
        mx = (x1 + x2) / 2.0
        for c in range(n_cols):
            if col_seps[c] <= mx < col_seps[c + 1] or (
                c == n_cols - 1 and col_seps[c] <= mx <= col_seps[c + 1]
            ):
                hits[c] += 1
                t = str(tb.get("text") or "").strip()
                if t:
                    col_texts[c].append(t)
                break

    def _neighbor_has_substance(c: int) -> bool:
        for t in (c - 1, c + 1):
            if 0 <= t < n_cols and any(not _looks_like_ocr_fragment(x) for x in col_texts[t]):
                return True
        return False

    # 标记幽灵列：双线缝优先（避免缝内误落入的数字被当成序号列保留）
    ghost = []
    for c in range(n_cols):
        # 双线框缝：无条件并入（OCR 中心落入缝隙属误命中）
        if _is_double_line_seam(widths[c], median_w):
            ghost.append(True)
            continue
        # 整列多为行序号：保留（与 html drop_evidenceless 一致）
        if is_index_column(col_texts[c]):
            ghost.append(False)
            continue
        narrow = widths[c] < min_width_ratio * median_w
        if not narrow:
            ghost.append(False)
            continue
        if hits[c] == 0:
            ghost.append(True)
            continue
        only_frag = bool(col_texts[c]) and all(
            _looks_like_ocr_fragment(t) for t in col_texts[c]
        )
        ghost.append(bool(only_frag and _neighbor_has_substance(c)))
    if not any(ghost):
        return cells

    # 映射：幽灵列并入右侧非幽灵列；末列幽灵并入左侧
    remap = list(range(n_cols))
    for c in range(n_cols):
        if not ghost[c]:
            continue
        target = None
        for t in range(c + 1, n_cols):
            if not ghost[t]:
                target = t
                break
        if target is None:
            for t in range(c - 1, -1, -1):
                if not ghost[t]:
                    target = t
                    break
        if target is not None:
            remap[c] = target

    # 压缩列号
    kept_old = sorted({remap[c] for c in range(n_cols) if not ghost[c] or remap[c] != c})
    # 更清晰：最终保留的物理列 = 非幽灵列
    survivors = [c for c in range(n_cols) if not ghost[c]]
    if not survivors:
        return cells
    old_to_new = {old: new for new, old in enumerate(survivors)}

    def map_col(c: int) -> int:
        r = remap[c]
        # 若 remap 指向幽灵，再跟一次
        while ghost[r] and remap[r] != r:
            r = remap[r]
            if r == remap[r]:
                break
        if r in old_to_new:
            return old_to_new[r]
        # fallback nearest survivor
        return min(old_to_new.values())

    def resolve_old(c: int) -> int:
        """幽灵列跟随 remap 到最终非幽灵旧列号。"""
        r = remap[c]
        guard = 0
        while ghost[r] and remap[r] != r and guard < n_cols + 2:
            r = remap[r]
            guard += 1
        return r

    # 每个幸存列吸收「remap 到它」的幽灵缝物理区间，避免缝内 OCR 落空
    surv_left: Dict[int, float] = {}
    surv_right: Dict[int, float] = {}
    for s in survivors:
        covered = [s] + [g for g in range(n_cols) if ghost[g] and resolve_old(g) == s]
        surv_left[s] = float(col_seps[min(covered)])
        surv_right[s] = float(col_seps[max(covered) + 1])

    out: List[Dict[str, Any]] = []
    for cell in cells:
        nc = dict(cell)
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        # 只按本格覆盖的「非幽灵列」重映射。若把格内幽灵列跟随并入右侧邻列，
        # 会把左侧 colspan（如 P100「分散液」c1-3）错误扩进「组成」列，
        # 引发逻辑重叠 → light 误升级 aggressive，二级表头被压扁。
        own_survivors = [i for i in range(cs, ce + 1) if not ghost[i]]
        if own_survivors:
            new_cs = min(old_to_new[i] for i in own_survivors)
            new_ce = max(old_to_new[i] for i in own_survivors)
        else:
            new_cs = min(map_col(i) for i in range(cs, ce + 1))
            new_ce = max(map_col(i) for i in range(cs, ce + 1))
        nc["col_start"] = new_cs
        nc["col_end"] = new_ce
        _refresh_spans(nc)
        left_old = survivors[new_cs]
        right_old = survivors[new_ce]
        left = surv_left[left_old]
        right = surv_right[right_old]
        x1, y1, x2, y2 = _cell_bbox(cell)
        nc["polygon"] = _rebuild_polygon(left, y1, right, y2)
        nc["x_key"] = left
        nc["y_key"] = y1
        out.append(nc)

    logger.info(
        "TSR 幽灵列合并: %d -> %d cols (ghost=%s)",
        n_cols,
        len(survivors),
        [i for i, g in enumerate(ghost) if g],
    )
    return out


def merge_ghost_rows(
    cells: List[Dict[str, Any]],
    boxes: Sequence[Dict[str, Any]],
    *,
    min_height_ratio: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    与 merge_ghost_columns 对称：双线框产生的极矮缝行 / 无文本窄行并入邻行。

    只改逻辑行号与多边形 y 范围，不改列。用于防止缝行把表体拉成 rowspan 碎格、
    以及 split_subtables 因缝隙 Y 间距误拆成多张表。
    """
    if not cells:
        return cells
    row_seps, col_seps = _derive_seps(cells)
    if len(row_seps) < 3:
        return cells

    n_rows = len(row_seps) - 1
    heights = [row_seps[i + 1] - row_seps[i] for i in range(n_rows)]
    median_h = _robust_main_size(heights)

    hits = [0] * n_rows
    row_texts: List[List[str]] = [[] for _ in range(n_rows)]
    for tb in boxes:
        x1, y1, x2, y2 = _tb_bbox(tb)
        my = (y1 + y2) / 2.0
        for r in range(n_rows):
            if row_seps[r] <= my < row_seps[r + 1] or (
                r == n_rows - 1 and row_seps[r] <= my <= row_seps[r + 1]
            ):
                hits[r] += 1
                t = str(tb.get("text") or "").strip()
                if t:
                    row_texts[r].append(t)
                break

    def _neighbor_has_substance(r: int) -> bool:
        for t in (r - 1, r + 1):
            if 0 <= t < n_rows and any(
                not _looks_like_ocr_fragment(x) for x in row_texts[t]
            ):
                return True
        return False

    ghost = []
    for r in range(n_rows):
        if _is_double_line_seam(heights[r], median_h):
            ghost.append(True)
            continue
        short = heights[r] < min_height_ratio * median_h
        if not short:
            ghost.append(False)
            continue
        if hits[r] == 0:
            ghost.append(True)
            continue
        only_frag = bool(row_texts[r]) and all(
            _looks_like_ocr_fragment(t) for t in row_texts[r]
        )
        ghost.append(bool(only_frag and _neighbor_has_substance(r)))
    if not any(ghost):
        return cells

    remap = list(range(n_rows))
    for r in range(n_rows):
        if not ghost[r]:
            continue
        target = None
        # 优先并入下方非幽灵行（表体语义更稳）；否则向上
        for t in range(r + 1, n_rows):
            if not ghost[t]:
                target = t
                break
        if target is None:
            for t in range(r - 1, -1, -1):
                if not ghost[t]:
                    target = t
                    break
        if target is not None:
            remap[r] = target

    survivors = [r for r in range(n_rows) if not ghost[r]]
    if not survivors:
        return cells
    old_to_new = {old: new for new, old in enumerate(survivors)}

    def map_row(r: int) -> int:
        cur = remap[r]
        while ghost[cur] and remap[cur] != cur:
            cur = remap[cur]
            if cur == remap[cur]:
                break
        if cur in old_to_new:
            return old_to_new[cur]
        return min(old_to_new.values())

    def resolve_old(r: int) -> int:
        cur = remap[r]
        guard = 0
        while ghost[cur] and remap[cur] != cur and guard < n_rows + 2:
            cur = remap[cur]
            guard += 1
        return cur

    surv_top: Dict[int, float] = {}
    surv_bot: Dict[int, float] = {}
    for s in survivors:
        covered = [s] + [g for g in range(n_rows) if ghost[g] and resolve_old(g) == s]
        surv_top[s] = float(row_seps[min(covered)])
        surv_bot[s] = float(row_seps[max(covered) + 1])

    out: List[Dict[str, Any]] = []
    for cell in cells:
        nc = dict(cell)
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        own_survivors = [i for i in range(rs, re + 1) if not ghost[i]]
        if own_survivors:
            new_rs = min(old_to_new[i] for i in own_survivors)
            new_re = max(old_to_new[i] for i in own_survivors)
        else:
            new_rs = min(map_row(i) for i in range(rs, re + 1))
            new_re = max(map_row(i) for i in range(rs, re + 1))
        nc["row_start"] = new_rs
        nc["row_end"] = new_re
        _refresh_spans(nc)
        top_old = survivors[new_rs]
        bot_old = survivors[new_re]
        top = surv_top[top_old]
        bot = surv_bot[bot_old]
        x1, y1, x2, y2 = _cell_bbox(cell)
        nc["polygon"] = _rebuild_polygon(x1, top, x2, bot)
        nc["x_key"] = x1
        nc["y_key"] = top
        out.append(nc)

    logger.info(
        "TSR 幽灵行合并: %d -> %d rows (ghost=%s)",
        n_rows,
        len(survivors),
        [i for i, g in enumerate(ghost) if g],
    )
    return out


def unmerge_bad_rowspans(
    cells: List[Dict[str, Any]],
    boxes: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """大 rowspan 内若每行恰好 1 个非 CJK 文本框 → 拆成原子行。"""
    if not cells or not boxes:
        return cells
    row_seps, col_seps = _derive_seps(cells)
    if len(row_seps) < 3:
        return cells

    def row_index(cy: float) -> int:
        for r in range(len(row_seps) - 1):
            if row_seps[r] <= cy < row_seps[r + 1]:
                return r
        if row_seps and cy >= row_seps[-1]:
            return len(row_seps) - 2
        return -1

    out: List[Dict[str, Any]] = []
    changed = False
    for cell in cells:
        row_span = int(cell.get("row_span") or 1)
        col_span = int(cell.get("col_span") or 1)
        if col_span != 1 or row_span < 2:
            out.append(cell)
            continue

        contained = _texts_in_cell(cell, boxes)
        row_start = int(cell["row_start"])
        row_end = int(cell["row_end"])
        if len(contained) != row_span:
            out.append(cell)
            continue

        by_row: Dict[int, List[Dict[str, Any]]] = {
            r: [] for r in range(row_start, row_end + 1)
        }
        ok = True
        for tb in contained:
            x1, y1, x2, y2 = _tb_bbox(tb)
            my = (y1 + y2) / 2.0
            ri = row_index(my)
            if ri < row_start or ri > row_end:
                ok = False
                break
            by_row[ri].append(tb)
        if not ok or any(len(v) != 1 for v in by_row.values()):
            out.append(cell)
            continue

        texts = [
            str(by_row[r][0].get("text") or "").strip()
            for r in range(row_start, row_end + 1)
        ]
        all_same = all(t == texts[0] for t in texts)
        none_cjk = all(not _has_cjk(t) for t in texts)
        # 实施例N / 合成例N / iii-N / OXL-* / 序号：CJK 不同也不应纵向粘连
        if not (all_same or none_cjk or are_independent_row_labels(texts)):
            out.append(cell)
            continue

        col_start = int(cell["col_start"])
        for r in range(row_start, row_end + 1):
            if r >= len(row_seps) - 1 or col_start >= len(col_seps) - 1:
                continue
            x1 = float(col_seps[col_start])
            x2 = float(col_seps[col_start + 1])
            y1 = float(row_seps[r])
            y2 = float(row_seps[r + 1])
            poly = _rebuild_polygon(x1, y1, x2, y2)
            out.append(
                {
                    "polygon": poly,
                    "x_key": x1,
                    "y_key": y1,
                    "row_start": r,
                    "row_end": r,
                    "col_start": col_start,
                    "col_end": col_start,
                    "row_span": 1,
                    "col_span": 1,
                    "texts": [],
                    "text": "",
                }
            )
        changed = True

    if changed:
        logger.info("TSR 拆错误纵向合并完成")
    return out


def split_bad_colspans(
    cells: List[Dict[str, Any]],
    boxes: Sequence[Dict[str, Any]],
    *,
    min_support_rows: int = 2,
) -> List[Dict[str, Any]]:
    """
    宽格内稳定双簇文本（如 B1|4g）→ 在中位间隙处拆成两列，并重映射全局列号。
    """
    if not cells or not boxes:
        return cells

    row_seps, col_seps = _derive_seps(cells)
    if len(col_seps) < 2:
        return cells

    # 找需要拆的 (col_start,col_end) 宽格候选：col_span==1 但物理宽且双簇
    # 实际对每个原子列区间检测双簇
    n_cols = len(col_seps) - 1
    splits: List[Tuple[int, float]] = []  # (col_idx, split_x) insert after col

    for c in range(n_cols):
        left, right = col_seps[c], col_seps[c + 1]
        width = right - left
        if width < 20:
            continue
        # 收集落入该列的文本中心 x，按行分组
        by_row: Dict[int, List[float]] = defaultdict(list)
        for tb in boxes:
            x1, y1, x2, y2 = _tb_bbox(tb)
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if not (left <= mx < right or (c == n_cols - 1 and left <= mx <= right)):
                continue
            # 框不能几乎占满整列
            if (x2 - x1) > 0.7 * width:
                continue
            ri = 0
            for r in range(len(row_seps) - 1):
                if row_seps[r] <= my < row_seps[r + 1]:
                    ri = r
                    break
            by_row[ri].append(mx)

        support = 0
        split_xs: List[float] = []
        for xs in by_row.values():
            if len(xs) < 2:
                continue
            xs = sorted(xs)
            # 最大间隙
            gaps = [(xs[i + 1] - xs[i], (xs[i] + xs[i + 1]) / 2.0) for i in range(len(xs) - 1)]
            gaps.sort(reverse=True)
            best_gap, mid = gaps[0]
            if best_gap < 0.15 * width:
                continue
            # 左右都有点
            left_n = sum(1 for x in xs if x < mid)
            right_n = sum(1 for x in xs if x >= mid)
            if left_n < 1 or right_n < 1:
                continue
            support += 1
            split_xs.append(mid)

        # 双数值列（酸当量|双键当量）即使仅 2 行支持也切；
        # 表头成对短标签（種類|添加量）仅 1 行双簇也可切。
        need = min_support_rows
        dual_numeric = 0
        for xs in by_row.values():
            if len(xs) >= 2:
                dual_numeric += 1
        if dual_numeric >= 2:
            need = min(need, 2)

        header_pair = False
        for ri, xs in by_row.items():
            if len(xs) < 2:
                continue
            # 顶部两行视为表头带
            if ri > 1:
                continue
            y0 = row_seps[ri] if ri < len(row_seps) else 0.0
            y1 = row_seps[ri + 1] if ri + 1 < len(row_seps) else y0 + 1.0
            short_n = 0
            for tb in boxes:
                bx1, by1, bx2, by2 = _tb_bbox(tb)
                mx, my = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                if not (y0 <= my < y1):
                    continue
                if not (left <= mx < right):
                    continue
                t = str(tb.get("text") or "").strip()
                compact = "".join(t.split())
                if 1 <= len(compact) <= 6:
                    short_n += 1
            if short_n >= 2:
                header_pair = True
                break
        if header_pair:
            need = 1

        if support >= need and split_xs:
            splits.append((c, float(np.median(split_xs))))

    if not splits:
        return cells

    # 从右往左插入，避免索引错乱
    splits.sort(key=lambda t: t[0], reverse=True)
    new_col_seps = list(col_seps)
    # 记录旧列 -> 新列范围映射
    # 先构建完整新 seps
    insert_at: Dict[int, float] = {c: x for c, x in splits}
    rebuilt = [new_col_seps[0]]
    old_col_to_new_range: Dict[int, Tuple[int, int]] = {}
    new_idx = 0
    for c in range(n_cols):
        if c in insert_at:
            sx = insert_at[c]
            # 确保在区间内
            lo, hi = new_col_seps[c], new_col_seps[c + 1]
            sx = min(max(sx, lo + 2), hi - 2)
            rebuilt.append(sx)
            old_col_to_new_range[c] = (new_idx, new_idx + 1)
            new_idx += 2
            rebuilt.append(hi)
        else:
            rebuilt.append(new_col_seps[c + 1])
            old_col_to_new_range[c] = (new_idx, new_idx)
            new_idx += 1

    out: List[Dict[str, Any]] = []
    for cell in cells:
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        # 若原子列被拆且 cell 是单列，按文本决定落左还是拆成两格
        if cs == ce and cs in insert_at:
            contained = _texts_in_cell(cell, boxes)
            sx = insert_at[cs]
            left_tbs = []
            right_tbs = []
            for tb in contained:
                mx = (_tb_bbox(tb)[0] + _tb_bbox(tb)[2]) / 2.0
                (left_tbs if mx < sx else right_tbs).append(tb)
            new_cs, new_ce = old_col_to_new_range[cs]
            y1 = float(row_seps[rs]) if rs < len(row_seps) else _cell_bbox(cell)[1]
            y2 = float(row_seps[re + 1]) if re + 1 < len(row_seps) else _cell_bbox(cell)[3]
            if left_tbs and right_tbs and new_cs != new_ce:
                # 拆两格
                for ncol, tbs in ((new_cs, left_tbs), (new_ce, right_tbs)):
                    x1 = float(rebuilt[ncol])
                    x2 = float(rebuilt[ncol + 1])
                    poly = _rebuild_polygon(x1, y1, x2, y2)
                    out.append(
                        {
                            "polygon": poly,
                            "x_key": x1,
                            "y_key": y1,
                            "row_start": rs,
                            "row_end": re,
                            "col_start": ncol,
                            "col_end": ncol,
                            "row_span": re - rs + 1,
                            "col_span": 1,
                            "texts": [],
                            "text": "",
                        }
                    )
                continue
            # 只落一侧或无文本：整格映射到覆盖范围
            nc = dict(cell)
            nc["col_start"] = new_cs
            nc["col_end"] = new_ce
            _refresh_spans(nc)
            x1 = float(rebuilt[new_cs])
            x2 = float(rebuilt[new_ce + 1])
            nc["polygon"] = _rebuild_polygon(x1, y1, x2, y2)
            nc["x_key"] = x1
            out.append(nc)
            continue

        # 多列/其它：映射起止
        new_starts = [old_col_to_new_range[i][0] for i in range(cs, ce + 1)]
        new_ends = [old_col_to_new_range[i][1] for i in range(cs, ce + 1)]
        nc = dict(cell)
        nc["col_start"] = min(new_starts)
        nc["col_end"] = max(new_ends)
        _refresh_spans(nc)
        x1 = float(rebuilt[nc["col_start"]])
        x2 = float(rebuilt[nc["col_end"] + 1])
        _, y1, _, y2 = _cell_bbox(cell)
        nc["polygon"] = _rebuild_polygon(x1, y1, x2, y2)
        nc["x_key"] = x1
        out.append(nc)

    logger.info("TSR 双簇补列: inserted %d splits", len(splits))
    return out


# 混合表：竖线明显、横线稀疏（专利实验表常见「有竖无线」）
_HYBRID_V_MIN = 0.0035
_HYBRID_H_MAX = 0.0045
# 真正偏「全有线」时要求横线也够，不能只靠竖线把 dens 抬高
_WIRED_H_MIN = 0.0040
_WIRED_V_MIN = 0.0030
# 欠切：格子相对 OCR 框过少（大格假高覆盖时仍能触发纠偏）
_UNDERSEG_CELL_TO_BOX = 0.22
_UNDERSEG_MIN_COLS_VS_BOXES = 40


def line_density_axes(image: np.ndarray) -> Tuple[float, float]:
    """返回 (h_dens, v_dens)：横/竖形态学开运算像素占比。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    h, w = binary.shape[:2]
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 40, 10), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 40, 10)))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    area = float(h * w + 1)
    return (
        float(np.count_nonzero(h_lines)) / area,
        float(np.count_nonzero(v_lines)) / area,
    )


def line_density_score(image: np.ndarray) -> float:
    """粗略线密度（H+V 合计）；兼容旧调用。"""
    h_dens, v_dens = line_density_axes(image)
    return h_dens + v_dens


def is_hybrid_line_table(h_dens: float, v_dens: float) -> bool:
    """竖线多、横线少 → 混合表（有线 unet 易欠切列）。"""
    if v_dens >= _HYBRID_V_MIN and h_dens <= _HYBRID_H_MAX:
        return True
    # 竖线显著多于横线（形态学横线常被文字抬高，用倍率兜底）
    if v_dens >= _HYBRID_V_MIN and v_dens >= h_dens * 1.35:
        return True
    return False


def looks_fully_wired(h_dens: float, v_dens: float) -> bool:
    """横竖线都够密，才适合偏置到 wired。"""
    return h_dens >= _WIRED_H_MIN and v_dens >= _WIRED_V_MIN


def cell_grid_stats(cells: Sequence[Dict[str, Any]]) -> Tuple[int, int, int]:
    """返回 (n_cols, n_rows, n_cells)。"""
    if not cells:
        return 0, 0, 0
    row_seps, col_seps = _derive_seps(list(cells))
    n_cols = max(0, len(col_seps) - 1)
    n_rows = max(0, len(row_seps) - 1)
    return n_cols, n_rows, len(cells)


def looks_undersegmented(
    cells: Sequence[Dict[str, Any]],
    text_boxes: Sequence[Dict[str, Any]],
) -> bool:
    """格子相对 OCR 框过少，或中位格过宽 → 疑似欠切。"""
    if not cells:
        return True
    n_boxes = len(text_boxes)
    if not n_boxes:
        return False
    if len(cells) < max(8, int(n_boxes * _UNDERSEG_CELL_TO_BOX)):
        return True
    n_cols, _n_rows, _n = cell_grid_stats(cells)
    if n_cols > 0 and n_cols < max(4, n_boxes // _UNDERSEG_MIN_COLS_VS_BOXES):
        return True
    widths = []
    for c in cells:
        x1, _y1, x2, _y2 = _cell_bbox(c)
        widths.append(max(0.0, x2 - x1))
    if widths and n_cols <= 14:
        xs1 = [_cell_bbox(c)[0] for c in cells]
        xs2 = [_cell_bbox(c)[2] for c in cells]
        table_w = max(xs2) - min(xs1) + 1e-6
        if float(np.median(widths)) > 0.16 * table_w:
            return True
    return False


def looks_oversegmented(
    cells: Sequence[Dict[str, Any]],
    text_boxes: Sequence[Dict[str, Any]],
) -> bool:
    """列/行/格数相对 OCR 过多 → 过切（后处理易塌缩）。

    注意：有线完整表常见「行数 ≈ 文本框数/4~6」；阈值须高于该量级，
    避免把正常的 30+ 行数据表误判为过切（进而拒绝框线回退）。
    """
    if not cells or not text_boxes:
        return False
    n_boxes = len(text_boxes)
    n_cols, n_rows, n = cell_grid_stats(cells)
    if n_cols >= 32:
        return True
    # 行过切：逻辑行显著多于「每行约 3 个 OCR 框」的上限，且至少 40 行
    if n_rows >= 40 and n_rows > max(24, n_boxes // 3):
        return True
    if n > n_boxes * 1.05 and (n_cols >= 28 or n_rows >= 40):
        return True
    return False


def structure_quality_score(
    cells: Sequence[Dict[str, Any]],
    text_boxes: Sequence[Dict[str, Any]],
) -> float:
    """
    结构可用性粗分（0~1）：兼顾覆盖率、格数相对 OCR、列/行是否离谱。
    用于 wired↔lineless 选型，避免「列最多但过切」赢过可后处理的网格。
    """
    if not cells:
        return 0.0
    boxes = list(text_boxes or [])
    n_boxes = max(len(boxes), 1)
    n_cols, n_rows, n = cell_grid_stats(cells)
    cov = coverage_score(list(cells), boxes) if boxes else 0.0

    ratio = float(n) / float(n_boxes)
    # 峰值约 0.45~0.7（有空格/合并格的表）
    cell_term = float(np.exp(-((ratio - 0.55) ** 2) / (2 * 0.28**2)))

    # 专利表常见 8~20 列
    col_term = float(np.exp(-((float(n_cols) - 14.0) ** 2) / (2 * 7.0**2)))
    row_target = max(10.0, float(n_boxes) / 16.0)
    row_term = float(
        np.exp(-((float(n_rows) - row_target) ** 2) / (2 * (row_target * 0.7) ** 2))
    )

    score = 0.28 * cov + 0.32 * cell_term + 0.25 * col_term + 0.15 * row_term
    if looks_undersegmented(cells, boxes):
        score *= 0.72
    if looks_oversegmented(cells, boxes):
        score *= 0.55
    return float(max(0.0, min(1.0, score)))



def coverage_score(
    cells: List[Dict[str, Any]], boxes: Sequence[Dict[str, Any]]
) -> float:
    """OCR 框中心落入任一 cell 的比例。"""
    if not boxes:
        return 0.0
    if not cells:
        return 0.0
    hit = 0
    for tb in boxes:
        x1, y1, x2, y2 = _tb_bbox(tb)
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        for cell in cells:
            cx1, cy1, cx2, cy2 = _cell_bbox(cell)
            if cx1 <= mx <= cx2 and cy1 <= my <= cy2:
                hit += 1
                break
    return hit / max(len(boxes), 1)


def split_underspanned_rows(
    cells: List[Dict[str, Any]],
    boxes: Sequence[Dict[str, Any]],
    *,
    min_support_cols: int = 3,
    gap_ratio: float = 0.45,
) -> List[Dict[str, Any]]:
    """
    少行表子行切分：对每个原子行，若多个列独立按 y 聚出相同 k 个带，则插入 row_seps。

    对称于 split_bad_colspans。用于行间无框线、TSR 把多数据行合成一行的场景。
    """
    if not cells or not boxes:
        return cells

    row_seps, col_seps = _derive_seps(cells)
    if len(row_seps) < 2 or len(col_seps) < 3:
        return cells

    n_rows = len(row_seps) - 1
    n_cols = len(col_seps) - 1
    # 用文本框高度估计行内间隙阈值（勿用被合并的超高 row_sep，否则永远切不开）
    box_heights = []
    for tb in boxes:
        _x1, y1, _x2, y2 = _tb_bbox(tb)
        bh = y2 - y1
        if 2 <= bh <= 80:
            box_heights.append(bh)
    median_box_h = float(np.median(box_heights)) if box_heights else 14.0
    base_gap_thresh = max(3.0, gap_ratio * median_box_h)

    # 表头带识别：从顶部开始连续若干“非数字/长标签行”，跳过子行切分
    # （避免把表头折行当作“子行”拆开）
    def _row_is_header(r: int) -> bool:
        y0, y1 = row_seps[r], row_seps[r + 1]
        row_texts: List[str] = []
        for tb in boxes:
            _x1, by1, _x2, by2 = _tb_bbox(tb)
            my = (by1 + by2) / 2.0
            if not (y0 <= my < y1 or (r == n_rows - 1 and y0 <= my <= y1)):
                continue
            t = str(tb.get("text") or "").strip()
            if t:
                row_texts.append(t)
        if not row_texts:
            return False
        has_long = any(len(t) > 12 for t in row_texts)
        digit_n = sum(1 for t in row_texts if any(ch.isdigit() for ch in t))
        digit_ratio = digit_n / max(len(row_texts), 1)
        # 长标签 + 数字少（或完全没有）更像表头
        return has_long and digit_ratio < 0.45

    header_until = 0
    for r in range(n_rows):
        if _row_is_header(r):
            header_until = r + 1
        else:
            break

    # splits: row_idx -> list of split_y ascending
    row_splits: Dict[int, List[float]] = {}

    for r in range(n_rows):
        if r < header_until:
            continue
        y0, y1 = row_seps[r], row_seps[r + 1]
        row_h = y1 - y0
        # 行高至少能放下约 2.5 行文字才考虑切
        # 注意：表头带已跳过，剩余行更像数据行，此时 gap 阈值才有意义
        if row_h < base_gap_thresh * 2.5 + median_box_h:
            continue

        # 每列的 y 中心列表
        col_ys: Dict[int, List[float]] = defaultdict(list)
        for tb in boxes:
            bx1, by1, bx2, by2 = _tb_bbox(tb)
            mx, my = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            if not (y0 <= my < y1 or (r == n_rows - 1 and y0 <= my <= y1)):
                continue
            # 框几乎占满整行高度 → 不参与（跨行标题 / 竖排标签）
            if (by2 - by1) > 0.55 * row_h:
                continue
            for c in range(n_cols):
                if col_seps[c] <= mx < col_seps[c + 1] or (
                    c == n_cols - 1 and col_seps[c] <= mx <= col_seps[c + 1]
                ):
                    col_ys[c].append(my)
                    break

        # 对每列做 1D 间隙聚类：
        # gap_thresh 使用“该行内相邻文本中心的间隙分布”推导出来的更严格阈值，
        # 只有明显大于折行/换行尺度的间隙才会被当成“子行切分”。
        diffs: List[float] = []
        for _c, ys in col_ys.items():
            if len(ys) < 2:
                continue
            ys_sorted = sorted(ys)
            for k in range(len(ys_sorted) - 1):
                d = ys_sorted[k + 1] - ys_sorted[k]
                # 只统计更可能对应“同一格内换行”的小间隙
                if d < 2.5 * median_box_h:
                    diffs.append(d)
        local_gap_thresh = base_gap_thresh
        if diffs:
            q75 = float(np.quantile(diffs, 0.75))
            local_gap_thresh = max(base_gap_thresh, q75 * 1.8)

        def cluster_ys(ys: List[float]) -> List[Tuple[float, float]]:
            if len(ys) < 2:
                return []
            ys = sorted(ys)
            bands: List[List[float]] = [[ys[0]]]
            for y in ys[1:]:
                if y - bands[-1][-1] > local_gap_thresh:
                    bands.append([y])
                else:
                    bands[-1].append(y)
            if len(bands) < 2:
                return []
            return [(min(b), max(b)) for b in bands]

        col_bands: Dict[int, List[Tuple[float, float]]] = {}
        for c, ys in col_ys.items():
            bands = cluster_ys(ys)
            if bands:
                col_bands[c] = bands

        if len(col_bands) < min_support_cols:
            continue

        # 行内文本偏长 → 多为折行表头，跳过（数据行多为短码/数值）
        row_texts = []
        for tb in boxes:
            _x1, y1, _x2, y2 = _tb_bbox(tb)
            my = (y1 + y2) / 2.0
            if y0 <= my < y1 or (r == n_rows - 1 and y0 <= my <= y1):
                row_texts.append(str(tb.get("text") or "").strip())
        if row_texts:
            short_n = sum(1 for t in row_texts if len(t) <= 12)
            if short_n / len(row_texts) < 0.55:
                continue

        # 多数列的 band 数一致；两行合并也允许切开，但仍要求多列共同支持
        k_counts: Dict[int, int] = defaultdict(int)
        for bands in col_bands.values():
            k_counts[len(bands)] += 1
        k_best, support = max(k_counts.items(), key=lambda kv: kv[1])
        need_support = max(min_support_cols, min(4, len(col_bands)))
        if k_best < 2 or support < need_support:
            continue

        # 行必须明显偏高：至少能放下 k_best 行文字并留出间隙
        if row_h < max(2.2 * median_box_h, k_best * 1.4 * median_box_h):
            continue

        # 每带高度应接近「单行文字」尺度
        band_h = row_h / float(k_best)
        if band_h < 0.8 * median_box_h or band_h > 2.8 * median_box_h:
            continue

        supporting = {
            c: bands
            for c, bands in col_bands.items()
            if len(bands) == k_best
        }
        if len(supporting) < max(need_support, int(np.ceil(0.6 * len(col_bands)))):
            continue

        # 带间隙中位作为切点
        split_ys: List[float] = []
        for bi in range(k_best - 1):
            gaps = []
            for bands in supporting.values():
                mid = (bands[bi][1] + bands[bi + 1][0]) / 2.0
                gaps.append(mid)
            split_ys.append(float(np.median(gaps)))

        # 切点必须落在行内且互相分开
        valid = []
        prev = y0
        min_sep = max(2.0, 0.5 * median_box_h)
        for sy in split_ys:
            if prev + min_sep < sy < y1 - min_sep:
                valid.append(sy)
                prev = sy
        if len(valid) < 1:
            continue
        # 允许略少于 k_best-1（边缘带并掉时），但至少切开
        row_splits[r] = valid

    if not row_splits:
        return cells

    # 重建 row_seps 与 old→new 映射
    rebuilt: List[float] = [row_seps[0]]
    old_row_to_new_range: Dict[int, Tuple[int, int]] = {}
    new_idx = 0
    for r in range(n_rows):
        if r in row_splits:
            for sy in row_splits[r]:
                rebuilt.append(sy)
            rebuilt.append(row_seps[r + 1])
            n_new = len(row_splits[r]) + 1
            old_row_to_new_range[r] = (new_idx, new_idx + n_new - 1)
            new_idx += n_new
        else:
            rebuilt.append(row_seps[r + 1])
            old_row_to_new_range[r] = (new_idx, new_idx)
            new_idx += 1

    out: List[Dict[str, Any]] = []
    for cell in cells:
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        # 单行且被拆：按文本 y 落入决定拆成多格或整格映射
        if rs == re and rs in row_splits:
            contained = _texts_in_cell(cell, boxes)
            new_rs, new_re = old_row_to_new_range[rs]
            split_ys = row_splits[rs]
            # 边界：rebuilt[new_rs .. new_re+1]
            bands_idx = list(range(new_rs, new_re + 1))
            buckets: Dict[int, List[Dict[str, Any]]] = {i: [] for i in bands_idx}

            def band_of(my: float) -> int:
                for bi, nrow in enumerate(bands_idx):
                    lo = rebuilt[nrow]
                    hi = rebuilt[nrow + 1]
                    if lo <= my < hi or (bi == len(bands_idx) - 1 and lo <= my <= hi):
                        return nrow
                return bands_idx[0]

            for tb in contained:
                my = (_tb_bbox(tb)[1] + _tb_bbox(tb)[3]) / 2.0
                buckets[band_of(my)].append(tb)

            nonempty_bands = [i for i, tbs in buckets.items() if tbs]
            x1 = float(col_seps[cs]) if cs < len(col_seps) else _cell_bbox(cell)[0]
            x2 = (
                float(col_seps[ce + 1])
                if ce + 1 < len(col_seps)
                else _cell_bbox(cell)[2]
            )
            if len(nonempty_bands) >= 2:
                for nrow in bands_idx:
                    y1 = float(rebuilt[nrow])
                    y2 = float(rebuilt[nrow + 1])
                    poly = _rebuild_polygon(x1, y1, x2, y2)
                    out.append(
                        {
                            "polygon": poly,
                            "x_key": x1,
                            "y_key": y1,
                            "row_start": nrow,
                            "row_end": nrow,
                            "col_start": cs,
                            "col_end": ce,
                            "row_span": 1,
                            "col_span": ce - cs + 1,
                            "texts": [],
                            "text": "",
                        }
                    )
                continue
            # 只落一侧或无文本：映射到覆盖范围
            nc = dict(cell)
            nc["row_start"] = new_rs
            nc["row_end"] = new_re
            _refresh_spans(nc)
            y1 = float(rebuilt[new_rs])
            y2 = float(rebuilt[new_re + 1])
            nc["polygon"] = _rebuild_polygon(x1, y1, x2, y2)
            nc["y_key"] = y1
            out.append(nc)
            continue

        # 多行/其它：映射起止
        new_starts = [old_row_to_new_range[i][0] for i in range(rs, re + 1)]
        new_ends = [old_row_to_new_range[i][1] for i in range(rs, re + 1)]
        nc = dict(cell)
        nc["row_start"] = min(new_starts)
        nc["row_end"] = max(new_ends)
        _refresh_spans(nc)
        _, y1, _, y2 = _cell_bbox(cell)
        y1 = float(rebuilt[nc["row_start"]])
        y2 = float(rebuilt[nc["row_end"] + 1])
        x1 = float(col_seps[cs]) if cs < len(col_seps) else _cell_bbox(cell)[0]
        x2 = (
            float(col_seps[ce + 1])
            if ce + 1 < len(col_seps)
            else _cell_bbox(cell)[2]
        )
        nc["polygon"] = _rebuild_polygon(x1, y1, x2, y2)
        nc["y_key"] = y1
        out.append(nc)

    n_ins = sum(len(v) for v in row_splits.values())
    logger.info(
        "TSR 子行切分: rows=%s inserted %d splits",
        sorted(row_splits.keys()),
        n_ins,
    )
    return out


def _cluster_1d(
    values: List[float],
    *,
    gap_thresh: float,
) -> List[List[int]]:
    """对一维坐标做间隙聚类，返回每簇的原始下标列表。"""
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    clusters: List[List[int]] = [[order[0]]]
    for idx in order[1:]:
        prev = values[clusters[-1][-1]]
        if values[idx] - prev > gap_thresh:
            clusters.append([idx])
        else:
            clusters[-1].append(idx)
    return clusters


def _snap_x_range_to_cols(
    x_lo: float,
    x_hi: float,
    col_seps: List[float],
    *,
    fallback: Tuple[int, int],
) -> Tuple[int, int]:
    """将物理 x 区间吸附到逻辑列 [cs, ce]。"""
    n_cols = len(col_seps) - 1
    if n_cols <= 0:
        return fallback
    mid = 0.5 * (x_lo + x_hi)
    # 优先：区间覆盖的列
    covered: List[int] = []
    for c in range(n_cols):
        left, right = col_seps[c], col_seps[c + 1]
        overlap = min(x_hi, right) - max(x_lo, left)
        if overlap > 0.25 * (right - left):
            covered.append(c)
    if covered:
        return covered[0], covered[-1]
    # 回退：中心所在列
    for c in range(n_cols):
        if col_seps[c] <= mid < col_seps[c + 1] or (
            c == n_cols - 1 and col_seps[c] <= mid <= col_seps[c + 1]
        ):
            return c, c
    return fallback


def _fill_col_spans_among_groups(
    spans: List[Tuple[int, int]],
    *,
    cs: int,
    ce: int,
) -> List[Tuple[int, int]]:
    """
    将若干已吸附的列区间按左→右填满 [cs, ce]，避免子表头之间留下空洞列。
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda t: (t[0], t[1]))
    seeds = [max(cs, min(ce, (a + b) // 2)) for a, b in ordered]
    uniq_seeds: List[int] = []
    for s in seeds:
        if not uniq_seeds or uniq_seeds[-1] != s:
            uniq_seeds.append(s)
    if len(uniq_seeds) == 1:
        return [(cs, ce)]
    boundaries = [cs]
    for i in range(len(uniq_seeds) - 1):
        mid = (uniq_seeds[i] + uniq_seeds[i + 1] + 1) // 2
        boundaries.append(max(boundaries[-1] + 1, min(ce, mid)))
    boundaries.append(ce + 1)
    out: List[Tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i + 1] - 1
        if b >= a:
            out.append((a, b))
    return out


def _body_merge_spans(
    work: List[Dict[str, Any]],
    *,
    after_row: int,
    seg_hi: int,
    cs: int,
    ce: int,
) -> List[Tuple[int, int]]:
    """取身行在 [cs,ce] 上的合并区间，用于表头与身列拓扑对齐。"""
    body_row = None
    for r in range(after_row + 1, seg_hi + 1):
        row_cells = [
            c
            for c in work
            if int(c["row_start"]) == r and int(c["row_end"]) == r
        ]
        if not row_cells:
            continue
        wide_n = sum(
            1 for c in row_cells if int(c["col_end"]) - int(c["col_start"]) + 1 >= 2
        )
        if wide_n >= max(2, len(row_cells) // 2):
            continue
        body_row = r
        break
    if body_row is None:
        return []
    spans: List[Tuple[int, int]] = []
    for c in work:
        if int(c["row_start"]) != body_row:
            continue
        a, b = int(c["col_start"]), int(c["col_end"])
        if b < cs or a > ce:
            continue
        spans.append((max(a, cs), min(b, ce)))
    return sorted(set(spans), key=lambda t: t[0])


def _align_spans_to_body(
    raw_spans: List[Tuple[int, int]],
    body_spans: List[Tuple[int, int]],
    *,
    cs: int,
    ce: int,
) -> List[Tuple[int, int]]:
    """
    将 OCR 簇吸附的列区间对齐到身行合并拓扑。
    若身行某格为 colspan>=2，则对应表头保持合并，避免单标签被拆散后串行。
    """
    if not raw_spans:
        return []
    if not body_spans:
        return _fill_col_spans_among_groups(raw_spans, cs=cs, ce=ce)

    seeds = [max(cs, min(ce, (a + b) // 2)) for a, b in raw_spans]
    # 身行若全是原子列，没有合并先验，改用 OCR 簇填满
    if all(a == b for a, b in body_spans):
        return _fill_col_spans_among_groups(raw_spans, cs=cs, ce=ce)

    hit_spans: List[Tuple[int, int]] = []
    for ba, bb in body_spans:
        if any(ba <= s <= bb for s in seeds):
            hit_spans.append((ba, bb))
    if not hit_spans:
        return _fill_col_spans_among_groups(raw_spans, cs=cs, ce=ce)
    # 已是身行拓扑：直接使用（仅补齐左右到 [cs,ce] 的空隙）
    hit_spans = sorted(hit_spans, key=lambda t: t[0])
    fixed = [list(s) for s in hit_spans]
    fixed[0][0] = cs
    fixed[-1][1] = ce
    out: List[Tuple[int, int]] = []
    for i, (a, b) in enumerate(fixed):
        if i + 1 < len(fixed):
            na = fixed[i + 1][0]
            if b + 1 < na:
                mid = (b + na) // 2
                b = mid
                fixed[i + 1][0] = mid + 1
        out.append((a, b))
    return out


def _make_empty_cell(
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> Dict[str, Any]:
    poly = _rebuild_polygon(x1, y1, x2, y2)
    return {
        "polygon": poly,
        "x_key": x1,
        "y_key": y1,
        "row_start": row_start,
        "row_end": row_end,
        "col_start": col_start,
        "col_end": col_end,
        "row_span": row_end - row_start + 1,
        "col_span": col_end - col_start + 1,
        "texts": [],
        "text": "",
    }


def needs_monomer_header_reconstruct(
    cells: List[Dict[str, Any]],
    boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> bool:
    """存在「单体[…]」宽格且其内叠有多个中文子表头 OCR → 应拆表头（P97）。"""
    if not cells:
        return False
    boxes = list(boxes) if boxes else []
    monomer_re = re.compile(r"单体\s*[\[［]")
    has_monomer = any(
        monomer_re.search(str(c.get("text") or "")) for c in cells
    ) or any(monomer_re.search(str(tb.get("text") or "")) for tb in boxes)
    if not has_monomer:
        return False
    mid_re = re.compile(
        r"(二羧酸|双氨基|封端剂|三官能|四官能|四羧酸|二胺及其|二羟基|有机硅烷)"
    )
    for cell in cells:
        span = int(
            cell.get("col_span")
            or (int(cell["col_end"]) - int(cell["col_start"]) + 1)
        )
        x1, y1, x2, y2 = _cell_bbox(cell)
        # 只要宽或高足够大的格（父表头常 rowspan 盖住子带）
        if span < 2 and (x2 - x1) < 120 and (y2 - y1) < 48:
            continue
        n_mid = 0
        for tb in boxes:
            t = str(tb.get("text") or "")
            if not (mid_re.search(t) or monomer_re.search(t)):
                continue
            bx1, by1, bx2, by2 = _tb_bbox(tb)
            mx, my = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            if x1 - 3 <= mx <= x2 + 3 and y1 - 3 <= my <= y2 + 3:
                n_mid += 1
        if n_mid >= 2:
            return True
    return False


def reconstruct_header_cells(
    cells: List[Dict[str, Any]],
    boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    按子表分段，用表头 OCR 的 Y/X 聚类重建多级表头结构。

    - 相对行号在表头带内的宽格才处理（不再硬编码只处理 row_start==row_end，允许跨行宽表头拆分）
    - Y 向多带 → 拆成多级表头行（如「单体[mol%]」+「三官能/四官能…」）
    - 每带内仅 1 个 X 簇 → 保持合并格（避免单标签被拆散串行）
    - 每带内 ≥2 个 X 簇 → 吸附到身列边界拆成若干格（可保留 colspan>1）
    """
    if not cells:
        return cells
    boxes = list(boxes) if boxes else []
    if not boxes:
        return cells

    from ..utils.segments import find_row_segments

    row_seps, col_seps = _derive_seps(cells)
    if len(col_seps) < 3 or len(row_seps) < 2:
        return cells

    segments = find_row_segments(cells, text_boxes=boxes)
    if not segments:
        min_r = min(int(c["row_start"]) for c in cells)
        max_r = max(int(c["row_end"]) for c in cells)
        segments = [(min_r, max_r)]

    # 文本框高度中位数 → Y/X 聚类间隙
    box_hs = []
    box_ws = []
    for tb in boxes:
        x1, y1, x2, y2 = _tb_bbox(tb)
        h, w = y2 - y1, x2 - x1
        if 2 <= h <= 80:
            box_hs.append(h)
        if 2 <= w <= 400:
            box_ws.append(w)
    median_h = float(np.median(box_hs)) if box_hs else 14.0
    median_w = float(np.median(box_ws)) if box_ws else 40.0
    y_gap = max(6.0, 0.85 * median_h)
    x_gap = max(12.0, 0.55 * median_w)

    # 工作副本
    work = [dict(c) for c in cells]
    total_y_splits = 0
    total_x_splits = 0

    for _pass in range(4):  # 有限轮次，避免死循环
        row_seps, col_seps = _derive_seps(work)
        segments = find_row_segments(work, text_boxes=boxes)
        changed = False

        for seg_lo, seg_hi in segments:
            # 段内无身行则跳过（纯题注段）
            body_rows = [
                r
                for r in range(seg_lo, seg_hi + 1)
                if any(
                    int(c["row_start"]) == r
                    and int(c["row_end"]) == r
                    and int(c.get("col_span") or 1) == 1
                    for c in work
                )
            ]
            # 表头带：段内相对前 2 行；若段很短则最多到身行前
            header_hi = min(seg_lo + 1, seg_hi)
            if body_rows:
                header_hi = min(header_hi, max(body_rows[0] - 1, seg_lo))

            # 找该段要处理的宽格（不再要求单行，支持处理 TSR 返回的跨多行的大表头）
            candidates = [
                c
                for c in work
                if seg_lo <= int(c["row_start"]) <= header_hi
                and int(c["col_end"]) - int(c["col_start"]) + 1 >= 2
            ]
            if not candidates:
                continue

            for cell in candidates:
                cs, ce = int(cell["col_start"]), int(cell["col_end"])
                rs = int(cell["row_start"])
                re_ = int(cell["row_end"])  # 修复冲突，改为 re_
                cx1, cy1, cx2, cy2 = _cell_bbox(cell)
                # 落入宽格的 OCR
                hits: List[Tuple[float, float, float, float, Dict[str, Any]]] = []
                for tb in boxes:
                    bx1, by1, bx2, by2 = _tb_bbox(tb)
                    mx, my = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                    if cx1 - 3 <= mx <= cx2 + 3 and cy1 - 3 <= my <= cy2 + 3:
                        hits.append((mx, my, bx1, bx2, tb))
                if len(hits) < 2:
                    continue

                mys = [h[1] for h in hits]
                y_clusters = _cluster_1d(mys, gap_thresh=y_gap)
                # 过滤过小的噪声簇
                y_clusters = [cl for cl in y_clusters if len(cl) >= 1]
                if not y_clusters:
                    continue

                # P97：几何 Y 过近时「单体[…]」与中段子表头常被聚成一带 → 按关键词强制分带
                _mid_kw = re.compile(
                    r"(二羧酸|双氨基|封端剂|三官能|四官能|四羧酸|二胺及其|二羟基|有机硅烷|二甲酰)"
                )
                _par_kw = re.compile(r"单体\s*[\[［]")
                if len(y_clusters) == 1 and len(hits) >= 2:
                    parent_idx = [
                        i
                        for i, h in enumerate(hits)
                        if _par_kw.search(str(h[4].get("text") or ""))
                    ]
                    child_idx = [
                        i
                        for i, h in enumerate(hits)
                        if _mid_kw.search(str(h[4].get("text") or ""))
                    ]
                    if parent_idx and child_idx:
                        rest = [
                            i
                            for i in range(len(hits))
                            if i not in parent_idx and i not in child_idx
                        ]
                        # 其余框并入更近的一带
                        def _band_mean(idxs: List[int]) -> float:
                            return float(sum(hits[i][1] for i in idxs) / max(len(idxs), 1))

                        py, cy = _band_mean(parent_idx), _band_mean(child_idx)
                        for i in rest:
                            if abs(hits[i][1] - py) <= abs(hits[i][1] - cy):
                                parent_idx.append(i)
                            else:
                                child_idx.append(i)
                        # 上带为 Y 更小者
                        if py <= cy:
                            y_clusters = [parent_idx, child_idx]
                        else:
                            y_clusters = [child_idx, parent_idx]

                # 每 Y 带内再做 X 聚类
                band_xgroups: List[List[List[int]]] = []
                for ycl in y_clusters:
                    mxs = [hits[i][0] for i in ycl]
                    # 映射回 hits 下标的簇
                    local = _cluster_1d(mxs, gap_thresh=x_gap)
                    # local 存的是 ycl 内的局部下标 → 转 hits 下标
                    groups = [[ycl[j] for j in loc] for loc in local]
                    band_xgroups.append(groups)

                n_y = len(y_clusters)
                max_xgroups = max(len(g) for g in band_xgroups)
                # 无需任何拆分
                if n_y == 1 and max_xgroups <= 1:
                    continue

                # 身列证据：该宽格覆盖列在下方是否多为原子格
                def _atomic_below(col: int) -> bool:
                    for c2 in work:
                        if int(c2["col_start"]) != col or int(c2["col_end"]) != col:
                            continue
                        if int(c2["row_start"]) > rs and seg_lo <= int(c2["row_start"]) <= seg_hi:
                            return True
                    return False

                covered = list(range(cs, ce + 1))
                body_support = sum(1 for c in covered if _atomic_below(c))
                # X 拆分需要身列支持；纯 Y 拆分（父级+子级）只要多带即可
                allow_x_split = body_support >= max(2, int(0.5 * len(covered) + 0.5)) or max_xgroups >= 2

                # ---- 构建替换格（可能插入一行）----
                # 计算每带的 y 范围
                band_yranges: List[Tuple[float, float]] = []
                for ycl in y_clusters:
                    tops, bots = [], []
                    for i in ycl:
                        _mx, _my, _bx1, _bx2, tb = hits[i]
                        _x1, y1, _x2, y2 = _tb_bbox(tb)
                        tops.append(y1)
                        bots.append(y2)
                    band_yranges.append((min(tops), max(bots)))

                # 下层 OCR 是否已落入其它逻辑格（说明子表头行已存在，勿再插行）
                lower_owned = 0
                lower_total = 0
                if n_y >= 2:
                    for ycl in y_clusters[1:]:
                        for i in ycl:
                            lower_total += 1
                            mx, my = hits[i][0], hits[i][1]
                            for c2 in work:
                                if (
                                    int(c2["row_start"]) == rs
                                    and int(c2["row_end"]) == re_
                                    and int(c2["col_start"]) == cs
                                    and int(c2["col_end"]) == ce
                                ):
                                    continue
                                if int(c2["row_start"]) <= rs:
                                    continue
                                bx1, by1, bx2, by2 = _cell_bbox(c2)
                                if bx1 - 2 <= mx <= bx2 + 2 and by1 - 2 <= my <= by2 + 2:
                                    lower_owned += 1
                                    break
                # 下一逻辑行在覆盖列上已有多个格 → 可能是已存在的子表头行
                next_row_overlap = [
                    c2
                    for c2 in work
                    if int(c2["row_start"]) == re_ + 1
                    and int(c2["col_end"]) >= cs
                    and int(c2["col_start"]) <= ce
                ]
                next_is_subheader = False
                if len(next_row_overlap) >= 2:
                    # 用 OCR 判断下一行是子表头还是数据行
                    y0 = float(row_seps[re_ + 1]) if re_ + 1 < len(row_seps) else 0.0
                    y1 = (
                        float(row_seps[re_ + 2])
                        if re_ + 2 < len(row_seps)
                        else y0 + 1.0
                    )
                    next_texts = []
                    for tb in boxes:
                        bx1, by1, bx2, by2 = _tb_bbox(tb)
                        my = (by1 + by2) / 2.0
                        mx = (bx1 + bx2) / 2.0
                        if not (y0 - 2 <= my <= y1 + 2):
                            continue
                        if not (float(col_seps[cs]) - 2 <= mx <= float(col_seps[ce + 1]) + 2):
                            continue
                        t = str(tb.get("text") or "").strip()
                        if t:
                            next_texts.append(t)
                    joined = " ".join(next_texts)
                    if next_texts and not re.search(
                        r"(合成例|实施例|実施例|比較例|比较例)\s*\d*", joined
                    ):
                        # 长中文占比高 → 子表头；短码/数值多 → 数据行
                        long_cjk = sum(
                            1
                            for t in next_texts
                            if _has_cjk(t) and len(re.sub(r"\s+", "", t)) >= 4
                        )
                        next_is_subheader = long_cjk >= max(1, len(next_texts) // 3)
                    elif not next_texts:
                        # 归属前无 OCR 字：≥2 个下级格且父为宽格 → 当作已有子表头行
                        next_is_subheader = (ce - cs + 1) >= 3

                children_already_exist = (
                    (
                        lower_total > 0
                        and lower_owned >= max(1, int(0.5 * lower_total + 0.5))
                    )
                    or next_is_subheader
                )

                # 子表头行已在：禁止再按 X 拆父格（P97 下表会把「单体[mol%]」拆碎）
                if children_already_exist and max_xgroups >= 2 and n_y <= 1:
                    # 若宽格物理框盖住子带，仍裁掉下半以免 OCR 串入父格
                    if re_ + 1 < len(row_seps):
                        split_y = float(row_seps[re_ + 1])
                        if cy1 + 3 < split_y < cy2 - 3:
                            cell["polygon"] = _rebuild_polygon(cx1, cy1, cx2, split_y)
                            cell["y_key"] = cy1
                            changed = True
                    continue

                top_texts = [
                    str(hits[i][4].get("text") or "")
                    for i in y_clusters[0]
                ]
                top_joined = "".join(top_texts)
                top_is_parent = bool(
                    re.search(r"(单体\s*[\[［]|聚合物)", top_joined)
                ) or (
                    len(re.sub(r"\s+", "", top_joined)) <= 16
                    and ("[" in top_joined or "［" in top_joined or "%" in top_joined)
                )
                bottom_max_x = (
                    max(len(g) for g in band_xgroups[1:]) if n_y >= 2 else 0
                )

                # 仅当「上带父级标题 + 下带多列标签」且当前是单行表头时才插行；
                # 避免把已经跨越多行的宽格强行打断
                need_y_split = (
                    rs == re_
                    and n_y >= 2
                    and not children_already_exist
                    and top_is_parent
                    and len(band_xgroups[0]) == 1
                    and bottom_max_x >= 2
                )

                # 子表头行已存在：仅收缩父格物理框到上带，避免 OCR 串入
                if children_already_exist and n_y >= 2 and len(band_xgroups[0]) <= 1:
                    split_y = 0.5 * (band_yranges[0][1] + band_yranges[1][0])
                    if cy1 + 3 < split_y < cy2 - 3:
                        cell["polygon"] = _rebuild_polygon(cx1, cy1, cx2, split_y)
                        cell["y_key"] = cy1
                        changed = True
                    continue

                # 单带多 X，或者已经是跨多行的大表头且有多X：只做列拆，不插行
                if not need_y_split and max_xgroups >= 2 and allow_x_split:
                    # 对于多带情况，取 X 分组最多的那个带作为切分基准
                    groups = max(band_xgroups, key=len)
                    new_cells: List[Dict[str, Any]] = []
                    y1 = float(row_seps[rs]) if rs < len(row_seps) else cy1
                    y2 = (
                        float(row_seps[re_ + 1])
                        if re_ + 1 < len(row_seps)
                        else cy2
                    )
                    for g in groups:
                        xs_lo = min(hits[i][2] for i in g)
                        xs_hi = max(hits[i][3] for i in g)
                        ncs, nce = _snap_x_range_to_cols(
                            xs_lo, xs_hi, col_seps, fallback=(cs, ce)
                        )
                        ncs = max(cs, min(ncs, ce))
                        nce = max(ncs, min(nce, ce))
                        x1 = float(col_seps[ncs])
                        x2 = float(col_seps[nce + 1])
                        new_cells.append(
                            _make_empty_cell(
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                                row_start=rs,
                                row_end=re_,
                                col_start=ncs,
                                col_end=nce,
                            )
                        )
                    if len(new_cells) >= 2:
                        work = [
                            c
                            for c in work
                            if not (
                                int(c["row_start"]) == rs
                                and int(c["row_end"]) == re_
                                and int(c["col_start"]) == cs
                                and int(c["col_end"]) == ce
                            )
                        ]
                        work.extend(new_cells)
                        total_x_splits += 1
                        changed = True
                        break
                    continue

                if need_y_split and n_y >= 2:
                    # 在第一带与第二带之间插入行切分
                    split_y = 0.5 * (band_yranges[0][1] + band_yranges[1][0])
                    if not (cy1 + 3 < split_y < cy2 - 3):
                        continue
                    insert_at = rs
                    remapped: List[Dict[str, Any]] = []
                    for c in work:
                        # 丢弃原宽格，后面按 Y/X 簇重建
                        if (
                            int(c["row_start"]) == rs
                            and int(c["row_end"]) == re_
                            and int(c["col_start"]) == cs
                            and int(c["col_end"]) == ce
                        ):
                            continue
                        nc = dict(c)
                        crs, cre = int(c["row_start"]), int(c["row_end"])
                        
                        peer_text = str(c.get("text") or "")
                        is_example_peer = bool(
                            re.search(
                                r"(合成例|实施例|実施例|比較例|比较例|对照例|参考例)",
                                peer_text,
                            )
                        )
                        
                        if crs == insert_at and is_example_peer:
                            # 实施例列头不跨多级表头行，保持原样
                            pass
                        elif cre >= insert_at:
                            # 被切分的行或跨过切分行的单元格，row_end 顺延一行
                            nc["row_end"] = cre + 1
                            if crs == insert_at and not is_example_peer:
                                _refresh_spans(nc)
                                x1, py1, x2, py2 = _cell_bbox(c)
                                nc["polygon"] = _rebuild_polygon(x1, py1, x2, max(py2, cy2))
                                remapped.append(nc)
                                continue

                        if crs > insert_at:
                            nc["row_start"] = crs + 1
                            nc["row_end"] = cre + 1
                        
                        _refresh_spans(nc)
                        remapped.append(nc)

                    # 上带：通常 1 个 X 簇 → 整段合并
                    top_groups = band_xgroups[0]
                    y1_top = cy1
                    y2_top = split_y
                    if len(top_groups) <= 1 or not allow_x_split:
                        remapped.append(
                            _make_empty_cell(
                                x1=float(col_seps[cs]),
                                y1=y1_top,
                                x2=float(col_seps[ce + 1]),
                                y2=y2_top,
                                row_start=rs,
                                row_end=rs,
                                col_start=cs,
                                col_end=ce,
                            )
                        )
                    else:
                        for g in top_groups:
                            xs_lo = min(hits[i][2] for i in g)
                            xs_hi = max(hits[i][3] for i in g)
                            ncs, nce = _snap_x_range_to_cols(
                                xs_lo, xs_hi, col_seps, fallback=(cs, ce)
                            )
                            ncs = max(cs, min(ncs, ce))
                            nce = max(ncs, min(nce, ce))
                            remapped.append(
                                _make_empty_cell(
                                    x1=float(col_seps[ncs]),
                                    y1=y1_top,
                                    x2=float(col_seps[nce + 1]),
                                    y2=y2_top,
                                    row_start=rs,
                                    row_end=rs,
                                    col_start=ncs,
                                    col_end=nce,
                                )
                            )

                    # 下带：所有后续 Y 带合并到 rs+1，再按 X 聚类
                    bottom_hit_idxs: List[int] = []
                    for ycl in y_clusters[1:]:
                        bottom_hit_idxs.extend(ycl)
                    bottom_mxs = [hits[i][0] for i in bottom_hit_idxs]
                    bottom_local = _cluster_1d(bottom_mxs, gap_thresh=x_gap)
                    bottom_groups = [
                        [bottom_hit_idxs[j] for j in loc] for loc in bottom_local
                    ]
                    y1_bot = split_y
                    y2_bot = cy2
                    if len(bottom_groups) <= 1:
                        remapped.append(
                            _make_empty_cell(
                                x1=float(col_seps[cs]),
                                y1=y1_bot,
                                x2=float(col_seps[ce + 1]),
                                y2=y2_bot,
                                row_start=rs + 1,
                                row_end=re_ + 1,
                                col_start=cs,
                                col_end=ce,
                            )
                        )
                    else:
                        raw_spans: List[Tuple[int, int]] = []
                        for g in bottom_groups:
                            xs_lo = min(hits[i][2] for i in g)
                            xs_hi = max(hits[i][3] for i in g)
                            ncs, nce = _snap_x_range_to_cols(
                                xs_lo, xs_hi, col_seps, fallback=(cs, ce)
                            )
                            ncs = max(cs, min(ncs, ce))
                            nce = max(ncs, min(nce, ce))
                            raw_spans.append((ncs, nce))
                        body_spans = _body_merge_spans(
                            work,
                            after_row=rs,
                            seg_hi=seg_hi,
                            cs=cs,
                            ce=ce,
                        )
                        filled = _align_spans_to_body(
                            raw_spans, body_spans, cs=cs, ce=ce
                        )
                        for ncs, nce in filled:
                            remapped.append(
                                _make_empty_cell(
                                    x1=float(col_seps[ncs]),
                                    y1=y1_bot,
                                    x2=float(col_seps[nce + 1]),
                                    y2=y2_bot,
                                    row_start=rs + 1,
                                    row_end=re_ + 1,
                                    col_start=ncs,
                                    col_end=nce,
                                )
                            )
                            total_x_splits += 1

                    work = remapped
                    total_y_splits += 1
                    changed = True
                    break

            if changed:
                break
        if not changed:
            break

    if total_y_splits or total_x_splits:
        logger.info(
            "表头重建: y_splits=%d x_group_splits=%d cells=%d",
            total_y_splits,
            total_x_splits,
            len(work),
        )
    return dedupe_overlapping_cells(work)

def explode_header_colspans_by_body(
    cells: List[Dict[str, Any]],
    boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """兼容旧名：转发到证据驱动的表头重建。"""
    return reconstruct_header_cells(cells, boxes)


def refine_tsr_cells_light(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """轻量后处理：先裁左行头过宽 colspan，再去重叠，保留 TSR 拓扑。"""
    if not cells:
        return cells
    from .row_header import clip_narrow_label_colspans, clip_row_header_child_overlaps

    cells = clip_row_header_child_overlaps(cells)
    cells = clip_narrow_label_colspans(cells)
    return dedupe_overlapping_cells(cells)


def logic_conflict_ratio(cells: Sequence[Dict[str, Any]]) -> float:
    """逻辑网格位置重叠比例：重叠位置数 / 总覆盖位置数（0 表示完整无重叠的合法网格）。"""
    if not cells:
        return 0.0
    occupied: set = set()
    conflicts = 0
    total = 0
    for c in cells:
        rs, re_ = int(c["row_start"]), int(c["row_end"])
        cs, ce = int(c["col_start"]), int(c["col_end"])
        for r in range(rs, re_ + 1):
            for col in range(cs, ce + 1):
                total += 1
                key = (r, col)
                if key in occupied:
                    conflicts += 1
                else:
                    occupied.add(key)
    return conflicts / max(total, 1)


_MONOMER_PARENT_RE = re.compile(r"单体\s*[\[［]")
_MONOMER_LEFT_ANCHOR_RE = re.compile(r"聚合物")
# 单体带右侧硬锚点：比率/当量等指标列（不含烯键式——P98/P94 烯键式是单体子列）
# 「来自/来源于具有…」覆盖 CN110（来自）与 CN111（来源于）两种 OCR 写法；
# 勿只锚「酸当量」——否则子表头行上的比率列会被当成单体子列（P92）。
_MONOMER_RIGHT_ANCHOR_RE = re.compile(
    r"(含有比率|含有比例|含有率|氟比率|酸当量|双键当量|"
    r"来自具有|来源于具有|所占的比率|所占比率)"
)
# 烯键式/烯属：子表头行且落在聚合物↔右硬锚之间 → 单体子列（不 promote）
_ENE_HEADER_RE = re.compile(r"(烯键式|烯属不饱和|不饱和双键)")
_CAP_AGENT_RE = re.compile(r"封端剂")
# 子头类别（异类标签墙，避免彼此吞列）
_CHILD_CAT_ACID_RE = re.compile(r"(四羧酸|二羧酸|酸酐)")
_CHILD_CAT_AMINE_RE = re.compile(r"(二胺|双氨基|硅烷|羟基二胺)")
_CHILD_CAT_COPOLY_RE = re.compile(r"(共聚成分|有机硅烷)")
# 表体化学代号（含可选括号用量）：BFE / MeTMS / cyEpoTMS / NA(40)
_CHEM_BODY_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9\-\+']{1,24}"
    r"(?:\s*[\(（]\s*\d+(?:\.\d+)?\s*[\)）])?$"
)
_PAREN_AMOUNT_RE = re.compile(r"^[\(（]\s*\d+(?:\.\d+)?\s*[\)）]$")


def _child_header_category(label: str) -> str:
    """子表头粗分类，用于对齐时的异类硬墙。"""
    t = label or ""
    if _ENE_HEADER_RE.search(t):
        return "ene"
    if _CAP_AGENT_RE.search(t):
        return "cap"
    # 三/四/二官能硅烷互为异类（须先于泛「硅烷/有机硅烷」规则）
    m_fn = re.search(r"([一二三四四五六七八九十\d]+)官能", t)
    if m_fn and re.search(r"硅烷", t):
        return f"silane_{m_fn.group(1)}"
    if _CHILD_CAT_ACID_RE.search(t):
        return "acid"
    if _CHILD_CAT_AMINE_RE.search(t):
        return "amine"
    if _CHILD_CAT_COPOLY_RE.search(t):
        return "copoly"
    if re.search(r"[\u4e00-\u9fff]", t):
        return "cjk"
    return "other"


_DASH_OR_BLANK_RE = re.compile(r"^[-—–−~～]+$")


def _body_col_has_substance(
    col: int,
    bodies: Sequence[Dict[str, Any]],
    boxes: Sequence[Dict[str, Any]],
    col_seps: Sequence[float],
) -> bool:
    """表体列是否含化学代号/数值等实质内容（短横/空不算）。"""
    texts: List[str] = []
    for b in bodies:
        if int(b["col_start"]) <= col <= int(b["col_end"]):
            t = re.sub(r"\s+", "", str(b.get("text") or ""))
            if t:
                texts.append(t)
    if col + 1 < len(col_seps) and bodies:
        x1 = float(col_seps[col])
        x2 = float(col_seps[col + 1])
        y_lo = min(_cell_bbox(b)[1] for b in bodies)
        y_hi = max(_cell_bbox(b)[3] for b in bodies)
        for tb in boxes:
            tx1, ty1, tx2, ty2 = _tb_bbox(tb)
            mx, my = 0.5 * (tx1 + tx2), 0.5 * (ty1 + ty2)
            if x1 <= mx <= x2 and y_lo <= my <= y_hi:
                t = re.sub(r"\s+", "", str(tb.get("text") or ""))
                if t:
                    texts.append(t)
    for t in texts:
        if not t or _DASH_OR_BLANK_RE.fullmatch(t):
            continue
        if _CHEM_BODY_RE.fullmatch(t) or _PAREN_AMOUNT_RE.fullmatch(t):
            return True
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            return True
        if re.search(r"[A-Za-z]{2,}", t):
            return True
    return False


def _cell_label_text(
    cell: Dict[str, Any], boxes: Sequence[Dict[str, Any]]
) -> str:
    parts = [str(tb.get("text") or "") for tb in _texts_in_cell(cell, boxes)]
    own = str(cell.get("text") or "")
    if own:
        parts.append(own)
    return re.sub(r"\s+", "", "".join(parts))


def _set_cell_cols(
    cell: Dict[str, Any],
    cs: int,
    ce: int,
    col_seps: Sequence[float],
) -> None:
    """更新逻辑列并按 col_seps 重写多边形 x 范围（保留原 y）。"""
    cs = int(cs)
    ce = int(ce)
    if ce < cs:
        return
    cell["col_start"] = cs
    cell["col_end"] = ce
    _refresh_spans(cell)
    if len(col_seps) > ce + 1:
        x1 = float(col_seps[cs])
        x2 = float(col_seps[ce + 1])
        _, y1, _, y2 = _cell_bbox(cell)
        cell["polygon"] = _rebuild_polygon(x1, y1, x2, y2)


def _set_cell_rows(
    cell: Dict[str, Any],
    rs: int,
    re: int,
    row_seps: Sequence[float],
) -> None:
    """更新逻辑行并按 row_seps 重写多边形 y 范围（保留原 x）。"""
    rs = int(rs)
    re = int(re)
    if re < rs:
        return
    cell["row_start"] = rs
    cell["row_end"] = re
    _refresh_spans(cell)
    if len(row_seps) > re + 1:
        y1 = float(row_seps[rs])
        y2 = float(row_seps[re + 1])
        x1, _, x2, _ = _cell_bbox(cell)
        cell["polygon"] = _rebuild_polygon(x1, y1, x2, y2)


def _cell_x_center(cell: Dict[str, Any]) -> float:
    x1, _y1, x2, _y2 = _cell_bbox(cell)
    return 0.5 * (x1 + x2)


def _align_monomer_children_to_body(
    children: List[Dict[str, Any]],
    body_cells: List[Dict[str, Any]],
    col_seps: Sequence[float],
    *,
    band_lo: int,
    band_hi: int,
    boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> int:
    """按表体列把单体子表头对齐到连续列带。

    1) 下一子头 OCR/多边形左缘作硬墙（左对齐跨列头：SiDA 归二胺而非封端剂）；
    2) 收尾：左子头右缘若拖着已越过其文案、且更靠近右邻的「仅短横/空」列，
       则让给右邻（P93：三官能不得吞四官能前空列；P97：含 AcrTMS 的列保留）。
    """
    if len(children) < 1 or len(body_cells) < 2:
        return 0
    boxes = list(boxes) if boxes else []
    bodies = sorted(
        [
            c
            for c in body_cells
            if band_lo <= int(c["col_start"]) <= band_hi
            or band_lo <= int(c["col_end"]) <= band_hi
        ],
        key=_cell_x_center,
    )
    kids = sorted(children, key=_cell_x_center)
    if len(bodies) < 2 or not kids:
        return 0

    body_atom_cols = sorted(
        {
            col
            for b in bodies
            for col in range(int(b["col_start"]), int(b["col_end"]) + 1)
            if band_lo <= col <= band_hi
        }
    )
    if len(body_atom_cols) < 2:
        return 0

    def _col_x(col: int) -> float:
        if col + 1 < len(col_seps):
            return 0.5 * (float(col_seps[col]) + float(col_seps[col + 1]))
        return float(col_seps[min(col, len(col_seps) - 1)])

    def _kid_left_x(kid: Dict[str, Any]) -> float:
        # 左对齐跨列头：左缘才是本列带起点；中心会偏到右邻类
        tbs = _texts_in_cell(kid, boxes)
        if tbs:
            return min(float(_tb_bbox(tb)[0]) for tb in tbs)
        return float(_cell_bbox(kid)[0])

    def _kid_text_right(kid: Dict[str, Any]) -> float:
        tbs = _texts_in_cell(kid, boxes)
        if tbs:
            return max(float(_tb_bbox(tb)[2]) for tb in tbs)
        return float(_cell_bbox(kid)[2])

    def _kid_label(kid: Dict[str, Any]) -> str:
        t = _cell_label_text(kid, boxes)
        if t.strip():
            return t
        tbs = _texts_in_cell(kid, boxes)
        if tbs:
            return "".join(str(tb.get("text") or "") for tb in tbs)
        return t

    kid_meta = [
        (
            k,
            _kid_left_x(k),
            _child_header_category(_kid_label(k)),
        )
        for k in kids
    ]
    ordered_meta = sorted(
        kid_meta, key=lambda t: (int(t[0]["col_start"]), t[1])
    )

    # 下一子头左缘为墙：左子头吃到墙左侧最后一列
    cuts: List[int] = []
    for i in range(len(ordered_meta) - 1):
        wall_x = float(ordered_meta[i + 1][1])
        cut = band_lo - 1
        for c in range(band_lo, band_hi + 1):
            if _col_x(c) < wall_x:
                cut = c
        max_cut = band_hi - (len(ordered_meta) - 1 - i)
        min_cut = band_lo + i
        cut = max(min_cut, min(cut, max_cut))
        cuts.append(cut)

    spans: List[Tuple[Dict[str, Any], int, int]] = []
    lo = band_lo
    for i, (kid, _xc, _cat) in enumerate(ordered_meta):
        if i < len(cuts):
            hi = cuts[i]
        else:
            hi = band_hi
        if hi < lo:
            hi = lo
        spans.append((kid, lo, hi))
        lo = hi + 1

    # 收尾：左子头不得拖着越过文案、更靠右邻的短横/空列（P93）
    for i in range(len(spans) - 1):
        left_kid, l_lo, l_hi = spans[i]
        right_kid, r_lo, r_hi = spans[i + 1]
        ltr = _kid_text_right(left_kid)
        rtl = _kid_left_x(right_kid)
        while l_hi > l_lo:
            cx = _col_x(l_hi)
            if cx <= ltr + 1.0:
                break
            if _body_col_has_substance(l_hi, bodies, boxes, col_seps):
                break
            if abs(cx - rtl) > abs(cx - ltr):
                break
            l_hi -= 1
            r_lo = l_hi + 1
        spans[i] = (left_kid, l_lo, l_hi)
        spans[i + 1] = (right_kid, r_lo, r_hi)

    changed = 0
    for kid, new_cs, new_ce in spans:
        old_cs, old_ce = int(kid["col_start"]), int(kid["col_end"])
        if (old_cs, old_ce) != (new_cs, new_ce):
            _set_cell_cols(kid, new_cs, new_ce, col_seps)
            changed += 1
            logger.info(
                "repair_monomer_parent_spans: child align %d-%d → %d-%d",
                old_cs,
                old_ce,
                new_cs,
                new_ce,
            )
    return changed


def repair_monomer_parent_spans(
    cells: List[Dict[str, Any]],
    boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    按子表分段，把过窄/错位的「单体[…]」父格对齐到同段单体子列并集。

    只改已有格子的 col_start/col_end 与多边形，不插列、不融合框线。
    用于轻量 TSR 路径修复 P98 类分段异形表；亦覆盖无右侧「含有比率」
    锚点、单体带到表缘结束的 P46/P47 类表。
    """
    if not cells:
        return cells
    # 允许无 OCR boxes：单元测试与已写入 cell.text 的拓扑修复仍可用
    boxes = list(boxes) if boxes else []

    from ..utils.segments import find_row_segments

    work = [dict(c) for c in cells]
    row_seps, col_seps = _derive_seps(work)
    if len(col_seps) < 3:
        return work

    segments = find_row_segments(work, text_boxes=boxes)
    if not segments:
        # 无分段信息时整表当作一段（有 text 的拓扑仍可修）
        max_r = max(int(c["row_end"]) for c in work)
        segments = [(0, max_r)]

    removed: set = set()
    changed = 0

    for seg_lo, seg_hi in segments:
        seg_cells = [
            c
            for c in work
            if id(c) not in removed
            and seg_lo <= int(c["row_start"]) <= seg_hi
        ]
        if not seg_cells:
            continue

        parents = [
            c
            for c in seg_cells
            if _MONOMER_PARENT_RE.search(_cell_label_text(c, boxes))
        ]
        if not parents:
            continue
        # 取段内最靠上的单体父格
        parent = min(
            parents,
            key=lambda c: (int(c["row_start"]), int(c["col_start"])),
        )
        prs = int(parent["row_start"])
        pre = int(parent["row_end"])

        left_anchor_ce = -1
        right_anchor_cs = 10**9
        for c in seg_cells:
            label = _cell_label_text(c, boxes)
            if not label:
                continue
            if _MONOMER_LEFT_ANCHOR_RE.search(label):
                left_anchor_ce = max(left_anchor_ce, int(c["col_end"]))
            if _MONOMER_RIGHT_ANCHOR_RE.search(label):
                right_anchor_cs = min(right_anchor_cs, int(c["col_start"]))

        if left_anchor_ce < 0:
            continue

        # 无右侧「含有比率」等锚点时：单体带到段内最右列（P46/P47）
        max_seg_col = max(int(c["col_end"]) for c in seg_cells)
        if right_anchor_cs >= 10**9:
            band_lo = left_anchor_ce + 1
            band_hi = max_seg_col
        else:
            if right_anchor_cs <= left_anchor_ce + 1:
                continue
            band_lo = left_anchor_ce + 1
            band_hi = right_anchor_cs - 1

        if band_hi < band_lo:
            continue

        def _in_band(c: Dict[str, Any]) -> bool:
            return int(c["col_end"]) >= band_lo and int(c["col_start"]) <= band_hi

        def _is_body_label(label: str) -> bool:
            t = (label or "").strip()
            if not t:
                return False
            if re.search(r"(合成例|实施例|実施例|比較例|比较例)", t):
                return True
            compact = re.sub(r"\s+", "", t)
            return bool(_CHEM_BODY_RE.fullmatch(compact) or _PAREN_AMOUNT_RE.fullmatch(compact))

        def _is_child_header_label(label: str) -> bool:
            t = (label or "").strip()
            if not t or _is_body_label(t):
                return False
            if _MONOMER_PARENT_RE.search(t) or _MONOMER_LEFT_ANCHOR_RE.search(t):
                return False
            if _MONOMER_RIGHT_ANCHOR_RE.search(t):
                return False
            # 中文子表头（含 P98 烯键式子列）或封端剂等
            return bool(
                re.search(r"[\u4e00-\u9fff]", t)
                or re.search(r"(封端剂|硅烷|衍生物|化合物|共聚)", t)
            )

        # 子表头行：父格之后、第一个带内 ≥2 个非体数据格的行（不要求紧邻 pre+1）
        children: List[Dict[str, Any]] = []
        child_row = pre + 1
        for r in range(pre + 1, seg_hi + 1):
            cand = [
                c
                for c in seg_cells
                if int(c["row_start"]) == r
                and c is not parent
                and _in_band(c)
                and not _MONOMER_PARENT_RE.search(_cell_label_text(c, boxes))
                and not _MONOMER_LEFT_ANCHOR_RE.search(_cell_label_text(c, boxes))
            ]
            if len(cand) < 2:
                continue
            # 若该行已是表体化学代号，则不是子表头
            if any(_is_body_label(_cell_label_text(c, boxes)) for c in cand):
                break
            if any(_is_child_header_label(_cell_label_text(c, boxes)) for c in cand):
                children = cand
                child_row = r
                break

        # 表体行：优先含合成例/化学代号的行
        body_rows = sorted(
            {
                int(c["row_start"])
                for c in seg_cells
                if int(c["row_start"]) > child_row
            }
        )
        body_cells: List[Dict[str, Any]] = []
        fallback_body: List[Dict[str, Any]] = []
        first_body_row: Optional[int] = None
        for br in body_rows:
            cand = [
                c
                for c in seg_cells
                if int(c["row_start"]) == br and _in_band(c)
            ]
            if len(cand) < 2:
                continue
            if any(_is_body_label(_cell_label_text(c, boxes)) for c in cand):
                body_cells = cand
                first_body_row = br
                break
            if not fallback_body and not any(
                _is_child_header_label(_cell_label_text(c, boxes)) for c in cand
            ):
                fallback_body = cand
                first_body_row = br
        if not body_cells:
            body_cells = fallback_body

        # 子表头可能跨多逻辑行（P46：二羧酸@r4 + 封端剂@r5）：收齐首表体行之前带内表头格
        if first_body_row is not None:
            header_kids = [
                c
                for c in seg_cells
                if c is not parent
                and pre < int(c["row_start"]) < first_body_row
                and _in_band(c)
                and _is_child_header_label(_cell_label_text(c, boxes))
            ]
            if len(header_kids) >= 2:
                children = header_kids
            elif len(header_kids) > len(children):
                children = header_kids

        # 烯键式若在子表头行且落在聚合物↔右硬锚之间，保留在带内（不再按封端剂门控剔出）

        # 父格列范围：子表头并集优先，否则表体并集；二者皆有时取并集更稳
        span_cells: List[Dict[str, Any]] = []
        if len(children) >= 2:
            span_cells.extend(children)
        if len(body_cells) >= 2:
            span_cells.extend(body_cells)
        if len(span_cells) < 2:
            span_cells = [
                c
                for c in seg_cells
                if c is not parent
                and _in_band(c)
                and not _MONOMER_LEFT_ANCHOR_RE.search(_cell_label_text(c, boxes))
            ]
        if len(span_cells) < 2:
            continue

        target_cs = min(int(c["col_start"]) for c in span_cells)
        target_ce = max(int(c["col_end"]) for c in span_cells)
        target_cs = max(target_cs, band_lo)
        target_ce = min(target_ce, band_hi)
        if target_ce < target_cs:
            continue

        old_cs, old_ce = int(parent["col_start"]), int(parent["col_end"])
        if (old_cs, old_ce) != (target_cs, target_ce):
            _set_cell_cols(parent, target_cs, target_ce, col_seps)
            changed += 1
            logger.info(
                "repair_monomer_parent_spans: seg[%d,%d] 单体 %d-%d → %d-%d",
                seg_lo,
                seg_hi,
                old_cs,
                old_ce,
                target_cs,
                target_ce,
            )

        # 子表头按表体列物理位置对齐（修复「三官能」只占 1 列等）
        if children and body_cells:
            changed += _align_monomer_children_to_body(
                children,
                body_cells,
                col_seps,
                band_lo=target_cs,
                band_hi=target_ce,
                boxes=boxes,
            )
            # 清掉子表头行上空壳，避免对齐后 colspan 已被拉开、空格仍占位
            for c in list(seg_cells):
                if c is parent or id(c) in removed:
                    continue
                if int(c["row_start"]) != child_row:
                    continue
                if _cell_label_text(c, boxes).strip():
                    continue
                cs, ce = int(c["col_start"]), int(c["col_end"])
                covered = any(
                    int(k["col_start"]) <= ce and int(k["col_end"]) >= cs
                    for k in children
                )
                if covered or (target_cs <= cs and ce <= target_ce):
                    removed.add(id(c))
                    changed += 1

        # 聚合物行起点若低于单体父格，上延 rowspan 盖住父行，避免左侧空角被并进「单体」
        for c in seg_cells:
            if c is parent or id(c) in removed:
                continue
            label = _cell_label_text(c, boxes)
            if not _MONOMER_LEFT_ANCHOR_RE.search(label):
                continue
            if int(c["col_end"]) > left_anchor_ce:
                continue
            old_rs = int(c["row_start"])
            if old_rs > prs:
                _set_cell_rows(c, prs, int(c["row_end"]), row_seps)
                changed += 1
                logger.info(
                    "repair_monomer_parent_spans: 聚合物上延 row %d → %d",
                    old_rs,
                    prs,
                )

        # 去掉父行上落在新父格内部的空壳占位格
        for c in list(seg_cells):
            if c is parent or id(c) in removed:
                continue
            if int(c["row_start"]) != prs:
                continue
            if int(c["col_start"]) >= target_cs and int(c["col_end"]) <= target_ce:
                label = _cell_label_text(c, boxes)
                if not label.strip():
                    removed.add(id(c))
                    changed += 1

        # 收缩与同级兄弟逻辑重叠的过宽格（不主动制造空洞微列）
        for row_cells in (children, body_cells):
            ordered = sorted(
                [c for c in row_cells if id(c) not in removed],
                key=lambda c: int(c["col_start"]),
            )
            for i, c in enumerate(ordered):
                cs, ce = int(c["col_start"]), int(c["col_end"])
                if ce <= cs:
                    continue
                nxt = None
                for o in ordered[i + 1 :]:
                    if int(o["col_start"]) > cs:
                        nxt = o
                        break
                if nxt is None:
                    if ce > target_ce:
                        _set_cell_cols(c, cs, target_ce, col_seps)
                        changed += 1
                    continue
                ncs = int(nxt["col_start"])
                if ce >= ncs:
                    new_ce = ncs - 1
                    if new_ce >= cs:
                        _set_cell_cols(c, cs, new_ce, col_seps)
                        changed += 1
                        logger.info(
                            "repair_monomer_parent_spans: shrink overlap "
                            "cols %d-%d → %d-%d",
                            cs,
                            ce,
                            cs,
                            new_ce,
                        )

    if removed:
        work = [c for c in work if id(c) not in removed]
    if changed:
        work = dedupe_overlapping_cells(work)
    return work


# 表体行上误落的中段/侧栏表头（P94/P98：四羧酸二酐、酸当量等）
_LIFTABLE_HEADER_RE = re.compile(
    r"(酸当量|双键当量|四羧酸二酐|二羧酸酐|封端剂|"
    r"具有.{2,40}(?:化合物|共聚成分|衍生物|不饱和羧酸|不饱和化合物))"
)
_SYNTHESIS_ROW_RE = re.compile(r"(合成例|实施例|実施例|比較例|比较例)")
# 侧栏整列表头：仅比率/当量等强制上延；烯键式/烯属默认视为单体子头（见 promote）
_SIDE_HEADER_RE = re.compile(
    r"(含有比率|含有率|烯键式|烯属不饱和|不饱和双键|氟比率|酸当量|双键当量|"
    r"来自具有|来源于具有)"
)
_SIDE_HEADER_FORCE_RE = re.compile(
    r"(含有比率|含有率|氟比率|酸当量|双键当量|来自具有|来源于具有)"
)


def _compact_cell_text(cell: Dict[str, Any]) -> str:
    return re.sub(r"\s+", "", str(cell.get("text") or ""))


def _looks_like_body_value(text: str) -> bool:
    """化学代号/用量/纯数字等表体值，不可当表头上提。"""
    t = re.sub(r"\s+", "", text or "")
    if not t:
        return False
    if _SYNTHESIS_ROW_RE.search(t):
        return True
    if re.fullmatch(r"[-—–−~～]|[-—–−]{2,}", t):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return True
    if _CHEM_BODY_RE.fullmatch(t) or _PAREN_AMOUNT_RE.fullmatch(t):
        return True
    # 带克/摩尔的配方描述（表体）
    if re.search(r"\d+(?:\.\d+)?\s*(?:g|mol|g/mol)", t, re.I):
        return True
    return False


def lift_misplaced_header_labels(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    把落在「合成例」表体行上的中段/侧栏表头字上提到上方表头行。

    P94/P98：TSR 漏建子表头格时，IoA 把「四羧酸二酐」「酸当量」等打进合成例行，
    既挤掉真实数据槽位，又造成「字没了」的观感。
    """
    if not cells:
        return cells

    work = [dict(c) for c in cells]
    row_seps, col_seps = _derive_seps(work)
    if len(row_seps) < 3 or len(col_seps) < 3:
        return work

    # 含合成例标签的行
    synth_rows: set[int] = set()
    for c in work:
        if int(c["col_start"]) > 1:
            continue
        if _SYNTHESIS_ROW_RE.search(str(c.get("text") or "")):
            synth_rows.add(int(c["row_start"]))
    if not synth_rows:
        return work

    def _occupies(rs: int, re_: int, cs: int, ce: int, skip: Any = None) -> Optional[Dict[str, Any]]:
        for o in work:
            if o is skip:
                continue
            if int(o["row_end"]) < rs or int(o["row_start"]) > re_:
                continue
            if int(o["col_end"]) < cs or int(o["col_start"]) > ce:
                continue
            return o
        return None

    def _header_target_row(body_row: int) -> Optional[int]:
        """合成例行之上最近的「聚合物/单体」所在行（父表头行）。"""
        best = None
        for c in work:
            rs = int(c["row_start"])
            if rs >= body_row:
                continue
            t = str(c.get("text") or "")
            if _MONOMER_PARENT_RE.search(t) or _MONOMER_LEFT_ANCHOR_RE.search(t):
                if best is None or rs > best:
                    best = rs
        return best

    lifted = 0
    for cell in list(work):
        rs = int(cell["row_start"])
        if rs not in synth_rows:
            continue
        if int(cell["row_end"]) != rs:
            continue
        raw = str(cell.get("text") or "").strip()
        if not raw or _looks_like_body_value(raw):
            continue
        compact = _compact_cell_text(cell)
        if not _LIFTABLE_HEADER_RE.search(compact):
            continue
        # 格内同时含表头词与独立数值 → 勿整格上提（避免误清数据）
        if re.search(r"(?<![A-Za-z])\d+(?:\.\d+)?", compact) and re.search(
            r"(酸当量|双键当量|四羧酸|衍生物|化合物|共聚成分)", compact
        ):
            # 纯「酸当量[g/mol]」里的单位不算数据值
            if not re.fullmatch(
                r".*(?:酸当量|双键当量)\s*[\[［]?[^\]］]*[\]］]?", compact
            ):
                continue

        parent_row = _header_target_row(rs)
        if parent_row is None:
            continue
        # 仅酸当量/比率等侧栏；烯键式在表1-3 中是单体子头，不能当侧栏
        is_side = bool(
            re.search(r"(酸当量|双键当量|含有比率|含有率|来自具有|来源于具有)", compact)
        )
        child_row = parent_row + 1
        if child_row >= rs:
            child_row = parent_row
        target_rs = parent_row if is_side else child_row
        target_re = child_row if is_side and child_row > parent_row else target_rs

        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        owner = _occupies(target_rs, target_re, cs, ce, skip=cell)
        if owner is not None:
            ot = str(owner.get("text") or "").strip()
            if ot and ot != raw:
                # 已被其它字占用，不强行覆盖
                continue
            owner["text"] = raw
            owner["texts"] = list(cell.get("texts") or [])
            if is_side and int(owner["row_end"]) < target_re:
                _set_cell_rows(owner, int(owner["row_start"]), target_re, row_seps)
            cell["text"] = ""
            cell["texts"] = []
        else:
            # 原地挪到表头行，避免新建格被去重吃掉
            _set_cell_rows(cell, target_rs, target_re, row_seps)
        lifted += 1

    if lifted:
        logger.info("lift_misplaced_header_labels: 上提 %d 个误落表头", lifted)
        work = dedupe_overlapping_cells(work)
        work = _expand_monomer_over_child_headers(work)
    return work


def _expand_monomer_over_child_headers(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """上提子表头后，把「单体」父格 colspan 扩到覆盖同段子头列。"""
    row_seps, col_seps = _derive_seps(cells)
    if len(col_seps) < 3:
        return cells
    changed = 0
    for parent in cells:
        pt = str(parent.get("text") or "")
        if not _MONOMER_PARENT_RE.search(pt):
            continue
        prs, pre = int(parent["row_start"]), int(parent["row_end"])
        child_row = pre + 1
        kids = [
            c
            for c in cells
            if int(c["row_start"]) == child_row == int(c["row_end"])
            and int(c["col_start"]) >= int(parent["col_start"])
            and str(c.get("text") or "").strip()
            and not _looks_like_body_value(str(c.get("text") or ""))
            and not _SIDE_HEADER_FORCE_RE.search(_compact_cell_text(c))
        ]
        if not kids:
            continue
        # 勿把已是整列外侧的烯键式扩进单体（rowspan≥2 且已在父格右侧）
        kids = [
            k
            for k in kids
            if not (
                _ENE_HEADER_RE.search(str(k.get("text") or ""))
                and int(k["row_end"]) > int(k["row_start"])
                and int(k["col_start"]) > int(parent["col_end"])
            )
        ]
        if not kids:
            continue
        new_ce = max(int(parent["col_end"]), max(int(k["col_end"]) for k in kids))
        if new_ce > int(parent["col_end"]):
            _set_cell_cols(parent, int(parent["col_start"]), new_ce, col_seps)
            changed += 1
    if changed:
        logger.info("expand_monomer_over_child_headers: 扩展 %d 个单体父格", changed)
    return cells


def promote_side_header_rowspans(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    将侧栏整列表头（含有比率/酸当量/来源于…）从仅占子表头行上延到父表头行。

    烯键式/烯属：若仅占子表头行且落在单体带内（或将被单体覆盖），不上延；
    仅对 `_SIDE_HEADER_FORCE_RE`（比率/当量/来源于）强制上延。
    """
    if not cells:
        return cells

    work = [dict(c) for c in cells]
    row_seps, col_seps = _derive_seps(work)
    if len(row_seps) < 3:
        return work

    # 找「单体」父格行，作为上延目标
    parent_rows: List[int] = []
    for c in work:
        if _MONOMER_PARENT_RE.search(str(c.get("text") or "")):
            parent_rows.append(int(c["row_start"]))
    if not parent_rows:
        return work
    parent_rows = sorted(set(parent_rows))

    def _col_free(row: int, cs: int, ce: int, skip: Any) -> bool:
        for o in work:
            if o is skip:
                continue
            if int(o["row_start"]) > row or int(o["row_end"]) < row:
                continue
            if int(o["col_end"]) < cs or int(o["col_start"]) > ce:
                continue
            if str(o.get("text") or "").strip():
                return False
        return True

    def _clear_empty_occupants(row: int, cs: int, ce: int, skip: Any) -> None:
        doomed = []
        for o in work:
            if o is skip:
                continue
            if int(o["row_start"]) > row or int(o["row_end"]) < row:
                continue
            if int(o["col_end"]) < cs or int(o["col_start"]) > ce:
                continue
            if str(o.get("text") or "").strip():
                continue
            doomed.append(o)
        if doomed:
            ids = {id(x) for x in doomed}
            work[:] = [c for c in work if id(c) not in ids]

    changed = 0
    for cell in work:
        t = _compact_cell_text(cell)
        if not t or not _SIDE_HEADER_FORCE_RE.search(t):
            continue  # 烯键式等非强制侧栏：保持子表头行
        if _MONOMER_PARENT_RE.search(t) or _MONOMER_LEFT_ANCHOR_RE.search(t):
            continue
        rs, re_ = int(cell["row_start"]), int(cell["row_end"])
        if re_ > rs:
            continue  # 已跨行
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        target_parent = None
        for pr in parent_rows:
            if rs == pr + 1:
                target_parent = pr
                break
        if target_parent is None:
            continue
        if not _col_free(target_parent, cs, ce, skip=cell):
            continue
        _clear_empty_occupants(target_parent, cs, ce, skip=cell)
        _set_cell_rows(cell, target_parent, re_, row_seps)
        changed += 1
        logger.info(
            "promote_side_header_rowspans: %s row %d → %d-%d",
            t[:20],
            rs,
            target_parent,
            re_,
        )

    if changed:
        work = dedupe_overlapping_cells(work)
    return work


_RATIO_SIDE_RE = re.compile(
    r"(来自具有|来源于具有|来源于|来自具有|含有比率|含有比例|含有率|所占的比率|所占比率)"
)
_EQUIV_SIDE_RE = re.compile(r"(?:全)?酸当量|双键当量")
_SUBHEADER_LEAK_RE = re.compile(
    r"(二羧酸|四羧酸|双氨基酚|封端剂|三官能|四官能|二官能|共聚成分|二甲酰|有机硅烷)"
)
_SHORT_SIDE_HEADER_RE = re.compile(
    r"^(?:来源于具有|来自具有)(?:[\[［【]mo[l1]%\s*[\]］】]?)?.{0,10}$"
)


def demote_ene_inside_monomer_band(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """已跨父+子行的烯键式若落在右硬锚左侧，收回子表头行并扩单体父格覆盖之。"""
    if not cells:
        return cells
    work = [dict(c) for c in cells]
    row_seps, col_seps = _derive_seps(work)
    if len(row_seps) < 3 or len(col_seps) < 3:
        return work

    parents = [
        c
        for c in work
        if _MONOMER_PARENT_RE.search(str(c.get("text") or ""))
    ]
    if not parents:
        return work

    right_anchor_cols = []
    for c in work:
        t = _compact_cell_text(c)
        if _MONOMER_RIGHT_ANCHOR_RE.search(t) and not _ENE_HEADER_RE.search(t):
            right_anchor_cols.append(int(c["col_start"]))
    min_right = min(right_anchor_cols) if right_anchor_cols else None

    changed = 0
    for parent in parents:
        prs, pre = int(parent["row_start"]), int(parent["row_end"])
        pcs, pce = int(parent["col_start"]), int(parent["col_end"])
        child_row = pre + 1
        for ene in work:
            t = str(ene.get("text") or "")
            if not _ENE_HEADER_RE.search(t):
                continue
            ers, ere = int(ene["row_start"]), int(ene["row_end"])
            ecs, ece = int(ene["col_start"]), int(ene["col_end"])
            # 仅处理已跨到父行、且紧挨单体右侧或已部分重叠的烯键式
            if ere <= ers:
                continue
            if ers != prs:
                continue
            if min_right is not None and ecs >= min_right:
                continue
            if ece < pcs:
                continue
            # 收到子表头行
            if ere >= child_row:
                _set_cell_rows(ene, child_row, child_row, row_seps)
            # 父格扩到覆盖烯键式
            new_ce = max(pce, ece)
            if new_ce > pce:
                _set_cell_cols(parent, pcs, new_ce, col_seps)
                pce = new_ce
            changed += 1
            logger.info(
                "demote_ene_inside_monomer_band: ene %d-%d → child row %d, 单体→%d",
                ecs,
                ece,
                child_row,
                pce,
            )
    if changed:
        work = dedupe_overlapping_cells(work)
    return work


def split_glued_side_headers(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """侧栏格同时含比率类与当量类，且右侧有空邻列/占位洞 → 当量挪到空列。"""
    if not cells:
        return cells
    work = [dict(c) for c in cells]
    row_seps, col_seps = _derive_seps(work)
    changed = 0

    def _full_text(c: Dict[str, Any]) -> str:
        parts = [str(t.get("text") or "") for t in (c.get("texts") or [])]
        own = str(c.get("text") or "")
        if own:
            parts.append(own)
        return "\n".join(p for p in parts if p)

    for cell in list(work):
        raw = _full_text(cell)
        compact = re.sub(r"\s+", "", raw)
        if not (_RATIO_SIDE_RE.search(compact) and _EQUIV_SIDE_RE.search(compact)):
            continue
        rs, re_ = int(cell["row_start"]), int(cell["row_end"])
        ce = int(cell["col_end"])
        empty = None
        for o in work:
            if o is cell:
                continue
            if int(o["row_start"]) > re_ or int(o["row_end"]) < rs:
                continue
            ocs = int(o["col_start"])
            if ocs < ce + 1 or ocs > ce + 2:
                continue
            ot = re.sub(r"\s+", "", _full_text(o))
            # 可接收当量：空、或无比率/当量主标签的占位
            if _RATIO_SIDE_RE.search(ot) and len(ot) > 8:
                continue
            if _EQUIV_SIDE_RE.search(ot) and not _RATIO_SIDE_RE.search(ot):
                continue
            if _MONOMER_PARENT_RE.search(ot) or _MONOMER_LEFT_ANCHOR_RE.search(ot):
                continue
            if _ENE_HEADER_RE.search(ot) or _CAP_AGENT_RE.search(ot):
                continue
            if _SYNTHESIS_ROW_RE.search(ot) or _looks_like_body_value(ot):
                continue
            empty = o
            break
        # 右侧无空格对象、但与「双键当量」之间有占位洞 → 插入新格
        if empty is None and len(col_seps) > ce + 2:
            next_equiv = None
            for o in work:
                if o is cell:
                    continue
                if int(o["row_start"]) > re_ or int(o["row_end"]) < rs:
                    continue
                if int(o["col_start"]) <= ce:
                    continue
                ot = re.sub(r"\s+", "", _full_text(o))
                if _EQUIV_SIDE_RE.search(ot) and "双键" in ot:
                    next_equiv = o
                    break
            if next_equiv is not None and int(next_equiv["col_start"]) >= ce + 2:
                hole = ce + 1
                # 洞列未被占用
                occupied = any(
                    int(o["col_start"]) <= hole <= int(o["col_end"])
                    and not (int(o["row_start"]) > re_ or int(o["row_end"]) < rs)
                    for o in work
                    if o is not cell
                )
                if not occupied and hole + 1 < len(col_seps):
                    x1 = float(col_seps[hole])
                    x2 = float(col_seps[hole + 1])
                    y1 = float(row_seps[rs]) if rs < len(row_seps) else 0.0
                    y2 = (
                        float(row_seps[re_ + 1])
                        if re_ + 1 < len(row_seps)
                        else y1 + 10.0
                    )
                    empty = _make_empty_cell(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        row_start=rs,
                        row_end=re_,
                        col_start=hole,
                        col_end=hole,
                    )
                    work.append(empty)
                    logger.info(
                        "split_glued_side_headers: 插入占位洞 col %d", hole
                    )
        if empty is None:
            continue
        equiv_key = "双键当量" if "双键当量" in compact else "酸当量"
        idx = raw.find(equiv_key)
        if idx < 0:
            idx = raw.find("酸当量")
            if idx < 0:
                idx = raw.find("双键当量")
        if idx < 0:
            m = _EQUIV_SIDE_RE.search(compact)
            if not m:
                continue
            ratio_text = compact[: m.start()]
            equiv_text = compact[m.start() :]
        else:
            ratio_text = raw[:idx].strip()
            equiv_text = raw[idx:].strip()
        if not ratio_text or not equiv_text:
            continue
        if "双键" in equiv_text:
            empty["text"] = "双键当量[g/mol]"
        else:
            empty["text"] = "酸当量[g/mol]"
        empty["texts"] = []
        cell["text"] = ratio_text
        cell["texts"] = []
        changed += 1
        logger.info("split_glued_side_headers: 拆出当量到 col %d", empty["col_start"])
    if changed:
        logger.info("split_glued_side_headers: 拆分 %d 处", changed)
        work = dedupe_overlapping_cells(work)
    return work


def refill_short_side_headers_from_ocr(
    cells: List[Dict[str, Any]],
    boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """侧栏头过短时，用 cell polygon（略扩）内 OCR 碎片并集补全。"""
    if not cells or not boxes:
        return cells
    work = [dict(c) for c in cells]
    changed = 0
    for cell in work:
        t = re.sub(r"\s+", "", str(cell.get("text") or ""))
        if not t or not _MONOMER_RIGHT_ANCHOR_RE.search(t):
            continue
        # 仅补「来源于/来自具有…」短比率头；勿动完整酸/双键当量列
        if not _SHORT_SIDE_HEADER_RE.match(t):
            continue
        x1, y1, x2, y2 = _cell_bbox(cell)
        # 短头往往 OCR 框偏出 cell，横向加宽、纵向略扩（勿过大以免吞邻列单位）
        pad_x = max(24.0, 0.35 * max(x2 - x1, 20.0))
        pad_y = max(10.0, 0.28 * max(y2 - y1, 20.0))
        region = (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)
        frags = []
        for tb in boxes:
            tx1, ty1, tx2, ty2 = _tb_bbox(tb)
            cx = 0.5 * (tx1 + tx2)
            cy = 0.5 * (ty1 + ty2)
            if region[0] <= cx <= region[2] and region[1] <= cy <= region[3]:
                frag = str(tb.get("text") or "").strip()
                if not frag:
                    continue
                fc = re.sub(r"\s+", "", frag)
                if _looks_like_body_value(fc) or re.fullmatch(r"\d+(?:\.\d+)?", fc):
                    continue
                # 裸单位 [g/mol] 属当量列，勿并进短比率头
                if re.fullmatch(r"[\[［]?g\s*/\s*mo[l1][\]］]?", fc, re.I):
                    continue
                # 只收比率/来源类碎片，勿把邻列当量或表体拼进来
                # OCR 常把 mol 识成 mo1；允许 [mo1%]/[mol%]
                if not (
                    _MONOMER_RIGHT_ANCHOR_RE.search(fc)
                    or re.search(
                        r"(结构单元|所占|含有|比率|比例|[\[［]mo[l1]\s*%|mo[l1]\s*%)",
                        fc,
                    )
                ):
                    continue
                # 纯当量碎片留给邻列
                if _EQUIV_SIDE_RE.search(fc) and not _RATIO_SIDE_RE.search(fc):
                    continue
                frags.append((ty1, tx1, frag))
        if not frags:
            continue
        frags.sort()
        # 丢掉纯数字/表体值碎片，避免把 350/330 吸进侧栏头
        clean = []
        for ty, tx, frag in frags:
            fc = re.sub(r"\s+", "", frag)
            if _looks_like_body_value(fc) or re.fullmatch(r"\d+(?:\.\d+)?", fc):
                continue
            if _SYNTHESIS_ROW_RE.search(fc):
                continue
            if re.fullmatch(r"[\[［]?g\s*/\s*mo[l1][\]］]?", fc, re.I):
                continue
            if _EQUIV_SIDE_RE.search(fc) and not _RATIO_SIDE_RE.search(fc):
                continue
            clean.append((ty, tx, frag))
        if not clean:
            continue
        merged = re.sub(r"\s+", "", "".join(f for _, _, f in clean))
        if len(merged) <= len(t):
            continue
        # 合并后若仍像「比率+数字」粘连，去掉尾部纯数字
        merged2 = re.sub(r"\d+(?:\.\d+)?$", "", merged)
        if len(merged2) >= 8:
            merged = merged2
            clean_text = "\n".join(f for _, _, f in clean)
            clean_text = re.sub(r"\n?\d+(?:\.\d+)?\s*$", "", clean_text)
        else:
            clean_text = "\n".join(f for _, _, f in clean)
        # 去掉误并的裸 [g/mol] 行
        lines = [
            ln
            for ln in clean_text.split("\n")
            if not re.fullmatch(
                r"\s*[\[［]?g\s*/\s*mo[l1][\]］]?\s*", ln or "", re.I
            )
        ]
        clean_text = "\n".join(lines).strip()
        if not clean_text or len(re.sub(r"\s+", "", clean_text)) <= len(t):
            continue
        cell["text"] = clean_text
        cell["texts"] = []
        changed += 1
        logger.info(
            "refill_short_side_headers_from_ocr: %s → %s",
            t[:20],
            re.sub(r"\s+", "", clean_text)[:40],
        )
    if changed:
        logger.info("refill_short_side_headers_from_ocr: 补全 %d 个侧栏头", changed)
    return work


def sanitize_side_header_body_leak(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """侧栏头尾部粘上的表体数字/短横杠去掉（酸当量[g/mol]350）。"""
    if not cells:
        return cells
    work = [dict(c) for c in cells]
    changed = 0
    for cell in work:
        raw = str(cell.get("text") or "")
        if not raw:
            continue
        compact = re.sub(r"\s+", "", raw)
        if not (
            _SIDE_HEADER_FORCE_RE.search(compact)
            or _MONOMER_RIGHT_ANCHOR_RE.search(compact)
        ):
            continue
        # 去掉尾部纯数字 / 单独短横
        cleaned = re.sub(r"(?:[\d.]+|[-—–−])+$", "", compact)
        if cleaned == compact or len(cleaned) < 4:
            continue
        # 尽量保留换行结构：从 raw 末尾剥数字
        new_raw = re.sub(r"(?:\s*[\d.]+|\s*[-—–−])+\s*$", "", raw).rstrip()
        if not new_raw:
            new_raw = cleaned
        cell["text"] = new_raw
        cell["texts"] = []
        changed += 1
    if changed:
        logger.info("sanitize_side_header_body_leak: 清理 %d 处", changed)
    return work


def fill_empty_child_headers_from_ocr(
    cells: List[Dict[str, Any]],
    boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """单体带子表头行空格：用水平对齐且落在表头带附近的 OCR 补文案（脂环式等掉出格外）。"""
    if not cells or not boxes:
        return cells
    work = [dict(c) for c in cells]
    row_seps, col_seps = _derive_seps(work)
    header_like = re.compile(
        r"(具有.{2,40}(?:共聚成分|化合物|衍生物|基团)|"
        r"四羧酸|二羧酸|二胺|封端剂|有机硅烷|烯键式|烯属)"
    )
    changed = 0
    parents = [
        c for c in work if _MONOMER_PARENT_RE.search(str(c.get("text") or ""))
    ]
    for parent in parents:
        prs, pre = int(parent["row_start"]), int(parent["row_end"])
        pcs, pce = int(parent["col_start"]), int(parent["col_end"])
        child_row = pre + 1
        y_lo = float(row_seps[prs]) if prs < len(row_seps) else 0.0
        y_hi = (
            float(row_seps[child_row + 1])
            if child_row + 1 < len(row_seps)
            else y_lo + 80.0
        )
        # 允许 OCR 略高出/低于子表头行
        y_pad = max(20.0, 0.35 * (y_hi - y_lo))
        for cell in work:
            if int(cell["row_start"]) != child_row == int(cell["row_end"]):
                continue
            if int(cell["col_end"]) < pcs or int(cell["col_start"]) > pce:
                continue
            if str(cell.get("text") or "").strip():
                continue
            x1, _y1, x2, _y2 = _cell_bbox(cell)
            pad_x = max(6.0, 0.15 * (x2 - x1))
            frags = []
            for tb in boxes:
                tx1, ty1, tx2, ty2 = _tb_bbox(tb)
                cx = 0.5 * (tx1 + tx2)
                cy = 0.5 * (ty1 + ty2)
                if not (x1 - pad_x <= cx <= x2 + pad_x):
                    continue
                if not (y_lo - y_pad <= cy <= y_hi + y_pad):
                    continue
                frag = str(tb.get("text") or "").strip()
                if not frag or not header_like.search(re.sub(r"\s+", "", frag)):
                    continue
                # 已落入其它有字格的 OCR 跳过
                taken = False
                for o in work:
                    if o is cell or not str(o.get("text") or "").strip():
                        continue
                    ox1, oy1, ox2, oy2 = _cell_bbox(o)
                    if ox1 <= cx <= ox2 and oy1 <= cy <= oy2:
                        taken = True
                        break
                if taken:
                    continue
                frags.append((ty1, tx1, frag))
            if not frags:
                continue
            frags.sort()
            cell["text"] = "\n".join(f for _, _, f in frags)
            cell["texts"] = []
            changed += 1
            logger.info(
                "fill_empty_child_headers_from_ocr: col %d ← %s",
                cell["col_start"],
                cell["text"][:30],
            )
    if changed:
        logger.info("fill_empty_child_headers_from_ocr: 补全 %d 个空子头", changed)

    # 子表头行在单体带内的空洞列：若有水平对齐的表头 OCR，则建格填入
    for parent in parents:
        prs, pre = int(parent["row_start"]), int(parent["row_end"])
        pcs, pce = int(parent["col_start"]), int(parent["col_end"])
        child_row = pre + 1
        if child_row + 1 >= len(row_seps) or pce + 1 >= len(col_seps):
            continue
        y1 = float(row_seps[child_row])
        y2 = float(row_seps[child_row + 1])
        y_pad = max(20.0, 0.35 * (y2 - y1))
        for col in range(pcs, pce + 1):
            covered = any(
                int(c["row_start"]) <= child_row <= int(c["row_end"])
                and int(c["col_start"]) <= col <= int(c["col_end"])
                and str(c.get("text") or "").strip()
                for c in work
            )
            if covered:
                continue
            if col + 1 >= len(col_seps):
                continue
            x1 = float(col_seps[col])
            x2 = float(col_seps[col + 1])
            pad_x = max(6.0, 0.15 * (x2 - x1))
            frags = []
            for tb in boxes:
                tx1, ty1, tx2, ty2 = _tb_bbox(tb)
                cx = 0.5 * (tx1 + tx2)
                cy = 0.5 * (ty1 + ty2)
                if not (x1 - pad_x <= cx <= x2 + pad_x):
                    continue
                if not (y1 - y_pad <= cy <= y2 + y_pad):
                    continue
                frag = str(tb.get("text") or "").strip()
                if not frag or not header_like.search(re.sub(r"\s+", "", frag)):
                    continue
                fc = re.sub(r"\s+", "", frag)
                if _looks_like_body_value(fc) or re.fullmatch(r"\d+(?:\.\d+)?", fc):
                    continue
                if _ENE_HEADER_RE.search(fc):
                    ene_exists = any(
                        _ENE_HEADER_RE.search(
                            re.sub(r"\s+", "", str(o.get("text") or ""))
                        )
                        for o in work
                    )
                    if ene_exists:
                        continue
                taken = False
                for o in work:
                    if not str(o.get("text") or "").strip():
                        continue
                    ox1, oy1, ox2, oy2 = _cell_bbox(o)
                    if ox1 <= cx <= ox2 and oy1 <= cy <= oy2:
                        taken = True
                        break
                if taken:
                    continue
                frags.append((ty1, tx1, frag))
            if not frags:
                continue
            frags.sort()
            slot = None
            for c in work:
                if (
                    int(c["row_start"]) == child_row == int(c["row_end"])
                    and int(c["col_start"]) == col == int(c["col_end"])
                    and not str(c.get("text") or "").strip()
                ):
                    slot = c
                    break
            if slot is None:
                slot = _make_empty_cell(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    row_start=child_row,
                    row_end=child_row,
                    col_start=col,
                    col_end=col,
                )
                work.append(slot)
            slot["text"] = "\n".join(f for _, _, f in frags)
            slot["texts"] = []
            slot["text"] = re.sub(
                r"(的?共聚成分)(?:[\s　]*的?共聚成分)+",
                r"\1",
                slot["text"],
            )
            changed += 1
            logger.info(
                "fill_empty_child_headers_from_ocr: 建格 col %d ← %s",
                col,
                slot["text"][:30],
            )
    if changed:
        work = dedupe_overlapping_cells(work)
    return work


def _copoly_child_cat_key(text: str) -> Optional[str]:
    """共聚/烯键子头类别键，用于截断补全时防串列。"""
    t = re.sub(r"\s+", "", text or "")
    if not t:
        return None
    if _ENE_HEADER_RE.search(t):
        return "ene"
    if "脂环" in t:
        return "ali"
    if "芳香" in t:
        return "aro"
    if "酸性" in t:
        return "acid"
    return None


def _monomer_child_header_incomplete(text: str) -> bool:
    """子表头是否明显截断（缺共聚成分/化合物等）。"""
    c = re.sub(r"\s+", "", text or "")
    if not c:
        return True
    if _ENE_HEADER_RE.search(c):
        return not re.search(r"(化合物|衍生物)", c)
    if any(k in c for k in ("酸性", "芳香", "脂环")):
        return "共聚成分" not in c
    if c.startswith("具有") and len(c) < 16:
        return True
    # 专利表常见双段堆叠：缺第二段化学类名时仍收同列 OCR
    if "二羧酸" in c and "二甲酰" not in c:
        return True
    if "双氨基酚" in c and "二羟基" not in c:
        return True
    if "有机硅烷" in c and "低聚" not in c and "三官能" not in c:
        # 「四官能有机硅烷」常另有「…低聚物」行；三官能通常单段
        return True
    return False


_CHILD_HEADER_FRAG_RE = re.compile(
    r"(具有.{1,40}(?:共聚成分|化合物|衍生物|基团)|"
    r"的?共聚成分|双键基团|的?不饱和化合物|"
    r"四羧酸|二羧酸|二胺|封端剂|有机硅烷|烯键式|烯属|"
    r"二甲酰|二羟基|低聚物|及其衍生物|衍生物)"
)


def refill_truncated_monomer_child_headers_from_ocr(
    cells: List[Dict[str, Any]],
    boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """过窄子头格：OCR 常落在格左侧。按邻列中点墙在表头带内收碎片补全。"""
    if not cells or not boxes:
        return cells
    work = [dict(c) for c in cells]
    row_seps, col_seps = _derive_seps(work)
    if len(row_seps) < 3:
        return work
    changed = 0
    parents = [
        c for c in work if _MONOMER_PARENT_RE.search(str(c.get("text") or ""))
    ]
    for parent in parents:
        prs, pre = int(parent["row_start"]), int(parent["row_end"])
        pcs, pce = int(parent["col_start"]), int(parent["col_end"])
        child_row = pre + 1
        if child_row + 1 >= len(row_seps):
            continue
        px1, _py1, px2, _py2 = _cell_bbox(parent)
        y_lo = float(row_seps[prs])
        y_hi = float(row_seps[child_row + 1])
        y_pad = max(16.0, 0.25 * (y_hi - y_lo))
        kids = [
            c
            for c in work
            if int(c["row_start"]) == child_row == int(c["row_end"])
            and int(c["col_end"]) >= pcs
            and int(c["col_start"]) <= pce
            and str(c.get("text") or "").strip()
            and not _looks_like_body_value(str(c.get("text") or ""))
            and not _SIDE_HEADER_FORCE_RE.search(_compact_cell_text(c))
            and not _MONOMER_RIGHT_ANCHOR_RE.search(_compact_cell_text(c))
        ]
        if len(kids) < 2:
            continue
        kids = sorted(kids, key=lambda c: (int(c["col_start"]), _cell_bbox(c)[0]))
        for i, kid in enumerate(kids):
            raw = str(kid.get("text") or "")
            if not _monomer_child_header_incomplete(raw):
                continue
            kx1, _ky1, kx2, _ky2 = _cell_bbox(kid)
            if i > 0:
                prev_x2 = _cell_bbox(kids[i - 1])[2]
                wall_lo = 0.5 * (prev_x2 + kx1)
            else:
                wall_lo = min(px1, kx1) - 20.0
            if i + 1 < len(kids):
                next_x1 = _cell_bbox(kids[i + 1])[0]
                wall_hi = 0.5 * (kx2 + next_x1)
            else:
                wall_hi = max(px2, kx2) + 20.0
            if wall_hi <= wall_lo + 4.0:
                continue
            kid_cat = _copoly_child_cat_key(raw)
            frags: List[Tuple[float, float, str]] = []
            for tb in boxes:
                tx1, ty1, tx2, ty2 = _tb_bbox(tb)
                cx = 0.5 * (tx1 + tx2)
                cy = 0.5 * (ty1 + ty2)
                if not (wall_lo <= cx <= wall_hi):
                    continue
                if not (y_lo - y_pad <= cy <= y_hi + y_pad):
                    continue
                frag = str(tb.get("text") or "").strip()
                if not frag:
                    continue
                fc = re.sub(r"\s+", "", frag)
                if _looks_like_body_value(fc) or re.fullmatch(r"\d+(?:\.\d+)?", fc):
                    continue
                if not _CHILD_HEADER_FRAG_RE.search(fc):
                    continue
                frag_cat = _copoly_child_cat_key(fc)
                if kid_cat and frag_cat and frag_cat != kid_cat:
                    continue
                # 无类别碎片（如「的共聚成分」）仅在本墙内收
                frags.append((ty1, tx1, frag))
            if not frags:
                continue
            frags.sort()
            merged = "\n".join(f for _, _, f in frags)
            merged = re.sub(
                r"(的?共聚成分)(?:[\s　]*的?共聚成分)+",
                r"\1",
                merged,
            )
            mc = re.sub(r"\s+", "", merged)
            tc = re.sub(r"\s+", "", raw)
            if len(mc) <= len(tc):
                continue
            # 补全后仍须属同类，避免串成烯键式
            if kid_cat and _copoly_child_cat_key(merged) not in (None, kid_cat):
                continue
            kid["text"] = merged
            kid["texts"] = []
            changed += 1
            logger.info(
                "refill_truncated_monomer_child_headers_from_ocr: %s → %s",
                tc[:18],
                mc[:40],
            )
    if changed:
        logger.info(
            "refill_truncated_monomer_child_headers_from_ocr: 补全 %d 处",
            changed,
        )
    return work


def absorb_empty_child_header_gaps(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """单体带子表头行：有字子头向两侧吸收空槽，铺满父格列带。"""
    if not cells:
        return cells
    work = [dict(c) for c in cells]
    row_seps, col_seps = _derive_seps(work)
    changed = 0
    removed: set = set()
    parents = [
        c for c in work if _MONOMER_PARENT_RE.search(str(c.get("text") or ""))
    ]
    for parent in parents:
        pre = int(parent["row_end"])
        pcs, pce = int(parent["col_start"]), int(parent["col_end"])
        child_row = pre + 1
        kids = [
            c
            for c in work
            if id(c) not in removed
            and int(c["row_start"]) == child_row == int(c["row_end"])
            and int(c["col_end"]) >= pcs
            and int(c["col_start"]) <= pce
            and str(c.get("text") or "").strip()
            and not _looks_like_body_value(str(c.get("text") or ""))
            and not _SIDE_HEADER_FORCE_RE.search(_compact_cell_text(c))
        ]
        if len(kids) < 2:
            continue
        kids = sorted(kids, key=lambda c: int(c["col_start"]))
        # 夹在两有字子头之间的空格并入较近一侧
        empties = [
            c
            for c in work
            if id(c) not in removed
            and int(c["row_start"]) == child_row == int(c["row_end"])
            and pcs <= int(c["col_start"]) <= pce
            and not str(c.get("text") or "").strip()
        ]
        for emp in empties:
            ecs, ece = int(emp["col_start"]), int(emp["col_end"])
            left = [k for k in kids if int(k["col_end"]) < ecs]
            right = [k for k in kids if int(k["col_start"]) > ece]
            target = None
            if left and right:
                # 较近的一侧
                if ecs - int(left[-1]["col_end"]) <= int(right[0]["col_start"]) - ece:
                    target = left[-1]
                else:
                    target = right[0]
            elif left:
                target = left[-1]
            elif right:
                target = right[0]
            if target is None:
                continue
            ncs = min(int(target["col_start"]), ecs)
            nce = max(int(target["col_end"]), ece)
            _set_cell_cols(target, ncs, nce, col_seps)
            removed.add(id(emp))
            changed += 1
    if removed:
        work = [c for c in work if id(c) not in removed]
    if changed:
        logger.info("absorb_empty_child_header_gaps: 吸收 %d 处空槽", changed)
        work = dedupe_overlapping_cells(work)
    return work


def sanitize_side_header_texts(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """侧栏头尾部粘上的表体数字/化学代号剥掉。"""
    if not cells:
        return cells
    work = [dict(c) for c in cells]
    changed = 0
    for cell in work:
        raw = str(cell.get("text") or "")
        if not raw.strip():
            continue
        compact = re.sub(r"\s+", "", raw)
        is_side = bool(
            _SIDE_HEADER_FORCE_RE.search(compact)
            or _MONOMER_RIGHT_ANCHOR_RE.search(compact)
        )
        is_copoly_child = bool(
            re.search(r"(脂环式|芳香族|酸性基团|共聚成分)", compact)
        )
        if not is_side and not is_copoly_child:
            continue
        if is_side:
            lines = [ln for ln in raw.split("\n") if ln.strip()]
            while lines:
                last = re.sub(r"\s+", "", lines[-1])
                if _looks_like_body_value(last) or re.fullmatch(
                    r"\d+(?:\.\d+)?", last
                ):
                    lines.pop()
                    changed += 1
                    continue
                break
            new_text = "\n".join(lines)
            # 同行粘连：酸当量[g/mol]350
            new_text2 = re.sub(
                r"((?:酸当量|双键当量)\s*\[[^\]]+\])\s*(?:\d+(?:\.\d+)?|[-—–−])\s*$",
                r"\1",
                new_text,
            )
            if new_text2 != raw:
                cell["text"] = new_text2
                cell["texts"] = []
                changed += 1
            elif new_text != raw:
                cell["text"] = new_text
                cell["texts"] = []
        # 脂环式格误吞烯键式后半段 → 只保留脂环式共聚成分
        raw2 = str(cell.get("text") or "")
        c2 = re.sub(r"\s+", "", raw2)
        if "脂环式" in c2 and (
            "烯键式" in c2
            or "双键基团" in c2
            or "不饱和双键" in c2
            or "不饱和化合物" in c2
        ):
            cut = re.split(
                r"(?=双键基团|烯键式|不饱和双键|不饱和化合物)",
                raw2,
                maxsplit=1,
            )[0].strip()
            if "脂环式" in cut and len(re.sub(r"\s+", "", cut)) >= 6:
                cell["text"] = cut
                cell["texts"] = []
                changed += 1
        # 重复「的共聚成分」
        raw3 = str(cell.get("text") or "")
        deduped = re.sub(
            r"(的?共聚成分)(?:[\s　]*的?共聚成分)+",
            r"\1",
            raw3,
        )
        if deduped != raw3:
            cell["text"] = deduped
            cell["texts"] = []
            changed += 1
        # 来源于…粘酸当量且右侧已有独立酸当量 → 只剥当量
        if not is_side:
            continue
        raw4 = str(cell.get("text") or "")
        c4 = re.sub(r"\s+", "", raw4)
        if _RATIO_SIDE_RE.search(c4) and _EQUIV_SIDE_RE.search(c4):
            rs, re_ = int(cell["row_start"]), int(cell["row_end"])
            ce = int(cell["col_end"])
            has_pure_equiv = any(
                _EQUIV_SIDE_RE.search(re.sub(r"\s+", "", str(o.get("text") or "")))
                and not _RATIO_SIDE_RE.search(
                    re.sub(r"\s+", "", str(o.get("text") or ""))
                )
                and int(o["col_start"]) > ce
                and not (int(o["row_start"]) > re_ or int(o["row_end"]) < rs)
                for o in work
                if o is not cell
            )
            if has_pure_equiv:
                stripped = re.split(r"酸当量|双键当量", raw4, maxsplit=1)[0].strip()
                if stripped and stripped != raw4:
                    cell["text"] = stripped
                    cell["texts"] = []
                    changed += 1
    # 同子表头行：烯键式若列序在脂环式左侧，交换列范围（P47）
    row_seps, col_seps = _derive_seps(work)
    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for c in work:
        if int(c["row_start"]) != int(c["row_end"]):
            continue
        by_row.setdefault(int(c["row_start"]), []).append(c)
    for _r, row_cells in by_row.items():
        ene = [
            c
            for c in row_cells
            if _ENE_HEADER_RE.search(re.sub(r"\s+", "", str(c.get("text") or "")))
        ]
        ali = [
            c
            for c in row_cells
            if "脂环式" in re.sub(r"\s+", "", str(c.get("text") or ""))
        ]
        if len(ene) == 1 and len(ali) == 1:
            e, a = ene[0], ali[0]
            if int(e["col_start"]) < int(a["col_start"]):
                ecs, ece = int(e["col_start"]), int(e["col_end"])
                acs, ace = int(a["col_start"]), int(a["col_end"])
                _set_cell_cols(e, acs, ace, col_seps)
                _set_cell_cols(a, ecs, ece, col_seps)
                changed += 1
                logger.info(
                    "sanitize_side_header_texts: 交换脂环式/烯键式列 %d-%d ↔ %d-%d",
                    ecs,
                    ece,
                    acs,
                    ace,
                )
    if changed:
        logger.info("sanitize_side_header_texts: 清理 %d 处", changed)
    return work


def split_left_anchor_subheader_leak(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """左锚(聚合物)文本吞入子头词 → 拆到右侧空/过窄子头槽。"""
    if not cells:
        return cells
    work = [dict(c) for c in cells]
    row_seps, col_seps = _derive_seps(work)
    changed = 0
    for cell in list(work):
        raw = str(cell.get("text") or "")
        if not _MONOMER_LEFT_ANCHOR_RE.search(raw):
            continue
        if not _SUBHEADER_LEAK_RE.search(raw):
            continue
        # 拆出泄漏子串（聚合物后紧跟的子头词）
        m = _SUBHEADER_LEAK_RE.search(raw)
        if not m:
            continue
        # 「聚合物」本身若被吃进，保留到泄漏起点
        poly_m = re.search(r"聚合物", raw)
        if poly_m and poly_m.end() <= m.start():
            keep = raw[: m.start()].strip() or "聚合物"
            leak = raw[m.start() :].strip()
        else:
            leak = raw[m.start() :].strip()
            keep = "聚合物"
        if not leak or leak == keep:
            continue
        rs, re_ = int(cell["row_start"]), int(cell["row_end"])
        ce = int(cell["col_end"])
        # 优先：子表头行上、紧邻右侧的空格
        target = None
        child_row = re_ if re_ == rs else (rs + 1 if re_ > rs else rs)
        # 聚合物常 rowspan 盖父+子；泄漏应落到子表头行
        for o in work:
            if o is cell:
                continue
            if int(o["row_start"]) != child_row and not (
                int(o["row_start"]) <= child_row <= int(o["row_end"])
            ):
                if int(o["row_start"]) != child_row == int(o["row_end"]):
                    continue
            if int(o["row_start"]) == child_row == int(o["row_end"]):
                if int(o["col_start"]) <= ce:
                    continue
                if int(o["col_start"]) > ce + 3:
                    continue
                ot = str(o.get("text") or "").strip()
                if ot and not _MONOMER_PARENT_RE.search(ot):
                    # 已有子头则跳过，找空槽
                    if _SUBHEADER_LEAK_RE.search(ot) or re.search(
                        r"[\u4e00-\u9fff]", ot
                    ):
                        continue
                target = o
                break
        if target is None:
            for o in work:
                if int(o["row_start"]) != child_row == int(o["row_end"]):
                    continue
                if int(o["col_start"]) <= ce:
                    continue
                if str(o.get("text") or "").strip():
                    continue
                target = o
                break
        if target is None:
            continue
        cell["text"] = keep
        cell["texts"] = []
        target["text"] = leak
        target["texts"] = []
        # 若聚合物 rowspan 盖住子行，把目标行钉在子表头行
        if int(target["row_start"]) != child_row:
            _set_cell_rows(target, child_row, child_row, row_seps)
        changed += 1
        logger.info(
            "split_left_anchor_subheader_leak: 聚合物泄漏 → col %d (%s)",
            target["col_start"],
            leak[:20],
        )
    if changed:
        logger.info("split_left_anchor_subheader_leak: 修复 %d 处", changed)
    return work


def split_undersplit_monomer_child_headers(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """子头格数 < 表体化学列数，且存在粘连中文 colspan>1 → 按表体列 x 切开。"""
    if not cells:
        return cells
    work = [dict(c) for c in cells]
    row_seps, col_seps = _derive_seps(work)
    if len(col_seps) < 4:
        return work

    parents = [
        c for c in work if _MONOMER_PARENT_RE.search(str(c.get("text") or ""))
    ]
    changed = 0
    for parent in parents:
        prs, pre = int(parent["row_start"]), int(parent["row_end"])
        pcs, pce = int(parent["col_start"]), int(parent["col_end"])
        child_row = pre + 1
        kids = [
            c
            for c in work
            if int(c["row_start"]) == child_row == int(c["row_end"])
            and int(c["col_end"]) >= pcs
            and int(c["col_start"]) <= pce
            and str(c.get("text") or "").strip()
            and not _looks_like_body_value(str(c.get("text") or ""))
            and not _SIDE_HEADER_FORCE_RE.search(_compact_cell_text(c))
        ]
        body = [
            c
            for c in work
            if int(c["row_start"]) > child_row
            and pcs <= int(c["col_start"]) <= pce
            and _looks_like_body_value(str(c.get("text") or ""))
        ]
        # 按原子列计数
        body_cols = sorted(
            {
                col
                for b in body
                for col in range(int(b["col_start"]), int(b["col_end"]) + 1)
                if pcs <= col <= pce
            }
        )
        if len(kids) < 2 or len(body_cols) <= len(kids):
            continue
        glued = [k for k in kids if int(k["col_span"]) > 1]
        if not glued:
            continue
        # 对每个粘连格：若文本含 ≥2 个类别词，按覆盖的 body 列均分/按 x 切
        for gk in glued:
            text = str(gk.get("text") or "")
            cats = set()
            for part in re.split(r"[\s/／]+", text):
                cats.add(_child_header_category(part))
            cats.discard("other")
            if len(cats) < 2 and not re.search(
                r"(共聚成分|衍生物|化合物).{0,6}(共聚成分|衍生物|化合物|不饱和)",
                re.sub(r"\s+", "", text),
            ):
                # 粘连：两个「具有…」片段
                if text.count("具有") < 2 and text.count("共聚") < 2:
                    continue
            gcs, gce = int(gk["col_start"]), int(gk["col_end"])
            cover = [c for c in body_cols if gcs <= c <= gce]
            if len(cover) < 2:
                continue
            # 仅当文本含 ≥2 个「具有…」独立片段才切开（勿按换行切开同一子头）
            parts = [p.strip() for p in re.split(r"(?=具有)", text) if p.strip()]
            if len(parts) < 2:
                continue
            # 各类别须能区分，避免把「具有A具有B」式合法单头误切
            part_cats = {_child_header_category(p) for p in parts}
            part_cats.discard("other")
            if len(part_cats) < 2 and text.count("共聚成分") < 2:
                continue
            n = min(len(cover), len(parts))
            _set_cell_cols(gk, cover[0], cover[0], col_seps)
            gk["text"] = parts[0]
            gk["texts"] = []
            for i in range(1, n):
                col = cover[i]
                slot = None
                for o in work:
                    if int(o["row_start"]) == child_row == int(o["row_end"]) and int(
                        o["col_start"]
                    ) == col == int(o["col_end"]):
                        slot = o
                        break
                if slot is None:
                    x1 = float(col_seps[col]) if col < len(col_seps) else 0
                    x2 = float(col_seps[col + 1]) if col + 1 < len(col_seps) else x1 + 10
                    _, y1, _, y2 = _cell_bbox(gk)
                    slot = _make_empty_cell(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        row_start=child_row,
                        row_end=child_row,
                        col_start=col,
                        col_end=col,
                    )
                    work.append(slot)
                if not str(slot.get("text") or "").strip():
                    slot["text"] = parts[i] if i < len(parts) else parts[-1]
                    slot["texts"] = []
                    _set_cell_cols(slot, col, col, col_seps)
            changed += 1
            logger.info(
                "split_undersplit_monomer_child_headers: 切开粘连子头 → %d 列",
                n,
            )
    if changed:
        work = dedupe_overlapping_cells(work)
    return work


def merge_stacked_chem_amount_cells(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """合并同列上下拆开的化学代号与括号用量：STR + (30) → STR\n(30)。

    仅处理表体原子列；中间可夹空逻辑行。避免把已含用量的格再次拼接。
    """
    if len(cells) < 2:
        return cells

    work = [dict(c) for c in cells]
    by_col: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for c in work:
        if int(c["col_start"]) != int(c["col_end"]):
            continue
        by_col[int(c["col_start"])].append(c)

    dropped: set = set()
    changed = 0
    for _col, group in by_col.items():
        group = sorted(group, key=lambda c: (int(c["row_start"]), int(c["row_end"])))
        i = 0
        while i < len(group):
            upper = group[i]
            if id(upper) in dropped:
                i += 1
                continue
            ut = re.sub(r"\s+", "", str(upper.get("text") or ""))
            if not ut or _PAREN_AMOUNT_RE.fullmatch(ut):
                i += 1
                continue
            if _CHEM_BODY_RE.fullmatch(ut) and _PAREN_AMOUNT_RE.search(ut):
                i += 1
                continue
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9\-\+']{1,24}", ut):
                i += 1
                continue
            lower = None
            for j in range(i + 1, len(group)):
                cand = group[j]
                if id(cand) in dropped:
                    continue
                ct = re.sub(r"\s+", "", str(cand.get("text") or ""))
                if not ct:
                    continue
                if int(cand["row_start"]) > int(upper["row_end"]) + 3:
                    break
                if _PAREN_AMOUNT_RE.fullmatch(ct):
                    lower = cand
                break
            if lower is None:
                i += 1
                continue
            lt = str(lower.get("text") or "").strip()
            prev = str(upper.get("text") or "").strip()
            upper["text"] = f"{prev}\n{lt}" if prev else lt
            upper["row_end"] = max(int(upper["row_end"]), int(lower["row_end"]))
            _refresh_spans(upper)
            ux1, uy1, ux2, uy2 = _cell_bbox(upper)
            lx1, ly1, lx2, ly2 = _cell_bbox(lower)
            upper["polygon"] = _rebuild_polygon(
                min(ux1, lx1), min(uy1, ly1), max(ux2, lx2), max(uy2, ly2)
            )
            dropped.add(id(lower))
            changed += 1
            i += 1

    if not changed:
        return cells
    out = [c for c in work if id(c) not in dropped]
    logger.info("合并化学代号+用量竖拆格: %d", changed)
    return dedupe_overlapping_cells(out)



_SYNTHESIS_LABEL_RE = re.compile(
    r"(合成例|实施例|実施例|比較例|比较例|对照例|参考例)\s*\d*"
)


def _set_logic_rows(cell: Dict[str, Any], rs: int, re: int) -> None:
    """只改逻辑行号，保留物理多边形（避免用错 row_seps 扭曲几何）。"""
    rs, re = int(rs), int(re)
    if re < rs:
        return
    cell["row_start"] = rs
    cell["row_end"] = re
    _refresh_spans(cell)


def normalize_oversegmented_table_rows(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """压缩 TSR 过切行：表头压成 2 级，合成例与同行数据对齐为单行。

    针对「聚合物 / 单体[…] / 合成例」专利表：去掉表头空行、把跨行
    代号+用量格收回单行，并把错位的合成例标签拉到数据行起点。
    """
    if len(cells) < 3:
        return cells

    joined = " ".join(str(c.get("text") or "") for c in cells)
    if not (
        _MONOMER_PARENT_RE.search(joined)
        and _MONOMER_LEFT_ANCHOR_RE.search(joined)
        and _SYNTHESIS_LABEL_RE.search(joined)
    ):
        return cells

    from ..utils.segments import _HEADER_CAPTION_RE, find_row_segments

    work = [dict(c) for c in cells]
    segments = find_row_segments(work)
    seg_in = len(segments) if segments else 1
    if not segments:
        min_r = min(int(c["row_start"]) for c in work)
        max_r = max(int(c["row_end"]) for c in work)
        segments = [(min_r, max_r)]
        seg_in = 1

    rebuilt_segs: List[List[Dict[str, Any]]] = []
    changed = 0

    for seg_lo, seg_hi in segments:
        seg = [c for c in work if seg_lo <= int(c["row_start"]) <= seg_hi]
        if not seg:
            continue

        # 1) 合成例与右侧近邻数据同行对齐，并压成单行（勿把上方表头当 peer）
        labels = [
            c
            for c in seg
            if _SYNTHESIS_LABEL_RE.search(str(c.get("text") or ""))
        ]
        # 其它合成例所在行：禁止跨行互拉（P98：跳过侧栏表头 peer 后，邻行数据会把 13/14 并到一行）
        synth_rows = {int(c["row_start"]) for c in labels}
        for lab in labels:
            lr = int(lab["row_start"])
            peers = []
            for c in seg:
                if c is lab:
                    continue
                t = str(c.get("text") or "").strip()
                if not t:
                    continue
                if int(c["col_start"]) <= int(lab["col_end"]):
                    continue
                # 必须起点落在合成例附近，排除上方跨很多行的表头格
                crs = int(c["row_start"])
                if abs(crs - lr) > 2:
                    continue
                # 绝不能把另一条合成例行上的格当成 peer
                if crs in synth_rows and crs != lr:
                    continue
                if _MONOMER_PARENT_RE.search(t) or _MONOMER_LEFT_ANCHOR_RE.search(t):
                    continue
                # 去空白再判：OCR 常把「双键当量」拆成「双键当\n量」导致漏跳过（P98）
                t_compact = re.sub(r"\s+", "", t)
                if re.search(
                    r"(衍生物|化合物|共聚成分|有机硅烷|封端剂|低聚物|"
                    r"酸当量|双键当量|四羧酸|二羧酸酐|含有比率|含有率|"
                    r"来自具有|来源于具有|烯键式|烯属不饱和)",
                    t_compact,
                ) and not _CHEM_BODY_RE.fullmatch(t_compact):
                    # 中文子表头 / 侧栏整列表头，不可当合成例 peer
                    continue
                peers.append(c)
            if not peers:
                if int(lab["row_end"]) != int(lab["row_start"]):
                    _set_logic_rows(lab, lr, lr)
                    changed += 1
                continue
            target = min([lr] + [int(c["row_start"]) for c in peers])
            # P93：禁止把合成例+化学格拽到仍含「聚合物/单体」的表头行
            headerish = any(
                (
                    _MONOMER_PARENT_RE.search(str(c.get("text") or ""))
                    or _MONOMER_LEFT_ANCHOR_RE.search(str(c.get("text") or ""))
                )
                and int(c["row_start"]) <= target <= int(c["row_end"])
                for c in seg
            )
            if headerish:
                target = lr
            for c in [lab, *peers]:
                if int(c["row_start"]) != target or int(c["row_end"]) != target:
                    _set_logic_rows(c, target, target)
                    changed += 1

        # 2) 划分表头 / 表体：以合成例行为表体起点
        body_starts = sorted(
            {
                int(c["row_start"])
                for c in seg
                if _SYNTHESIS_LABEL_RE.search(str(c.get("text") or ""))
            }
        )
        if not body_starts:
            rebuilt_segs.append(seg)
            continue
        first_body = body_starts[0]

        header_cells = [
            c
            for c in seg
            if int(c["row_start"]) < first_body and str(c.get("text") or "").strip()
        ]
        body_cells = [
            c
            for c in seg
            if int(c["row_start"]) >= first_body and str(c.get("text") or "").strip()
        ]
        empty_header = [
            c
            for c in seg
            if int(c["row_start"]) < first_body
            and not str(c.get("text") or "").strip()
        ]
        if empty_header:
            changed += len(empty_header)

        parents = []
        children = []
        for c in header_cells:
            t = str(c.get("text") or "").strip()
            # 表题不应进入子表头行（会触发 find_row_segments 按 caption 切开）
            if _HEADER_CAPTION_RE.search(t):
                continue
            if _MONOMER_LEFT_ANCHOR_RE.search(t):
                parents.append(("polymer", c))
            elif _MONOMER_PARENT_RE.search(t):
                parents.append(("monomer", c))
            else:
                children.append(c)

        local: List[Dict[str, Any]] = []

        for kind, c in parents:
            nc = dict(c)
            if kind == "polymer":
                _set_logic_rows(nc, 0, 1 if children else 0)
            else:
                _set_logic_rows(nc, 0, 0)
            local.append(nc)
            changed += 1

        for c in children:
            nc = dict(c)
            _set_logic_rows(nc, 1, 1)
            local.append(nc)
            changed += 1

        # 表体按当前 row_start 分组，压成连续局部行
        body_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for c in body_cells:
            body_groups[int(c["row_start"])].append(c)

        local_r = 2 if (parents and children) else (1 if parents or children else 0)
        for old_r in sorted(body_groups.keys()):
            for c in body_groups[old_r]:
                nc = dict(c)
                _set_logic_rows(nc, local_r, local_r)
                local.append(nc)
                changed += 1
            local_r += 1

        rebuilt_segs.append(local)

    if not changed:
        return cells

    # 3) 段拼接为全局连续行号
    final: List[Dict[str, Any]] = []
    global_row = 0
    for seg_cells in rebuilt_segs:
        if not seg_cells:
            continue
        max_local = max(int(c["row_end"]) for c in seg_cells)
        for c in seg_cells:
            nc = dict(c)
            _set_logic_rows(
                nc,
                global_row + int(c["row_start"]),
                global_row + int(c["row_end"]),
            )
            final.append(nc)
        global_row += max_local + 1

    if not final:
        return cells

    out = dedupe_overlapping_cells(final)
    # 误伤回退：P96/P97/P98 等多段合成例表会被压成残缺表头+丢行；
    # 非空格大幅下降时保留原文归属结果。
    def _ne(cs: List[Dict[str, Any]]) -> int:
        return sum(1 for c in cs if str(c.get("text") or "").strip())

    ne_in, ne_out = _ne(cells), _ne(out)
    synth_in = sum(
        1 for c in cells if _SYNTHESIS_LABEL_RE.search(str(c.get("text") or ""))
    )
    synth_out = sum(
        1 for c in out if _SYNTHESIS_LABEL_RE.search(str(c.get("text") or ""))
    )

    def _pure_num_cells(cs: List[Dict[str, Any]]) -> int:
        n = 0
        for c in cs:
            t = re.sub(r"\s+", "", str(c.get("text") or ""))
            if re.fullmatch(r"\d+(?:\.\d+)?", t):
                n += 1
        return n

    num_in, num_out = _pure_num_cells(cells), _pure_num_cells(out)
    seg_out = len(find_row_segments(out) or []) or 1
    # 非空格/合成例/纯数字格/分段掉太多 → 回退
    if ne_in >= 12 and (
        ne_out <= max(8, int(ne_in * 0.70))
        or len(out) < max(12, int(len(cells) * 0.50))
        or (synth_in >= 2 and synth_out < synth_in)
        or (num_in >= 4 and num_out < max(2, num_in - 1))
        or (seg_in >= 2 and seg_out < seg_in)
    ):
        logger.info(
            "normalize_oversegmented_table_rows: 回退 %d→%d cells "
            "(ne %d→%d synth %d→%d num %d→%d seg %d→%d)",
            len(cells),
            len(out),
            ne_in,
            ne_out,
            synth_in,
            synth_out,
            num_in,
            num_out,
            seg_in,
            seg_out,
        )
        return cells

    logger.info(
        "normalize_oversegmented_table_rows: %d → %d cells",
        len(cells),
        len(out),
    )
    return out


def refine_tsr_cells(
    cells: List[Dict[str, Any]],
    boxes: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """激进 TSR 结构后处理：幽灵列合并、拆 rowspan/colspan、双簇补列。"""
    if not cells:
        return cells
    from .row_header import clip_narrow_label_colspans, clip_row_header_child_overlaps

    cells = clip_row_header_child_overlaps(cells)
    cells = clip_narrow_label_colspans(cells)
    cells = dedupe_overlapping_cells(cells)
    cells = merge_ghost_columns(cells, boxes)
    cells = dedupe_overlapping_cells(cells)
    cells = merge_ghost_rows(cells, boxes)
    cells = dedupe_overlapping_cells(cells)
    cells = unmerge_bad_rowspans(cells, boxes)
    cells = split_underspanned_rows(cells, boxes)
    cells = split_bad_colspans(cells, boxes)
    cells = dedupe_overlapping_cells(cells)
    
    # ======= 此前漏掉的修复：多列表头拆分 =======
    # 将 TSR 误判合并的多列表头（colspan>=2），根据文本的 X 轴聚类拆分为多个独立的单元格
    cells = reconstruct_header_cells(cells, boxes)
    # ============================================

    return cells