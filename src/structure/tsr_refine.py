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


def merge_ghost_columns(
    cells: List[Dict[str, Any]],
    boxes: Sequence[Dict[str, Any]],
    *,
    min_width_ratio: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    几乎无文本落入且物理宽度异常窄的逻辑列 → 合并到右侧邻列。
    窄列若仅有短碎片 OCR（而邻列有实质文本）也视为幽灵列。
    """
    if not cells:
        return cells
    row_seps, col_seps = _derive_seps(cells)
    if len(col_seps) < 3:
        return cells

    n_cols = len(col_seps) - 1
    widths = [col_seps[i + 1] - col_seps[i] for i in range(n_cols)]
    median_w = float(np.median(widths)) if widths else 1.0

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

    # 标记幽灵列：窄 + 无文本，或窄 + 仅碎片且邻列有实质文本
    ghost = []
    for c in range(n_cols):
        # 【修正】若完全无文本命中，无视宽度直接判定为幽灵列
        if hits[c] == 0:
            ghost.append(True)
            continue
            
        narrow = widths[c] < min_width_ratio * median_w
        if not narrow:
            ghost.append(False)
            continue
        # 整列多为行序号：保留（与 html drop_evidenceless 一致）
        if is_index_column(col_texts[c]):
            ghost.append(False)
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
        # survivors[i] 是旧列号；左边界 = col_seps[survivors[new_cs]]
        left = float(col_seps[survivors[new_cs]])
        right_old = survivors[new_ce]
        right = float(col_seps[right_old + 1])
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
    """列/行/格数相对 OCR 过多 → 过切（后处理易塌缩）。"""
    if not cells or not text_boxes:
        return False
    n_boxes = len(text_boxes)
    n_cols, n_rows, n = cell_grid_stats(cells)
    if n_cols >= 32:
        return True
    if n_rows >= 32 and n_rows > max(16, n_boxes // 6):
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

                children_already_exist = (
                    (
                        lower_total > 0
                        and lower_owned >= max(1, int(0.5 * lower_total + 0.5))
                    )
                    or next_is_subheader
                )

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
_MONOMER_RIGHT_ANCHOR_RE = re.compile(
    r"(含有比率|酸当量|双键当量|含有率)"
)


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


def repair_monomer_parent_spans(
    cells: List[Dict[str, Any]],
    boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    按子表分段，把过窄/错位的「单体[…]」父格对齐到同段单体子列并集。

    只改已有格子的 col_start/col_end 与多边形，不插列、不融合框线。
    用于轻量 TSR 路径修复 P98 类分段异形表。
    """
    if not cells:
        return cells
    boxes = list(boxes) if boxes else []
    if not boxes:
        return cells

    from ..utils.segments import find_row_segments

    work = [dict(c) for c in cells]
    row_seps, col_seps = _derive_seps(work)
    if len(col_seps) < 3:
        return work

    segments = find_row_segments(work, text_boxes=boxes)
    if not segments:
        return work

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

        if left_anchor_ce < 0 or right_anchor_cs >= 10**9:
            continue
        if right_anchor_cs <= left_anchor_ce + 1:
            continue

        band_lo = left_anchor_ce + 1
        band_hi = right_anchor_cs - 1

        # 子表头行：父格下一行（多级表头）
        child_row = pre + 1
        if child_row > seg_hi:
            child_row = prs + 1
        children = [
            c
            for c in seg_cells
            if int(c["row_start"]) == child_row
            and int(c["col_end"]) >= band_lo
            and int(c["col_start"]) <= band_hi
            and c is not parent
            and max(int(c.get("row_span") or 1), 1) == 1
        ]

        # 表体行：段内第一个「非表头」行上的中间带格子
        body_rows = sorted(
            {
                int(c["row_start"])
                for c in seg_cells
                if int(c["row_start"]) == int(c["row_end"])
                and int(c["row_start"]) > child_row
            }
        )
        body_cells: List[Dict[str, Any]] = []
        for br in body_rows:
            cand = [
                c
                for c in seg_cells
                if int(c["row_start"]) == br == int(c["row_end"])
                and int(c["col_end"]) >= band_lo
                and int(c["col_start"]) <= band_hi
            ]
            # 至少两格才像单体数据带
            if len(cand) >= 2:
                body_cells = cand
                break

        span_cells = children if len(children) >= 2 else body_cells
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
    cells = unmerge_bad_rowspans(cells, boxes)
    cells = split_underspanned_rows(cells, boxes)
    cells = split_bad_colspans(cells, boxes)
    cells = dedupe_overlapping_cells(cells)
    
    # ======= 此前漏掉的修复：多列表头拆分 =======
    # 将 TSR 误判合并的多列表头（colspan>=2），根据文本的 X 轴聚类拆分为多个独立的单元格
    cells = reconstruct_header_cells(cells, boxes)
    # ============================================

    return cells