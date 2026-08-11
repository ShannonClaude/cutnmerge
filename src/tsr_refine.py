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

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]"
)


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
    survivors = [c for k, c in enumerate(kept) if k not in drop]
    if len(survivors) < 3:
        return survivors

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
    return [c for k, c in enumerate(survivors) if k not in drop2]


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


def merge_ghost_columns(
    cells: List[Dict[str, Any]],
    boxes: Sequence[Dict[str, Any]],
    *,
    min_width_ratio: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    几乎无文本落入且物理宽度异常窄的逻辑列 → 合并到右侧邻列。
    """
    if not cells:
        return cells
    row_seps, col_seps = _derive_seps(cells)
    if len(col_seps) < 3:
        return cells

    n_cols = len(col_seps) - 1
    widths = [col_seps[i + 1] - col_seps[i] for i in range(n_cols)]
    median_w = float(np.median(widths)) if widths else 1.0

    # 每列文本命中数
    hits = [0] * n_cols
    for tb in boxes:
        x1, y1, x2, y2 = _tb_bbox(tb)
        mx = (x1 + x2) / 2.0
        for c in range(n_cols):
            if col_seps[c] <= mx < col_seps[c + 1] or (
                c == n_cols - 1 and col_seps[c] <= mx <= col_seps[c + 1]
            ):
                hits[c] += 1
                break

    # 标记幽灵列：窄 + 无文本（或极少）
    ghost = [
        (widths[c] < min_width_ratio * median_w and hits[c] == 0)
        for c in range(n_cols)
    ]
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
        new_cs = min(map_col(i) for i in range(cs, ce + 1))
        new_ce = max(map_col(i) for i in range(cs, ce + 1))
        nc["col_start"] = new_cs
        nc["col_end"] = new_ce
        _refresh_spans(nc)
        # 更新物理框到新 col_seps 子集
        new_col_seps = [col_seps[survivors[0]]]
        for s in survivors:
            new_col_seps.append(col_seps[s + 1])
        # 简化：用 survivors 边界重建
        xs0 = col_seps[survivors[new_cs]] if new_cs < len(survivors) else col_seps[0]
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
        if not (all_same or none_cjk):
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
            if best_gap < 0.25 * width:
                continue
            # 左右都有点
            left_n = sum(1 for x in xs if x < mid)
            right_n = sum(1 for x in xs if x >= mid)
            if left_n < 1 or right_n < 1:
                continue
            support += 1
            split_xs.append(mid)

        if support >= min_support_rows and split_xs:
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


def line_density_score(image: np.ndarray) -> float:
    """粗略线密度：用于判断是否更像有线表。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    h, w = binary.shape[:2]
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 40, 10), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 40, 10)))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    return float(np.count_nonzero(h_lines) + np.count_nonzero(v_lines)) / float(h * w + 1)


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


def refine_tsr_cells(
    cells: List[Dict[str, Any]],
    boxes: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """TSR 结构后处理流水线。"""
    if not cells:
        return cells
    cells = dedupe_overlapping_cells(cells)
    cells = merge_ghost_columns(cells, boxes)
    cells = dedupe_overlapping_cells(cells)
    cells = unmerge_bad_rowspans(cells, boxes)
    cells = split_underspanned_rows(cells, boxes)
    cells = split_bad_colspans(cells, boxes)
    cells = dedupe_overlapping_cells(cells)
    return cells
