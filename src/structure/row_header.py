"""左侧多层行头：按物理位置裁掉父格对右侧子格的过宽 colspan。

TSR 常把「聚酰亚胺组成」「(b)化合物」等左侧分类格的逻辑列拉得过宽，
盖住右侧酸酐 / 品名等子格。可视化仍画各自 polygon，看起来标准；
HTML 按逻辑占用渲染时会把子标签并进父格。

本模块只改 col_start/col_end（及 col_span），不改 polygon。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 单行标签物理宽于逻辑 colspan 时的裁剪阈值（相对中位列宽）
_NARROW_LABEL_WIDTH_RATIO = 1.35

# 可从父格文本剥离到空兄弟格的子行头标签
_PEEL_SUBLABEL_RES = (
    re.compile(r"^酸酐$"),
    re.compile(r"^二胺$"),
    re.compile(r"^封端剂$"),
    re.compile(r"^封剂$"),
    re.compile(r"^化合物\s*[\(（]?[bB][\)）]?$"),
    re.compile(r"^[\(（][bB][\)）]化合物$"),
    re.compile(r"^jER[-\u2010]?\d+$", re.I),
    re.compile(r"^[ABC]$"),
    re.compile(r"^苯胺$"),
    re.compile(r"^其他$"),
)
_B_COMPOUND_PARENT_RE = re.compile(
    r"化合物.*[\(（][bB][\)）]|[\(（][bB][\)）].*化合物",
    re.I,
)
_C_COMPOUND_PARENT_RE = re.compile(
    r"[\(（][cC][\)）].*醌二",
    re.I,
)
_CHEM_NAME_RE = re.compile(
    r"^(?:jER[-\u2010]?\d+|OXT[-\u2010]?\d+|EP\d+\w*|NC\d+\w*|EPICLON\d+)$",
    re.I,
)
_CHEM_NAME_FIND_RE = re.compile(
    r"(?:jER[-\u2010]?\d+|OXT[-\u2010]?\d+|EP\d+\w*|NC\d+\w*|EPICLON\d+)",
    re.I,
)
_CATEGORY_LABELS = frozenset({"封端剂", "封剂"})
_SUBROW_LABELS = frozenset({"苯胺"})
# 子格左缘贴着父格右缘时的像素容差（共享竖线）
_RIGHT_EDGE_TOL = 8.0
# 宽度远小于父格且又矮又窄，视为角落幽灵格（如 P100 分散液右上碎片）
_SLIVER_WIDTH_RATIO = 0.20
_SLIVER_MIN_PX = 12.0
# 插入品名格时相对父格宽度的默认比例
_INSERT_LABEL_WIDTH_RATIO = 0.45
_INSERT_LABEL_MIN_PX = 40.0
_INSERT_LABEL_MAX_PX = 180.0


def _bbox(cell: Dict[str, Any]) -> Tuple[float, float, float, float]:
    poly = np.asarray(cell.get("polygon"), dtype=np.float64).reshape(-1, 2)
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )


def _logic_area(cell: Dict[str, Any]) -> int:
    return (
        (int(cell["row_end"]) - int(cell["row_start"]) + 1)
        * (int(cell["col_end"]) - int(cell["col_start"]) + 1)
    )


def _rows_subset(child: Dict[str, Any], parent: Dict[str, Any]) -> bool:
    return (
        int(child["row_start"]) >= int(parent["row_start"])
        and int(child["row_end"]) <= int(parent["row_end"])
    )


def _cols_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return not (
        int(a["col_end"]) < int(b["col_start"])
        or int(b["col_end"]) < int(a["col_start"])
    )


def _is_sliver_child(child: Dict[str, Any], parent: Dict[str, Any]) -> bool:
    """角落碎片：宽度远小于父格，且高度也不成一行分类标签。"""
    try:
        cx1, cy1, cx2, cy2 = _bbox(child)
        px1, py1, px2, py2 = _bbox(parent)
    except (TypeError, ValueError):
        return False
    cw = max(cx2 - cx1, 0.0)
    ch = max(cy2 - cy1, 0.0)
    pw = max(px2 - px1, 1.0)
    ph = max(py2 - py1, 1.0)
    narrow = cw < max(_SLIVER_MIN_PX, _SLIVER_WIDTH_RATIO * pw)
    short = ch < 0.50 * ph
    return narrow and short


def is_physically_right_child(
    parent: Dict[str, Any],
    child: Dict[str, Any],
) -> bool:
    """子格物理框在父格右侧（贴边或中心已出父格右界），且不是角落幽灵。

    允许与父格同一 col_start：TSR 常把品名格也吸附到分类列起点。
    """
    if child is parent:
        return False
    if not _rows_subset(child, parent):
        return False
    if not _cols_overlap(child, parent):
        return False
    if _logic_area(child) >= _logic_area(parent):
        return False
    if _is_sliver_child(child, parent):
        return False
    try:
        px1, _, px2, _ = _bbox(parent)
        cx1, _, cx2, _ = _bbox(child)
    except (TypeError, ValueError):
        return int(child["col_start"]) > int(parent["col_start"])
    child_cx = 0.5 * (cx1 + cx2)
    parent_w = max(px2 - px1, 1.0)
    tol = max(_RIGHT_EDGE_TOL, 0.15 * parent_w)
    return (
        cx1 >= px2 - tol
        or child_cx >= px2
        or cx1 >= px1 + 0.35 * parent_w
    )


def _refresh_colspan(cell: Dict[str, Any]) -> None:
    cell["col_span"] = int(cell["col_end"]) - int(cell["col_start"]) + 1


def clip_row_header_child_overlaps(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    跨行父格逻辑列盖住右侧子格时，把父格 col_end 收到子格左侧。

    若子格与父格同一列起点（物理上仍在右），父格收成单列，再把子格右移一列。
    不改 polygon。原地修改并返回 cells。
    """
    if len(cells) < 2:
        return cells

    parents = [
        c
        for c in cells
        if int(c.get("row_span") or (int(c["row_end"]) - int(c["row_start"]) + 1))
        > 1
    ]
    if not parents:
        return cells

    changed = 0
    for parent in parents:
        clip_at: int | None = None
        for child in cells:
            if not is_physically_right_child(parent, child):
                continue
            cand = int(child["col_start"]) - 1
            clip_at = cand if clip_at is None else min(clip_at, cand)
        if clip_at is None:
            continue
        pcs = int(parent["col_start"])
        pce = int(parent["col_end"])
        # 同列起点时 clip_at = pcs-1，仍保留父格起始列
        new_end = max(pcs, min(pce, clip_at))
        if new_end >= pce:
            continue
        parent["col_end"] = new_end
        _refresh_colspan(parent)
        changed += 1
        logger.info(
            "左行头父格裁列: (%s,%s)-(%s,%s) col_end %s → %s",
            parent.get("row_start"),
            parent.get("row_end"),
            pcs,
            pce,
            pce,
            new_end,
        )

    # 第二遍：仍与父格逻辑重叠、但物理在右的子格，整段右移到父格右侧
    shifted = 0
    for parent in parents:
        pcs = int(parent["col_start"])
        pce = int(parent["col_end"])
        dest = pce + 1
        for child in cells:
            if child is parent:
                continue
            if not is_physically_right_child(parent, child):
                continue
            ccs, cce = int(child["col_start"]), int(child["col_end"])
            if ccs >= dest:
                continue
            span = cce - ccs
            child["col_start"] = dest
            child["col_end"] = dest + span
            _refresh_colspan(child)
            shifted += 1
            logger.info(
                "左行头子格右移: text=%r col %s-%s → %s-%s",
                str(child.get("text") or "")[:20],
                ccs,
                cce,
                child["col_start"],
                child["col_end"],
            )

    if changed or shifted:
        logger.info("左行头父格裁列完成: clip=%d shift=%d", changed, shifted)
    return cells


def _median_atomic_col_width(cells: List[Dict[str, Any]]) -> float:
    widths: List[float] = []
    for c in cells:
        if int(c.get("col_span") or 1) != 1:
            continue
        try:
            x1, _, x2, _ = _bbox(c)
        except (TypeError, ValueError):
            continue
        w = max(x2 - x1, 0.0)
        if w > 1e-6:
            widths.append(w)
    if not widths:
        return 0.0
    return float(np.median(widths))


def clip_narrow_label_colspans(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """左区单行标签逻辑 colspan 过宽、物理框却窄时，收到单列。"""
    if not cells:
        return cells
    median_w = _median_atomic_col_width(cells)
    if median_w <= 1e-6:
        return cells
    thresh = median_w * _NARROW_LABEL_WIDTH_RATIO
    changed = 0
    for cell in cells:
        if int(cell.get("row_span") or 1) != 1:
            continue
        if int(cell.get("col_span") or 1) <= 1:
            continue
        if int(cell["col_start"]) > 3:
            continue
        try:
            x1, _, x2, _ = _bbox(cell)
        except (TypeError, ValueError):
            continue
        if (x2 - x1) >= thresh:
            continue
        old_ce = int(cell["col_end"])
        cell["col_end"] = int(cell["col_start"])
        _refresh_colspan(cell)
        changed += 1
        logger.info(
            "窄标签裁列: text=%r col_end %s → %s",
            str(cell.get("text") or "")[:24],
            old_ce,
            cell["col_end"],
        )
    if changed:
        logger.info("窄标签裁列完成: %d 格", changed)
    return cells


def _tb_centroid(tb: Dict[str, Any]) -> Tuple[float, float]:
    poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
    return float(poly[:, 0].mean()), float(poly[:, 1].mean())


def _point_in_cell(cx: float, cy: float, cell: Dict[str, Any]) -> bool:
    try:
        x1, y1, x2, y2 = _bbox(cell)
    except (TypeError, ValueError):
        return False
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _is_known_sublabel_text(text: str) -> bool:
    """已知子行头标签（苯胺、A/B/C、jER828…）不算空碎片。"""
    compact = re.sub(r"\s+", "", (text or "").strip())
    if not compact:
        return False
    if compact in {"A", "B", "C"} or compact in _SUBROW_LABELS:
        return True
    return _token_matches_sublabel(compact)


def _is_empty_or_frag(cell: Dict[str, Any]) -> bool:
    t = str(cell.get("text") or "").strip()
    if not t:
        return True
    compact = re.sub(r"\s+", "", t)
    if _is_known_sublabel_text(compact):
        return False
    return len(compact) <= 2 or compact in {"-", "—", "_"}


def _join_cell_texts(texts: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for tb in texts:
        t = str(tb.get("text") or "").strip()
        if t and t not in parts:
            parts.append(t)
    return "\n".join(parts)


def _refresh_cell_text(cell: Dict[str, Any]) -> None:
    cell["text"] = _join_cell_texts(cell.get("texts") or [])


def _empty_siblings(parent: Dict[str, Any], cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prs, pre = int(parent["row_start"]), int(parent["row_end"])
    pcs, pce = int(parent["col_start"]), int(parent["col_end"])
    out: List[Dict[str, Any]] = []
    for c in cells:
        if c is parent:
            continue
        rs, re_ = int(c["row_start"]), int(c["row_end"])
        if rs < prs or re_ > pre:
            continue
        cs = int(c["col_start"])
        if cs <= pcs:
            continue
        # 右侧相邻或重叠子格（酸酐 col1 不必与父格 col0 逻辑重叠）
        if cs > pce + 1 and not _cols_overlap(parent, c):
            continue
        if not _is_empty_or_frag(c):
            continue
        out.append(c)
    return out


def _peel_text_token_from_parent(
    parent: Dict[str, Any],
    token: str,
    target: Dict[str, Any],
) -> bool:
    """从 parent.text 去掉 token，写入 target（无 OCR box 时仅改 text）。"""
    token = (token or "").strip()
    if not token:
        return False
    parent_text = str(parent.get("text") or "")
    # 父格整段就是该 token：禁止 self-peel 清空标签
    parent_compact = re.sub(r"\s+", "", parent_text)
    token_compact = re.sub(r"\s+", "", token)
    if parent_compact == token_compact:
        return False
    lines = [ln.strip() for ln in parent_text.split("\n") if ln.strip()]
    if len(lines) >= 2 and token in {"A", "B", "C"}:
        for i, ln in enumerate(lines):
            if ln == token:
                lines.pop(i)
                parent["text"] = "\n".join(lines).strip()
                target["text"] = token
                target["texts"] = list(target.get("texts") or [])
                return True
    if token not in parent_text.replace("\n", " "):
        # 尝试无空格粘连：化合物jER828
        compact_p = re.sub(r"\s+", "", parent_text)
        compact_t = re.sub(r"\s+", "", token)
        if compact_t not in compact_p:
            return False
        parent["text"] = compact_p.replace(compact_t, "", 1).strip()
    else:
        parent["text"] = re.sub(
            re.escape(token) + r"\s*",
            "",
            parent_text,
            count=1,
        ).strip()
    target["text"] = token
    target["texts"] = list(target.get("texts") or [])
    logger.info(
        "左行头文本剥离: %r → (%s,%s) 自 (%s,%s)",
        token[:30],
        target.get("row_start"),
        target.get("col_start"),
        parent.get("row_start"),
        parent.get("col_start"),
    )
    return True


def _token_matches_sublabel(token: str) -> bool:
    t = re.sub(r"\s+", "", (token or "").strip())
    if not t:
        return False
    for pat in _PEEL_SUBLABEL_RES:
        if pat.fullmatch(t):
            return True
    return bool(_CHEM_NAME_RE.fullmatch(t))


def _make_label_sibling(
    parent: Dict[str, Any],
    row: int,
    *,
    cells: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """在父格右侧插入空品名原子格（逻辑列紧挨父格）。"""
    pcs = int(parent["col_start"])
    pce = int(parent["col_end"])
    dest_col = pce + 1
    # 若目标列已被占用，仍复用该列起点插入单列格（后续 peel 填字）
    try:
        px1, py1, px2, py2 = _bbox(parent)
    except (TypeError, ValueError):
        px1, py1, px2, py2 = 0.0, 0.0, 80.0, 40.0
    pw = max(px2 - px1, 1.0)
    ph = max(py2 - py1, 1.0)
    prs, pre = int(parent["row_start"]), int(parent["row_end"])
    n_rows = max(1, pre - prs + 1)
    row_h = ph / n_rows
    y1 = py1 + (row - prs) * row_h
    y2 = y1 + row_h
    # 右侧邻格左缘优先，否则按父宽比例估算
    right_x = None
    for c in cells:
        if c is parent:
            continue
        if int(c["row_start"]) > row or int(c["row_end"]) < row:
            continue
        if int(c["col_start"]) <= pce:
            continue
        try:
            cx1, _, _, _ = _bbox(c)
        except (TypeError, ValueError):
            continue
        if cx1 > px2 - 1:
            right_x = cx1 if right_x is None else min(right_x, cx1)
    if right_x is None or right_x <= px2 + 2:
        w = min(
            _INSERT_LABEL_MAX_PX,
            max(_INSERT_LABEL_MIN_PX, pw * _INSERT_LABEL_WIDTH_RATIO),
        )
        right_x = px2 + w
    poly = np.array(
        [[px2, y1], [right_x, y1], [right_x, y2], [px2, y2]],
        dtype=np.float64,
    )
    return {
        "polygon": poly,
        "row_start": row,
        "row_end": row,
        "col_start": dest_col,
        "col_end": dest_col,
        "row_span": 1,
        "col_span": 1,
        "texts": [],
        "text": "",
    }


def _ensure_compound_label_siblings(
    parent: Dict[str, Any],
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """(b)/(c) 大父格右侧缺品名格时按原子行插入空格。"""
    compact = re.sub(r"\s+", "", str(parent.get("text") or ""))
    need_b = bool(_B_COMPOUND_PARENT_RE.search(compact)) or (
        "化合物" in compact and bool(_CHEM_NAME_FIND_RE.search(compact))
    )
    need_c = bool(_C_COMPOUND_PARENT_RE.search(compact))
    if not need_b and not need_c:
        return cells
    prs, pre = int(parent["row_start"]), int(parent["row_end"])
    pce = int(parent["col_end"])
    dest = pce + 1
    # (b) 只补首行；(c) 补每一原子行
    rows = list(range(prs, pre + 1)) if need_c else [prs]
    inserted = 0
    for row in rows:
        exists = any(
            int(c["row_start"]) == row
            and int(c["row_end"]) == row
            and int(c["col_start"]) == dest
            for c in cells
        )
        if exists:
            continue
        cells.append(_make_label_sibling(parent, row, cells=cells))
        inserted += 1
    if inserted:
        logger.info(
            "左行头补品名格: parent=(%s,%s) +%d",
            parent.get("row_start"),
            parent.get("col_start"),
            inserted,
        )
    return cells


def _abc_letter_target(
    parent: Dict[str, Any],
    cells: List[Dict[str, Any]],
    letter: str,
) -> Optional[Dict[str, Any]]:
    """rowspan=3 时 A/B/C 分别对应父格首/中/末行的右侧标签格。"""
    prs, pre = int(parent["row_start"]), int(parent["row_end"])
    dest = int(parent["col_end"]) + 1
    want_row: Optional[int] = None
    if pre - prs == 2:
        want_row = {"A": prs, "B": prs + 1, "C": pre}.get(letter)
    if want_row is None:
        return None
    for c in cells:
        if c is parent:
            continue
        if int(c["row_start"]) != want_row or int(c["row_end"]) != want_row:
            continue
        if int(c["col_start"]) == dest:
            return c
    return None


def _peel_glued_abc_from_parent(
    parent: Dict[str, Any],
    siblings: List[Dict[str, Any]],
    cells: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """从 (c)醌二A / 叠氮化合B 等粘连行末尾抽出 A/B/C。"""
    parent_text = str(parent.get("text") or "").strip()
    if not _C_COMPOUND_PARENT_RE.search(re.sub(r"\s+", "", parent_text)):
        return 0
    abc_sibs = sorted(
        [s for s in siblings if _is_empty_or_frag(s)],
        key=lambda c: int(c["row_start"]),
    )
    lines = [ln.strip() for ln in parent_text.split("\n") if ln.strip()]
    # 已有独立 A/B/C 行时交给后续逻辑，避免与粘连末字母双重剥离
    if any(re.sub(r"\s+", "", ln) in {"A", "B", "C"} for ln in lines):
        return 0
    peeled = 0
    sib_i = 0
    new_lines: List[str] = []
    for ln in lines:
        compact_ln = re.sub(r"\s+", "", ln)
        letter: Optional[str] = None
        rest = ln
        m = re.search(r"([ABC])$", compact_ln)
        if m:
            letter = m.group(1)
            rest = re.sub(r"[ABC]\s*$", "", ln).rstrip()
        if letter is None or not rest:
            new_lines.append(ln)
            continue
        target = None
        if cells:
            target = _abc_letter_target(parent, cells, letter)
        existing = str((target or {}).get("text") or "").strip()
        if target is not None and existing == letter:
            new_lines.append(rest)
            peeled += 1
            continue
        if target is not None and (not existing or _is_empty_or_frag(target)):
            target["text"] = letter
            target["texts"] = list(target.get("texts") or [])
            new_lines.append(rest)
            peeled += 1
            logger.info(
                "左行头粘连字母剥离: %r → (%s,%s)",
                letter,
                target.get("row_start"),
                target.get("col_start"),
            )
            continue
        if sib_i < len(abc_sibs):
            target = abc_sibs[sib_i]
            target["text"] = letter
            target["texts"] = list(target.get("texts") or [])
            new_lines.append(rest)
            sib_i += 1
            peeled += 1
            logger.info(
                "左行头粘连字母剥离: %r → (%s,%s)",
                letter,
                target.get("row_start"),
                target.get("col_start"),
            )
        else:
            new_lines.append(ln)
    if peeled:
        parent["text"] = "\n".join(new_lines).strip()
    return peeled


def peel_row_header_text(
    cells: List[Dict[str, Any]],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    IoA 后：把误落入跨行父格的子行头 OCR/文本剥离到空兄弟格。
    """
    if len(cells) < 2:
        return cells

    parents = [
        c
        for c in cells
        if int(c["col_start"]) <= 2
        and (
            int(c.get("row_span") or (int(c["row_end"]) - int(c["row_start"]) + 1)) > 1
            or int(c.get("col_span") or 1) > 1
        )
    ]
    peeled = 0

    # 先为 (b)/(c) 补品名空格，再 peel
    for parent in parents:
        _ensure_compound_label_siblings(parent, cells)

    for parent in parents:
        # 父格整段即子标签（如「苯胺」colspan=2）：勿当 peel 源清空
        parent_compact = re.sub(r"\s+", "", str(parent.get("text") or ""))
        if parent_compact and _is_known_sublabel_text(parent_compact):
            continue

        siblings = _empty_siblings(parent, cells)
        if not siblings:
            continue

        texts = list(parent.get("texts") or [])
        for tb in texts[:]:
            cx, cy = _tb_centroid(tb)
            best: Dict[str, Any] | None = None
            best_area = float("inf")
            for sib in siblings:
                if not _point_in_cell(cx, cy, sib):
                    continue
                area = _logic_area(sib)
                if area < best_area:
                    best_area = area
                    best = sib
            if best is None:
                continue
            parent["texts"] = [t for t in texts if t is not tb]
            texts = parent["texts"]
            best.setdefault("texts", []).append(tb)
            _refresh_cell_text(parent)
            _refresh_cell_text(best)
            peeled += 1

        # 按行拆多行父格文本到空兄弟格（(c) 的 A/B/C 等）
        lines = [ln.strip() for ln in str(parent.get("text") or "").split("\n") if ln.strip()]
        if len(lines) >= 2:
            by_row: Dict[int, List[Dict[str, Any]]] = {}
            for sib in siblings:
                by_row.setdefault(int(sib["row_start"]), []).append(sib)
            for rs in sorted(by_row.keys()):
                row_sibs = sorted(by_row[rs], key=lambda c: int(c["col_start"]))
                if not row_sibs or not lines:
                    continue
                line = lines[0]
                if _token_matches_sublabel(line) or _CHEM_NAME_RE.fullmatch(line):
                    if _peel_text_token_from_parent(parent, line, row_sibs[0]):
                        lines.pop(0)
                        peeled += 1

        # 从单行粘连文本剥离子标签（酸酐、jER828…）
        parent_text = str(parent.get("text") or "").strip()
        if parent_text:
            compact = re.sub(r"\s+", "", parent_text)
            for sib in sorted(siblings, key=lambda c: (int(c["row_start"]), int(c["col_start"]))):
                if not _is_empty_or_frag(sib):
                    continue
                peeled_one = False
                for name in ("酸酐", "二胺", "封端剂", "封剂", "其他"):
                    if compact.endswith(name) or name in compact:
                        if _peel_text_token_from_parent(parent, name, sib):
                            parent_text = str(parent.get("text") or "")
                            compact = re.sub(r"\s+", "", parent_text)
                            peeled += 1
                            peeled_one = True
                            break
                if peeled_one:
                    continue
                # 化合物 ER828 → 剥 jER828（OCR 丢 j）
                if re.search(r"化合物\s*ER828", parent_text, re.I):
                    if _peel_text_token_from_parent(parent, "jER828", sib):
                        parent_text = str(parent.get("text") or "")
                        compact = re.sub(r"\s+", "", parent_text)
                        peeled += 1
                        continue
                for pat in _PEEL_SUBLABEL_RES:
                    m = pat.search(parent_text)
                    if not m:
                        continue
                    token = m.group(0)
                    # 勿把父格自身整段标签剥到兄弟格
                    if re.sub(r"\s+", "", parent_text) == re.sub(r"\s+", "", token):
                        continue
                    if _peel_text_token_from_parent(parent, token, sib):
                        parent_text = str(parent.get("text") or "")
                        peeled += 1
                        break
                cm = _CHEM_NAME_FIND_RE.search(parent_text)
                if cm and _is_empty_or_frag(sib):
                    if _peel_text_token_from_parent(parent, cm.group(0), sib):
                        parent_text = str(parent.get("text") or "")
                        peeled += 1

        # (b)化合物 父格粘连品名：先剥化学名，再规范父格为 (b)化合物
        parent_text = str(parent.get("text") or "").strip()
        compact_p = re.sub(r"\s+", "", parent_text)
        if _B_COMPOUND_PARENT_RE.search(compact_p) or (
            "化合物" in compact_p and _CHEM_NAME_FIND_RE.search(compact_p)
        ):
            # 刷新 siblings（可能刚插入）
            siblings = _empty_siblings(parent, cells)
            for sib in sorted(siblings, key=lambda c: int(c["row_start"])):
                if not _is_empty_or_frag(sib):
                    continue
                cm = _CHEM_NAME_FIND_RE.search(parent_text)
                if cm and _peel_text_token_from_parent(parent, cm.group(0), sib):
                    parent_text = str(parent.get("text") or "")
                    compact_p = re.sub(r"\s+", "", parent_text)
                    peeled += 1
                    break
            if re.search(r"[\(（][bB][\)）]", compact_p) or "化合物" in compact_p:
                if re.search(r"[\(（][bB][\)）]", compact_p):
                    parent["text"] = "(b)化合物"

        # (c)醌二叠氮：粘连字母 + 独立 A/B/C
        siblings = _empty_siblings(parent, cells)
        glued = _peel_glued_abc_from_parent(parent, siblings, cells)
        peeled += glued
        parent_text = str(parent.get("text") or "").strip()
        if _C_COMPOUND_PARENT_RE.search(re.sub(r"\s+", "", parent_text)):
            abc_sibs = sorted(
                [s for s in siblings if _is_empty_or_frag(s)],
                key=lambda c: int(c["row_start"]),
            )
            lines = [
                ln.strip()
                for ln in parent_text.split("\n")
                if ln.strip()
            ]
            for sib in abc_sibs:
                if not lines:
                    break
                while lines and lines[0] not in {"A", "B", "C"}:
                    lines.pop(0)
                if not lines or lines[0] not in {"A", "B", "C"}:
                    break
                letter = lines[0]
                if _peel_text_token_from_parent(parent, letter, sib):
                    lines.pop(0)
                    peeled += 1

    for parent in parents:
        compact_c = re.sub(r"\s+", "", str(parent.get("text") or ""))
        if "醌二" in compact_c:
            parent["text"] = "(c)醌二叠氮化合物"
            _fill_missing_c_compound_b(parent, cells)

    if peeled:
        logger.info("左行头文本剥离完成: %d 次", peeled)
    return relocate_misplaced_category_labels(cells)


# 保留别名供测试/外部引用（常量已上移）
# _CATEGORY_LABELS / _SUBROW_LABELS 定义见文件顶部


_MONOMER_BAND_RE = re.compile(r"单体\s*[\[［]")
_MONOMER_MID_HEADER_RE = re.compile(r"(四羧酸|二胺|二羧酸|双氨基酚|衍生物)")


def _category_label_under_monomer_band(cells: Sequence[Dict[str, Any]], cell: Dict[str, Any]) -> bool:
    """封端剂落在「单体[…]」列带内（或与四羧酸/二胺等同子表头行）→ 勿当左侧行头挪走。"""
    cs = int(cell["col_start"])
    ce = int(cell["col_end"])
    rs = int(cell["row_start"])
    for p in cells:
        if p is cell:
            continue
        t = str(p.get("text") or "")
        if not _MONOMER_BAND_RE.search(t):
            continue
        pcs, pce = int(p["col_start"]), int(p["col_end"])
        if cs <= pce and ce >= pcs:
            return True
    # 同行已有单体中段子表头 → 也视为列中表头
    for sib in cells:
        if sib is cell:
            continue
        if int(sib["row_start"]) != rs:
            continue
        if _MONOMER_MID_HEADER_RE.search(str(sib.get("text") or "")):
            return True
    return False


def relocate_misplaced_category_labels(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    把误入数据列的行头类别标签（如 OCR 截断的「封剂」）移回空 rowspan 父格，
    并规范为「封端剂」。

    P96 类：封端剂本是「单体」下中段子表头，不得迁到左上角空 rowspan。
    """
    if len(cells) < 2:
        return cells

    moved = 0
    for c in cells:
        raw = str(c.get("text") or "").strip()
        compact = re.sub(r"\s+", "", raw)
        if compact not in _CATEGORY_LABELS:
            continue
        cs = int(c["col_start"])
        if cs <= 2:
            if compact == "封剂":
                c["text"] = "封端剂"
            continue
        # 单体列带内的封端剂是列中表头，不是左侧行头
        if _category_label_under_monomer_band(cells, c):
            if compact == "封剂":
                c["text"] = "封端剂"
            continue

        rs = int(c["row_start"])
        best_parent: Dict[str, Any] | None = None
        for parent in cells:
            if parent is c:
                continue
            if int(parent["col_start"]) > 2:
                continue
            prs, pre = int(parent["row_start"]), int(parent["row_end"])
            if not (prs <= rs <= pre):
                continue
            if str(parent.get("text") or "").strip():
                continue
            rsp = int(
                parent.get("row_span")
                or (int(parent["row_end"]) - int(parent["row_start"]) + 1)
            )
            if rsp < 2:
                continue
            if best_parent is None or int(parent["col_start"]) > int(
                best_parent["col_start"]
            ):
                best_parent = parent

        if best_parent is None:
            if compact == "封剂":
                c["text"] = "封端剂"
            continue

        best_parent["text"] = "封端剂"
        c["text"] = ""
        c["texts"] = []
        moved += 1

    if moved:
        logger.info("类别行头归位: %d 次", moved)
    cells = _relocate_photosensitive_section_prefix(cells)
    return _relocate_misplaced_subrow_labels(_normalize_truncated_chem_names(cells))


def _relocate_misplaced_subrow_labels(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """封端剂末行「苯胺」等子标签误入数据列时归位到空兄弟格。"""
    if len(cells) < 2:
        return cells
    moved = 0
    for c in cells:
        raw = str(c.get("text") or "").strip()
        compact = re.sub(r"\s+", "", raw)
        if compact not in _SUBROW_LABELS:
            continue
        if int(c["col_start"]) <= 2:
            continue
        rs = int(c["row_start"])
        best: Dict[str, Any] | None = None
        for sib in cells:
            if sib is c:
                continue
            if int(sib["row_start"]) != rs:
                continue
            if int(sib["col_start"]) >= int(c["col_start"]):
                continue
            if str(sib.get("text") or "").strip():
                continue
            if best is None or int(sib["col_start"]) > int(best["col_start"]):
                best = sib
        if best is None:
            continue
        best["text"] = raw
        c["text"] = ""
        c["texts"] = []
        moved += 1
    if moved:
        logger.info("子行标签归位: %d 次", moved)
    return cells


def _normalize_truncated_chem_names(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """修正格内常见化学名 OCR 截断（如 ER828 → jER828、端MAP → MAP）。"""
    for c in cells:
        t = str(c.get("text") or "")
        if re.search(r"(?<![jJ])ER828", t):
            t = re.sub(r"(?<![jJ])ER828", "jER828", t)
        t = re.sub(r"端\s*MAP\b", "MAP", t)
        c["text"] = t
    return cells


def _fill_missing_c_compound_b(
    parent: Dict[str, Any],
    cells: List[Dict[str, Any]],
) -> None:
    """(c) 父格 rowspan=3 且子标签为 A / 空 / C 时补 B。"""
    prs, pre = int(parent["row_start"]), int(parent["row_end"])
    if pre - prs != 2:
        return
    dest = int(parent["col_end"]) + 1
    by_row: Dict[int, Dict[str, Any]] = {}
    for c in cells:
        if c is parent:
            continue
        if int(c["row_start"]) != int(c["row_end"]):
            continue
        r = int(c["row_start"])
        if r < prs or r > pre:
            continue
        if int(c["col_start"]) != dest:
            continue
        by_row[r] = c
    if not all(r in by_row for r in range(prs, pre + 1)):
        grouped: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for c in cells:
            if c is parent:
                continue
            if int(c["row_start"]) != int(c["row_end"]):
                continue
            r = int(c["row_start"])
            if r < prs or r > pre:
                continue
            t = str(c.get("text") or "").strip()
            if t not in {"", "A", "B", "C"}:
                continue
            grouped.setdefault(int(c["col_start"]), {})[r] = c
        by_row = {}
        for _col, rows in sorted(grouped.items()):
            if all(k in rows for k in range(prs, pre + 1)):
                t0 = str(rows[prs].get("text") or "").strip()
                t2 = str(rows[pre].get("text") or "").strip()
                if t0 == "A" and t2 == "C":
                    by_row = rows
                    break
    if not all(r in by_row for r in range(prs, pre + 1)):
        return
    t0 = str(by_row[prs].get("text") or "").strip()
    t1 = str(by_row[prs + 1].get("text") or "").strip()
    t2 = str(by_row[pre].get("text") or "").strip()
    if t0 == "A" and t2 == "C" and not t1:
        by_row[prs + 1]["text"] = "B"


_PHOTOSENSITIVE_A_RE = re.compile(
    r"^感光性组\s*[（(]\s*a\s*[)）]\s*聚酰亚胺$",
    re.I,
)


def _relocate_photosensitive_section_prefix(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """把 (a) 格前缀「感光性组」挪回左侧大类行头。"""
    donor: Optional[Dict[str, Any]] = None
    for c in cells:
        compact = re.sub(r"\s+", "", str(c.get("text") or "").strip())
        if _PHOTOSENSITIVE_A_RE.fullmatch(compact) or (
            compact.startswith("感光性组")
            and "聚酰亚胺" in compact
            and re.search(r"[（(]a[)）]", compact, re.I)
        ):
            donor = c
            raw = str(c.get("text") or "")
            c["text"] = re.sub(r"^感光性组\s*", "", raw).strip()
            break
    if donor is None:
        return cells
    rs = int(donor["row_start"])
    best: Optional[Dict[str, Any]] = None
    for p in cells:
        if p is donor:
            continue
        if int(p["col_start"]) != 0:
            continue
        if int(p["row_start"]) > rs or int(p["row_end"]) < rs:
            continue
        rsp = int(p.get("row_span") or (int(p["row_end"]) - int(p["row_start"]) + 1))
        if rsp < 2:
            continue
        best = p
        break
    if best is None:
        return cells
    pt = re.sub(r"\s+", "", str(best.get("text") or ""))
    if "感光性组合物组成" not in pt:
        best["text"] = "感光性组合物组成（重量份）"
        logger.info("感光性组合物组成行头归位")
    return cells


_SECTION_PARENT_RE = re.compile(r"聚酰亚胺.*组成")
_METRIC_ROW_RE = re.compile(r"聚酰亚胺.*(?:酰亚胺化率|重均分子量)")


def _refresh_rowspan(cell: Dict[str, Any]) -> None:
    cell["row_span"] = int(cell["row_end"]) - int(cell["row_start"]) + 1


def extend_section_rowspan_over_metric_rows(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    P24/P25：左侧「聚酰亚胺组成（摩尔比）」大类行头若覆盖了
    「酰亚胺化率 / 重均分子量」测量行，应收缩 rowspan 使其只覆盖
    组成子行，测量行作为独立全宽标签行。
    """
    if not cells:
        return cells

    changed = 0
    for parent in cells:
        if int(parent["col_start"]) != 0:
            continue
        row_span = int(
            parent.get("row_span")
            or (int(parent["row_end"]) - int(parent["row_start"]) + 1)
        )
        if row_span < 3:
            continue
        text = str(parent.get("text") or "").strip()
        if not _SECTION_PARENT_RE.search(text):
            continue

        parent_start = int(parent["row_start"])
        parent_end = int(parent["row_end"])

        # 收集在 parent 范围内或紧邻其后的测量行
        metric_rows: List[int] = []
        for c in cells:
            if c is parent:
                continue
            rs, re_ = int(c["row_start"]), int(c["row_end"])
            if rs != re_:
                continue
            # 测量行在 parent 范围内或紧邻其后（+3 容差）
            if rs < parent_start:
                continue
            if rs > parent_end + 3:
                continue
            label = str(c.get("text") or "").strip()
            if not _METRIC_ROW_RE.search(label):
                continue
            cs = int(c["col_start"])
            if cs >= 4:
                continue
            metric_rows.append(rs)

        if not metric_rows:
            continue

        metric_rows = sorted(set(metric_rows))

        # 收缩 parent：row_end 设为测量行最小行号的前一行
        new_end = min(metric_rows) - 1
        if new_end < parent_start:
            new_end = parent_start
        if new_end != parent_end:
            old_end = parent_end
            parent["row_end"] = new_end
            _refresh_rowspan(parent)
            logger.info(
                "大类行头收缩排除测量行: %s row_end %d → %d",
                text[:24],
                old_end,
                new_end,
            )

        # 测量行标签单元格：确保从 col_start=0 开始，成为独立全宽标签
        metric_row_set = set(metric_rows)
        for c in cells:
            rs = int(c["row_start"])
            if rs not in metric_row_set:
                continue
            cs, ce = int(c["col_start"]), int(c["col_end"])
            label = str(c.get("text") or "").strip()

            # 空的 col0 占位单元格丢弃
            if cs == 0 and not label:
                c["_drop_render"] = True
                continue

            # 测量行标签：扩展到 col_start=0 使其独立
            if _METRIC_ROW_RE.search(label):
                if cs >= 1:
                    c["col_start"] = 0
                    _refresh_colspan(c)
                continue

        changed += 1

    if not changed:
        return cells
    return [c for c in cells if not c.get("_drop_render")]
