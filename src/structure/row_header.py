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
    re.compile(r"^jER[-\u2010]?\d+$", re.I),
    re.compile(r"^[ABC]$"),
    re.compile(r"^其他$"),
)
_CHEM_NAME_RE = re.compile(
    r"^(?:jER[-\u2010]?\d+|OXT[-\u2010]?\d+|EP\d+\w*|NC\d+\w*|EPICLON\d+)$",
    re.I,
)
_CHEM_NAME_FIND_RE = re.compile(
    r"(?:jER[-\u2010]?\d+|OXT[-\u2010]?\d+|EP\d+\w*|NC\d+\w*|EPICLON\d+)",
    re.I,
)
# 子格左缘贴着父格右缘时的像素容差（共享竖线）
_RIGHT_EDGE_TOL = 8.0
# 宽度远小于父格且又矮又窄，视为角落幽灵格（如 P100 分散液右上碎片）
_SLIVER_WIDTH_RATIO = 0.20
_SLIVER_MIN_PX = 12.0


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


def _is_empty_or_frag(cell: Dict[str, Any]) -> bool:
    t = str(cell.get("text") or "").strip()
    if not t:
        return True
    compact = re.sub(r"\s+", "", t)
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

    for parent in parents:
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
                    if _peel_text_token_from_parent(parent, token, sib):
                        parent_text = str(parent.get("text") or "")
                        peeled += 1
                        break
                cm = _CHEM_NAME_FIND_RE.search(parent_text)
                if cm and _is_empty_or_frag(sib):
                    if _peel_text_token_from_parent(parent, cm.group(0), sib):
                        parent_text = str(parent.get("text") or "")
                        peeled += 1

    if peeled:
        logger.info("左行头文本剥离完成: %d 次", peeled)
    return relocate_misplaced_category_labels(cells)


_CATEGORY_LABELS = frozenset({"封端剂", "封剂"})


def relocate_misplaced_category_labels(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    把误入数据列的行头类别标签（如 OCR 截断的「封剂」）移回空 rowspan 父格，
    并规范为「封端剂」。
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
    return _normalize_truncated_chem_names(cells)


def _normalize_truncated_chem_names(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """修正格内常见化学名 OCR 截断（如 ER828 → jER828）。"""
    for c in cells:
        t = str(c.get("text") or "")
        if re.search(r"(?<![jJ])ER828", t):
            c["text"] = re.sub(r"(?<![jJ])ER828", "jER828", t)
    return cells
