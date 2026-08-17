"""Re-run OCR on suspicious cells via a single packed montage."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_CJK_CHAR_RE = re.compile(r"[\u3400-\u9fff]")

from .label_patterns import (
    complete_truncated_component_header,
    fix_iii_ocr,
    split_value_grade,
)
from .matching import join_cell_texts, unmerge_filled_label_rowspans
from .models import predict_texts
from .ocr_post import _maybe_geometric_dash


def _cell_bbox(cell: Dict[str, Any]) -> Tuple[int, int, int, int]:
    poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
    return (
        int(np.floor(poly[:, 0].min())),
        int(np.floor(poly[:, 1].min())),
        int(np.ceil(poly[:, 0].max())),
        int(np.ceil(poly[:, 1].max())),
    )


def _crop_cell(image: np.ndarray, cell: Dict[str, Any], pad: int = 4) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _cell_bbox(cell)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def _cell_ink_ratio(binary: np.ndarray, cell: Dict[str, Any]) -> float:
    x1, y1, x2, y2 = _cell_bbox(cell)
    h, w = binary.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = binary[y1:y2, x1:x2]
    return float(np.count_nonzero(roi)) / float(max(roi.size, 1))


def _avg_score(texts: Sequence[Dict[str, Any]]) -> float:
    scores = [float(tb.get("score", 1.0)) for tb in texts if str(tb.get("text") or "").strip()]
    return float(np.mean(scores)) if scores else 0.0


def _looks_numeric(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and all(ch in "0123456789.-+×x/^%()[]{}<>Ω□cmgAJB " for ch in t)


def _char_skeleton(text: str) -> str:
    """
    列模板骨架：把字符映射到稳定类别，避免同义符号/数字位差导致模式失真。
    digits->9, latin->A, CJK->C, other keep.
    """
    t = (text or "").strip()
    if not t:
        return ""
    out: List[str] = []
    for ch in t:
        if ch.isdigit():
            out.append("9")
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            out.append("A")
        else:
            o = ord(ch)
            is_cjk = (0x3400 <= o <= 0x9FFF) or (0xF900 <= o <= 0xFAFF)
            if is_cjk:
                out.append("C")
            elif ch.isspace():
                continue
            else:
                out.append(ch)

    s = "".join(out)
    if not s:
        return ""
    # 压缩连续相同类别，降低 OCR 字符数量差异的敏感性
    compressed: List[str] = [s[0]]
    for ch in s[1:]:
        if ch == compressed[-1] and ch in {"9", "A", "C"}:
            continue
        compressed.append(ch)
    return "".join(compressed)


def _column_template_skeletons(cells: Sequence[Dict[str, Any]]) -> Dict[int, str]:
    """
    列模板一致性：
      - 对每个 col_start 收集所有非空 cell 的 skeleton
      - 若某 skeleton 频次足够高，则作为该列模板
    """
    from collections import Counter

    col_to_sks: Dict[int, List[str]] = {}
    for c in cells:
        col = int(c.get("col_start") or 0)
        txt = str(c.get("text") or "").strip()
        if not txt:
            continue
        sk = _char_skeleton(txt)
        if not sk:
            continue
        col_to_sks.setdefault(col, []).append(sk)

    templates: Dict[int, str] = {}
    for col, sks in col_to_sks.items():
        if len(sks) < 3:
            continue
        cnt = Counter(sks)
        sk_mode, freq = cnt.most_common(1)[0]
        if freq >= max(2, int(np.ceil(len(sks) * 0.45))):
            templates[col] = sk_mode
    return templates


def _cell_leftover_ink_ratio(binary: np.ndarray, cell: Dict[str, Any], inset: int = 3) -> float:
    """单元格内、已归属文本框之外的前景占比。用于检出半识别。"""
    x1, y1, x2, y2 = _cell_bbox(cell)
    h, w = binary.shape[:2]
    x1 = max(0, x1 + inset)
    y1 = max(0, y1 + inset)
    x2 = min(w, x2 - inset)
    y2 = min(h, y2 - inset)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = binary[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    mask = roi.copy()
    for tb in cell.get("texts") or []:
        poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
        if poly.size < 4:
            continue
        tx1 = int(np.floor(poly[:, 0].min())) - x1 - 2
        ty1 = int(np.floor(poly[:, 1].min())) - y1 - 2
        tx2 = int(np.ceil(poly[:, 0].max())) - x1 + 2
        ty2 = int(np.ceil(poly[:, 1].max())) - y1 + 2
        tx1 = max(0, tx1)
        ty1 = max(0, ty1)
        tx2 = min(mask.shape[1], tx2)
        ty2 = min(mask.shape[0], ty2)
        if tx2 > tx1 and ty2 > ty1:
            mask[ty1:ty2, tx1:tx2] = 0
    return float(np.count_nonzero(mask)) / float(max(roi.size, 1))


def _looks_truncated_header(text: str) -> bool:
    t = (text or "").strip()
    return bool(re.fullmatch(r"\([A-Za-z]\d+\)(?:[\n/]+第\d+)?$", t, flags=re.S))


def _apply_component_header_completion(cells: List[Dict[str, Any]]) -> None:
    rows: dict[int, list[Dict[str, Any]]] = {}
    for cell in cells:
        rows.setdefault(int(cell.get("row_start") or 0), []).append(cell)
    for row_cells in rows.values():
        donors = [str(c.get("text") or "") for c in row_cells]
        for cell in row_cells:
            old = str(cell.get("text") or "")
            new = complete_truncated_component_header(old, donors)
            if new != old:
                cell["text"] = new


def _coerce_dash_column_ones(cells: List[Dict[str, Any]], binary: np.ndarray) -> None:
    """破折号列里把孤立「1/l/I」按几何横线收成 '-'。"""
    cols: dict[int, list[str]] = {}
    for cell in cells:
        cols.setdefault(int(cell.get("col_start") or 0), []).append(
            str(cell.get("text") or "").strip()
        )
    dash_cols = {
        col
        for col, texts in cols.items()
        if texts
        and sum(1 for t in texts if t == "-") >= max(2, int(np.ceil(len(texts) * 0.4)))
    }
    if not dash_cols:
        return
    for cell in cells:
        if int(cell.get("col_start") or 0) not in dash_cols:
            continue
        text = str(cell.get("text") or "").strip()
        if text not in {"1", "l", "I", "一", "|", ""}:
            continue
        if text == "":
            if _cell_ink_ratio(binary, cell) >= 0.01:
                cell["text"] = "-"
            continue
        tbs = cell.get("texts") or []
        if tbs:
            dummy = dict(tbs[0])
            dummy["text"] = text
            fixed = _maybe_geometric_dash(dummy, text, binary=binary)
            if fixed == "-":
                cell["text"] = "-"
                continue
        cell["text"] = "-"


def _looks_garbled_cjk(text: str) -> bool:
    """密表化学中文常见乱码启发式：短串内重复生僻字或无意义碎片。"""
    t = (text or "").strip()
    if len(t) < 2:
        return False
    # 明显乱码片段
    if re.search(r"(后居游|别房|品民房|路学总|即房号|品市安|66-7X)", t):
        return True
    cjk = re.findall(r"[\u3400-\u9fff]", t)
    if len(cjk) >= 3 and len(set(cjk)) <= max(2, len(cjk) // 3):
        return True
    return False


def _suspicious_indices(cells: Sequence[Dict[str, Any]], binary: np.ndarray) -> List[int]:
    suspects: List[int] = []
    cols: dict[int, list[str]] = {}
    for cell in cells:
        cols.setdefault(int(cell["col_start"]), []).append(str(cell.get("text") or "").strip())

    templates = _column_template_skeletons(cells)

    numeric_cols = {
        col
        for col, texts in cols.items()
        if texts and sum(1 for t in texts if _looks_numeric(t)) >= max(2, int(np.ceil(len(texts) * 0.6)))
    }

    for i, cell in enumerate(cells):
        text = str(cell.get("text") or "").strip()
        texts = cell.get("texts") or []
        scores = [float(tb.get("score", 1.0)) for tb in texts if str(tb.get("text") or "").strip()]
        low_conf = bool(scores) and min(scores) < 0.75
        multiline = any(bool(tb.get("needs_reocr")) for tb in texts)
        vertical = any(bool(tb.get("maybe_vertical")) for tb in texts)
        empty_but_inky = not text and _cell_ink_ratio(binary, cell) >= 0.02
        leftover_ink = bool(text) and _cell_leftover_ink_ratio(binary, cell) >= 0.02
        truncated_header = _looks_truncated_header(text)
        numeric_mismatch = int(cell["col_start"]) in numeric_cols and text and not _looks_numeric(text)

        template_mismatch = False
        col_template = templates.get(int(cell.get("col_start") or 0))
        if col_template and text:
            template_mismatch = _char_skeleton(text) != col_template

        # 细长竖排格
        x1, y1, x2, y2 = _cell_bbox(cell)
        cw, ch = max(x2 - x1, 1), max(y2 - y1, 1)
        tall_narrow = ch >= 2.5 * cw and ch >= 40

        garbled = _looks_garbled_cjk(text)
        # 末行/短串数字崩坏：如 "0 0" "+0"
        broken_num = bool(re.fullmatch(r"[0+\-\s]{1,6}", text)) and int(cell.get("col_start") or 0) in numeric_cols

        if (
            low_conf
            or multiline
            or vertical
            or empty_but_inky
            or leftover_ink
            or truncated_header
            or numeric_mismatch
            or template_mismatch
            or tall_narrow
            or garbled
            or broken_num
        ):
            suspects.append(i)
    return suspects


def _pack_crops(crops: Sequence[np.ndarray], target_h: int = 80, gap: int = 16, max_width: int = 1800) -> tuple[np.ndarray, List[Tuple[int, int, int, int]], List[float]]:
    scaled: List[np.ndarray] = []
    scales: List[float] = []
    for crop in crops:
        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            scaled.append(np.full((target_h, target_h, 3), 255, dtype=np.uint8))
            scales.append(1.0)
            continue
        scale = target_h / float(h)
        new_w = max(8, int(round(w * scale)))
        scaled.append(cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC))
        scales.append(scale)

    rects: List[Tuple[int, int, int, int]] = []
    x = gap
    y = gap
    row_h = 0
    canvas_w = max_width
    canvas_h = gap
    for crop in scaled:
        h, w = crop.shape[:2]
        if x + w + gap > canvas_w:
            x = gap
            y += row_h + gap
            row_h = 0
        rects.append((x, y, x + w, y + h))
        x += w + gap
        row_h = max(row_h, h)
        canvas_h = max(canvas_h, y + h + gap)

    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    for crop, (x1, y1, x2, y2) in zip(scaled, rects):
        canvas[y1:y2, x1:x2] = crop
    return canvas, rects, scales


def _boxes_in_rect(text_boxes: Sequence[Dict[str, Any]], rect: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
    x1, y1, x2, y2 = rect
    out = []
    for tb in text_boxes:
        poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
        cx = float(poly[:, 0].mean())
        cy = float(poly[:, 1].mean())
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            ntb = dict(tb)
            ntb["polygon"] = np.asarray(tb["polygon"], dtype=np.float64) - np.array([[x1, y1]])
            out.append(ntb)
    return out


def _montage_cache_extra(cells: Sequence[Dict[str, Any]], image: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(str(image.shape).encode("utf-8"))
    for cell in cells:
        h.update(str(_cell_bbox(cell)).encode("utf-8"))
        h.update(str(cell.get("text") or "").encode("utf-8", errors="ignore"))
    return "reocr=" + h.hexdigest()


def apply_reocr_to_cells(
    image: np.ndarray,
    cells: List[Dict[str, Any]],
    *,
    binary: np.ndarray,
    ocr_engine=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    max_cells: int = 24,
) -> List[Dict[str, Any]]:
    if not cells:
        return cells
    suspect_ids = _suspicious_indices(cells, binary)
    if not suspect_ids:
        return cells
    suspect_ids = suspect_ids[:max_cells]
    templates = _column_template_skeletons(cells)
    
    def _ocr_candidate_for_cell(
        cell_idx: int, *, rotate: int = 0
    ) -> Optional[List[Dict[str, Any]]]:
        """
        对单个可疑 cell 做定向二次 OCR：
          1) 按原分辨率放大到更稳定的字高尺度
          2) 可选 90/180/270° 旋转（竖排标签）
          3) OCR 后把 polygon 映射回“未放大、未旋转”的 cell crop 坐标系
        """
        cell = cells[cell_idx]
        crop, _bbox = _crop_cell(image, cell, pad=6)
        if crop.size == 0:
            return None
        h0, w0 = crop.shape[:2]
        if h0 <= 0 or w0 <= 0:
            return None

        # 放大：小字倾向更大倍率；大字不必过度放大
        target_h = int(max(160, min(420, h0 * 2.0)))
        scale = float(target_h) / float(h0)
        w1 = max(8, int(round(w0 * scale)))
        crop_scaled = cv2.resize(crop, (w1, target_h), interpolation=cv2.INTER_CUBIC)
        rot = int(rotate) % 360
        if rot == 90:
            crop_scaled = cv2.rotate(crop_scaled, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            crop_scaled = cv2.rotate(crop_scaled, cv2.ROTATE_180)
        elif rot == 270:
            crop_scaled = cv2.rotate(crop_scaled, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cache_extra = (
            f"reocr|cell={cell_idx}|rot={rot}|s={scale:.3f}|"
            f"{_cell_bbox(cell)}|{str(cell.get('col_start') or '')}"
        )
        tbs = predict_texts(
            crop_scaled,
            ocr_engine=ocr_engine,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_extra=cache_extra,
        )
        if not tbs:
            return None

        w_scaled, h_scaled = float(crop_scaled.shape[1]), float(crop_scaled.shape[0])
        mapped: List[Dict[str, Any]] = []
        for i, tb in enumerate(tbs):
            ntb = dict(tb)
            text = fix_iii_ocr(str(tb.get("text") or ""))
            g = split_value_grade(text)
            if g is not None:
                text = f"{g[0]}\n{g[1]}"
            ntb["text"] = text
            if rot == 0:
                poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2) / scale
            elif rot == 180:
                poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2).copy()
                poly[:, 0] = w_scaled - poly[:, 0]
                poly[:, 1] = h_scaled - poly[:, 1]
                poly = poly / scale
            else:
                # 90/270：按阅读序合成框即可（整格替换不依赖精确坐标）
                y0 = 4.0 + i * 12.0
                poly = np.array(
                    [
                        [2.0, y0],
                        [max(w0 - 2.0, 8.0), y0],
                        [max(w0 - 2.0, 8.0), y0 + 10.0],
                        [2.0, y0 + 10.0],
                    ],
                    dtype=np.float64,
                )
            ntb["polygon"] = poly
            mapped.append(ntb)
        return mapped

    for idx in suspect_ids:
        cell = cells[idx]
        old_text_from_texts = (
            join_cell_texts(cell.get("texts") or []).strip()
            if cell.get("texts")
            else ""
        )
        old_text = (old_text_from_texts or str(cell.get("text") or "")).strip()
        old_score = _avg_score(cell.get("texts") or [])
        x1, y1, x2, y2 = _cell_bbox(cell)
        tall = (y2 - y1) >= 2.5 * max(x2 - x1, 1)
        rotations = [0, 180]
        if tall or any(bool(tb.get("maybe_vertical")) for tb in (cell.get("texts") or [])):
            rotations = [0, 90, 180, 270]

        candidates: List[Tuple[float, str, List[Dict[str, Any]]]] = []
        for rot in rotations:
            tbs = _ocr_candidate_for_cell(idx, rotate=rot)
            if not tbs:
                continue
            new_text = join_cell_texts(tbs).strip()
            if not new_text:
                continue
            candidates.append((_avg_score(tbs), new_text, tbs))

        if not candidates:
            continue

        leftover = _cell_leftover_ink_ratio(binary, cell) >= 0.02
        truncated = _looks_truncated_header(old_text)
        # 短串半识别（如「2」对「23A+」）才按更长文本取；长表头不因旋转多字被覆盖
        incomplete = leftover and len(old_text) <= 8
        prefer_longer = truncated or incomplete
        if prefer_longer:
            new_score, new_text, new_tbs = max(
                candidates, key=lambda x: (len(x[1]), x[0])
            )
        else:
            new_score, new_text, new_tbs = max(candidates, key=lambda x: x[0])

        # 表题/表外格不做二次覆盖
        if old_text and (
            (int(cell.get("row_start") or 0) <= 1 and ("表" in old_text or old_text.startswith("[")))
            or ("表" in old_text and "[" in old_text)
            or (old_text.startswith("[") and len(old_text) <= 6)
        ):
            continue

        if not old_text:
            cell["texts"] = new_tbs
            cell["text"] = new_text
            continue

        garbled_old = _looks_garbled_cjk(old_text) or bool(
            re.fullmatch(r"[0+\-\s]{1,6}", old_text)
        )
        if garbled_old:
            if len(new_text) >= 1 and new_score >= max(0.35, old_score - 0.1):
                cell["texts"] = new_tbs
                cell["text"] = new_text
            continue

        if prefer_longer and len(new_text) > len(old_text) and new_score >= 0.35:
            cell["texts"] = new_tbs
            cell["text"] = new_text
            continue

        if new_score < old_score + 0.05:
            continue
        if len(new_text) < 0.6 * max(1, len(old_text)):
            continue

        col_template = templates.get(int(cell.get("col_start") or 0))
        if col_template and _char_skeleton(new_text) != col_template:
            continue

        cell["texts"] = new_tbs
        cell["text"] = new_text

    _apply_component_header_completion(cells)
    _coerce_dash_column_ones(cells, binary)
    # 二次 OCR 后可能修好行标签，再拆一次错误 rowspan
    cells = unmerge_filled_label_rowspans(cells)
    return cells


def _inset_xyxy(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    min_px: int = 4,
    frac: float = 0.12,
) -> Tuple[int, int, int, int]:
    """Shrink a bbox to drop grid-line ink on the borders."""
    w = max(x2 - x1, 0)
    h = max(y2 - y1, 0)
    dx = min(max(min_px, int(round(w * frac))), max(0, w // 3))
    dy = min(max(min_px, int(round(h * frac))), max(0, h // 3))
    return x1 + dx, y1 + dy, x2 - dx, y2 - dy


def _interior_ink_ratio(binary: np.ndarray, cell: Dict[str, Any]) -> float:
    x1, y1, x2, y2 = _cell_bbox(cell)
    ix1, iy1, ix2, iy2 = _inset_xyxy(x1, y1, x2, y2)
    h, w = binary.shape[:2]
    ix1 = max(0, ix1)
    iy1 = max(0, iy1)
    ix2 = min(w, ix2)
    iy2 = min(h, iy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    roi = binary[iy1:iy2, ix1:ix2]
    return float(np.count_nonzero(roi)) / float(max(roi.size, 1))


def _compact_header_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip()


def _looks_like_vertical_header_text(text: str, score: float) -> bool:
    """Short CJK header; reject digits/codes/border noise."""
    if score < 0.6:
        return False
    compact = _compact_header_text(text)
    if not (2 <= len(compact) <= 8):
        return False
    cjk_n = len(_CJK_CHAR_RE.findall(compact))
    if cjk_n < max(2, int(round(0.6 * len(compact)))):
        return False
    non_cjk = sum(1 for ch in compact if not _CJK_CHAR_RE.fullmatch(ch) and ch not in "()（）[]【】")
    if non_cjk >= max(2, len(compact) - 1):
        return False
    if re.fullmatch(r"[\d.\-+\s|丨lI]+", compact):
        return False
    return True


def _neighbor_texts(
    cells: Sequence[Dict[str, Any]], cell: Dict[str, Any]
) -> List[str]:
    rs = int(cell["row_start"])
    cs = int(cell["col_start"])
    out: List[str] = []
    for other in cells:
        if other is cell:
            continue
        if int(other["row_start"]) != rs:
            continue
        ocs = int(other["col_start"])
        if ocs in {cs - 1, cs + 1}:
            t = _compact_header_text(str(other.get("text") or ""))
            if t:
                out.append(t)
    return out


def recover_empty_vertical_headers(
    image: np.ndarray,
    cells: List[Dict[str, Any]],
    *,
    binary: np.ndarray,
    ocr_engine=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    max_cells: int = 8,
) -> List[Dict[str, Any]]:
    """
    定点补检：表头带空格 + 内缩后仍有墨迹 + 窄高 → 旋转 90/270° OCR。

    真正空白的表头（只有框线）内缩后墨迹≈0，不发请求。
    """
    if not cells or image is None or binary is None:
        return cells

    header_hi = 1
    candidates: List[int] = []
    for i, cell in enumerate(cells):
        if int(cell.get("row_start") or 0) > header_hi:
            continue
        if max(int(cell.get("col_span") or 1), 1) != 1:
            continue
        if str(cell.get("text") or "").strip():
            continue
        x1, y1, x2, y2 = _cell_bbox(cell)
        ix1, iy1, ix2, iy2 = _inset_xyxy(x1, y1, x2, y2)
        iw, ih = max(ix2 - ix1, 1), max(iy2 - iy1, 1)
        if ih < 36 or ih < 1.8 * iw:
            continue
        if _interior_ink_ratio(binary, cell) < 0.04:
            continue
        candidates.append(i)
        if len(candidates) >= max_cells:
            break

    if not candidates:
        return cells

    slots: List[Tuple[int, int]] = []  # (cell_idx, rot)
    crops: List[np.ndarray] = []
    for idx in candidates:
        crop, _bbox = _crop_cell(image, cells[idx], pad=2)
        if crop.size == 0:
            continue
        # 再内缩一圈，减少框线进 OCR
        ch, cw = crop.shape[:2]
        mx = min(max(3, cw // 8), max(0, cw // 3))
        my = min(max(3, ch // 8), max(0, ch // 3))
        inner = crop[my : max(my + 1, ch - my), mx : max(mx + 1, cw - mx)]
        if inner.size == 0:
            inner = crop
        for rot in (90, 270):
            if rot == 90:
                rotated = cv2.rotate(inner, cv2.ROTATE_90_CLOCKWISE)
            else:
                rotated = cv2.rotate(inner, cv2.ROTATE_90_COUNTERCLOCKWISE)
            crops.append(rotated)
            slots.append((idx, rot))

    if not crops:
        return cells

    canvas, rects, _scales = _pack_crops(crops, target_h=96, gap=20, max_width=1600)
    cache_extra = "vheader=" + hashlib.sha1(
        np.ascontiguousarray(canvas).tobytes()
    ).hexdigest()[:16]
    tbs = predict_texts(
        canvas,
        ocr_engine=ocr_engine,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        cache_extra=cache_extra,
    )

    by_cell: Dict[int, List[Tuple[float, str, List[Dict[str, Any]]]]] = {}
    for (cell_idx, _rot), rect in zip(slots, rects):
        hit = _boxes_in_rect(tbs, rect)
        if not hit:
            continue
        text = join_cell_texts(hit).strip()
        if not text:
            continue
        score = _avg_score(hit)
        by_cell.setdefault(cell_idx, []).append((score, text, hit))

    filled = 0
    for idx, opts in by_cell.items():
        cell = cells[idx]
        neighbors = set(_neighbor_texts(cells, cell))
        accepted: List[Tuple[float, str, List[Dict[str, Any]]]] = []
        for score, text, hit in opts:
            if not _looks_like_vertical_header_text(text, score):
                continue
            if _compact_header_text(text) in neighbors:
                continue
            accepted.append((score, text, hit))
        if not accepted:
            continue
        best_score, best_text, best_tbs = max(accepted, key=lambda x: (x[0], len(x[1])))
        cell["texts"] = best_tbs
        cell["text"] = best_text
        filled += 1
        logger.info(
            "竖排空表头补回 col=%s: %r (score=%.3f)",
            cell.get("col_start"),
            best_text,
            best_score,
        )

    if filled:
        logger.info("竖排空表头补检: candidates=%d filled=%d", len(candidates), filled)
    else:
        logger.info("竖排空表头补检: candidates=%d filled=0", len(candidates))
    return cells
