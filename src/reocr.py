"""Re-run OCR on suspicious cells via a single packed montage."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .label_patterns import fix_iii_ocr, split_value_grade
from .matching import join_cell_texts, unmerge_filled_label_rowspans
from .models import predict_texts


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

        if new_score < old_score + 0.05:
            continue
        if len(new_text) < 0.6 * max(1, len(old_text)):
            continue

        col_template = templates.get(int(cell.get("col_start") or 0))
        if col_template and _char_skeleton(new_text) != col_template:
            continue

        cell["texts"] = new_tbs
        cell["text"] = new_text

    # 二次 OCR 后可能修好行标签，再拆一次错误 rowspan
    cells = unmerge_filled_label_rowspans(cells)
    return cells
