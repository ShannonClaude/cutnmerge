"""行标签 / 数值等级等通用文本模式（结构拆分与 OCR 后处理共用）。"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

# 独立行标签：相邻行各自成格，不应因缺横线被 rowspan 粘在一起
_INDEPENDENT_LABEL_RE = re.compile(
    r"^(?:"
    r"(?:实[施試]例|实[施試]形態|実施例|比較例|比较例|合成例|对照例|参考例|態様)"
    r"[IVXivx０-９\d]+|"
    r"(?:iii|ii|i|III|II|I|ｌｉｉ|ｌｉ)[-－]?\d+|"
    r"(?:OXL|OXE|OX)[-－]?[A-Za-z0-9]+|"
    r"\d{1,3}"
    r")$",
    re.IGNORECASE,
)

# 数值 + 评价等级（叠放格常见）
_VALUE_GRADE_GLUED_RE = re.compile(
    r"^([<>]?\d+(?:\.\d+)?)([AB]\+?)$",
    re.IGNORECASE,
)
_VALUE_GRADE_SPACED_RE = re.compile(
    r"^([<>]?\d+(?:\.\d+)?)\s+([AB]\+?)$",
    re.IGNORECASE,
)

# iii-N 常见 OCR 误识
_III_OCR_RE = re.compile(
    r"(?i)\b(?:111|11[ilI]|ll[il1]|lii)\s*[-－]\s*(\d+)\b"
)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


def is_independent_row_label(text: str) -> bool:
    """单行文本是否像「实施例N / 合成例N / iii-N / OXL-* / 纯序号」。"""
    t = normalize_spaces(text)
    if not t:
        return False
    return bool(_INDEPENDENT_LABEL_RE.fullmatch(t))


def are_independent_row_labels(texts: Sequence[str]) -> bool:
    """多行文本是否均为独立行标签（应拆开 rowspan）。"""
    cleaned = [normalize_spaces(t) for t in texts if (t or "").strip()]
    if len(cleaned) < 2:
        return False
    return all(is_independent_row_label(t) for t in cleaned)


def split_value_grade(text: str) -> Optional[Tuple[str, str]]:
    """
    将「40A」「40 A」「100A+」拆成 (数值, 等级)。

    仅处理整段即为此模式的情况，避免误伤化学式。
    """
    t = (text or "").strip()
    if not t or "\n" in t:
        return None
    m = _VALUE_GRADE_GLUED_RE.fullmatch(t) or _VALUE_GRADE_SPACED_RE.fullmatch(t)
    if not m:
        return None
    grade = m.group(2)
    grade = grade[0].upper() + grade[1:]
    return m.group(1), grade


def fix_iii_ocr(text: str) -> str:
    """把 111-3 / 11i-5 等纠正为 iii-N。"""
    if not text:
        return text
    return _III_OCR_RE.sub(r"iii-\1", text)


def extract_independent_labels_from_joined(text: str) -> List[str]:
    """从已拼接的多行/粘连文本中抽出独立行标签序列。"""
    if not text:
        return []
    parts: List[str] = []
    for line in re.split(r"[\n|/]+", text):
        line = line.strip()
        if not line:
            continue
        # 同行无空格粘连：实施例51实施例52
        compact = normalize_spaces(line)
        ms = list(
            re.finditer(
                r"(?:实[施試]例|実[施試]例|実施例|比較例|比较例|合成例)"
                r"[IVXivx０-９\d]+|"
                r"(?:iii|ii|i)\s*[-－]?\s*\d+|"
                r"(?:OXL|OXE)\s*[-－]?\s*[A-Za-z0-9]+|"
                r"\d{1,3}(?!\d)",
                compact,
                flags=re.IGNORECASE,
            )
        )
        if len(ms) >= 2 and all(is_independent_row_label(m.group(0)) for m in ms):
            parts.extend(m.group(0) for m in ms)
        elif is_independent_row_label(line):
            parts.append(normalize_spaces(line))
        elif is_independent_row_label(compact):
            parts.append(compact)
    return parts
