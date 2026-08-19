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
    r"\([A-Z]{2}\)|"
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
# OCR 常把「40A」读成「A40」，「25A+」读成「25 +」
_VALUE_GRADE_REVERSED_RE = re.compile(
    r"^([AB]\+?)([<>]?\d{2,}(?:\.\d+)?)$",
    re.IGNORECASE,
)
_VALUE_GRADE_TRAILING_PLUS_RE = re.compile(
    r"^([<>]?\d+(?:\.\d+)?)\s*\+$",
)

_COMPONENT_HEADER_DONOR_RE = re.compile(
    r"^\(([A-Za-z])(\d+)\)[\n/]+第(\d+)(\S+)$",
    re.S,
)
_COMPONENT_HEADER_TARGET_RE = re.compile(
    r"^\(([A-Za-z])(\d+)\)(?:[\n/]+第(\d+))?$",
    re.S,
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


_ROW_INDEX_RE = re.compile(r"^\d{1,3}$")


def is_row_index_text(text: str) -> bool:
    """单格是否像表体行序号（1–3 位纯数字）。"""
    t = normalize_spaces(text)
    return bool(t) and bool(_ROW_INDEX_RE.fullmatch(t))


def is_index_column(
    texts: Sequence[str],
    *,
    min_nonempty: int = 3,
    min_ratio: float = 0.8,
) -> bool:
    """
    列级判定：多数非空格为行序号 → 不应当「碎片幽灵列」删除/合并。

    用于 drop_evidenceless_columns / merge_ghost_columns，避免窄序号列被吃掉。
    """
    nonempty = [str(t).strip() for t in texts if str(t or "").strip()]
    if len(nonempty) < min_nonempty:
        return False
    n_idx = sum(1 for t in nonempty if is_row_index_text(t))
    return (n_idx / len(nonempty)) >= min_ratio


def split_value_grade(text: str) -> Optional[Tuple[str, str]]:
    """
    将「40A」「40 A」「100A+」拆成 (数值, 等级)。

    仅处理整段即为此模式的情况，避免误伤化学式。
    """
    t = (text or "").strip()
    if not t or "\n" in t:
        return None
    m = _VALUE_GRADE_GLUED_RE.fullmatch(t) or _VALUE_GRADE_SPACED_RE.fullmatch(t)
    if m:
        grade = m.group(2)
        grade = grade[0].upper() + grade[1:]
        return m.group(1), grade
    m = _VALUE_GRADE_REVERSED_RE.fullmatch(t)
    if m:
        grade = m.group(1)
        grade = grade[0].upper() + grade[1:]
        return m.group(2), grade
    m = _VALUE_GRADE_TRAILING_PLUS_RE.fullmatch(t)
    if m:
        return m.group(1), "A+"
    return None


def complete_truncated_component_header(text: str, donors: Sequence[str]) -> str:
    """
    同行表头「(A1)/第1树脂」补全截断的「(A2)」或「(A2)/第2」。
    后缀取自完整 donor，不硬编码「树脂」。
    """
    t = (text or "").strip()
    split_m = re.fullmatch(
        r"^\(([A-Za-z])(\d+)\)[\n/]+第(\d+)[\n/]+(\S+)$",
        t,
        flags=re.S,
    )
    if split_m and split_m.group(2) == split_m.group(3):
        t = (
            f"({split_m.group(1)}{split_m.group(2)})\n"
            f"第{split_m.group(2)}{split_m.group(4)}"
        )
    tm = _COMPONENT_HEADER_TARGET_RE.fullmatch(t)
    if not tm:
        return t
    letter, num = tm.group(1), tm.group(2)
    partial = tm.group(3)
    if partial and partial != num:
        return t
    for donor in donors:
        dm = _COMPONENT_HEADER_DONOR_RE.fullmatch((donor or "").strip())
        if not dm:
            continue
        if dm.group(2) != dm.group(3):
            continue
        suffix = dm.group(4)
        if not suffix:
            continue
        return f"({letter}{num})\n第{num}{suffix}"
    return t


def fix_iii_ocr(text: str) -> str:
    """把 111-3 / 11i-5 等纠正为 iii-N。"""
    if not text:
        return text
    return _III_OCR_RE.sub(r"iii-\1", text)


# 表题/图题碎片：从中抽出的裸数字不能当作行标签（否则 [表1-2]+聚合物 会被拆成 1|2）
_CAPTION_CHUNK_RE = re.compile(
    r"\[?\s*(?:表|図|图)\s*[\d\-ー－]+\s*\]?",
    re.IGNORECASE,
)


def extract_independent_labels_from_joined(text: str) -> List[str]:
    """从已拼接的多行/粘连文本中抽出独立行标签序列。"""
    if not text:
        return []
    parts: List[str] = []
    for line in re.split(r"[\n|/]+", text):
        line = line.strip()
        if not line:
            continue
        # 去掉表题碎片后再解析，避免 [表1-2] 贡献假数字标签
        line_wo_cap = _CAPTION_CHUNK_RE.sub(" ", line).strip()
        if not line_wo_cap:
            continue
        # 同行无空格粘连：实施例51实施例52
        compact = normalize_spaces(line_wo_cap)
        # 多标签扫描：不含裸 \d{1,3}（裸数字只允许整行 fullmatch）
        ms = list(
            re.finditer(
                r"(?:实[施試]例|実[施試]例|実施例|比較例|比较例|合成例)"
                r"[IVXivx０-９\d]+|"
                r"(?:iii|ii|i)\s*[-－]?\s*\d+|"
                r"(?:OXL|OXE)\s*[-－]?\s*[A-Za-z0-9]+|"
                r"\([A-Z]{2}\)",
                compact,
                flags=re.IGNORECASE,
            )
        )
        if len(ms) >= 2 and all(is_independent_row_label(m.group(0)) for m in ms):
            covered = sum(len(m.group(0)) for m in ms)
            # 匹配必须覆盖绝大部分文本，避免「表题数字 + 正文」误抽
            if covered >= max(2, int(0.7 * len(compact))):
                parts.extend(m.group(0) for m in ms)
                continue
        if is_independent_row_label(line_wo_cap):
            parts.append(normalize_spaces(line_wo_cap))
        elif is_independent_row_label(compact):
            parts.append(compact)
    return parts
