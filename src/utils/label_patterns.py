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


# 实施例/比较例 等行标签 + 右侧纯序号列（组合物表常见双列）
_EXAMPLE_ROW_HEAD = (
    r"(?:实[施試]例|実[施試]例|比較例|比较例|合成例|对照例|参考例)"
)
_CLEAN_EXAMPLE_LABEL_RE = re.compile(
    rf"^(?P<head>{_EXAMPLE_ROW_HEAD})\s*(?P<num>\d{{1,3}})\s*$"
)
# 「实施例 32 32」：标签内尾号与自身序号重复
_DUP_EXAMPLE_INDEX_RE = re.compile(
    rf"^(?P<head>{_EXAMPLE_ROW_HEAD})\s*(?P<num>\d{{1,3}})(?:\s+|\n+)(?P=num)\s*$"
)
# 「实施例 2 3」/「实施例 3 2」：序号被空格拆碎
_FRAG_EXAMPLE_INDEX_RE = re.compile(
    rf"^(?P<head>{_EXAMPLE_ROW_HEAD})\s+"
    rf"(?P<digits>(?:\d{{1,3}}(?:\s+|\n+))+\d{{1,3}})\s*$"
)


def parse_clean_example_label(text: str) -> Optional[Tuple[str, str]]:
    """「实施例 33」→ (head, num)；否则 None。"""
    m = _CLEAN_EXAMPLE_LABEL_RE.fullmatch((text or "").strip())
    if not m:
        return None
    return m.group("head"), m.group("num")


def best_digit_split_for_example_label(
    head: str,
    digits: str,
    *,
    local_frac: Optional[float] = None,
) -> Optional[int]:
    """
    在数字串上选切开点：左侧构成独立行标，右侧优先 2–3 位组合物编号。

    例：比较例 + 186 → split_at=1（1|86）；比较例 + 489 → 4|89。
    """
    if len(digits) < 2:
        return None
    ideal: Optional[int] = None
    if local_frac is not None:
        n = len(digits)
        idx = int(round(max(0.0, min(1.0, local_frac)) * n))
        ideal = max(1, min(n - 1, idx))
    scored: List[Tuple[int, int]] = []
    for split_at in range(1, len(digits)):
        left_d = digits[:split_at]
        right_d = digits[split_at:]
        label = f"{head}{left_d}"
        if not is_independent_row_label(label):
            continue
        score = 0
        if len(right_d) in (2, 3):
            score += 10
        elif len(right_d) == 1:
            score -= 2
        if len(left_d) <= 2:
            score += 3
        if right_d.startswith("0") and len(right_d) > 1:
            score -= 6
        if ideal is not None:
            score -= abs(split_at - ideal)
        scored.append((score, split_at))
    if not scored:
        return None
    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored[0][1]


def split_example_local_and_composition_id(
    text: str,
) -> Optional[Tuple[str, str]]:
    """
    从粘连行标中剥离「局部序号 | 组合物编号」。

    - 「比较例186」「比较例489」→ (「比较例 1」,「86」) / (「比较例 4」,「89」)
    - 「比较例 3 88」「比较例 1 86」→ (「比较例 3」,「88」) / (「比较例 1」,「86」)
    - 已干净的「比较例 1」「比较例 86」→ None（避免误剥）
    """
    lab = (text or "").strip()
    if not lab:
        return None

    spaced = re.fullmatch(
        rf"^(?P<head>{_EXAMPLE_ROW_HEAD})\s+"
        rf"(?P<local>\d{{1,3}})\s+(?P<comp>\d{{2,3}})\s*$",
        lab,
    )
    if spaced:
        head = spaced.group("head")
        local, comp = spaced.group("local"), spaced.group("comp")
        if not is_independent_row_label(f"{head}{local}"):
            return None
        return f"{head} {local}", comp

    clean = _CLEAN_EXAMPLE_LABEL_RE.fullmatch(lab)
    if not clean:
        return None
    head, num = clean.group("head"), clean.group("num")
    # 少于 3 位：比较例1 / 比较例86 —— 无法可靠区分「局部+组合物」
    if len(num) < 3:
        return None
    split_at = best_digit_split_for_example_label(head, num)
    if split_at is None:
        return None
    local, comp = num[:split_at], num[split_at:]
    if len(comp) not in (2, 3) or len(local) > 2:
        return None
    if not is_independent_row_label(f"{head}{local}"):
        return None
    return f"{head} {local}", comp


def repair_glued_example_label_index(
    label: str,
    index: str,
) -> Optional[Tuple[str, str]]:
    """
    纠正标签列与相邻序号列的串位（需列级多数干净对作门控后再调用）。

    - 「实施例 32 32」+ 空 → 「实施例 32」|「32」
    - 「实施例 32 32」+「32」→ 去掉标签尾部重复
    - 「实施例 2 3」+「23」/「实施例 3 2」+「32」→ 标签数字拼回与序号列一致

    组合物号粘连（比较例186|Bk-1）由 split_example_local_and_composition_id
    + html_formatter Path B 处理，不在此函数。
    """
    lab = (label or "").strip()
    idx = (index or "").strip()
    if not lab:
        return None

    dup = _DUP_EXAMPLE_INDEX_RE.fullmatch(lab)
    if dup:
        head, num = dup.group("head"), dup.group("num")
        fixed_lab = f"{head} {num}"
        if not idx:
            return fixed_lab, num
        if idx == num:
            return fixed_lab, idx
        return None

    if not is_row_index_text(idx):
        return None
    frag = _FRAG_EXAMPLE_INDEX_RE.fullmatch(lab)
    if not frag:
        return None
    digits = re.findall(r"\d{1,3}", frag.group("digits"))
    if len(digits) < 2:
        return None
    if "".join(digits) != idx:
        return None
    return f"{frag.group('head')} {idx}", idx


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
