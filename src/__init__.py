"""复杂表格解耦提取 Pipeline 核心模块。"""

from __future__ import annotations

__all__ = ["extract_table_markdown", "extract_table_output"]


def __getattr__(name: str):
    if name == "extract_table_markdown":
        from .pipeline import extract_table_markdown

        return extract_table_markdown
    if name == "extract_table_output":
        from .pipeline import extract_table_output

        return extract_table_output
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
