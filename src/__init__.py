"""复杂表格解耦提取 Pipeline 核心模块。"""

from __future__ import annotations

__all__ = ["extract_table_markdown"]


def __getattr__(name: str):
    if name == "extract_table_markdown":
        from .pipeline import extract_table_markdown

        return extract_table_markdown
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
