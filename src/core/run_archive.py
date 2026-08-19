"""批量运行存档：记录成功/失败，下次默认只处理失败项与新文件。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

from .config import ROOT

DEFAULT_ARCHIVE_PATH = ROOT / "data" / "run_archive.json"

# 控制台/JSON 中截断过长异常，避免存档膨胀
_MAX_ERROR_LEN = 800


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def file_fingerprint(path: Path) -> Dict[str, int]:
    st = path.stat()
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def shorten_error(exc: BaseException, *, limit: int = _MAX_ERROR_LEN) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


class RunArchive:
    """以文件名为键的成功/失败清单，落盘 data/run_archive.json。"""

    def __init__(self, path: Path = DEFAULT_ARCHIVE_PATH) -> None:
        self.path = path
        self.items: Dict[str, Dict[str, Any]] = {}
        self.updated_at: str = ""

    def load(self) -> None:
        if not self.path.is_file():
            self.items = {}
            self.updated_at = ""
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.items = {}
            self.updated_at = ""
            return
        raw_items = data.get("items") if isinstance(data, dict) else None
        items: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_items, dict):
            for name, rec in raw_items.items():
                if isinstance(name, str) and isinstance(rec, dict):
                    items[name] = rec
        elif isinstance(raw_items, list):
            for rec in raw_items:
                if isinstance(rec, dict) and rec.get("name"):
                    items[str(rec["name"])] = rec
        self.items = items
        self.updated_at = str(data.get("updated_at") or "") if isinstance(data, dict) else ""

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now_iso()
        payload = {
            "version": 1,
            "updated_at": self.updated_at,
            "items": dict(sorted(self.items.items())),
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def prune_missing(self, present_names: Iterable[str]) -> int:
        """去掉输入目录里已经不存在的条目，返回删除数量。"""
        keep = set(present_names)
        gone = [name for name in self.items if name not in keep]
        for name in gone:
            del self.items[name]
        return len(gone)

    def _same_file(self, path: Path, rec: Dict[str, Any]) -> bool:
        try:
            fp = file_fingerprint(path)
        except OSError:
            return False
        return (
            int(rec.get("size") or -1) == fp["size"]
            and int(rec.get("mtime_ns") or -1) == fp["mtime_ns"]
        )

    def select(
        self,
        images: List[Path],
        *,
        force_all: bool = False,
        output_present: Callable[[Path], bool] | None = None,
    ) -> Tuple[List[Path], List[Path], Dict[str, int]]:
        """按存档筛选：失败 / 新文件 / 内容变化 / 产物缺失 → 处理；已成功且未改 → 跳过。

        output_present: 可选回调，传入后已成功且输入未变的图片还要检查
        产物是否还在；返回 False（产物缺失）则重新处理。
        """
        to_process: List[Path] = []
        skipped: List[Path] = []
        stats = {"new": 0, "retry": 0, "changed": 0, "force": 0, "out_missing": 0, "skip": 0}
        for path in images:
            rec = self.items.get(path.name)
            if force_all:
                to_process.append(path)
                stats["force"] += 1
                continue
            if rec is None:
                to_process.append(path)
                stats["new"] += 1
                continue
            status = str(rec.get("status") or "")
            if status == "failed":
                to_process.append(path)
                stats["retry"] += 1
                continue
            if status == "success" and not self._same_file(path, rec):
                to_process.append(path)
                stats["changed"] += 1
                continue
            if status == "success":
                if output_present is not None and not output_present(path):
                    to_process.append(path)
                    stats["out_missing"] += 1
                    continue
                skipped.append(path)
                stats["skip"] += 1
                continue
            to_process.append(path)
            stats["retry"] += 1
        return to_process, skipped, stats

    def mark_success(self, path: Path, *, save: bool = True) -> None:
        fp = file_fingerprint(path)
        self.items[path.name] = {
            "name": path.name,
            "status": "success",
            "size": fp["size"],
            "mtime_ns": fp["mtime_ns"],
            "error": None,
            "updated_at": _now_iso(),
        }
        if save:
            self.save()

    def mark_failed(self, path: Path, error: str) -> None:
        try:
            fp = file_fingerprint(path)
        except OSError:
            fp = {"size": 0, "mtime_ns": 0}
        text = (error or "").strip() or "unknown error"
        if len(text) > _MAX_ERROR_LEN:
            text = text[: _MAX_ERROR_LEN - 1] + "…"
        self.items[path.name] = {
            "name": path.name,
            "status": "failed",
            "size": fp["size"],
            "mtime_ns": fp["mtime_ns"],
            "error": text,
            "updated_at": _now_iso(),
        }
        self.save()

    def failed_records(self) -> List[Dict[str, Any]]:
        rows = [
            rec
            for rec in self.items.values()
            if str(rec.get("status") or "") == "failed"
        ]
        rows.sort(key=lambda rec: str(rec.get("name") or ""))
        return rows


def format_select_summary(stats: Dict[str, int], n_process: int, n_skip: int) -> str:
    bits: List[str] = []
    if stats.get("retry"):
        bits.append(f"失败重试 {stats['retry']}")
    if stats.get("new"):
        bits.append(f"新文件 {stats['new']}")
    if stats.get("changed"):
        bits.append(f"已改动 {stats['changed']}")
    if stats.get("force"):
        bits.append(f"强制全量 {stats['force']}")
    if stats.get("skip"):
        bits.append(f"跳过已成功 {stats['skip']}")
    if stats.get("out_missing"):
        bits.append(f"产物缺失 {stats['out_missing']}")
    detail = "，".join(bits) if bits else "无筛选"
    return f"存档筛选: {detail} → 本次处理 {n_process} 张，跳过 {n_skip} 张"
