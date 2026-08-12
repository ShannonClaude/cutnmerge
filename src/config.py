"""从项目根目录加载 .env 配置。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
_ENV_LOADED = False


def load_env(dotenv_path: Path | None = None, *, override: bool = False) -> None:
    """加载 .env（幂等；override=True 时强制重新读取）。"""
    global _ENV_LOADED
    if _ENV_LOADED and not override:
        return
    path = dotenv_path or (ROOT / ".env")
    load_dotenv(path, override=override)
    _ENV_LOADED = True


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    return float(value)


def _as_optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def _as_csv_list(value: str | None) -> list[str]:
    if value is None or value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings:
    """PaddleOCR 云端相关配置。"""

    def __init__(self) -> None:
        load_env()
        self.access_token: str = (os.getenv("PADDLEOCR_ACCESS_TOKEN") or "").strip()
        self.base_url: str = (os.getenv("PADDLEOCR_BASE_URL") or "").strip()
        # ocr | doc_parsing
        self.task: str = (os.getenv("PADDLEOCR_TASK") or "ocr").strip().lower()
        self.ocr_model: str = (os.getenv("PADDLEOCR_OCR_MODEL") or "PP-OCRv6").strip()
        self.vl_model: str = (
            os.getenv("PADDLEOCR_VL_MODEL") or "PaddleOCR-VL-1.6"
        ).strip()
        self.request_timeout: float = _as_float(
            os.getenv("PADDLEOCR_REQUEST_TIMEOUT"), 300.0
        )
        self.poll_timeout: float = _as_float(
            os.getenv("PADDLEOCR_POLL_TIMEOUT"), 600.0
        )
        self.use_layout_detection: bool = _as_bool(
            os.getenv("PADDLEOCR_USE_LAYOUT_DETECTION"), True
        )
        self.use_doc_orientation_classify: bool = _as_bool(
            os.getenv("PADDLEOCR_USE_DOC_ORIENTATION_CLASSIFY"), False
        )
        self.use_doc_unwarping: bool = _as_bool(
            os.getenv("PADDLEOCR_USE_DOC_UNWARPING"), False
        )
        self.use_chart_recognition: bool = _as_bool(
            os.getenv("PADDLEOCR_USE_CHART_RECOGNITION"), False
        )
        self.prettify_markdown: bool = _as_bool(
            os.getenv("PADDLEOCR_PRETTIFY_MARKDOWN"), False
        )
        self.use_textline_orientation: bool = _as_bool(
            os.getenv("PADDLEOCR_USE_TEXTLINE_ORIENTATION"), False
        )
        self.text_det_thresh: float | None = _as_optional_float(
            os.getenv("PADDLEOCR_TEXT_DET_THRESH")
        )
        self.text_det_box_thresh: float | None = _as_optional_float(
            os.getenv("PADDLEOCR_TEXT_DET_BOX_THRESH")
        )
        self.text_det_unclip_ratio: float | None = _as_optional_float(
            os.getenv("PADDLEOCR_TEXT_DET_UNCLIP_RATIO")
        )
        self.text_rec_score_thresh: float | None = _as_optional_float(
            os.getenv("PADDLEOCR_TEXT_REC_SCORE_THRESH")
        )
        self.markdown_ignore_labels: list[str] = _as_csv_list(
            os.getenv("PADDLEOCR_MARKDOWN_IGNORE_LABELS")
        )

    def require_token(self) -> str:
        if not self.access_token or self.access_token == "your-access-token-here":
            raise RuntimeError(
                "未配置有效的 PADDLEOCR_ACCESS_TOKEN。"
                "请在项目根目录 .env 中填写 AI Studio Access Token。"
            )
        return self.access_token


def get_settings() -> Settings:
    return Settings()
