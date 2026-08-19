from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_env

load_env()

from src.core.pipeline import _load_image, predict_texts, load_ocr, _extract_via_tsr
from src.output.html_formatter import _top_header_band_end, _column_has_body_content

inp = ROOT / "data/input"
img = next(p for p in inp.iterdir() if "P26X194" in p.name)
image = _load_image(str(img))
ocr = load_ocr()
text_boxes = predict_texts(image, ocr_engine=ocr, use_cache=True)
_, dbg = _extract_via_tsr(image, text_boxes, ioa_threshold=0.5, compress_empty_cols=True)
cells = dbg[0].cells
print("ncells", len(cells))
he = _top_header_band_end(cells)
print("header_end", he, "col0 body", _column_has_body_content(cells, 0, he))
for c in sorted(cells, key=lambda x: (int(x["row_start"]), int(x["col_start"]))):
    if int(c["row_start"]) <= 1:
        print(
            int(c["row_start"]),
            int(c["col_start"]),
            int(c["col_end"]),
            repr(str(c.get("text") or "")[:30]),
        )
