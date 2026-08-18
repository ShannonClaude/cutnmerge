# cutnmerge

复杂表格解耦提取：**结构识别**与 **OCR 文本** 分离处理，再按 IoA（交叠面积比）将文本归属到单元格，默认输出 **HTML**（保留 `rowspan`/`colspan`）+ **Markdown**（由最终 HTML 转换）。

适用于有框线/弱框线的专利表、实验参数表等截图。默认使用 RapidAI TableStructureRec（有线/无线分流）做结构，并经拓扑后处理与网格证据校验；`--structure lines` / `lore` 可切换结构来源。文本一律走 PaddleOCR 云端 API（本地预处理后上传、坐标映回原图），结果可本地缓存以免重复消耗额度，OCR 产物（json/csv/标注图）默认落盘 `data/ocr/`。

---

# 中文版

> 面向中国开发者。The English version is below [English Version](#english-version)。

## 一、模块区分

### 1. 入口与主流程（`src/core`）

| 文件 | 职责 |
|------|------|
| `main.py` | CLI 入口：参数解析、扫描 `data/input/`、批量调用 pipeline、按 `--format` 写结果 |
| `src/core/pipeline.py` | 主流程编排：方向归正 → 去偏斜 → 结构识别 → OCR → 网格证据校验 → IoA 填格 → 输出 |
| `src/core/config.py` | 从项目根目录加载 `.env`（AI Studio Token、模型名、超时、上传预处理/检测参数、REOCR/TSR_AGGRESSIVE 开关） |
| `src/core/models.py` | 模型加载与推断：ModelScope LORE（`--structure lore`）+ `predict_texts` 云端 OCR 封装 |

### 2. 预处理（`src/preprocess`）

| 文件 | 职责 |
|------|------|
| `src/preprocess/orient.py` | 方向归正：投影剖面定轴向（0/90/270），OCR 置信度消歧 180° |

### 3. 结构识别（`src/structure`）— 三选一，`--structure` 指定

| 文件 | 职责 |
|------|------|
| `src/structure/tsr.py` | **默认** TableStructureRec：`table_cls` 有线/无线分流 → `tsr_refine` 取单元格框与逻辑拓扑；`need_ocr=False`，OCR 仍走云端 |
| `src/structure/tsr_refine.py` | TSR 拓扑后处理：去重叠、幽灵列合并、错误 row/colspan 拆分、子行切分/容器格抑制 |
| `src/structure/grid_evidence.py` | **网格证据校验层**（TSR 路径核心）：在文本归属前，用“线证据 + 空白走廊证据”修正错分行/列边界，防止表头折行被切行、幽灵行列错位 |
| `src/structure/grid_fusion.py` | 将 TSR 拓扑与框线派生的网格分隔线融合（补回 TSR 漏掉的行/列边界） |
| `src/structure/lines.py` | 框线网格重建：基于 OpenCV 检测水平/竖分隔线（Separator），供 `lines` 结构模式与证据校验共用 |
| `src/structure/refine.py` | 框线网格（lines 路径）后处理：按 OCR 文本聚类补列、拆错误纵向合并 |
| `src/structure/hline_repair.py` | 表头带假横切修复：按列检测横线墨迹，局部恢复 rowspan |

> `--structure auto` 已废弃，行为等价于 `tsr`。

### 4. OCR 与文本后处理（`src/ocr`）

| 文件 | 职责 |
|------|------|
| `src/ocr/cloud_ocr.py` | **云端 OCR 全流程**：本地预处理（去透明合成、矮图短边放大、长边钳制、JPEG 编码）→ HTTP 提交 → 轮询 → 下载 JSONL → 解析文本框 → 坐标映回原图；重试/限速；产物（json/csv/标注图/API 原图）落盘 |
| `src/ocr/ocr_cache.py` | 本地缓存：按图片内容 hash + 模型配置落盘，避免重复消耗云端额度 |
| `src/ocr/ocr_post.py` | OCR 文本后处理：符号规范化（全半角、空格）与基于墨迹证据的幻觉清理 |
| `src/ocr/reocr.py` | 可疑单元格二次 OCR：将多个可疑格子拼成一张图一次性重识别（`--reocr` / `REOCR`） |

### 5. 匹配、输出与共用工具

| 文件 | 职责 |
|------|------|
| `src/matching/matching.py` | IoA 文本归属：跨列切分、行标签粘连拆分；多格命中取面积最小；单元格内文本排序拼接 |
| `src/output/formatter.py` | 表格外自由文本格式化（主流程 Markdown 已改由 html2md 从最终 HTML 转换） |
| `src/output/html2md.py` | 最终 **HTML → Markdown** 转换（BeautifulSoup 解析合并单元格、表头深度） |
| `src/output/html_formatter.py` | 基于逻辑拓扑的 **HTML** 表格生成（保留 `rowspan`/`colspan`） |
| `src/utils/geometry.py` | 坐标几何工具：多边形转换、IoA 计算 |
| `src/utils/segments.py` | 子表行段切分（多级表头不误切、表体后重复表头仍切），供结构修复与渲染共用 |
| `src/utils/label_patterns.py` | 行标签/数值等级等通用文本模式，结构拆分与 OCR 后处理共用 |

### 6. 工具与测试（`tools/`）

| 文件 | 职责 |
|------|------|
| `tools/regress.py` | 回归统计：cells / coverage / 丢字数（与 `data/debug/_regress_baseline.txt` 基线比对） |
| `tools/inspect_grid.py` | 网格结构检查（调试用） |
| `tools/run_hline_verify.py` | 在 P96/P97/P98/P109/P135 上跑 pipeline 并打印 HTML 检查摘要 |
| `tools/test_*.py` | 单元/回归测试：hline_repair、窄序号列保留、OCR 文本修复、行段切分、sticky 行标粘连切分 |

## 二、安装与环境

- Python 3.10+（建议）
- 可选：NVIDIA GPU + CUDA（ModelScope LORE 更快；无 GPU 走 CPU 兼容路径）
- 需要[百度 AI Studio](https://aistudio.baidu.com/account/accessToken) Access Token（云端 OCR）

```bash
cd cutnmerge
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
# source venv/bin/activate

pip install -r requirements.txt
```

首次使用 TableStructureRec（默认）或 ModelScope LORE（`--structure lore`）时会下载模型，需可访问外网。

## 三、配置

```bash
cp .env.example .env
# 编辑 .env，至少填写 PADDLEOCR_ACCESS_TOKEN
```

| 变量 | 说明 | 默认 |
|------|------|------|
| `PADDLEOCR_ACCESS_TOKEN` | AI Studio Token（**必填**） | — |
| `PADDLEOCR_BASE_URL` | 自定义 API 地址 | 官方默认 |
| `PADDLEOCR_TASK` | `ocr`（当前云端路径仅支持 `ocr`，细粒度文本框供 IoA 填格；`doc_parsing` 已不再支持） | `ocr` |
| `PADDLEOCR_OCR_MODEL` | OCR 任务模型名 | `PP-OCRv6` |
| `PADDLEOCR_REQUEST_TIMEOUT` | 请求超时（秒） | `300` |
| `PADDLEOCR_POLL_TIMEOUT` | 轮询超时（秒） | `600` |
| `PADDLEOCR_JOBS_URL` | 自定义 OCR job API 地址 | 官方默认 |
| `PADDLEOCR_PREPROCESS_MAX_LONG_SIDE` | 上传前本地预处理：长边钳制上限 | `2200` |
| `PADDLEOCR_PREPROCESS_MIN_SHORT_SIDE` | 上传前本地预处理：矮图短边放大下限（利于小字召回） | `720` |
| `PADDLEOCR_PREPROCESS_JPEG_QUALITY` | 上传 JPEG 质量 | `90` |
| `PADDLEOCR_SAVE_ARTIFACTS` | OCR 产物（json/csv/标注图/API 原图）落盘开关 | `true` |
| `PADDLEOCR_ARTIFACT_DIR` | 产物目录 | `data/ocr/` |
| `PADDLEOCR_TEXT_DET_LIMIT_TYPE` | 检测边长模式 `min`/`max`（留空不传）；`min` 且投影长边超云端上限(~4000)时自动钳制 | — |
| `PADDLEOCR_TEXT_DET_LIMIT_SIDE_LEN` | 检测边长阈值 | — |
| `PADDLEOCR_TEXT_DET_THRESH` / `PADDLEOCR_TEXT_DET_BOX_THRESH` / `PADDLEOCR_TEXT_DET_UNCLIP_RATIO` | 文本检测阈值参数 | — |
| `PADDLEOCR_TEXT_REC_SCORE_THRESH` | 文本识别置信度阈值 | — |
| `PADDLEOCR_MARKDOWN_IGNORE_LABELS` | 忽略的版面标签（逗号分隔） | — |
| `PADDLEOCR_USE_DOC_ORIENTATION_CLASSIFY` / `PADDLEOCR_USE_DOC_UNWARPING` / `PADDLEOCR_USE_TEXTLINE_ORIENTATION` | 随请求上传的检测开关（方向分类/去卷曲/文本行方向） | `false` |
| `PADDLEOCR_USE_LAYOUT_DETECTION` / `PADDLEOCR_USE_CHART_RECOGNITION` / `PADDLEOCR_PRETTIFY_MARKDOWN` | doc_parsing（VL）遗留开关，当前 HTTP 路径不生效 | — |
| `REOCR` | 可疑单元格二次 OCR 总开关（费时/耗 API） | `false` |
| `REOCR_MAX_CELLS` | 每张图最多二次 OCR 的单元格数 | `24` |
| `TSR_AGGRESSIVE` | 激进结构后处理（补列/重建表头/横切表头）；`false` 表示信任 TableStructureRec 拓扑 | `false` |

## 四、使用说明（CLI）

将表格图片放入 `data/input/`（支持 png / jpg / jpeg / bmp / tif / tiff / webp）：

```bash
# 批量：扫描 data/input/，写出 data/output/<同名>.html + .md + 两张彩图
python main.py

# 单图
python main.py --image data/input/demo.png
python main.py --image data/input/demo.png --output data/output/demo.html

# 输出格式（默认 both：html + md，md 由最终 HTML 经 html2md 转换）
python main.py --format both    # 默认：html + md
python main.py --format html    # 仅 html，保留 rowspan/colspan
python main.py --format md      # 仅 md

# 结构来源（模块区分见上文）
python main.py --structure tsr     # 默认 TableStructureRec（有线/无线分流 + 拓扑后处理 + 网格证据校验）
python main.py --structure lines   # 框线网格（对照/debug）
python main.py --structure lore    # ModelScope LORE
python main.py --structure tsr --tsr-kind wired   # 强制有线引擎
python main.py --structure tsr --tsr-kind lineless # 强制无线引擎
python main.py --structure tsr --fallback-lines   # TSR 覆盖率过低时回退框线

# 结构后处理力度
python main.py --tsr-light        # 默认：信任库拓扑（轻量路径）
python main.py --tsr-aggressive   # 激进后处理：补列/重建表头/横切表头

# 可疑单元格二次 OCR（费时/耗 API）
python main.py --reocr
python main.py --reocr --reocr-max-cells 12
python main.py --no-reocr                # 显式关闭（覆盖 .env REOCR）

# 建议用同一批失败样例对比结构来源
python main.py --structure lines --debug
python main.py --structure tsr --debug

# 其它常用参数
python main.py --ioa-threshold 0.5          # 文本归属 IoA 阈值
python main.py --no-deskew                   # 关闭倾斜校正
python main.py --max-skew-angle 15           # Deskew 最大校正角（度）
python main.py --orientation auto            # 默认；也可 none / 0 / 90 / 180 / 270
python main.py --no-cache                    # 禁用 OCR 缓存，每次请求云端
python main.py --refresh-cache               # 强制重拉 OCR 并覆盖缓存
python main.py --keep-empty-cols             # 保留整列为空的列（默认压缩删除）
python main.py --debug                       # 网格叠加图写到 data/debug/
python main.py --no-vis                      # 关闭表格划线可视化（默认写 *_table_vis.png / *_table_vis_logic.png）
```

回归统计：

```bash
python tools/regress.py
python tools/regress.py --refresh-cache --baseline data/debug/_regress_baseline.txt
```

## 五、目录结构

```
cutnmerge/
├── main.py              # CLI 入口
├── requirements.txt
├── .env.example
├── tools/
│   ├── regress.py       # 回归统计
│   ├── inspect_grid.py  # 网格检查
│   ├── run_hline_verify.py
│   └── test_*.py        # 单元/回归测试
├── data/
│   ├── input/           # 待处理图片
│   ├── output/          # HTML / Markdown 结果
│   ├── cache/           # OCR 本地缓存（gitignore）
│   ├── ocr/             # OCR 产物：json/csv/标注图（gitignore）
│   └── debug/           # --debug 叠加图（gitignore）
└── src/
    ├── core/            # 编排、配置、模型加载
    │   ├── pipeline.py
    │   ├── config.py
    │   └── models.py
    ├── preprocess/      # 方向归正
    │   └── orient.py
    ├── structure/       # 表格拓扑
    │   ├── tsr.py
    │   ├── tsr_refine.py
    │   ├── grid_evidence.py
    │   ├── grid_fusion.py
    │   ├── lines.py
    │   ├── refine.py
    │   └── hline_repair.py
    ├── ocr/             # 云端 OCR 与文本后处理
    │   ├── cloud_ocr.py
    │   ├── ocr_cache.py
    │   ├── ocr_post.py
    │   └── reocr.py
    ├── matching/        # IoA 填格
    │   └── matching.py
    ├── output/          # HTML / Markdown 渲染
    │   ├── formatter.py
    │   ├── html2md.py
    │   └── html_formatter.py
    └── utils/           # 几何、文本模式、行段切分
        ├── geometry.py
        ├── label_patterns.py
        └── segments.py
```

## 六、代码调用

```python
from src.core.pipeline import extract_table_output

out = extract_table_output(
    "data/input/demo.png",
    structure="tsr",
    ioa_threshold=0.5,
)
print(out["html"])
# out["md"] 为展开式 Markdown（可选）
```

入口会通过 `src.core.config.load_env()` 加载项目根目录 `.env`。

## 七、依赖说明

| 用途 | 包 |
|------|-----|
| ModelScope LORE | `modelscope`、`torch` 等 |
| TableStructureRec | `wired_table_rec`、`lineless_table_rec`、`table_cls` |
| 云端 OCR | `paddleocr`（官方客户端，无需本地 PaddlePaddle）、`requests`（HTTP 提交/轮询） |
| 图像 / 几何 | `opencv-python`、`numpy`、`shapely` |
| 配置 | `python-dotenv` |

`tsr` 默认 `need_ocr=False`，不依赖 `rapidocr`；若个别环境仍报缺 RapidOCR，可额外 `pip install rapidocr`。

## 八、常见问题

**未配置 Token**  
确保 `.env` 中 `PADDLEOCR_ACCESS_TOKEN` 已填写且不是占位符 `your-access-token-here`。

**Windows 控制台乱码**  
结果文件为 UTF-8；控制台可能无法打印部分字符，以 `data/output/*.html` 为准。

**中文路径图片**  
已通过 `np.fromfile` + `imdecode` 处理，可直接使用含中文的文件名路径。

**想反复调结构参数**  
默认启用 OCR 缓存；调 `--structure` / IoA 等时不必加 `--refresh-cache`，可避免重复扣额度。

**结构结果差怎么排查**  
按“结构来源对比 + 证据校验 + 后处理力度”逐层试：`--structure lines --debug` → `--structure tsr --debug` → `--tsr-aggressive`；对比 `data/debug/*_grid.png` 叠加图。

---

---

# English Version

> For international developers. 中文版见上文 [中文版](#中文版).

## 1. Module Map

### Entry & Main Pipeline (`src/core`)

| File | Responsibility |
|------|----------------|
| `main.py` | CLI entry: argument parsing, batch scan of `data/input/`, calls pipeline, writes results per `--format` |
| `src/core/pipeline.py` | Orchestrates the full flow: orientation → deskew → structure → OCR → grid evidence check → IoA matching → output |
| `src/core/config.py` | Loads `.env` from the project root (token, model names, timeouts, upload-preprocess / detection params, REOCR/TSR_AGGRESSIVE flags) |
| `src/core/models.py` | Model loading & inference: ModelScope LORE (`--structure lore`) + `predict_texts` cloud OCR wrapper |

### Preprocessing (`src/preprocess`)

| File | Responsibility |
|------|----------------|
| `src/preprocess/orient.py` | Orientation normalization: projection profile picks axis (0/90/270), OCR confidence disambiguates 180° |

### Structure Recognition (`src/structure`) — choose one via `--structure`

| File | Responsibility |
|------|----------------|
| `src/structure/tsr.py` | **Default** TableStructureRec: `table_cls` wired/lineless split → cell boxes + logical topology via `tsr_refine`; `need_ocr=False`, OCR still goes to the cloud |
| `src/structure/tsr_refine.py` | TSR topology post-processing: overlap removal, ghost-column merge, wrong row/colspan splitting, sub-row splitting / container-cell suppression |
| `src/structure/grid_evidence.py` | **Grid evidence validation** (core of the TSR path): before text matching, fixes wrong row/column boundaries using "line evidence + blank-corridor evidence" — prevents header wrapping from being split into rows and ghost row/column misalignment |
| `src/structure/grid_fusion.py` | Fuses TSR topology with line-derived grid separators (recovers row/column boundaries TSR missed) |
| `src/structure/lines.py` | Line-grid reconstruction: OpenCV-based detection of horizontal/vertical separators; shared by `lines` structure mode and evidence checking |
| `src/structure/refine.py` | Line-grid post-processing: adds missing columns by OCR text clustering, splits wrong vertical merges |
| `src/structure/hline_repair.py` | Fake header cross-line repair: detects horizontal ink per column, locally restores `rowspan` |

> `--structure auto` is deprecated; it behaves identically to `tsr`.

### OCR & Text Post-Processing (`src/ocr`)

| File | Responsibility |
|------|----------------|
| `src/ocr/cloud_ocr.py` | **Full cloud OCR flow**: local preprocessing (alpha compositing, short-side upscale, long-side clamp, JPEG) → HTTP submit → poll → JSONL download → box parsing → coordinates mapped back to the original image; retry/rate-limiting; artifacts (json/csv/annotated/API image) written to disk |
| `src/ocr/ocr_cache.py` | Local cache keyed by image content hash + model config, avoids burning cloud quota |
| `src/ocr/ocr_post.py` | OCR text normalization (full/half-width, spaces) and ink-density-based hallucination cleanup |
| `src/ocr/reocr.py` | Second-pass OCR for suspicious cells: packs them into one montage image for a single re-recognition (`--reocr` / `REOCR`) |

### Matching, Output & Shared Utils

| File | Responsibility |
|------|----------------|
| `src/matching/matching.py` | IoA text assignment: cross-column splits, row-label sticking splits; multi-cell hits take the smallest area; sorts text inside cells |
| `src/output/formatter.py` | Free-text formatting (text outside tables; the pipeline's Markdown is now produced by html2md from the final HTML) |
| `src/output/html2md.py` | Final **HTML → Markdown** conversion (BeautifulSoup: merged cells, header depth) |
| `src/output/html_formatter.py` | Logical-topology **HTML** generation (preserves `rowspan`/`colspan`) |
| `src/utils/geometry.py` | Coordinate geometry utilities: polygon conversion, IoA computation |
| `src/utils/segments.py` | Sub-table row-segment splitting (multi-level headers not wrongly cut; repeated headers after body still split); shared by repair and rendering |
| `src/utils/label_patterns.py` | Generic text patterns (row labels, numeric grades) shared by structure splitting and OCR post-processing |

### Tools & Tests (`tools/`)

| File | Responsibility |
|------|----------------|
| `tools/regress.py` | Regression stats: cells / coverage / dropped chars, diffed against the `data/debug/_regress_baseline.txt` baseline |
| `tools/inspect_grid.py` | Grid-structure inspection (debugging) |
| `tools/run_hline_verify.py` | Runs the pipeline on P96/P97/P98/P109/P135 and prints a short HTML check summary |
| `tools/test_*.py` | Unit/regression tests: hline_repair, narrow index-column preservation, OCR text-fix, row-segment splitting, sticky row-label splitting |

## 2. Installation & Environment

- Python 3.10+ (recommended)
- Optional: NVIDIA GPU + CUDA (faster ModelScope LORE; CPU-compatible fallback without GPU)
- A [Baidu AI Studio](https://aistudio.baidu.com/account/accessToken) Access Token (cloud OCR)

```bash
cd cutnmerge
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
# source venv/bin/activate

pip install -r requirements.txt
```

Models are downloaded on first use of TableStructureRec (default) or ModelScope LORE (`--structure lore`); internet access is required.

## 3. Configuration

```bash
cp .env.example .env
# Edit .env — at minimum set PADDLEOCR_ACCESS_TOKEN
```

| Variable | Description | Default |
|----------|-------------|---------|
| `PADDLEOCR_ACCESS_TOKEN` | AI Studio Token (**required**) | — |
| `PADDLEOCR_BASE_URL` | Custom API base URL | official default |
| `PADDLEOCR_TASK` | `ocr` (the cloud path only supports `ocr` — fine-grained boxes for IoA filling; `doc_parsing` is no longer supported) | `ocr` |
| `PADDLEOCR_OCR_MODEL` | OCR task model | `PP-OCRv6` |
| `PADDLEOCR_REQUEST_TIMEOUT` | Request timeout (s) | `300` |
| `PADDLEOCR_POLL_TIMEOUT` | Poll timeout (s) | `600` |
| `PADDLEOCR_JOBS_URL` | Custom OCR job API URL | official default |
| `PADDLEOCR_PREPROCESS_MAX_LONG_SIDE` | Pre-upload preprocessing: long-side clamp | `2200` |
| `PADDLEOCR_PREPROCESS_MIN_SHORT_SIDE` | Pre-upload preprocessing: short-side upscale floor for short images (better small-text recall) | `720` |
| `PADDLEOCR_PREPROCESS_JPEG_QUALITY` | Upload JPEG quality | `90` |
| `PADDLEOCR_SAVE_ARTIFACTS` | Master switch for writing OCR artifacts (json/csv/annotated/API image) | `true` |
| `PADDLEOCR_ARTIFACT_DIR` | Artifact directory | `data/ocr/` |
| `PADDLEOCR_TEXT_DET_LIMIT_TYPE` | Detection side mode `min`/`max` (empty = not sent); with `min`, auto-clamps if the projected long side would hit the cloud limit (~4000) | — |
| `PADDLEOCR_TEXT_DET_LIMIT_SIDE_LEN` | Detection side length | — |
| `PADDLEOCR_TEXT_DET_THRESH` / `PADDLEOCR_TEXT_DET_BOX_THRESH` / `PADDLEOCR_TEXT_DET_UNCLIP_RATIO` | Text-detection thresholds | — |
| `PADDLEOCR_TEXT_REC_SCORE_THRESH` | Text-recognition score threshold | — |
| `PADDLEOCR_MARKDOWN_IGNORE_LABELS` | Layout labels to ignore (comma-separated) | — |
| `PADDLEOCR_USE_DOC_ORIENTATION_CLASSIFY` / `PADDLEOCR_USE_DOC_UNWARPING` / `PADDLEOCR_USE_TEXTLINE_ORIENTATION` | Detection toggles sent with the request (orientation / unwarping / textline orientation) | `false` |
| `PADDLEOCR_USE_LAYOUT_DETECTION` / `PADDLEOCR_USE_CHART_RECOGNITION` / `PADDLEOCR_PRETTIFY_MARKDOWN` | Leftovers from the doc_parsing (VL) path; no effect on the current HTTP path | — |
| `REOCR` | Master switch for second-pass OCR on suspicious cells (slow / costs API quota) | `false` |
| `REOCR_MAX_CELLS` | Max cells re-OCR'd per image | `24` |
| `TSR_AGGRESSIVE` | Aggressive structure post-processing (fill columns / rebuild header / cross-line header); `false` = trust TableStructureRec topology | `false` |

## 4. Usage (CLI)

Put table images into `data/input/` (png / jpg / jpeg / bmp / tif / tiff / webp):

```bash
# Batch: scan data/input/, write data/output/<same-name>.html + .md + two colored visualizations
python main.py

# Single image
python main.py --image data/input/demo.png
python main.py --image data/input/demo.png --output data/output/demo.html

# Output format (default both: html + md; md is converted from the final HTML by html2md)
python main.py --format both    # default: html + md
python main.py --format html    # html only, preserves rowspan/colspan
python main.py --format md      # md only

# Structure source (see Module Map above)
python main.py --structure tsr     # default: TableStructureRec (wired/lineless + topology + grid evidence)
python main.py --structure lines   # line grid (comparison/debug)
python main.py --structure lore    # ModelScope LORE
python main.py --structure tsr --tsr-kind wired     # force wired engine
python main.py --structure tsr --tsr-kind lineless  # force lineless engine
python main.py --structure tsr --fallback-lines     # fall back to line grid if TSR coverage too low

# Structure post-processing strength
python main.py --tsr-light        # default: trust library topology (lightweight path)
python main.py --tsr-aggressive   # aggressive: fill columns / rebuild header / cross-line header

# Second-pass OCR for suspicious cells (slow / costs API quota)
python main.py --reocr
python main.py --reocr --reocr-max-cells 12
python main.py --no-reocr                # explicitly disable (overrides .env REOCR)

# Compare structure sources on the same failing samples
python main.py --structure lines --debug
python main.py --structure tsr --debug

# Other common options
python main.py --ioa-threshold 0.5          # IoA threshold for text-to-cell assignment
python main.py --no-deskew                   # disable skew correction
python main.py --max-skew-angle 15           # max deskew angle (degrees)
python main.py --orientation auto            # default; also none / 0 / 90 / 180 / 270
python main.py --no-cache                    # disable OCR cache, always hit the cloud
python main.py --refresh-cache               # force re-OCR and overwrite cache
python main.py --keep-empty-cols             # keep fully-empty columns (dropped by default)
python main.py --debug                       # write grid overlay images to data/debug/
python main.py --no-vis                      # disable table-line visualization (default writes *_table_vis.png / *_table_vis_logic.png)
```

Regression stats:

```bash
python tools/regress.py
python tools/regress.py --refresh-cache --baseline data/debug/_regress_baseline.txt
```

## 5. Directory Layout

```
cutnmerge/
├── main.py              # CLI entry
├── requirements.txt
├── .env.example
├── tools/
│   ├── regress.py       # regression stats
│   ├── inspect_grid.py  # grid inspection
│   ├── run_hline_verify.py
│   └── test_*.py        # unit/regression tests
├── data/
│   ├── input/           # images to process
│   ├── output/          # HTML / Markdown results
│   ├── cache/           # local OCR cache (gitignored)
│   ├── ocr/             # OCR artifacts: json/csv/annotated images (gitignored)
│   └── debug/           # --debug overlay images (gitignored)
└── src/
    ├── core/            # orchestration, config, model loading
    │   ├── pipeline.py
    │   ├── config.py
    │   └── models.py
    ├── preprocess/      # orientation
    │   └── orient.py
    ├── structure/       # table topology
    │   ├── tsr.py
    │   ├── tsr_refine.py
    │   ├── grid_evidence.py
    │   ├── grid_fusion.py
    │   ├── lines.py
    │   ├── refine.py
    │   └── hline_repair.py
    ├── ocr/             # cloud OCR and text post-processing
    │   ├── cloud_ocr.py
    │   ├── ocr_cache.py
    │   ├── ocr_post.py
    │   └── reocr.py
    ├── matching/        # IoA cell filling
    │   └── matching.py
    ├── output/          # HTML / Markdown rendering
    │   ├── formatter.py
    │   ├── html2md.py
    │   └── html_formatter.py
    └── utils/           # geometry, text patterns, row segments
        ├── geometry.py
        ├── label_patterns.py
        └── segments.py
```

## 6. Library Use

```python
from src.core.pipeline import extract_table_output

out = extract_table_output(
    "data/input/demo.png",
    structure="tsr",
    ioa_threshold=0.5,
)
print(out["html"])
# out["md"] is the unrolled Markdown (optional)
```

The entry point loads the project-root `.env` via `src.core.config.load_env()`.

## 7. Dependencies

| Purpose | Packages |
|---------|----------|
| ModelScope LORE | `modelscope`, `torch`, etc. |
| TableStructureRec | `wired_table_rec`, `lineless_table_rec`, `table_cls` |
| Cloud OCR | `paddleocr` (official client; no local PaddlePaddle needed), `requests` (HTTP submit/poll) |
| Image / geometry | `opencv-python`, `numpy`, `shapely` |
| Config | `python-dotenv` |

`tsr` runs with `need_ocr=False` by default and does not depend on `rapidocr`; if a particular environment still complains about RapidOCR, install it separately: `pip install rapidocr`.

## 8. FAQ

**Token not configured**  
Make sure `PADDLEOCR_ACCESS_TOKEN` in `.env` is filled in and not the placeholder `your-access-token-here`.

**Windows console garbled output**  
Result files are UTF-8; the console may fail to print some characters — rely on `data/output/*.html`.

**Images with non-ASCII paths**  
Handled via `np.fromfile` + `imdecode`; Chinese filenames work directly.

**Iterating on structure parameters**  
OCR caching is on by default; no need to add `--refresh-cache` when tuning `--structure` / IoA, avoiding repeated quota consumption.

**Debugging poor structure results**  
Try layers in order: `--structure lines --debug` → `--structure tsr --debug` → `--tsr-aggressive`; compare the overlay images in `data/debug/*_grid.png`.
