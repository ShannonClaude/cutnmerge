# cutnmerge

复杂表格解耦提取：结构识别与 OCR 文本分离，再按 IoA 归属填格，默认输出 **HTML**（保留 `rowspan`/`colspan`）。

适用于有框线/弱框线的专利表、实验参数表等截图。默认使用 RapidAI TableStructureRec（有线/无线分流）做结构，并经拓扑后处理；`--structure lines` 仍可用于对照。文本一律走 PaddleOCR 云端 API，结果可本地缓存以免重复消耗额度。

## 流程概览

1. **Orientation**（可选，默认 auto）：OCR 框形态判定是否需转 90°，置信度消歧 180°  
2. **Deskew**（可选）：霍夫倾斜校正  
3. **OCR**：云端 PP-OCR，命中 `data/cache/` 则跳过；经符号规范化 / 墨迹密度幻觉过滤  
4. **结构**：默认 `tsr`（`table_cls` → wired/lineless + `tsr_refine`，含子行切分 / 容器格抑制）；可选 `lines` / `lore`  
5. **匹配**：文本框与单元格 IoA 归属（跨列切分 / 行标签粘连拆分；多格命中取面积最小）  
6. **输出**：HTML 表格写入 `data/output/`（`--format md|both` 可导出 Markdown）

`tsr` 模式只取单元格框与逻辑拓扑；**不用**库内 RapidOCR / `pred_html`，OCR 仍用云端，HTML 由本仓库自拼。

## 环境要求

- Python 3.10+（建议）
- 可选：NVIDIA GPU + CUDA（ModelScope LORE 更快；无 GPU 时会走 CPU 兼容路径）
- [百度 AI Studio](https://aistudio.baidu.com/account/accessToken) Access Token（云端 OCR）

## 安装

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

## 配置

```bash
cp .env.example .env
# 编辑 .env，至少填写 PADDLEOCR_ACCESS_TOKEN
```

| 变量 | 说明 | 默认 |
|------|------|------|
| `PADDLEOCR_ACCESS_TOKEN` | AI Studio Token（必填） | — |
| `PADDLEOCR_BASE_URL` | 自定义 API 地址 | 官方默认 |
| `PADDLEOCR_TASK` | `ocr` 或 `doc_parsing` | `ocr` |
| `PADDLEOCR_OCR_MODEL` | OCR 模型名 | `PP-OCRv6` |
| `PADDLEOCR_VL_MODEL` | 版面解析模型名 | `PaddleOCR-VL-1.6` |
| `PADDLEOCR_REQUEST_TIMEOUT` | 请求超时（秒） | `300` |
| `PADDLEOCR_POLL_TIMEOUT` | 轮询超时（秒） | `600` |

其余 `PADDLEOCR_USE_*` / `PADDLEOCR_PRETTIFY_MARKDOWN` 主要影响 `doc_parsing`，见 `.env.example`。

填格场景建议保持 `PADDLEOCR_TASK=ocr`，以拿到细粒度文本框。

## 用法

将表格图片放入 `data/input/`（支持 png / jpg / jpeg / bmp / tif / tiff / webp）：

```bash
# 批量：扫描 data/input/，写出 data/output/<同名>.html
python main.py

# 单图
python main.py --image data/input/demo.png
python main.py --image data/input/demo.png --output data/output/demo.html

# 输出格式
python main.py --format html    # 默认
python main.py --format md
python main.py --format both

# 结构来源
python main.py --structure tsr     # 默认 TableStructureRec
python main.py --structure lines   # 框线网格（对照/debug）
python main.py --structure lore    # ModelScope LORE
# --structure auto 已废弃，行为等价于 tsr

python main.py --structure tsr --fallback-lines   # TSR 覆盖率过低时回退框线

# 建议用同一批失败样例对比
python main.py --structure lines --debug
python main.py --structure tsr --debug

# 其它常用参数
python main.py --ioa-threshold 0.5
python main.py --no-deskew
python main.py --max-skew-angle 15
python main.py --orientation auto   # 默认；也可 none / 0 / 90 / 180 / 270
python main.py --no-cache          # 禁用 OCR 缓存，每次请求云端
python main.py --refresh-cache     # 强制重拉 OCR 并覆盖缓存
python main.py --keep-empty-cols   # 保留整列为空的列
python main.py --debug             # 写出网格叠加图到 data/debug/
```

回归统计：

```bash
python tools/regress.py
python tools/regress.py --refresh-cache --baseline data/debug/_regress_baseline.txt
```

## 目录结构

```
cutnmerge/
├── main.py              # CLI 入口
├── requirements.txt
├── .env.example
├── tools/
│   └── regress.py       # 回归统计
├── data/
│   ├── input/           # 待处理图片
│   ├── output/          # HTML / Markdown 结果
│   ├── cache/           # OCR 本地缓存（gitignore）
│   └── debug/           # --debug 叠加图（gitignore）
└── src/
    ├── pipeline.py      # 主流程
    ├── orient.py        # 方向归正（90/180）
    ├── lines.py         # 框线网格
    ├── tsr.py           # TableStructureRec 结构
    ├── tsr_refine.py    # TSR 拓扑后处理
    ├── models.py        # LORE + 云端 OCR
    ├── matching.py      # IoA 填格
    ├── formatter.py     # Markdown 组装
    ├── html_formatter.py# HTML（rowspan/colspan）
    ├── ocr_post.py      # OCR 规范化
    ├── config.py        # .env 配置
    └── ocr_cache.py     # OCR 缓存
```

## 代码调用

```python
from src.pipeline import extract_table_output

out = extract_table_output(
    "data/input/demo.png",
    structure="tsr",
    ioa_threshold=0.5,
)
print(out["html"])
# out["md"] 为展开式 Markdown（可选）
```

入口会通过 `src.config.load_env()` 加载项目根目录 `.env`。

## 依赖说明

| 用途 | 包 |
|------|-----|
| ModelScope LORE | `modelscope`、`torch` 等 |
| TableStructureRec | `wired_table_rec`、`lineless_table_rec`、`table_cls` |
| 云端 OCR | `paddleocr`（官方客户端，无需本地 PaddlePaddle） |
| 图像 / 几何 | `opencv-python`、`numpy`、`shapely` |
| 配置 | `python-dotenv` |

`tsr` 默认 `need_ocr=False`，不依赖 `rapidocr`；若个别环境仍报缺 RapidOCR，可额外 `pip install rapidocr`。

## 常见问题

**未配置 Token**  
确保 `.env` 中 `PADDLEOCR_ACCESS_TOKEN` 已填写且不是占位符 `your-access-token-here`。

**Windows 控制台乱码**  
结果文件为 UTF-8；控制台可能无法打印部分字符，以 `data/output/*.html` 为准。

**中文路径图片**  
已通过 `np.fromfile` + `imdecode` 处理，可直接使用含中文的文件名路径。

**想反复调结构参数**  
默认启用 OCR 缓存；调 `--structure` / IoA 等时不必加 `--refresh-cache`，可避免重复扣额度。
