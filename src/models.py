"""模型加载与推断：读光 LORE 表格结构识别 + 云端 PaddleOCR 文本提取。

【Phase 1 重构说明】
LORE（Logical Location Regression Network）模型的推理结果并不只是单元格的
物理四点坐标（polygons），它还会同时输出每个单元格的**逻辑拓扑位置**
（ModelScope 内部称为 logi，对外以 OutputKeys.BOXES 形式给出）：

    boxes[i] = [row_start, row_end, col_start, col_end]

即该单元格在表格逻辑网格中横跨的起止行、起止列（inclusive）。这是 LORE
论文本身要解决的核心问题——不用几何聚类去"猜"表格的行列结构，而是由模型
端到端直接预测逻辑网格拓扑。因此本文件不再只提取 polygons，而是把 polygons
与 logi 按下标一一对应，一并挂在每个 cell 对象上，供下游 matching /
formatter 直接使用逻辑索引，而不是再去猜测行列。

【四级缺陷重构 —— 稀疏表格列向左错位】
阿里读光（LORE）遇到空白单元格时往往不输出框，导致后方有内容的单元格其
预测的 col_start 被错误缩小。因此在 predict_cells 末尾增加物理列坐标校正：
对全表单元格左右边界做距离聚类（边界吸附），用物理列索引覆盖 LORE 的
col_start/col_end。

【行索引物理 Y 校正】
对全表单元格上下边界做距离聚类（边界吸附），用物理行索引覆盖 LORE 的
row_start/row_end。边界吸附可避免跨列/跨行格中心点制造幽灵行列。
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .config import get_settings

# 阿里读光 LORE 表格结构识别模型（ModelScope 官方 ID）
LORE_MODEL_ID = "damo/cv_resnet-transformer_table-structure-recognition_lore"

_lore_pipeline = None
_ocr_client = None


def _enable_lore_cpu_fallback() -> None:
    """
    ModelScope LORE 推理代码多处硬编码 .cuda()，在仅 CPU 环境下会直接报错。
    无 CUDA 时将 Tensor.cuda() 变为空操作（保持当前 device）。
    """
    import torch

    if torch.cuda.is_available():
        return
    if getattr(torch.Tensor.cuda, "_lore_cpu_patched", False):
        return

    _orig_cuda = torch.Tensor.cuda

    def _cuda_noop(self, *args, **kwargs):  # noqa: ANN001
        return self

    _cuda_noop._lore_cpu_patched = True  # type: ignore[attr-defined]
    torch.Tensor.cuda = _cuda_noop  # type: ignore[method-assign]


def load_lore_model():
    """
    加载 ModelScope 读光表格结构识别 Pipeline（懒加载单例）。

    【任务候选顺序说明】
    只有 `lineless_table_recognition` 这个任务在 ModelScope 的
    TASK_OUTPUTS 映射中同时声明了 POLYGONS + BOXES（即物理坐标 + 逻辑拓扑，
    参见 modelscope/outputs/outputs.py 中
    `Tasks.lineless_table_recognition: [OutputKeys.POLYGONS, OutputKeys.BOXES]`）。
    而 `table_recognition` 等旧任务只声明了 POLYGONS，不含逻辑拓扑。
    本次重构的核心诉求就是拿到逻辑拓扑，因此该任务必须排在候选列表最前面；
    其余任务名仅作为不同 ModelScope 版本下的兼容兜底保留，一旦命中但缺少
    拓扑信息，会在 predict_cells 中显式报错，而不是静默退化。
    """
    global _lore_pipeline
    if _lore_pipeline is not None:
        return _lore_pipeline

    _enable_lore_cpu_fallback()

    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    task_candidates = []
    for name in (
        "lineless_table_recognition",  # 唯一保证 POLYGONS + BOXES(logi) 的任务
        "table_recognition",
        "table_structure_recognition",
    ):
        if hasattr(Tasks, name):
            task_candidates.append(getattr(Tasks, name))
    task_candidates.extend(
        [
            "lineless-table-recognition",
            "table-recognition",
            "table-structure-recognition",
        ]
    )

    last_err: Exception | None = None
    for task in task_candidates:
        try:
            _lore_pipeline = pipeline(task, model=LORE_MODEL_ID)
            return _lore_pipeline
        except Exception as exc:  # noqa: BLE001 — 逐个尝试兼容不同 modelscope 版本
            last_err = exc
            continue

    raise RuntimeError(
        f"无法加载读光表格结构模型 {LORE_MODEL_ID}"
    ) from last_err


def load_ocr():
    """
    加载 PaddleOCR 官方云端客户端（不跑本地推理）。

    Token / 模型名等均从项目根目录 .env 读取。
    """
    global _ocr_client
    if _ocr_client is not None:
        return _ocr_client

    from paddleocr import PaddleOCRClient

    settings = get_settings()
    kwargs: Dict[str, Any] = {
        "token": settings.require_token(),
        "request_timeout": settings.request_timeout,
        "poll_timeout": settings.poll_timeout,
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url

    _ocr_client = PaddleOCRClient(**kwargs)
    return _ocr_client


def _to_quad(pts: Any) -> Optional[np.ndarray]:
    """将任意四点坐标转为 shape (4, 2) 的 float64 数组。"""
    arr = np.asarray(pts, dtype=np.float64).reshape(-1)
    if arr.size == 4:
        # [x1, y1, x2, y2] 轴对齐框 → 四点
        x1, y1, x2, y2 = arr.tolist()
        return np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float64,
        )
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] < 4:
        return None
    return arr[:4].copy()


def _top_left_from_quad(quad: np.ndarray) -> tuple[float, float]:
    sums = quad.sum(axis=1)
    tl_idx = int(np.argmin(sums))
    return float(quad[tl_idx, 0]), float(quad[tl_idx, 1])


def _extract_polygons_from_result(result: Any) -> List[np.ndarray]:
    """
    兼容解析 LORE / ModelScope 多种返回结构中的单元格四点多边形。

    常见形态：
    - {"polygons": [[[x,y], ...], ...]}
    - {"output": {"polygons": ...}}
    - {"polygons": np.ndarray}
    - 直接为 list / ndarray
    """
    candidates: List[Any] = []

    if result is None:
        return []

    if isinstance(result, dict):
        for key in ("polygons", "cells", "boxes", "cell_polys", "polys"):
            if key in result and result[key] is not None:
                candidates.append(result[key])
        for nest_key in ("output", "outputs", "result"):
            nested = result.get(nest_key)
            if isinstance(nested, dict):
                for key in ("polygons", "cells", "boxes", "cell_polys", "polys"):
                    if key in nested and nested[key] is not None:
                        candidates.append(nested[key])
            elif isinstance(nested, (list, tuple, np.ndarray)):
                candidates.append(nested)
    elif isinstance(result, (list, tuple, np.ndarray)):
        candidates.append(result)

    polygons: List[np.ndarray] = []
    for cand in candidates:
        items = cand
        if isinstance(items, np.ndarray):
            if items.ndim == 3 and items.shape[1] >= 4 and items.shape[2] == 2:
                for i in range(items.shape[0]):
                    quad = _to_quad(items[i])
                    if quad is not None:
                        polygons.append(quad)
                if polygons:
                    return polygons
            if items.ndim == 2 and items.shape[1] == 8:
                for row in items:
                    quad = _to_quad(row.reshape(4, 2))
                    if quad is not None:
                        polygons.append(quad)
                if polygons:
                    return polygons
            items = list(items)

        if not isinstance(items, (list, tuple)):
            continue

        for item in items:
            if isinstance(item, (list, tuple, np.ndarray)):
                arr = np.asarray(item)
                if arr.size >= 8 or arr.size == 4:
                    quad = _to_quad(arr)
                    if quad is not None:
                        polygons.append(quad)
                    continue
            if isinstance(item, dict):
                for k in ("polygon", "points", "box", "bbox", "poly"):
                    if k in item:
                        quad = _to_quad(item[k])
                        if quad is not None:
                            polygons.append(quad)
                        break

        if polygons:
            return polygons

    return polygons


def _extract_logi_from_result(result: Any) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    从 LORE / ModelScope 返回结构中解析每个单元格的**逻辑拓扑位置**（logi）。

    ModelScope 的 lineless_table_recognition pipeline 在
    modelscope/models/cv/table_recognition/model_lore.py 中，
    postprocess 阶段产出：

        result = {
            OutputKeys.POLYGONS: slct_dets,                 # 物理四点坐标
            OutputKeys.BOXES: slct_logi.cpu().numpy(),       # 逻辑拓扑 (N,4)
        }

    其中 BOXES（即 logi）每行是 4 个数：
        [row_start, row_end, col_start, col_end]

    表示该单元格横跨的起止逻辑行 / 起止逻辑列（闭区间，可能为 0-based 或
    1-based，具体由 predict_cells 统一归一化处理）。

    本函数做多路 key 兼容解析（"boxes" / "logi" / 嵌套 output.boxes 等），
    与 _extract_polygons_from_result 的兼容思路保持一致；若确实找不到任何
    拓扑信息，返回 None（调用方需据此判断当前任务是否支持逻辑拓扑输出）。
    """
    if result is None:
        return None

    candidates: List[Any] = []
    if isinstance(result, dict):
        for key in ("boxes", "logi", "logic", "logical_boxes"):
            if key in result and result[key] is not None:
                candidates.append(result[key])
        for nest_key in ("output", "outputs", "result"):
            nested = result.get(nest_key)
            if isinstance(nested, dict):
                for key in ("boxes", "logi", "logic", "logical_boxes"):
                    if key in nested and nested[key] is not None:
                        candidates.append(nested[key])

    for cand in candidates:
        arr = np.asarray(cand, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 4:
            continue
        logi: List[Tuple[int, int, int, int]] = []
        for row in arr:
            row_start, row_end, col_start, col_end = row.tolist()
            logi.append(
                (
                    int(round(row_start)),
                    int(round(row_end)),
                    int(round(col_start)),
                    int(round(col_end)),
                )
            )
        if logi:
            return logi

    return None


def _cluster_1d_centers(
    values: List[float],
    thresh: float = 15.0,
) -> List[float]:
    """
    对一维坐标做贪心距离聚类，返回按升序排列的簇中心列表。

    算法：先排序，再从左到右扫描——若当前值与当前簇均值的差 > thresh，
    则开启新簇；否则并入当前簇并更新均值。最终每个簇的均值即为物理列中心。
    """
    if not values:
        return []

    sorted_vals = sorted(float(v) for v in values)
    clusters: List[List[float]] = [[sorted_vals[0]]]
    means: List[float] = [sorted_vals[0]]

    for v in sorted_vals[1:]:
        if abs(v - means[-1]) > thresh:
            clusters.append([v])
            means.append(v)
        else:
            clusters[-1].append(v)
            means[-1] = float(np.mean(clusters[-1]))

    # 按中心升序返回（贪心从左到右已基本有序，再保险排序一次）
    return sorted(means)


def _snap_to_boundary(value: float, boundaries: np.ndarray) -> int:
    """将坐标吸附到最近边界，返回边界下标。"""
    return int(np.argmin(np.abs(boundaries - value)))


def _correct_columns_by_physical_x(
    cells: List[Dict[str, Any]],
    cluster_thresh: float = 12.0,
) -> List[Dict[str, Any]]:
    """
    用物理框左右边界聚类，校正 LORE 预测的 col_start / col_end。

    【边界吸附】替代旧版中心点聚类：跨列合并单元格的中心 X 会落在两列中间，
    被误判成幽灵列。改为汇总全表 x_min ∪ x_max 聚类成列边界序列，
    col_start = idx(左边界)，col_end = idx(右边界) - 1。

    row_* 保持不变。原地修改并返回 cells。
    """
    if not cells:
        return cells

    edges: List[float] = []
    polys: List[np.ndarray] = []
    for cell in cells:
        poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
        polys.append(poly)
        edges.append(float(poly[:, 0].min()))
        edges.append(float(poly[:, 0].max()))

    boundaries = _cluster_1d_centers(edges, thresh=cluster_thresh)
    if len(boundaries) < 2:
        return cells

    bounds_arr = np.asarray(boundaries, dtype=np.float64)

    for cell, poly in zip(cells, polys):
        x_min = float(poly[:, 0].min())
        x_max = float(poly[:, 0].max())
        left = _snap_to_boundary(x_min, bounds_arr)
        right = _snap_to_boundary(x_max, bounds_arr)
        if right <= left:
            # 退化：保留原 span，以左边界为起点
            col_span = max(int(cell.get("col_span") or 1), 1)
            right = min(left + col_span, len(boundaries) - 1)
            if right <= left:
                right = left + 1 if left + 1 < len(boundaries) else left
        cell["col_start"] = left
        cell["col_end"] = max(right - 1, left)
        cell["col_span"] = int(cell["col_end"]) - int(cell["col_start"]) + 1

    min_col = min(int(c["col_start"]) for c in cells)
    if min_col != 0:
        for cell in cells:
            cell["col_start"] = int(cell["col_start"]) - min_col
            cell["col_end"] = int(cell["col_end"]) - min_col
            cell["col_span"] = int(cell["col_end"]) - int(cell["col_start"]) + 1

    return cells


def _median_cell_height(
    cells: List[Dict[str, Any]],
    *,
    single_row_only: bool = True,
) -> float:
    """
    计算单元格物理高度的中位数；无有效高度时返回 0。

    默认只统计 row_span==1 的单元格，避免跨行大格拉高中位数、
    导致行聚类阈值过大、相邻数据行被误合并。
    """
    heights: List[float] = []
    for cell in cells:
        if single_row_only and max(int(cell.get("row_span") or 1), 1) > 1:
            continue
        poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
        h = float(poly[:, 1].max() - poly[:, 1].min())
        if h > 1e-6:
            heights.append(h)
    # 若过滤后为空，回退到全表
    if not heights and single_row_only:
        return _median_cell_height(cells, single_row_only=False)
    if not heights:
        return 0.0
    return float(np.median(heights))


def _resolve_row_cluster_thresh(
    cells: List[Dict[str, Any]],
    cluster_thresh: Optional[float] = None,
    height_ratio: float = 0.45,
    min_thresh: float = 6.0,
    max_thresh: float = 20.0,
) -> float:
    """
    解析行聚类阈值。

    - 若调用方显式传入 cluster_thresh，直接使用；
    - 否则按「全表单元格物理高度中位数 × height_ratio」动态计算，
      并夹紧到 [min_thresh, max_thresh]，避免分辨率差异导致过合并/过拆分。
    """
    if cluster_thresh is not None:
        return float(cluster_thresh)

    median_h = _median_cell_height(cells)
    if median_h <= 0:
        return 10.0
    return float(np.clip(median_h * height_ratio, min_thresh, max_thresh))


def _correct_rows_by_physical_y(
    cells: List[Dict[str, Any]],
    cluster_thresh: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    用物理框上下边界聚类，校正 LORE 预测的 row_start / row_end。

    【边界吸附】替代旧版中心点聚类：格内换行文字 / 跨行大格的中心 Y
    会制造幽灵行。改为汇总全表 y_min ∪ y_max 聚类成行边界序列，
    row_start = idx(顶边界)，row_end = idx(底边界) - 1。

    阈值默认取单行格高度中位数的一小部分（偏紧），避免过合并。
    col_* 保持不变。原地修改并返回 cells。
    """
    if not cells:
        return cells

    # 边界聚类阈值比中心聚类更紧：用较小比例
    if cluster_thresh is None:
        median_h = _median_cell_height(cells)
        thresh = float(np.clip(median_h * 0.25, 4.0, 14.0)) if median_h > 0 else 8.0
    else:
        thresh = float(cluster_thresh)

    edges: List[float] = []
    polys: List[np.ndarray] = []
    for cell in cells:
        poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
        polys.append(poly)
        edges.append(float(poly[:, 1].min()))
        edges.append(float(poly[:, 1].max()))

    boundaries = _cluster_1d_centers(edges, thresh=thresh)
    if len(boundaries) < 2:
        return cells

    bounds_arr = np.asarray(boundaries, dtype=np.float64)

    for cell, poly in zip(cells, polys):
        y_top = float(poly[:, 1].min())
        y_bot = float(poly[:, 1].max())
        top = _snap_to_boundary(y_top, bounds_arr)
        bot = _snap_to_boundary(y_bot, bounds_arr)
        if bot <= top:
            row_span = max(int(cell.get("row_span") or 1), 1)
            bot = min(top + row_span, len(boundaries) - 1)
            if bot <= top:
                bot = top + 1 if top + 1 < len(boundaries) else top
        cell["row_start"] = top
        cell["row_end"] = max(bot - 1, top)
        cell["row_span"] = int(cell["row_end"]) - int(cell["row_start"]) + 1

    min_row = min(int(c["row_start"]) for c in cells)
    if min_row != 0:
        for cell in cells:
            cell["row_start"] = int(cell["row_start"]) - min_row
            cell["row_end"] = int(cell["row_end"]) - min_row
            cell["row_span"] = int(cell["row_end"]) - int(cell["row_start"]) + 1

    return cells


def predict_cells(
    image: Union[str, np.ndarray],
    lore_pipe=None,
) -> List[Dict[str, Any]]:
    """
    使用读光 LORE 模型预测所有单元格的物理坐标 + 逻辑拓扑。

    Returns:
        Cell 列表，每项含:
        - polygon: (4,2) ndarray，物理四点坐标（供 IoA 匹配使用）
        - x_key / y_key: 左上角坐标（仅用于调试展示与格内文本排序，
          不再承担"猜测行结构"的职责——该职责已由下方逻辑拓扑字段取代）
        - row_start / row_end / col_start / col_end: LORE 原生预测的逻辑网格
          拓扑坐标（已归一化为从 0 开始，闭区间）；其中 col_* 会再经
          物理列中心聚类校正，防止稀疏表列向左塌陷；row_* 会再经
          物理行中心聚类校正，防止稀疏/不均匀表行错位
        - row_span / col_span: 由拓扑坐标推导出的跨行 / 跨列数
        - texts / text: 预留，由 matching 填充

    Raises:
        RuntimeError: 若模型结果不包含逻辑拓扑（BOXES/logi），说明当前加载
            的任务/模型版本不支持拓扑输出，需确认 load_lore_model 使用的是
            `lineless_table_recognition` 任务。
    """
    pipe = lore_pipe or load_lore_model()
    result = pipe(image)

    quads = _extract_polygons_from_result(result)
    if not quads:
        return []

    logi = _extract_logi_from_result(result)
    if logi is None:
        raise RuntimeError(
            "读光 LORE 模型返回结果中未找到逻辑拓扑信息（row/col 起止索引）。"
            "本 Pipeline 依赖 LORE 原生逻辑拓扑而非几何聚类来重建表格行列结构，"
            "请确认 load_lore_model 加载的任务是 `lineless_table_recognition`"
            "（该任务的 TASK_OUTPUTS 同时包含 POLYGONS 与 BOXES/logi），"
            "而不是仅输出 POLYGONS 的旧版 `table_recognition` 任务。"
        )
    if len(logi) != len(quads):
        raise RuntimeError(
            f"读光 LORE 模型返回的物理坐标数量（{len(quads)}）与逻辑拓扑数量"
            f"（{len(logi)}）不一致，无法按下标一一对应，请检查模型/版本兼容性。"
        )

    # ---- 逻辑索引归一化：兼容模型可能输出 0-based 或 1-based 的起止索引 ----
    # 统一减去全表最小值，保证最终的行列坐标一定从 0 开始，
    # 这样后续 formatter 里按 row_start/col_start 直接建矩阵下标才不会越界。
    min_row = min(r for r, _, _, _ in logi)
    min_col = min(c for _, _, c, _ in logi)

    cells: List[Dict[str, Any]] = []
    for quad, (row_start, row_end, col_start, col_end) in zip(quads, logi):
        x_key, y_key = _top_left_from_quad(quad)

        norm_row_start = row_start - min_row
        norm_row_end = row_end - min_row
        norm_col_start = col_start - min_col
        norm_col_end = col_end - min_col

        # 防御：若模型偶发预测出 end < start，交换保证区间合法
        if norm_row_end < norm_row_start:
            norm_row_start, norm_row_end = norm_row_end, norm_row_start
        if norm_col_end < norm_col_start:
            norm_col_start, norm_col_end = norm_col_end, norm_col_start

        cells.append(
            {
                "polygon": quad,
                "x_key": x_key,
                "y_key": y_key,
                # ---- LORE 原生逻辑拓扑（本次重构的核心数据）----
                "row_start": norm_row_start,
                "row_end": norm_row_end,
                "col_start": norm_col_start,
                "col_end": norm_col_end,
                "row_span": norm_row_end - norm_row_start + 1,
                "col_span": norm_col_end - norm_col_start + 1,
                "texts": [],
                "text": "",
            }
        )

    # ---- 物理列坐标校正：用 center_x 聚类覆盖被稀疏空白格压塌的 col_start ----
    cells = _correct_columns_by_physical_x(cells, cluster_thresh=15.0)
    # ---- 物理行坐标校正：用 center_y 聚类覆盖被稀疏/不均匀行错位的 row_start ----
    return _correct_rows_by_physical_y(cells)


def _ensure_image_file(image: Union[str, np.ndarray]) -> tuple[str, Optional[str]]:
    """
    云端 API 只接受 file_path / file_url。

    Returns:
        (path, temp_path_to_cleanup_or_None)
    """
    if isinstance(image, str):
        if not os.path.isfile(image):
            raise FileNotFoundError(f"无法读取图像: {image}")
        return image, None

    if isinstance(image, np.ndarray):
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        if not cv2.imwrite(tmp_path, image):
            os.unlink(tmp_path)
            raise RuntimeError("无法将 ndarray 图像写入临时文件")
        return tmp_path, tmp_path

    raise TypeError("image 须为文件路径或 numpy.ndarray")


def _append_text_box(
    out: List[Dict[str, Any]],
    text: Any,
    box: Any,
    score: float = 1.0,
) -> None:
    text_s = str(text).strip() if text is not None else ""
    if not text_s:
        return
    quad = _to_quad(box)
    if quad is None:
        return
    top_left = _top_left_from_quad(quad)
    out.append(
        {
            "polygon": quad,
            "text": text_s,
            "score": float(score),
            "top_left": top_left,
        }
    )


def _parse_pruned_result_to_text_boxes(pruned: Any) -> List[Dict[str, Any]]:
    """从云端 OCR 的 prunedResult 中提取带坐标的文本块。"""
    text_boxes: List[Dict[str, Any]] = []
    if not isinstance(pruned, dict):
        return text_boxes

    texts = pruned.get("rec_texts")
    polys = (
        pruned.get("rec_polys")
        or pruned.get("dt_polys")
        or pruned.get("rec_boxes")
        or pruned.get("dt_boxes")
    )
    scores = pruned.get("rec_scores") or []
    if isinstance(texts, (list, tuple)) and isinstance(polys, (list, tuple, np.ndarray)):
        poly_list = list(polys)
        for i, text in enumerate(texts):
            box = poly_list[i] if i < len(poly_list) else None
            score = float(scores[i]) if i < len(scores) else 1.0
            if box is not None:
                _append_text_box(text_boxes, text, box, score)
    return text_boxes


def _parse_cloud_result_to_text_boxes(result: Any) -> List[Dict[str, Any]]:
    """解析 PaddleOCRClient.ocr 返回对象。"""
    text_boxes: List[Dict[str, Any]] = []
    pages = getattr(result, "pages", None) or []
    for page in pages:
        pruned = getattr(page, "pruned_result", None)
        if pruned is None and isinstance(getattr(page, "raw", None), dict):
            pruned = page.raw.get("prunedResult") or page.raw.get("pruned_result")
        text_boxes.extend(_parse_pruned_result_to_text_boxes(pruned))
    return text_boxes


def predict_texts(
    image: Union[str, np.ndarray],
    ocr_engine=None,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_extra: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    使用云端 PaddleOCR 提取整图文本块（含检测坐标）。

    调用 PP-OCRv5/v6，返回细粒度文本框供 IoA 填格。
    use_cache：命中 data/cache/ 则跳过云端；refresh_cache 强制重拉并覆盖。

    Returns:
        文本块列表，每项含 polygon / text / score / top_left(x,y)
    """
    from paddleocr import OCROptions

    from .ocr_cache import load_ocr_cache, save_ocr_cache

    if use_cache and not refresh_cache:
        cached = load_ocr_cache(image, extra=cache_extra)
        if cached is not None:
            return cached

    client = ocr_engine or load_ocr()
    settings = get_settings()
    file_path, tmp_path = _ensure_image_file(image)

    try:
        options = OCROptions(
            use_doc_orientation_classify=settings.use_doc_orientation_classify,
            use_doc_unwarping=settings.use_doc_unwarping,
            use_textline_orientation=settings.use_textline_orientation,
            text_det_thresh=settings.text_det_thresh,
            text_det_box_thresh=settings.text_det_box_thresh,
            text_det_unclip_ratio=settings.text_det_unclip_ratio,
            text_rec_score_thresh=settings.text_rec_score_thresh,
        )
        result = client.ocr(
            file_path=file_path,
            model=settings.ocr_model,
            options=options,
        )
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    text_boxes = _parse_cloud_result_to_text_boxes(result)
    if use_cache or refresh_cache:
        try:
            save_ocr_cache(image, text_boxes, extra=cache_extra)
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning("写入 OCR 缓存失败: %s", exc)
    return text_boxes
