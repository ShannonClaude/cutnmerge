"""坐标几何工具：多边形转换、IoA 计算。

【Phase 1 重构说明】
本文件原先还承担着"同行单元格 Y 坐标聚类归一化"（cluster_rows_by_y）的职责，
即通过物理坐标去猜测表格的行结构。这种纯几何聚类方式在单元格存在轻微抖动、
倾斜、合并单元格等场景下非常脆弱（容差稍有不合适就会把两行错误合并，或把
同一行拆成两行）。

读光 LORE 模型本身已经在 models.py 中输出了每个单元格的逻辑拓扑坐标
（row_start/row_end/col_start/col_end），行列结构应直接来自模型的逻辑
预测结果，而不是靠 Y 轴聚类重新猜测一遍。因此本文件已**彻底删除**
cluster_rows_by_y 及其专用辅助函数 _top_left_xy，只保留与"行结构猜测"
无关的纯几何原语：多边形转换与 IoA（Intersection over Area）计算，
供 matching.py 做文本框到逻辑单元格的物理归属判定。
"""

from __future__ import annotations

from shapely.geometry import Polygon
from shapely.validation import make_valid
import numpy as np


def polygon_to_shapely(pts: np.ndarray) -> Polygon:
    """将 (N,2) 坐标数组转为 Shapely Polygon，并尽量修复无效几何。"""
    poly = Polygon(np.asarray(pts, dtype=np.float64))
    if not poly.is_valid:
        # buffer(0) / make_valid 可修复自相交等常见 OCR/检测毛刺
        fixed = poly.buffer(0)
        if fixed.is_empty:
            fixed = make_valid(poly)
        poly = fixed
    # MultiPolygon 时取面积最大的一块，避免后续交集运算异常
    if poly.geom_type == "MultiPolygon":
        poly = max(list(poly.geoms), key=lambda g: g.area)
    return poly


def compute_ioa(text_poly: Polygon, cell_poly: Polygon) -> float:
    """
    计算 IoA（Intersection over Area），而非标准 IoU。

    公式：
        IoA = 文本框与单元格的交集面积 / 文本框自身面积

    设计动机：
    - IoU 在「小文本框落入大单元格」时会被分母中的单元格面积稀释，容易误判为不匹配；
    - IoA 只衡量「文本有多少比例落在单元格内」，更适合 OCR 文本到单元格的归属判定。

    边界：
    - 文本面积为 0（退化线段/点）时返回 0.0；
    - 无效几何已在 polygon_to_shapely 中尽量修复。
    """
    text_area = float(text_poly.area)
    if text_area <= 1e-9:
        return 0.0

    try:
        inter = text_poly.intersection(cell_poly)
        inter_area = float(inter.area)
    except Exception:
        # 极端拓扑错误时兜底
        try:
            inter = text_poly.buffer(0).intersection(cell_poly.buffer(0))
            inter_area = float(inter.area)
        except Exception:
            return 0.0

    return inter_area / text_area
