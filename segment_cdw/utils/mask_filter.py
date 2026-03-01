# -*- coding: utf-8 -*-
"""
SAM2 mask post-processing module
--------------------------------

用途：
- 接收 mask_fn(img_np) 的输出 pred_items（通常是 List[Dict]）
- 进行实例级过滤（面积、形状、bbox 等）
- 进行 mask 级 NMS 去重（优先用 pycocotools 计算 IoU；无则 fallback）
- 输出统一的 MaskItem 列表（后续裁剪、分类、统计都用这个结构）

输入 pred_items 的典型结构（兼容多种字段名）：
- 每个元素为 dict
- mask 字段支持：
  - item["segmentation"] : (H, W) bool/uint8/float
  - item["mask"]         : (H, W) bool/uint8/float
  - item["masks"]        : (N, H, W) 或 (H, W)
- 分数字段可选：
  - item["predicted_iou"]
  - item["stability_score"]

输出 filtered_items 的结构：
- List[MaskItem]
  - mask: (H, W) bool
  - bbox_xyxy: (x1, y1, x2, y2)  半开区间
  - area: int
  - score: float（优先 predicted_iou，其次 stability_score，否则 1.0）
  - predicted_iou / stability_score: Optional[float]
  - src_index: int（原始 pred_items 索引）
  - meta: Dict[str, Any]（保留其它字段，已移除大 mask）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from pycocotools import mask as mask_utils
    _HAS_COCO = True
except Exception:
    _HAS_COCO = False

__all__ = ["MaskItem", "filter_sam2_masks"]


@dataclass
class MaskItem:
    """
    过滤后的实例 mask 数据结构（后续 pipeline 建议统一用它）
    """
    mask: np.ndarray  # (H, W) bool
    bbox_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2) half-open
    area: int
    score: float
    predicted_iou: Optional[float]
    stability_score: Optional[float]
    src_index: int
    meta: Dict[str, Any]


def _extract_mask(item: Dict[str, Any], image_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    从 item 中提取 mask，返回 shape=(H,W) bool
    支持字段：segmentation / mask / masks
    """
    H, W = image_hw
    cand = None
    for k in ("segmentation", "mask", "masks"):
        if k in item:
            cand = item[k]
            break
    if cand is None:
        return None

    m = np.asarray(cand)

    # (N,H,W) -> 取第一张
    if m.ndim == 3 and m.shape[1:] == (H, W):
        m = m[0]

    if m.ndim != 2 or m.shape != (H, W):
        return None

    if m.dtype != np.bool_:
        m = m > 0.5

    return m


def _compute_bbox_xyxy(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return (0, 0, 0, 0)
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1  # half-open
    y2 = int(ys.max()) + 1
    return (x1, y1, x2, y2)


def _bbox_wh(b: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1), max(0, y2 - y1)


def _compactness(mask: np.ndarray) -> float:
    """
    紧致度：4πA/P^2（边界越“细长/锯齿”越小）
    perimeter 用 4-neighbor 边界近似，足够做过滤。
    """
    m = mask.astype(np.uint8)
    A = float(m.sum())
    if A <= 0:
        return 0.0

    up = np.pad(m[:-1, :], ((1, 0), (0, 0)), mode="constant")
    dn = np.pad(m[1:, :], ((0, 1), (0, 0)), mode="constant")
    lf = np.pad(m[:, :-1], ((0, 0), (1, 0)), mode="constant")
    rt = np.pad(m[:, 1:], ((0, 0), (0, 1)), mode="constant")

    edge = (m != up) + (m != dn) + (m != lf) + (m != rt)
    P = float(edge.sum())
    if P <= 0:
        return 0.0

    return float(4.0 * np.pi * A / (P * P))


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _get_score_fields(item: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], float]:
    """
    你的定义：score = predicted_iou * stability_score

    - 若其中一个缺失：按 0 处理（更保守，避免“没分数的 mask”混进来）
    - 若你更希望缺失时当 1.0，可以把缺失默认值从 0 改为 1
    """
    piou = _to_float(item.get("predicted_iou", None))
    stab = _to_float(item.get("stability_score", None))

    # 保守策略：缺失即 0（更严格）
    piou_v = 0.0 if piou is None else float(piou)
    stab_v = 0.0 if stab is None else float(stab)

    score = piou_v * stab_v
    return piou, stab, float(score)


def _encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    # pycocotools 需要 Fortran order 的 uint8
    m = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(m)
    if isinstance(rle.get("counts", None), bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _mask_iou_np(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum(dtype=np.int64)
    return float(inter) / float(union) if union > 0 else 0.0


def _greedy_nms(items: List[MaskItem], iou_thr: float) -> List[MaskItem]:
    """
    mask 级贪心 NMS：
    - 按 score 降序
    - 与已保留 mask 的 IoU > iou_thr 则丢弃
    """
    if not items:
        return []

    items_sorted = sorted(items, key=lambda x: x.score, reverse=True)
    keep: List[MaskItem] = []

    if _HAS_COCO:
        rles = [_encode_rle(it.mask) for it in items_sorted]
        kept_rles: List[Dict[str, Any]] = []

        for it, rle in zip(items_sorted, rles):
            if not kept_rles:
                keep.append(it)
                kept_rles.append(rle)
                continue
            ious = mask_utils.iou(kept_rles, [rle], [0] )  # (K,1)
            max_iou = float(np.max(ious)) if ious.size else 0.0
            if max_iou <= iou_thr:
                keep.append(it)
                kept_rles.append(rle)
        return keep

    # fallback（没有 pycocotools 时）
    for it in items_sorted:
        ok = True
        for kt in keep:
            if _mask_iou_np(it.mask, kt.mask) > iou_thr:
                ok = False
                break
        if ok:
            keep.append(it)
    return keep


def filter_sam2_masks(
    pred_items: Sequence[Dict[str, Any]],
    image_hw: Tuple[int, int],
    *,
    min_area: int = 200,
    max_area_frac: float = 0.20,
    min_score: float = 0.0,
    min_pred_iou: Optional[float] = None,
    min_stability: Optional[float] = None,
    min_bbox_side: int = 20,
    max_aspect_ratio: float = 8.0,
    min_compactness: float = 0.005,
    nms_iou_thr: float = 0.85,
    keep_meta: bool = True,
) -> List[MaskItem]:
    """
    过滤 + NMS 去重，输出 MaskItem 列表。

    关键参数解释：
    - min_area: 太小的碎片过滤
    - max_area_frac: 面积占整图比例超过该阈值的“超大罩子”过滤
    - min_score: score 下限（score 来源：predicted_iou > stability_score > 1.0）
    - min_pred_iou / min_stability: 若提供则作为硬阈值
    - min_bbox_side: bbox 最短边太小过滤（后续分类也更稳）
    - max_aspect_ratio: bbox 过细长过滤（裂缝/边缘伪实例）
    - min_compactness: 紧致度下限
    - nms_iou_thr: mask 级去重阈值
    """
    H, W = image_hw
    img_area = float(H * W)

    items: List[MaskItem] = []

    for idx, raw in enumerate(pred_items):
        if not isinstance(raw, dict):
            continue

        mask = _extract_mask(raw, image_hw)
        if mask is None:
            continue

        area = int(mask.sum())
        if area <= 0:
            continue
        if area < int(min_area):
            continue
        if area > int(max_area_frac * img_area):
            continue

        bbox = _compute_bbox_xyxy(mask)
        bw, bh = _bbox_wh(bbox)
        if bw < min_bbox_side or bh < min_bbox_side:
            continue
        ar = (max(bw, bh) / max(1, min(bw, bh)))
        if ar > max_aspect_ratio:
            continue

        comp = _compactness(mask)
        if comp < min_compactness:
            continue

        piou, stab, score = _get_score_fields(raw)
        if score < float(min_score):
            continue
        if min_pred_iou is not None and (piou is None or piou < float(min_pred_iou)):
            continue
        if min_stability is not None and (stab is None or stab < float(min_stability)):
            continue

        meta = dict(raw) if keep_meta else {}
        # 避免 meta 里保存大数组
        for k in ("segmentation", "mask", "masks"):
            meta.pop(k, None)

        items.append(
            MaskItem(
                mask=mask,
                bbox_xyxy=bbox,
                area=area,
                score=float(score),
                predicted_iou=piou,
                stability_score=stab,
                src_index=int(idx),
                meta=meta,
            )
        )

    # 去重：同一颗粒多 mask 很常见
    items = _greedy_nms(items, iou_thr=float(nms_iou_thr))
    return items
