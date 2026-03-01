"""伪标签可靠性评分。"""

from __future__ import annotations

from typing import Optional

import numpy as np

from distill.core.structures import InstancePrediction


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(max_val, float(value)))


def _aspect_penalty(bbox: tuple[float, float, float, float]) -> float:
    _, _, w, h = bbox
    if w <= 0 or h <= 0:
        return 0.0
    aspect = max(w / h, h / w)
    return _clamp(1.0 / aspect)


def _area_ratio(mask: Optional[object], bbox: tuple[float, float, float, float]) -> float:
    if mask is None:
        return 1.0
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    area = float(arr.sum())
    _, _, w, h = bbox
    bbox_area = max(1.0, float(w * h))
    return _clamp(area / bbox_area)


def compute_reliability(instance: InstancePrediction) -> float:
    """为预测实例计算可靠性得分。"""

    base_score = float(getattr(instance, "score", 0.0))
    predicted_iou = instance.meta.get("predicted_iou") if hasattr(instance, "meta") else None
    stability = instance.meta.get("stability_score") if hasattr(instance, "meta") else None

    if predicted_iou is not None and stability is not None:
        base_score = float(predicted_iou) * float(stability)

    area_ratio = _area_ratio(instance.mask, instance.bbox)
    aspect_penalty = _aspect_penalty(instance.bbox)

    reliability = base_score * area_ratio * max(0.2, aspect_penalty)
    return _clamp(reliability)
