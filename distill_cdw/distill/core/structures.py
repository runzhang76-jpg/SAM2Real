"""教师输出与伪标签的共享数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch

    TorchTensor = torch.Tensor
except Exception:  # pragma: no cover - torch 可选，仅用于类型提示
    torch = None
    TorchTensor = Any  # type: ignore

MaskType = Union[np.ndarray, TorchTensor]
BBoxType = Tuple[float, float, float, float]


@dataclass
class InstancePrediction:
    """教师端的单实例预测（bbox 使用 COCO 风格 xywh）。"""

    image_id: int
    bbox: BBoxType
    class_id: int
    score: float
    reliability: float
    mask: Optional[MaskType] = None
    rle: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 友好的字典（默认不序列化 mask）。"""
        return {
            "image_id": self.image_id,
            "bbox": list(self.bbox),
            "class_id": self.class_id,
            "score": float(self.score),
            "reliability": float(self.reliability),
            "rle": self.rle,
            "meta": self.meta,
        }


@dataclass
class PseudoLabelInstance:
    """学生监督使用的伪标签实例（bbox 使用 COCO 风格 xywh）。"""

    image_id: int
    bbox: BBoxType
    class_id: int
    score: float
    reliability: float
    mask: Optional[MaskType] = None
    rle: Optional[Dict[str, Any]] = None
    instance_id: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 友好的字典（默认不序列化 mask）。"""
        return {
            "id": self.instance_id,
            "image_id": self.image_id,
            "bbox": list(self.bbox),
            "category_id": self.class_id,
            "score": float(self.score),
            "reliability": float(self.reliability),
            "segmentation": self.rle,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PseudoLabelInstance":
        """从 JSON 友好的字典反序列化。"""
        bbox = data.get("bbox", [0.0, 0.0, 0.0, 0.0])
        return PseudoLabelInstance(
            image_id=int(data.get("image_id", 0)),
            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            class_id=int(data.get("category_id", data.get("class_id", 0))),
            score=float(data.get("score", 0.0)),
            reliability=float(data.get("reliability", 0.0)),
            mask=None,
            rle=data.get("segmentation"),
            instance_id=data.get("id"),
            meta=data.get("meta", {}),
        )


def instances_mean_reliability(instances: Sequence[PseudoLabelInstance]) -> float:
    """计算一组实例的平均可靠性。"""
    if not instances:
        return 0.0
    return float(sum(inst.reliability for inst in instances) / max(1, len(instances)))


def ensure_tensor_mask(mask: MaskType) -> MaskType:
    """在 torch 可用时将 numpy mask 转为 torch 张量。"""
    if torch is None:
        return mask
    if isinstance(mask, np.ndarray):
        return torch.from_numpy(mask)
    return mask
