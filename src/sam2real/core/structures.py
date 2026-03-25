""""""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch

    TorchTensor = torch.Tensor
except Exception:  # pragma: no cover - torch
    torch = None
    TorchTensor = Any  # type: ignore

MaskType = Union[np.ndarray, TorchTensor]
BBoxType = Tuple[float, float, float, float]


@dataclass
class InstancePrediction:
    """bbox  COCO  xywh"""

    image_id: int
    bbox: BBoxType
    class_id: int
    score: float
    reliability: float
    mask: Optional[MaskType] = None
    rle: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """ JSON  mask"""
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
    """bbox  COCO  xywh"""

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
        """ JSON  mask"""
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
        """ JSON """
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
    """"""
    if not instances:
        return 0.0
    return float(sum(inst.reliability for inst in instances) / max(1, len(instances)))


def ensure_tensor_mask(mask: MaskType) -> MaskType:
    """ torch  numpy mask  torch """
    if torch is None:
        return mask
    if isinstance(mask, np.ndarray):
        return torch.from_numpy(mask)
    return mask
