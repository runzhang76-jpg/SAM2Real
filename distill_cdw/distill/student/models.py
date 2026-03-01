"""学生模型定义与构建工厂。"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from distill.core.registry import Registry
from distill.core.structures import InstancePrediction
from distill.student.base import StudentModel
from distill.utils.logging import get_logger

import torch
import torch.nn as nn

STUDENT_REGISTRY = Registry("student")


def _mask_to_bbox(mask: np.ndarray) -> List[float]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max()) + 1.0
    y2 = float(ys.max()) + 1.0
    return [x1, y1, x2 - x1, y2 - y1]


class ToyStudent(StudentModel):
    """用于跑通流程的最小学生模型。"""

    def __init__(self, in_channels: int = 3, num_classes: int = 2) -> None:
        if nn is None:
            raise RuntimeError("PyTorch 不可用，无法构建 ToyStudent。")
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Conv2d(32, 1, kernel_size=1)
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, images: "torch.Tensor", targets: Any = None) -> Dict[str, Any]:
        feats = self.encoder(images)
        return {
            "mask_logits": self.mask_head(feats),
            "class_logits": self.class_head(feats),
            "features": feats,
        }

    def predict(self, images: Any, **kwargs: Any) -> List[List[InstancePrediction]]:
        if torch is None or F is None:
            raise RuntimeError("PyTorch 不可用，无法进行预测。")
        self.eval()
        if isinstance(images, torch.Tensor):
            tensor = images
        else:
            arr = np.asarray(images)
            if arr.ndim == 3:
                arr = arr[None, ...]
            tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0

        with torch.no_grad():
            outputs = self.forward(tensor)
            mask_logits = outputs.get("mask_logits")
            class_logits = outputs.get("class_logits")
            if mask_logits is None or class_logits is None:
                return [[] for _ in range(tensor.shape[0])]
            mask_prob = torch.sigmoid(mask_logits)
            class_prob = torch.softmax(class_logits, dim=1)

        results: List[List[InstancePrediction]] = []
        for i in range(tensor.shape[0]):
            mask = (mask_prob[i, 0].cpu().numpy() > 0.5)
            if mask.sum() == 0:
                results.append([])
                continue
            bbox = _mask_to_bbox(mask)
            cls_id = int(torch.argmax(class_prob[i]).item())
            score = float(class_prob[i].max().item())
            results.append(
                [
                    InstancePrediction(
                        image_id=-1,
                        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        class_id=cls_id,
                        score=score,
                        reliability=score,
                        mask=mask,
                        rle=None,
                        meta={"source": "toy"},
                    )
                ]
            )
        return results

    def load_weights(self, path: str) -> None:
        if torch is None:
            raise RuntimeError("PyTorch 不可用，无法加载权重。")
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state, strict=False)

    def save_weights(self, path: str) -> None:
        if torch is None:
            raise RuntimeError("PyTorch 不可用，无法保存权重。")
        torch.save(self.state_dict(), path)


@STUDENT_REGISTRY.register("toy")
def _build_toy(cfg: Dict[str, Any]) -> StudentModel:
    return ToyStudent(
        in_channels=int(cfg.get("in_channels", 3)),
        num_classes=int(cfg.get("num_classes", 2)),
    )


@STUDENT_REGISTRY.register("tiny_mask")
def _build_tiny(cfg: Dict[str, Any]) -> StudentModel:
    return _build_toy(cfg)


@STUDENT_REGISTRY.register("yolov8s_seg")
def _build_yolov8(cfg: Dict[str, Any]) -> StudentModel:
    from distill.student.yolov8_adapter import YOLOv8SegStudent
    return YOLOv8SegStudent(cfg)


def build_student(cfg: Dict[str, Any]) -> StudentModel:
    """学生模型构建工厂函数。"""

    logger = get_logger("distill")
    name = str(cfg.get("name", "toy"))
    backend = str(cfg.get("backend", "")).lower()
    if name.startswith("yolo") and backend and backend != "ultralytics":
        logger.warning("student.backend=%s 与 %s 不匹配，建议使用 ultralytics", backend, name)
    if name in STUDENT_REGISTRY.list_keys():
        return STUDENT_REGISTRY.build(name, cfg)
    logger.warning("未知学生模型: %s，回退到 ToyStudent", name)
    return _build_toy(cfg)
