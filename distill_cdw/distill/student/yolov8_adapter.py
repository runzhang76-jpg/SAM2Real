"""YOLOv8 实例分割学生模型适配器。"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from distill.core.structures import InstancePrediction
from distill.student.base import StudentModel
from distill.utils.logging import get_logger

import torch
import torch.nn.functional as F


def _to_numpy_batch(images: Any) -> List[np.ndarray]:
    """将输入图像批量转换为 HWC 的 uint8 numpy 列表。"""

    if torch is not None and isinstance(images, torch.Tensor):
        batch = []
        for i in range(images.shape[0]):
            img = images[i].detach().cpu()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)
            arr = img.numpy()
            if arr.dtype.kind == "f":
                max_val = float(arr.max()) if arr.size else 1.0
                if max_val <= 1.5:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255)
            arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            batch.append(arr)
        return batch

    if isinstance(images, list):
        batch = []
        for img in images:
            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            batch.append(arr)
        return batch

    raise TypeError("无法识别的图像输入类型")


def _bbox_xywh_to_center_norm(bbox: Tuple[float, float, float, float], width: int, height: int) -> List[float]:
    x, y, w, h = bbox
    cx = x + w * 0.5
    cy = y + h * 0.5
    return [cx / width, cy / height, w / width, h / height]


def _bbox_xyxy_to_xywh(bbox_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def _mask_from_bbox(bbox: Tuple[float, float, float, float], height: int, width: int) -> "torch.Tensor":
    x, y, w, h = bbox
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(width, int(x + w))
    y2 = min(height, int(y + h))
    mask = torch.zeros(height, width, dtype=torch.float32)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0
    return mask


class YOLOv8SegStudent(StudentModel):
    """
    YOLOv8 实例分割学生模型适配器。

    约定输入 targets 为 List[List[PseudoLabelInstance]]。
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.logger = get_logger("distill")
        super().__init__()

        self.cfg = cfg
        self.device = str(cfg.get("device", "auto"))
        self.weights = str(cfg.get("weights", "yolov8s-seg.pt")) or "yolov8s-seg.pt"
        self.params = cfg.get("params", {})
        self.conf = float(self.params.get("conf", 0.25))
        self.iou = float(self.params.get("iou", 0.7))
        self.imgsz = int(self.params.get("imgsz", 640))

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._setup_yolo(self.weights, train_mode=True)

    def _ensure_ultralytics_args(self) -> None:
        """补齐 Ultralytics 训练所需的 args，并确保支持属性访问。"""

        raw_args = getattr(self.net, "args", None)
        args_dict: Dict[str, Any] = {}
        if raw_args is None:
            args_dict = {}
        elif isinstance(raw_args, Mapping):
            args_dict = dict(raw_args)
        else:
            try:
                args_dict = dict(vars(raw_args))
            except Exception:
                args_dict = {}

        defaults = {
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "overlap_mask": True,
        }
        overrides: Dict[str, Any] = {}
        overrides.update(self.cfg.get("ultralytics_args", {}))
        overrides.update(self.cfg.get("yolo_args", {}))

        final_args = {**defaults, **args_dict, **overrides}
        self.net.args = SimpleNamespace(**final_args)

    def _setup_yolo(self, weights: str, train_mode: bool) -> None:
        """初始化 Ultralytics YOLO 并绑定底层网络。"""

        from ultralytics import YOLO

        yolo = YOLO(weights)
        object.__setattr__(self, "_yolo", yolo)
        if "_yolo" in self._modules:
            self._modules.pop("_yolo")
        self.net = self._yolo.model
        self._ensure_ultralytics_args()
        self.net.to(self.device)
        self.net.requires_grad_(True)
        self.net.train(train_mode)

    def _build_ultralytics_batch(self, images: "torch.Tensor", targets: Optional[Any]) -> Dict[str, Any]:
        """将伪标签转换为 Ultralytics 训练 batch 格式。"""

        if torch is None:
            raise RuntimeError("PyTorch 不可用，无法构建 batch。")

        batch_size, _, height, width = images.shape
        batch_idx: List[int] = []
        cls_list: List[float] = []
        bbox_list: List[List[float]] = []
        masks_list: List["torch.Tensor"] = []

        if targets is None:
            targets = [[] for _ in range(batch_size)]

        for img_i, insts in enumerate(targets):
            for inst in insts:
                bbox_list.append(_bbox_xywh_to_center_norm(inst.bbox, width, height))
                cls_list.append(float(inst.class_id))
                batch_idx.append(int(img_i))
                if inst.mask is not None:
                    mask = inst.mask
                    if not isinstance(mask, torch.Tensor):
                        mask = torch.as_tensor(np.asarray(mask), dtype=torch.float32)
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask[0]
                    if mask.shape != (height, width):
                        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(height, width), mode="nearest").squeeze(0).squeeze(0)
                    masks_list.append(mask.float())
                else:
                    masks_list.append(_mask_from_bbox(inst.bbox, height, width))

        if not bbox_list:
            empty = torch.zeros((0, 4), device=images.device)
            return {
                "img": images,
                "batch_idx": torch.zeros((0, 1), device=images.device, dtype=torch.int64),
                "cls": torch.zeros((0, 1), device=images.device, dtype=torch.float32),
                "bboxes": empty,
                "masks": torch.zeros((0, height, width), device=images.device),
            }

        batch = {
            "img": images,
            "batch_idx": torch.tensor(batch_idx, device=images.device, dtype=torch.int64).view(-1, 1),
            "cls": torch.tensor(cls_list, device=images.device, dtype=torch.float32).view(-1, 1),
            "bboxes": torch.tensor(bbox_list, device=images.device, dtype=torch.float32),
            "masks": torch.stack(masks_list).to(images.device),
        }
        return batch

    def _normalize_loss_items(self, loss_items: Any) -> Dict[str, float]:
        if loss_items is None:
            return {}
        if isinstance(loss_items, dict):
            return {f"loss_{k}": float(v) for k, v in loss_items.items()}
        if isinstance(loss_items, torch.Tensor):
            values = loss_items.detach().cpu().flatten().tolist()
            keys = ["box", "cls", "dfl", "mask"]
            mapped = {}
            for idx, val in enumerate(values):
                key = keys[idx] if idx < len(keys) else f"item_{idx}"
                mapped[f"loss_{key}"] = float(val)
            return mapped
        return {}

    def forward(self, images: "torch.Tensor", targets: Optional[Any] = None) -> Dict[str, Any]:
        images = images.to(self.device)
        outputs: Dict[str, Any] = {}

        preds = self.net(images)
        outputs["preds"] = preds

        if targets is not None and hasattr(self.net, "loss"):
            batch = self._build_ultralytics_batch(images, targets)
            loss, loss_items = self.net.loss(batch, preds=preds)
            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.sum()
            outputs["loss_supervised"] = loss
            outputs["loss_items"] = self._normalize_loss_items(loss_items)

        return outputs

    def predict(self, images: Any, **kwargs: Any) -> List[List[InstancePrediction]]:
        images_np = _to_numpy_batch(images)
        results = self._yolo.predict(
            images_np,
            imgsz=int(kwargs.get("imgsz", self.imgsz)),
            conf=float(kwargs.get("conf", self.conf)),
            iou=float(kwargs.get("iou", self.iou)),
            device=self.device,
            verbose=False,
        )

        outputs: List[List[InstancePrediction]] = []
        for res in results:
            preds: List[InstancePrediction] = []
            boxes = getattr(res, "boxes", None)
            masks = getattr(res, "masks", None)
            if boxes is None:
                outputs.append(preds)
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes, "xyxy") else np.zeros((0, 4))
            cls = boxes.cls.detach().cpu().numpy() if hasattr(boxes, "cls") else np.zeros((len(xyxy),))
            conf = boxes.conf.detach().cpu().numpy() if hasattr(boxes, "conf") else np.zeros((len(xyxy),))
            mask_data = None
            if masks is not None and hasattr(masks, "data"):
                mask_data = masks.data.detach().cpu().numpy()
            for i in range(len(xyxy)):
                bbox_xywh = _bbox_xyxy_to_xywh(tuple(xyxy[i].tolist()))
                mask = None
                if mask_data is not None and i < mask_data.shape[0]:
                    mask = mask_data[i] > 0.5
                preds.append(
                    InstancePrediction(
                        image_id=-1,
                        bbox=bbox_xywh,
                        class_id=int(cls[i]) if i < len(cls) else 0,
                        score=float(conf[i]) if i < len(conf) else 0.0,
                        reliability=float(conf[i]) if i < len(conf) else 0.0,
                        mask=mask,
                        rle=None,
                        meta={"source": "yolov8"},
                    )
                )
            outputs.append(preds)
        return outputs

    def load_weights(self, path: str) -> None:
        self._setup_yolo(path, train_mode=self.training)
        self.logger.info("YOLO 权重已加载: %s", path)

    def save_weights(self, path: str) -> None:
        if torch is None:
            raise RuntimeError("PyTorch 不可用，无法保存权重。")
        torch.save(self.net.state_dict(), path)
        self.logger.info("YOLO 权重已保存: %s", path)
