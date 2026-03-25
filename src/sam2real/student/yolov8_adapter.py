"""YOLOv8 """

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sam2real.core.structures import InstancePrediction
from sam2real.student.base import StudentModel
from sam2real.utils.logging import get_logger
from sam2real.utils.paths import resolve_project_path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_numpy_batch(images: Any) -> List[np.ndarray]:
    """ HWC  uint8 numpy """

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

    raise TypeError("")


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


def _parse_class_id_map(raw_map: Any) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    if not isinstance(raw_map, dict):
        return mapping
    for key, value in raw_map.items():
        try:
            mapping[int(key)] = int(value)
        except Exception:
            continue
    return mapping


def _default_class_names(num_classes: int) -> Dict[int, str]:
    return {idx: f"class_{idx}" for idx in range(max(0, int(num_classes)))}


class YOLOv8SegStudent(StudentModel):
    """
    YOLOv8

     targets  List[List[PseudoLabelInstance]]
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.logger = get_logger("distill")
        super().__init__()

        self.cfg = cfg
        self.device = str(cfg.get("device", "auto"))
        self.weights = resolve_project_path(cfg.get("weights", "yolov8s-seg.pt")) or "yolov8s-seg.pt"
        self.params = cfg.get("params", {})
        self.conf = float(self.params.get("conf", 0.25))
        self.iou = float(self.params.get("iou", 0.7))
        self.imgsz = int(self.params.get("imgsz", 640))
        self.max_det = int(self.params.get("max_det", 300))
        self.num_classes = int(cfg.get("num_classes", self.params.get("num_classes", 1)))
        self.train_to_orig_class_id = _parse_class_id_map(cfg.get("train_to_orig_class_id"))
        self.kd_heads_cfg = self.params.get("kd_heads", {})
        self.kd_heads_enabled = bool(self.kd_heads_cfg.get("enabled", True))
        self._kd_hook_feature: Optional[torch.Tensor] = None
        self._kd_feature_warned = False
        self._kd_hook_handle = None
        object.__setattr__(self, "_eval_yolo", None)

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._setup_yolo(self.weights, train_mode=True)
        self._setup_kd_heads()

    def _ensure_ultralytics_args(self) -> None:
        """ Ultralytics  args"""

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
            # Our training batch provides instance-stacked masks [N, H, W] plus batch_idx,
            # which matches Ultralytics overlap_mask=False semantics.
            "overlap_mask": False,
        }
        overrides: Dict[str, Any] = {}
        overrides.update(self.cfg.get("ultralytics_args", {}))
        overrides.update(self.cfg.get("yolo_args", {}))

        final_args = {**defaults, **args_dict, **overrides}
        self.net.args = SimpleNamespace(**final_args)

    def _setup_yolo(self, weights: str, train_mode: bool) -> None:
        """ Ultralytics YOLO """

        from ultralytics import YOLO
        from ultralytics.nn.tasks import SegmentationModel

        yolo = YOLO(weights)
        loaded_model = yolo.model
        loaded_nc = getattr(getattr(loaded_model, "model", [None])[-1], "nc", None)
        if loaded_nc is not None and int(loaded_nc) != self.num_classes:
            cfg = deepcopy(getattr(loaded_model, "yaml", {}))
            if not cfg:
                raise RuntimeError(f"Unable to rebuild YOLO segmentation model for nc={self.num_classes}: missing yaml.")
            cfg["nc"] = int(self.num_classes)
            rebuilt_model = SegmentationModel(cfg=cfg, ch=int(cfg.get("ch", 3)), nc=self.num_classes, verbose=False)
            rebuilt_model.load(loaded_model, verbose=True)
            rebuilt_model.names = _default_class_names(self.num_classes)
            yolo.model = rebuilt_model
            yolo.predictor = None
            self.logger.info(
                "rebuilt YOLO segmentation head for %d classes from pretrained nc=%s",
                self.num_classes,
                loaded_nc,
            )
        object.__setattr__(self, "_yolo", yolo)
        if "_yolo" in self._modules:
            self._modules.pop("_yolo")
        self.net = self._yolo.model
        if not hasattr(self.net, "names") or len(getattr(self.net, "names", {}) or {}) != self.num_classes:
            self.net.names = _default_class_names(self.num_classes)
        self._ensure_ultralytics_args()
        self.net.to(self.device)
        self.net.requires_grad_(True)
        self.net.train(train_mode)
        self._clear_eval_session()
        if self.kd_heads_enabled and hasattr(self, "kd_mask_head"):
            self._register_kd_feature_hook()

    def _build_eval_yolo(self) -> Any:
        from ultralytics import YOLO

        eval_yolo = YOLO(self.weights)
        eval_net = deepcopy(self.net).to(self.device)
        eval_net.requires_grad_(False)
        eval_net.eval()
        eval_yolo.model = eval_net
        eval_yolo.predictor = None
        return eval_yolo

    def _clear_eval_session(self) -> None:
        object.__setattr__(self, "_eval_yolo", None)
        if "_eval_yolo" in self._modules:
            self._modules.pop("_eval_yolo")

    def begin_predict_session(self) -> None:
        self._clear_eval_session()
        object.__setattr__(self, "_eval_yolo", self._build_eval_yolo())
        if "_eval_yolo" in self._modules:
            self._modules.pop("_eval_yolo")

    def end_predict_session(self) -> None:
        self._clear_eval_session()

    def _get_predict_yolo(self) -> Tuple[Any, bool]:
        if self._eval_yolo is not None:
            return self._eval_yolo, False
        return self._build_eval_yolo(), True

    def _setup_kd_heads(self) -> None:
        if not self.kd_heads_enabled:
            self.kd_mask_head = None
            self.kd_cls_pool = None
            self.kd_cls_head = None
            return

        if not hasattr(nn, "LazyConv2d") or not hasattr(nn, "LazyLinear"):
            raise RuntimeError("KD heads require torch.nn.LazyConv2d and torch.nn.LazyLinear.")
        self.kd_mask_head = nn.LazyConv2d(1, kernel_size=1)
        self.kd_cls_pool = nn.AdaptiveAvgPool2d(1)
        self.kd_cls_head = nn.LazyLinear(self.num_classes)
        self._register_kd_feature_hook()

    def _register_kd_feature_hook(self) -> None:
        if not self.kd_heads_enabled:
            return
        if self._kd_hook_handle is not None:
            self._kd_hook_handle.remove()
            self._kd_hook_handle = None

        hook_module = self.net
        model_seq = getattr(self.net, "model", None)
        try:
            if model_seq is not None and len(model_seq) > 0:
                hook_module = model_seq[-1]
        except Exception:
            hook_module = self.net

        def _capture_feature(_module: nn.Module, inputs: Tuple[Any, ...]) -> None:
            candidate = inputs[0] if inputs else None
            self._kd_hook_feature = self._select_kd_feature(candidate)

        self._kd_hook_handle = hook_module.register_forward_pre_hook(_capture_feature)

    def _select_kd_feature(self, value: Any) -> Optional["torch.Tensor"]:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, Mapping):
            tensors = [self._select_kd_feature(item) for item in value.values()]
            tensors = [item for item in tensors if item is not None]
            return tensors[-1] if tensors else None
        if isinstance(value, (list, tuple)):
            tensors = [self._select_kd_feature(item) for item in value]
            tensors = [item for item in tensors if item is not None]
            return tensors[-1] if tensors else None
        return None

    def _forward_kd_heads(self, feature: Optional["torch.Tensor"], image_size: Tuple[int, int]) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        if not self.kd_heads_enabled or feature is None:
            return outputs
        if self.kd_mask_head is None or self.kd_cls_pool is None or self.kd_cls_head is None:
            return outputs

        outputs["features"] = feature
        kd_mask_logits = self.kd_mask_head(feature)
        if kd_mask_logits.shape[-2:] != image_size:
            kd_mask_logits = F.interpolate(kd_mask_logits, size=image_size, mode="bilinear", align_corners=False)
        outputs["kd_mask_logits"] = kd_mask_logits

        pooled = self.kd_cls_pool(feature).flatten(1)
        outputs["kd_cls_logits"] = self.kd_cls_head(pooled)
        return outputs

    def _build_ultralytics_batch(self, images: "torch.Tensor", targets: Optional[Any]) -> Dict[str, Any]:
        """ Ultralytics  batch """

        if torch is None:
            raise RuntimeError("PyTorch  batch")

        if bool(getattr(self.net.args, "overlap_mask", False)):
            raise RuntimeError(
                "YOLOv8 overlap_mask=True expects per-image index masks, but this adapter provides "
                "instance-stacked masks with batch_idx. Set overlap_mask=False."
            )

        batch_size, _, height, width = images.shape
        batch_idx: List[int] = []
        cls_list: List[float] = []
        bbox_list: List[List[float]] = []
        masks_list: List["torch.Tensor"] = []

        if targets is None:
            targets = [[] for _ in range(batch_size)]

        for img_i, insts in enumerate(targets):
            for inst in insts:
                cls_id = int(inst.class_id)
                if cls_id < 0 or cls_id >= self.num_classes:
                    raise ValueError(
                        f"Target class_id={cls_id} is outside [0, {self.num_classes - 1}] for YOLOv8. "
                        "COCO-style category ids must be remapped to contiguous 0-based training ids."
                    )
                bbox_list.append(_bbox_xywh_to_center_norm(inst.bbox, width, height))
                cls_list.append(float(cls_id))
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

        self._kd_hook_feature = None
        preds = self.net(images)
        outputs["preds"] = preds
        if self.kd_heads_enabled:
            kd_outputs = self._forward_kd_heads(self._kd_hook_feature, images.shape[-2:])
            outputs.update(kd_outputs)
            if not kd_outputs and not self._kd_feature_warned:
                self.logger.warning("KD heads enabled but no YOLO feature map was captured; soft distillation terms will stay zero.")
                self._kd_feature_warned = True

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
        was_training = self.training
        net_was_training = self.net.training
        outputs: List[List[InstancePrediction]] = []
        predict_yolo, transient_predictor = self._get_predict_yolo()
        try:
            results = predict_yolo.predict(
                images_np,
                imgsz=int(kwargs.get("imgsz", self.imgsz)),
                conf=float(kwargs.get("conf", self.conf)),
                iou=float(kwargs.get("iou", self.iou)),
                max_det=int(kwargs.get("max_det", self.max_det)),
                device=self.device,
                verbose=False,
            )
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
                mask_polygons: List[Any] = []
                orig_shape = tuple(int(v) for v in getattr(res, "orig_shape", (0, 0)))
                if masks is not None and hasattr(masks, "data"):
                    mask_data = masks.data.detach().cpu().numpy()
                if masks is not None and hasattr(masks, "xy"):
                    mask_polygons = list(masks.xy)
                if len(xyxy) > 0 and mask_data is None:
                    self.logger.warning(
                        "YOLO predict returned %d boxes but no masks; COCO segm export will drop these predictions.",
                        len(xyxy),
                    )
                invalid_raw_classes: List[int] = []
                for i in range(len(xyxy)):
                    raw_cls = int(cls[i]) if i < len(cls) else 0
                    if raw_cls < 0 or raw_cls >= self.num_classes:
                        invalid_raw_classes.append(raw_cls)
                        continue
                    bbox_xywh = _bbox_xyxy_to_xywh(tuple(xyxy[i].tolist()))
                    mask = None
                    if mask_data is not None and i < mask_data.shape[0]:
                        candidate_mask = mask_data[i] > 0.5
                        if candidate_mask.shape == orig_shape:
                            mask = candidate_mask
                    polygon = None
                    if i < len(mask_polygons):
                        polygon = np.asarray(mask_polygons[i], dtype=np.float32)
                    preds.append(
                        InstancePrediction(
                            image_id=-1,
                            bbox=bbox_xywh,
                            class_id=self.train_to_orig_class_id.get(raw_cls, raw_cls),
                            score=float(conf[i]) if i < len(conf) else 0.0,
                            reliability=float(conf[i]) if i < len(conf) else 0.0,
                            mask=mask,
                            rle=None,
                            meta={
                                "source": "yolov8",
                                "raw_train_class_id": raw_cls,
                                "predict_conf": float(kwargs.get("conf", self.conf)),
                                "predict_iou": float(kwargs.get("iou", self.iou)),
                                "predict_max_det": int(kwargs.get("max_det", self.max_det)),
                                "has_mask": bool(mask is not None),
                                "mask_shape": tuple(mask_data[i].shape) if mask_data is not None and i < mask_data.shape[0] else None,
                                "orig_shape": orig_shape,
                                "mask_polygon": polygon,
                            },
                        )
                    )
                if invalid_raw_classes:
                    self.logger.warning(
                        "YOLO predict produced raw class ids outside [0, %d): invalid=%s names_len=%d head_nc=%s",
                        self.num_classes,
                        sorted(set(invalid_raw_classes))[:16],
                        len(getattr(self.net, "names", {}) or {}),
                        getattr(getattr(self.net, "model", [None])[-1], "nc", None),
                    )
                outputs.append(preds)
        finally:
            self.net.requires_grad_(True)
            self.net.train(net_was_training)
            self.train(was_training)
            if transient_predictor:
                predict_yolo = None
        return outputs

    def load_weights(self, path: str) -> None:
        self._setup_yolo(resolve_project_path(path), train_mode=self.training)
        self.logger.info("YOLO : %s", path)

    def save_weights(self, path: str) -> None:
        if torch is None:
            raise RuntimeError("PyTorch ")
        torch.save(self.net.state_dict(), path)
        self.logger.info("YOLO : %s", path)
