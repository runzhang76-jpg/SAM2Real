"""Instance classifier adapter with pluggable backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from matmatch2real.core.structures import InstancePrediction
from matmatch2real.teacher.dinov3_classifier import DinoV3Classifier
from matmatch2real.teacher.model_classifier import ModelClassifier
from matmatch2real.utils.logging import get_logger
from matmatch2real.utils.paths import resolve_project_path


CategoryMap = Dict[int, Union[int, str]]


def _try_int(value: Any) -> Optional[int]:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _load_category_map(path: Optional[Any]) -> Optional[CategoryMap]:
    if not path:
        return None
    if isinstance(path, dict):
        mapping: CategoryMap = {}
        for key, value in path.items():
            key_id = _try_int(key)
            if key_id is None:
                continue
            value_id = _try_int(value)
            if value_id is not None:
                mapping[key_id] = value_id
            elif isinstance(value, str):
                mapping[key_id] = value
        return mapping or None
    if isinstance(path, list):
        mapping = {}
        for item in path:
            if isinstance(item, dict) and "src" in item and "dst" in item:
                src_id = _try_int(item["src"])
                dst_id = _try_int(item["dst"])
                if src_id is not None and dst_id is not None:
                    mapping[src_id] = dst_id
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                src_id = _try_int(item[0])
                dst_id = _try_int(item[1])
                if src_id is not None and dst_id is not None:
                    mapping[src_id] = dst_id
        return mapping or None
    if not isinstance(path, (str, Path)):
        return None
    file_path = Path(resolve_project_path(path))
    data = json.loads(file_path.read_text(encoding="utf-8"))
    mapping = {}
    for key, value in data.items():
        key_id = _try_int(key)
        if key_id is None:
            continue
        value_id = _try_int(value)
        if value_id is not None:
            mapping[key_id] = value_id
        elif isinstance(value, str):
            mapping[key_id] = value
    return mapping or None


def _square_crop_coords(
    img_hw: Tuple[int, int], bbox_xyxy: Tuple[float, float, float, float], margin_ratio: float = 0.1
) -> Tuple[int, int, int, int]:
    height, width = img_hw
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw, bh) * (1.0 + 2.0 * margin_ratio)
    half = side * 0.5
    x1n = int(max(0, np.floor(cx - half)))
    y1n = int(max(0, np.floor(cy - half)))
    x2n = int(min(width, np.ceil(cx + half)))
    y2n = int(min(height, np.ceil(cy + half)))
    return x1n, y1n, x2n, y2n


def _masked_square_crop(
    img: np.ndarray,
    mask: Optional[np.ndarray],
    bbox_xyxy: Tuple[float, float, float, float],
    margin_ratio: float = 0.1,
    background_mode: str = "zero",
    min_size: int = 4,
) -> np.ndarray:
    # Crop by bbox then optionally suppress background outside the mask.
    height, width = img.shape[:2]
    x1n, y1n, x2n, y2n = _square_crop_coords((height, width), bbox_xyxy, margin_ratio)
    if x2n <= x1n or y2n <= y1n:
        return np.zeros((0, 0, img.shape[2]), dtype=img.dtype)
    img_crop = img[y1n:y2n, x1n:x2n].copy()
    if img_crop.size == 0 or img_crop.shape[0] < min_size or img_crop.shape[1] < min_size:
        return np.zeros((0, 0, img.shape[2]), dtype=img.dtype)
    if mask is None or background_mode == "keep":
        return img_crop
    mask_crop = mask[y1n:y2n, x1n:x2n]
    if mask_crop.size == 0:
        return np.zeros((0, 0, img.shape[2]), dtype=img.dtype)
    if mask_crop.dtype != bool:
        mask_crop = mask_crop.astype(bool)
    if not np.any(mask_crop):
        return np.zeros((0, 0, img.shape[2]), dtype=img.dtype)
    if background_mode == "white":
        img_crop[~mask_crop] = 255
    else:
        img_crop[~mask_crop] = 0
    return img_crop


def _collect_instance_patches(
    instances: List[InstancePrediction],
    image_np: np.ndarray,
    crop_cfg: Dict[str, Any],
) -> Tuple[List[int], List[np.ndarray]]:
    valid_indices: List[int] = []
    patches: List[np.ndarray] = []
    margin_ratio = float(crop_cfg.get("margin_ratio", 0.1))
    background_mode = str(crop_cfg.get("background_mode", "zero")).strip().lower()
    min_size = int(crop_cfg.get("min_size", 4))
    for idx, inst in enumerate(instances):
        x, y, w, h = inst.bbox
        bbox_xyxy = (float(x), float(y), float(x + w), float(y + h))
        mask_np: Optional[np.ndarray] = None
        if inst.mask is not None:
            if isinstance(inst.mask, torch.Tensor):
                mask_np = inst.mask.detach().cpu().numpy()
            else:
                mask_np = np.asarray(inst.mask)
            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                mask_np = mask_np[0]
        patch = _masked_square_crop(
            image_np,
            mask_np,
            bbox_xyxy,
            margin_ratio=margin_ratio,
            background_mode=background_mode,
            min_size=min_size,
        )
        if patch.size == 0:
            continue
        valid_indices.append(idx)
        patches.append(patch)
    return valid_indices, patches


class ClassifierAdapter:
    """
    Unified classifier adapter.

    Supported classifier.type:
    - "model" (existing branch)
    - "dinov3"
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.logger = get_logger("distill")
        self.score_threshold = float(cfg.get("score_threshold", 0.0))
        self.crop_cfg = self._resolve_crop_cfg(cfg)
        self.combine_score = str(cfg.get("combine_score", "mul")).lower()
        self.category_map = _load_category_map(cfg.get("category_map"))
        self.type = self._resolve_type(cfg)
        self.classifier = self._build_classifier()

    @staticmethod
    def _resolve_crop_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        crop_cfg = dict(cfg.get("crop", {}))
        if "margin_ratio" not in crop_cfg:
            crop_cfg["margin_ratio"] = float(cfg.get("margin_ratio", 0.1))
        crop_cfg["background_mode"] = str(crop_cfg.get("background_mode", "zero")).strip().lower()
        crop_cfg["min_size"] = int(crop_cfg.get("min_size", 4))
        return crop_cfg

    @staticmethod
    def _resolve_type(cfg: Dict[str, Any]) -> str:
        raw_type = str(cfg.get("type", "")).strip().lower()
        if raw_type:
            return raw_type
        legacy_backend = str(cfg.get("backend", "none")).strip().lower()
        if legacy_backend in {"segment_cdw", "model"}:
            return "model"
        if legacy_backend in {"none", ""}:
            return "none"
        return legacy_backend

    def _build_classifier(self) -> Optional[Any]:
        if self.type == "none":
            self.logger.info("Classifier disabled by type=none")
            return None
        if self.type == "model":
            model_cfg = dict(self.cfg)
            model_cfg.update(self.cfg.get("model", {}))
            return ModelClassifier(model_cfg, logger=self.logger)
        if self.type == "dinov3":
            dino_cfg = dict(self.cfg)
            dino_cfg.update(self.cfg.get("dinov3", {}))
            mode = str(dino_cfg.get("mode", "prototype")).strip().lower()
            if mode == "knn":
                dino_cfg.update(dino_cfg.get("knn", {}))
            else:
                dino_cfg["mode"] = "prototype"
                dino_cfg.update(dino_cfg.get("prototype", {}))
            return DinoV3Classifier(dino_cfg, logger=self.logger)
        self.logger.warning("Unknown classifier.type=%s; classifier disabled", self.type)
        return None

    def classify(self, instances: List[InstancePrediction], image_np: Optional[np.ndarray] = None) -> List[InstancePrediction]:
        if not instances:
            return instances
        if self.classifier is None:
            return instances
        if image_np is None:
            self.logger.warning("image_np is required for classifier crop; skip classify")
            return instances

        valid_indices, patches = _collect_instance_patches(instances, image_np, crop_cfg=self.crop_cfg)
        if not patches:
            return instances

        try:
            cls_outputs = self.classifier.predict_patches(patches)
        except Exception as exc:
            self.logger.warning("classification failed: %s", exc)
            return instances

        if len(cls_outputs) != len(valid_indices):
            self.logger.warning(
                "classifier output size mismatch: outputs=%d, instances=%d",
                len(cls_outputs),
                len(valid_indices),
            )
            return instances

        updated: List[Optional[InstancePrediction]] = list(instances)
        for inst_idx, pred in zip(valid_indices, cls_outputs):
            inst = updated[inst_idx]
            if inst is None:
                continue
            raw_cls = pred.get("category_id")
            category_score = float(pred.get("category_score", 0.0))
            class_id = _try_int(raw_cls)

            mapped: Union[int, str, None] = None
            if class_id is not None and self.category_map is not None:
                mapped = self.category_map.get(class_id, class_id)
            elif class_id is not None:
                mapped = class_id

            inst.meta["category_score"] = category_score
            inst.meta["cls_prob"] = category_score
            inst.meta["mask_score"] = float(inst.score)
            if class_id is not None:
                inst.meta["category_id"] = class_id
            if "neighbors" in pred:
                inst.meta["knn_neighbors"] = pred["neighbors"]
            if "prototype_similarity" in pred:
                inst.meta["prototype_similarity"] = float(pred["prototype_similarity"])
            if "prototype_index" in pred:
                inst.meta["prototype_index"] = int(pred["prototype_index"])

            if isinstance(mapped, str):
                inst.meta["class_name"] = mapped
            elif isinstance(mapped, int):
                inst.class_id = int(mapped)
            elif class_id is not None:
                inst.class_id = class_id
            else:
                label = str(raw_cls) if raw_cls is not None else "unknown"
                inst.meta["class_name"] = label

            if self.combine_score == "mul":
                inst.score = float(inst.score) * category_score
            elif self.combine_score == "replace":
                inst.score = category_score

            if self.score_threshold > 0.0 and inst.score < self.score_threshold:
                updated[inst_idx] = None

        return [inst for inst in updated if inst is not None]
