"""训练与验证的变换。"""

from __future__ import annotations

import random
from typing import Callable, Dict, Any, List, Optional, Tuple

from distill.core.structures import PseudoLabelInstance

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch 可选
    torch = None
    F = None  # type: ignore


class Compose:
    """组合多个变换。"""

    def __init__(self, transforms: List[Callable[[Dict[str, Any]], Dict[str, Any]]]) -> None:
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToFloat:
    """将图像转换为 float 并缩放到 [0,1]。"""

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if torch is None:
            return sample
        image = sample["image"]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        sample["image"] = image
        return sample


class Normalize:
    """对图像进行归一化。"""

    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if torch is None:
            return sample
        image = sample["image"]
        mean = torch.tensor(self.mean, device=image.device).view(-1, 1, 1)
        std = torch.tensor(self.std, device=image.device).view(-1, 1, 1)
        sample["image"] = (image - mean) / std
        return sample


class RandomHorizontalFlip:
    """随机水平翻转。"""

    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if torch is None:
            return sample
        if random.random() > self.prob:
            return sample

        image = sample["image"]
        _, height, width = image.shape
        image = torch.flip(image, dims=[2])
        sample["image"] = image

        instances: List[PseudoLabelInstance] = sample.get("instances", [])
        for inst in instances:
            x, y, w, h = inst.bbox
            new_x = max(0.0, float(width) - float(x) - float(w))
            inst.bbox = (new_x, float(y), float(w), float(h))
            if inst.mask is not None:
                mask = inst.mask
                if isinstance(mask, torch.Tensor):
                    inst.mask = torch.flip(mask, dims=[1])
                else:
                    inst.mask = mask[:, ::-1]

        meta = sample.get("meta", {})
        meta["flipped"] = True
        sample["meta"] = meta
        return sample


class Resize:
    """将图像与 mask 缩放到指定大小。"""

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if torch is None or F is None:
            return sample
        image = sample["image"]
        _, height, width = image.shape
        new_h, new_w = self.size
        if height == new_h and width == new_w:
            return sample

        image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        sample["image"] = image

        scale_x = new_w / width
        scale_y = new_h / height
        instances: List[PseudoLabelInstance] = sample.get("instances", [])
        for inst in instances:
            x, y, w, h = inst.bbox
            inst.bbox = (x * scale_x, y * scale_y, w * scale_x, h * scale_y)
            if inst.mask is not None:
                mask = inst.mask
                if not isinstance(mask, torch.Tensor):
                    mask = torch.as_tensor(mask, dtype=torch.float32)
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask[0]
                mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0).squeeze(0)
                inst.mask = mask

        meta = sample.get("meta", {})
        meta["height"] = new_h
        meta["width"] = new_w
        sample["meta"] = meta
        return sample


def _build_transforms(tf_cfg: Dict[str, Any], enable_flip: bool) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """按配置构建变换序列。"""

    transforms: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = []
    if tf_cfg.get("to_float", True):
        transforms.append(ToFloat())

    if enable_flip:
        flip_prob = float(tf_cfg.get("flip_prob", 0.5))
        if flip_prob > 0:
            transforms.append(RandomHorizontalFlip(prob=flip_prob))

    resize = tf_cfg.get("resize")
    if isinstance(resize, (list, tuple)) and len(resize) == 2:
        transforms.append(Resize((int(resize[0]), int(resize[1]))))

    mean = tf_cfg.get("mean")
    std = tf_cfg.get("std")
    if mean is not None and std is not None:
        transforms.append(Normalize(tuple(mean), tuple(std)))

    if not transforms:
        return lambda x: x
    return Compose(transforms)


def build_train_transforms(cfg: Dict[str, Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """构建训练变换。"""

    data_cfg = cfg.get("data", {})
    train_cfg = data_cfg.get("train", {})
    tf_cfg = train_cfg.get("transforms", {})
    return _build_transforms(tf_cfg, enable_flip=True)


def build_eval_transforms(cfg: Dict[str, Any], override: Optional[Dict[str, Any]] = None) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """构建评估变换。"""

    data_cfg = cfg.get("data", {})
    eval_cfg = data_cfg.get("eval", {})
    tf_cfg = override if isinstance(override, dict) else eval_cfg.get("transforms", {})
    return _build_transforms(tf_cfg, enable_flip=False)
