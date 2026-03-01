"""实例分类模型适配器。"""

from __future__ import annotations
import time
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from torchvision import transforms

from distill.core.structures import InstancePrediction
from distill.utils.logging import get_logger
from segment_cdw.classier_network.CNNs import ResNet50, seresnext50_32x4d


import torch
import torch.nn.functional as F



try:
    from PIL import Image
except Exception:  # pragma: no cover - 可选依赖
    Image = None  # type: ignore


def _square_crop_coords(
    img_hw: Tuple[int, int], bbox_xyxy: Tuple[float, float, float, float], margin_ratio: float = 0.1
) -> Tuple[int, int, int, int]:
    """根据 bbox 计算正方形裁剪区域（含 margin），返回坐标。"""

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
) -> np.ndarray:
    """使用 mask 抠出目标后再做正方形裁剪，非 mask 区域置黑。"""

    height, width = img.shape[:2]
    x1n, y1n, x2n, y2n = _square_crop_coords((height, width), bbox_xyxy, margin_ratio)
    img_crop = img[y1n:y2n, x1n:x2n].copy()
    if mask is None:
        return img_crop
    mask_crop = mask[y1n:y2n, x1n:x2n]
    if mask_crop.dtype != bool:
        mask_crop = mask_crop.astype(bool)
    img_crop[~mask_crop] = 0
    return img_crop


def _torch_masked_square_crop(
    img_t: "torch.Tensor",
    mask_t: Optional["torch.Tensor"],
    bbox_xyxy: Tuple[float, float, float, float],
    margin_ratio: float,
    patch_size: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> "torch.Tensor":
    """GPU-friendly crop + resize + normalize."""
    if torch is None:
        raise RuntimeError("PyTorch 不可用，无法执行 GPU crop。")
    h, w = int(img_t.shape[1]), int(img_t.shape[2])
    x1n, y1n, x2n, y2n = _square_crop_coords((h, w), bbox_xyxy, margin_ratio)
    patch = img_t[:, y1n:y2n, x1n:x2n].clone()
    if mask_t is not None:
        if mask_t.dtype != torch.bool:
            mask_t = mask_t > 0.5
        mask_crop = mask_t[y1n:y2n, x1n:x2n]
        patch[:, ~mask_crop] = 0.0
    patch = patch.unsqueeze(0)
    patch = torch.nn.functional.interpolate(patch, size=(patch_size, patch_size), mode="bilinear", align_corners=False)
    patch = patch.squeeze(0)
    mean_t = torch.tensor(mean, device=patch.device).view(3, 1, 1)
    std_t = torch.tensor(std, device=patch.device).view(3, 1, 1)
    patch = (patch - mean_t) / std_t
    return patch


def _try_int(value: Any) -> Optional[int]:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


CategoryMap = Dict[int, Union[int, str]]


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
        mapping: CategoryMap = {}
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
    try:
        file_path = Path(path)
    except TypeError:
        return None
    data = json.loads(file_path.read_text(encoding="utf-8"))
    mapping: CategoryMap = {}
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


def _ensure_segment_cdw_sys_path(cfg: Dict[str, Any]) -> Optional[Path]:
    """确保 segment_cdw 与 classier_network 都可被导入。"""

    def _add_path(path: Path) -> None:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    candidates: List[Path] = []
    user_path = cfg.get("segment_cdw_path") or cfg.get("segment_cdw_root") or cfg.get("segment_cdw_dir")
    if user_path:
        candidates.append(Path(user_path))

    try:
        for parent in Path(__file__).resolve().parents:
            seg_dir = parent / "segment_cdw"
            if seg_dir.exists():
                candidates.append(seg_dir)
                candidates.append(parent)
                break
    except Exception:
        pass

    for path in candidates:
        if not path.exists():
            continue
        if path.name == "segment_cdw":
            _add_path(path)
            _add_path(path.parent)
            return path
        seg_dir = path / "segment_cdw"
        if seg_dir.exists():
            _add_path(seg_dir)
            _add_path(path)
            return seg_dir
        if (path / "classier_network").exists():
            _add_path(path)
            return path
    return None


class ClassifierPatchTransform:
    """统一的 patch 处理流程：resize + to_tensor + normalize。"""

    def __init__(
        self,
        patch_size: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ) -> None:
        self.patch_size = patch_size
        self.norm = transforms.Normalize(mean=mean, std=std)

    def __call__(self, patch: np.ndarray) -> "torch.Tensor":
        patch_img = Image.fromarray(patch).resize((self.patch_size, self.patch_size), resample=Image.BILINEAR)
        tensor = torch.from_numpy(np.array(patch_img)).permute(2, 0, 1).float() / 255.0
        tensor = self.norm(tensor)
        return tensor


def _classify_patches_batch(
    model: "torch.nn.Module",
    patches: List["torch.Tensor"],
    device: str,
    batch_size: int ,
) -> List[Tuple[int, float]]:
    """对 patch 批量分类，返回 [(class_id, class_prob), ...]。"""

    if torch is None or F is None:
        raise RuntimeError("PyTorch/torchvision 未安装，无法进行分类。")
    if len(patches) == 0:
        return []

    outputs: List[Tuple[int, float]] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = torch.stack(patches[i : i + batch_size]).to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            conf, cls = torch.max(probs, dim=1)
            outputs.extend([(int(c), float(p)) for c, p in zip(cls.cpu(), conf.cpu())])
    return outputs


class ClassifierAdapter:
    """
    用于为实例分配类别标签的分类器封装。

    backend:
      - "segment_cdw": 
      - "none": 
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.logger = get_logger("distill")
        self.num_classes = int(cfg.get("num_classes", 1))
        self.score_threshold = float(cfg.get("score_threshold", 0.0))
        self.backend = str(cfg.get("backend", "none")).lower()
        self.patch_size = int(cfg.get("patch_size", 224))
        self.margin_ratio = float(cfg.get("margin_ratio", 0.1))
        self.batch_size = int(cfg.get("batch_size", 32))
        self.normalize_mean = tuple(cfg.get("normalize_mean", cfg.get("mean", (0.485, 0.456, 0.406))))
        self.normalize_std = tuple(cfg.get("normalize_std", cfg.get("std", (0.229, 0.224, 0.225))))
        self.combine_score = str(cfg.get("combine_score", "mul")).lower()
        self.category_map = _load_category_map(cfg.get("category_map"))
        self.use_gpu_crop = bool(cfg.get("use_gpu_crop", False))
        self._warned_norm = False
        self.patch_transform = ClassifierPatchTransform(
            patch_size=self.patch_size,
            mean=self.normalize_mean,
            std=self.normalize_std,
        )

        self.model = None
        self.device = str(cfg.get("device"))

        if self.backend == "segment_cdw":
            self._build_segment_cdw_model()
        elif self.backend != "none":
            self.logger.warning("未知分类器 backend: %s", self.backend)

    def _build_segment_cdw_model(self) -> None:
        _ensure_segment_cdw_sys_path(self.cfg)

        ckpt_path = self.cfg.get("checkpoint", "")
        if not ckpt_path:
            self.logger.warning("未提供分类模型权重路径，跳过加载")
            return

        model = seresnext50_32x4d(pretrained=False, out_features=self.num_classes)
        model = torch.nn.DataParallel(model)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(self.device)
        model.eval()
        self.model = model
        self.logger.info("分类模型已加载: %s (device=%s)", ckpt_path, self.device)

    def classify(self, instances: List[InstancePrediction], 
                 image_np: Optional[np.ndarray] = None) -> List[InstancePrediction]:
        if not instances:
            self.logger.warning("实例为空")
            return instances
        if self.model is None or image_np is None:
            self.logger.warning("模型为空或原图为空")
            return instances

        patches: List["torch.Tensor"] = []
        valid_instances: List[InstancePrediction] = []
        t5 = time.perf_counter()
        image_t: Optional["torch.Tensor"] = None
        use_gpu_crop = self.use_gpu_crop and self.device.startswith("cuda") and torch is not None
        if use_gpu_crop:
            img_t = torch.from_numpy(image_np).to(self.device)
            if img_t.ndim == 3:
                img_t = img_t.permute(2, 0, 1)
            image_t = img_t.float() / 255.0
        for inst in instances:
            x, y, w, h = inst.bbox
            bbox_xyxy = (float(x), float(y), float(x + w), float(y + h))
            mask = inst.mask
            if mask is not None:
                if torch is not None and isinstance(mask, torch.Tensor):
                    mask_t = mask.detach().to(self.device) if use_gpu_crop else mask.detach().cpu().numpy()
                else:
                    mask_np = np.asarray(mask)
                    if use_gpu_crop:
                        mask_t = torch.from_numpy(mask_np).to(self.device)
                    else:
                        mask = mask_np
                if use_gpu_crop:
                    if mask_t.ndim == 3 and mask_t.shape[0] == 1:
                        mask_t = mask_t[0]
                else:
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask[0]
            else:
                mask_t = None
            
            # 裁剪patch
            if use_gpu_crop and image_t is not None:
                patch_tensor = _torch_masked_square_crop(
                    image_t,
                    mask_t,
                    bbox_xyxy,
                    margin_ratio=self.margin_ratio,
                    patch_size=self.patch_size,
                    mean=self.normalize_mean,
                    std=self.normalize_std,
                )
            else:
                patch = _masked_square_crop(image_np, mask, bbox_xyxy, margin_ratio=self.margin_ratio)
                patch_tensor = self.patch_transform(patch)
  
            patches.append(patch_tensor)
            valid_instances.append(inst)

        t6 = time.perf_counter()
        print(f"裁剪用时: {(t6 - t5) * 1000:.2f} ms")
        
        if not patches:
            return instances

        try:
            t7 = time.perf_counter()
            cls_outputs = _classify_patches_batch(
                self.model,
                patches,
                device=self.device,
                batch_size=self.batch_size,
            )
            t8 = time.perf_counter()
            print(f"分类用时: {(t8 - t7) * 1000:.2f} ms")
        except Exception as exc:
            self.logger.warning("分类失败: %s", exc)
            return instances

        filtered: List[InstancePrediction] = []
        for inst, (class_id, class_prob) in zip(valid_instances, cls_outputs):
            mapped = self.category_map.get(class_id, class_id) if self.category_map else class_id
            inst.meta["cls_prob"] = float(class_prob)
            inst.meta["mask_score"] = float(inst.score)
            if isinstance(mapped, str):
                inst.meta["class_name"] = mapped
                inst.class_id = int(class_id)
            else:
                inst.class_id = int(mapped)

            if self.combine_score == "mul":
                inst.score = float(inst.score) * float(class_prob)
            elif self.combine_score == "replace":
                inst.score = float(class_prob)

            if self.score_threshold > 0.0 and inst.score < self.score_threshold:
                continue
            filtered.append(inst)

        return filtered
