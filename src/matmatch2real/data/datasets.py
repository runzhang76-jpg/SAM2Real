""""""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json

from matmatch2real.core.structures import PseudoLabelInstance
from matmatch2real.data.pseudolabel_io import decode_segmentation, read_pseudolabels
from matmatch2real.data.transforms import build_train_transforms, build_eval_transforms
from matmatch2real.student.distill_soft.io import load_teacher_soft_sample, resolve_teacher_soft_path
from matmatch2real.utils.logging import get_logger

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - torch
    torch = None
    DataLoader = object  # type: ignore
    Dataset = object  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL
    Image = None  # type: ignore


def _load_image(path: Path) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required for dataset loading.")
    if Image is None:
        # PIL
        return torch.zeros(3, 256, 256)
    image = Image.open(path).convert("RGB")
    byte_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    img = byte_tensor.view(image.size[1], image.size[0], 3).permute(2, 0, 1)
    return img.float() / 255.0


def _collect_image_paths(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])


def _pad_images(images: List["torch.Tensor"]) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required for dataset loading.")
    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)
    batch = torch.zeros(len(images), images[0].shape[0], max_h, max_w)
    for i, img in enumerate(images):
        h, w = img.shape[-2], img.shape[-1]
        batch[i, :, :h, :w] = img
    return batch


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


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


def _infer_class_id_maps_from_annotation_ids(class_ids: List[int], num_classes: int) -> Tuple[Dict[int, int], Dict[int, int]]:
    valid_ids = sorted({int(cid) for cid in class_ids if int(cid) >= 0})
    if not valid_ids:
        return {}, {}

    candidate_ids = list(valid_ids)
    if len(candidate_ids) == num_classes + 1 and 0 in candidate_ids:
        candidate_ids = [cid for cid in candidate_ids if cid != 0]

    if len(candidate_ids) != num_classes:
        return {}, {}

    orig_to_train = {orig_id: train_id for train_id, orig_id in enumerate(candidate_ids)}
    train_to_orig = {train_id: orig_id for orig_id, train_id in orig_to_train.items()}
    return orig_to_train, train_to_orig


def prepare_class_mappings(cfg: Dict[str, Any], data_root: Path) -> None:
    student_cfg = cfg.setdefault("student", {})
    num_classes = int(student_cfg.get("num_classes", 0) or 0)
    if num_classes <= 0:
        return
    logger = get_logger("distill")

    existing_orig_to_train = _parse_class_id_map(student_cfg.get("orig_to_train_class_id"))
    existing_train_to_orig = _parse_class_id_map(student_cfg.get("train_to_orig_class_id"))
    if existing_orig_to_train and existing_train_to_orig:
        return

    inferred_orig_to_train: Dict[int, int] = {}
    inferred_train_to_orig: Dict[int, int] = {}

    train_cfg = cfg.get("data", {}).get("train", {})
    train_root = _resolve_path(data_root, str(train_cfg.get("root_dir", ""))) if train_cfg.get("root_dir") else data_root
    pseudolabel_path = _resolve_path(train_root, str(train_cfg.get("pseudolabel_path", "pseudolabels.json")))
    if pseudolabel_path.exists():
        try:
            with pseudolabel_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            ann_ids = [int(ann.get("category_id", ann.get("class_id", -1))) for ann in payload.get("annotations", [])]
            inferred_orig_to_train, inferred_train_to_orig = _infer_class_id_maps_from_annotation_ids(ann_ids, num_classes)
        except Exception:
            inferred_orig_to_train, inferred_train_to_orig = {}, {}

    if not inferred_orig_to_train:
        eval_cfg = cfg.get("data", {}).get("eval", {})
        gt_json = str(eval_cfg.get("gt_json", "")).strip()
        if gt_json:
            eval_root = _resolve_path(data_root, str(eval_cfg.get("root_dir", ""))) if eval_cfg.get("root_dir") else data_root
            ann_path = _resolve_path(eval_root, gt_json)
            if ann_path.exists():
                try:
                    with ann_path.open("r", encoding="utf-8") as f:
                        payload = json.load(f)
                    cat_ids = [int(cat.get("id", -1)) for cat in payload.get("categories", [])]
                    inferred_orig_to_train, inferred_train_to_orig = _infer_class_id_maps_from_annotation_ids(cat_ids, num_classes)
                except Exception:
                    inferred_orig_to_train, inferred_train_to_orig = {}, {}

    if inferred_orig_to_train and inferred_train_to_orig:
        student_cfg["orig_to_train_class_id"] = {int(k): int(v) for k, v in inferred_orig_to_train.items()}
        student_cfg["train_to_orig_class_id"] = {int(k): int(v) for k, v in inferred_train_to_orig.items()}
        logger.info(
            "prepared class-id mapping: orig_to_train=%s train_to_orig=%s",
            student_cfg["orig_to_train_class_id"],
            student_cfg["train_to_orig_class_id"],
        )


def _parse_image_info(image_info: Dict[str, Any], fallback_id: int) -> Tuple[int, str, int, int]:
    image_id = int(image_info.get("id", fallback_id))
    file_name = image_info.get("file_name", "")
    height = int(image_info.get("height", 256))
    width = int(image_info.get("width", 256))
    return image_id, file_name, height, width


def _load_image_or_empty(image_path: Optional[Path], height: int, width: int) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required for dataset loading.")
    if image_path and image_path.exists():
        return _load_image(image_path)
    return torch.zeros(3, height, width)


def _decode_instance_masks(instances: List[PseudoLabelInstance], height: int, width: int) -> None:
    for inst in instances:
        if inst.mask is not None or inst.rle is None:
            continue
        mask = decode_segmentation(inst.rle, height, width)
        if mask is None:
            continue
        inst.mask = mask


def _make_sample(
    image: "torch.Tensor",
    image_id: int,
    instances: List[PseudoLabelInstance],
    meta: Dict[str, Any],
    transforms: Optional[Any],
) -> Dict[str, Any]:
    sample = {
        "image": image,
        "image_id": image_id,
        "instances": instances,
        "meta": meta,
    }
    if transforms:
        sample = transforms(sample)
    return sample


def _as_float_tensor(value: Any) -> Optional["torch.Tensor"]:
    if torch is None or value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    return torch.as_tensor(value, dtype=torch.float32)


def _teacher_soft_debug_enabled(debug_cfg: Dict[str, Any]) -> bool:
    return bool(debug_cfg.get("enabled", False))


def _validate_teacher_soft_sample(
    teacher_soft: Dict[str, Any],
    image_hw: Tuple[int, int],
    *,
    context: str,
    strict: bool,
) -> Dict[str, float]:
    if torch is None:
        raise RuntimeError("PyTorch is required for dataset loading.")

    image_h, image_w = image_hw
    stats: Dict[str, float] = {}
    mask_logits = _as_float_tensor(teacher_soft.get("img_mask_logits"))
    if mask_logits is not None:
        if mask_logits.ndim != 2:
            raise AssertionError(f"{context}: img_mask_logits must be HxW, got {tuple(mask_logits.shape)}")
        if tuple(mask_logits.shape) != (image_h, image_w):
            raise AssertionError(
                f"{context}: img_mask_logits shape {tuple(mask_logits.shape)} does not match image {(image_h, image_w)}"
            )
        if not torch.isfinite(mask_logits).all():
            raise AssertionError(f"{context}: img_mask_logits contains NaN/Inf")
        stats["mask_min"] = float(mask_logits.min().item())
        stats["mask_max"] = float(mask_logits.max().item())

    boundary_map = _as_float_tensor(teacher_soft.get("img_boundary"))
    if boundary_map is not None:
        if boundary_map.ndim != 2:
            raise AssertionError(f"{context}: img_boundary must be HxW, got {tuple(boundary_map.shape)}")
        if tuple(boundary_map.shape) != (image_h, image_w):
            raise AssertionError(
                f"{context}: img_boundary shape {tuple(boundary_map.shape)} does not match image {(image_h, image_w)}"
            )
        if not torch.isfinite(boundary_map).all():
            raise AssertionError(f"{context}: img_boundary contains NaN/Inf")
        boundary_min = float(boundary_map.min().item())
        boundary_max = float(boundary_map.max().item())
        stats["boundary_min"] = boundary_min
        stats["boundary_max"] = boundary_max
        if strict and (boundary_min < -1e-4 or boundary_max > 1.0001):
            raise AssertionError(
                f"{context}: img_boundary out of range [0, 1], got min={boundary_min:.4f} max={boundary_max:.4f}"
            )

    class_soft = _as_float_tensor(teacher_soft.get("img_class_soft"))
    if class_soft is not None:
        if class_soft.ndim != 1:
            raise AssertionError(f"{context}: img_class_soft must be 1D, got {tuple(class_soft.shape)}")
        if not torch.isfinite(class_soft).all():
            raise AssertionError(f"{context}: img_class_soft contains NaN/Inf")
        class_sum = float(class_soft.sum().item())
        stats["class_sum"] = class_sum
        stats["class_min"] = float(class_soft.min().item())
        stats["class_max"] = float(class_soft.max().item())
        if strict and abs(class_sum - 1.0) > 1e-2:
            raise AssertionError(f"{context}: img_class_soft is not normalized, sum={class_sum:.4f}")

    img_score = _as_float_tensor(teacher_soft.get("img_score"))
    if img_score is not None:
        if not torch.isfinite(img_score).all():
            raise AssertionError(f"{context}: img_score contains NaN/Inf")
        stats["score"] = float(img_score.reshape(-1)[0].item())

    return stats


class RawImageDataset(Dataset):
    """"""

    def __init__(self, images_dir: Path, transforms: Optional[Any] = None, allow_empty: bool = False) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for dataset loading.")
        self.images_dir = images_dir
        self.transforms = transforms
        self.image_paths = _collect_image_paths(images_dir)
        if not self.image_paths and not allow_empty:
            raise FileNotFoundError(f"No images found in {images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.image_paths[index]
        image = _load_image(path)
        meta = {"path": str(path), "height": image.shape[-2], "width": image.shape[-1]}
        return _make_sample(image, index, [], meta, self.transforms)


class ImagePathDataset(Dataset):
    """"""

    def __init__(self, image_paths: List[Path], allow_empty: bool = False) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for dataset loading.")
        self.image_paths = image_paths
        if not self.image_paths and not allow_empty:
            raise FileNotFoundError("")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.image_paths[index]
        meta = {"path": str(path), "height": 0, "width": 0, "image_id": index}
        return {"image_id": index, "meta": meta}


class PseudoLabelDataset(Dataset):
    """"""

    def __init__(
        self,
        images_dir: Path,
        pseudolabel_path: Path,
        transforms: Optional[Any] = None,
        allow_empty: bool = False,
        teacher_soft_dir: Optional[Path] = None,
        debug_cfg: Optional[Dict[str, Any]] = None,
        orig_to_train_class_id: Optional[Dict[int, int]] = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for dataset loading.")
        self.logger = get_logger("distill")
        self.images_dir = images_dir
        self.transforms = transforms
        self.teacher_soft_dir = teacher_soft_dir
        self.debug_cfg = dict(debug_cfg or {})
        self._debug_limit = max(0, int(self.debug_cfg.get("max_samples", 8)))
        self._debug_count = 0
        self._missing_teacher_soft: set[str] = set()
        self.images, self.instances_by_image, _meta = read_pseudolabels(str(pseudolabel_path))
        self.orig_to_train_class_id = {int(k): int(v) for k, v in (orig_to_train_class_id or {}).items()}
        self._apply_class_id_mapping()
        if not self.images and not allow_empty:
            raise RuntimeError("Pseudo label file is empty.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_info = self.images[index]
        image_id, file_name, height, width = _parse_image_info(image_info, index)
        image_path = self.images_dir / file_name if file_name else None
        image = _load_image_or_empty(image_path, height, width)
        # Transforms mutate boxes/masks in place, so each sample must get detached copies.
        instances = [deepcopy(inst) for inst in self.instances_by_image.get(image_id, [])]
        _decode_instance_masks(instances, int(image.shape[-2]), int(image.shape[-1]))
        meta = {"path": str(image_path) if image_path else "", "height": image.shape[-2], "width": image.shape[-1]}
        sample = _make_sample(image, image_id, instances, meta, None)
        teacher_soft = self._load_teacher_soft(file_name)
        if teacher_soft is not None:
            sample["teacher_soft"] = teacher_soft
        if self.transforms:
            sample = self.transforms(sample)
        self._audit_teacher_soft(file_name, sample)
        return sample

    def _load_teacher_soft(self, file_name: str) -> Optional[Dict[str, Any]]:
        if self.teacher_soft_dir is None:
            return None
        soft_path = resolve_teacher_soft_path(self.teacher_soft_dir, file_name)
        if soft_path is None:
            return None
        teacher_soft = load_teacher_soft_sample(soft_path)
        if teacher_soft is None and str(soft_path) not in self._missing_teacher_soft:
            self._missing_teacher_soft.add(str(soft_path))
            self.logger.warning("teacher soft file missing or invalid: %s", soft_path)
        return teacher_soft

    def _audit_teacher_soft(self, file_name: str, sample: Dict[str, Any]) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for dataset loading.")
        teacher_soft = sample.get("teacher_soft")
        debug_enabled = _teacher_soft_debug_enabled(self.debug_cfg)
        strict = bool(self.debug_cfg.get("strict", True))
        image = sample.get("image")
        if not isinstance(image, torch.Tensor):
            return
        image_hw = (int(image.shape[-2]), int(image.shape[-1]))
        if teacher_soft is None:
            if debug_enabled and self._debug_count < self._debug_limit:
                self.logger.info(
                    "teacher_soft dataset-check file=%s exists=%s image_hw=%s",
                    file_name,
                    False,
                    image_hw,
                )
                self._debug_count += 1
            return
        stats = _validate_teacher_soft_sample(
            teacher_soft,
            image_hw,
            context=f"teacher_soft[{file_name}]",
            strict=strict,
        )
        if debug_enabled and self._debug_count < self._debug_limit:
            class_ids = teacher_soft.get("class_ids")
            class_ids_list = None
            if isinstance(class_ids, torch.Tensor):
                class_ids_list = [int(v) for v in class_ids.detach().cpu().flatten().tolist()]
            self.logger.info(
                "teacher_soft dataset-check file=%s exists=%s image_hw=%s mask_shape=%s mask_range=[%.4f, %.4f] "
                "boundary_range=[%.4f, %.4f] class_sum=%.4f class_range=[%.4f, %.4f] score=%.4f class_ids=%s src=%s",
                file_name,
                True,
                image_hw,
                tuple(_as_float_tensor(teacher_soft.get('img_mask_logits')).shape) if teacher_soft.get("img_mask_logits") is not None else None,
                stats.get("mask_min", 0.0),
                stats.get("mask_max", 0.0),
                stats.get("boundary_min", 0.0),
                stats.get("boundary_max", 0.0),
                stats.get("class_sum", 0.0),
                stats.get("class_min", 0.0),
                stats.get("class_max", 0.0),
                stats.get("score", 0.0),
                class_ids_list,
                teacher_soft.get("source_path", ""),
            )
            self._debug_count += 1

    def _apply_class_id_mapping(self) -> None:
        if not self.orig_to_train_class_id:
            return
        for instances in self.instances_by_image.values():
            for inst in instances:
                orig_class_id = int(inst.class_id)
                if orig_class_id not in self.orig_to_train_class_id:
                    raise ValueError(
                        f"Pseudo label class_id={orig_class_id} is not present in orig_to_train_class_id mapping "
                        f"{self.orig_to_train_class_id}"
                    )
                inst.meta["orig_category_id"] = orig_class_id
                inst.class_id = int(self.orig_to_train_class_id[orig_class_id])


class DummyDataset(Dataset):
    """"""

    def __init__(self, num_images: int, image_size: Tuple[int, int], num_instances: int = 1) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for dataset loading.")
        self.num_images = num_images
        self.image_size = image_size
        self.num_instances = num_instances

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, index: int) -> Dict[str, Any]:
        height, width = self.image_size
        image = torch.zeros(3, height, width)
        instances: List[PseudoLabelInstance] = []
        for inst_id in range(self.num_instances):
            bbox = (10.0, 10.0, float(width // 4), float(height // 4))
            mask = torch.zeros(height, width)
            mask[10 : 10 + height // 4, 10 : 10 + width // 4] = 1.0
            instances.append(
                PseudoLabelInstance(
                    image_id=index,
                    bbox=bbox,
                    class_id=1,
                    score=1.0,
                    reliability=1.0,
                    mask=mask,
                    instance_id=inst_id,
                )
            )
        return {
            "image": image,
            "image_id": index,
            "instances": instances,
            "meta": {"path": "", "height": height, "width": width},
        }


class CocoImageDataset(Dataset):
    """ COCO """

    def __init__(
        self,
        images_dir: Path,
        annotation_path: Path,
        transforms: Optional[Any] = None,
        allow_empty: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for dataset loading.")
        self.images_dir = images_dir
        self.transforms = transforms
        if not annotation_path.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")
        with annotation_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.images = payload.get("images", [])
        if not self.images and not allow_empty:
            raise RuntimeError("COCO annotation file has no images.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_info = self.images[index]
        image_id, file_name, height, width = _parse_image_info(image_info, index)
        image_path = self.images_dir / file_name if file_name else None
        image = _load_image_or_empty(image_path, height, width)
        meta = {"path": str(image_path) if image_path else "", "height": image.shape[-2], "width": image.shape[-1]}
        return _make_sample(image, image_id, [], meta, self.transforms)


class CocoImageMetaDataset(Dataset):
    """ COCO """

    def __init__(
        self,
        images_dir: Path,
        annotation_path: Path,
        allow_empty: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for dataset loading.")
        self.images_dir = images_dir
        if not annotation_path.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")
        with annotation_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.images = payload.get("images", [])
        if not self.images and not allow_empty:
            raise RuntimeError("COCO annotation file has no images.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_info = self.images[index]
        image_id, file_name, height, width = _parse_image_info(image_info, index)
        image_path = self.images_dir / file_name if file_name else None
        meta = {"path": str(image_path) if image_path else "", "height": height, "width": width, "image_id": image_id}
        return {"image_id": image_id, "meta": meta}


def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = [b["image"] for b in batch]
    if torch is None:
        raise RuntimeError("PyTorch is required for dataset loading.")
    images = _pad_images(images)
    teacher_soft = _collate_teacher_soft([b.get("teacher_soft") for b in batch])
    if teacher_soft is not None:
        mask_logits = teacher_soft.get("img_mask_logits")
        boundary_map = teacher_soft.get("img_boundary")
        if isinstance(mask_logits, torch.Tensor):
            assert mask_logits.shape[0] == images.shape[0], (
                f"teacher_soft batch mismatch: mask batch {mask_logits.shape[0]} vs images {images.shape[0]}"
            )
            assert tuple(mask_logits.shape[-2:]) == tuple(images.shape[-2:]), (
                f"teacher_soft mask/image shape mismatch: {tuple(mask_logits.shape[-2:])} vs {tuple(images.shape[-2:])}"
            )
        if isinstance(boundary_map, torch.Tensor):
            assert boundary_map.shape[0] == images.shape[0], (
                f"teacher_soft batch mismatch: boundary batch {boundary_map.shape[0]} vs images {images.shape[0]}"
            )
            assert tuple(boundary_map.shape[-2:]) == tuple(images.shape[-2:]), (
                f"teacher_soft boundary/image shape mismatch: {tuple(boundary_map.shape[-2:])} vs {tuple(images.shape[-2:])}"
            )
    result = {
        "images": images,
        "image_ids": [b["image_id"] for b in batch],
        "instances": [b.get("instances", []) for b in batch],
        "meta": [b.get("meta", {}) for b in batch],
    }
    if teacher_soft is not None:
        result["teacher_soft"] = teacher_soft
    return result


def _collate_teacher_soft(batch_soft: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for dataset loading.")
    if not any(item is not None for item in batch_soft):
        return None

    collated: Dict[str, Any] = {}
    batch_size = len(batch_soft)
    field_specs = {
        "img_mask_logits": 2,
        "img_boundary": 2,
        "img_class_soft": 1,
    }
    for key, ndim in field_specs.items():
        reference = next((item.get(key) for item in batch_soft if item is not None and item.get(key) is not None), None)
        if reference is None:
            collated[key] = None
            collated[f"has_{key}"] = torch.zeros(batch_size, dtype=torch.float32)
            continue
        ref_tensor = reference if isinstance(reference, torch.Tensor) else torch.as_tensor(reference, dtype=torch.float32)
        values: List[torch.Tensor] = []
        valid: List[float] = []
        for item in batch_soft:
            value = item.get(key) if item is not None else None
            if value is None:
                values.append(torch.zeros_like(ref_tensor, dtype=torch.float32))
                valid.append(0.0)
                continue
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
            if tuple(tensor.shape) != tuple(ref_tensor.shape):
                raise AssertionError(
                    f"teacher_soft field {key} shape mismatch inside batch: {tuple(tensor.shape)} vs {tuple(ref_tensor.shape)}"
                )
            if not torch.isfinite(tensor.float()).all():
                raise AssertionError(f"teacher_soft field {key} contains NaN/Inf during collate")
            values.append(tensor.float())
            valid.append(1.0)
        collated[key] = torch.stack(values)
        collated[f"has_{key}"] = torch.tensor(valid, dtype=torch.float32)
        if ndim == 1 and collated[key].ndim == 1:
            collated[key] = collated[key].unsqueeze(0)

    scores: List[torch.Tensor] = []
    for item in batch_soft:
        score = item.get("img_score") if item is not None else None
        if score is None:
            scores.append(torch.tensor(0.0, dtype=torch.float32))
        elif isinstance(score, torch.Tensor):
            scores.append(score.float().reshape(()))
        else:
            scores.append(torch.tensor(float(score), dtype=torch.float32))
    collated["img_score"] = torch.stack(scores)
    class_ids = next((item.get("class_ids") for item in batch_soft if item is not None and item.get("class_ids") is not None), None)
    if class_ids is not None:
        ref_class_ids = class_ids if isinstance(class_ids, torch.Tensor) else torch.as_tensor(class_ids, dtype=torch.int64)
        for item in batch_soft:
            value = item.get("class_ids") if item is not None else None
            if value is None:
                continue
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.int64)
            if tuple(tensor.shape) != tuple(ref_class_ids.shape) or not torch.equal(tensor.long(), ref_class_ids.long()):
                raise AssertionError("teacher_soft class_ids mismatch inside batch; class alignment is inconsistent.")
        collated["class_ids"] = ref_class_ids.long()
    return collated


def _teacher_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image_ids": [b.get("image_id") for b in batch],
        "meta": [b.get("meta", {}) for b in batch],
    }


def _build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    collate_fn: Optional[Any] = None,
) -> DataLoader:
    if torch is None:
        raise RuntimeError("PyTorch is required for dataloader.")
    if collate_fn is None:
        collate_fn = _collate_fn
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def build_train_dataset(cfg: Dict[str, Any], data_root: Path) -> Dataset:
    """"""

    logger = get_logger("distill")
    prepare_class_mappings(cfg, data_root)
    data_cfg = cfg.get("data", {})
    train_cfg = data_cfg.get("train", {})
    transforms = build_train_transforms(cfg)

    source = train_cfg.get("source", "offline")
    images_dir = _resolve_path(data_root, str(train_cfg.get("images_dir", "images")))
    pseudolabel_path = _resolve_path(data_root, str(train_cfg.get("pseudolabel_path", "pseudolabels.json")))
    allow_empty = bool(train_cfg.get("allow_empty", False))
    teacher_soft_dir = _resolve_teacher_soft_dir(cfg, data_root) if source == "offline" else None

    if teacher_soft_dir is not None:
        _validate_teacher_soft_augmentations(cfg)

    if source == "dummy":
        dummy_cfg = train_cfg.get("dummy", {})
        return DummyDataset(
            num_images=int(dummy_cfg.get("num_images", 8)),
            image_size=tuple(dummy_cfg.get("image_size", [256, 256])),
            num_instances=int(dummy_cfg.get("num_instances", 1)),
        )

    try:
        if source == "offline":
            return PseudoLabelDataset(
                images_dir,
                pseudolabel_path,
                transforms=transforms,
                allow_empty=allow_empty,
                teacher_soft_dir=teacher_soft_dir,
                debug_cfg=cfg.get("loss", {}).get("distill_soft", {}).get("debug", {}),
                orig_to_train_class_id=_parse_class_id_map(cfg.get("student", {}).get("orig_to_train_class_id")),
            )
        if source == "online":
            return RawImageDataset(images_dir, transforms=transforms, allow_empty=allow_empty)
        raise ValueError(f"Unknown data source: {source}")
    except Exception as exc:
        if train_cfg.get("fallback_to_dummy", False):
            logger.warning("dataset build failed (%s). Falling back to dummy data.", exc)
            dummy_cfg = train_cfg.get("dummy", {})
            return DummyDataset(
                num_images=int(dummy_cfg.get("num_images", 8)),
                image_size=tuple(dummy_cfg.get("image_size", [256, 256])),
                num_instances=int(dummy_cfg.get("num_instances", 1)),
            )
        raise


def _resolve_teacher_soft_dir(cfg: Dict[str, Any], data_root: Path) -> Optional[Path]:
    loss_cfg = cfg.get("loss", {}).get("distill_soft", {})
    if not loss_cfg.get("enabled", False):
        return None
    path = str(loss_cfg.get("teacher_soft_dir_train", "")).strip()
    if not path:
        return None
    return _resolve_path(data_root, path)


def _validate_teacher_soft_augmentations(cfg: Dict[str, Any]) -> None:
    train_cfg = cfg.get("data", {}).get("train", {})
    student_params = cfg.get("student", {}).get("params", {})
    for key in ("mosaic", "mixup"):
        train_flag = train_cfg.get(key, 0)
        student_flag = student_params.get(key, 0)
        if bool(train_flag) or bool(student_flag):
            raise NotImplementedError(f"teacher_soft alignment does not support {key} augmentation.")


def build_train_dataloader(cfg: Dict[str, Any], data_root: Path) -> DataLoader:
    """ DataLoader """

    dataset = build_train_dataset(cfg, data_root)
    data_cfg = cfg.get("data", {})
    train_cfg = data_cfg.get("train", {})
    return _build_dataloader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=bool(train_cfg.get("shuffle", True)),
        num_workers=int(train_cfg.get("num_workers", 0)),
    )


def build_teacher_dataset(cfg: Dict[str, Any], data_root: Path) -> Dataset:
    """"""

    data_cfg = cfg.get("data", {})
    train_cfg = data_cfg.get("train", {})
    teacher_cfg = data_cfg.get("teacher", {})

    source = str(teacher_cfg.get("source", "images"))
    root_dir = teacher_cfg.get("root_dir", train_cfg.get("root_dir", ""))
    teacher_root = _resolve_path(data_root, str(root_dir)) if root_dir else data_root
    images_dir = _resolve_path(teacher_root, str(teacher_cfg.get("images_dir", train_cfg.get("images_dir", "images"))))
    allow_empty = bool(teacher_cfg.get("allow_empty", False))

    if source == "coco":
        gt_json = teacher_cfg.get("gt_json", "")
        if not gt_json:
            raise ValueError("data.teacher.gt_json  COCO ")
        ann_path = _resolve_path(teacher_root, str(gt_json))
        return CocoImageMetaDataset(images_dir, ann_path, allow_empty=allow_empty)
    if source in {"images", "dir", "folder"}:
        image_paths = _collect_image_paths(images_dir)
        return ImagePathDataset(image_paths, allow_empty=allow_empty)
    raise ValueError(f" data.teacher.source: {source}")


def build_teacher_dataloader(cfg: Dict[str, Any], data_root: Path) -> DataLoader:
    """ DataLoader/"""

    dataset = build_teacher_dataset(cfg, data_root)
    data_cfg = cfg.get("data", {})
    train_cfg = data_cfg.get("train", {})
    teacher_cfg = data_cfg.get("teacher", {})
    batch_size = int(teacher_cfg.get("batch_size", train_cfg.get("batch_size", 1)))
    shuffle = bool(teacher_cfg.get("shuffle", False))
    num_workers = int(teacher_cfg.get("num_workers", 0))
    return _build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_teacher_collate_fn,
    )


def build_eval_dataset(cfg: Dict[str, Any], data_root: Path) -> Optional[Dataset]:
    """"""

    logger = get_logger("distill")
    prepare_class_mappings(cfg, data_root)
    data_cfg = cfg.get("data", {})
    eval_cfg = data_cfg.get("eval", {})
    if not eval_cfg.get("enabled", False):
        return None

    gt_json = eval_cfg.get("gt_json", "")
    if not gt_json:
        logger.warning("data.eval.gt_json ")
        return None

    train_cfg = data_cfg.get("train", {})
    eval_root = _resolve_path(data_root, str(eval_cfg.get("root_dir", ""))) if eval_cfg.get("root_dir") else data_root
    images_dir = _resolve_path(eval_root, str(eval_cfg.get("images_dir", train_cfg.get("images_dir", "images"))))
    ann_path = _resolve_path(eval_root, str(gt_json))

    allow_empty = bool(eval_cfg.get("allow_empty", False))
    transforms = build_eval_transforms(cfg, override=eval_cfg.get("transforms"))

    try:
        dataset = CocoImageDataset(images_dir, ann_path, transforms=transforms, allow_empty=allow_empty)
        setattr(dataset, "gt_json", str(ann_path))
        return dataset
    except Exception as exc:
        logger.warning(": %s", exc)
        return None


def build_eval_dataloader(cfg: Dict[str, Any], data_root: Path) -> Optional[DataLoader]:
    """ DataLoader"""

    dataset = build_eval_dataset(cfg, data_root)
    eval_cfg = cfg.get("data", {}).get("eval", {})
    batch_size = int(eval_cfg.get("batch_size", 1))
    num_workers = int(eval_cfg.get("num_workers", 0))
    return _build_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
