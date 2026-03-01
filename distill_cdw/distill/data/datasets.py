"""原始图像与伪标签数据集。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json

from distill.core.structures import PseudoLabelInstance
from distill.data.pseudolabel_io import read_pseudolabels
from distill.data.transforms import build_train_transforms, build_eval_transforms
from distill.utils.logging import get_logger

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - torch 可选
    torch = None
    DataLoader = object  # type: ignore
    Dataset = object  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL 可选
    Image = None  # type: ignore


def _load_image(path: Path) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required for dataset loading.")
    if Image is None:
        # PIL 不可用时回退为全零图像。
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


class RawImageDataset(Dataset):
    """不含伪标签的原始图像数据集。"""

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
    """仅返回图像路径与元信息，不加载图像本体。"""

    def __init__(self, image_paths: List[Path], allow_empty: bool = False) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for dataset loading.")
        self.image_paths = image_paths
        if not self.image_paths and not allow_empty:
            raise FileNotFoundError("教师数据集未找到图像。")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.image_paths[index]
        meta = {"path": str(path), "height": 0, "width": 0, "image_id": index}
        return {"image_id": index, "meta": meta}


class PseudoLabelDataset(Dataset):
    """包含离线伪标签的数据集。"""

    def __init__(
        self,
        images_dir: Path,
        pseudolabel_path: Path,
        transforms: Optional[Any] = None,
        allow_empty: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for dataset loading.")
        self.images_dir = images_dir
        self.transforms = transforms
        self.images, self.instances_by_image, _meta = read_pseudolabels(str(pseudolabel_path))
        if not self.images and not allow_empty:
            raise RuntimeError("Pseudo label file is empty.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_info = self.images[index]
        image_id, file_name, height, width = _parse_image_info(image_info, index)
        image_path = self.images_dir / file_name if file_name else None
        image = _load_image_or_empty(image_path, height, width)
        instances = self.instances_by_image.get(image_id, [])
        meta = {"path": str(image_path) if image_path else "", "height": image.shape[-2], "width": image.shape[-1]}
        return _make_sample(image, image_id, instances, meta, self.transforms)


class DummyDataset(Dataset):
    """用于快速自检的合成数据集。"""

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
    """仅用于评估的 COCO 图像数据集。"""

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
    """仅返回 COCO 图像路径与元信息，不加载图像本体。"""

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
    return {
        "images": images,
        "image_ids": [b["image_id"] for b in batch],
        "instances": [b.get("instances", []) for b in batch],
        "meta": [b.get("meta", {}) for b in batch],
    }


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
    """基于配置构建训练数据集的工厂函数。"""

    logger = get_logger("distill")
    data_cfg = cfg.get("data", {})
    train_cfg = data_cfg.get("train", {})
    transforms = build_train_transforms(cfg)

    source = train_cfg.get("source", "offline")
    images_dir = _resolve_path(data_root, str(train_cfg.get("images_dir", "images")))
    pseudolabel_path = _resolve_path(data_root, str(train_cfg.get("pseudolabel_path", "pseudolabels.json")))
    allow_empty = bool(train_cfg.get("allow_empty", False))

    if source == "dummy":
        dummy_cfg = train_cfg.get("dummy", {})
        return DummyDataset(
            num_images=int(dummy_cfg.get("num_images", 8)),
            image_size=tuple(dummy_cfg.get("image_size", [256, 256])),
            num_instances=int(dummy_cfg.get("num_instances", 1)),
        )

    try:
        if source == "offline":
            return PseudoLabelDataset(images_dir, pseudolabel_path, transforms=transforms, allow_empty=allow_empty)
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


def build_train_dataloader(cfg: Dict[str, Any], data_root: Path) -> DataLoader:
    """构建训练 DataLoader 的工厂函数。"""

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
    """构建教师端使用的轻量数据集（仅返回路径与元信息）。"""

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
            raise ValueError("data.teacher.gt_json 未配置，无法读取 COCO 图像列表。")
        ann_path = _resolve_path(teacher_root, str(gt_json))
        return CocoImageMetaDataset(images_dir, ann_path, allow_empty=allow_empty)
    if source in {"images", "dir", "folder"}:
        image_paths = _collect_image_paths(images_dir)
        return ImagePathDataset(image_paths, allow_empty=allow_empty)
    raise ValueError(f"不支持的 data.teacher.source: {source}")


def build_teacher_dataloader(cfg: Dict[str, Any], data_root: Path) -> DataLoader:
    """构建教师端 DataLoader，避免重复读取/变换图像。"""

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
    """基于配置构建评估数据集。"""

    logger = get_logger("distill")
    data_cfg = cfg.get("data", {})
    eval_cfg = data_cfg.get("eval", {})
    if not eval_cfg.get("enabled", False):
        return None

    gt_json = eval_cfg.get("gt_json", "")
    if not gt_json:
        logger.warning("data.eval.gt_json 未配置，跳过评估数据集构建。")
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
        logger.warning("评估数据集构建失败: %s", exc)
        return None


def build_eval_dataloader(cfg: Dict[str, Any], data_root: Path) -> Optional[DataLoader]:
    """构建评估 DataLoader。"""

    dataset = build_eval_dataset(cfg, data_root)
    eval_cfg = cfg.get("data", {}).get("eval", {})
    batch_size = int(eval_cfg.get("batch_size", 1))
    num_workers = int(eval_cfg.get("num_workers", 0))
    return _build_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
