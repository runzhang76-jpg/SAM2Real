#!/usr/bin/env python
"""Build class prototype database (.pth) from COCO-style annotations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PKG_ROOT = ROOT / "distill_cdw"
if PKG_ROOT.exists() and str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from distill.teacher.dinov3_knn_classifier import load_model
from distill.utils.config import load_config
from distill.utils.logging import setup_logger

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    from pycocotools import mask as mask_utils  # type: ignore
except Exception:  # pragma: no cover
    mask_utils = None  # type: ignore


def _str2bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return (x / denom).astype(np.float32, copy=False)


def _square_crop_coords(
    img_hw: Tuple[int, int],
    bbox_xyxy: Tuple[float, float, float, float],
    margin_ratio: float,
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


def _decode_segmentation(seg: Any, height: int, width: int) -> Optional[np.ndarray]:
    if seg is None or mask_utils is None:
        return None
    if isinstance(seg, dict) and "counts" in seg and "size" in seg:
        m = mask_utils.decode(seg)
        if m.ndim == 3:
            m = m[:, :, 0]
        return np.asarray(m, dtype=bool)
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, height, width)
        m = mask_utils.decode(rles)
        if m.ndim == 3:
            m = m.any(axis=2)
        return np.asarray(m, dtype=bool)
    return None


def _crop_patch(
    image: np.ndarray,
    bbox_xywh: List[float],
    mask: Optional[np.ndarray],
    margin_ratio: float,
) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox_xywh]
    x1n, y1n, x2n, y2n = _square_crop_coords(
        img_hw=(image.shape[0], image.shape[1]),
        bbox_xyxy=(x, y, x + w, y + h),
        margin_ratio=margin_ratio,
    )
    patch = image[y1n:y2n, x1n:x2n].copy()
    if patch.size == 0:
        return patch
    if mask is not None:
        m = mask[y1n:y2n, x1n:x2n]
        if m.dtype != bool:
            m = m.astype(bool)
        if m.size == 0:
            return np.zeros((0, 0, image.shape[2]), dtype=image.dtype)
        patch[~m] = 0
    return patch


def _resolve_dino_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cls_cfg = cfg.get("teacher", {}).get("classifier", {})
    proto_cfg = dict(cls_cfg)
    proto_cfg.update(cls_cfg.get("dinov3_prototype", {}))
    if not proto_cfg.get("repo_dir"):
        knn_cfg = cls_cfg.get("dinov3_knn", {})
        proto_cfg.update({k: v for k, v in knn_cfg.items() if k not in proto_cfg or not proto_cfg.get(k)})
    return proto_cfg


def _preprocess_patches(
    patches: List[np.ndarray],
    *,
    input_size: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> "torch.Tensor":
    if torch is None or Image is None:
        raise RuntimeError("PyTorch and PIL are required for DINOv3 feature extraction")
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    tensors = []
    for patch in patches:
        img = Image.fromarray(patch).resize((input_size, input_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32, copy=True) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        t = (t - mean_t) / std_t
        tensors.append(t)
    return torch.stack(tensors, dim=0)


def _extract_features(
    model: "torch.nn.Module",
    patches: List[np.ndarray],
    *,
    device: str,
    input_size: int,
    batch_size: int,
    feature_dim: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for feature extraction")
    if len(patches) == 0:
        return np.zeros((0, feature_dim), dtype=np.float32)
    feats: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            x = _preprocess_patches(
                patches[i : i + batch_size],
                input_size=input_size,
                mean=mean,
                std=std,
            ).to(device)
            f = model.forward_features(x)
            if isinstance(f, dict):
                if "x_norm_clstoken" in f:
                    f = f["x_norm_clstoken"]
                elif "x_norm_cls_token" in f:
                    f = f["x_norm_cls_token"]
                else:
                    raise KeyError("forward_features output dict missing x_norm_clstoken")
            if not isinstance(f, torch.Tensor):
                raise TypeError("forward_features must return Tensor or dict of Tensor")
            if f.ndim == 3:
                f = f[:, 0, :]
            feats.append(f.detach().cpu().to(torch.float32).numpy())
    out = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    if out.shape[1] != feature_dim:
        raise ValueError(f"DINO feature dim mismatch: model={out.shape[1]} config={feature_dim}")
    return _l2_normalize(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build class prototype database from GT annotations")
    parser.add_argument("--config", default="distill_cdw/configs/distill_default.yaml")
    parser.add_argument("--gt-json", default='../data/cdw_classify/dataset_seg/annotations/instances_test.json', help="COCO annotation json path")
    parser.add_argument("--images-root", default='../data/cdw_classify/dataset_seg/images/test', help="Images root directory")
    parser.add_argument("--use-mask", type=_str2bool, default=True, help="Mask-out background using GT segmentation")
    parser.add_argument("--margin-ratio", type=float, default=0, help="Crop margin; <0 uses config classifier.margin_ratio")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of GT instances")
    parser.add_argument("--min-per-class", type=int, default=1, help="Minimum samples required to export a class prototype")
    parser.add_argument("--output", default="distill_cdw/distill/teacher/class_prototypes_UseMask.pth", help="Output prototype database .pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("distill")

    if torch is None:
        raise RuntimeError("PyTorch is required")
    if Image is None:
        raise RuntimeError("PIL is required")

    cfg = load_config(args.config)
    dino_cfg = _resolve_dino_cfg(cfg)
    if not dino_cfg.get("repo_dir"):
        raise ValueError("DINO repo_dir is missing in config")
    if not dino_cfg.get("weights"):
        raise ValueError("DINO weights are missing in config")

    device = str(dino_cfg.get("device", "cuda")).strip()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = str(dino_cfg.get("model_name", "dinov3_vitb16")).strip()
    model_source = str(dino_cfg.get("model_source", "local")).strip() or "local"
    input_size = int(dino_cfg.get("input_size", 224))
    batch_size = int(dino_cfg.get("batch_size", 32))
    feature_dim = int(dino_cfg.get("feature_dim", 768))
    mean = tuple(dino_cfg.get("normalize_mean", dino_cfg.get("mean", (0.485, 0.456, 0.406))))
    std = tuple(dino_cfg.get("normalize_std", dino_cfg.get("std", (0.229, 0.224, 0.225))))

    cls_cfg = cfg.get("teacher", {}).get("classifier", {})
    margin_ratio = float(cls_cfg.get("margin_ratio", 0.1)) if args.margin_ratio < 0 else float(args.margin_ratio)

    logger.info("loading DINOv3 model: model=%s device=%s", model_name, device)
    model = load_model(
        repo_dir=str(dino_cfg["repo_dir"]),
        weights=str(dino_cfg["weights"]),
        model_name=model_name,
        device=device,
        source=model_source,
    )

    gt = json.loads(Path(args.gt_json).read_text(encoding="utf-8"))
    images = gt.get("images", [])
    anns = gt.get("annotations", [])
    categories = gt.get("categories", [])
    img_by_id = {int(x["id"]): x for x in images}
    cat_name_by_id = {int(c["id"]): str(c.get("name", c["id"])) for c in categories}

    if args.limit > 0:
        anns = anns[: args.limit]

    image_cache: Dict[int, np.ndarray] = {}
    patches: List[np.ndarray] = []
    patch_labels: List[int] = []
    patch_meta: List[Dict[str, Any]] = []
    skipped = 0

    images_root = Path(args.images_root)
    for ann in anns:
        image_id = int(ann.get("image_id", -1))
        info = img_by_id.get(image_id)
        if info is None:
            skipped += 1
            continue

        if image_id not in image_cache:
            file_name = str(info.get("file_name", ""))
            image_path = images_root / file_name
            if not image_path.exists():
                logger.warning("image not found: %s", image_path)
                skipped += 1
                continue
            image_cache[image_id] = np.array(Image.open(image_path).convert("RGB"), copy=True)
        image_np = image_cache[image_id]

        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            skipped += 1
            continue

        class_id = int(ann.get("category_id", -1))
        mask = None
        if args.use_mask:
            mask = _decode_segmentation(
                ann.get("segmentation"),
                height=int(info.get("height", image_np.shape[0])),
                width=int(info.get("width", image_np.shape[1])),
            )
        patch = _crop_patch(image=image_np, bbox_xywh=bbox, mask=mask, margin_ratio=margin_ratio)
        if patch.size == 0:
            skipped += 1
            continue

        patches.append(patch)
        patch_labels.append(class_id)
        patch_meta.append(
            {
                "ann_id": int(ann.get("id", -1)),
                "image_id": image_id,
                "file_name": str(info.get("file_name", "")),
                "category_id": class_id,
                "category_name": cat_name_by_id.get(class_id, str(class_id)),
                "bbox": bbox,
            }
        )

    logger.info("collected patches: total=%d skipped=%d", len(patches), skipped)
    if not patches:
        raise RuntimeError("No valid GT patches collected")

    features = _extract_features(
        model,
        patches,
        device=device,
        input_size=input_size,
        batch_size=batch_size,
        feature_dim=feature_dim,
        mean=mean,  # type: ignore[arg-type]
        std=std,  # type: ignore[arg-type]
    )

    features_by_class: Dict[int, List[np.ndarray]] = defaultdict(list)
    for feat, class_id in zip(features, patch_labels):
        features_by_class[int(class_id)].append(feat)

    labels: List[int] = []
    counts: List[int] = []
    prototypes: List[np.ndarray] = []
    class_stats: List[Dict[str, Any]] = []

    for class_id in sorted(features_by_class.keys()):
        feat_list = features_by_class[class_id]
        count = len(feat_list)
        if count < int(args.min_per_class):
            logger.warning("skip class=%s(%d): samples=%d < min_per_class=%d", cat_name_by_id.get(class_id, str(class_id)), class_id, count, int(args.min_per_class))
            continue
        feat_mat = np.stack(feat_list, axis=0).astype(np.float32, copy=False)
        proto = feat_mat.mean(axis=0, keepdims=True)
        proto = _l2_normalize(proto)[0]
        labels.append(class_id)
        counts.append(count)
        prototypes.append(proto)
        class_stats.append(
            {
                "category_id": class_id,
                "category_name": cat_name_by_id.get(class_id, str(class_id)),
                "count": count,
            }
        )
        logger.info("class=%s(%d): samples=%d", cat_name_by_id.get(class_id, str(class_id)), class_id, count)

    if not prototypes:
        raise RuntimeError("No class prototypes were built")

    prototype_matrix = np.stack(prototypes, axis=0).astype(np.float32, copy=False)
    output = {
        "labels": torch.tensor(labels, dtype=torch.int64),
        "counts": torch.tensor(counts, dtype=torch.int64),
        "prototypes": torch.from_numpy(prototype_matrix),
        "category_names": {int(k): v for k, v in cat_name_by_id.items()},
        "class_stats": class_stats,
        "feature_dim": int(feature_dim),
        "config": {
            "model_name": model_name,
            "model_source": model_source,
            "repo_dir": str(dino_cfg["repo_dir"]),
            "weights": str(dino_cfg["weights"]),
            "device": device,
            "input_size": input_size,
            "batch_size": batch_size,
            "use_mask": bool(args.use_mask),
            "margin_ratio": margin_ratio,
            "min_per_class": int(args.min_per_class),
        },
        "source": {
            "gt_json": str(args.gt_json),
            "images_root": str(args.images_root),
            "num_instances": len(patches),
            "skipped": skipped,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    logger.info(
        "saved prototype database: %s (classes=%d, instances=%d, feature_dim=%d)",
        out_path,
        len(labels),
        len(patches),
        feature_dim,
    )


if __name__ == "__main__":
    main()
