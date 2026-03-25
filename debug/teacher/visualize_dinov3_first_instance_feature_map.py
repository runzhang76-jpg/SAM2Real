#!/usr/bin/env python
"""Visualize the DINOv3 patch-token feature map for the first GT instance of one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sam2real.config.loader import load_config
from sam2real.teacher.classifier_adapter import _masked_square_crop
from sam2real.teacher.dinov3_classifier import DinoV3FeatureExtractor
from sam2real.utils.logging import setup_logger
from sam2real.utils.paths import resolve_project_path


def _str2bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def _decode_compressed_rle_counts(text: str) -> List[int]:
    counts: List[int] = []
    index = 0
    while index < len(text):
        value = 0
        shift = 0
        while True:
            char = ord(text[index]) - 48
            index += 1
            value |= (char & 0x1F) << shift
            if (char & 0x20) == 0:
                if char & 0x10:
                    value |= -1 << (shift + 5)
                break
            shift += 5
        if len(counts) > 2:
            value += counts[-2]
        counts.append(int(value))
    return counts


def _decode_rle(segmentation: Dict[str, Any]) -> np.ndarray:
    size = segmentation.get("size", [])
    if len(size) != 2:
        raise ValueError(f"Invalid RLE size: {size}")
    height, width = int(size[0]), int(size[1])
    counts_raw = segmentation.get("counts")
    if isinstance(counts_raw, str):
        counts = _decode_compressed_rle_counts(counts_raw)
    elif isinstance(counts_raw, list):
        counts = [int(value) for value in counts_raw]
    else:
        raise ValueError(f"Unsupported RLE counts type: {type(counts_raw).__name__}")

    flat = np.zeros(height * width, dtype=np.uint8)
    start = 0
    value = 0
    for count in counts:
        end = start + int(count)
        if value == 1:
            flat[start:end] = 1
        start = end
        value = 1 - value
    return flat.reshape((height, width), order="F")


def _decode_polygons(segmentation: List[Any], height: int, width: int) -> np.ndarray:
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue
        coords = [(float(polygon[idx]), float(polygon[idx + 1])) for idx in range(0, len(polygon), 2)]
        draw.polygon(coords, outline=1, fill=1)
    return np.asarray(image, dtype=np.uint8)


def _decode_segmentation(segmentation: Any, height: int, width: int) -> Optional[np.ndarray]:
    if segmentation is None:
        return None
    if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
        mask = _decode_rle(segmentation)
    elif isinstance(segmentation, list):
        mask = _decode_polygons(segmentation, height, width)
    else:
        return None
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return np.asarray(mask > 0, dtype=np.uint8)


def _infer_token_grid(tokens: torch.Tensor, input_size: int, patch_size: Optional[int]) -> Tuple[int, int]:
    patch_count = int(tokens.shape[1])
    side = int(round(patch_count ** 0.5))
    if side * side == patch_count:
        return side, side
    if patch_size is not None and patch_size > 0:
        side = int(input_size // patch_size)
        if side * side == patch_count:
            return side, side
    raise ValueError(f"Cannot infer token grid for patch_count={patch_count}")


def _reduce_patch_tokens(tokens: np.ndarray, reduction: str) -> np.ndarray:
    if reduction == "mean":
        return tokens.mean(axis=2)
    if reduction == "max":
        return tokens.max(axis=2)
    if reduction == "l2":
        return np.linalg.norm(tokens, axis=2)
    raise ValueError(f"Unsupported reduction={reduction}")


def _normalize_map(feature_map: np.ndarray) -> np.ndarray:
    feature_map = np.asarray(feature_map, dtype=np.float32)
    min_value = float(feature_map.min()) if feature_map.size else 0.0
    max_value = float(feature_map.max()) if feature_map.size else 1.0
    if max_value - min_value < 1e-8:
        return np.zeros_like(feature_map, dtype=np.float32)
    return (feature_map - min_value) / (max_value - min_value)


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _resize_map(feature_map: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    height, width = target_hw
    image = Image.fromarray(np.clip(feature_map * 255.0, 0, 255).astype(np.uint8), mode="L")
    image = image.resize((width, height), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _overlay_heatmap(image_np: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    color = (_jet_colormap(heatmap) * 255.0).astype(np.uint8)
    base = image_np.astype(np.float32)
    overlay = base * (1.0 - alpha) + color.astype(np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _draw_bbox(image_np: np.ndarray, bbox_xywh: List[float]) -> np.ndarray:
    image = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image)
    x, y, w, h = [float(v) for v in bbox_xywh]
    draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=3)
    return np.asarray(image, dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DINOv3 feature map of the first GT instance in one image.")
    parser.add_argument("--config", default="configs/teacher/teacher_default.yaml")
    parser.add_argument("--gt-json", default="../data/cdw_classify/dataset_seg/annotations/instances_test.json")
    parser.add_argument("--images-root", default="../data/cdw_classify/dataset_seg/images/test")
    parser.add_argument("--image-id", type=int, default=540 , help="COCO image id to inspect.")
    parser.add_argument("--instance-index", type=int, default=0, help="Which GT instance to use after sorting by ann id.")
    parser.add_argument("--use-mask", type=_str2bool, default=False, help="Use GT segmentation to suppress background.")
    parser.add_argument("--margin-ratio", type=float, default=-1.0, help="Crop margin; negative uses teacher config.")
    parser.add_argument("--background-mode", default="", help="Override crop background mode: keep/zero/white.")
    parser.add_argument("--reduction", choices=("l2", "mean", "max"), default="l2", help="Channel reduction for patch tokens.")
    parser.add_argument("--overlay-alpha", type=float, default=0.45, help="Heatmap alpha when blending over the crop.")
    parser.add_argument("--out-dir", default="outputs/dinov3_feature_maps", help="Directory for saved debug outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("distill")

    cfg = load_config(args.config)
    cls_cfg = cfg.get("teacher", {}).get("classifier", {})
    dino_cfg = dict(cls_cfg.get("dinov3", {}))
    extractor = DinoV3FeatureExtractor(dino_cfg, logger=logger)

    crop_cfg = dict(cls_cfg.get("crop", {}))
    margin_ratio = float(crop_cfg.get("margin_ratio", cls_cfg.get("margin_ratio", 0.1)))
    if args.margin_ratio >= 0:
        margin_ratio = float(args.margin_ratio)
    background_mode = str(crop_cfg.get("background_mode", "zero")).strip().lower() or "zero"
    if str(args.background_mode).strip():
        background_mode = str(args.background_mode).strip().lower()
    min_size = int(crop_cfg.get("min_size", 4))

    gt_path = Path(resolve_project_path(args.gt_json))
    images_root = Path(resolve_project_path(args.images_root))
    out_root = Path(resolve_project_path(args.out_dir))

    payload = json.loads(gt_path.read_text(encoding="utf-8"))
    image_by_id = {int(item["id"]): item for item in payload.get("images", [])}
    image_info = image_by_id.get(int(args.image_id))
    if image_info is None:
        raise KeyError(f"image_id={args.image_id} not found in {gt_path}")

    annotations = [ann for ann in payload.get("annotations", []) if int(ann.get("image_id", -1)) == int(args.image_id)]
    annotations.sort(key=lambda ann: int(ann.get("id", 0)))
    if not annotations:
        raise ValueError(f"No GT instances found for image_id={args.image_id}")
    if args.instance_index < 0 or args.instance_index >= len(annotations):
        raise IndexError(f"instance_index={args.instance_index} out of range, total_instances={len(annotations)}")

    annotation = annotations[int(args.instance_index)]
    file_name = str(image_info.get("file_name", ""))
    image_path = images_root / file_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_np = np.asarray(Image.open(image_path).convert("RGB"))
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox in annotation id={annotation.get('id')}")

    mask = None
    if args.use_mask:
        mask = _decode_segmentation(
            annotation.get("segmentation"),
            height=int(image_info.get("height", image_np.shape[0])),
            width=int(image_info.get("width", image_np.shape[1])),
        )

    x, y, w, h = [float(v) for v in bbox]
    crop = _masked_square_crop(
        image_np,
        mask,
        bbox_xyxy=(x, y, x + w, y + h),
        margin_ratio=margin_ratio,
        background_mode=background_mode,
        min_size=min_size,
    )
    if crop.size == 0:
        raise RuntimeError("The selected instance crop is empty after applying the crop settings.")

    extractor.model.eval()
    with torch.no_grad():
        inputs = extractor._preprocess_patches([crop]).to(extractor.device)
        features = extractor.model.forward_features(inputs)
        if not isinstance(features, dict) or "x_norm_patchtokens" not in features:
            raise KeyError("DINOv3 forward_features output missing x_norm_patchtokens")
        patch_tokens = features["x_norm_patchtokens"].detach().cpu()

    patch_size = int(getattr(extractor.model, "patch_size", 0) or 0) or None
    grid_h, grid_w = _infer_token_grid(patch_tokens, input_size=extractor.input_size, patch_size=patch_size)
    patch_tokens_np = patch_tokens[0].to(torch.float32).numpy().reshape(grid_h, grid_w, -1)
    feature_map = _reduce_patch_tokens(patch_tokens_np, reduction=args.reduction)
    feature_map_norm = _normalize_map(feature_map)
    heatmap_large = _resize_map(feature_map_norm, target_hw=(crop.shape[0], crop.shape[1]))

    overlay_np = _overlay_heatmap(crop, heatmap_large, alpha=float(args.overlay_alpha))
    bbox_image_np = _draw_bbox(image_np, bbox)

    category_name_by_id = {int(item["id"]): str(item.get("name", item["id"])) for item in payload.get("categories", [])}
    category_id = int(annotation.get("category_id", -1))
    ann_id = int(annotation.get("id", -1))
    sample_dir = out_root / f"image_{args.image_id}_ann_{ann_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(bbox_image_np).save(sample_dir / "image_with_bbox.png")
    Image.fromarray(crop).save(sample_dir / "instance_crop.png")
    Image.fromarray(np.clip(feature_map_norm * 255.0, 0, 255).astype(np.uint8), mode="L").save(sample_dir / "feature_map_grid.png")
    Image.fromarray(np.clip(heatmap_large * 255.0, 0, 255).astype(np.uint8), mode="L").save(sample_dir / "feature_map_resized.png")
    Image.fromarray(overlay_np).save(sample_dir / "feature_overlay.png")
    np.save(sample_dir / "patch_tokens.npy", patch_tokens_np)
    np.save(sample_dir / "feature_map.npy", feature_map.astype(np.float32))

    meta = {
        "config": str(args.config),
        "gt_json": str(gt_path),
        "images_root": str(images_root),
        "image_id": int(args.image_id),
        "instance_index": int(args.instance_index),
        "annotation_id": ann_id,
        "file_name": file_name,
        "category_id": category_id,
        "category_name": category_name_by_id.get(category_id, str(category_id)),
        "bbox_xywh": [float(v) for v in bbox],
        "use_mask": bool(args.use_mask),
        "margin_ratio": margin_ratio,
        "background_mode": background_mode,
        "crop_shape": [int(v) for v in crop.shape],
        "input_size": int(extractor.input_size),
        "patch_size": patch_size,
        "token_grid_hw": [grid_h, grid_w],
        "token_shape": [int(v) for v in patch_tokens_np.shape],
        "reduction": args.reduction,
        "output_dir": str(sample_dir),
    }
    (sample_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("saved crop to %s", sample_dir / "instance_crop.png")
    logger.info("saved feature overlay to %s", sample_dir / "feature_overlay.png")
    logger.info("saved raw patch tokens to %s", sample_dir / "patch_tokens.npy")


if __name__ == "__main__":
    main()
