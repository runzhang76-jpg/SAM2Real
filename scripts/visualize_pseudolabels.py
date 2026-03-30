#!/usr/bin/env python
""""""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pycocotools import mask as mask_utils  # type: ignore
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matmatch2real.utils.visualize import overlay_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize pseudolabels on a selected image")
    parser.add_argument(
        "--pred",
        default="../data/cdw_classify/dataset_seg/annotations/instances_test.json",
        help="Pseudo label JSON path",
    )
    parser.add_argument(
        "--out",
        default="outputs/gt_overlay.png",
        help="Output image path",
    )
    parser.add_argument("--images-root", default="../data/cdw_classify/dataset_seg/images/test", help="Root for relative file_name")
    parser.add_argument("--image-id", type=int, default=-1, help="Target image id; -1 means random")
    parser.add_argument(
        "--draw-boxes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to draw bounding boxes on top of the mask overlay.",
    )
    parser.add_argument("--seed", type=int, default=41312, help="Random seed")
    return parser.parse_args()


def _pick_image(images: List[Dict[str, Any]], image_id: int) -> Dict[str, Any]:
    if image_id < 0:
        return random.choice(images)
    for image in images:
        if int(image.get("id", -1)) == image_id:
            return image
    raise ValueError(f"image_id={image_id} not found in pseudolabel images[]")


def _resolve_image_path(file_name: str, images_root: str) -> Path:
    path = Path(file_name)
    if path.is_absolute():
        return path
    if images_root:
        return Path(images_root) / path
    return path


def _decode_segmentation(seg: Any, height: int, width: int) -> np.ndarray:
    if isinstance(seg, dict) and "counts" in seg and "size" in seg:
        return mask_utils.decode(seg).astype(bool)
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, height, width)
        return mask_utils.decode(rles).any(axis=2)
    if isinstance(seg, dict) and seg.get("format") == "bitmap":
        return np.asarray(seg["mask"], dtype=np.uint8).astype(bool)
    raise ValueError(" segmentation ")


def _color_for_category(category_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(category_id) + 12345)
    return tuple(int(x) for x in rng.integers(64, 256, size=3))


def _draw_boxes(
    image: np.ndarray,
    annotations: List[Dict[str, Any]],
    category_names: Dict[int, str],
) -> np.ndarray:
    canvas = Image.fromarray(image)
    drawer = ImageDraw.Draw(canvas)
    for ann in annotations:
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x, y, w, h = [float(v) for v in bbox]
        if w <= 0 or h <= 0:
            continue
        category_id = int(ann.get("category_id", -1))
        color = _color_for_category(category_id)
        drawer.rectangle((x, y, x + w, y + h), outline=color, width=3)
        label = category_names.get(category_id, f"class_{category_id}")
        drawer.text((x + 2, max(0, y - 14)), label, fill=color)
    return np.asarray(canvas)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    payload = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    images: List[Dict[str, Any]] = payload.get("images", [])
    annotations: List[Dict[str, Any]] = payload.get("annotations", [])
    category_names = {
        int(cat.get("id", -1)): str(cat.get("name", f"class_{int(cat.get('id', -1))}"))
        for cat in payload.get("categories", [])
        if isinstance(cat, dict)
    }

    if not images:
        raise ValueError("pseudolabel JSON has no images[]")

    img_info = _pick_image(images, args.image_id)
    image_id = int(img_info["id"])
    file_name = str(img_info.get("file_name", ""))
    image_path = _resolve_image_path(file_name, args.images_root)
    image = np.asarray(Image.open(image_path).convert("RGB"))

    masks = []
    selected_annotations: List[Dict[str, Any]] = []
    height = int(img_info.get("height", image.shape[0]))
    width = int(img_info.get("width", image.shape[1]))
    for ann in annotations:
        if int(ann.get("image_id", -1)) != image_id:
            continue
        selected_annotations.append(ann)
        seg = ann.get("segmentation")
        if seg is None:
            continue
        masks.append(_decode_segmentation(seg, height, width))

    overlaid = overlay_masks(image, masks, alpha=0.9)
    if args.draw_boxes:
        overlaid = _draw_boxes(overlaid, selected_annotations, category_names)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlaid).save(out_path)
    print(
        f"saved: {out_path} | image_id={image_id} | file={file_name} | "
        f"masks={len(masks)} | anns={len(selected_annotations)} | draw_boxes={args.draw_boxes}"
    )


if __name__ == "__main__":
    main()
