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
        default=r"D:/zhangrun/project/PointCould4AggGradation\dataset/18-railway-ballast-particles\data\photos_of_ballast_particles\particle_01/PRE_annotations/annotations.json",
        help="Pseudo label JSON path",
    )
    parser.add_argument(
        "--out",
        default="outputs/gt_overlay.png",
        help="Output image path for a single image",
    )
    parser.add_argument("--all", action="store_true", help="Visualize every image in the pseudo-label JSON")
    parser.add_argument(
        "--out-dir",
        default=r"outputs/railway_ballast/visualize_pseudolabels",
        help="Output directory used with --all; preserves image-relative paths",
    )
    parser.add_argument("--images-root", 
                        default=r"D:/zhangrun/project/PointCould4AggGradation\dataset/18-railway-ballast-particles\data\photos_of_ballast_particles\particle_01/images", 
                        help="Root for relative file_name")
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


def _normalized_output_path(path: Path) -> Path:
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        return path.with_suffix(".jpg")
    return path


def _visualize_image(
    img_info: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    category_names: Dict[int, str],
    images_root: str,
    out_path: Path,
    draw_boxes: bool,
) -> tuple[int, int, Path]:
    image_id = int(img_info["id"])
    file_name = str(img_info.get("file_name", ""))
    image_path = _resolve_image_path(file_name, images_root)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    height = int(img_info.get("height", image.shape[0]))
    width = int(img_info.get("width", image.shape[1]))

    masks = []
    selected_annotations: List[Dict[str, Any]] = []
    for ann in annotations:
        if int(ann.get("image_id", -1)) != image_id:
            continue
        selected_annotations.append(ann)
        seg = ann.get("segmentation")
        if seg is not None:
            masks.append(_decode_segmentation(seg, height, width))

    overlaid = overlay_masks(image, masks, alpha=0.9)
    if draw_boxes:
        overlaid = _draw_boxes(overlaid, selected_annotations, category_names)
    out_path = _normalized_output_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlaid).save(out_path)
    return len(masks), len(selected_annotations), out_path


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

    if args.all:
        out_dir = Path(args.out_dir)
        for index, img_info in enumerate(images, start=1):
            file_name = str(img_info.get("file_name", ""))
            relative_path = Path(file_name) if file_name else Path(f"image_{img_info['id']}.png")
            masks, anns, out_path = _visualize_image(
                img_info, annotations, category_names, args.images_root, out_dir / relative_path, args.draw_boxes
            )
            print(f"[{index}/{len(images)}] saved: {out_path} | masks={masks} | anns={anns}")
        print(f"visualized {len(images)} images to: {out_dir}")
        return

    img_info = _pick_image(images, args.image_id)
    masks, anns, out_path = _visualize_image(
        img_info, annotations, category_names, args.images_root, Path(args.out), args.draw_boxes
    )
    print(
        f"saved: {out_path} | image_id={img_info['id']} | file={img_info.get('file_name', '')} | "
        f"masks={masks} | anns={anns} | draw_boxes={args.draw_boxes}"
    )
    return

    # PIL 保存图片时必须具有可识别的扩展名


if __name__ == "__main__":
    main()
