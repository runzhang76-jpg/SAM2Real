#!/usr/bin/env python
"""随机选择一张图像并叠加伪标签。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pycocotools import mask as mask_utils  # type: ignore
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PKG_ROOT = ROOT / "distill_cdw"
if PKG_ROOT.exists() and str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from distill.utils.visualize import overlay_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize pseudolabels on a random image")
    parser.add_argument(
        "--pred",
        default="../data/sam2/dataset/segment_cdw_coco_dataset/pseudolabels/pseudolabels.json",
        help="Pseudo label JSON path",
    )
    parser.add_argument(
        "--out",
        default="outputs/pseudolabel_overlay.png",
        help="Output image path",
    )
    parser.add_argument("--images-root", default="", help="Root for relative file_name")
    parser.add_argument("--seed", type=int, default=41312, help="Random seed")
    return parser.parse_args()


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
    raise ValueError("未知 segmentation 格式")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    payload = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    images: List[Dict[str, Any]] = payload.get("images", [])
    annotations: List[Dict[str, Any]] = payload.get("annotations", [])

    img_info = random.choice(images)
    image_id = int(img_info["id"])
    file_name = str(img_info.get("file_name", ""))
    image_path = _resolve_image_path(file_name, args.images_root)
    image = np.asarray(Image.open(image_path).convert("RGB"))

    masks = []
    height = int(img_info.get("height", image.shape[0]))
    width = int(img_info.get("width", image.shape[1]))
    for ann in annotations:
        if int(ann.get("image_id", -1)) != image_id:
            continue
        seg = ann.get("segmentation")
        if seg is None:
            continue
        masks.append(_decode_segmentation(seg, height, width))

    overlaid = overlay_masks(image, masks, alpha=0.9)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlaid).save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
