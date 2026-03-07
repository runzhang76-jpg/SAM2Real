#!/usr/bin/env python
"""Visualize teacher inference results exported as COCO results JSON."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PKG_ROOT = ROOT / "distill_cdw"
if PKG_ROOT.exists() and str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from distill.utils.visualize import overlay_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize teacher inference results")
    parser.add_argument(
        "--pred",
        default="../data/cdw_classify/subtest/pseudolabels_results.json",
        help="Teacher results JSON path (COCO results list)",
    )
    parser.add_argument(
        "--meta",
        default="../data/cdw_classify/subtest/pseudolabels.json",
        help="JSON containing images[] (and optional categories[])",
    )
    parser.add_argument(
        "--images-root",
        default="../data/cdw_classify/subtest/test",
        help="Root for relative image file_name in meta JSON",
    )
    parser.add_argument("--image-id", type=int, default=2, help="Target image id; -1 means random")
    parser.add_argument("--score-thr", type=float, default=0.0, help="Minimum score to visualize")
    parser.add_argument("--topk", type=int, default=150, help="Keep top-k instances by score")
    parser.add_argument("--alpha", type=float, default=0.7, help="Mask overlay alpha")
    parser.add_argument("--font-size", type=int, default=30, help="Label font size")
    parser.add_argument("--seed", type=int, default=41312, help="Random seed")
    parser.add_argument(
        "--out",
        default="outputs/teacher_results_overlay.png",
        help="Output image path",
    )
    return parser.parse_args()


def _load_results(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            return payload["results"]
        if isinstance(payload.get("annotations"), list):
            return payload["annotations"]
    raise ValueError(f"Unsupported results format: {path}")


def _load_meta(path: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    if not isinstance(images, list) or len(images) == 0:
        raise ValueError(f"meta JSON has no images[]: {path}")
    img_by_id: Dict[int, Dict[str, Any]] = {}
    for img in images:
        img_id = int(img.get("id"))
        img_by_id[img_id] = img

    cat_name_by_id: Dict[int, str] = {}
    categories = payload.get("categories", [])
    if isinstance(categories, list):
        for cat in categories:
            try:
                cid = int(cat.get("id"))
                raw_name = cat.get("name", None)
                if raw_name is None or str(raw_name).strip() == "" or str(raw_name).lower() == "none":
                    cat_name_by_id[cid] = f"class_{cid}"
                else:
                    cat_name_by_id[cid] = str(raw_name)
            except Exception:
                continue
    return img_by_id, cat_name_by_id


def _resolve_image_path(file_name: str, images_root: str) -> Path:
    p = Path(file_name)
    if p.is_absolute():
        return p
    if images_root:
        return Path(images_root) / p
    return p


def _decode_segmentation(seg: Any, height: int, width: int) -> Optional[np.ndarray]:
    if seg is None:
        return None
    if isinstance(seg, dict) and "counts" in seg and "size" in seg:
        decoded = mask_utils.decode(seg)
        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]
        return decoded.astype(bool)
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, height, width)
        return mask_utils.decode(rles).any(axis=2)
    return None


def _pick_image_id(results: List[Dict[str, Any]], image_id: int, seed: int) -> int:
    if image_id >= 0:
        return image_id
    candidates = sorted({int(r.get("image_id", -1)) for r in results if int(r.get("image_id", -1)) >= 0})
    if not candidates:
        raise ValueError("No valid image_id found in results")
    rng = random.Random(seed)
    return rng.choice(candidates)


def _color_for_category(category_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(category_id + 12345)
    color = rng.integers(0, 255, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_label(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    color: Tuple[int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    x0 = int(max(0, x))
    y0 = int(max(0, y - (int(font.size) + 8 if hasattr(font, "size") else 20)))
    left, top, right, bottom = draw.textbbox((x0 + 4, y0 + 2), text, font=font)
    draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=color)
    draw.text((x0 + 4, y0 + 2), text, fill=(255, 255, 255), font=font)


def main() -> None:
    args = parse_args()
    results = _load_results(Path(args.pred))
    img_by_id, cat_name_by_id = _load_meta(Path(args.meta))

    target_image_id = _pick_image_id(results, args.image_id, args.seed)
    if target_image_id not in img_by_id:
        raise ValueError(f"image_id={target_image_id} not found in meta images[]")

    img_info = img_by_id[target_image_id]
    file_name = str(img_info.get("file_name", ""))
    img_path = _resolve_image_path(file_name, args.images_root)
    image = np.asarray(Image.open(img_path).convert("RGB"))
    height = int(img_info.get("height", image.shape[0]))
    width = int(img_info.get("width", image.shape[1]))

    selected = [
        r for r in results
        if int(r.get("image_id", -1)) == target_image_id and float(r.get("score", 0.0)) >= args.score_thr
    ]
    selected.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    if args.topk > 0:
        selected = selected[: args.topk]

    masks: List[np.ndarray] = []
    colors: List[Tuple[int, int, int]] = []
    for r in selected:
        category_id = int(r.get("category_id", -1))
        mask = _decode_segmentation(r.get("segmentation"), height, width)
        if mask is None:
            continue
        masks.append(mask)
        colors.append(_color_for_category(category_id))

    overlaid = overlay_masks(image, masks=masks, alpha=args.alpha, colors=colors)
    canvas = Image.fromarray(overlaid)
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(8, int(args.font_size)))

    for r in selected:
        bbox = r.get("bbox", None)
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x, y, w, h = [float(v) for v in bbox]
        category_id = int(r.get("category_id", -1))
        score = float(r.get("score", 0.0))
        cname = cat_name_by_id.get(category_id, f"class_{category_id}")
        if cname is None or str(cname).strip() == "" or str(cname).lower() == "none":
            cname = f"class_{category_id}"
        color = _color_for_category(category_id)
        draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
        _draw_label(draw, x, y, f"{cname} {score:.3f}", color, font)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(
        f"saved: {out_path} | image_id={target_image_id} | file={file_name} | instances={len(selected)}"
    )


if __name__ == "__main__":
    main()
