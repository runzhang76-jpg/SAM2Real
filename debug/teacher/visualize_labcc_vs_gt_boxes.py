#!/usr/bin/env python
"""Visualize GT boxes vs Lab-CC prompt boxes on one image."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sam2real.teacher.prompt_generators import LabCCBoxPromptGenerator
from sam2real.config.loader import load_config


def _xywh_to_xyxy(b: List[float]) -> List[float]:
    x, y, w, h = [float(v) for v in b]
    return [x, y, x + w, y + h]


def _box_iou_xyxy(a: List[float], b: List[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _greedy_match_stats(gt_boxes: List[List[float]], lab_boxes: List[List[float]], iou_thr: float) -> Dict[str, float]:
    used_gt = set()
    matched = 0
    for lb in lab_boxes:
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gt_boxes):
            if j in used_gt:
                continue
            iou = _box_iou_xyxy(lb, gb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            used_gt.add(best_j)
            matched += 1

    gt_n = len(gt_boxes)
    lab_n = len(lab_boxes)
    precision = matched / lab_n if lab_n > 0 else 0.0
    recall = matched / gt_n if gt_n > 0 else 0.0
    return {
        "matched": float(matched),
        "precision": float(precision),
        "recall": float(recall),
        "gt_count": float(gt_n),
        "lab_count": float(lab_n),
    }


def _draw_boxes(image: np.ndarray, boxes: List[List[float]], color: Tuple[int, int, int], title: str) -> Image.Image:
    canvas = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    for b in boxes:
        x0, y0, x1, y1 = [float(v) for v in b]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
    draw.rectangle((8, 8, 8 + 300, 38), fill=(0, 0, 0))
    draw.text((12, 12), f"{title} (n={len(boxes)})", fill=(255, 255, 255))
    return canvas


def _stack_h(images: List[Image.Image]) -> Image.Image:
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    out = Image.new("RGB", (sum(widths), max(heights)), color=(20, 20, 20))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GT boxes vs Lab-CC prompt boxes")
    parser.add_argument("--gt-json", default="../data/cdw_classify/dataset_seg/annotations/instances_test.json")
    parser.add_argument("--images-root", default="../data/cdw_classify/dataset_seg/images/test")
    parser.add_argument("--image-id", type=int, default=146, help="Target image id; -1 means random")
    parser.add_argument("--seed", type=int, default=41312, help="Random seed")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for matching stats")
    parser.add_argument("--out", default="outputs/labcc_vs_gt_boxes.png", help="Output visualization path")
    parser.add_argument("--config", default="", help="Optional config YAML to read lab_cc params")
    parser.add_argument("--l-thresh-min", type=int, default=8)
    parser.add_argument("--a-thresh", type=int, default=100)
    parser.add_argument("--b-thresh", type=int, default=100)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--open-kernel", type=int, default=3)
    parser.add_argument("--min-cc-area", type=int, default=8000)
    parser.add_argument("--min-box-w", type=int, default=100)
    parser.add_argument("--min-box-h", type=int, default=100)
    parser.add_argument("--force-square", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument(
        "--high-overlap-filter",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=False,
        help="If true, remove highly-overlapped boxes and keep larger ones",
    )
    parser.add_argument("--high-overlap-thresh", type=float, default=0.9)
    parser.add_argument("--nms-thresh", type=float, default=0.7)
    parser.add_argument("--max-prompts-per-image", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    gt_path = Path(args.gt_json)
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    images = gt.get("images", [])
    anns = gt.get("annotations", [])
    if not images:
        raise ValueError(f"No images in {gt_path}")

    img_info = None
    if args.image_id >= 0:
        for x in images:
            if int(x.get("id", -1)) == args.image_id:
                img_info = x
                break
        if img_info is None:
            raise ValueError(f"image_id={args.image_id} not found")
    else:
        img_info = random.choice(images)

    image_id = int(img_info["id"])
    file_name = str(img_info.get("file_name", ""))
    image_path = Path(args.images_root) / file_name
    image_np = np.asarray(Image.open(image_path).convert("RGB"))

    gt_boxes = [_xywh_to_xyxy(a["bbox"]) for a in anns if int(a.get("image_id", -1)) == image_id and "bbox" in a]

    lab_cfg: Dict[str, Any] = {
        "l_thresh_min": args.l_thresh_min,
        "a_thresh": args.a_thresh,
        "b_thresh": args.b_thresh,
        "close_kernel": args.close_kernel,
        "open_kernel": args.open_kernel,
        "min_cc_area": args.min_cc_area,
        "min_box_w": args.min_box_w,
        "min_box_h": args.min_box_h,
        "force_square": bool(args.force_square),
        "high_overlap_filter": bool(args.high_overlap_filter),
        "high_overlap_thresh": args.high_overlap_thresh,
        "nms_thresh": args.nms_thresh,
        "max_prompts_per_image": args.max_prompts_per_image,
        "save_debug": False,
    }
    if args.config:
        cfg = load_config(args.config)
        lab_cfg.update(cfg.get("teacher", {}).get("prompt_generator", {}).get("lab_cc_boxes", {}))

    generator = LabCCBoxPromptGenerator(lab_cfg)
    lab_boxes_np = generator.generate_boxes(image_np, image_meta=img_info)
    lab_boxes = lab_boxes_np.tolist()

    stats = _greedy_match_stats(gt_boxes, lab_boxes, iou_thr=args.iou_thr)
    print(
        f"image_id={image_id}, file={file_name}, "
        f"gt={int(stats['gt_count'])}, lab_cc={int(stats['lab_count'])}, "
        f"matched@{args.iou_thr:.2f}={int(stats['matched'])}, "
        f"precision={stats['precision']:.3f}, recall={stats['recall']:.3f}"
    )

    gt_view = _draw_boxes(image_np, gt_boxes, color=(80, 255, 80), title="GT boxes")
    lab_view = _draw_boxes(image_np, lab_boxes, color=(80, 160, 255), title="Lab-CC boxes")

    overlay = Image.fromarray(image_np.copy())
    draw = ImageDraw.Draw(overlay)
    for b in gt_boxes:
        draw.rectangle(tuple(b), outline=(80, 255, 80), width=3)
    for b in lab_boxes:
        draw.rectangle(tuple(b), outline=(80, 160, 255), width=2)
    draw.rectangle((8, 8, 8 + 520, 44), fill=(0, 0, 0))
    draw.text(
        (12, 12),
        f"Overlay | m@{args.iou_thr:.2f}={int(stats['matched'])} P={stats['precision']:.3f} R={stats['recall']:.3f}",
        fill=(255, 255, 255),
    )

    merged = _stack_h([gt_view, lab_view, overlay])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
