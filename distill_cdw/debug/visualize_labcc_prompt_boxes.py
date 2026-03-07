#!/usr/bin/env python
"""Visualize Lab-CC prompt boxes on a plain white canvas."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PKG_ROOT = ROOT / "distill_cdw"
if PKG_ROOT.exists() and str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from distill.teacher.prompt_generators import LabCCBoxPromptGenerator
from distill.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Lab-CC prompt boxes on white background")
    parser.add_argument("--image", default='../data/cdw_classify/subtest/test/Co9_20260112214151959.jpg', help="Input image path")
    parser.add_argument("--out", default="outputs/labcc_prompt_boxes_white.png", help="Output image path")
    parser.add_argument("--config", default="", help="Optional config YAML to read lab_cc params")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--line-width", type=int, default=3, help="Rectangle line width")
    parser.add_argument("--l-thresh-min", type=int, default=18)
    parser.add_argument("--a-thresh", type=int, default=131)
    parser.add_argument("--b-thresh", type=int, default=133)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--open-kernel", type=int, default=3)
    parser.add_argument("--min-cc-area", type=int, default=1500)
    parser.add_argument("--min-box-w", type=int, default=100)
    parser.add_argument("--min-box-h", type=int, default=100)
    parser.add_argument("--force-square", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument(
        "--high-overlap-filter",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=False,
    )
    parser.add_argument("--high-overlap-thresh", type=float, default=0.9)
    parser.add_argument("--nms-thresh", type=float, default=0.7)
    parser.add_argument("--max-prompts-per-image", type=int, default=200)
    return parser.parse_args()


def build_labcc_cfg(args: argparse.Namespace) -> Dict[str, Any]:
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
    return lab_cfg


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    image_path = Path(args.image)
    image_np = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image_np.shape[:2]

    generator = LabCCBoxPromptGenerator(build_labcc_cfg(args))
    boxes = generator.generate_boxes(
        image_np,
        image_meta={"file_name": image_path.name, "height": height, "width": width},
    )

    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    line_width = max(1, int(args.line_width))
    for box in boxes.tolist():
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle((x0, y0, x1, y1), outline=(0, 0, 0), width=line_width)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    print(f"image={image_path}")
    print(f"num_boxes={int(boxes.shape[0])}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
