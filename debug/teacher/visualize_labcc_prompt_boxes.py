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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matmatch2real.teacher.prompt_generators import build_prompt_generator
from matmatch2real.config.loader import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Lab-CC prompt boxes on white background")
    parser.add_argument("--image", default=r'../data\cdw_classify\dataset_seg\images\test/Co9_20260112194439586.jpg', help="Input image path")
    parser.add_argument("--out", default="outputs/labcc_prompt_boxes_white.png", help="Output image path")
    parser.add_argument("--config", default="configs/teacher/teacher_default.yaml", help="Config YAML to read prompt-generator settings")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--line-width", type=int, default=3, help="Rectangle line width")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    image_path = Path(args.image)
    image_np = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image_np.shape[:2]

    cfg = load_config(args.config)
    prompt_cfg = cfg.get("teacher", {}).get("prompt_generator", {})
    if str(prompt_cfg.get("type", "")).lower() != "lab_cc_boxes":
        raise ValueError("visualize_labcc_prompt_boxes.py requires teacher.prompt_generator.type=lab_cc_boxes")
    generator = build_prompt_generator(prompt_cfg)
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
