#!/usr/bin/env python
"""Profile numpy vs torch postprocess timing on a single image."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any, Dict, List
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import numpy as np

from sam2real.teacher.sam2_teacher import SAM2Teacher, SegmentCDWAdapter
from sam2real.teacher.postprocess import PostProcessPipeline, convert_instances
from sam2real.config.loader import load_config
from sam2real.utils.logging import setup_logger

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile postprocess paths")
    parser.add_argument("--config", default="sam2real-main/configs/teacher/distill_default.yaml")
    parser.add_argument("--image", default="Co3_20250714172336216.jpg", help="Image path (required)")
    parser.add_argument("--device", default="auto", help="torch device for torch path")
    parser.add_argument("--repeat", type=int, default=3)
    return parser.parse_args()


def _load_image(path: Path) -> np.ndarray:
    if Image is None:
        raise RuntimeError("PIL not available")
    return np.asarray(Image.open(path).convert("RGB"))


def _prepare_post_cfg(cfg: Dict[str, Any], use_torch: bool, device: str) -> Dict[str, Any]:
    post_cfg = copy.deepcopy(cfg.get("teacher", {}).get("postprocess", {}))
    runtime = post_cfg.get("runtime", {})
    runtime["use_torch"] = bool(use_torch)
    runtime["device"] = device
    post_cfg["runtime"] = runtime
    return post_cfg


def _run(post_cfg: Dict[str, Any], raw_preds: List[Dict[str, Any]], meta: Dict[str, Any]) -> int:
    pipeline = PostProcessPipeline(post_cfg)
    processed = pipeline(list(raw_preds), meta)
    inst = convert_instances(
        processed,
        image_hw=(meta["height"], meta["width"]),
        image_id=int(meta["image_id"]),
        class_id=0,
        cfg=post_cfg,
        encode_rle=False,
    )
    return len(inst)


def main() -> None:
    args = parse_args()
    logger = setup_logger("distill")
    if not args.image:
        raise ValueError("--image is required")
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    cfg = load_config(args.config)
    img_np = _load_image(image_path)
    meta = {
        "path": str(image_path),
        "file_name": image_path.name,
        "height": int(img_np.shape[0]),
        "width": int(img_np.shape[1]),
        "image_id": 0,
    }

    teacher = SAM2Teacher(cfg.get("teacher", {}))
    if teacher.adapter is None:
        raise RuntimeError("Teacher adapter is not configured")
    if isinstance(teacher.adapter, SegmentCDWAdapter):
        raw_preds = teacher.adapter.run([img_np], [meta])[0]
    else:
        raw_preds = teacher.adapter.generate([img_np], [meta])[0]

    post_np = _prepare_post_cfg(cfg, use_torch=False, device="cpu")
    post_torch = _prepare_post_cfg(cfg, use_torch=True, device=args.device)

    for name, post_cfg in [("numpy", post_np), ("torch", post_torch)]:
        t0 = time.perf_counter()
        count = 0
        for _ in range(args.repeat):
            count = _run(post_cfg, raw_preds, meta)
        t1 = time.perf_counter()
        logger.info("profile %s: repeat=%d time=%.3fs instances=%d", name, args.repeat, t1 - t0, count)


if __name__ == "__main__":
    main()
