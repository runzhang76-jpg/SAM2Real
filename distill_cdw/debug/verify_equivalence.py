#!/usr/bin/env python
"""Compare numpy vs torch postprocess outputs for the same image."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PKG_ROOT = ROOT / "distill_cdw"
if PKG_ROOT.exists() and str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from distill.core.structures import InstancePrediction
from distill.teacher.sam2_teacher import SAM2Teacher, SegmentCDWAdapter
from distill.teacher.postprocess import PostProcessPipeline, convert_instances
from distill.utils.config import load_config
from distill.utils.logging import setup_logger

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify postprocess equivalence")
    parser.add_argument("--config", default="distill_cdw/configs/distill_default.yaml")
    parser.add_argument("--image", default="Co3_20250714172336216.jpg", help="Image path (required)")
    parser.add_argument("--device", default="auto", help="torch device for torch path")
    parser.add_argument("--topk", type=int, default=50, help="TopK instances to compare")
    return parser.parse_args()


def _load_image(path: Path) -> np.ndarray:
    if Image is None:
        raise RuntimeError("PIL not available")
    return np.asarray(Image.open(path).convert("RGB"))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _prepare_post_cfg(cfg: Dict[str, Any], use_torch: bool, device: str) -> Dict[str, Any]:
    post_cfg = copy.deepcopy(cfg.get("teacher", {}).get("postprocess", {}))
    runtime = post_cfg.get("runtime", {})
    runtime["use_torch"] = bool(use_torch)
    runtime["device"] = device
    post_cfg["runtime"] = runtime
    return post_cfg


def _run_postprocess(
    post_cfg: Dict[str, Any],
    raw_preds: List[Dict[str, Any]],
    image_meta: Dict[str, Any],
) -> List[InstancePrediction]:
    pipeline = PostProcessPipeline(post_cfg)
    processed = pipeline(list(raw_preds), image_meta)
    return convert_instances(
        processed,
        image_hw=(image_meta["height"], image_meta["width"]),
        image_id=int(image_meta["image_id"]),
        class_id=0,
        cfg=post_cfg,
        encode_rle=False,
    )


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

    inst_np = _run_postprocess(post_np, raw_preds, meta)
    inst_torch = _run_postprocess(post_torch, raw_preds, meta)

    inst_np = sorted(inst_np, key=lambda x: x.score, reverse=True)[: args.topk]
    inst_torch = sorted(inst_torch, key=lambda x: x.score, reverse=True)[: args.topk]

    logger.info("instances: numpy=%d torch=%d", len(inst_np), len(inst_torch))

    # match by highest mask IoU
    used = set()
    ious: List[float] = []
    score_diff: List[float] = []
    bbox_diff: List[float] = []
    for t in inst_torch:
        best_iou = 0.0
        best_j = -1
        for j, n in enumerate(inst_np):
            if j in used:
                continue
            if t.mask is None or n.mask is None:
                continue
            iou = _mask_iou(np.asarray(t.mask), np.asarray(n.mask))
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            used.add(best_j)
            n = inst_np[best_j]
            ious.append(best_iou)
            score_diff.append(abs(float(t.score) - float(n.score)))
            tb = t.bbox
            nb = n.bbox
            bbox_diff.append(sum(abs(float(tb[i]) - float(nb[i])) for i in range(4)))

    summary = {
        "match_count": len(ious),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "min_iou": float(np.min(ious)) if ious else 0.0,
        "mean_score_abs_diff": float(np.mean(score_diff)) if score_diff else 0.0,
        "mean_bbox_abs_diff": float(np.mean(bbox_diff)) if bbox_diff else 0.0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
