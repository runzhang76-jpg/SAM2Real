"""
在稀疏 COCO 验证集上跑 SAM2，输出 Recall / IoU。

Example:
python -m segment_cdw.eval_sparse \
  --data-root segment_cdw/data \
  --annotation segment_cdw/data/annotations/instances.json \
  --config configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --device cuda \
  --iou-threshold 0.5 \
  --max-images 50
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from segment_cdw.dataset import CocoSparseDataset
from segment_cdw.metrics import aggregate, summarize_image


def evaluate_dataset(
    mask_generator: SAM2AutomaticMaskGenerator,
    dataset: CocoSparseDataset,
    iou_threshold: float = 0.5,
    max_images: int | None = None,
) -> Dict[str, float]:
    summaries: List[Dict] = []
    num_images = len(dataset) if max_images is None else min(len(dataset), max_images)
    for idx in range(num_images):
        image, target = dataset[idx]
        if target["masks"].shape[0] == 0:
            continue

        preds = mask_generator.generate(np.array(image))
        pred_masks = [p["segmentation"].astype(bool) for p in preds]
        gt_masks = list(target["masks"])

        summaries.append(
            summarize_image(pred_masks, gt_masks, iou_threshold=iou_threshold)
        )

    return aggregate(summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sparse samples with SAM2 recall/IoU."
    )
    parser.add_argument(
        "--data-root",
        default="segment_cdw/data",
        type=Path,
        help="Dataset root containing images/ and annotations/.",
    )
    parser.add_argument(
        "--annotation",
        default=None,
        type=Path,
        help="COCO annotation json. Defaults to <data-root>/annotations/instances.json",
    )
    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        type=str,
        help="SAM2 config yaml path.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sam2.1_hiera_tiny.pt",
        type=str,
        help="Checkpoint path for SAM2 weights.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Device for inference (cuda, mps, or cpu).",
    )
    parser.add_argument(
        "--points-per-side",
        default=32,
        type=int,
        help="Grid resolution for automatic point prompts.",
    )
    parser.add_argument(
        "--iou-threshold",
        default=0.5,
        type=float,
        help="IoU threshold used for recall.",
    )
    parser.add_argument(
        "--max-images",
        default=None,
        type=int,
        help="Optional cap on number of images to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation = (
        args.annotation
        if args.annotation is not None
        else args.data_root / "annotations" / "instances.json"
    )

    dataset = CocoSparseDataset(
        data_root=args.data_root,
        annotation_file=annotation,
    )

    sam_model = build_sam2(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device=args.device,
    )
    mask_generator = SAM2AutomaticMaskGenerator(
        sam_model,
        points_per_side=args.points_per_side,
        output_mode="binary_mask",
    )

    metrics = evaluate_dataset(
        mask_generator,
        dataset,
        iou_threshold=args.iou_threshold,
        max_images=args.max_images,
    )

    print("----- Sparse Validation -----")
    print(f"Images evaluated: {len(dataset) if args.max_images is None else min(len(dataset), args.max_images)}")
    print(f"Objects evaluated: {metrics['num_objects']}")
    print(f"Recall@{args.iou_threshold:.2f}: {metrics['recall']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")


if __name__ == "__main__":
    main()
