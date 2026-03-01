#!/usr/bin/env python3
"""
SAM2 -> mask 过滤 -> 导出可用于分类的小图与 CSV 标签。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from utils.dataset import CocoSparseDataset
from utils.mask_filter import filter_sam2_masks, MaskItem


def square_crop_coords(
    img_hw: Tuple[int, int], bbox_xyxy: Tuple[float, float, float, float], margin_ratio: float = 0.1
) -> Tuple[int, int, int, int]:
    """根据 bbox 计算正方形裁剪区域（含 margin），返回坐标。"""
    h, w = img_hw
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw, bh) * (1.0 + 2.0 * margin_ratio)
    half = side * 0.5
    x1n = int(max(0, np.floor(cx - half)))
    y1n = int(max(0, np.floor(cy - half)))
    x2n = int(min(w, np.ceil(cx + half)))
    y2n = int(min(h, np.ceil(cy + half)))
    return x1n, y1n, x2n, y2n


def masked_square_crop(
    img: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    margin_ratio: float = 0.1,
) -> np.ndarray:
    """用 mask 抠出目标后再做正方形裁剪，非 mask 区域置黑。"""
    h, w = img.shape[:2]
    x1n, y1n, x2n, y2n = square_crop_coords((h, w), bbox_xyxy, margin_ratio)
    img_crop = img[y1n:y2n, x1n:x2n].copy()
    mask_crop = mask[y1n:y2n, x1n:x2n]
    if mask_crop.dtype != bool:
        mask_crop = mask_crop.astype(bool)
    img_crop[~mask_crop] = 0
    return img_crop


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export masked patches + CSV labels")
    parser.add_argument("--dataset-root", type=Path, default="segment_cdw/data/dataset/segment_cdw_coco_dataset/")
    parser.add_argument("--ann-file", type=Path, default="segment_cdw/data/dataset/segment_cdw_coco_dataset/annotations/instances_val.json")
    parser.add_argument("--image-dir", type=str, default="images_val")
    parser.add_argument("--sam-config", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam-ckpt", type=str, default="checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default="segment_cdw/data/patches")
    parser.add_argument("--output-csv", type=Path, default="segment_cdw/data/patches.csv")
    parser.add_argument("--margin-ratio", type=float, default=0.1)
    parser.add_argument("--patch-size", type=int, default=224, help="导出小图尺寸（正方形）")
    parser.add_argument("--min-mask-area", type=int, default=0)
    parser.add_argument("--max-mask-area-frac", type=float, default=0.20)
    parser.add_argument("--points-per-side", type=int, default=24)
    parser.add_argument("--points-per-batch", type=int, default=16)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.6)
    parser.add_argument("--stability-thresh", type=float, default=0.7)
    parser.add_argument("--stability-score-offset", type=float, default=1.0)
    parser.add_argument("--mask-thresh", type=float, default=0.0)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    parser.add_argument("--crop-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-overlap-ratio", type=float, default=0.34)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--min-mask-region-area", type=int, default=0)
    parser.add_argument("--multimask-output", type=int, default=1)
    parser.add_argument("--img-downsample", action="store_true", default=False, help="进入 SAM 前对原图下采样")
    parser.add_argument("--img-downsample-factor", type=float, default=4.0, help="原图下采样因子（>1 缩小）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CocoSparseDataset(
        data_root=args.dataset_root,
        annotation_file=args.ann_file,
        image_dir=args.image_dir,
    )
    coco_gt: COCO = dataset.coco

    sam_model = build_sam2(
        config_file=args.sam_config,
        ckpt_path=args.sam_ckpt,
        device=args.device,
    )
    mask_gen = SAM2AutomaticMaskGenerator(
        sam_model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
        stability_score_offset=args.stability_score_offset,
        mask_threshold=args.mask_thresh,
        box_nms_thresh=args.box_nms_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_nms_thresh=args.crop_nms_thresh,
        crop_overlap_ratio=args.crop_overlap_ratio,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
        output_mode="binary_mask",
        multimask_output=bool(args.multimask_output),
    )

    rows: List[Tuple[str, int]] = []
    for idx in range(len(dataset)):
        img_pil, target = dataset[idx]
        img_np = np.array(img_pil)
        orig_h, orig_w = img_np.shape[:2]
        sam_img = img_np
        if args.img_downsample and args.img_downsample_factor > 1.0:
            new_w = max(1, int(orig_w / args.img_downsample_factor))
            new_h = max(1, int(orig_h / args.img_downsample_factor))
            sam_img = np.array(
                Image.fromarray(img_np).resize((new_w, new_h), resample=Image.BILINEAR)
            )
        image_id = int(target["image_id"])
        pred_items = mask_gen.generate(sam_img)
        if args.img_downsample and args.img_downsample_factor > 1.0:
            resized_items = []
            for item in pred_items:
                seg = item.get("segmentation", item.get("mask", None))
                if seg is None:
                    resized_items.append(item)
                    continue
                seg_img = Image.fromarray(np.asarray(seg, dtype=np.uint8) * 255)
                seg_img = seg_img.resize((orig_w, orig_h), resample=Image.NEAREST)
                new_item = dict(item)
                new_item["segmentation"] = np.array(seg_img, copy=False) > 127
                resized_items.append(new_item)
            pred_items = resized_items
        filtered = filter_sam2_masks(
            pred_items,
            image_hw=(img_np.shape[0], img_np.shape[1]),
            min_area=args.min_mask_area,
            max_area_frac=args.max_mask_area_frac,
        )

        ann_ids = coco_gt.getAnnIds(imgIds=[image_id], iscrowd=0)
        gt_anns = coco_gt.loadAnns(ann_ids)
        gt_masks: List[np.ndarray] = []
        gt_cats: List[int] = []
        for ann in gt_anns:
            m = mask_utils.decode(mask_utils.frPyObjects(ann["segmentation"], img_np.shape[0], img_np.shape[1]))
            if m.ndim == 3:
                m = np.any(m, axis=2)
            gt_masks.append(m.astype(bool))
            gt_cats.append(int(ann["category_id"]))

        for j, item in enumerate(filtered):
            if len(gt_masks) == 0:
                continue
            best_iou = -1.0
            best_cat = None
            for gm, cid in zip(gt_masks, gt_cats):
                iou = mask_iou(item.mask, gm)
                if iou > best_iou:
                    best_iou = iou
                    best_cat = cid
            if best_cat is None:
                continue
            patch = masked_square_crop(img_np, item.mask, item.bbox_xyxy, margin_ratio=args.margin_ratio)
            patch_img = Image.fromarray(patch).resize((args.patch_size, args.patch_size), resample=Image.BILINEAR)
            patch_name = f"{image_id}_{j:04d}.jpg"
            patch_path = args.output_dir / patch_name
            patch_img.save(patch_path)
            rows.append((patch_name, int(best_cat)))

    with args.output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
