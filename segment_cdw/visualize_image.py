#!/usr/bin/env python3
"""
单张图像可视化：SAM2 自动分割并叠加 mask。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from segment_cdw.utils.mask_filter import filter_sam2_masks
from segment_cdw.utils.visualize import save_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SAM2 masks on one image")
    parser.add_argument("--image", type=Path, default="segment_cdw/data/dataset/segment_cdw_coco_dataset/images_val/Co1_20250714170609083.jpg")
    parser.add_argument("--ann-file", type=Path, default="segment_cdw/data/dataset/segment_cdw_coco_dataset/annotations/instances_val.json", help="COCO 标注文件路径")
    parser.add_argument("--output", type=Path, default="segment_cdw/vis_mask.png", help="输出图像路径")
    parser.add_argument("--sam-config", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam-ckpt", type=str, default="checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--device", type=str, default="cuda")
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
    parser.add_argument("--img-downsample-factor", type=float, default=2.0, help="原图下采样因子（>1 缩小）")
    parser.add_argument("--vis-max-masks", type=int, default=200, help="可视化最多叠加的预测掩码数量")
    parser.add_argument("--min-mask-area", type=int, default=0, help="过滤过小 mask")
    parser.add_argument("--max-mask-area-frac", type=float, default=0.20, help="过滤过大 mask 的面积比例阈值")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img_np = np.array(Image.open(args.image).convert("RGB"))
    orig_h, orig_w = img_np.shape[:2]

    sam_img = img_np
    if args.img_downsample and args.img_downsample_factor > 1.0:
        new_w = max(1, int(orig_w / args.img_downsample_factor))
        new_h = max(1, int(orig_h / args.img_downsample_factor))
        sam_img = np.array(
            Image.fromarray(img_np).resize((new_w, new_h), resample=Image.BILINEAR)
        )

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

    # mask 过滤
    filtered = filter_sam2_masks(
        pred_items,
        image_hw=(orig_h, orig_w),
        min_area=args.min_mask_area,
        max_area_frac=args.max_mask_area_frac,
    )

    # 读取 GT，并为每个 GT 选一个最佳 IoU 的预测 mask
    coco = COCO(str(args.ann_file))
    img_name = args.image.name
    img_ids = [i for i in coco.getImgIds() if coco.loadImgs(i)[0]["file_name"] == img_name]
    if not img_ids:
        raise ValueError(f"Image not found in COCO annotations: {img_name}")
    img_id = img_ids[0]
    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=0)
    gt_anns = coco.loadAnns(ann_ids)
    gt_masks: List[np.ndarray] = []
    for ann in gt_anns:
        m = mask_utils.decode(mask_utils.frPyObjects(ann["segmentation"], orig_h, orig_w))
        if m.ndim == 3:
            m = np.any(m, axis=2)
        gt_masks.append(m.astype(bool))

    pred_masks: List[np.ndarray] = [m.mask for m in filtered]
    matched_masks: List[np.ndarray] = []
    for gt in gt_masks:
        if not pred_masks:
            break
        best_iou = -1.0
        best_mask = None
        for pm in pred_masks:
            inter = np.logical_and(pm, gt).sum()
            union = np.logical_or(pm, gt).sum()
            iou = 0.0 if union == 0 else float(inter / union)
            if iou > best_iou:
                best_iou = iou
                best_mask = pm
        if best_mask is not None:
            matched_masks.append(best_mask)

    save_overlay(
        img_np,
        matched_masks,
        gt_masks=[],
        save_path=args.output,
        orig_hw=None,
        max_masks=args.vis_max_masks,
        downsample_factor=1.0,
    )


if __name__ == "__main__":
    main()
