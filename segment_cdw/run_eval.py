#!/usr/bin/env python3
"""
SAM2 + mask 过滤 + patch 分类 + COCOeval 的完整评估 pipeline。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch import nn

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


from utils.mask_filter import filter_sam2_masks, MaskItem

from classier_network.CNNs import ResNet50, seresnext50_32x4d


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


def mask_to_rle(mask_bool: np.ndarray) -> Dict:
    """将 bool mask 编码成 COCO RLE（counts 为 str）。"""
    mask_uint8 = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = mask_utils.encode(mask_uint8)
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def classify_patches_batch(
    model: torch.nn.Module,
    patches: List[np.ndarray],
    device: str,
    batch_size: int = 32,
    normalize: bool = True,
) -> List[Tuple[int, float]]:
    """对 patch 批量分类，返回 [(class_id, class_prob), ...]。"""
    if len(patches) == 0:
        return []
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    norm = transforms.Normalize(mean=mean, std=std)
    outputs: List[Tuple[int, float]] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i : i + batch_size]
            tensor = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float() / 255.0
            if normalize:
                tensor = torch.stack([norm(t) for t in tensor])
            tensor = tensor.to(device)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            conf, cls = torch.max(probs, dim=1)
            outputs.extend([(int(c), float(p)) for c, p in zip(cls.cpu(), conf.cpu())])
    return outputs


def build_coco_results_for_image(
    image_id: int,
    img_np: np.ndarray,
    filtered_masks: List[MaskItem],
    classifier: torch.nn.Module,
    device: str,
    inst_start: int = 0,
    margin_ratio: float = 0.1,
    patch_size: int = 224,
    category_map: Optional[Dict[int, int]] = None,
    cls_batch_size: int = 32,
    cls_normalize: bool = True,
) -> Tuple[List[Dict], int]:
    """将 MaskItem 列表转换为 COCO results，并返回下一个实例编号。"""
    results = []
    inst_id = inst_start
    patches: List[np.ndarray] = []
    for item in filtered_masks:
        # 用 mask 抠出目标区域，背景置黑后再裁剪
        patch = masked_square_crop(img_np, item.mask, item.bbox_xyxy, margin_ratio=margin_ratio)
        patch_img = Image.fromarray(patch).resize((patch_size, patch_size), resample=Image.BILINEAR)
        patches.append(np.array(patch_img))

    cls_outputs = classify_patches_batch(
        classifier,
        patches,
        device=device,
        batch_size=cls_batch_size,
        normalize=cls_normalize,
    )

    for item, (class_id, class_prob) in zip(filtered_masks, cls_outputs):
        if category_map is not None:
            class_id = category_map.get(class_id, class_id)
        score = float(class_prob) * float(item.score)
        x1, y1, x2, y2 = item.bbox_xyxy
        bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        rle = mask_to_rle(item.mask)
        results.append(
            {
                "image_id": int(image_id),
                "category_id": int(class_id),
                "segmentation": rle,
                "bbox": bbox_xywh,
                "score": score,
            }
        )
        inst_id += 1
    return results, inst_id


def load_category_map(path: Optional[str]) -> Optional[Dict[int, int]]:
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    return {int(k): int(v) for k, v in data.items()}


def _norm_label(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def build_category_map_from_coco(coco_gt: COCO) -> Dict[int, int]:
    """
    将分类模型的 label_id 映射到 COCO category_id。
    默认分类 label_map:
      0: crushed_stone
      1: brick
      2: concrete
      3: ceramic
    """
    label_map = {
        0: "crushed_stone",
        1: "brick",
        2: "concrete",
        3: "ceramic",
    }
    cat_name_to_id = {}
    for cat in coco_gt.loadCats(coco_gt.getCatIds()):
        name = _norm_label(cat["name"])
        if name == "cermic":
            name = "ceramic"
        cat_name_to_id[name] = int(cat["id"])

    mapped = {}
    for label_id, name in label_map.items():
        key = _norm_label(name)
        coco_id = cat_name_to_id.get(key)
        if coco_id is not None:
            mapped[int(label_id)] = int(coco_id)
    return mapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 -> COCOeval pipeline")
    # 数据集/标注
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default="../data/sam2/dataset/segment_cdw_coco_dataset/",
        help="数据集根目录",
    )
    parser.add_argument(
        "--ann-file",
        type=Path,
        default="../data/sam2/dataset/segment_cdw_coco_dataset/annotations/instances_test.json",
        help="COCO 标注文件路径",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="images_test",
        help="图像子目录（相对 dataset_root）",
    )
    # SAM2 模型
    parser.add_argument(
        "--sam-config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help="SAM2 配置文件",
    )
    parser.add_argument(
        "--sam-ckpt",
        type=str,
        default="checkpoints/sam2.1_hiera_tiny.pt",
        help="SAM2 权重路径",
    )
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=24,
        help="SAM 网格密度 points_per_side",
    )
    parser.add_argument(
        "--points-per-batch",
        type=int,
        default=16,
        help="SAM points_per_batch",
    )
    parser.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=0.6,
        help="SAM pred_iou_thresh",
    )
    parser.add_argument(
        "--stability-thresh",
        type=float,
        default=0.7,
        help="SAM stability_score_thresh",
    )
    parser.add_argument(
        "--stability-score-offset",
        type=float,
        default=1.0,
        help="SAM stability_score_offset",
    )
    parser.add_argument(
        "--mask-thresh",
        type=float,
        default=0.0,
        help="SAM mask_threshold",
    )
    parser.add_argument(
        "--box-nms-thresh",
        type=float,
        default=0.7,
        help="SAM box_nms_thresh",
    )
    parser.add_argument(
        "--crop-n-layers",
        type=int,
        default=0,
        help="SAM crop_n_layers",
    )
    parser.add_argument(
        "--crop-nms-thresh",
        type=float,
        default=0.7,
        help="SAM crop_nms_thresh",
    )
    parser.add_argument(
        "--crop-overlap-ratio",
        type=float,
        default=0.34,
        help="SAM crop_overlap_ratio",
    )
    parser.add_argument(
        "--crop-n-points-downscale-factor",
        type=int,
        default=1,
        help="SAM crop_n_points_downscale_factor",
    )
    parser.add_argument(
        "--min-mask-region-area",
        type=int,
        default=0,
        help="SAM min_mask_region_area",
    )
    parser.add_argument(
        "--multimask-output",
        type=int,
        default=1,
        help="SAM multimask_output (1 True / 0 False)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="运行设备（cuda/mps/cpu）",
    )
    # 输出
    parser.add_argument(
        "--output-json",
        type=Path,
        default="segment_cdw/data/results/coco/classifier_coco_result.json",
        help="COCO 预测结果输出路径",
    )
    parser.add_argument(
        "--metrics-txt",
        type=Path,
        default="segment_cdw/data/results/coco/classifier_coco_eval_result.json",
        help="COCOeval 指标输出路径（可选）",
    )
    parser.add_argument(
        "--category-map",
        type=str,
        default=None,
        help="类别映射 JSON（可选）",
    )
    # 裁剪与过滤
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.1,
        help="裁剪时的外扩比例",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="分类模型输入尺寸",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=0,
        help="过滤过小 mask 的面积阈值",
    )
    parser.add_argument(
        "--max-mask-area-frac",
        type=float,
        default=0.20,
        help="过滤过大 mask 的面积比例阈值（相对整图）",
    )
    parser.add_argument(
        "--img-downsample",
        action="store_true",
        default=True,
        help="进入 SAM 前对原图下采样",
    )
    parser.add_argument(
        "--img-downsample-factor",
        type=float,
        default=4.0,
        help="原图下采样因子（>1 缩小）",
    )
    # 分类模型
    parser.add_argument(
        "--cls-ckpt",
        type=Path,
        default="segment_cdw/reseresnext50_32x4d_0_6epochs_accuracy0.817_weights.pth",
        help="分类模型权重路径",
    )
    parser.add_argument(
        "--cls-num-classes",
        type=int,
        default=4,
        help="分类模型类别数",
    )
    parser.add_argument(
        "--cls-batch-size",
        type=int,
        default=32,
        help="分类批量大小",
    )
    parser.add_argument(
        "--cls-normalize",
        action="store_true",
        default=True,
        help="分类输入是否做 ImageNet 归一化",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coco_gt = COCO(str(args.ann_file))
    image_dir = args.dataset_root / args.image_dir

    # 构建 SAM2
    sam_model = build_sam2(
        config_file=args.sam_config,
        ckpt_path=args.sam_ckpt,
        device=args.device,
    )
    print("Loaded SAM2 model")
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

    # 构建分类模型
    classifier = seresnext50_32x4d(pretrained=False, out_features=args.cls_num_classes)
    classifier = nn.DataParallel(classifier)
    print(f'classifier load from {args.cls_ckpt}')
    state_dict = torch.load(args.cls_ckpt, map_location='cpu')
    # classifier.load_state_dict(state_dict['model_state_dict'])
    
    classifier.load_state_dict(state_dict)
    classifier.to(args.device)
    classifier.eval()
    print("Loaded classifier model")

    category_map = load_category_map(args.category_map)
    if category_map is None:
        category_map = build_category_map_from_coco(coco_gt)
        print(f"Using default category_map: {category_map}")

    results: List[Dict] = []
    inst_id = 0
    img_ids = sorted(coco_gt.getImgIds())

    for img_id in img_ids:
        info = coco_gt.loadImgs(img_id)[0]
        img_path = image_dir / info["file_name"]
        img_np = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = img_np.shape[:2]
        sam_img = img_np
        if args.img_downsample and args.img_downsample_factor > 1.0:
            new_w = max(1, int(orig_w / args.img_downsample_factor))
            new_h = max(1, int(orig_h / args.img_downsample_factor))
            sam_img = np.array(
                Image.fromarray(img_np).resize((new_w, new_h), resample=Image.BILINEAR)
            )
        # 1) SAM2 预测
        pred_items = mask_gen.generate(sam_img)
        print(f"Mask prediction done: image_id={img_id} preds={len(pred_items)}")

        # 若进行了下采样，将预测 mask 还原回原图尺寸
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

        # 2) mask 过滤
        filtered = filter_sam2_masks(
            pred_items,
            image_hw=(img_np.shape[0], img_np.shape[1]),
            min_area=args.min_mask_area,
            max_area_frac=args.max_mask_area_frac,
        )
        print(f"Mask filtering done: image_id={img_id} kept={len(filtered)}")

        # 3) 批量分类 + 生成 COCO results
        per_img_results, inst_id = build_coco_results_for_image(
            image_id=img_id,
            img_np=img_np,
            filtered_masks=filtered,
            classifier=classifier,
            device=args.device,
            inst_start=inst_id,
            margin_ratio=args.margin_ratio,
            patch_size=args.patch_size,
            category_map=category_map,
            cls_batch_size=args.cls_batch_size,
            cls_normalize=args.cls_normalize,
        )
        print(f"Classification done: image_id={img_id} results={len(per_img_results)}")
        results.extend(per_img_results)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, ensure_ascii=False))

    # 5) COCOeval 评估
    if results:
        if "info" not in coco_gt.dataset:
            coco_gt.dataset["info"] = {}
        if "licenses" not in coco_gt.dataset:
            coco_gt.dataset["licenses"] = []
        coco_dt = coco_gt.loadRes(results)
        evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
        evaluator.params.maxDets = [1, 10, 200]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        print("COCOeval finished")
        if args.metrics_txt:
            args.metrics_txt.write_text("\n".join([str(x) for x in evaluator.stats]))


if __name__ == "__main__":
    main()
