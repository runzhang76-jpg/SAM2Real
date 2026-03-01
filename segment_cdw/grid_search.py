"""
对 SAM2AutomaticMaskGenerator 做超参数网格搜索，并支持切换提示方式（全局自动、手动点/框）。
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image
from tqdm import tqdm

# 确保可以找到仓库根目录，避免 ModuleNotFoundError
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_cdw.dataset import CocoSparseDataset
from segment_cdw.metrics import coco_eval_metrics


def parse_float_list(val: str) -> List[float]:
    """解析逗号分隔的浮点数列表。"""
    return [float(x) for x in val.split(",") if x.strip()]


def parse_int_list(val: str) -> List[int]:
    """解析逗号分隔的整数列表。"""
    return [int(x) for x in val.split(",") if x.strip()]


def parse_points(val: str) -> np.ndarray:
    """
    将形如 "x1,y1;x2,y2" 的字符串解析为 Nx2 点阵列。
    默认全部前景点，labels 全 1。
    """
    pts = []
    for pair in val.split(";"):
        if not pair.strip():
            continue
        x_str, y_str = pair.split(",")
        pts.append([float(x_str), float(y_str)])
    return np.array(pts, dtype=np.float32)


def parse_box(val: str) -> np.ndarray:
    """解析 "x1,y1,x2,y2" 字符串为长度 4 的数组。"""
    parts = [float(x) for x in val.split(",") if x.strip()]
    if len(parts) != 4:
        raise ValueError("box 需要 4 个值: x1,y1,x2,y2")
    return np.array(parts, dtype=np.float32)


def evaluate_dataset(
    dataset: CocoSparseDataset,
    mask_fn,
    downsample_enabled: bool = False,
    downsample_factor: float = 1.0,
    mask_downsample: bool = False,
    mask_downsample_factor: float = 2.0,
) -> Dict[str, float]:
    """逐图调用 mask_fn(image_np) -> [mask]，按规则计算 Recall/IoU。"""
    coco_results: List[Dict] = []
    img_ids: List[int] = []
    num_images = len(dataset)
    iterator = range(num_images)
    pbar = tqdm(iterator, total=num_images, desc=f"eval", leave=False)
    for idx in pbar:
        image, target = dataset[idx]

        if target["masks"].shape[0] == 0:
            continue

        gt_masks = list(target["masks"])
        img_np = np.array(image, copy=False)
        tqdm.write(
            f"Image {idx+1}/{num_images} | name={target['file_name']} | "
        )

        if downsample_enabled and downsample_factor > 1.0:
            new_w = max(1, int(image.width / downsample_factor))
            new_h = max(1, int(image.height / downsample_factor))
            image_resized = image.resize((new_w, new_h), resample=Image.BILINEAR)
            img_np = np.array(image_resized, copy=False)

        pred_items = mask_fn(img_np)

        pred_masks: List[np.ndarray] = []
        pred_scores: List[float] = []

        for item in pred_items:
            seg = item.get("segmentation", item.get("mask", None))

            piou = item.get("predicted_iou", item.get("score", 1.0))
            stab = item.get("stability_score", item.get("stability", 1.0))
            score = float(piou) * float(stab)

            pred_masks.append(np.asarray(seg, dtype=bool))
            pred_scores.append(score)

        if downsample_enabled and downsample_factor > 1.0:
            restored = []
            for pm in pred_masks:
                pm_img = Image.fromarray(np.asarray(pm, dtype=np.uint8) * 255)
                pm_img = pm_img.resize((image.width, image.height), resample=Image.NEAREST)
                restored.append(np.array(pm_img, copy=False) > 127)
            pred_masks = restored

        if mask_downsample and mask_downsample_factor > 1.0:
            tqdm.write("  mask_downsample is ignored for COCOeval metrics.")

        # 组织为 COCO 格式预测
        image_id = int(target["image_id"])
        img_ids.append(image_id)
        cat_ids = dataset.coco.getCatIds()
        default_cat = 1
        for pm, score in zip(pred_masks, pred_scores):
            pm_uint8 = np.asarray(pm, dtype=np.uint8)
            rle = mask_utils.encode(np.asfortranarray(pm_uint8))
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id": default_cat,
                    "segmentation": rle,
                    "score": float(score),
                }
            )

        msg = (
            f"[{idx+1}/{num_images}] img={target['file_name']} "
            f"preds={len(pred_masks)} gts={len(gt_masks)}"
        )
        pbar.set_postfix_str(f"preds {len(pred_masks)}")
        tqdm.write(msg)

    return coco_eval_metrics(dataset.coco, coco_results, img_ids)


def build_mask_fn_auto(
    sam_model,
    points_per_side: int,
    points_per_batch: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    stability_score_offset: float,
    mask_threshold: float,
    box_nms_thresh: float,
    crop_n_layers: int,
    crop_nms_thresh: float,
    crop_overlap_ratio: float,
    crop_n_points_downscale_factor: int,
    min_mask_region_area: int,
    use_m2m: bool = False,
) -> callable:
    """返回自动提示生成掩码的函数。"""
    generator = SAM2AutomaticMaskGenerator(
        sam_model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        mask_threshold=mask_threshold,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        crop_overlap_ratio=crop_overlap_ratio,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
        output_mode="binary_mask",
        use_m2m=use_m2m,
        multimask_output=True
    )

    def _fn(image_np: np.ndarray) -> List[Dict]:
        return generator.generate(image_np)

    return _fn


def build_mask_fn_manual(
    sam_model,
    points: np.ndarray | None,
    point_labels: np.ndarray | None,
    box: np.ndarray | None,
) -> callable:
    """返回使用手动点/框提示的掩码函数（单次多掩码，取得分最高）。"""
    predictor = SAM2ImagePredictor(sam_model)

    def _fn(image_np: np.ndarray) -> List[Dict]:
        predictor.set_image(image_np)
        masks, scores, _ = predictor.predict(
            point_coords=points if points is not None else None,
            point_labels=point_labels if point_labels is not None else None,
            box=box if box is not None else None,
            multimask_output=True,
            normalize_coords=True,
        )
        best_idx = int(np.argmax(scores))
        return [
            {
                "segmentation": masks[best_idx],
                "score": float(scores[best_idx]),
            }
        ]

    return _fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM2 自动掩码生成器超参网格搜索"
    )
    parser.add_argument("--data-root", type=Path, default="segment_cdw/data/sparse_coco_segment_cdw")
    parser.add_argument("--annotation", type=Path, default="segment_cdw\data\sparse_coco_segment_cdw/annotation/annotations_coco_0.json", 
                        help="COCO 标注路径，默认 data/sparse_coco_segment_cdw/annotation/instances.json",)
    parser.add_argument("--config", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml", 
                        help="SAM2 配置文件路径",)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2.1_hiera_tiny.pt",
                        help="SAM2 权重文件",)
    parser.add_argument("--device", type=str, default="cuda")
    # 超参数网格
    parser.add_argument("--points-per-side-list", type=str, default="16, 24, 32, 40, 48", 
                        help="网格密度： points_per_side 候选值，逗号分隔（从稀疏到较密的 3 个档位）",)
    parser.add_argument("--points-per-batch-list", type=str, default="32", 
                        help="points_per_batch 候选值，逗号分隔（一次送进模型的点数，通常固定）",)
    parser.add_argument("--pred-iou-thresh-list",type=str, default="0.5, 0.6, 0.7, 0.8, 0.9", 
                        help="pred_iou_thresh 候选值，逗号分隔（mask 质量阈值，从偏召回到偏精度）",)
    parser.add_argument("--stability-thresh-list", type=str, default="0.5, 0.6, 0.7, 0.8, 0.9", 
                        help="stability_score_thresh 候选值，逗号分隔（形状稳定性，从宽松到严格）",)
    parser.add_argument("--stability-score-offset-list", type=str, default="1.0", 
                        help="stability_score_offset 候选值，逗号分隔（通常固定为 1.0）",)
    parser.add_argument("--mask-thresh-list", type=str, default="0.0", 
                        help="mask_threshold 候选值，逗号分隔（像素级二值化阈值，第一阶段固定）",)
    parser.add_argument("--box-nms-thresh-list",type=str, default="0.7",
                        help="box_nms_thresh 候选值，逗号分隔（bbox NMS 阈值，建议先固定 0.7）",)
    parser.add_argument("--crop-n-layers-list",type=str,default="0, 1", 
                        help="crop_n_layers 候选值，逗号分隔（金字塔裁剪层数，0 或 1）",)
    parser.add_argument("--crop-nms-thresh-list", type=str, default="0.7", 
                        help="crop_nms_thresh 候选值，逗号分隔（crop NMS 阈值，先固定为 0.7）",)
    parser.add_argument("--crop-overlap-ratio-list", type=str, default="0.34", 
                        help="crop_overlap_ratio 候选值，逗号分隔（crop 重叠比例，默认约 0.34）",)
    parser.add_argument("--crop-n-points-downscale-factor-list", type=str, default="2",
                        help="crop_n_points_downscale_factor 候选值，逗号分隔（crop 内点密度缩放因子）",)
    parser.add_argument("--min-mask-region-area-list", type=str, default="0", 
                        help="min_mask_region_area 候选值，逗号分隔（单位：像素，用于过滤小碎块）",)
    parser.add_argument("--multimask-output-list",type=str,default="1",
                        help="multimask_output 候选值，逗号分隔（1 表示 True，0 表示 False）",)
    # 提示形式
    parser.add_argument("--prompt-type", type=str, default="auto", choices=["auto", "points", "box"],
                        help="auto 使用全图点网格；points/box 使用手动提示",)
    parser.add_argument("--points", type=str, default="",
                        help="手动点提示: \"x1,y1;x2,y2\"，仅 prompt-type=points 生效",)
    parser.add_argument("--box", type=str, default="",
                        help="手动框提示: \"x1,y1,x2,y2\"，仅 prompt-type=box 生效",)
    # 稀疏样本iou阈值
    parser.add_argument("--iou-threshold", type=float,default=0.5, help="Recall 计算的 IoU 阈值",)
    parser.add_argument("--max-images", type=int, default=None,help="可选：限制评估图片数量",)
    # 掩码下采样
    parser.add_argument("--mask-downsample", action="store_true", default=False, help="评价前对预测/GT 掩码进行最近邻下采样")
    parser.add_argument("--mask-downsample-factor", type=float, default=2.0, help="掩码下采样因子（>1 缩小）")
    # 下采样
    parser.add_argument("--downsample", action="store_true", default=False, help="开启图像/标签下采样")
    parser.add_argument("--downsample-factor", type=float, default=4.0, help="下采样因子 (>1 生效)")
    # 结果保存
    parser.add_argument("--results-out", type=Path, default="segment_cdw/data/sparse_coco_segment_cdw/results/result.json",
                        help="每个网格跑完即保存的结果文件（JSON）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(">>> 运行配置")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    annotation = (
        args.annotation
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

    results = []
    results_out: Path = args.results_out

    if args.prompt_type == "auto":
        pps_list = parse_int_list(args.points_per_side_list)
        ppb_list = parse_int_list(args.points_per_batch_list)
        iou_list = parse_float_list(args.pred_iou_thresh_list)  
        stab_t_list = parse_float_list(args.stability_thresh_list)
        stab_o_list = parse_float_list(args.stability_score_offset_list)
        mask_list = parse_float_list(args.mask_thresh_list)
        nms_list = parse_float_list(args.box_nms_thresh_list)
        crop_n_l_list = parse_int_list(args.crop_n_layers_list)
        crop_n_t_list = parse_float_list(args.crop_nms_thresh_list)
        crop_o_r_list = parse_float_list(args.crop_overlap_ratio_list)
        crop_n_p_list = parse_int_list(args.crop_n_points_downscale_factor_list)
        min_mask_list = parse_int_list(args.min_mask_region_area_list)

        for (
            pps, 
            ppb, 
            iou_t, 
            stab_t, 
            stab_o, 
            mask_t, 
            nms_t, 
            crop_n_l, 
            crop_n_t, 
            crop_o_r, 
            crop_n_p, 
            min_mask, 
            ) in itertools.product(
            pps_list, 
            ppb_list, 
            iou_list, 
            stab_t_list, 
            stab_o_list, 
            mask_list, 
            nms_list, 
            crop_n_l_list, 
            crop_n_t_list, 
            crop_o_r_list, 
            crop_n_p_list, 
            min_mask_list, 
        ):
            mask_fn = build_mask_fn_auto(
                sam_model,
                points_per_side=pps,
                points_per_batch=ppb,
                pred_iou_thresh=iou_t,      
                stability_score_thresh=stab_t,
                stability_score_offset=stab_o,
                mask_threshold=mask_t,
                box_nms_thresh=nms_t,
                crop_n_layers=crop_n_l,
                crop_nms_thresh=crop_n_t,
                crop_overlap_ratio=crop_o_r,
                crop_n_points_downscale_factor=crop_n_p,
                min_mask_region_area=min_mask,
            )
            metrics = evaluate_dataset(
                dataset,
                mask_fn,
                downsample_enabled=args.downsample,
                downsample_factor=args.downsample_factor,
                mask_downsample=args.mask_downsample,
                mask_downsample_factor=args.mask_downsample_factor,
            )
            results.append(
            {
                "points_per_side": pps,
                "points_per_batch": ppb,
                "pred_iou_thresh": iou_t,
                "stability_thresh": stab_t,
                "stability_score_offset": stab_o,
                "mask_thresh": mask_t,
                "box_nms_thresh": nms_t,
                "crop_n_layers": crop_n_l,
                "crop_nms_thresh": crop_n_t,
                "crop_overlap_ratio": crop_o_r,
                "crop_n_points_downscale_factor": crop_n_p,
                "min_mask_region_area": min_mask,
                "mAP": metrics.get("mAP", 0.0),
                "mAR": metrics.get("mAR", 0.0),
                "AP50": metrics.get("AP50", 0.0),
                "AP75": metrics.get("AP75", 0.0),
                "AP95": metrics.get("AP95", 0.0),
                "num_preds": metrics.get("num_preds", 0),
            }
        )
            results_out.parent.mkdir(parents=True, exist_ok=True)
            results_out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        # 手动提示仅评一次
        points = parse_points(args.points) if args.prompt_type == "points" and args.points else None
        point_labels = (
            np.ones(len(points), dtype=np.int64) if points is not None else None
        )
        box = parse_box(args.box) if args.prompt_type == "box" and args.box else None
        mask_fn = build_mask_fn_manual(sam_model, points, point_labels, box)
        metrics = evaluate_dataset(
            dataset,
            mask_fn,
            iou_threshold=args.iou_threshold,
            max_images=args.max_images,
            downsample_enabled=args.downsample,
            downsample_factor=args.downsample_factor,
            device=args.device,
            hyper_params={"prompt_type": args.prompt_type, "points": args.points, "box": args.box},
            mask_downsample=args.mask_downsample,
            mask_downsample_factor=args.mask_downsample_factor,
        )
        results.append(
            {
                "prompt_type": args.prompt_type,
                "points": args.points,
                "box": args.box,
                "mAP": metrics.get("mAP", 0.0),
                "mAR": metrics.get("mAR", 0.0),
                "AP50": metrics.get("AP50", 0.0),
                "AP75": metrics.get("AP75", 0.0),
                "AP95": metrics.get("AP95", 0.0),
                "num_preds": metrics.get("num_preds", 0),
            }
        )
        results_out.parent.mkdir(parents=True, exist_ok=True)
        results_out.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    print("----- 网格搜索结果 -----")
    if results:
        header = [
            "prompt",
            "pps",
            "ppb",
            "pred_iou",
            "stab",
            "stab_off",
            "mask_thr",
            "box_nms",
            "crop_n",
            "crop_nms",
            "crop_ovl",
            "crop_down",
            "min_area",
            "mAP",
            "AP50",
            "AP75",
            "AP95",
            "mAR",
            "preds",
        ]
        print(" | ".join(header))
        for res in results:
            row = [
                res.get("prompt_type", "auto"),
                str(res.get("points_per_side", "-")),
                str(res.get("points_per_batch", "-")),
                f"{res.get('pred_iou_thresh', '-')}",
                f"{res.get('stability_thresh', '-')}",
                f"{res.get('stability_score_offset', '-')}",
                f"{res.get('mask_thresh', '-')}",
                f"{res.get('box_nms_thresh', '-')}",
                f"{res.get('crop_n_layers', '-')}",
                f"{res.get('crop_nms_thresh', '-')}",
                f"{res.get('crop_overlap_ratio', '-')}",
                f"{res.get('crop_n_points_downscale_factor', '-')}",
                f"{res.get('min_mask_region_area', '-')}",
                f"{res.get('mAP', 0.0):.3f}",
                f"{res.get('AP50', 0.0):.3f}",
                f"{res.get('AP75', 0.0):.3f}",
                f"{res.get('AP95', 0.0):.3f}",
                f"{res.get('mAR', 0.0):.3f}",
                str(res.get("num_preds", "-")),
            ]
            print(" | ".join(row))
    else:
        print("No results computed.")


if __name__ == "__main__":
    main()
