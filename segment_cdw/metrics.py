from typing import Dict, Iterable, List, Sequence

import numpy as np
from pycocotools.cocoeval import COCOeval


def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算两个布尔掩码的 IoU。"""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def summarize_image(
    pred_masks: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    iou_threshold: float = 0.5,
) -> Dict[str, float | List[float]]:
    """
    对每个 GT 掩码在所有预测掩码中取最大 IoU。
    未匹配到的预测不计惩罚（忽略 FP）。
    """
    best_ious: List[float] = []
    multi_match_flags: List[bool] = []
    frag_ratios: List[float] = []
    for gt_mask in gt_masks:
        ious = [mask_iou(pred, gt_mask) for pred in pred_masks]
        best_ious.append(max(ious))

        # 1) 统计与该 GT 相交且 IoU > 阈值的预测数量
        k = sum(iou > iou_threshold for iou in ious)
        multi_match_flags.append(k > 1)

        # 2) 计算碎裂率（frag_ratio）
        # 只考虑与 GT 有交集的预测
        intersections = []
        for pred in pred_masks:
            inter = np.logical_and(pred, gt_mask).sum()
            if inter > 0:
                intersections.append(inter)
    
        if len(intersections) == 0:
            frag_ratios.append(0.0)
        else:
            a_sum = float(np.sum(intersections))
            a_max = float(np.max(intersections))
            frag_ratios.append(a_max / a_sum if a_sum > 0 else 0.0)

    hits = sum(iou >= iou_threshold for iou in best_ious)
    total = len(gt_masks)
    return {
        "recall": hits / total if total else 0.0,
        "mean_iou": float(np.mean(best_ious)) if best_ious else 0.0,
        "ious": best_ious,
        # 过分割相关指标
        "multi_match_ratio": float(np.mean(multi_match_flags)) if multi_match_flags else 0.0,  # GT 被多个预测覆盖（IoU>阈值）的比例
        "frag_ratio_mean": float(np.mean(frag_ratios)) if frag_ratios else 0.0,  # 碎裂率均值
        "frag_ratios": frag_ratios,
    }


def aggregate(summaries: Iterable[Dict[str, float | List[float]]]) -> Dict[str, float]:
    """汇总多张图片的指标，得到数据级别的统计。"""
    recalls: List[float] = []
    ious: List[float] = []
    multi_match: List[float] = []
    frag_ratios_all: List[float] = []
    for summary in summaries:
        recalls.append(float(summary.get("recall", 0.0)))
        ious.extend(summary.get("ious", []))  # type: ignore[arg-type]
        multi_match.append(float(summary.get("multi_match_ratio", 0.0)))
        frag_ratios_all.extend(summary.get("frag_ratios", []))  # type: ignore[arg-type]
    total_samples = len(recalls)
    return {
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "num_objects": len(ious),
        "multi_match_ratio": float(np.mean(multi_match)) if multi_match else 0.0,
        "frag_ratio_mean": float(np.mean(frag_ratios_all)) if frag_ratios_all else 0.0,
    }


def coco_eval_metrics(coco_gt, coco_results: List[Dict], img_ids: List[int]) -> Dict[str, float]:
    if len(coco_results) == 0:
        return {"mAP": 0.0, "mAR": 0.0, "num_preds": 0}

    coco_dt = coco_gt.loadRes(coco_results)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
    evaluator.params.imgIds = sorted(set(img_ids))

    evaluator.params.maxDets = [1, 10, 200]
    evaluator.params.useCats = 0

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    precision = evaluator.eval["precision"]
    iou_thrs = evaluator.params.iouThrs
    area_labels = evaluator.params.areaRngLbl
    max_dets = evaluator.params.maxDets

    aind = area_labels.index("all")
    mind = len(max_dets) - 1

    print(float(stats[8]))
    
    def _ap_at_iou(iou: float) -> float:
        idx = np.where(np.isclose(iou_thrs, iou))[0]
        if len(idx) == 0:
            return 0.0
        t = int(idx[0])
        pr = precision[t, :, :, aind, mind]
        pr = pr[pr > -1]
        return float(np.mean(pr)) if pr.size else 0.0

    return {
        "mAP": float(stats[0]),
        "mAR": float(stats[8]),
        "num_preds": len(coco_results),
        "AP50": _ap_at_iou(0.50),
        "AP75": _ap_at_iou(0.75),
        "AP95": _ap_at_iou(0.95),
    }
