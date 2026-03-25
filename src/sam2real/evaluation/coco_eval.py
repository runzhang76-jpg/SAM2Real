"""COCO evaluation wrapper with graceful fallback."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from sam2real.core.structures import InstancePrediction
from sam2real.utils.logging import get_logger

try:
    from pycocotools.coco import COCO  # type: ignore
    from pycocotools.cocoeval import COCOeval  # type: ignore
    from pycocotools import mask as mask_utils  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    COCO = None  # type: ignore
    COCOeval = None  # type: ignore
    mask_utils = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


def _encode_mask(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    if mask_utils is None:
        return None
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(encoded.get("counts"), bytes):
        encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def _encode_polygon(segmentation: Any, height: int, width: int) -> Optional[Dict[str, Any]]:
    if mask_utils is None:
        return None
    polygon = np.asarray(segmentation, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[0] < 3:
        return None
    rles = mask_utils.frPyObjects([polygon.reshape(-1).tolist()], int(height), int(width))
    encoded = mask_utils.merge(rles)
    if isinstance(encoded.get("counts"), bytes):
        encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def _to_coco_results(image_id: int, preds: List[InstancePrediction]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    results: List[Dict[str, Any]] = []
    stats = {
        "polygon_exports": 0,
        "raw_mask_exports": 0,
        "shape_mismatch_drops": 0,
        "missing_segmentation_drops": 0,
    }
    for inst in preds:
        seg = inst.rle
        meta = inst.meta if isinstance(inst.meta, dict) else {}
        orig_shape = tuple(meta.get("orig_shape", (0, 0)))
        polygon = meta.get("mask_polygon")
        if seg is None and polygon is not None and len(orig_shape) == 2:
            seg = _encode_polygon(polygon, int(orig_shape[0]), int(orig_shape[1]))
            if seg is not None:
                stats["polygon_exports"] += 1
        if seg is None and inst.mask is not None:
            mask = np.asarray(inst.mask)
            if len(orig_shape) == 2 and tuple(mask.shape) != tuple(orig_shape):
                stats["shape_mismatch_drops"] += 1
                continue
            seg = _encode_mask(mask)
            if seg is not None:
                stats["raw_mask_exports"] += 1
        if seg is None:
            stats["missing_segmentation_drops"] += 1
            continue
        x, y, w, h = inst.bbox
        results.append(
            {
                "image_id": int(image_id),
                "category_id": int(inst.class_id),
                "segmentation": seg,
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(inst.score),
            }
        )
    return results, stats


class CocoEvaluator:
    """Evaluate segmentation predictions with pycocotools."""

    def __init__(self, dataloader: Any, gt_json: Optional[str] = None, iou_types: Optional[Iterable[str]] = None) -> None:
        self.logger = get_logger("distill")
        self.dataloader = dataloader
        self.gt_json = gt_json
        self._coco_gt = COCO(gt_json) if COCO is not None and gt_json else None
        self.iou_types = list(iou_types) if iou_types is not None else ["segm"]

    def _predict_batch(self, model: Any, images: Any) -> List[List[InstancePrediction]]:
        if hasattr(model, "predict"):
            return model.predict(images)
        batch_size = int(len(images)) if images is not None else 0
        self.logger.warning("Model is missing a predict() method; skipping COCO evaluation.")
        return [[] for _ in range(batch_size)]

    def evaluate(self, model: Any) -> Dict[str, Any]:
        if COCO is None or COCOeval is None or self._coco_gt is None:
            self.logger.warning("pycocotools is unavailable or gt_json is missing; skipping COCO evaluation.")
            return {}

        restore_training = bool(getattr(model, "training", False)) if hasattr(model, "training") else None
        uses_predict_session = hasattr(model, "begin_predict_session") and hasattr(model, "end_predict_session")
        if uses_predict_session:
            model.begin_predict_session()
        elif torch is not None and hasattr(model, "eval"):
            model.eval()

        results: List[Dict[str, Any]] = []
        total_pred_instances = 0
        total_pred_masks = 0
        total_coco_instances = 0
        pred_class_ids: List[int] = []
        export_stats = {
            "polygon_exports": 0,
            "raw_mask_exports": 0,
            "shape_mismatch_drops": 0,
            "missing_segmentation_drops": 0,
        }
        conf = getattr(model, "conf", None)
        iou = getattr(model, "iou", None)
        max_det = getattr(model, "max_det", None)
        gt_cat_ids = sorted(self._coco_gt.getCatIds()) if self._coco_gt is not None else []

        try:
            with torch.no_grad() if torch is not None else nullcontext():
                for batch in self.dataloader:
                    images = batch.get("images")
                    image_ids = batch.get("image_ids", [])
                    preds_batch = self._predict_batch(model, images)
                    for idx, preds in enumerate(preds_batch):
                        image_id = int(image_ids[idx]) if idx < len(image_ids) else idx
                        total_pred_instances += len(preds)
                        total_pred_masks += sum(
                            1
                            for inst in preds
                            if inst.mask is not None
                            or inst.rle is not None
                            or (
                                isinstance(getattr(inst, "meta", None), dict)
                                and getattr(inst, "meta", {}).get("mask_polygon") is not None
                            )
                        )
                        pred_class_ids.extend(int(inst.class_id) for inst in preds)
                        coco_results, item_stats = _to_coco_results(image_id, preds)
                        for key, value in item_stats.items():
                            export_stats[key] += int(value)
                        total_coco_instances += len(coco_results)
                        results.extend(coco_results)

            if not results:
                self.logger.warning(
                    "COCO eval debug: raw_preds=%d raw_masks=%d coco_valid=%d conf=%s iou=%s max_det=%s gt_cat_ids=%s pred_cat_ids=%s",
                    total_pred_instances,
                    total_pred_masks,
                    total_coco_instances,
                    conf,
                    iou,
                    max_det,
                    gt_cat_ids,
                    sorted(set(pred_class_ids))[:16],
                )
                self.logger.warning("No valid predictions for COCO evaluation; skipping.")
                return {}

            self.logger.info(
                "COCO eval prediction stats: raw_preds=%d raw_masks=%d coco_valid=%d polygon_exports=%d raw_mask_exports=%d "
                "shape_mismatch_drops=%d missing_segmentation_drops=%d conf=%s iou=%s max_det=%s gt_cat_ids=%s pred_cat_ids=%s",
                total_pred_instances,
                total_pred_masks,
                total_coco_instances,
                export_stats["polygon_exports"],
                export_stats["raw_mask_exports"],
                export_stats["shape_mismatch_drops"],
                export_stats["missing_segmentation_drops"],
                conf,
                iou,
                max_det,
                gt_cat_ids,
                sorted(set(pred_class_ids))[:16],
            )

            if gt_cat_ids and pred_class_ids:
                pred_cat_set = set(pred_class_ids)
                if not pred_cat_set.issubset(set(gt_cat_ids)):
                    self.logger.warning(
                        "Predicted category ids are not a subset of gt category ids: pred=%s gt=%s",
                        sorted(pred_cat_set),
                        gt_cat_ids,
                    )

            coco_dt = self._coco_gt.loadRes(results)
            metrics: Dict[str, Any] = {}
            for iou_type in self.iou_types:
                evaluator = COCOeval(self._coco_gt, coco_dt, iouType=iou_type)
                evaluator.params.maxDets = [1, 10, 200]
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()
                stats = evaluator.stats.tolist() if hasattr(evaluator.stats, "tolist") else list(evaluator.stats)
                metrics.update(
                    {
                        f"{iou_type}_AP": float(stats[0]),
                        f"{iou_type}_AP50": float(stats[1]),
                        f"{iou_type}_AP75": float(stats[2]),
                        f"{iou_type}_APs": float(stats[3]),
                        f"{iou_type}_APm": float(stats[4]),
                        f"{iou_type}_APl": float(stats[5]),
                    }
                )
            return metrics
        finally:
            if uses_predict_session:
                model.end_predict_session()
            if not uses_predict_session and restore_training is not None and hasattr(model, "train"):
                model.train(restore_training)
