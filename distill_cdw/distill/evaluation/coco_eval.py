"""COCO 评估封装，支持优雅降级。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from distill.core.structures import InstancePrediction
from distill.utils.logging import get_logger

try:
    from pycocotools.coco import COCO  # type: ignore
    from pycocotools.cocoeval import COCOeval  # type: ignore
    from pycocotools import mask as mask_utils  # type: ignore
except Exception:  # pragma: no cover - 可选依赖
    COCO = None  # type: ignore
    COCOeval = None  # type: ignore
    mask_utils = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover - 可选依赖
    torch = None  # type: ignore


def _encode_mask(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    if mask_utils is None:
        return None
    m = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(m)
    if isinstance(rle.get("counts", None), bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _to_coco_results(
    image_id: int,
    preds: List[InstancePrediction],
) -> List[Dict[str, Any]]:
    results = []
    for inst in preds:
        seg = inst.rle
        if seg is None and inst.mask is not None:
            seg = _encode_mask(np.asarray(inst.mask))
        if seg is None:
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
    return results


class CocoEvaluator:
    """COCO 评估器封装。"""

    def __init__(self, dataloader: Any, gt_json: Optional[str] = None, iou_types: Optional[Iterable[str]] = None) -> None:
        self.logger = get_logger("distill")
        self.dataloader = dataloader
        self.gt_json = gt_json
        self._coco_gt = COCO(gt_json) if COCO is not None and gt_json else None
        self.iou_types = list(iou_types) if iou_types is not None else ["segm"]

    def _predict_batch(self, model: Any, images: Any) -> List[List[InstancePrediction]]:
        if hasattr(model, "predict"):
            return model.predict(images)
        self.logger.warning("模型缺少 predict 接口，跳过 COCO 评估")
        return [[] for _ in range(len(images))]

    def evaluate(self, model: Any) -> Dict[str, Any]:
        if COCO is None or COCOeval is None or self._coco_gt is None:
            self.logger.warning("pycocotools 不可用或 gt_json 缺失，跳过 COCO 评估")
            return {}

        if torch is not None and hasattr(model, "eval"):
            model.eval()

        results: List[Dict[str, Any]] = []
        with torch.no_grad() if torch is not None else _nullcontext():
            for batch in self.dataloader:
                images = batch.get("images")
                image_ids = batch.get("image_ids", [])
                preds_batch = self._predict_batch(model, images)
                for idx, preds in enumerate(preds_batch):
                    image_id = int(image_ids[idx]) if idx < len(image_ids) else idx
                    results.extend(_to_coco_results(image_id, preds))

        if not results:
            self.logger.warning("无有效预测结果，跳过 COCO 评估")
            return {}

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


class _nullcontext:  # pragma: no cover - 小型兼容性辅助
    def __enter__(self) -> None:
        return None

    def __exit__(self, *excinfo: Any) -> None:
        return None
