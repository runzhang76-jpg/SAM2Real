#!/usr/bin/env python
"""COCO  vs GT """

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pycocotools.coco import COCO  # type: ignore
from pycocotools.cocoeval import COCOeval  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _str2bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO eval for pseudolabels")
    parser.add_argument(
        "--pred",
        default="../data/cdw_classify/dataset_seg/pseudolabels/pseudolabels_results.json",
        help="COCO results JSON path",
    )
    parser.add_argument(
        "--gt",
        default="../data/cdw_classify/dataset_seg/annotations/instances_test.json",
        help="COCO GT JSON path",
    )
    parser.add_argument("--iou-type", default="segm", help="segm or bbox")
    parser.add_argument(
        "--class-agnostic",
        type=_str2bool,
        default=False,
        help="Evaluate in class-agnostic mode (true/false)",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional path to save summary JSON including per-class metrics.",
    )
    return parser.parse_args()


def _safe_mean(values: np.ndarray) -> float:
    valid = values[values > -1]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


def _find_iou_index(iou_thresholds: np.ndarray, target: float) -> int:
    matches = np.where(np.isclose(iou_thresholds, target))[0]
    if matches.size == 0:
        raise ValueError(f"IoU threshold {target} not found in evaluator params")
    return int(matches[0])


def _build_per_class_metrics(evaluator: COCOeval, coco_gt: COCO) -> List[Dict[str, Any]]:
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    params = evaluator.params
    area_index = 0
    max_det_index = len(params.maxDets) - 1
    ap50_index = _find_iou_index(np.asarray(params.iouThrs), 0.50)
    ap75_index = _find_iou_index(np.asarray(params.iouThrs), 0.75)

    category_names = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in coco_gt.dataset.get("categories", [])
    }

    rows: List[Dict[str, Any]] = []
    for class_index, category_id in enumerate(params.catIds):
        class_precision = precision[:, :, class_index, area_index, max_det_index]
        class_recall = recall[:, class_index, area_index, max_det_index]
        row = {
            "category_id": int(category_id),
            "category_name": category_names.get(int(category_id), str(category_id)),
            "ap": _safe_mean(class_precision),
            "ap50": _safe_mean(class_precision[ap50_index : ap50_index + 1]),
            "ap75": _safe_mean(class_precision[ap75_index : ap75_index + 1]),
            "ar": _safe_mean(class_recall),
        }
        rows.append(row)
    return rows


def _format_metric(value: float) -> str:
    return "nan" if np.isnan(value) else f"{value:.6f}"


def main() -> None:
    args = parse_args()
    pred_payload = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    preds = pred_payload if isinstance(pred_payload, list) else pred_payload.get("results", pred_payload.get("annotations", []))
    coco_gt = COCO(args.gt)
    coco_dt = coco_gt.loadRes(preds)
    evaluator = COCOeval(coco_gt, coco_dt, iouType=args.iou_type)
    if args.class_agnostic:
        evaluator.params.useCats = 0
    evaluator.params.maxDets = [1, 10, 1000]
    print(
        f"eval: iou_type={args.iou_type}, class_agnostic={bool(args.class_agnostic)}, "
        f"num_preds={len(preds)}"
    )
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    summary: Dict[str, Any] = {
        "pred": str(Path(args.pred).resolve()),
        "gt": str(Path(args.gt).resolve()),
        "iou_type": str(args.iou_type),
        "class_agnostic": bool(args.class_agnostic),
        "num_preds": len(preds),
        "overall_stats": [float(value) for value in evaluator.stats.tolist()],
    }

    if not args.class_agnostic:
        per_class = _build_per_class_metrics(evaluator, coco_gt)
        summary["per_class"] = per_class
        print("\nPer-class metrics:")
        print(f"{'category_id':>12}  {'category_name':<20}  {'AP':>10}  {'AP50':>10}  {'AP75':>10}  {'AR':>10}")
        for row in per_class:
            print(
                f"{row['category_id']:>12}  "
                f"{row['category_name']:<20.20}  "
                f"{_format_metric(float(row['ap'])):>10}  "
                f"{_format_metric(float(row['ap50'])):>10}  "
                f"{_format_metric(float(row['ap75'])):>10}  "
                f"{_format_metric(float(row['ar'])):>10}"
            )
    else:
        print("\nPer-class metrics skipped because class_agnostic=true.")

    if str(args.out_json).strip():
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSummary JSON: {out_path.resolve()}")


if __name__ == "__main__":
    main()
