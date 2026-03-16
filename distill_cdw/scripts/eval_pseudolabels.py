#!/usr/bin/env python
"""COCO 评估：结果文件 vs GT 文件。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pycocotools.coco import COCO  # type: ignore
from pycocotools.cocoeval import COCOeval  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PKG_ROOT = ROOT / "distill_cdw"
if PKG_ROOT.exists() and str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))


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
    return parser.parse_args()


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


if __name__ == "__main__":
    main()
