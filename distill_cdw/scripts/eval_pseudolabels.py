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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO eval for pseudolabels")
    parser.add_argument(
        "--pred",
        default="../data/sam2/dataset/segment_cdw_coco_dataset_t/pseudolabels/pseudolabels_results.json",
        help="COCO results JSON path",
    )
    parser.add_argument(
        "--gt",
        default="../data/sam2/dataset/segment_cdw_coco_dataset_t/annotations/instances_test_c.json",
        help="COCO GT JSON path",
    )
    parser.add_argument("--iou-type", default="segm", help="segm or bbox")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_payload = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    preds = pred_payload if isinstance(pred_payload, list) else pred_payload.get("results", pred_payload.get("annotations", []))
    coco_gt = COCO(args.gt)
    coco_dt = coco_gt.loadRes(preds)
    evaluator = COCOeval(coco_gt, coco_dt, iouType=args.iou_type)
    evaluator.params.maxDets = [1, 10, 200]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()


if __name__ == "__main__":
    main()
