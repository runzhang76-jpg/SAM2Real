#!/usr/bin/env python
"""将伪标签实例类别统一改为指定值。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Force all pseudolabel categories to a fixed id")
    parser.add_argument(
        "--input",
        default="../data/sam2/dataset/segment_cdw_coco_dataset_t/pseudolabels/pseudolabels_results.json",
        help="Input pseudolabel JSON path",
    )
    parser.add_argument(
        "--output",
        default="../data/sam2/dataset/segment_cdw_coco_dataset_t/pseudolabels/pseudolabels_results_sc.json",
        help="Output JSON path",
    )
    parser.add_argument("--class-id", type=int, default=1, help="Target category id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        anns = payload
    else:
        anns = payload.get("annotations", payload.get("results", []))
    for ann in anns:
        ann["category_id"] = int(args.class_id)
        if "class_id" in ann:
            ann["class_id"] = int(args.class_id)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
