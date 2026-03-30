#!/usr/bin/env python
"""Count per-category instance totals from a COCO-style annotation file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT_JSON = PROJECT_ROOT.parent / "data" / "cdw_classify" / "dataset_seg" / "annotations" / "instances_test.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count instances for each category in a COCO annotation json.")
    parser.add_argument(
        "--gt-json",
        default=r'..\data\cdw_classify\dataset_seg\annotations\instances_train.json',
        help="Path to a COCO-style annotation json file.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("count", "id", "name"),
        default="count",
        help="How to sort the output rows.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Also print categories defined in categories[] but with zero instances.",
    )
    return parser.parse_args()


def _load_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Annotation payload must be a JSON object.")
    return payload


def _build_rows(payload: Dict[str, Any], include_empty: bool) -> Tuple[List[Dict[str, Any]], int]:
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    if not isinstance(annotations, list):
        raise ValueError("annotations must be a list.")
    if not isinstance(categories, list):
        categories = []

    category_names: Dict[int, str] = {}
    for category in categories:
        if not isinstance(category, dict):
            continue
        try:
            category_id = int(category.get("id", -1))
        except Exception:
            continue
        category_names[category_id] = str(category.get("name", f"class_{category_id}"))

    counts: Counter[int] = Counter()
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        try:
            category_id = int(ann.get("category_id", -1))
        except Exception:
            category_id = -1
        counts[category_id] += 1

    visible_ids = set(counts.keys())
    if include_empty:
        visible_ids.update(category_names.keys())

    rows: List[Dict[str, Any]] = []
    for category_id in visible_ids:
        rows.append(
            {
                "category_id": int(category_id),
                "category_name": category_names.get(int(category_id), f"class_{int(category_id)}"),
                "instance_count": int(counts.get(int(category_id), 0)),
            }
        )
    return rows, len(annotations)


def _sort_rows(rows: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
    if sort_by == "id":
        return sorted(rows, key=lambda row: int(row["category_id"]))
    if sort_by == "name":
        return sorted(rows, key=lambda row: (str(row["category_name"]).lower(), int(row["category_id"])))
    return sorted(rows, key=lambda row: (-int(row["instance_count"]), int(row["category_id"])))


def _print_rows(rows: List[Dict[str, Any]], total_instances: int, gt_json: Path) -> None:
    print(f"annotation_file: {gt_json}")
    print(f"total_instances: {total_instances}")
    print(f"num_categories_shown: {len(rows)}")
    print("")

    if not rows:
        print("No category statistics found.")
        return

    id_width = max(len("category_id"), max(len(str(row["category_id"])) for row in rows))
    name_width = max(len("category_name"), max(len(str(row["category_name"])) for row in rows))
    count_width = max(len("instance_count"), max(len(str(row["instance_count"])) for row in rows))

    header = (
        f"{'category_id':>{id_width}}  "
        f"{'category_name':<{name_width}}  "
        f"{'instance_count':>{count_width}}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{int(row['category_id']):>{id_width}}  "
            f"{str(row['category_name']):<{name_width}}  "
            f"{int(row['instance_count']):>{count_width}}"
        )


def main() -> None:
    args = parse_args()
    gt_json = Path(args.gt_json).expanduser().resolve()
    payload = _load_payload(gt_json)
    rows, total_instances = _build_rows(payload, include_empty=bool(args.include_empty))
    rows = _sort_rows(rows, args.sort_by)
    _print_rows(rows, total_instances=total_instances, gt_json=gt_json)


if __name__ == "__main__":
    main()
