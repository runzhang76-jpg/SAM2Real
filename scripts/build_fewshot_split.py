#!/usr/bin/env python
"""Generate few-shot manifests and data YAML for stage-2 fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sam2real.hard_distill.manifests import (
    DEFAULT_GENERATED_ROOT,
    ensure_base_manifests,
    load_class_names,
    resolve_dataset_root,
    write_data_yaml,
    write_manifest,
)
from sam2real.hard_distill.shot_sampler import (
    DEFAULT_DATASET_ROOT,
    sample_image_count_records,
    sample_k_shot_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build few-shot manifests for hard-distill stage-2 fine-tuning.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Dataset root.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--shot", type=int, help="K images per class.")
    mode_group.add_argument("--image-count", type=int, help="Total number of training images to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--generated-root",
        default=str(DEFAULT_GENERATED_ROOT),
        help="Directory used to store manifests, YAML, and summary files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(Path(args.dataset_root))
    generated_root = Path(args.generated_root).resolve()
    manifests_root = generated_root / "manifests"
    ensure_base_manifests(dataset_root, manifests_root=manifests_root)

    if args.shot is not None:
        sample = sample_k_shot_records(dataset_root, shot=int(args.shot), seed=int(args.seed))
        tag = f"shot_{int(args.shot)}_seed_{int(args.seed)}"
    else:
        sample = sample_image_count_records(dataset_root, image_count=int(args.image_count), seed=int(args.seed))
        tag = f"images_{int(args.image_count)}_seed_{int(args.seed)}"

    records = sample["records"]
    summary = dict(sample["summary"])

    train_manifest_path = write_manifest(
        [record.image_path for record in records],
        manifests_root / f"{tag}.txt",
    )

    names = load_class_names(dataset_root)
    data_yaml_path = write_data_yaml(
        output_path=generated_root / f"{tag}.yaml",
        dataset_root=dataset_root,
        train=str(train_manifest_path),
        val=str(manifests_root / "val.txt"),
        test=str(manifests_root / "test.txt"),
        names=names,
    )

    summary.update(
        {
            "train_manifest": str(train_manifest_path),
            "data_yaml": str(data_yaml_path),
        }
    )
    summary_path = generated_root / f"{tag}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
