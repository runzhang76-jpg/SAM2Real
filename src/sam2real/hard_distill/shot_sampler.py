"""Few-shot image sampling utilities for hard-label distillation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Dict, List, Sequence

from sam2real.utils.paths import PROJECT_ROOT, resolve_project_path


DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / "data" / "cdw_classify" / "dataset_seg"


@dataclass(frozen=True)
class ImageLabelRecord:
    """One labeled training image and the classes it contains."""

    image_path: Path
    label_path: Path
    stem: str
    classes: tuple[int, ...]


def _iter_label_files(labels_dir: Path) -> List[Path]:
    return sorted([path for path in labels_dir.rglob("*.txt") if path.is_file()])


def _iter_image_files(images_dir: Path) -> Dict[str, Path]:
    image_files: Dict[str, Path] = {}
    for path in sorted(images_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image_files[path.stem] = path
    return image_files


def _parse_label_classes(label_path: Path) -> tuple[int, ...]:
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return tuple()
    classes = set()
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        classes.add(int(float(parts[0])))
    return tuple(sorted(classes))


def load_train_records(dataset_root: Path) -> List[ImageLabelRecord]:
    dataset_root = Path(resolve_project_path(dataset_root))
    labels_dir = dataset_root / "labels" / "val"
    images_dir = dataset_root / "images" / "val"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Training labels directory not found: {labels_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Training images directory not found: {images_dir}")

    image_by_stem = _iter_image_files(images_dir)
    records: List[ImageLabelRecord] = []
    missing_images: List[Path] = []
    for label_path in _iter_label_files(labels_dir):
        classes = _parse_label_classes(label_path)
        if not classes:
            continue
        image_path = image_by_stem.get(label_path.stem)
        if image_path is None:
            missing_images.append(label_path)
            continue
        records.append(
            ImageLabelRecord(
                image_path=image_path.resolve(),
                label_path=label_path.resolve(),
                stem=label_path.stem,
                classes=classes,
            )
        )
    if missing_images:
        missing_preview = ", ".join(str(path) for path in missing_images[:8])
        raise FileNotFoundError(f"Missing matching training images for label files: {missing_preview}")
    if not records:
        raise RuntimeError(f"No valid labeled training records found under {labels_dir}")
    return records


def _build_class_buckets(records: Sequence[ImageLabelRecord]) -> Dict[int, List[ImageLabelRecord]]:
    buckets: Dict[int, List[ImageLabelRecord]] = {}
    for record in records:
        for class_id in record.classes:
            buckets.setdefault(class_id, []).append(record)
    return {class_id: sorted(items, key=lambda item: item.stem) for class_id, items in buckets.items()}


def sample_k_shot_records(dataset_root: Path, shot: int, seed: int = 0) -> Dict[str, Any]:
    if shot <= 0:
        raise ValueError(f"shot must be >= 1, got {shot}")

    records = load_train_records(dataset_root)
    class_buckets = _build_class_buckets(records)
    classes_sorted = sorted(class_buckets.keys(), key=lambda class_id: (len(class_buckets[class_id]), class_id))
    rng = random.Random(int(seed))

    selected_by_stem: Dict[str, ImageLabelRecord] = {}
    class_coverage = {class_id: 0 for class_id in classes_sorted}

    shuffled_buckets: Dict[int, List[ImageLabelRecord]] = {}
    for class_id, items in class_buckets.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        shuffled_buckets[class_id] = shuffled

    for class_id in classes_sorted:
        available = len(class_buckets[class_id])
        if available < shot:
            raise ValueError(
                f"class {class_id} only has {available} labeled training images, cannot satisfy shot={shot}"
            )

        while class_coverage[class_id] < shot:
            remaining = [item for item in shuffled_buckets[class_id] if item.stem not in selected_by_stem]
            if not remaining:
                remaining = shuffled_buckets[class_id]
            chosen = remaining[0]
            if chosen.stem not in selected_by_stem:
                selected_by_stem[chosen.stem] = chosen
                for covered_class in chosen.classes:
                    if covered_class in class_coverage:
                        class_coverage[covered_class] += 1
            else:
                # All candidates are already selected. Coverage should already be saturated unless the
                # requested shot is impossible, which is handled above.
                class_coverage[class_id] = sum(1 for item in selected_by_stem.values() if class_id in item.classes)

    selected_records = sorted(selected_by_stem.values(), key=lambda item: item.stem)
    summary = {
        "dataset_root": str(Path(resolve_project_path(dataset_root))),
        "sampling_mode": "per_class_shot",
        "shot": int(shot),
        "seed": int(seed),
        "selected_images": len(selected_records),
        "selected_stems": [item.stem for item in selected_records],
        "class_stats": {
            str(class_id): {
                "available_images": len(class_buckets[class_id]),
                "selected_images": sum(1 for item in selected_records if class_id in item.classes),
            }
            for class_id in classes_sorted
        },
    }
    return {"records": selected_records, "summary": summary}


def sample_image_count_records(dataset_root: Path, image_count: int, seed: int = 0) -> Dict[str, Any]:
    if image_count <= 0:
        raise ValueError(f"image_count must be >= 1, got {image_count}")

    records = load_train_records(dataset_root)
    if image_count > len(records):
        raise ValueError(
            f"Requested image_count={image_count}, but only {len(records)} labeled training images are available"
        )

    rng = random.Random(int(seed))
    shuffled = list(records)
    rng.shuffle(shuffled)
    selected_records = sorted(shuffled[:image_count], key=lambda item: item.stem)

    class_buckets = _build_class_buckets(records)
    classes_sorted = sorted(class_buckets.keys())
    summary = {
        "dataset_root": str(Path(resolve_project_path(dataset_root))),
        "sampling_mode": "image_count",
        "image_count": int(image_count),
        "seed": int(seed),
        "available_images": len(records),
        "selected_images": len(selected_records),
        "selected_stems": [item.stem for item in selected_records],
        "class_stats": {
            str(class_id): {
                "available_images": len(class_buckets[class_id]),
                "selected_images": sum(1 for item in selected_records if class_id in item.classes),
            }
            for class_id in classes_sorted
        },
    }
    return {"records": selected_records, "summary": summary}


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "ImageLabelRecord",
    "load_train_records",
    "sample_image_count_records",
    "sample_k_shot_records",
]
