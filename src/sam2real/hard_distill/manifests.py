"""Manifest and data-yaml helpers for hard-label distillation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from sam2real.hard_distill.shot_sampler import DEFAULT_DATASET_ROOT
from sam2real.utils.paths import PROJECT_ROOT, resolve_project_path


DEFAULT_GENERATED_ROOT = PROJECT_ROOT / "configs" / "hard_distill" / "generated"
DEFAULT_MANIFESTS_ROOT = DEFAULT_GENERATED_ROOT / "manifests"


def resolve_dataset_root(dataset_root: Path | str) -> Path:
    return Path(resolve_project_path(dataset_root or DEFAULT_DATASET_ROOT))


def _load_label_summary(dataset_root: Path) -> Dict[str, Any]:
    summary_path = dataset_root / "labels" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Label summary not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_class_names(dataset_root: Path) -> List[str]:
    summary = _load_label_summary(dataset_root)
    names = summary.get("names", [])
    if not isinstance(names, list) or not names:
        raise ValueError(f"Invalid names list in {dataset_root / 'labels' / 'summary.json'}")
    return [str(name) for name in names]


def _iter_split_images(images_dir: Path) -> List[Path]:
    return sorted(
        [
            path.resolve()
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )


def ensure_split_manifest(dataset_root: Path, split_name: str, manifests_root: Path = DEFAULT_MANIFESTS_ROOT) -> Path:
    dataset_root = resolve_dataset_root(dataset_root)
    manifests_root.mkdir(parents=True, exist_ok=True)
    images_dir = dataset_root / "images" / split_name
    if not images_dir.exists():
        raise FileNotFoundError(f"Split images directory not found: {images_dir}")
    manifest_path = manifests_root / f"{split_name}.txt"
    image_paths = _iter_split_images(images_dir)
    manifest_path.write_text("\n".join(str(path) for path in image_paths) + ("\n" if image_paths else ""), encoding="utf-8")
    return manifest_path


def write_manifest(image_paths: Iterable[Path], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_paths = [str(Path(path).resolve()) for path in image_paths]
    output_path.write_text("\n".join(resolved_paths) + ("\n" if resolved_paths else ""), encoding="utf-8")
    return output_path


def write_data_yaml(
    *,
    output_path: Path,
    dataset_root: Path,
    train: str,
    val: str,
    test: str,
    names: List[str],
) -> Path:
    payload = {
        "path": str(dataset_root.resolve()),
        "train": str(train),
        "val": str(val),
        "test": str(test),
        "names": {index: name for index, name in enumerate(names)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return output_path


def ensure_base_manifests(dataset_root: Path, manifests_root: Path = DEFAULT_MANIFESTS_ROOT) -> Dict[str, Path]:
    dataset_root = resolve_dataset_root(dataset_root)
    outputs: Dict[str, Path] = {}
    for split_name in ("train", "val", "test", "pseudo"):
        outputs[split_name] = ensure_split_manifest(dataset_root, split_name, manifests_root=manifests_root)
    return outputs


def build_pseudo_stage_yaml(
    dataset_root: Path,
    output_path: Path,
    manifests_root: Path = DEFAULT_MANIFESTS_ROOT,
) -> Path:
    dataset_root = resolve_dataset_root(dataset_root)
    manifests = ensure_base_manifests(dataset_root, manifests_root=manifests_root)
    names = load_class_names(dataset_root)
    return write_data_yaml(
        output_path=output_path,
        dataset_root=dataset_root,
        train=str(manifests["pseudo"]),
        val=str(manifests["val"]),
        test=str(manifests["test"]),
        names=names,
    )


__all__ = [
    "DEFAULT_GENERATED_ROOT",
    "DEFAULT_MANIFESTS_ROOT",
    "build_pseudo_stage_yaml",
    "ensure_base_manifests",
    "ensure_split_manifest",
    "load_class_names",
    "resolve_dataset_root",
    "write_data_yaml",
    "write_manifest",
]
