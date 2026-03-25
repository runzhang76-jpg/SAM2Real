"""Build Ultralytics-style segmentation label files from COCO annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from sam2real.utils.paths import PROJECT_ROOT, resolve_project_path


DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / "data" / "cdw_classify" / "dataset_seg"
DEFAULT_SPLITS: Dict[str, Dict[str, str]] = {
    "train": {
        "annotation": "annotations/instances_train.json",
        "images_dir": "images/train",
    },
    "val": {
        "annotation": "annotations/instances_val.json",
        "images_dir": "images/val",
    },
    "test": {
        "annotation": "annotations/instances_test.json",
        "images_dir": "images/test",
    },
    "pseudo": {
        "annotation": "pseudolabels/pseudolabels.json",
        "images_dir": "images/pseudo",
    },
}


def _sanitize_name(name: str) -> str:
    return name.strip().replace(" ", "_")


def _load_payload(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_category_mapping(payloads: Iterable[Dict[str, Any]]) -> Tuple[Dict[int, int], Dict[int, str]]:
    categories: Dict[int, str] = {}
    for payload in payloads:
        for category in payload.get("categories", []):
            category_id = int(category.get("id", -1))
            if category_id <= 0:
                continue
            raw_name = str(category.get("name", f"class_{category_id}"))
            categories[category_id] = _sanitize_name(raw_name)
    if not categories:
        raise ValueError("No foreground categories found in payloads.")
    sorted_ids = sorted(categories.keys())
    category_id_to_index = {category_id: index for index, category_id in enumerate(sorted_ids)}
    index_to_name = {category_id_to_index[category_id]: categories[category_id] for category_id in sorted_ids}
    return category_id_to_index, index_to_name


def _decode_compressed_rle_counts(text: str) -> List[int]:
    counts: List[int] = []
    index = 0
    while index < len(text):
        value = 0
        shift = 0
        while True:
            char = ord(text[index]) - 48
            index += 1
            value |= (char & 0x1F) << shift
            if (char & 0x20) == 0:
                if char & 0x10:
                    value |= -1 << (shift + 5)
                break
            shift += 5
        if len(counts) > 2:
            value += counts[-2]
        counts.append(int(value))
    return counts


def _decode_rle(segmentation: Dict[str, Any]) -> np.ndarray:
    size = segmentation.get("size", [])
    if len(size) != 2:
        raise ValueError(f"Invalid RLE size: {size}")
    height, width = int(size[0]), int(size[1])
    counts_raw = segmentation.get("counts")
    if isinstance(counts_raw, str):
        counts = _decode_compressed_rle_counts(counts_raw)
    elif isinstance(counts_raw, list):
        counts = [int(value) for value in counts_raw]
    else:
        raise ValueError(f"Unsupported RLE counts type: {type(counts_raw).__name__}")

    flat = np.zeros(height * width, dtype=np.uint8)
    start = 0
    value = 0
    for count in counts:
        end = start + int(count)
        if value == 1:
            flat[start:end] = 1
        start = end
        value = 1 - value
    if start != flat.size:
        raise ValueError(f"Decoded RLE size mismatch: got {start}, expected {flat.size}")
    return flat.reshape((height, width), order="F")


def _decode_polygons(segmentation: List[Any], height: int, width: int) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Polygon decoding requires Pillow.") from exc

    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue
        coords = [(float(polygon[idx]), float(polygon[idx + 1])) for idx in range(0, len(polygon), 2)]
        draw.polygon(coords, outline=1, fill=1)
    return np.asarray(image, dtype=np.uint8)


def _decode_segmentation(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
        mask = _decode_rle(segmentation)
    elif isinstance(segmentation, list):
        mask = _decode_polygons(segmentation, height, width)
    else:
        raise ValueError(f"Unsupported segmentation format: {type(segmentation).__name__}")

    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return np.asarray(mask, dtype=np.uint8)


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _mask_to_polygons(mask: np.ndarray, min_points: int, min_area: float) -> List[np.ndarray]:
    mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))

    try:
        import cv2  # type: ignore

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons: List[np.ndarray] = []
        for contour in contours:
            pts = contour.reshape(-1, 2).astype(np.float32)
            if pts.shape[0] < min_points:
                continue
            if float(cv2.contourArea(contour)) < min_area:
                continue
            polygons.append(pts)
        return polygons
    except Exception:
        try:
            from skimage import measure  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("mask contour extraction requires opencv-python or scikit-image") from exc

        polygons = []
        for contour in measure.find_contours(mask_u8, level=0.5):
            pts = np.flip(contour, axis=1).astype(np.float32)
            if pts.shape[0] < min_points:
                continue
            if _polygon_area(pts) < min_area:
                continue
            polygons.append(pts)
        return polygons


def _normalize_polygon(points: np.ndarray, width: int, height: int) -> List[float]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: width={width}, height={height}")
    normalized = points.astype(np.float32, copy=True)
    normalized[:, 0] = np.clip(normalized[:, 0] / float(width), 0.0, 1.0)
    normalized[:, 1] = np.clip(normalized[:, 1] / float(height), 0.0, 1.0)
    return normalized.reshape(-1).tolist()


def _annotation_to_rows(
    annotation: Dict[str, Any],
    image_info: Dict[str, Any],
    category_id_to_index: Dict[int, int],
    min_points: int,
    min_area: float,
) -> List[str]:
    category_id = int(annotation.get("category_id", -1))
    if category_id not in category_id_to_index:
        return []

    height = int(image_info.get("height", 0))
    width = int(image_info.get("width", 0))
    segmentation = annotation.get("segmentation")
    if segmentation is None:
        return []

    mask = _decode_segmentation(segmentation, height, width)
    polygons = _mask_to_polygons(mask, min_points=min_points, min_area=min_area)
    if not polygons:
        return []

    class_index = category_id_to_index[category_id]
    rows: List[str] = []
    for polygon in polygons:
        normalized = _normalize_polygon(polygon, width=width, height=height)
        if len(normalized) < min_points * 2:
            continue
        values = " ".join(f"{value:.6f}" for value in normalized)
        rows.append(f"{class_index} {values}")
    return rows


def _label_path(labels_root: Path, split_name: str, file_name: str) -> Path:
    relative = Path(file_name).with_suffix(".txt")
    return labels_root / split_name / relative


def _clear_split_dir(split_dir: Path) -> None:
    if not split_dir.exists():
        return
    for path in sorted(split_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def convert_split(
    *,
    split_name: str,
    annotation_path: Path,
    images_dir: Path,
    labels_root: Path,
    category_id_to_index: Dict[int, int],
    min_points: int,
    min_area: float,
    clear_existing: bool,
) -> Dict[str, int]:
    payload = _load_payload(annotation_path)
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    image_by_id = {int(image["id"]): image for image in images}
    rows_by_image: Dict[int, List[str]] = {int(image["id"]): [] for image in images}

    split_dir = labels_root / split_name
    if clear_existing:
        _clear_split_dir(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    label_rows = 0
    for annotation in annotations:
        image_id = int(annotation.get("image_id", -1))
        image_info = image_by_id.get(image_id)
        if image_info is None:
            skipped += 1
            continue
        rows = _annotation_to_rows(
            annotation,
            image_info,
            category_id_to_index=category_id_to_index,
            min_points=min_points,
            min_area=min_area,
        )
        if not rows:
            skipped += 1
            continue
        rows_by_image[image_id].extend(rows)
        label_rows += len(rows)

    empty_files = 0
    missing_images = 0
    for image_id, image_info in image_by_id.items():
        file_name = str(image_info.get("file_name", ""))
        image_path = images_dir / file_name
        if not image_path.exists():
            missing_images += 1
        output_path = _label_path(labels_root, split_name, file_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = rows_by_image.get(image_id, [])
        if not rows:
            empty_files += 1
        output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    return {
        "images": len(images),
        "annotations": len(annotations),
        "label_files": len(image_by_id),
        "label_rows": label_rows,
        "empty_files": empty_files,
        "missing_images": missing_images,
        "skipped_annotations": skipped,
    }


def build_yolo_labels(
    dataset_root: Path,
    *,
    labels_dir_name: str = "labels",
    clear_existing: bool = False,
    min_points: int = 3,
    min_area: float = 4.0,
) -> Dict[str, Any]:
    dataset_root = Path(resolve_project_path(dataset_root))
    labels_root = dataset_root / labels_dir_name

    payloads = [_load_payload(dataset_root / split_cfg["annotation"]) for split_cfg in DEFAULT_SPLITS.values()]
    category_id_to_index, index_to_name = _build_category_mapping(payloads)

    summary: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "labels_root": str(labels_root),
        "class_id_map": category_id_to_index,
        "names": [index_to_name[index] for index in sorted(index_to_name.keys())],
        "splits": {},
    }

    for split_name, split_cfg in DEFAULT_SPLITS.items():
        split_summary = convert_split(
            split_name=split_name,
            annotation_path=dataset_root / split_cfg["annotation"],
            images_dir=dataset_root / split_cfg["images_dir"],
            labels_root=labels_root,
            category_id_to_index=category_id_to_index,
            min_points=min_points,
            min_area=min_area,
            clear_existing=clear_existing,
        )
        summary["splits"][split_name] = split_summary

    summary_path = labels_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert COCO annotations and pseudolabels into Ultralytics label txt files.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Dataset root containing annotations/, pseudolabels/, images/.")
    parser.add_argument("--labels-dir-name", default="labels", help="Output labels directory name under dataset root.")
    parser.add_argument("--clear-existing", action="store_true", help="Delete existing split label files before writing.")
    parser.add_argument("--min-points", type=int, default=3, help="Minimum polygon vertices kept per instance.")
    parser.add_argument("--min-area", type=float, default=4.0, help="Minimum polygon area in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_yolo_labels(
        Path(args.dataset_root),
        labels_dir_name=str(args.labels_dir_name),
        clear_existing=bool(args.clear_existing),
        min_points=int(args.min_points),
        min_area=float(args.min_area),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


__all__ = ["build_yolo_labels", "main", "parse_args"]
