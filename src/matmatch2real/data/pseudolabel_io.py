""" IO JSON/NPZ/COCO-like"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from matmatch2real.core.structures import PseudoLabelInstance
from matmatch2real.utils.logging import get_logger

from pycocotools import mask as mask_utils
_HAS_COCO = True


def _encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    if not _HAS_COCO:
        return {}
    m = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(m)
    if isinstance(rle.get("counts", None), bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _mask_to_segmentation(mask: np.ndarray) -> Dict[str, Any]:
    if _HAS_COCO:
        return _encode_rle(mask)
    return {"format": "bitmap", "mask": mask.astype(np.uint8).tolist()}


def decode_segmentation(segmentation: Any, height: int, width: int) -> Optional[np.ndarray]:
    """Decode COCO-style segmentation into a binary mask."""
    if segmentation is None:
        return None
    if isinstance(segmentation, np.ndarray):
        mask = np.asarray(segmentation)
        if mask.ndim != 2:
            return None
        return (mask > 0).astype(np.uint8)
    if isinstance(segmentation, dict):
        if _HAS_COCO and "counts" in segmentation and "size" in segmentation:
            try:
                mask = mask_utils.decode(segmentation)
            except Exception:
                return None
            if mask.ndim == 3:
                mask = mask[..., 0]
            return (np.asarray(mask) > 0).astype(np.uint8)
        if segmentation.get("format") == "bitmap":
            bitmap = segmentation.get("mask")
            if bitmap is None:
                return None
            mask = np.asarray(bitmap, dtype=np.uint8)
            if mask.ndim != 2:
                return None
            return (mask > 0).astype(np.uint8)
        return None
    if isinstance(segmentation, list):
        if not _HAS_COCO:
            return None
        try:
            rles = mask_utils.frPyObjects(segmentation, int(height), int(width))
            merged = mask_utils.merge(rles)
            mask = mask_utils.decode(merged)
        except Exception:
            return None
        if mask.ndim == 3:
            mask = mask[..., 0]
        return (np.asarray(mask) > 0).astype(np.uint8)
    return None


def _normalize_npz_payload(payload: Any) -> Any:
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    return payload


def read_pseudolabels(path: str) -> Tuple[List[Dict[str, Any]], Dict[int, List[PseudoLabelInstance]], Dict[str, Any]]:
    """ JSON  NPZ """

    logger = get_logger("distill")
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Pseudo label file not found: {path}")

    if file_path.suffix.lower() == ".npz":
        data = np.load(file_path, allow_pickle=True)
        images = _normalize_npz_payload(data.get("images", []))
        annotations = _normalize_npz_payload(data.get("annotations", []))
        meta = _normalize_npz_payload(data.get("meta", {}))
    else:
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        meta = payload.get("meta", {})

    instances_by_image: Dict[int, List[PseudoLabelInstance]] = {}
    for ann in annotations:
        try:
            inst = PseudoLabelInstance.from_dict(ann)
        except Exception as exc:
            logger.warning("skip invalid annotation: %s", exc)
            continue
        instances_by_image.setdefault(inst.image_id, []).append(inst)

    logger.info("loaded pseudo labels: images=%d instances=%d", len(images), len(annotations))
    return images, instances_by_image, meta


def write_pseudolabels(
    path: str,
    images: List[Dict[str, Any]],
    instances_by_image: Dict[int, List[PseudoLabelInstance]],
    meta: Dict[str, Any],
) -> None:
    """ JSON  NPZ"""

    annotations: List[Dict[str, Any]] = []
    for insts in instances_by_image.values():
        for inst in insts:
            ann = inst.to_dict()
            if inst.rle is not None:
                ann["segmentation"] = inst.rle
            elif inst.mask is not None:
                mask = inst.mask
                if not isinstance(mask, np.ndarray):
                    mask = np.asarray(mask)
                ann["segmentation"] = _mask_to_segmentation(mask)
            annotations.append(ann)

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [],
        "meta": meta,
    }

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.suffix.lower() == ".npz":
        np.savez_compressed(
            file_path,
            images=np.array(images, dtype=object),
            annotations=np.array(annotations, dtype=object),
            categories=np.array([], dtype=object),
            meta=np.array(meta, dtype=object),
        )
        return

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
