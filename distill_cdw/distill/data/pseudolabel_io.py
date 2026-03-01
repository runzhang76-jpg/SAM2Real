"""伪标签 IO 工具（JSON/NPZ/COCO-like）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from distill.core.structures import PseudoLabelInstance
from distill.utils.logging import get_logger

try:
    from pycocotools import mask as mask_utils

    _HAS_COCO = True
except Exception:  # pragma: no cover - 可选依赖
    _HAS_COCO = False


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


def _normalize_npz_payload(payload: Any) -> Any:
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    return payload


def read_pseudolabels(path: str) -> Tuple[List[Dict[str, Any]], Dict[int, List[PseudoLabelInstance]], Dict[str, Any]]:
    """从 JSON 或 NPZ 读取伪标签。"""

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
    """将伪标签写入 JSON 或 NPZ。"""

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
