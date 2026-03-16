#!/usr/bin/env python
"""Run DINOv3 prototype classifier on GT object crops for debugging."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PKG_ROOT = ROOT / "distill_cdw"
if PKG_ROOT.exists() and str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from distill.teacher.dinov3_prototype_classifier import DinoV3PrototypeClassifier
from distill.utils.config import load_config
from distill.utils.logging import setup_logger

try:
    from pycocotools import mask as mask_utils  # type: ignore
except Exception:
    mask_utils = None  # type: ignore


CategoryMap = Dict[int, Union[int, str]]


def _str2bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def _try_int(value: Any) -> Optional[int]:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _load_category_map(path: Optional[Any]) -> Optional[CategoryMap]:
    if not path:
        return None
    if isinstance(path, dict):
        mapping: CategoryMap = {}
        for key, value in path.items():
            k = _try_int(key)
            if k is None:
                continue
            v_int = _try_int(value)
            if v_int is not None:
                mapping[k] = v_int
            elif isinstance(value, str):
                mapping[k] = value
        return mapping or None
    return None


def _square_crop_coords(
    img_hw: Tuple[int, int],
    bbox_xyxy: Tuple[float, float, float, float],
    margin_ratio: float,
) -> Tuple[int, int, int, int]:
    height, width = img_hw
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw, bh) * (1.0 + 2.0 * margin_ratio)
    half = side * 0.5
    x1n = int(max(0, np.floor(cx - half)))
    y1n = int(max(0, np.floor(cy - half)))
    x2n = int(min(width, np.ceil(cx + half)))
    y2n = int(min(height, np.ceil(cy + half)))
    return x1n, y1n, x2n, y2n


def _decode_segmentation(seg: Any, height: int, width: int) -> Optional[np.ndarray]:
    if seg is None or mask_utils is None:
        return None
    if isinstance(seg, dict) and "counts" in seg and "size" in seg:
        m = mask_utils.decode(seg)
        if m.ndim == 3:
            m = m[:, :, 0]
        return np.asarray(m, dtype=bool)
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, height, width)
        m = mask_utils.decode(rles)
        if m.ndim == 3:
            m = m.any(axis=2)
        return np.asarray(m, dtype=bool)
    return None


def _crop_patch(
    image: np.ndarray,
    bbox_xywh: List[float],
    mask: Optional[np.ndarray],
    margin_ratio: float,
) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox_xywh]
    x1n, y1n, x2n, y2n = _square_crop_coords(
        img_hw=(image.shape[0], image.shape[1]),
        bbox_xyxy=(x, y, x + w, y + h),
        margin_ratio=margin_ratio,
    )
    patch = image[y1n:y2n, x1n:x2n].copy()
    if patch.size == 0:
        return patch
    if mask is not None:
        m = mask[y1n:y2n, x1n:x2n]
        if m.dtype != bool:
            m = m.astype(bool)
        if m.size == 0:
            return np.zeros((0, 0, image.shape[2]), dtype=image.dtype)
        patch[~m] = 0
    return patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug DINOv3 prototype classification with GT object boxes")
    parser.add_argument("--config", default="distill_cdw/configs/distill_default.yaml")
    parser.add_argument("--gt-json", default="../data/cdw_classify/dataset_seg/annotations/instances_test.json")
    parser.add_argument("--images-root", default="../data/cdw_classify/dataset_seg/images/test")
    parser.add_argument("--use-mask", type=_str2bool, default=True, help="Mask-out background using GT segmentation")
    parser.add_argument("--margin-ratio", type=float, default=0, help="Crop margin; <0 uses config classifier.margin_ratio")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of GT instances")
    parser.add_argument("--out-json", default="outputs/gt_prototype_predictions.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("distill")

    cfg = load_config(args.config)
    cls_cfg = cfg.get("teacher", {}).get("classifier", {})
    proto_cfg = dict(cls_cfg)
    proto_cfg.update(cls_cfg.get("dinov3_prototype", {}))

    margin_ratio = float(cls_cfg.get("margin_ratio", 0.1)) if args.margin_ratio < 0 else float(args.margin_ratio)
    category_map = _load_category_map(cls_cfg.get("category_map"))

    if str(cls_cfg.get("type", "")).lower() not in {"prototype_similarity", "dinov3_prototype"}:
        logger.warning(
            "teacher.classifier.type is not prototype_similarity/dinov3_prototype; script still uses dinov3_prototype config fields"
        )

    classifier = DinoV3PrototypeClassifier(cfg=proto_cfg, logger=logger)

    gt = json.loads(Path(args.gt_json).read_text(encoding="utf-8"))
    images = gt.get("images", [])
    anns = gt.get("annotations", [])
    img_by_id = {int(x["id"]): x for x in images}
    cat_name_by_id = {int(c["id"]): str(c.get("name", c["id"])) for c in gt.get("categories", [])}

    if args.limit > 0:
        anns = anns[: args.limit]

    image_cache: Dict[int, np.ndarray] = {}
    patches: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    skipped = 0

    for ann in anns:
        image_id = int(ann.get("image_id", -1))
        info = img_by_id.get(image_id, None)
        if info is None:
            skipped += 1
            continue

        if image_id not in image_cache:
            file_name = str(info.get("file_name", ""))
            image_path = Path(args.images_root) / file_name
            if not image_path.exists():
                logger.warning("image not found: %s", image_path)
                skipped += 1
                continue
            image_cache[image_id] = np.array(Image.open(image_path).convert("RGB"), copy=True)
        image_np = image_cache[image_id]

        bbox = ann.get("bbox", None)
        if not isinstance(bbox, list) or len(bbox) != 4:
            skipped += 1
            continue

        gt_cat = int(ann.get("category_id", -1))
        mask = None
        if args.use_mask:
            mask = _decode_segmentation(
                ann.get("segmentation", None),
                height=int(info.get("height", image_np.shape[0])),
                width=int(info.get("width", image_np.shape[1])),
            )
        patch = _crop_patch(image=image_np, bbox_xywh=bbox, mask=mask, margin_ratio=margin_ratio)
        if patch.size == 0:
            skipped += 1
            continue

        patches.append(patch)
        rows.append(
            {
                "ann_id": int(ann.get("id", -1)),
                "image_id": image_id,
                "file_name": str(info.get("file_name", "")),
                "gt_category_id": gt_cat,
                "gt_category_name": cat_name_by_id.get(gt_cat, str(gt_cat)),
                "bbox": bbox,
            }
        )

    preds = classifier.predict_patches(patches)
    if len(preds) != len(rows):
        raise RuntimeError(f"prediction size mismatch: preds={len(preds)} rows={len(rows)}")

    confusion: Counter = Counter()
    correct = 0
    for row, pred in zip(rows, preds):
        raw_pred = pred.get("category_id")
        pred_int = _try_int(raw_pred)
        mapped = category_map.get(pred_int, pred_int) if (pred_int is not None and category_map is not None) else pred_int
        mapped_int = _try_int(mapped) if mapped is not None else None

        final_pred = mapped_int if mapped_int is not None else pred_int
        score = float(pred.get("category_score", 0.0))
        similarity = float(pred.get("prototype_similarity", 0.0))
        is_correct = (final_pred is not None and int(final_pred) == int(row["gt_category_id"]))

        row["pred_category_id_raw"] = raw_pred
        row["pred_category_id_mapped"] = final_pred
        row["pred_category_score"] = score
        row["prototype_similarity"] = similarity
        row["prototype_index"] = int(pred.get("prototype_index", -1))
        row["correct"] = bool(is_correct)

        if final_pred is not None:
            confusion[(int(row["gt_category_id"]), int(final_pred))] += 1
        if is_correct:
            correct += 1

    total = len(rows)
    acc = correct / total if total > 0 else 0.0
    logger.info(
        "GT->Prototype done: total=%d skipped=%d correct=%d acc=%.4f use_mask=%s margin_ratio=%.3f",
        total,
        skipped,
        correct,
        acc,
        bool(args.use_mask),
        margin_ratio,
    )

    per_class_total: Dict[int, int] = defaultdict(int)
    per_class_correct: Dict[int, int] = defaultdict(int)
    for r in rows:
        c = int(r["gt_category_id"])
        per_class_total[c] += 1
        if r["correct"]:
            per_class_correct[c] += 1
    for c in sorted(per_class_total.keys()):
        t = per_class_total[c]
        cc = per_class_correct[c]
        logger.info(
            "class=%s(%d): total=%d correct=%d acc=%.4f",
            cat_name_by_id.get(c, str(c)),
            c,
            t,
            cc,
            (cc / t) if t > 0 else 0.0,
        )

    out = {
        "summary": {
            "total": total,
            "skipped": skipped,
            "correct": correct,
            "accuracy": acc,
            "use_mask": bool(args.use_mask),
            "margin_ratio": margin_ratio,
        },
        "per_instance": rows,
        "confusion": [
            {"gt_category_id": g, "pred_category_id": p, "count": c}
            for (g, p), c in sorted(confusion.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        ],
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("saved debug result to %s", out_path)


if __name__ == "__main__":
    main()
