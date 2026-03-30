#!/usr/bin/env python
"""Run model classifier branch on GT object crops for debugging."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matmatch2real.core.structures import InstancePrediction
from matmatch2real.teacher.classifier_adapter import ClassifierAdapter
from matmatch2real.config.loader import load_config
from matmatch2real.utils.logging import setup_logger

try:
    from pycocotools import mask as mask_utils  # type: ignore
except Exception:
    mask_utils = None  # type: ignore


def _str2bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug model classifier using GT boxes")
    parser.add_argument("--config", default="matmatch2real-main/configs/teacher/distill_default.yaml")
    parser.add_argument("--gt-json", default="")
    parser.add_argument("--images-root", default="")
    parser.add_argument("--use-mask", type=_str2bool, default=True, help="Mask-out background using GT segmentation")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of GT instances")
    parser.add_argument(
        "--gt-id-offset",
        type=int,
        default=0,
        help="Offset added to GT category id before comparison (e.g. -1 for 1-based GT vs 0-based model)",
    )
    parser.add_argument("--out-json", default="outputs/gt_model_predictions.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("distill")

    cfg = load_config(args.config)
    cls_cfg = dict(cfg.get("teacher", {}).get("classifier", {}))
    if not bool(cls_cfg.get("enabled", False)):
        raise ValueError("teacher.classifier.enabled is false")
    # Force model backend for this debug script.
    cls_cfg["type"] = "model"

    adapter = ClassifierAdapter(cls_cfg)
    if adapter.classifier is None:
        raise RuntimeError("Failed to initialize classifier model backend")

    gt = json.loads(Path(args.gt_json).read_text(encoding="utf-8"))
    images = gt.get("images", [])
    anns = gt.get("annotations", [])
    img_by_id = {int(x["id"]): x for x in images}
    cat_name_by_id = {int(c["id"]): str(c.get("name", c["id"])) for c in gt.get("categories", [])}

    if args.limit > 0:
        anns = anns[: args.limit]

    anns_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in anns:
        anns_by_image[int(ann.get("image_id", -1))].append(ann)

    rows: List[Dict[str, Any]] = []
    confusion: Counter = Counter()
    per_class_total: Dict[int, int] = defaultdict(int)
    per_class_correct: Dict[int, int] = defaultdict(int)
    skipped = 0
    total = 0
    correct = 0

    for image_id, image_anns in anns_by_image.items():
        info = img_by_id.get(image_id, None)
        if info is None:
            skipped += len(image_anns)
            continue
        file_name = str(info.get("file_name", ""))
        image_path = Path(args.images_root) / file_name
        if not image_path.exists():
            logger.warning("image not found: %s", image_path)
            skipped += len(image_anns)
            continue
        image_np = np.asarray(Image.open(image_path).convert("RGB"))
        h = int(info.get("height", image_np.shape[0]))
        w = int(info.get("width", image_np.shape[1]))

        instances: List[InstancePrediction] = []
        ann_ids: List[int] = []
        gt_ids_raw: List[int] = []
        gt_bboxes: List[List[float]] = []
        for ann in image_anns:
            bbox = ann.get("bbox", None)
            if not isinstance(bbox, list) or len(bbox) != 4:
                skipped += 1
                continue
            gt_cat = int(ann.get("category_id", -1))
            mask = _decode_segmentation(ann.get("segmentation", None), height=h, width=w) if args.use_mask else None
            inst = InstancePrediction(
                image_id=image_id,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                class_id=0,
                score=1.0,
                reliability=1.0,
                mask=mask,
                rle=None,
                meta={"gt_category_id": gt_cat},
            )
            instances.append(inst)
            ann_ids.append(int(ann.get("id", -1)))
            gt_ids_raw.append(gt_cat)
            gt_bboxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])

        if not instances:
            continue
        preds = adapter.classify(instances, image_np=image_np)

        # adapter may filter by score_threshold; use position-safe pairing by count.
        n = min(len(preds), len(ann_ids))
        if len(preds) != len(ann_ids):
            logger.warning(
                "prediction count mismatch on image_id=%d: preds=%d gt=%d",
                image_id,
                len(preds),
                len(ann_ids),
            )
            skipped += abs(len(ann_ids) - len(preds))
        for i in range(n):
            pred = preds[i]
            gt_raw = gt_ids_raw[i]
            gt_eval = int(gt_raw) + int(args.gt_id_offset)
            pred_id = int(pred.class_id)
            score = float(pred.meta.get("category_score", pred.meta.get("cls_prob", pred.score)))
            ok = pred_id == gt_eval

            total += 1
            if ok:
                correct += 1
            confusion[(gt_eval, pred_id)] += 1
            per_class_total[gt_eval] += 1
            if ok:
                per_class_correct[gt_eval] += 1

            rows.append(
                {
                    "ann_id": ann_ids[i],
                    "image_id": image_id,
                    "file_name": file_name,
                    "gt_category_id_raw": gt_raw,
                    "gt_category_id_eval": gt_eval,
                    "gt_category_name": cat_name_by_id.get(gt_raw, str(gt_raw)),
                    "bbox": gt_bboxes[i],
                    "pred_category_id": pred_id,
                    "pred_category_score": score,
                    "correct": bool(ok),
                }
            )

    acc = correct / total if total > 0 else 0.0
    logger.info(
        "GT->Model done: total=%d skipped=%d correct=%d acc=%.4f use_mask=%s gt_id_offset=%d",
        total,
        skipped,
        correct,
        acc,
        bool(args.use_mask),
        int(args.gt_id_offset),
    )
    for c in sorted(per_class_total.keys()):
        t = per_class_total[c]
        cc = per_class_correct[c]
        logger.info("class_eval_id=%d: total=%d correct=%d acc=%.4f", c, t, cc, (cc / t) if t > 0 else 0.0)

    out = {
        "summary": {
            "total": total,
            "skipped": skipped,
            "correct": correct,
            "accuracy": acc,
            "use_mask": bool(args.use_mask),
            "gt_id_offset": int(args.gt_id_offset),
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
