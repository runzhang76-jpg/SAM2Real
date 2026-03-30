#!/usr/bin/env python
"""Evaluate Lab-CC prompt boxes against GT boxes over a COCO-style dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matmatch2real.teacher.prompt_generators import LabCCBoxPromptGenerator
from matmatch2real.config.loader import load_config


def _xywh_to_xyxy(b: List[float]) -> List[float]:
    x, y, w, h = [float(v) for v in b]
    return [x, y, x + w, y + h]


def _box_iou_xyxy(a: List[float], b: List[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _greedy_match_stats(gt_boxes: List[List[float]], pred_boxes: List[List[float]], iou_thr: float) -> Dict[str, float]:
    used_gt = set()
    matched = 0
    matched_iou_sum = 0.0
    for pb in pred_boxes:
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gt_boxes):
            if j in used_gt:
                continue
            iou = _box_iou_xyxy(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            used_gt.add(best_j)
            matched += 1
            matched_iou_sum += best_iou

    gt_n = len(gt_boxes)
    pred_n = len(pred_boxes)
    precision = matched / pred_n if pred_n > 0 else 0.0
    recall = matched / gt_n if gt_n > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_matched_iou = matched_iou_sum / matched if matched > 0 else 0.0
    return {
        "matched": float(matched),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "gt_count": float(gt_n),
        "pred_count": float(pred_n),
        "mean_matched_iou": float(mean_matched_iou),
    }


def _parse_iou_thrs(value: str) -> List[float]:
    thrs: List[float] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        thrs.append(float(part))
    if not thrs:
        raise ValueError("No valid IoU thresholds parsed")
    return thrs


def _build_labcc_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    lab_cfg: Dict[str, Any] = {
        "l_thresh_min": args.l_thresh_min,
        "a_thresh": args.a_thresh,
        "b_thresh": args.b_thresh,
        "close_kernel": args.close_kernel,
        "open_kernel": args.open_kernel,
        "min_cc_area": args.min_cc_area,
        "min_box_w": args.min_box_w,
        "min_box_h": args.min_box_h,
        "force_square": bool(args.force_square),
        "high_overlap_filter": bool(args.high_overlap_filter),
        "high_overlap_thresh": args.high_overlap_thresh,
        "nms_thresh": args.nms_thresh,
        "max_prompts_per_image": args.max_prompts_per_image,
        "save_debug": False,
    }
    if args.config:
        cfg = load_config(args.config)
        lab_cfg.update(cfg.get("teacher", {}).get("prompt_generator", {}).get("lab_cc_boxes", {}))
    return lab_cfg


def _aggregate_stats(per_image_stats: Iterable[Dict[str, float]]) -> Dict[str, float]:
    stats = list(per_image_stats)
    total_gt = sum(item["gt_count"] for item in stats)
    total_pred = sum(item["pred_count"] for item in stats)
    total_matched = sum(item["matched"] for item in stats)
    precision = total_matched / total_pred if total_pred > 0 else 0.0
    recall = total_matched / total_gt if total_gt > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_matched_iou = (
        sum(item["mean_matched_iou"] * item["matched"] for item in stats) / total_matched
        if total_matched > 0
        else 0.0
    )
    return {
        "images": float(len(stats)),
        "gt_count": float(total_gt),
        "pred_count": float(total_pred),
        "matched": float(total_matched),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": float(mean_matched_iou),
    }


def _draw_pred_boxes(
    image_hw: Tuple[int, int],
    pred_boxes: List[List[float]],
    *,
    image_np: np.ndarray | None = None,
    mode: str = "raw",
    line_width: int = 3,
) -> Image.Image:
    height, width = image_hw
    mode = str(mode).strip().lower()
    if mode == "raw":
        if image_np is None:
            raise ValueError("image_np is required when vis-mode=raw")
        canvas = Image.fromarray(np.array(image_np, copy=True))
        box_color = (0, 0, 0)
    elif mode == "black":
        canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
        box_color = (255, 255, 255)
    else:
        canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
        box_color = (0, 0, 0)
    draw = ImageDraw.Draw(canvas)
    for box in pred_boxes:
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle((x0, y0, x1, y1), outline=box_color, width=line_width)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Lab-CC prompt boxes over all images in COCO GT")
    parser.add_argument("--gt-json", default='../data/cdw_classify/dataset_seg/annotations/instances_test.json', help="COCO GT json path")
    parser.add_argument("--images-root", default='../data/cdw_classify/dataset_seg/images/test', help="Image root directory")
    parser.add_argument("--config", default="configs/teacher/teacher_default.yaml", help="Optional config YAML to read lab_cc params")
    parser.add_argument("--iou-thrs", default="0.3,0.5,0.7", help="Comma-separated IoU thresholds")
    parser.add_argument("--limit", type=int, default=1, help="Limit evaluated images; -1 means all")
    parser.add_argument("--save-json", default="outputs/labcc/result.json", help="Optional path to save detailed results json")
    parser.add_argument("--save-vis-dir", default="outputs/labcc/", help="Optional directory to save predicted-box visualizations")
    parser.add_argument("--vis-mode", choices=["raw", "white", "black"], default="white", help="Visualization mode")
    parser.add_argument("--vis-line-width", type=int, default=3, help="Visualization rectangle line width")
    parser.add_argument("--verbose-every", type=int, default=50, help="Log progress every N images")
    parser.add_argument("--l-thresh-min", type=int, default=18)
    parser.add_argument("--a-thresh", type=int, default=131)
    parser.add_argument("--b-thresh", type=int, default=133)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--open-kernel", type=int, default=3)
    parser.add_argument("--min-cc-area", type=int, default=1200)
    parser.add_argument("--min-box-w", type=int, default=100)
    parser.add_argument("--min-box-h", type=int, default=100)
    parser.add_argument("--force-square", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument(
        "--high-overlap-filter",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=False,
    )
    parser.add_argument("--high-overlap-thresh", type=float, default=0.9)
    parser.add_argument("--nms-thresh", type=float, default=0.7)
    parser.add_argument("--max-prompts-per-image", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iou_thrs = _parse_iou_thrs(args.iou_thrs)

    gt_path = Path(args.gt_json)
    payload = json.loads(gt_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    anns = payload.get("annotations", [])
    if not images:
        raise ValueError(f"No images found in {gt_path}")

    ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in anns:
        image_id = int(ann.get("image_id", -1))
        ann_by_image.setdefault(image_id, []).append(ann)

    if args.limit > 0:
        images = images[: args.limit]

    generator = LabCCBoxPromptGenerator(_build_labcc_cfg(args))
    images_root = Path(args.images_root)
    vis_dir = Path(args.save_vis_dir) if str(args.save_vis_dir).strip() else None
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    all_details: List[Dict[str, Any]] = []
    all_by_thr: Dict[float, List[Dict[str, float]]] = {thr: [] for thr in iou_thrs}
    zero_pred_images = 0
    missing_images = 0

    for idx, img_info in enumerate(images, start=1):
        image_id = int(img_info.get("id", idx - 1))
        file_name = str(img_info.get("file_name", ""))
        image_path = images_root / file_name
        if not image_path.exists():
            missing_images += 1
            continue

        image_np = np.asarray(Image.open(image_path).convert("RGB"))
        gt_boxes = [
            _xywh_to_xyxy(ann["bbox"])
            for ann in ann_by_image.get(image_id, [])
            if isinstance(ann.get("bbox"), list) and len(ann["bbox"]) == 4
        ]
        pred_boxes_np = generator.generate_boxes(
            image_np,
            image_meta={
                "image_id": image_id,
                "file_name": file_name,
                "height": image_np.shape[0],
                "width": image_np.shape[1],
            },
        )
        pred_boxes = pred_boxes_np.tolist()
        if len(pred_boxes) == 0:
            zero_pred_images += 1

        if vis_dir is not None:
            vis_image = _draw_pred_boxes(
                image_hw=(image_np.shape[0], image_np.shape[1]),
                pred_boxes=pred_boxes,
                image_np=image_np,
                mode=args.vis_mode,
                line_width=max(1, int(args.vis_line_width)),
            )
            stem = Path(file_name).stem if file_name else f"image_{image_id}"
            vis_path = vis_dir / f"{stem}_{image_id}_pred_boxes.png"
            vis_image.save(vis_path)

        per_thr: Dict[str, Dict[str, float]] = {}
        for thr in iou_thrs:
            stats = _greedy_match_stats(gt_boxes, pred_boxes, thr)
            all_by_thr[thr].append(stats)
            per_thr[f"{thr:.2f}"] = stats

        all_details.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "gt_count": len(gt_boxes),
                "pred_count": len(pred_boxes),
                "metrics": per_thr,
            }
        )

        if args.verbose_every > 0 and idx % args.verbose_every == 0:
            print(f"[{idx}/{len(images)}] processed {file_name} pred_boxes={len(pred_boxes)} gt_boxes={len(gt_boxes)}")

    print(f"evaluated_images={len(all_details)} missing_images={missing_images} zero_pred_images={zero_pred_images}")
    print("Lab-CC params:")
    print(json.dumps(_build_labcc_cfg(args), ensure_ascii=False, indent=2))

    summary: Dict[str, Any] = {
        "evaluated_images": len(all_details),
        "missing_images": missing_images,
        "zero_pred_images": zero_pred_images,
        "iou_thresholds": iou_thrs,
        "metrics": {},
    }
    for thr in iou_thrs:
        agg = _aggregate_stats(all_by_thr[thr])
        summary["metrics"][f"{thr:.2f}"] = agg
        print(
            f"IoU@{thr:.2f} | images={int(agg['images'])} gt={int(agg['gt_count'])} "
            f"pred={int(agg['pred_count'])} matched={int(agg['matched'])} "
            f"P={agg['precision']:.4f} R={agg['recall']:.4f} F1={agg['f1']:.4f} "
            f"mean_matched_iou={agg['mean_matched_iou']:.4f}"
        )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "per_image": all_details,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved_json={out_path}")


if __name__ == "__main__":
    main()
