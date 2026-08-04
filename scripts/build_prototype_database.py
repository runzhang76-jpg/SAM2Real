#!/usr/bin/env python
"""Build class prototype database (.pth) from COCO-style annotations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matmatch2real.teacher.dinov3_classifier import load_model
from matmatch2real.config.loader import load_config
from matmatch2real.utils.logging import setup_logger
from matmatch2real.utils.paths import resolve_project_path

import torch
from PIL import Image
from pycocotools import mask as mask_utils



def _str2bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return (x / denom).astype(np.float32, copy=False)


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


def _resolve_dino_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cls_cfg = cfg.get("teacher", {}).get("classifier", {})
    dino_cfg = dict(cls_cfg.get("dinov3", {}))
    merged_cfg = dict(dino_cfg)
    merged_cfg["mode"] = "prototype"
    merged_cfg.update(dino_cfg.get("prototype", {}))
    return merged_cfg


def _resolve_repo_path(value: Any) -> str:
    return resolve_project_path(value)


def _normalize_config_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    teacher_cfg = dict(cfg.get("teacher", {}))
    classifier_cfg = dict(teacher_cfg.get("classifier", {}))

    dino_cfg = dict(classifier_cfg.get("dinov3", {}))
    for key in ("repo_dir", "weights"):
        if dino_cfg.get(key):
            dino_cfg[key] = _resolve_repo_path(dino_cfg.get(key))

    knn_cfg = dict(dino_cfg.get("knn", {}))
    if knn_cfg.get("database_csv"):
        knn_cfg["database_csv"] = _resolve_repo_path(knn_cfg.get("database_csv"))
    dino_cfg["knn"] = knn_cfg

    prototype_cfg = dict(dino_cfg.get("prototype", {}))
    if prototype_cfg.get("prototype_pth"):
        prototype_cfg["prototype_pth"] = _resolve_repo_path(prototype_cfg.get("prototype_pth"))
    dino_cfg["prototype"] = prototype_cfg

    classifier_cfg["dinov3"] = dino_cfg

    teacher_cfg["classifier"] = classifier_cfg

    normalized = dict(cfg)
    normalized["teacher"] = teacher_cfg
    return normalized


def _preprocess_patches(
    patches: List[np.ndarray],
    *,
    input_size: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> "torch.Tensor":
    if torch is None or Image is None:
        raise RuntimeError("PyTorch and PIL are required for DINOv3 feature extraction")
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    tensors = []
    for patch in patches:
        img = Image.fromarray(patch).resize((input_size, input_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32, copy=True) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        t = (t - mean_t) / std_t
        tensors.append(t)
    return torch.stack(tensors, dim=0)


def _extract_features(
    model: "torch.nn.Module",
    patches: List[np.ndarray],
    *,
    device: str,
    input_size: int,
    batch_size: int,
    feature_dim: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for feature extraction")
    if len(patches) == 0:
        return np.zeros((0, feature_dim), dtype=np.float32)
    feats: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            x = _preprocess_patches(
                patches[i : i + batch_size],
                input_size=input_size,
                mean=mean,
                std=std,
            ).to(device)
            f = model.forward_features(x)
            if isinstance(f, dict):
                if "x_norm_clstoken" in f:
                    f = f["x_norm_clstoken"]
                elif "x_norm_cls_token" in f:
                    f = f["x_norm_cls_token"]
                else:
                    raise KeyError("forward_features output dict missing x_norm_clstoken")
            if not isinstance(f, torch.Tensor):
                raise TypeError("forward_features must return Tensor or dict of Tensor")
            if f.ndim == 3:
                f = f[:, 0, :]
            feats.append(f.detach().cpu().to(torch.float32).numpy())
    out = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    if out.shape[1] != feature_dim:
        raise ValueError(f"DINO feature dim mismatch: model={out.shape[1]} config={feature_dim}")
    return _l2_normalize(out)


def _validate_ratio(name: str, value: float) -> float:
    ratio = float(value)
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {ratio}")
    return ratio


def _resolve_min_max_class_ratio(value: Optional[float], *, seed: int) -> Tuple[float, str]:
    if value is not None:
        return _validate_ratio("min_max_class_ratio", value), "argument"
    rng = np.random.default_rng(seed)
    ratio = float(rng.uniform(np.nextafter(0.5, 1.0), 1.0))
    return ratio, "random"


def _largest_remainder_allocation(
    capacities: Dict[int, int],
    target_total: int,
) -> Dict[int, int]:
    if target_total <= 0 or not capacities:
        return {int(class_id): 0 for class_id in capacities}

    total_capacity = sum(int(v) for v in capacities.values())
    if total_capacity <= 0:
        return {int(class_id): 0 for class_id in capacities}
    if target_total >= total_capacity:
        return {int(class_id): int(v) for class_id, v in capacities.items()}

    raw_targets: Dict[int, float] = {}
    allocation: Dict[int, int] = {}
    for class_id, capacity in capacities.items():
        raw = target_total * float(capacity) / float(total_capacity)
        raw_targets[int(class_id)] = raw
        allocation[int(class_id)] = min(int(capacity), int(np.floor(raw)))

    assigned = sum(allocation.values())
    remaining = target_total - assigned
    if remaining <= 0:
        return allocation

    order = sorted(
        capacities.keys(),
        key=lambda class_id: (
            raw_targets[int(class_id)] - np.floor(raw_targets[int(class_id)]),
            capacities[int(class_id)] - allocation[int(class_id)],
            -int(class_id),
        ),
        reverse=True,
    )
    for class_id in order:
        if remaining <= 0:
            break
        cid = int(class_id)
        spare = int(capacities[cid]) - int(allocation[cid])
        if spare <= 0:
            continue
        take = min(spare, remaining)
        allocation[cid] += take
        remaining -= take

    return allocation


def _largest_remainder_from_targets(
    raw_targets: Dict[int, float],
    capacities: Dict[int, int],
    desired_total: int,
) -> Dict[int, int]:
    allocation: Dict[int, int] = {}
    for class_id, raw_target in raw_targets.items():
        allocation[int(class_id)] = min(int(capacities[int(class_id)]), int(np.floor(raw_target)))

    assigned = sum(allocation.values())
    remaining = max(0, int(desired_total) - assigned)
    if remaining <= 0:
        return allocation

    order = sorted(
        raw_targets.keys(),
        key=lambda class_id: (
            raw_targets[int(class_id)] - np.floor(raw_targets[int(class_id)]),
            capacities[int(class_id)] - allocation[int(class_id)],
            -int(class_id),
        ),
        reverse=True,
    )
    for class_id in order:
        if remaining <= 0:
            break
        cid = int(class_id)
        if allocation[cid] >= int(capacities[cid]):
            continue
        allocation[cid] += 1
        remaining -= 1
    return allocation


def _build_exact_ratio_targets(
    raw_counts: Dict[int, int],
    *,
    target_total: int,
    min_max_class_ratio: float,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    if not raw_counts:
        return {}, {"requested_total": 0, "selected_total": 0, "actual_ratio": None}

    class_ids = sorted(raw_counts.keys())
    if len(class_ids) == 1:
        only_id = int(class_ids[0])
        count = min(int(raw_counts[only_id]), int(target_total))
        return {only_id: count}, {
            "requested_total": int(target_total),
            "selected_total": int(count),
            "actual_ratio": 1.0 if count > 0 else None,
        }

    count_values = [int(raw_counts[class_id]) for class_id in class_ids]
    raw_min = min(count_values)
    raw_max = max(count_values)

    if raw_max == raw_min:
        even_target = min(sum(count_values), int(target_total))
        even_counts = _largest_remainder_allocation(
            {int(class_id): int(raw_counts[class_id]) for class_id in class_ids},
            even_target,
        )
        return even_counts, {
            "requested_total": int(target_total),
            "selected_total": int(sum(even_counts.values())),
            "actual_ratio": 1.0 if even_counts else None,
        }

    weights: Dict[int, float] = {}
    for class_id in class_ids:
        count = float(raw_counts[int(class_id)])
        position = (count - float(raw_min)) / max(1e-12, float(raw_max - raw_min))
        weights[int(class_id)] = float(min_max_class_ratio + (1.0 - min_max_class_ratio) * position)

    scale_from_caps = min(
        float(raw_counts[int(class_id)]) / max(weights[int(class_id)], 1e-12)
        for class_id in class_ids
    )
    scale_from_total = float(target_total) / max(sum(weights.values()), 1e-12)
    scale = min(scale_from_caps, scale_from_total)

    raw_targets = {int(class_id): float(scale * weights[int(class_id)]) for class_id in class_ids}
    feasible_total = min(int(target_total), int(np.floor(sum(raw_targets))) + len(class_ids))
    selected_counts = _largest_remainder_from_targets(raw_targets, raw_counts, feasible_total)

    positive_counts = [int(v) for v in selected_counts.values() if int(v) > 0]
    actual_ratio = None
    if positive_counts:
        actual_ratio = float(min(positive_counts) / max(max(positive_counts), 1))

    return selected_counts, {
        "requested_total": int(target_total),
        "selected_total": int(sum(selected_counts.values())),
        "actual_ratio": actual_ratio,
        "raw_min": int(raw_min),
        "raw_max": int(raw_max),
        "scale_from_caps": float(scale_from_caps),
        "scale_from_total": float(scale_from_total),
    }


def _sample_annotations_by_class(
    anns: Sequence[Dict[str, Any]],
    *,
    total_sample_ratio: float,
    min_max_class_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[int, int], Dict[str, Any]]:
    anns_by_class: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in anns:
        class_id = int(ann.get("category_id", -1))
        anns_by_class[class_id].append(ann)

    raw_counts = {class_id: len(items) for class_id, items in anns_by_class.items()}
    raw_total = len(anns)
    target_total = int(np.floor(raw_total * total_sample_ratio))
    if total_sample_ratio > 0 and raw_total > 0:
        target_total = max(1, target_total)
    target_total = min(target_total, raw_total)

    selected_counts, ratio_summary = _build_exact_ratio_targets(
        raw_counts,
        target_total=target_total,
        min_max_class_ratio=min_max_class_ratio,
    )
    rng_pick = np.random.default_rng(seed)
    selected_anns: List[Dict[str, Any]] = []
    for class_id in sorted(anns_by_class.keys()):
        pool = anns_by_class[class_id]
        take = int(selected_counts.get(class_id, 0))
        if take <= 0:
            continue
        if take >= len(pool):
            selected_anns.extend(pool)
            continue
        picked = rng_pick.choice(len(pool), size=take, replace=False)
        selected_anns.extend(pool[int(idx)] for idx in np.sort(picked))

    summary = {
        "raw_total": raw_total,
        "balanced_total": len(selected_anns),
        "selected_total": len(selected_anns),
        "target_total": int(target_total),
        "class_cap": None,
        "actual_min_max_ratio": ratio_summary.get("actual_ratio"),
        "requested_total": ratio_summary.get("requested_total"),
        "ratio_limited": bool(ratio_summary.get("selected_total", 0) < int(target_total)),
        "raw_min": ratio_summary.get("raw_min"),
        "raw_max": ratio_summary.get("raw_max"),
    }
    return selected_anns, raw_counts, selected_counts, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build class prototype database from GT annotations")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "teacher" / "teacher_default.yaml"))
    parser.add_argument("--gt-json", default="../data/cdw_classify/dataset_seg/annotations/instances_test.json", help="COCO annotation json path")
    parser.add_argument("--images-root", default="../data/cdw_classify/dataset_seg/images/test", help="Images root directory")
    parser.add_argument("--use-mask", type=_str2bool, default=False, help="Mask-out background using GT segmentation")
    parser.add_argument("--margin-ratio", type=float, default=0, help="Crop margin; <0 uses config classifier.margin_ratio")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of GT instances")
    parser.add_argument("--total-sample-ratio", type=float, default=1.0, help="Select this fraction of all GT instances in the JSON")
    parser.add_argument("--min-max-class-ratio", type=float, default=1.0, help="Target min_class_count / max_class_count; if omitted, sample a random ratio > 0.5")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for class-balanced sampling")
    parser.add_argument("--min-per-class", type=int, default=4, help="Minimum samples required to export a class prototype")
    parser.add_argument("--output", default="lib/class_prototypes_sampleR1p0_classR0p9.pth", help="Output prototype database .pth")
    return parser.parse_args()


def _resolve_dataset_root(root_dir: str) -> Path:
    if not str(root_dir or "").strip():
        return PROJECT_ROOT
    return Path(resolve_project_path(root_dir))


def _resolve_gt_context(cfg: Dict[str, Any], args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.gt_json and args.images_root:
        return Path(resolve_project_path(args.gt_json)), Path(resolve_project_path(args.images_root))

    data_cfg = cfg.get("data", {})
    for split_name in ("eval", "teacher", "train"):
        split_cfg = data_cfg.get(split_name, {})
        gt_json = str(split_cfg.get("gt_json") or split_cfg.get("coco_gt") or "").strip()
        images_dir = str(split_cfg.get("images_dir", "")).strip()
        if not gt_json or not images_dir:
            continue
        root_dir = str(split_cfg.get("root_dir", data_cfg.get("root_dir", ""))).strip()
        dataset_root = _resolve_dataset_root(root_dir)
        return (
            Path(resolve_project_path(gt_json, base_dir=dataset_root)),
            Path(resolve_project_path(images_dir, base_dir=dataset_root)),
        )

    raise ValueError("Unable to infer --gt-json/--images-root from config; please pass them explicitly.")


def main() -> None:
    args = parse_args()
    logger = setup_logger("distill")

    if torch is None:
        raise RuntimeError("PyTorch is required")
    if Image is None:
        raise RuntimeError("PIL is required")

    total_sample_ratio = _validate_ratio("total_sample_ratio", args.total_sample_ratio)
    effective_min_max_class_ratio, min_max_class_ratio_source = _resolve_min_max_class_ratio(
        args.min_max_class_ratio,
        seed=int(args.seed),
    )

    cfg = _normalize_config_paths(load_config(args.config))
    dino_cfg = _resolve_dino_cfg(cfg)
    if not dino_cfg.get("repo_dir"):
        raise ValueError("DINO repo_dir is missing in config")
    if not dino_cfg.get("weights"):
        raise ValueError("DINO weights are missing in config")

    device = str(dino_cfg.get("device", "cuda")).strip()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = str(dino_cfg.get("model_name", "dinov3_vitb16")).strip()
    model_source = str(dino_cfg.get("model_source", "local")).strip() or "local"
    input_size = int(dino_cfg.get("input_size", 224))
    batch_size = int(dino_cfg.get("batch_size", 32))
    feature_dim = int(dino_cfg.get("feature_dim", 768))
    mean = tuple(dino_cfg.get("normalize_mean", dino_cfg.get("mean", (0.485, 0.456, 0.406))))
    std = tuple(dino_cfg.get("normalize_std", dino_cfg.get("std", (0.229, 0.224, 0.225))))

    cls_cfg = cfg.get("teacher", {}).get("classifier", {})
    margin_ratio = float(cls_cfg.get("margin_ratio", 0.1)) if args.margin_ratio < 0 else float(args.margin_ratio)

    logger.info("loading DINOv3 model: model=%s device=%s", model_name, device)
    model = load_model(
        repo_dir=str(dino_cfg["repo_dir"]),
        weights=str(dino_cfg["weights"]),
        model_name=model_name,
        device=device,
        source=model_source,
    )

    gt_json_path, images_root = _resolve_gt_context(cfg, args)
    gt = json.loads(gt_json_path.read_text(encoding="utf-8"))
    images = gt.get("images", [])
    anns = gt.get("annotations", [])
    categories = gt.get("categories", [])
    img_by_id = {int(x["id"]): x for x in images}
    cat_name_by_id = {int(c["id"]): str(c.get("name", c["id"])) for c in categories}

    if args.limit > 0:
        anns = anns[: args.limit]

    sampled_anns, raw_class_counts, selected_class_counts, sample_summary = _sample_annotations_by_class(
        anns,
        total_sample_ratio=total_sample_ratio,
        min_max_class_ratio=effective_min_max_class_ratio,
        seed=int(args.seed),
    )
    if sample_summary["selected_total"] <= 0:
        raise RuntimeError("Sampling removed all GT instances; adjust --total-sample-ratio/--min-max-class-ratio")
    if sample_summary["ratio_limited"]:
        logger.warning(
            "exact class-ratio target reduced selected instances: requested=%d selected=%d target_ratio=%.4f actual_ratio=%.4f",
            int(sample_summary["target_total"]),
            int(sample_summary["selected_total"]),
            float(effective_min_max_class_ratio),
            float(sample_summary["actual_min_max_ratio"] or 0.0),
        )
    logger.info(
        "sampling GT instances: raw=%d requested=%d selected=%d total_ratio=%.4f target_min_max_ratio=%.4f actual_min_max_ratio=%s seed=%d",
        int(sample_summary["raw_total"]),
        int(sample_summary["target_total"]),
        int(sample_summary["selected_total"]),
        total_sample_ratio,
        float(effective_min_max_class_ratio),
        "n/a" if sample_summary["actual_min_max_ratio"] is None else f"{float(sample_summary['actual_min_max_ratio']):.4f}",
        int(args.seed),
    )
    logger.info(
        "min-max class ratio source: %s",
        min_max_class_ratio_source,
    )
    anns = sampled_anns

    image_cache: Dict[int, np.ndarray] = {}
    patches: List[np.ndarray] = []
    patch_labels: List[int] = []
    patch_meta: List[Dict[str, Any]] = []
    skipped = 0

    for ann in anns:
        image_id = int(ann.get("image_id", -1))
        info = img_by_id.get(image_id)
        if info is None:
            skipped += 1
            continue

        if image_id not in image_cache:
            file_name = str(info.get("file_name", ""))
            image_path = images_root / file_name
            if not image_path.exists():
                logger.warning("image not found: %s", image_path)
                skipped += 1
                continue
            image_cache[image_id] = np.array(Image.open(image_path).convert("RGB"), copy=True)
        image_np = image_cache[image_id]

        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            skipped += 1
            continue

        class_id = int(ann.get("category_id", -1))
        mask = None
        if args.use_mask:
            mask = _decode_segmentation(
                ann.get("segmentation"),
                height=int(info.get("height", image_np.shape[0])),
                width=int(info.get("width", image_np.shape[1])),
            )
        patch = _crop_patch(image=image_np, bbox_xywh=bbox, mask=mask, margin_ratio=margin_ratio)
        if patch.size == 0:
            skipped += 1
            continue

        patches.append(patch)
        patch_labels.append(class_id)
        patch_meta.append(
            {
                "ann_id": int(ann.get("id", -1)),
                "image_id": image_id,
                "file_name": str(info.get("file_name", "")),
                "category_id": class_id,
                "category_name": cat_name_by_id.get(class_id, str(class_id)),
                "bbox": bbox,
            }
        )

    logger.info("collected patches: total=%d skipped=%d", len(patches), skipped)
    if not patches:
        raise RuntimeError("No valid GT patches collected")

    features = _extract_features(
        model,
        patches,
        device=device,
        input_size=input_size,
        batch_size=batch_size,
        feature_dim=feature_dim,
        mean=mean,  # type: ignore[arg-type]
        std=std,  # type: ignore[arg-type]
    )

    features_by_class: Dict[int, List[np.ndarray]] = defaultdict(list)
    for feat, class_id in zip(features, patch_labels):
        features_by_class[int(class_id)].append(feat)

    labels: List[int] = []
    counts: List[int] = []
    prototypes: List[np.ndarray] = []
    class_stats: List[Dict[str, Any]] = []

    for class_id in sorted(features_by_class.keys()):
        feat_list = features_by_class[class_id]
        count = len(feat_list)
        if count < int(args.min_per_class):
            logger.warning("skip class=%s(%d): samples=%d < min_per_class=%d", cat_name_by_id.get(class_id, str(class_id)), class_id, count, int(args.min_per_class))
            continue
        feat_mat = np.stack(feat_list, axis=0).astype(np.float32, copy=False)
        proto = feat_mat.mean(axis=0, keepdims=True)
        proto = _l2_normalize(proto)[0]
        labels.append(class_id)
        counts.append(count)
        prototypes.append(proto)
        class_stats.append(
            {
                "category_id": class_id,
                "category_name": cat_name_by_id.get(class_id, str(class_id)),
                "raw_count": int(raw_class_counts.get(class_id, 0)),
                "selected_count": int(selected_class_counts.get(class_id, 0)),
                "count": count,
            }
        )
        logger.info(
            "class=%s(%d): raw=%d selected=%d used=%d",
            cat_name_by_id.get(class_id, str(class_id)),
            class_id,
            int(raw_class_counts.get(class_id, 0)),
            int(selected_class_counts.get(class_id, 0)),
            count,
        )

    if not prototypes:
        raise RuntimeError("No class prototypes were built")

    prototype_matrix = np.stack(prototypes, axis=0).astype(np.float32, copy=False)
    output = {
        "labels": torch.tensor(labels, dtype=torch.int64),
        "counts": torch.tensor(counts, dtype=torch.int64),
        "prototypes": torch.from_numpy(prototype_matrix),
        "category_names": {int(k): v for k, v in cat_name_by_id.items()},
        "class_stats": class_stats,
        "feature_dim": int(feature_dim),
        "config": {
            "model_name": model_name,
            "model_source": model_source,
            "repo_dir": str(dino_cfg["repo_dir"]),
            "weights": str(dino_cfg["weights"]),
            "device": device,
            "input_size": input_size,
            "batch_size": batch_size,
            "use_mask": bool(args.use_mask),
            "margin_ratio": margin_ratio,
            "total_sample_ratio": float(total_sample_ratio),
            "min_max_class_ratio": float(effective_min_max_class_ratio),
            "min_max_class_ratio_arg": None if args.min_max_class_ratio is None else float(args.min_max_class_ratio),
            "min_max_class_ratio_source": min_max_class_ratio_source,
            "seed": int(args.seed),
            "min_per_class": int(args.min_per_class),
        },
        "source": {
            "gt_json": str(gt_json_path),
            "images_root": str(images_root),
            "raw_num_instances": int(sample_summary["raw_total"]),
            "balanced_num_instances": int(sample_summary["selected_total"]),
            "num_instances": len(patches),
            "class_cap": sample_summary["class_cap"],
            "actual_min_max_ratio": sample_summary["actual_min_max_ratio"],
            "ratio_limited": bool(sample_summary["ratio_limited"]),
            "skipped": skipped,
        },
    }

    out_path = Path(_resolve_repo_path(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    logger.info(
        "saved prototype database: %s (classes=%d, instances=%d, feature_dim=%d)",
        out_path,
        len(labels),
        len(patches),
        feature_dim,
    )


if __name__ == "__main__":
    main()
