#!/usr/bin/env python
"""Build pseudo labels and teacher soft files for student training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matmatch2real.core.structures import InstancePrediction, PseudoLabelInstance
from matmatch2real.data.remote import ensure_dataset_available
from matmatch2real.teacher.postprocess import convert_instances
from matmatch2real.teacher.reliability import compute_reliability
from matmatch2real.teacher.matmatch_teacher import SAM2Teacher, SegmentCDWAdapter
from matmatch2real.config.loader import load_config
from matmatch2real.utils.logging import setup_logger
from matmatch2real.utils.paths import resolve_project_path

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    from pycocotools import mask as mask_utils  # type: ignore
except Exception:  # pragma: no cover
    mask_utils = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pseudo labels and teacher soft files")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "teacher" / "distill_default.yaml"), help="Path to config")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split label for default outputs")
    parser.add_argument("--output", default="pseudolabels/test_pseudolabels.json", help="Output pseudo label JSON path")
    parser.add_argument("--output-results", default="", help="Output COCO results JSON path")
    parser.add_argument("--teacher-soft-dir", default="teacher_soft/train", help="Output directory for teacher soft npz files")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of images")
    parser.add_argument("--coco-gt", default="", help="COCO GT JSON path override")
    parser.add_argument("--images-root", default="", help="Images root dir override")
    parser.add_argument("--use-folder", action="store_true", help="Read images from folder instead of COCO")
    parser.add_argument("--encode-workers", type=int, default=0, help="ProcessPool workers for RLE encode")
    parser.add_argument("--mask-logit-scale", type=float, default=8.0, help="Absolute logit value used to encode binary masks")
    parser.add_argument("--save-boundary", type=_str2bool, default=True, help="Write boundary_map into teacher soft files")
    parser.add_argument("--save-class-soft", type=_str2bool, default=True, help="Write class_soft into teacher soft files")
    return parser.parse_args()


def _str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _load_image_np(path: Path) -> "np.ndarray":
    if Image is None or np is None:
        raise RuntimeError("PIL and numpy are required for export.")
    img = Image.open(path).convert("RGB")
    return np.array(img, copy=True)


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _resolve_repo_path(value: Any) -> str:
    return resolve_project_path(value)


def _normalize_teacher_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    teacher_cfg = dict(cfg.get("teacher", {}))

    sam2_cfg = dict(teacher_cfg.get("sam2", {}))
    sam2_cfg["config_file"] = _resolve_repo_path(sam2_cfg.get("config_file"))
    sam2_cfg["ckpt_path"] = _resolve_repo_path(sam2_cfg.get("ckpt_path"))
    teacher_cfg["sam2"] = sam2_cfg

    classifier_cfg = dict(teacher_cfg.get("classifier", {}))

    model_cfg = dict(classifier_cfg.get("model", {}))
    if model_cfg.get("checkpoint"):
        model_cfg["checkpoint"] = _resolve_repo_path(model_cfg.get("checkpoint"))
    classifier_cfg["model"] = model_cfg

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


def _resolve_images_root(data_root: Path, cfg: Dict[str, Any], args: argparse.Namespace) -> Path:
    if args.images_root:
        return Path(args.images_root)
    train_cfg = cfg.get("data", {}).get("train", {})
    root = train_cfg.get("images_root") or train_cfg.get("images_dir") or "images"
    return _resolve_path(data_root, str(root))


def _resolve_coco_gt(cfg: Dict[str, Any], args: argparse.Namespace, data_root: Path) -> Optional[Path]:
    if args.coco_gt:
        return Path(args.coco_gt)
    train_cfg = cfg.get("data", {}).get("train", {})
    gt = train_cfg.get("coco_gt") or train_cfg.get("gt_json") or ""
    if not gt:
        return None
    return _resolve_path(data_root, str(gt))


def _resolve_output_path(data_root: Path, args: argparse.Namespace) -> Path:
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("pseudolabels") / f"{args.split}_pseudolabels.json"
    if not output_path.is_absolute():
        output_path = data_root / output_path
    return output_path


def _resolve_results_path(output_path: Path, args: argparse.Namespace) -> Path:
    if args.output_results:
        return Path(args.output_results)
    return output_path.with_name(f"{output_path.stem}_results.json")


def _resolve_teacher_soft_dir(data_root: Path, cfg: Dict[str, Any], args: argparse.Namespace) -> Path:
    if args.teacher_soft_dir:
        path = Path(args.teacher_soft_dir)
    else:
        loss_cfg = cfg.get("loss", {}).get("distill_soft", {})
        key = {
            "train": "teacher_soft_dir_train",
            "val": "teacher_soft_dir_val",
            "test": "teacher_soft_dir_val",
        }.get(args.split, "teacher_soft_dir_train")
        default_dir = loss_cfg.get(key) or f"teacher_soft/{args.split}"
        path = Path(str(default_dir))
    if not path.is_absolute():
        path = data_root / path
    return path


def _load_coco_images(coco_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    return payload.get("images", [])


def _iter_folder_images(images_root: Path) -> List[Dict[str, Any]]:
    image_paths = sorted([p for p in images_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    images: List[Dict[str, Any]] = []
    for idx, path in enumerate(image_paths):
        images.append({"id": idx, "file_name": str(path.relative_to(images_root)), "height": 0, "width": 0})
    return images


def _resolve_image_path(images_root: Path, file_name: str) -> Path:
    path = Path(file_name)
    return path if path.is_absolute() else images_root / path


def _encode_rle(mask: "np.ndarray") -> Dict[str, Any]:
    if mask_utils is None:
        raise RuntimeError("pycocotools is required for RLE encoding.")
    m = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(m)
    if isinstance(rle.get("counts"), bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _instance_to_coco_ann(inst: PseudoLabelInstance, image_id: int, image_name: str, ann_id: int) -> Optional[Dict[str, Any]]:
    x, y, w, h = inst.bbox
    if w <= 0 or h <= 0:
        return None
    mask = _instance_mask_to_numpy(inst)
    if mask is None:
        return None
    area = float(mask.sum())
    return {
        "iscrowd": False,
        "image_id": int(image_id),
        "image_name": image_name,
        "category_id": int(inst.class_id),
        "id": int(ann_id),
        "segmentation": _encode_rle(mask),
        "area": area,
        "bbox": [float(x), float(y), float(w), float(h)],
    }


def _instance_to_coco_result(inst: PseudoLabelInstance, image_id: int) -> Optional[Dict[str, Any]]:
    x, y, w, h = inst.bbox
    if w <= 0 or h <= 0:
        return None
    mask = _instance_mask_to_numpy(inst)
    if mask is None:
        return None
    return {
        "image_id": int(image_id),
        "category_id": int(inst.class_id),
        "segmentation": _encode_rle(mask),
        "bbox": [float(x), float(y), float(w), float(h)],
        "score": float(inst.score),
    }


def _set_worker_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _encode_instances_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    _set_worker_env()
    image_id = int(payload["image_id"])
    image_name = str(payload["image_name"])
    insts = payload["instances"]
    anns: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    for inst in insts:
        pseudo = PseudoLabelInstance(
            image_id=image_id,
            bbox=tuple(inst["bbox"]),
            class_id=int(inst["class_id"]),
            score=float(inst["score"]),
            reliability=float(inst.get("reliability", inst["score"])),
            mask=inst.get("mask"),
            rle=inst.get("rle"),
        )
        ann = _instance_to_coco_ann(pseudo, image_id, image_name, 0)
        if ann is not None:
            ann.pop("id", None)
            anns.append(ann)
        result = _instance_to_coco_result(pseudo, image_id)
        if result is not None:
            results.append(result)
    return {"image_id": image_id, "anns": anns, "results": results}


def _serialize_instance(inst: PseudoLabelInstance) -> Dict[str, Any]:
    mask = _instance_mask_to_numpy(inst)
    return {
        "bbox": list(inst.bbox),
        "class_id": int(inst.class_id),
        "score": float(inst.score),
        "reliability": float(inst.reliability),
        "mask": mask,
        "rle": inst.rle,
    }


def _instance_mask_to_numpy(inst: PseudoLabelInstance) -> Optional["np.ndarray"]:
    if np is None:
        raise RuntimeError("numpy is required for export.")
    mask = inst.mask
    if mask is not None and hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return (mask > 0.5).astype(np.uint8)
    if inst.rle is not None and mask_utils is not None:
        decoded = mask_utils.decode(inst.rle).astype(np.uint8)
        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]
        return decoded
    return None


def _build_categories(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    cls_cfg = cfg.get("teacher", {}).get("classifier", {})
    cat_map = cls_cfg.get("category_map")
    if isinstance(cat_map, dict) and cat_map:
        cats: List[Dict[str, Any]] = []
        for key, value in cat_map.items():
            try:
                cid = int(str(key).split(":")[0])
            except Exception:
                continue
            name = str(value)
            if ":" in name:
                name = name.split(":", 1)[-1].strip().strip("'").strip('"')
            cats.append({"id": cid, "name": name, "supercategory": None})
        if cats:
            return sorted(cats, key=lambda x: x["id"])
    return [
        {"id": 0, "name": "crushed_stone", "supercategory": None},
        {"id": 1, "name": "brick", "supercategory": None},
        {"id": 2, "name": "concrete", "supercategory": None},
        {"id": 3, "name": "ceramic", "supercategory": None},
    ]


def _run_teacher_pipeline(teacher: SAM2Teacher, image_np: "np.ndarray", meta: Dict[str, Any], image_id: int) -> List[PseudoLabelInstance]:
    raw_preds = (
        teacher.adapter.run([image_np], [meta])
        if isinstance(teacher.adapter, SegmentCDWAdapter)
        else teacher.adapter.generate([image_np], [meta])
    )
    raw = raw_preds[0] if raw_preds else []
    height = int(meta.get("height", image_np.shape[0]))
    width = int(meta.get("width", image_np.shape[1]))

    if raw and isinstance(raw[0], InstancePrediction):
        processed = list(raw)
    else:
        filtered = teacher.postprocess(list(raw), meta)
        processed = convert_instances(
            filtered,
            image_hw=(height, width),
            image_id=image_id,
            class_id=0,
            cfg=teacher.post_cfg,
            encode_rle=bool(teacher.post_cfg.get("encode_rle", False)),
        )

    if teacher.classifier is not None:
        processed = teacher.classifier.classify(processed, image_np=image_np)

    output: List[PseudoLabelInstance] = []
    for inst in processed:
        inst.reliability = compute_reliability(inst)
        output.append(
            PseudoLabelInstance(
                image_id=inst.image_id,
                bbox=inst.bbox,
                class_id=inst.class_id,
                score=inst.score,
                reliability=inst.reliability,
                mask=inst.mask,
                rle=inst.rle,
                meta=inst.meta,
            )
        )
    return output


def _mask_to_logits(mask: "np.ndarray", scale: float) -> "np.ndarray":
    logits = np.full(mask.shape, -abs(float(scale)), dtype=np.float32)
    logits[mask > 0] = abs(float(scale))
    return logits


def _mask_boundary(mask: "np.ndarray") -> "np.ndarray":
    m = mask.astype(bool)
    up = np.pad(m[:-1, :], ((1, 0), (0, 0)), mode="constant")
    dn = np.pad(m[1:, :], ((0, 1), (0, 0)), mode="constant")
    lf = np.pad(m[:, :-1], ((0, 0), (1, 0)), mode="constant")
    rt = np.pad(m[:, 1:], ((0, 0), (0, 1)), mode="constant")
    boundary = (m != up) | (m != dn) | (m != lf) | (m != rt)
    return boundary.astype(np.float32)


def _class_soft_from_instance(
    inst: PseudoLabelInstance,
    num_classes: int,
    class_to_index: Dict[int, int],
) -> Optional["np.ndarray"]:
    if num_classes <= 0:
        return None
    cls_index = class_to_index.get(int(inst.class_id))
    if cls_index is None or cls_index >= num_classes:
        return None
    score = float(inst.meta.get("category_score", inst.meta.get("cls_prob", inst.score)))
    score = max(0.0, min(1.0, score))
    probs = np.zeros((num_classes,), dtype=np.float32)
    if num_classes == 1:
        probs[0] = 1.0
        return probs
    fill = (1.0 - score) / max(1, num_classes - 1)
    probs.fill(fill)
    probs[cls_index] = score
    probs /= probs.sum().clip(min=1e-6)
    return probs


def _write_teacher_soft_file(
    output_path: Path,
    instances: List[PseudoLabelInstance],
    image_hw: Tuple[int, int],
    categories: List[Dict[str, Any]],
    mask_logit_scale: float,
    save_boundary: bool,
    save_class_soft: bool,
) -> None:
    height, width = image_hw
    masks: List[np.ndarray] = []
    scores: List[float] = []
    boundaries: List[np.ndarray] = []
    class_softs: List[np.ndarray] = []
    class_to_index = {int(cat["id"]): idx for idx, cat in enumerate(sorted(categories, key=lambda x: int(x["id"])))}
    num_classes = len(class_to_index)

    for inst in instances:
        mask = _instance_mask_to_numpy(inst)
        if mask is None:
            continue
        if mask.shape != (height, width):
            mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), resample=Image.NEAREST) > 127, copy=False)
            mask = mask.astype(np.uint8)
        masks.append(_mask_to_logits(mask, mask_logit_scale))
        scores.append(float(inst.score))
        if save_boundary:
            boundaries.append(_mask_boundary(mask))
        if save_class_soft:
            probs = _class_soft_from_instance(inst, num_classes=num_classes, class_to_index=class_to_index)
            if probs is not None:
                class_softs.append(probs)

    payload: Dict[str, Any] = {
        "mask_logits": np.stack(masks).astype(np.float16) if masks else np.zeros((0, height, width), dtype=np.float16),
        "score": np.array(scores, dtype=np.float32),
    }
    if save_boundary:
        payload["boundary_map"] = (
            np.stack(boundaries).astype(np.float16)
            if boundaries
            else np.zeros((0, height, width), dtype=np.float16)
        )
    if save_class_soft:
        payload["class_soft"] = (
            np.stack(class_softs).astype(np.float16)
            if class_softs
            else np.zeros((0, num_classes), dtype=np.float16)
        )
        payload["class_ids"] = np.array([int(cat["id"]) for cat in sorted(categories, key=lambda x: int(x["id"]))], dtype=np.int32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def main() -> None:
    if np is None:
        raise RuntimeError("numpy is required for export.")
    args = parse_args()
    cfg = _normalize_teacher_paths(load_config(args.config))
    logger = setup_logger("distill")
    data_root = ensure_dataset_available(cfg)

    images_root = _resolve_images_root(data_root, cfg, args)
    coco_gt = _resolve_coco_gt(cfg, args, data_root)
    output_path = _resolve_output_path(data_root, args)
    results_path = _resolve_results_path(output_path, args)
    if not results_path.is_absolute():
        results_path = Path(results_path)
    teacher_soft_dir = _resolve_teacher_soft_dir(data_root, cfg, args)

    if args.use_folder or coco_gt is None:
        images_info = _iter_folder_images(images_root)
        logger.info("folder mode: images=%d", len(images_info))
    else:
        images_info = _load_coco_images(coco_gt)
        logger.info("COCO mode: images=%d from %s", len(images_info), coco_gt)

    if args.limit > 0:
        images_info = images_info[: args.limit]

    teacher = SAM2Teacher(cfg.get("teacher", {}))
    if teacher.adapter is None:
        raise RuntimeError("Teacher adapter is not configured; cannot export training artifacts.")
    categories = _build_categories(cfg)

    logger.info(
        "export -> pseudo=%s | results=%s | soft=%s",
        output_path,
        results_path,
        teacher_soft_dir,
    )

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    ann_id = 1
    encode_workers = int(args.encode_workers)
    pool = None
    futures = []

    iterator: Iterable[Dict[str, Any]] = images_info
    if tqdm is not None:
        iterator = tqdm(images_info, total=len(images_info), desc="Export training artifacts", dynamic_ncols=True)

    for img in iterator:
        image_id = int(img.get("id"))
        file_name = str(img.get("file_name", ""))
        image_path = _resolve_image_path(images_root, file_name)
        image_np = _load_image_np(image_path)
        height = int(img.get("height", 0)) or int(image_np.shape[0])
        width = int(img.get("width", 0)) or int(image_np.shape[1])

        meta = {
            "path": str(image_path),
            "file_name": file_name,
            "height": height,
            "width": width,
            "image_id": image_id,
        }
        preds = _run_teacher_pipeline(teacher, image_np, meta, image_id=image_id)

        soft_path = teacher_soft_dir / Path(file_name).with_suffix(".npz")
        _write_teacher_soft_file(
            soft_path,
            preds,
            image_hw=(height, width),
            categories=categories,
            mask_logit_scale=float(args.mask_logit_scale),
            save_boundary=bool(args.save_boundary),
            save_class_soft=bool(args.save_class_soft),
        )

        images.append({"id": image_id, "file_name": file_name, "height": height, "width": width})
        image_name = Path(file_name).name

        if encode_workers > 0 and pool is None:
            pool = ProcessPoolExecutor(max_workers=encode_workers, initializer=_set_worker_env)

        if pool is None:
            for inst in preds:
                ann = _instance_to_coco_ann(inst, image_id, image_name, ann_id)
                if ann is None:
                    continue
                annotations.append(ann)
                ann_id += 1
                result = _instance_to_coco_result(inst, image_id)
                if result is not None:
                    results.append(result)
        else:
            payload = {
                "image_id": image_id,
                "image_name": image_name,
                "instances": [_serialize_instance(inst) for inst in preds],
            }
            futures.append(pool.submit(_encode_instances_worker, payload))

    if pool is not None:
        for fut in as_completed(futures):
            out = fut.result()
            for ann in out["anns"]:
                ann["id"] = int(ann_id)
                annotations.append(ann)
                ann_id += 1
            results.extend(out["results"])
        pool.shutdown(wait=True)

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "meta": {
            "source": "sam2",
            "coco_gt": str(coco_gt) if coco_gt else "",
            "teacher_soft_dir": str(teacher_soft_dir),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("pseudo labels exported to %s", output_path)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("pseudo label results exported to %s", results_path)
    logger.info("teacher soft files exported to %s", teacher_soft_dir)


if __name__ == "__main__":
    main()
