#!/usr/bin/env python
"""从教师管线导出离线伪标签（COCO 对齐）。"""

from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PKG_ROOT = ROOT / "distill_cdw"
if PKG_ROOT.exists() and str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from distill.core.structures import PseudoLabelInstance
from distill.data.remote import ensure_dataset_available
from distill.teacher.sam2_teacher import SAM2Teacher
from distill.utils.config import load_config
from distill.utils.logging import setup_logger

try:
    import numpy as np
except Exception:  # pragma: no cover - 可选依赖
    np = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover - 可选依赖
    Image = None  # type: ignore

try:
    from pycocotools import mask as mask_utils  # type: ignore
except Exception:  # pragma: no cover - 可选依赖
    mask_utils = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - 可选依赖
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pseudo labels from teacher (COCO aligned)")
    parser.add_argument("--config", default="distill_cdw/configs/distill_default.yaml", help="Path to config")
    parser.add_argument("--output", default="", help="Output pseudo label JSON path")
    parser.add_argument("--output-results", default="", help="Output COCO results JSON path")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of images")
    parser.add_argument("--coco-gt", 
                        default="../data/sam2/dataset/segment_cdw_coco_dataset/annotations/instances_train.json", 
                        help="COCO GT JSON path (override config)")
    parser.add_argument("--images-root", 
                        default="../data/sam2/dataset/segment_cdw_coco_dataset/images_train", 
                        help="Images root dir (override config)")
    parser.add_argument("--use-folder", action="store_true", help="Read images from folder instead of COCO")
    parser.add_argument("--encode-workers", type=int, default=0, help="ProcessPool workers for RLE encode")
    return parser.parse_args()


def _load_image_np(path: Path) -> "np.ndarray":
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def _resolve_images_root(data_root: Path, cfg: Dict[str, Any], args: argparse.Namespace) -> Path:
    if args.images_root:
        return Path(args.images_root)
    train_cfg = cfg.get("data", {}).get("train", {})
    root = train_cfg.get("images_root") or train_cfg.get("images_dir") or "images"
    root_path = Path(root)
    return root_path if root_path.is_absolute() else data_root / root_path


def _resolve_coco_gt(cfg: Dict[str, Any], args: argparse.Namespace, data_root: Path) -> Optional[Path]:
    if args.coco_gt:
        return Path(args.coco_gt)
    train_cfg = cfg.get("data", {}).get("train", {})
    gt = train_cfg.get("coco_gt") or train_cfg.get("gt_json") or ""
    if not gt:
        return None
    gt_path = Path(gt)
    return gt_path if gt_path.is_absolute() else data_root / gt_path


def _load_coco_images(coco_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    return payload.get("images", [])


def _iter_folder_images(images_root: Path) -> List[Dict[str, Any]]:
    image_paths = sorted([p for p in images_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    images = []
    for idx, path in enumerate(image_paths):
        images.append(
            {
                "id": idx,
                "file_name": str(path.relative_to(images_root)),
                "height": 0,
                "width": 0,
            }
        )
    return images


def _resolve_image_path(images_root: Path, file_name: str) -> Path:
    path = Path(file_name)
    if path.is_absolute():
        return path
    return images_root / path


def _encode_rle(mask: "np.ndarray") -> Dict[str, Any]:
    if mask_utils is None:
        raise RuntimeError("pycocotools 未安装，无法编码 RLE。")
    m = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(m)
    if isinstance(rle.get("counts", None), bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _instance_to_coco_ann(
    inst: PseudoLabelInstance,
    image_id: int,
    image_name: str,
    ann_id: int,
) -> Optional[Dict[str, Any]]:
    if np is None:
        raise RuntimeError("numpy 未安装，无法处理 mask。")
    x, y, w, h = inst.bbox
    if w <= 0 or h <= 0:
        return None
    mask = None
    if inst.mask is not None:
        mask = inst.mask
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask)
    elif inst.rle is not None:
        if mask_utils is None:
            raise RuntimeError("pycocotools 未安装，无法解码 RLE。")
        mask = mask_utils.decode(inst.rle).astype(np.uint8)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
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


def _instance_to_coco_result(
    inst: PseudoLabelInstance,
    image_id: int,
) -> Optional[Dict[str, Any]]:
    x, y, w, h = inst.bbox
    if w <= 0 or h <= 0:
        return None
    if inst.mask is None and inst.rle is None:
        return None
    mask = None
    if inst.mask is not None:
        mask = inst.mask
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask)
    elif inst.rle is not None:
        if mask_utils is None:
            raise RuntimeError("pycocotools 未安装，无法解码 RLE。")
        mask = mask_utils.decode(inst.rle).astype(np.uint8)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
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
        ann = _instance_to_coco_ann(
            PseudoLabelInstance(
                image_id=image_id,
                bbox=inst["bbox"],
                class_id=inst["class_id"],
                score=inst["score"],
                reliability=inst.get("reliability", inst["score"]),
                mask=inst.get("mask"),
                rle=inst.get("rle"),
            ),
            image_id=image_id,
            image_name=image_name,
            ann_id=0,
        )
        if ann is not None:
            ann.pop("id", None)
            anns.append(ann)
        res = _instance_to_coco_result(
            PseudoLabelInstance(
                image_id=image_id,
                bbox=inst["bbox"],
                class_id=inst["class_id"],
                score=inst["score"],
                reliability=inst.get("reliability", inst["score"]),
                mask=inst.get("mask"),
                rle=inst.get("rle"),
            ),
            image_id=image_id,
        )
        if res is not None:
            results.append(res)
    return {"image_id": image_id, "anns": anns, "results": results}


def _serialize_instance(inst: PseudoLabelInstance) -> Dict[str, Any]:
    mask = inst.mask
    if mask is not None and hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    return {
        "bbox": inst.bbox,
        "class_id": int(inst.class_id),
        "score": float(inst.score),
        "reliability": float(inst.reliability),
        "mask": mask,
        "rle": inst.rle,
    }


def _build_categories(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not bool(cfg.get("teacher", {}).get("classifier", {}).get("enabled", False)):
        return [
            {"id": 0, "name": "aggregates", "supercategory": None},
        ]
    cls_cfg = cfg.get("teacher", {}).get("classifier", {})
    cat_map = cls_cfg.get("category_map")
    if isinstance(cat_map, dict) and cat_map:
        cats = []
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("distill")
    data_root = ensure_dataset_available(cfg)

    images_root = _resolve_images_root(data_root, cfg, args)
    coco_gt = _resolve_coco_gt(cfg, args, data_root)

    if args.use_folder or coco_gt is None:
        images_info = _iter_folder_images(images_root)
        logger.info("使用文件夹模式，共 %d 张图像", len(images_info))
    else:
        images_info = _load_coco_images(coco_gt)
        logger.info("读取 COCO images=%d from %s", len(images_info), coco_gt)

    if args.limit > 0:
        images_info = images_info[: args.limit]

    teacher = SAM2Teacher(cfg.get("teacher", {}))
    categories = _build_categories(cfg)

    logger.info(
        "export -> %s | device=%s | classifier=%s | postprocess=%s",
        args.output or cfg.get("data", {}).get("train", {}).get("pseudolabel_path", "pseudolabels.json"),
        cfg.get("experiment", {}).get("device", "auto"),
        bool(cfg.get("teacher", {}).get("classifier", {}).get("enabled", False)),
        bool(cfg.get("teacher", {}).get("postprocess", {}).get("enabled", False)),
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
        iterator = tqdm(images_info, total=len(images_info), desc="Export pseudo labels", dynamic_ncols=True)

    for img in iterator:
        image_id = int(img.get("id"))
        file_name = str(img.get("file_name", ""))
        height = int(img.get("height", 0))
        width = int(img.get("width", 0))

        image_path = _resolve_image_path(images_root, file_name)
        image_np = _load_image_np(image_path)
        if height <= 0 or width <= 0:
            height, width = int(image_np.shape[0]), int(image_np.shape[1])

        meta = {
            "path": str(image_path),
            "file_name": file_name,
            "height": height,
            "width": width,
            "image_id": image_id,
        }
        preds = teacher.generate([image_np], [meta], image_ids=[image_id])[0]

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
        "meta": {"source": "sam2", "coco_gt": str(coco_gt) if coco_gt else ""},
    }

    train_cfg = cfg.get("data", {}).get("train", {})
    output_path = Path(args.output or train_cfg.get("pseudolabel_path", "pseudolabels.json"))
    if not output_path.is_absolute():
        output_path = data_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("pseudo labels exported to %s", output_path)

    results_path = Path(args.output_results) if args.output_results else None
    if results_path is None:
        stem = output_path.stem
        results_path = output_path.with_name(f"{stem}_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("pseudo label results exported to %s", results_path)


if __name__ == "__main__":
    main()
