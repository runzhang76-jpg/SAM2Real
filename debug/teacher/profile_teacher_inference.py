#!/usr/bin/env python
"""Profile teacher inference throughput, latency, and CUDA memory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from matmatch2real.config.loader import load_config
from matmatch2real.data.datasets import build_teacher_dataset
from matmatch2real.teacher import SAM2Teacher
from matmatch2real.utils.logging import setup_logger
from matmatch2real.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile teacher inference FPS, latency, and CUDA memory.")
    parser.add_argument("--config", default="configs/teacher/teacher_default.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Limit profiled images. 0 means all available images.")
    parser.add_argument("--batch-size", type=int, default=0, help="Override teacher batch size. 0 uses config.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup passes before measuring.")
    parser.add_argument("--repeat", type=int, default=1, help="Measured passes over the selected images.")
    parser.add_argument("--device", default="", help="Override teacher.sam2.device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--out-json",
        default="debug/outputs/teacher_inference_profile.json",
        help="Path to save the profiling summary JSON.",
    )
    return parser.parse_args()


def _resolve_runtime_device(cfg: Dict[str, Any], override: str) -> str:
    if str(override).strip():
        return str(override).strip()
    teacher_cfg = cfg.get("teacher", {})
    sam_cfg = teacher_cfg.get("sam2", {})
    device = str(sam_cfg.get("device", cfg.get("experiment", {}).get("device", "cpu"))).strip()
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_available():
        return "cuda:0"
    return device or "cpu"


def _is_cuda_device(device: str) -> bool:
    return str(device).lower().startswith("cuda") and torch.cuda.is_available()


def _maybe_sync(device: str) -> None:
    if _is_cuda_device(device):
        torch.cuda.synchronize(torch.device(device))


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _collect_samples(cfg: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    dataset = build_teacher_dataset(cfg, PROJECT_ROOT)
    total = len(dataset)
    if limit > 0:
        total = min(total, int(limit))
    samples: List[Dict[str, Any]] = []
    for index in range(total):
        sample = dataset[index]
        meta = dict(sample.get("meta", {}))
        image_id = int(sample.get("image_id", meta.get("image_id", index)))
        meta["image_id"] = image_id
        samples.append({"image_id": image_id, "meta": meta})
    return samples


def _iter_batches(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _prepare_batch(batch: List[Dict[str, Any]]) -> tuple[List[np.ndarray], List[Dict[str, Any]], List[int]]:
    images: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    image_ids: List[int] = []
    for item in batch:
        meta = dict(item["meta"])
        path_text = str(meta.get("path", "")).strip()
        if not path_text:
            raise ValueError(f"Teacher sample for image_id={item['image_id']} is missing meta.path")
        image_path = Path(path_text)
        if not image_path.exists():
            raise FileNotFoundError(f"Teacher image not found: {image_path}")
        image_np = _load_image(image_path)
        meta["height"] = int(image_np.shape[0])
        meta["width"] = int(image_np.shape[1])
        meta["file_name"] = meta.get("file_name", image_path.name)
        meta["path"] = str(image_path)
        images.append(image_np)
        metas.append(meta)
        image_ids.append(int(item["image_id"]))
    return images, metas, image_ids


def _run_pass(teacher: SAM2Teacher, samples: List[Dict[str, Any]], batch_size: int, device: str) -> Dict[str, Any]:
    batch_times: List[float] = []
    total_instances = 0
    for batch in _iter_batches(samples, batch_size):
        images, metas, image_ids = _prepare_batch(batch)
        _maybe_sync(device)
        t0 = time.perf_counter()
        outputs = teacher.generate(images, metas, image_ids=image_ids)
        _maybe_sync(device)
        t1 = time.perf_counter()
        batch_times.append(t1 - t0)
        total_instances += sum(len(instances) for instances in outputs)
    return {
        "batch_times_s": batch_times,
        "total_time_s": float(sum(batch_times)),
        "total_instances": int(total_instances),
    }


def main() -> None:
    args = parse_args()
    logger = setup_logger("distill")

    cfg = load_config(args.config)
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    runtime_device = _resolve_runtime_device(cfg, args.device)
    teacher_cfg = cfg.setdefault("teacher", {})
    if str(args.device).strip():
        teacher_cfg.setdefault("sam2", {})
        teacher_cfg["sam2"]["device"] = runtime_device

    teacher = SAM2Teacher(teacher_cfg, device=runtime_device)
    if teacher.adapter is None:
        raise RuntimeError("Teacher adapter is not configured; cannot profile inference.")

    samples = _collect_samples(cfg, limit=int(args.limit))
    if not samples:
        raise RuntimeError("No teacher samples found for profiling.")

    config_batch_size = int(cfg.get("data", {}).get("teacher", {}).get("batch_size", 1))
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else max(1, config_batch_size)

    logger.info(
        "teacher profile setup: images=%d batch_size=%d warmup=%d repeat=%d device=%s",
        len(samples),
        batch_size,
        int(args.warmup),
        int(args.repeat),
        runtime_device,
    )

    for warmup_idx in range(int(args.warmup)):
        stats = _run_pass(teacher, samples, batch_size=batch_size, device=runtime_device)
        logger.info(
            "warmup %d/%d: time=%.3fs instances=%d",
            warmup_idx + 1,
            int(args.warmup),
            stats["total_time_s"],
            stats["total_instances"],
        )

    if _is_cuda_device(runtime_device):
        device_obj = torch.device(runtime_device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_obj)
    else:
        device_obj = None

    measured_batch_times: List[float] = []
    total_instances = 0
    total_images = len(samples) * int(args.repeat)

    for repeat_idx in range(int(args.repeat)):
        stats = _run_pass(teacher, samples, batch_size=batch_size, device=runtime_device)
        measured_batch_times.extend(stats["batch_times_s"])
        total_instances += int(stats["total_instances"])
        logger.info(
            "profile pass %d/%d: time=%.3fs instances=%d",
            repeat_idx + 1,
            int(args.repeat),
            stats["total_time_s"],
            stats["total_instances"],
        )

    total_time_s = float(sum(measured_batch_times))
    avg_time_per_image_ms = (total_time_s / total_images * 1000.0) if total_images > 0 else 0.0
    avg_time_per_batch_ms = (total_time_s / len(measured_batch_times) * 1000.0) if measured_batch_times else 0.0
    fps = (total_images / total_time_s) if total_time_s > 0 else 0.0

    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
    if device_obj is not None:
        peak_allocated_mb = float(torch.cuda.max_memory_allocated(device_obj) / (1024.0 ** 2))
        peak_reserved_mb = float(torch.cuda.max_memory_reserved(device_obj) / (1024.0 ** 2))

    summary = {
        "config": str(Path(args.config)),
        "device": runtime_device,
        "timing_scope": "teacher.generate_only_excludes_image_decode",
        "num_unique_images": int(len(samples)),
        "num_profiled_images": int(total_images),
        "batch_size": int(batch_size),
        "warmup_passes": int(args.warmup),
        "repeat_passes": int(args.repeat),
        "total_time_s": total_time_s,
        "avg_time_per_image_ms": avg_time_per_image_ms,
        "avg_time_per_batch_ms": avg_time_per_batch_ms,
        "fps": fps,
        "total_instances": int(total_instances),
        "avg_instances_per_image": (float(total_instances) / float(total_images)) if total_images > 0 else 0.0,
        "peak_gpu_memory_allocated_mb": peak_allocated_mb,
        "peak_gpu_memory_reserved_mb": peak_reserved_mb,
        "batch_times_s": measured_batch_times,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("fps=%.3f", fps)
    logger.info("avg_time_per_image_ms=%.3f", avg_time_per_image_ms)
    logger.info("avg_time_per_batch_ms=%.3f", avg_time_per_batch_ms)
    logger.info("peak_gpu_memory_allocated_mb=%.2f", peak_allocated_mb)
    logger.info("peak_gpu_memory_reserved_mb=%.2f", peak_reserved_mb)
    logger.info("saved teacher profile summary to %s", out_path)


if __name__ == "__main__":
    main()
