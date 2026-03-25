#!/usr/bin/env python
"""Profile hard-distill YOLO inference throughput, latency, and CUDA memory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from ultralytics import YOLO

from sam2real.hard_distill.yolo_runner import DEFAULT_CONFIG, load_yaml_config, normalize_runtime_config
from sam2real.utils.paths import resolve_project_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile hard-distill YOLO inference FPS, latency, and CUDA memory.")
    parser.add_argument("--cfg", type=str, default=str(DEFAULT_CONFIG), help="Hard-distill config YAML.")
    parser.add_argument("--weights", "--model", dest="weights", type=str, default="", help="Checkpoint path.")
    parser.add_argument("--data", type=str, default="", help="Dataset YAML path. Defaults to cfg.data.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to profile: train/val/test/pseudo.")
    parser.add_argument("--limit", type=int, default=0, help="Limit profiled images. 0 means all images in the split.")
    parser.add_argument("--batch", type=int, default=0, help="Override inference batch size. 0 uses cfg.batch.")
    parser.add_argument("--imgsz", type=int, default=0, help="Override inference image size. 0 uses cfg.imgsz.")
    parser.add_argument("--device", type=str, default="", help="Override runtime device, e.g. 0 or cpu.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup passes before measuring.")
    parser.add_argument("--repeat", type=int, default=1, help="Measured passes over the selected images.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="YOLO max detections per image.")
    parser.add_argument(
        "--out-json",
        type=str,
        default="debug/outputs/hard_distill_yolo_inference_profile.json",
        help="Path to save the profiling summary JSON.",
    )
    return parser.parse_args()


def _resolve_text_path(value: str) -> Path:
    return Path(resolve_project_path(value))


def _normalize_device(value: str | int | None) -> str:
    if value is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    text = str(value).strip()
    if not text:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if text == "cuda":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return text


def _torch_device_for_runtime(device: str) -> torch.device:
    text = str(device).strip().lower()
    if text.isdigit():
        return torch.device(f"cuda:{int(text)}")
    if text.startswith("cuda"):
        return torch.device(text)
    return torch.device(text or "cpu")


def _is_cuda_device(device: str) -> bool:
    text = str(device).strip().lower()
    return torch.cuda.is_available() and (text.startswith("cuda") or text.isdigit())


def _maybe_sync(device: str) -> None:
    if _is_cuda_device(device):
        torch.cuda.synchronize(_torch_device_for_runtime(device))


def _collect_images_from_dir(images_dir: Path) -> List[Path]:
    return sorted([path.resolve() for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES])


def _load_split_entries(data_yaml_path: Path, split: str) -> List[Path]:
    payload = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Data YAML must be a mapping: {data_yaml_path}")
    if split not in payload:
        raise KeyError(f"Split '{split}' not found in data yaml: {data_yaml_path}")

    dataset_root = Path(str(payload.get("path", data_yaml_path.parent))).resolve()
    split_value = payload[split]
    split_path = Path(str(split_value))
    if not split_path.is_absolute():
        split_path = (dataset_root / split_path).resolve()

    if split_path.is_file():
        image_paths = [
            Path(line.strip()).resolve()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return image_paths
    if split_path.is_dir():
        return _collect_images_from_dir(split_path)
    raise FileNotFoundError(f"Split path not found: {split_path}")


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _iter_batches(items: List[np.ndarray], batch_size: int) -> Iterable[List[np.ndarray]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _count_detections(results: List[Any]) -> int:
    total = 0
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or not hasattr(boxes, "__len__"):
            continue
        total += len(boxes)
    return int(total)


def _run_pass(
    model: YOLO,
    images: List[np.ndarray],
    *,
    batch_size: int,
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
) -> Dict[str, Any]:
    batch_times: List[float] = []
    total_detections = 0
    for batch in _iter_batches(images, batch_size):
        _maybe_sync(device)
        t0 = time.perf_counter()
        results = model.predict(
            batch,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            verbose=False,
        )
        _maybe_sync(device)
        t1 = time.perf_counter()
        batch_times.append(t1 - t0)
        total_detections += _count_detections(results)
    return {
        "batch_times_s": batch_times,
        "total_time_s": float(sum(batch_times)),
        "total_detections": int(total_detections),
    }


def main() -> None:
    args = parse_args()

    cfg_path = _resolve_text_path(args.cfg)
    merged = load_yaml_config(cfg_path)
    if str(args.weights).strip():
        merged["model"] = str(args.weights)
    if str(args.data).strip():
        merged["data"] = str(args.data)
    if int(args.batch) > 0:
        merged["batch"] = int(args.batch)
    if int(args.imgsz) > 0:
        merged["imgsz"] = int(args.imgsz)
    if str(args.device).strip():
        merged["device"] = str(args.device)
    merged = normalize_runtime_config(merged)

    if "model" not in merged:
        raise KeyError("Hard-distill config must define `model`, or pass --weights.")
    if "data" not in merged:
        raise KeyError("Hard-distill config must define `data`, or pass --data.")

    weights_path = Path(str(merged["model"])).resolve()
    data_yaml_path = Path(str(merged["data"])).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_yaml_path}")

    batch_size = max(1, int(merged.get("batch", 1)))
    imgsz = int(merged.get("imgsz", 640))
    device = _normalize_device(merged.get("device"))
    split = str(args.split).strip()

    image_paths = _load_split_entries(data_yaml_path, split=split)
    if int(args.limit) > 0:
        image_paths = image_paths[: int(args.limit)]
    if not image_paths:
        raise RuntimeError(f"No images found for split='{split}' in {data_yaml_path}")

    images = [_load_image(path) for path in image_paths]
    model = YOLO(str(weights_path))

    for _ in range(int(args.warmup)):
        _run_pass(
            model,
            images,
            batch_size=batch_size,
            imgsz=imgsz,
            device=device,
            conf=float(args.conf),
            iou=float(args.iou),
            max_det=int(args.max_det),
        )

    if _is_cuda_device(device):
        device_obj = _torch_device_for_runtime(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_obj)
    else:
        device_obj = None

    measured_batch_times: List[float] = []
    total_detections = 0
    total_images = len(images) * int(args.repeat)

    for _ in range(int(args.repeat)):
        stats = _run_pass(
            model,
            images,
            batch_size=batch_size,
            imgsz=imgsz,
            device=device,
            conf=float(args.conf),
            iou=float(args.iou),
            max_det=int(args.max_det),
        )
        measured_batch_times.extend(stats["batch_times_s"])
        total_detections += int(stats["total_detections"])

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
        "cfg": str(cfg_path),
        "weights": str(weights_path),
        "data": str(data_yaml_path),
        "split": split,
        "device": device,
        "timing_scope": "ultralytics_yolo_predict_only_excludes_image_decode",
        "num_unique_images": int(len(images)),
        "num_profiled_images": int(total_images),
        "batch_size": int(batch_size),
        "imgsz": int(imgsz),
        "warmup_passes": int(args.warmup),
        "repeat_passes": int(args.repeat),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "max_det": int(args.max_det),
        "total_time_s": total_time_s,
        "avg_time_per_image_ms": avg_time_per_image_ms,
        "avg_time_per_batch_ms": avg_time_per_batch_ms,
        "fps": fps,
        "total_detections": int(total_detections),
        "avg_detections_per_image": (float(total_detections) / float(total_images)) if total_images > 0 else 0.0,
        "peak_gpu_memory_allocated_mb": peak_allocated_mb,
        "peak_gpu_memory_reserved_mb": peak_reserved_mb,
        "batch_times_s": measured_batch_times,
        "image_paths": [str(path) for path in image_paths],
    }

    out_path = _resolve_text_path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"FPS: {fps:.6f}")
    print(f"Average time per image (ms): {avg_time_per_image_ms:.6f}")
    print(f"Average time per batch (ms): {avg_time_per_batch_ms:.6f}")
    print(f"Peak GPU memory allocated (MB): {peak_allocated_mb:.2f}")
    print(f"Peak GPU memory reserved (MB): {peak_reserved_mb:.2f}")
    print(f"Summary JSON: {out_path}")


if __name__ == "__main__":
    main()
