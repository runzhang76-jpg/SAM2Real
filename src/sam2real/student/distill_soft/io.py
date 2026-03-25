"""Teacher soft file loading, aggregation, and geometric alignment helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch optional during static inspection
    torch = None
    F = None  # type: ignore


def _as_float_array(value: Any) -> np.ndarray:
    return np.array(value, dtype=np.float32, copy=True)


def resolve_teacher_soft_path(teacher_soft_dir: Path, file_name: str) -> Optional[Path]:
    if not file_name:
        return None
    relative = Path(file_name)
    candidates = [teacher_soft_dir / relative.with_suffix(".npz")]
    if relative.parent != Path("."):
        candidates.append(teacher_soft_dir / f"{relative.stem}.npz")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_teacher_soft_sample(npz_path: Path) -> Optional[Dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for teacher soft loading.")
    if not npz_path.exists():
        return None

    with np.load(npz_path, allow_pickle=False) as payload:
        if "mask_logits" not in payload:
            return None

        mask_logits = _as_float_array(payload["mask_logits"])
        if mask_logits.ndim == 2:
            mask_logits = mask_logits[None, ...]
        if mask_logits.ndim != 3:
            raise ValueError(f"Unexpected mask_logits shape in {npz_path}: {mask_logits.shape}")

        num_instances = int(mask_logits.shape[0])
        scores = _as_float_array(payload["score"]) if "score" in payload else np.ones((num_instances,), dtype=np.float32)
        scores = scores.reshape(-1)
        if scores.size < num_instances:
            pad = np.ones((num_instances - scores.size,), dtype=np.float32)
            scores = np.concatenate([scores, pad], axis=0)
        scores = scores[:num_instances]

        if num_instances > 0:
            img_mask_logits = mask_logits.max(axis=0)
        else:
            img_mask_logits = np.zeros(mask_logits.shape[-2:], dtype=np.float32)

        aggregated: Dict[str, Any] = {
            "img_mask_logits": torch.from_numpy(np.array(img_mask_logits, dtype=np.float32, copy=True)),
            "img_score": torch.tensor(float(scores.mean()) if scores.size else 0.0, dtype=torch.float32),
            "source_path": str(npz_path),
        }

        if "boundary_map" in payload:
            boundary_map = _as_float_array(payload["boundary_map"])
            if boundary_map.ndim == 2:
                boundary_map = boundary_map[None, ...]
            if boundary_map.ndim == 3:
                keep = min(boundary_map.shape[0], num_instances)
                if keep > 0:
                    aggregated["img_boundary"] = torch.from_numpy(np.array(boundary_map[:keep].max(axis=0), dtype=np.float32, copy=True))

        if "class_soft" in payload:
            class_soft = _as_float_array(payload["class_soft"])
            if class_soft.ndim == 1:
                class_soft = class_soft[None, ...]
            if class_soft.ndim == 2 and class_soft.shape[0] > 0:
                keep = min(class_soft.shape[0], num_instances, scores.shape[0])
                probs = class_soft[:keep]
                weights = scores[:keep]
                denom = float(weights.sum())
                if denom > 1e-6:
                    img_class_soft = (probs * weights[:, None]).sum(axis=0) / denom
                else:
                    img_class_soft = probs.mean(axis=0)
                aggregated["img_class_soft"] = torch.from_numpy(np.array(img_class_soft, dtype=np.float32, copy=True))
        if "class_ids" in payload:
            aggregated["class_ids"] = torch.from_numpy(np.array(payload["class_ids"], dtype=np.int64, copy=True))

    return aggregated


def resize_teacher_soft(teacher_soft: Dict[str, Any], size: Tuple[int, int]) -> Dict[str, Any]:
    if torch is None or F is None:
        return teacher_soft
    new_h, new_w = size
    for key in ("img_mask_logits", "img_boundary"):
        value = teacher_soft.get(key)
        if value is None:
            continue
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected teacher soft map shape for {key}: {tuple(tensor.shape)}")
        resized = F.interpolate(tensor.float(), size=(new_h, new_w), mode="bilinear", align_corners=False)
        teacher_soft[key] = resized.squeeze(0).squeeze(0)
    return teacher_soft


def flip_teacher_soft(teacher_soft: Dict[str, Any]) -> Dict[str, Any]:
    if torch is None:
        return teacher_soft
    for key in ("img_mask_logits", "img_boundary"):
        value = teacher_soft.get(key)
        if value is None:
            continue
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
        teacher_soft[key] = torch.flip(tensor, dims=[-1])
    return teacher_soft
