"""Torch-based postprocess helpers (batched, GPU-friendly)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from torchvision.ops import box_iou as tv_box_iou
except Exception:  # pragma: no cover
    tv_box_iou = None  # type: ignore


RawMaskPrediction = Dict[str, Any]


@dataclass
class TorchMaskBatch:
    masks: torch.Tensor  # (N, H, W) bool
    indices: List[int]  # indices in original preds
    scores: torch.Tensor  # (N,)
    pred_iou: torch.Tensor  # (N,)
    stability: torch.Tensor  # (N,)


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _score_of_pred(item: Dict[str, Any]) -> Tuple[float, float, float]:
    piou = _to_float(item.get("predicted_iou", None))
    stab = _to_float(item.get("stability_score", None))
    piou_v = 0.0 if piou is None else float(piou)
    stab_v = 0.0 if stab is None else float(stab)
    return float(piou_v * stab_v), piou_v, stab_v


def _extract_mask(item: Dict[str, Any], image_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
    height, width = image_hw
    cand = None
    for key in ("segmentation", "mask", "masks"):
        if key in item:
            cand = item[key]
            break
    if cand is None:
        return None

    if isinstance(cand, torch.Tensor):
        mask = cand
    else:
        mask = torch.as_tensor(cand)

    if mask.ndim == 3 and mask.shape[1:] == (height, width):
        mask = mask[0]

    if mask.ndim != 2 or mask.shape != (height, width):
        return None

    if mask.dtype != torch.bool:
        mask = mask > 0.5
    return mask


def build_mask_batch(
    pred_items: Sequence[RawMaskPrediction],
    image_hw: Tuple[int, int],
    device: torch.device,
) -> TorchMaskBatch:
    masks: List[torch.Tensor] = []
    indices: List[int] = []
    scores: List[float] = []
    piou_list: List[float] = []
    stab_list: List[float] = []
    for idx, raw in enumerate(pred_items):
        if not isinstance(raw, dict):
            continue
        mask = _extract_mask(raw, image_hw)
        if mask is None:
            continue
        score, piou, stab = _score_of_pred(raw)
        masks.append(mask)
        indices.append(int(idx))
        scores.append(score)
        piou_list.append(piou)
        stab_list.append(stab)
    if not masks:
        empty = torch.zeros((0,), device=device)
        return TorchMaskBatch(
            masks=torch.zeros((0, image_hw[0], image_hw[1]), dtype=torch.bool, device=device),
            indices=[],
            scores=empty,
            pred_iou=empty,
            stability=empty,
        )
    masks_t = torch.stack([m.to(device=device, dtype=torch.bool, non_blocking=True) for m in masks], dim=0)
    return TorchMaskBatch(
        masks=masks_t,
        indices=indices,
        scores=torch.tensor(scores, device=device, dtype=torch.float32),
        pred_iou=torch.tensor(piou_list, device=device, dtype=torch.float32),
        stability=torch.tensor(stab_list, device=device, dtype=torch.float32),
    )


def compute_bbox_xyxy(masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # masks: (N, H, W) bool
    n, h, w = masks.shape
    rows_any = masks.any(dim=2)  # (N, H)
    cols_any = masks.any(dim=1)  # (N, W)
    row_idx = torch.arange(h, device=masks.device).view(1, h).expand(n, h)
    col_idx = torch.arange(w, device=masks.device).view(1, w).expand(n, w)

    row_min = torch.where(rows_any, row_idx, torch.full_like(row_idx, h)).min(dim=1).values
    row_max = torch.where(rows_any, row_idx, torch.full_like(row_idx, -1)).max(dim=1).values
    col_min = torch.where(cols_any, col_idx, torch.full_like(col_idx, w)).min(dim=1).values
    col_max = torch.where(cols_any, col_idx, torch.full_like(col_idx, -1)).max(dim=1).values

    # x2/y2 are exclusive (+1)
    x1 = col_min
    y1 = row_min
    x2 = col_max + 1
    y2 = row_max + 1
    return x1, y1, x2, y2


def compute_compactness(masks: torch.Tensor) -> torch.Tensor:
    # compactness = 4*pi*A / P^2
    m = masks.to(torch.bool)
    area = m.sum(dim=(1, 2)).to(torch.float32)
    if area.numel() == 0:
        return area
    up = F.pad(m[:, :-1, :], (0, 0, 1, 0), value=False)
    dn = F.pad(m[:, 1:, :], (0, 0, 0, 1), value=False)
    lf = F.pad(m[:, :, :-1], (1, 0, 0, 0), value=False)
    rt = F.pad(m[:, :, 1:], (0, 1, 0, 0), value=False)
    edge = (m != up) | (m != dn) | (m != lf) | (m != rt)
    perimeter = edge.sum(dim=(1, 2)).to(torch.float32)
    eps = 1e-6
    comp = torch.where(
        perimeter > 0,
        (4.0 * torch.pi * area) / (perimeter * perimeter + eps),
        torch.zeros_like(area),
    )
    return comp


def mask_iou_with_kept(
    masks: torch.Tensor,
    kept_masks: torch.Tensor,
    areas: torch.Tensor,
    kept_areas: torch.Tensor,
) -> torch.Tensor:
    # masks: (1, H, W) or (N, H, W), kept_masks: (K, H, W)
    if kept_masks.numel() == 0:
        return torch.zeros((0,), device=masks.device, dtype=torch.float32)
    inter = (masks & kept_masks).sum(dim=(1, 2)).to(torch.float32)
    union = areas.view(-1, 1) + kept_areas.view(1, -1) - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def greedy_nms_mask(
    masks: torch.Tensor,
    scores: torch.Tensor,
    iou_thr: float,
    topk: int = 0,
) -> List[int]:
    if masks.numel() == 0:
        return []
    n = masks.shape[0]
    if topk > 0 and n > topk:
        scores, idx = torch.topk(scores, k=topk)
        masks = masks[idx]
        base_indices = idx.tolist()
    else:
        base_indices = list(range(n))
    order = torch.argsort(scores, descending=True)
    masks = masks[order]
    ordered_idx = [base_indices[i] for i in order.tolist()]
    areas = masks.sum(dim=(1, 2)).to(torch.float32)
    kept: List[int] = []
    kept_masks: Optional[torch.Tensor] = None
    kept_areas: Optional[torch.Tensor] = None
    for i, idx in enumerate(ordered_idx):
        m = masks[i : i + 1]
        a = areas[i : i + 1]
        if kept_masks is None:
            kept.append(idx)
            kept_masks = m
            kept_areas = a
            continue
        ious = mask_iou_with_kept(m, kept_masks, a, kept_areas)
        max_iou = float(ious.max()) if ious.numel() > 0 else 0.0
        if max_iou <= iou_thr:
            kept.append(idx)
            kept_masks = torch.cat([kept_masks, m], dim=0)
            kept_areas = torch.cat([kept_areas, a], dim=0)
    return kept


def greedy_nms_box(
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    iou_thr: float,
    topk: int = 0,
) -> List[int]:
    if boxes_xyxy.numel() == 0:
        return []
    n = boxes_xyxy.shape[0]
    if topk > 0 and n > topk:
        scores, idx = torch.topk(scores, k=topk)
        boxes_xyxy = boxes_xyxy[idx]
        base_indices = idx.tolist()
    else:
        base_indices = list(range(n))
    order = torch.argsort(scores, descending=True)
    boxes_xyxy = boxes_xyxy[order]
    ordered_idx = [base_indices[i] for i in order.tolist()]
    kept: List[int] = []
    kept_boxes: Optional[torch.Tensor] = None
    for i, idx in enumerate(ordered_idx):
        box = boxes_xyxy[i : i + 1]
        if kept_boxes is None:
            kept.append(idx)
            kept_boxes = box
            continue
        if tv_box_iou is None:
            # fallback: compute IoU manually
            x1 = torch.maximum(box[:, 0], kept_boxes[:, 0])
            y1 = torch.maximum(box[:, 1], kept_boxes[:, 1])
            x2 = torch.minimum(box[:, 2], kept_boxes[:, 2])
            y2 = torch.minimum(box[:, 3], kept_boxes[:, 3])
            inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
            area_a = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
            area_b = (kept_boxes[:, 2] - kept_boxes[:, 0]) * (kept_boxes[:, 3] - kept_boxes[:, 1])
            union = area_a + area_b - inter
            ious = torch.where(union > 0, inter / union, torch.zeros_like(union))
        else:
            ious = tv_box_iou(box, kept_boxes).view(-1)
        max_iou = float(ious.max()) if ious.numel() > 0 else 0.0
        if max_iou <= iou_thr:
            kept.append(idx)
            kept_boxes = torch.cat([kept_boxes, box], dim=0)
    return kept
