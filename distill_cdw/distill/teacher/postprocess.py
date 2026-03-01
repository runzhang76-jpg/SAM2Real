"""SAM2 实例提案的后处理工具与可插拔流水线。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Protocol

import numpy as np

from distill.core.structures import InstancePrediction
from distill.utils.logging import get_logger

try:
    from pycocotools import mask as mask_utils

    _HAS_COCO = True
except Exception:
    _HAS_COCO = False

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - 可选依赖
    torch = None  # type: ignore
    F = None  # type: ignore

try:
    from distill.teacher.postprocess_torch import (
        TorchMaskBatch,
        build_mask_batch,
        compute_bbox_xyxy,
        compute_compactness,
        greedy_nms_box,
        greedy_nms_mask,
    )
except Exception:  # pragma: no cover - 可选依赖
    TorchMaskBatch = None  # type: ignore
    build_mask_batch = None  # type: ignore
    compute_bbox_xyxy = None  # type: ignore
    compute_compactness = None  # type: ignore
    greedy_nms_box = None  # type: ignore
    greedy_nms_mask = None  # type: ignore

RawMaskPrediction = Dict[str, Any]

__all__ = [
    "MaskItem",
    "filter_sam2_masks",
    "convert_instances",
    "IdentityPostProcess",
    "PostProcessPipeline",
    "PostProcessStep",
    "PostProcessRuntimeConfig",
]


@dataclass
class MaskItem:
    """
    过滤后的实例 mask 数据结构。
    """

    mask: np.ndarray  # (H, W) bool
    bbox_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2) 半开区间
    area: int
    score: float
    predicted_iou: Optional[float]
    stability_score: Optional[float]
    src_index: int
    meta: Dict[str, Any]


@dataclass
class PostProcessRuntimeConfig:
    """运行时后处理配置（支持 torch 路径）。"""

    use_torch: bool = False
    device: str = "cpu"
    topk_pre: int = 0
    torch_fp16: bool = False
    torch_nms: bool = True
    topk_nms: int = 0
    log_stats: bool = False


class PostProcessStep(Protocol):
    """后处理步骤统一接口。"""

    name: str

    def __call__(self, preds: List[RawMaskPrediction], image_meta: Dict[str, Any]) -> List[RawMaskPrediction]:
        ...


def _get_image_hw(image_meta: Dict[str, Any]) -> Tuple[int, int]:
    height = int(image_meta.get("height", 0))
    width = int(image_meta.get("width", 0))
    return height, width


def _extract_mask(item: Dict[str, Any], image_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """从 item 中提取 mask，返回 shape=(H, W) 的 bool mask。"""

    height, width = image_hw
    cand = None
    for key in ("segmentation", "mask", "masks"):
        if key in item:
            cand = item[key]
            break
    if cand is None:
        return None

    mask = np.asarray(cand)

    if mask.ndim == 3 and mask.shape[1:] == (height, width):
        mask = mask[0]

    if mask.ndim != 2 or mask.shape != (height, width):
        return None

    if mask.dtype != np.bool_:
        mask = mask > 0.5

    return mask


def _build_mask_items(
    pred_items: Sequence[RawMaskPrediction],
    image_hw: Tuple[int, int],
    *,
    keep_meta: bool = True,
) -> List[MaskItem]:
    """将原始预测解析为 MaskItem 列表（不做过滤/去重）。"""

    items: List[MaskItem] = []
    for idx, raw in enumerate(pred_items):
        if not isinstance(raw, dict):
            continue
        mask = _extract_mask(raw, image_hw)
        if mask is None:
            continue
        bbox = _compute_bbox_xyxy(mask)
        piou, stab, score = _get_score_fields(raw)
        meta = dict(raw) if keep_meta else {}
        for key in ("segmentation", "mask", "masks"):
            meta.pop(key, None)
        items.append(
            MaskItem(
                mask=mask,
                bbox_xyxy=bbox,
                area=int(mask.sum()),
                score=float(score),
                predicted_iou=piou,
                stability_score=stab,
                src_index=int(idx),
                meta=meta,
            )
        )
    return items


def _compute_bbox_xyxy(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return (0, 0, 0, 0)
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return (x1, y1, x2, y2)


def _bbox_wh(b: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1), max(0, y2 - y1)


def _compactness(mask: np.ndarray) -> float:
    """紧致度：4πA/P^2（越细长/锯齿越小）。"""

    m = mask.astype(np.uint8)
    area = float(m.sum())
    if area <= 0:
        return 0.0

    up = np.pad(m[:-1, :], ((1, 0), (0, 0)), mode="constant")
    dn = np.pad(m[1:, :], ((0, 1), (0, 0)), mode="constant")
    lf = np.pad(m[:, :-1], ((0, 0), (1, 0)), mode="constant")
    rt = np.pad(m[:, 1:], ((0, 0), (0, 1)), mode="constant")

    edge = (m != up) + (m != dn) + (m != lf) + (m != rt)
    perimeter = float(edge.sum())
    if perimeter <= 0:
        return 0.0

    return float(4.0 * np.pi * area / (perimeter * perimeter))


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _get_score_fields(item: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], float]:
    """
    定义 score = predicted_iou * stability_score。
    """

    piou = _to_float(item.get("predicted_iou", None))
    stab = _to_float(item.get("stability_score", None))

    piou_v = 0.0 if piou is None else float(piou)
    stab_v = 0.0 if stab is None else float(stab)

    score = piou_v * stab_v
    return piou, stab, float(score)


def _score_of_pred(item: Dict[str, Any]) -> float:
    _, _, score = _get_score_fields(item)
    return float(score)


def _parse_runtime_cfg(cfg: Dict[str, Any]) -> PostProcessRuntimeConfig:
    runtime = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
    return PostProcessRuntimeConfig(
        use_torch=bool(runtime.get("use_torch", False)),
        device=str(runtime.get("device", "cpu")),
        topk_pre=int(runtime.get("topk_pre", 0)),
        torch_fp16=bool(runtime.get("torch_fp16", False)),
        torch_nms=bool(runtime.get("torch_nms", True)),
        topk_nms=int(runtime.get("topk_nms", 0)),
        log_stats=bool(runtime.get("log_stats", False)),
    )


def _encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    if not _HAS_COCO:
        return {}
    m = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(m)
    if isinstance(rle.get("counts", None), bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _mask_iou_np(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum(dtype=np.int64)
    return float(inter) / float(union) if union > 0 else 0.0


def _greedy_nms(items: List[MaskItem], iou_thr: float) -> List[MaskItem]:
    """mask 级贪心 NMS。"""

    if not items:
        return []

    items_sorted = sorted(items, key=lambda x: x.score, reverse=True)
    keep: List[MaskItem] = []

    if _HAS_COCO:
        rles = [_encode_rle(it.mask) for it in items_sorted]
        kept_rles: List[Dict[str, Any]] = []

        for it, rle in zip(items_sorted, rles):
            if not kept_rles:
                keep.append(it)
                kept_rles.append(rle)
                continue
            ious = mask_utils.iou(kept_rles, [rle], [0])
            max_iou = float(np.max(ious)) if ious.size else 0.0
            if max_iou <= iou_thr:
                keep.append(it)
                kept_rles.append(rle)
        return keep

    for it in items_sorted:
        ok = True
        for kt in keep:
            if _mask_iou_np(it.mask, kt.mask) > iou_thr:
                ok = False
                break
        if ok:
            keep.append(it)
    return keep


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def filter_sam2_masks(
    pred_items: Sequence[RawMaskPrediction],
    image_hw: Tuple[int, int],
    *,
    min_area: int = 200,
    max_area_frac: float = 0.20,
    min_score: float = 0.0,
    min_pred_iou: Optional[float] = None,
    min_stability: Optional[float] = None,
    min_bbox_side: int = 20,
    max_aspect_ratio: float = 8.0,
    min_compactness: float = 0.005,
    keep_meta: bool = True,
) -> List[MaskItem]:
    """
    质量过滤，输出 MaskItem 列表
    """

    height, width = image_hw
    img_area = float(height * width)

    items = _build_mask_items(pred_items, image_hw, keep_meta=keep_meta)
    filtered: List[MaskItem] = []

    for item in items:
        area = item.area
        if area <= 0:
            continue
        if area < int(min_area):
            continue
        if area > int(max_area_frac * img_area):
            continue

        bw, bh = _bbox_wh(item.bbox_xyxy)
        if bw < min_bbox_side or bh < min_bbox_side:
            continue
        ar = max(bw, bh) / max(1, min(bw, bh))
        if ar > max_aspect_ratio:
            continue

        comp = _compactness(item.mask)
        if comp < min_compactness:
            continue

        if item.score < float(min_score):
            continue
        if min_pred_iou is not None and (item.predicted_iou is None or item.predicted_iou < float(min_pred_iou)):
            continue
        if min_stability is not None and (item.stability_score is None or item.stability_score < float(min_stability)):
            continue

        filtered.append(item)

    return filtered


def _filter_sam2_masks_torch(
    pred_items: Sequence[RawMaskPrediction],
    image_hw: Tuple[int, int],
    *,
    min_area: int,
    max_area_frac: float,
    min_score: float,
    min_pred_iou: Optional[float],
    min_stability: Optional[float],
    min_bbox_side: int,
    max_aspect_ratio: float,
    min_compactness: float,
    runtime: PostProcessRuntimeConfig,
) -> List[int]:
    if torch is None or build_mask_batch is None or compute_bbox_xyxy is None or compute_compactness is None:
        return []
    device = torch.device(runtime.device if runtime.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    batch = build_mask_batch(pred_items, image_hw, device=device)
    if batch.masks.numel() == 0:
        return []

    masks = batch.masks
    scores = batch.scores
    areas = masks.sum(dim=(1, 2)).to(torch.int64)
    img_area = float(image_hw[0] * image_hw[1])

    x1, y1, x2, y2 = compute_bbox_xyxy(masks)
    bw = (x2 - x1).clamp(min=0)
    bh = (y2 - y1).clamp(min=0)
    ar = torch.maximum(bw, bh).to(torch.float32) / torch.maximum(torch.minimum(bw, bh), torch.ones_like(bw)).to(torch.float32)

    compact = compute_compactness(masks)

    keep = torch.ones((masks.shape[0],), device=device, dtype=torch.bool)
    keep &= areas > 0
    keep &= areas >= int(min_area)
    keep &= areas <= int(max_area_frac * img_area)
    keep &= bw >= int(min_bbox_side)
    keep &= bh >= int(min_bbox_side)
    keep &= ar <= float(max_aspect_ratio)
    keep &= compact >= float(min_compactness)
    keep &= scores >= float(min_score)

    if min_pred_iou is not None:
        keep &= batch.pred_iou >= float(min_pred_iou)
    if min_stability is not None:
        keep &= batch.stability >= float(min_stability)

    kept_indices = [batch.indices[i] for i in keep.nonzero(as_tuple=False).view(-1).tolist()]
    return kept_indices


def _nms_torch(
    pred_items: Sequence[RawMaskPrediction],
    image_hw: Tuple[int, int],
    *,
    method: str,
    iou_thr: float,
    min_score: float,
    runtime: PostProcessRuntimeConfig,
) -> List[int]:
    if torch is None or build_mask_batch is None:
        return []
    device = torch.device(runtime.device if runtime.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    batch = build_mask_batch(pred_items, image_hw, device=device)
    if batch.masks.numel() == 0:
        return []
    scores = batch.scores
    keep_mask = scores >= float(min_score)
    if keep_mask.sum().item() == 0:
        return []
    masks = batch.masks[keep_mask]
    scores = scores[keep_mask]
    indices = [batch.indices[i] for i in keep_mask.nonzero(as_tuple=False).view(-1).tolist()]

    if method == "box":
        x1, y1, x2, y2 = compute_bbox_xyxy(masks)
        boxes = torch.stack([x1, y1, x2, y2], dim=1).to(torch.float32)
        kept_local = greedy_nms_box(boxes, scores, iou_thr=float(iou_thr), topk=int(runtime.topk_nms))
    else:
        kept_local = greedy_nms_mask(masks, scores, iou_thr=float(iou_thr), topk=int(runtime.topk_nms))

    kept_indices = [indices[i] for i in kept_local]
    return kept_indices


def _bbox_xyxy_to_xywh(bbox_xyxy: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def convert_instances(
    pred_items: Sequence[RawMaskPrediction],
    image_hw: Tuple[int, int],
    *,
    image_id: int,
    class_id: int = 0,
    cfg: Optional[Dict[str, Any]] = None,
    encode_rle: bool = False,
) -> List[InstancePrediction]:
    """
    将 SAM2 原始输出转换为 InstancePrediction。

    输入格式（单个元素为 dict，字段兼容 SAM2AutomaticMaskGenerator 输出）：
    - segmentation / mask / masks: (H, W) bool/uint8/float
    - predicted_iou: Optional[float]
    - stability_score: Optional[float]

    输出格式：
    - InstancePrediction.bbox: (x, y, w, h)
    - InstancePrediction.mask: (H, W) bool
    - InstancePrediction.score: predicted_iou * stability_score

    注意：该函数仅做格式转换，不负责策略性后处理（filter/NMS）。
    """

    cfg = cfg or {}
    instances: List[InstancePrediction] = []
    items: Sequence[RawMaskPrediction] = pred_items

    for item in items:
        if not isinstance(item, dict):
            continue
        mask = _extract_mask(item, image_hw)
        if mask is None:
            continue
        bbox_xyxy = _compute_bbox_xyxy(mask)
        if bbox_xyxy == (0, 0, 0, 0):
            continue
        _, _, score = _get_score_fields(item)
        meta = dict(item)
        for key in ("segmentation", "mask", "masks"):
            meta.pop(key, None)
        rle = _encode_rle(mask) if encode_rle else None
        instances.append(
            InstancePrediction(
                image_id=int(image_id),
                bbox=_bbox_xyxy_to_xywh(bbox_xyxy),
                class_id=int(class_id),
                score=float(score),
                reliability=float(score),
                mask=mask,
                rle=rle,
                meta=meta,
            )
        )
    return instances


class IdentityPostProcess:
    """无后处理占位，保持输入原样返回。"""

    name = "identity"

    def __call__(self, preds: List[RawMaskPrediction], image_meta: Dict[str, Any]) -> List[RawMaskPrediction]:
        return preds


class QualityFilterStep:
    """质量过滤：面积/稳定性/预测 IoU 等规则。"""

    name = "quality_filter"

    def __init__(self, cfg: Dict[str, Any], runtime: Optional[PostProcessRuntimeConfig] = None) -> None:
        self.cfg = cfg  # step
        self.runtime = runtime

    def __call__(self, preds: List[RawMaskPrediction], image_meta: Dict[str, Any]) -> List[RawMaskPrediction]:
        height, width = _get_image_hw(image_meta)
        if height <= 0 or width <= 0:
            return preds
        if self.runtime and self.runtime.use_torch and torch is not None:
            kept = set(
                _filter_sam2_masks_torch(
                    preds,
                    image_hw=(height, width),
                    min_area=int(self.cfg.get("min_area", 200)),
                    max_area_frac=float(self.cfg.get("max_area_frac", 0.20)),
                    min_score=float(self.cfg.get("min_score", 0.0)),
                    min_pred_iou=self.cfg.get("min_pred_iou", None),
                    min_stability=self.cfg.get("min_stability", None),
                    min_bbox_side=int(self.cfg.get("min_bbox_side", 20)),
                    max_aspect_ratio=float(self.cfg.get("max_aspect_ratio", 8.0)),
                    min_compactness=float(self.cfg.get("min_compactness", 0.005)),
                    runtime=self.runtime,
                )
            )
        else:
            items = filter_sam2_masks(
                preds,
                image_hw=(height, width),
                min_area=int(self.cfg.get("min_area", 200)),
                max_area_frac=float(self.cfg.get("max_area_frac", 0.20)),
                min_score=float(self.cfg.get("min_score", 0.0)),
                min_pred_iou=self.cfg.get("min_pred_iou", None),
                min_stability=self.cfg.get("min_stability", None),
                min_bbox_side=int(self.cfg.get("min_bbox_side", 20)),
                max_aspect_ratio=float(self.cfg.get("max_aspect_ratio", 8.0)),
                min_compactness=float(self.cfg.get("min_compactness", 0.005)),
                keep_meta=True,
            )
            kept = {it.src_index for it in items}
        return [preds[i] for i in range(len(preds)) if i in kept]


class NmsStep:
    """NMS 去重（支持 mask/box 两种方式）。"""

    name = "nms"

    def __init__(self, cfg: Dict[str, Any], runtime: Optional[PostProcessRuntimeConfig] = None) -> None:
        self.cfg = cfg
        self.runtime = runtime

    def __call__(self, preds: List[RawMaskPrediction], image_meta: Dict[str, Any]) -> List[RawMaskPrediction]:
        height, width = _get_image_hw(image_meta)
        if height <= 0 or width <= 0:
            return preds
        method = str(self.cfg.get("method", "mask")).lower()
        iou_thr = float(self.cfg.get("iou_thresh", self.cfg.get("nms_iou_thr", 0.85)))

        min_score = float(self.cfg.get("min_score", 0.0))
        if self.runtime and self.runtime.use_torch and self.runtime.torch_nms and torch is not None:
            kept = set(
                _nms_torch(
                    preds,
                    image_hw=(height, width),
                    method=method,
                    iou_thr=iou_thr,
                    min_score=min_score,
                    runtime=self.runtime,
                )
            )
        else:
            items = _build_mask_items(preds, image_hw=(height, width), keep_meta=True)
            if min_score > 0:
                items = [it for it in items if it.score >= min_score]

            # nms去重
            if method == "box":
                items_sorted = sorted(items, key=lambda x: x.score, reverse=True)
                keep: List[MaskItem] = []
                for it in items_sorted:
                    if all(_bbox_iou(it.bbox_xyxy, kt.bbox_xyxy) <= iou_thr for kt in keep):
                        keep.append(it)
                kept = {it.src_index for it in keep}
            else:
                kept = {it.src_index for it in _greedy_nms(items, iou_thr=iou_thr)}
        return [preds[i] for i in range(len(preds)) if i in kept]


class PostProcessPipeline:
    """可配置的后处理流水线。"""

    _STEP_REGISTRY = {
        "quality_filter": QualityFilterStep,
        "nms": NmsStep,
    }

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.runtime = _parse_runtime_cfg(cfg)
        self.logger = get_logger("distill")
        self.steps: List[PostProcessStep] = []
        for step_cfg in cfg.get("steps", []):
            if not isinstance(step_cfg, dict):
                continue
            if not step_cfg.get("enabled", True):
                continue
            name = str(step_cfg.get("name", "")).lower()
            builder = self._STEP_REGISTRY.get(name)
            if builder is None:
                raise ValueError(f"未知后处理步骤: {name}")
            self.steps.append(builder(step_cfg, runtime=self.runtime))
        if self.runtime.use_torch and torch is None:
            self.logger.warning("postprocess.runtime.use_torch=True 但 torch 不可用，回退 numpy 路径")

    def __call__(self, preds: List[RawMaskPrediction], image_meta: Dict[str, Any]) -> List[RawMaskPrediction]:
        output = preds
        if self.runtime.log_stats:
            self.logger.info(
                "postprocess start: preds=%d device=%s use_torch=%s",
                len(output),
                self.runtime.device,
                self.runtime.use_torch,
            )
        if self.runtime.topk_pre and len(output) > self.runtime.topk_pre:
            scores = [(i, _score_of_pred(item)) for i, item in enumerate(output) if isinstance(item, dict)]
            scores.sort(key=lambda x: x[1], reverse=True)
            keep = {i for i, _ in scores[: self.runtime.topk_pre]}
            output = [output[i] for i in range(len(output)) if i in keep]
            if self.runtime.log_stats:
                self.logger.info("postprocess topk_pre=%d kept=%d", self.runtime.topk_pre, len(output))
        for step in self.steps:
            output = step(output, image_meta)
            if self.runtime.log_stats:
                self.logger.info("postprocess step=%s out=%d", getattr(step, "name", "unknown"), len(output))
        return output
