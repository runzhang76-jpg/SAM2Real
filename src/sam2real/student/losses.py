""""""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sam2real.core.structures import PseudoLabelInstance
from sam2real.student.distill_soft.losses import (
    boundary_distill_loss,
    soft_class_distill_loss,
    soft_mask_distill_loss,
)
from sam2real.utils.logging import get_logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch
    torch = None
    nn = None  # type: ignore
    F = None  # type: ignore


def _instances_to_mask(
    instances: List[PseudoLabelInstance], height: int, width: int, device: "torch.device"
) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch ")
    mask = torch.zeros(height, width, device=device)
    for inst in instances:
        if inst.mask is not None:
            inst_mask = inst.mask
            if hasattr(inst_mask, "shape") and tuple(inst_mask.shape) == (height, width):
                if not isinstance(inst_mask, torch.Tensor):
                    inst_mask = torch.as_tensor(inst_mask, device=device)
                else:
                    inst_mask = inst_mask.to(device)
                mask = torch.maximum(mask, inst_mask.float())
                continue
        # mask  bbox
        x, y, w, h = inst.bbox
        x1, y1 = int(x), int(y)
        x2, y2 = min(width, int(x + w)), min(height, int(y + h))
        mask[y1:y2, x1:x2] = 1.0
    return mask


def _reliability_weight(instances: List[PseudoLabelInstance], gamma: float) -> float:
    if not instances:
        return 0.0
    avg_rel = sum(inst.reliability for inst in instances) / max(1, len(instances))
    return float(max(0.0, min(1.0, avg_rel)) ** gamma)


def _pick_reliable_instance(instances: List[PseudoLabelInstance]) -> Optional[PseudoLabelInstance]:
    if not instances:
        return None
    return max(instances, key=lambda inst: inst.reliability)


def _as_tensor(value: Any, device: "torch.device") -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch ")
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.tensor(float(value), device=device)


def _infer_device(outputs: Dict[str, Any], batch: Dict[str, Any]) -> "torch.device":
    if torch is None:
        raise RuntimeError("PyTorch ")
    for value in outputs.values():
        if isinstance(value, torch.Tensor):
            return value.device
    images = batch.get("images")
    if isinstance(images, torch.Tensor):
        return images.device
    return torch.device("cpu")


def _align_features(student_feat: "torch.Tensor", teacher_feat: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
    if F is None:
        return student_feat, teacher_feat
    if student_feat.ndim == 4 and teacher_feat.ndim == 4:
        target_h = min(student_feat.shape[2], teacher_feat.shape[2])
        target_w = min(student_feat.shape[3], teacher_feat.shape[3])
        if student_feat.shape[2:] != (target_h, target_w):
            student_feat = F.adaptive_avg_pool2d(student_feat, (target_h, target_w))
        if teacher_feat.shape[2:] != (target_h, target_w):
            teacher_feat = F.adaptive_avg_pool2d(teacher_feat, (target_h, target_w))
        min_c = min(student_feat.shape[1], teacher_feat.shape[1])
        return student_feat[:, :min_c], teacher_feat[:, :min_c]
    if student_feat.ndim == 2 and teacher_feat.ndim == 2:
        min_c = min(student_feat.shape[1], teacher_feat.shape[1])
        return student_feat[:, :min_c], teacher_feat[:, :min_c]
    student_flat = student_feat.flatten(start_dim=1)
    teacher_flat = teacher_feat.flatten(start_dim=1)
    min_dim = min(student_flat.shape[1], teacher_flat.shape[1])
    return student_flat[:, :min_dim], teacher_flat[:, :min_dim]


def _build_weights(
    batch_instances: List[List[PseudoLabelInstance]],
    gamma: float,
    use_reliability: bool,
    device: "torch.device",
) -> Optional["torch.Tensor"]:
    if torch is None:
        raise RuntimeError("PyTorch ")
    if not batch_instances:
        return None
    weights = [
        _reliability_weight(instances, gamma) if use_reliability else 1.0
        for instances in batch_instances
    ]
    return torch.tensor(weights, device=device)


def _build_target_masks(
    batch_instances: List[List[PseudoLabelInstance]],
    height: int,
    width: int,
    device: "torch.device",
) -> Optional["torch.Tensor"]:
    if not batch_instances:
        return None
    masks = [_instances_to_mask(instances, height, width, device) for instances in batch_instances]
    if not masks:
        return None
    return torch.stack(masks).unsqueeze(1)


def _build_class_targets(batch_instances: List[List[PseudoLabelInstance]]) -> List[int]:
    targets: List[int] = []
    for instances in batch_instances:
        inst = _pick_reliable_instance(instances)
        targets.append(int(inst.class_id) if inst else 0)
    return targets


def _build_bbox_targets(batch_instances: List[List[PseudoLabelInstance]]) -> List[List[float]]:
    targets: List[List[float]] = []
    for instances in batch_instances:
        inst = _pick_reliable_instance(instances)
        if inst is None:
            targets.append([0.0, 0.0, 0.0, 0.0])
        else:
            targets.append(list(inst.bbox))
    return targets


def _tensor_stats(name: str, value: Any) -> Dict[str, float]:
    if torch is None or not isinstance(value, torch.Tensor):
        return {}
    tensor = value.detach().float()
    if tensor.numel() == 0:
        return {f"{name}_mean": 0.0, f"{name}_std": 0.0, f"{name}_min": 0.0, f"{name}_max": 0.0}
    return {
        f"{name}_mean": float(tensor.mean().item()),
        f"{name}_std": float(tensor.std(unbiased=False).item()),
        f"{name}_min": float(tensor.min().item()),
        f"{name}_max": float(tensor.max().item()),
    }


class DistillLoss(nn.Module):
    """"""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.logger = get_logger("distill")
        self.cfg = cfg
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.soft_debug_cfg = dict(self.cfg.get("distill_soft", {}).get("debug", {}))
        self._forward_calls = 0

    def _soft_profile(self) -> str:
        return str(self.cfg.get("distill_soft", {}).get("profile", "full")).strip().lower() or "full"

    def _soft_head_enabled(self, head: str, cfg_enabled: bool) -> bool:
        profile = self._soft_profile()
        if profile in {"baseline", "off", "none"}:
            return False
        if profile == "mask":
            return head == "mask" and cfg_enabled
        if profile in {"mask_boundary", "mask+boundary"}:
            return head in {"mask", "boundary"} and cfg_enabled
        if profile in {"mask_cls", "mask+cls"}:
            return head in {"mask", "cls"} and cfg_enabled
        if profile in {"full", "all"}:
            return cfg_enabled
        raise ValueError(f"Unsupported distill_soft.profile: {profile}")

    def _should_debug_soft(self) -> bool:
        if not bool(self.soft_debug_cfg.get("enabled", False)):
            return False
        every = max(1, int(self.soft_debug_cfg.get("log_every", 1)))
        return self._forward_calls % every == 0

    def _assert_finite(self, name: str, value: Any) -> None:
        if torch is None or not isinstance(value, torch.Tensor):
            return
        if not torch.isfinite(value).all():
            raise AssertionError(f"{name} contains NaN/Inf")

    def _validate_soft_inputs(self, outputs: Dict[str, Any], teacher_soft: Optional[Dict[str, Any]], logs: Dict[str, Any]) -> None:
        if torch is None or teacher_soft is None:
            return
        mask_student = outputs.get("kd_mask_logits")
        mask_teacher = teacher_soft.get("img_mask_logits")
        if isinstance(mask_student, torch.Tensor) and isinstance(mask_teacher, torch.Tensor):
            teacher_mask = mask_teacher.to(device=mask_student.device, dtype=torch.float32)
            if teacher_mask.ndim == 2:
                teacher_mask = teacher_mask.unsqueeze(0).unsqueeze(0)
            elif teacher_mask.ndim == 3:
                teacher_mask = teacher_mask.unsqueeze(1)
            if tuple(mask_student.shape) != tuple(teacher_mask.shape):
                raise AssertionError(
                    f"KD mask shape mismatch before loss: student={tuple(mask_student.shape)} teacher={tuple(teacher_mask.shape)}"
                )
            self._assert_finite("outputs.kd_mask_logits", mask_student)
            self._assert_finite("teacher_soft.img_mask_logits", teacher_mask)
            if self._should_debug_soft():
                logs.update(_tensor_stats("kd_mask_student", mask_student))
                logs.update(_tensor_stats("kd_mask_teacher", teacher_mask))

        boundary_map = teacher_soft.get("img_boundary")
        if isinstance(boundary_map, torch.Tensor):
            boundary_tensor = boundary_map.float()
            if boundary_tensor.ndim == 2:
                boundary_tensor = boundary_tensor.unsqueeze(0).unsqueeze(0)
            elif boundary_tensor.ndim == 3:
                boundary_tensor = boundary_tensor.unsqueeze(1)
            self._assert_finite("teacher_soft.img_boundary", boundary_tensor)
            if self._should_debug_soft():
                logs.update(_tensor_stats("kd_boundary", boundary_tensor))

        cls_student = outputs.get("kd_cls_logits")
        cls_teacher = teacher_soft.get("img_class_soft")
        if isinstance(cls_student, torch.Tensor) and isinstance(cls_teacher, torch.Tensor):
            teacher_cls = cls_teacher.to(device=cls_student.device, dtype=torch.float32)
            if teacher_cls.ndim == 1:
                teacher_cls = teacher_cls.unsqueeze(0)
            if tuple(cls_student.shape) != tuple(teacher_cls.shape):
                raise AssertionError(
                    f"KD class shape mismatch before loss: student={tuple(cls_student.shape)} teacher={tuple(teacher_cls.shape)}"
                )
            self._assert_finite("outputs.kd_cls_logits", cls_student)
            self._assert_finite("teacher_soft.img_class_soft", teacher_cls)
            if self._should_debug_soft():
                logs.update(_tensor_stats("kd_cls_student", cls_student))
                logs.update(_tensor_stats("kd_cls_teacher", teacher_cls))
                logs["kd_cls_teacher_sum_mean"] = float(teacher_cls.sum(dim=1).mean().item())

        sample_weights = teacher_soft.get("img_score")
        if isinstance(sample_weights, torch.Tensor):
            self._assert_finite("teacher_soft.img_score", sample_weights)
            if self._should_debug_soft():
                logs.update(_tensor_stats("kd_score_weight", sample_weights))

    def _compute_supervised_loss(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> Tuple["torch.Tensor", Dict[str, Any]]:
        device = _infer_device(outputs, batch)
        if "loss_supervised" in outputs:
            loss_supervised = _as_tensor(outputs["loss_supervised"], device)
            if loss_supervised.numel() > 1:
                loss_supervised = loss_supervised.sum()
            loss_items = outputs.get("loss_items", {})
            logs = {"loss_supervised": float(loss_supervised.detach().cpu())}
            if isinstance(loss_items, dict):
                logs.update({k: float(v) for k, v in loss_items.items()})
            return loss_supervised, logs

        mask_logits = outputs.get("mask_logits")
        if mask_logits is None:
            return _as_tensor(0.0, device), {"loss_supervised": 0.0}

        batch_instances = batch.get("instances", [])
        gamma = float(self.cfg.get("reliability", {}).get("gamma", 1.0))
        use_reliability = bool(self.cfg.get("reliability", {}).get("enabled", True))
        height, width = mask_logits.shape[-2], mask_logits.shape[-1]
        weights = _build_weights(batch_instances, gamma, use_reliability, mask_logits.device)
        target = _build_target_masks(batch_instances, height, width, mask_logits.device)

        if target is None or weights is None:
            return _as_tensor(0.0, device), {"loss_supervised": 0.0}

        weight_tensor = weights.view(-1, 1, 1, 1)

        bce_loss = self.bce(mask_logits, target)
        loss_mask = (bce_loss * weight_tensor).mean()

        pred = torch.sigmoid(mask_logits)
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        loss_dice = (dice * weight_tensor.view(-1)).mean()

        loss_supervised = loss_mask * float(self.cfg.get("mask", {}).get("weight", 1.0))
        loss_supervised = loss_supervised + loss_dice * float(self.cfg.get("dice", {}).get("weight", 1.0))

        logs = {
            "loss_supervised": float(loss_supervised.detach().cpu()),
            "loss_mask": float(loss_mask.detach().cpu()),
            "loss_dice": float(loss_dice.detach().cpu()),
            "reliability_weight": float(weights.mean().item()),
        }
        return loss_supervised, logs

    def _compute_distill_loss(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> Tuple["torch.Tensor", Dict[str, Any]]:
        device = _infer_device(outputs, batch)
        distill_cfg = self.cfg.get("distill", {})
        gamma = float(self.cfg.get("reliability", {}).get("gamma", 1.0))
        use_reliability = bool(self.cfg.get("reliability", {}).get("enabled", True))

        mask_weight = float(distill_cfg.get("mask_kl_weight", 0.0))
        cls_weight = float(distill_cfg.get("cls_kl_weight", 0.0))
        bbox_weight = float(distill_cfg.get("bbox_l1_weight", 0.0))
        feat_weight = float(distill_cfg.get("feat_l2_weight", 0.0))

        total = _as_tensor(0.0, device)
        logs: Dict[str, Any] = {}

        batch_instances = batch.get("instances", [])
        weights = _build_weights(batch_instances, gamma, use_reliability, device)

        if mask_weight > 0.0 and "mask_logits" in outputs and weights is not None:
            mask_logits = outputs["mask_logits"]
            height, width = mask_logits.shape[-2], mask_logits.shape[-1]
            target = _build_target_masks(batch_instances, height, width, mask_logits.device)
            if target is not None:
                weight_tensor = weights.view(-1, 1, 1, 1)
                loss_mask = self.bce(mask_logits, target)
                loss_mask = (loss_mask * weight_tensor).mean()
                total = total + loss_mask * mask_weight
                logs["loss_distill_mask"] = float(loss_mask.detach().cpu())

        if cls_weight > 0.0 and "class_logits" in outputs and weights is not None:
            class_logits = outputs["class_logits"]
            target_ids = _build_class_targets(batch_instances)
            if target_ids:
                target_tensor = torch.tensor(target_ids, device=class_logits.device)
                log_probs = F.log_softmax(class_logits, dim=1)
                one_hot = F.one_hot(target_tensor, num_classes=class_logits.shape[1]).float()
                kl = F.kl_div(log_probs, one_hot, reduction="none").sum(dim=1)
                weight_tensor = weights.to(class_logits.device)
                loss_cls = (kl * weight_tensor).mean()
                total = total + loss_cls * cls_weight
                logs["loss_distill_cls"] = float(loss_cls.detach().cpu())

        if bbox_weight > 0.0 and "pred_boxes" in outputs and weights is not None:
            pred_boxes = outputs["pred_boxes"]
            target_boxes = _build_bbox_targets(batch_instances)
            if target_boxes:
                target_tensor = torch.tensor(target_boxes, device=pred_boxes.device)
                loss_bbox = F.l1_loss(pred_boxes, target_tensor, reduction="none").mean(dim=1)
                weight_tensor = weights.to(pred_boxes.device)
                loss_bbox = (loss_bbox * weight_tensor).mean()
                total = total + loss_bbox * bbox_weight
                logs["loss_distill_bbox"] = float(loss_bbox.detach().cpu())

        if feat_weight > 0.0 and "features" in outputs and "teacher_features" in batch:
            student_feat = outputs["features"]
            teacher_feat = batch["teacher_features"]
            if isinstance(teacher_feat, torch.Tensor) and isinstance(student_feat, torch.Tensor):
                s_feat, t_feat = _align_features(student_feat, teacher_feat)
                loss_feat = F.mse_loss(s_feat, t_feat)
                total = total + loss_feat * feat_weight
                logs["loss_distill_feat"] = float(loss_feat.detach().cpu())

        return total, logs

    def _compute_soft_distill_loss(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> Tuple["torch.Tensor", Dict[str, Any]]:
        device = _infer_device(outputs, batch)
        soft_cfg = self.cfg.get("distill_soft", {})
        logs = {
            "loss_kd_softmask": 0.0,
            "loss_kd_boundary": 0.0,
            "loss_kd_softcls": 0.0,
        }
        total = _as_tensor(0.0, device)

        if not soft_cfg.get("enabled", False):
            return total, logs

        teacher_soft = batch.get("teacher_soft")
        score_weight = bool(soft_cfg.get("score_weight", True))
        logs["distill_soft_profile"] = self._soft_profile()
        self._validate_soft_inputs(outputs, teacher_soft, logs)

        mask_cfg = soft_cfg.get("mask", {})
        mask_loss = _as_tensor(0.0, device)
        per_pixel = None
        sample_weights = torch.ones(1, device=device)
        valid_mask = None
        if self._soft_head_enabled("mask", bool(mask_cfg.get("enable", False))):
            mask_loss, per_pixel, sample_weights, valid_mask = soft_mask_distill_loss(
                outputs.get("kd_mask_logits"),
                teacher_soft,
                device=device,
                mode=str(mask_cfg.get("mode", "bce")),
                use_score_weight=score_weight,
            )
            total = total + mask_loss * float(mask_cfg.get("weight", 1.0))
            logs["loss_kd_softmask"] = float(mask_loss.detach().cpu())

        boundary_cfg = soft_cfg.get("boundary", {})
        if self._soft_head_enabled("boundary", bool(boundary_cfg.get("enable", False))):
            boundary_loss = boundary_distill_loss(
                per_pixel,
                teacher_soft,
                device=device,
                alpha=float(boundary_cfg.get("alpha", 3.0)),
                sample_weights=sample_weights,
                valid_mask=valid_mask,
            )
            total = total + boundary_loss * float(boundary_cfg.get("weight", 1.0))
            logs["loss_kd_boundary"] = float(boundary_loss.detach().cpu())

        cls_cfg = soft_cfg.get("cls", {})
        if self._soft_head_enabled("cls", bool(cls_cfg.get("enable", False))):
            cls_loss = soft_class_distill_loss(
                outputs.get("kd_cls_logits"),
                teacher_soft,
                device=device,
                temperature=float(cls_cfg.get("temperature", 2.0)),
                use_score_weight=score_weight,
            )
            total = total + cls_loss * float(cls_cfg.get("weight", 1.0))
            logs["loss_kd_softcls"] = float(cls_loss.detach().cpu())

        if self._should_debug_soft():
            if isinstance(per_pixel, torch.Tensor):
                logs.update(_tensor_stats("kd_per_pixel", per_pixel))
            if isinstance(sample_weights, torch.Tensor):
                logs.update(_tensor_stats("kd_sample_weight", sample_weights))
            if isinstance(valid_mask, torch.Tensor):
                logs.update(_tensor_stats("kd_valid_mask", valid_mask.float()))
            logs["kd_softmask_norm"] = "sample-weighted mean of per-sample pixel mean"
            logs["kd_boundary_norm"] = "delta(weighted-base) with boundary in [0,1]"
            logs["kd_softcls_norm"] = "sample-weighted KL with T^2"

        return total, logs

    def forward(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        self._forward_calls += 1

        supervised_loss, sup_logs = self._compute_supervised_loss(outputs, batch)
        distill_loss, distill_logs = self._compute_distill_loss(outputs, batch)
        soft_distill_loss, soft_distill_logs = self._compute_soft_distill_loss(outputs, batch)

        alpha = float(self.cfg.get("distill", {}).get("alpha", 1.0))
        total = supervised_loss + distill_loss * alpha + soft_distill_loss
        total_distill = distill_loss + soft_distill_loss

        logs = {
            "total": total,
            "loss_distill": float(total_distill.detach().cpu()) if isinstance(total_distill, torch.Tensor) else total_distill,
            "loss_distill_legacy": float(distill_loss.detach().cpu()) if isinstance(distill_loss, torch.Tensor) else distill_loss,
            "loss_distill_soft": float(soft_distill_loss.detach().cpu()) if isinstance(soft_distill_loss, torch.Tensor) else soft_distill_loss,
            **sup_logs,
            **distill_logs,
            **soft_distill_logs,
        }
        return logs
