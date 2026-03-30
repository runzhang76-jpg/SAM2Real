"""Composable soft distillation losses for file-driven teacher supervision."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch optional during static inspection
    torch = None
    F = None  # type: ignore


def _zero(device: "torch.device") -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required for soft distillation losses.")
    return torch.tensor(0.0, device=device)


def _as_tensor(value: Any, device: "torch.device") -> Optional["torch.Tensor"]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _assert_finite(name: str, tensor: Optional["torch.Tensor"]) -> None:
    if tensor is None:
        return
    if torch is None:
        raise RuntimeError("PyTorch is required for soft distillation losses.")
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN/Inf")


def _expand_valid_mask(valid_mask: Optional["torch.Tensor"], target: "torch.Tensor") -> Optional["torch.Tensor"]:
    if valid_mask is None:
        return None
    if valid_mask.ndim == 1:
        view_shape = [valid_mask.shape[0]] + [1] * (target.ndim - 1)
        return valid_mask.view(*view_shape)
    return valid_mask


def _match_class_dim(teacher_prob: "torch.Tensor", num_classes: int) -> "torch.Tensor":
    if teacher_prob.shape[1] == num_classes:
        return teacher_prob
    raise AssertionError(
        f"teacher class_soft dim {teacher_prob.shape[1]} does not match student num_classes {num_classes}; "
        "do not silently truncate/pad class probabilities."
    )


def _build_sample_weights(
    teacher_soft: Optional[Dict[str, Any]],
    batch_size: int,
    device: "torch.device",
    use_score_weight: bool,
) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required for soft distillation losses.")
    if not use_score_weight or not teacher_soft or teacher_soft.get("img_score") is None:
        return torch.ones(batch_size, device=device)
    scores = _as_tensor(teacher_soft.get("img_score"), device)
    if scores is None:
        return torch.ones(batch_size, device=device)
    scores = scores.flatten()
    if scores.numel() < batch_size:
        pad = torch.zeros(batch_size - scores.numel(), device=device)
        scores = torch.cat([scores, pad], dim=0)
    scores = scores[:batch_size]
    _assert_finite("teacher_soft.img_score", scores)
    return scores.clamp_(0.0, 1.0)


def _reduce_loss(
    per_item_loss: "torch.Tensor",
    sample_weights: "torch.Tensor",
    valid_mask: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required for soft distillation losses.")
    batch_size = per_item_loss.shape[0]
    _assert_finite("per_item_loss", per_item_loss)
    _assert_finite("sample_weights", sample_weights)
    loss = per_item_loss.reshape(batch_size, -1).mean(dim=1)
    if valid_mask is not None:
        valid_vec = valid_mask.flatten().float()
        _assert_finite("valid_mask", valid_vec)
        sample_weights = sample_weights * valid_vec
    denom = sample_weights.sum().clamp_min(1e-6)
    return (loss * sample_weights).sum() / denom


def _binary_kl(student_prob: "torch.Tensor", teacher_prob: "torch.Tensor") -> "torch.Tensor":
    eps = 1e-6
    student_prob = student_prob.clamp(eps, 1.0 - eps)
    teacher_prob = teacher_prob.clamp(eps, 1.0 - eps)
    return teacher_prob * (teacher_prob.log() - student_prob.log()) + (1.0 - teacher_prob) * (
        (1.0 - teacher_prob).log() - (1.0 - student_prob).log()
    )


def soft_mask_distill_loss(
    student_logits: Optional["torch.Tensor"],
    teacher_soft: Optional[Dict[str, Any]],
    device: "torch.device",
    mode: str = "bce",
    use_score_weight: bool = True,
) -> Tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", Optional["torch.Tensor"]]:
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for soft distillation losses.")
    if student_logits is None or teacher_soft is None:
        return _zero(device), None, torch.ones(1, device=device), None

    teacher_logits = _as_tensor(teacher_soft.get("img_mask_logits"), device)
    if teacher_logits is None:
        return _zero(device), None, torch.ones(student_logits.shape[0], device=device), None

    if teacher_logits.ndim == 3:
        teacher_logits = teacher_logits.unsqueeze(1)
    elif teacher_logits.ndim == 2:
        teacher_logits = teacher_logits.unsqueeze(0).unsqueeze(0)
    if teacher_logits.shape != student_logits.shape:
        raise AssertionError(
            f"soft mask distill shape mismatch: student={tuple(student_logits.shape)} teacher={tuple(teacher_logits.shape)}"
        )

    valid_mask = _as_tensor(teacher_soft.get("has_img_mask_logits"), device)
    sample_weights = _build_sample_weights(teacher_soft, student_logits.shape[0], device, use_score_weight)
    teacher_prob = torch.sigmoid(teacher_logits)
    _assert_finite("student_logits", student_logits)
    _assert_finite("teacher_logits", teacher_logits)
    _assert_finite("teacher_prob", teacher_prob)

    if mode.lower() == "kl":
        per_pixel = _binary_kl(torch.sigmoid(student_logits), teacher_prob)
    else:
        per_pixel = F.binary_cross_entropy_with_logits(student_logits, teacher_prob, reduction="none")

    loss = _reduce_loss(per_pixel, sample_weights, valid_mask)
    return loss, per_pixel, sample_weights, valid_mask


def boundary_distill_loss(
    per_pixel_loss: Optional["torch.Tensor"],
    teacher_soft: Optional[Dict[str, Any]],
    device: "torch.device",
    alpha: float,
    sample_weights: "torch.Tensor",
    valid_mask: Optional["torch.Tensor"],
) -> "torch.Tensor":
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for soft distillation losses.")
    if per_pixel_loss is None or teacher_soft is None:
        return _zero(device)

    boundary_map = _as_tensor(teacher_soft.get("img_boundary"), device)
    if boundary_map is None:
        return _zero(device)
    if boundary_map.ndim == 3:
        boundary_map = boundary_map.unsqueeze(1)
    elif boundary_map.ndim == 2:
        boundary_map = boundary_map.unsqueeze(0).unsqueeze(0)
    if boundary_map.shape != per_pixel_loss.shape:
        raise AssertionError(
            f"boundary distill shape mismatch: loss={tuple(per_pixel_loss.shape)} boundary={tuple(boundary_map.shape)}"
        )

    boundary_valid = _as_tensor(teacher_soft.get("has_img_boundary"), device)
    if valid_mask is not None and boundary_valid is not None:
        valid_mask = valid_mask * boundary_valid
    elif boundary_valid is not None:
        valid_mask = boundary_valid

    _assert_finite("boundary_map", boundary_map)
    boundary_map = boundary_map.clamp(0.0, 1.0)
    weighted = per_pixel_loss * (1.0 + float(alpha) * boundary_map)
    base_loss = _reduce_loss(per_pixel_loss, sample_weights, valid_mask)
    weighted_loss = _reduce_loss(weighted, sample_weights, valid_mask)
    return torch.clamp(weighted_loss - base_loss, min=0.0)


def soft_class_distill_loss(
    student_logits: Optional["torch.Tensor"],
    teacher_soft: Optional[Dict[str, Any]],
    device: "torch.device",
    temperature: float,
    use_score_weight: bool = True,
) -> "torch.Tensor":
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for soft distillation losses.")
    if student_logits is None or teacher_soft is None:
        return _zero(device)

    teacher_prob = _as_tensor(teacher_soft.get("img_class_soft"), device)
    if teacher_prob is None:
        return _zero(device)
    if teacher_prob.ndim == 1:
        teacher_prob = teacher_prob.unsqueeze(0)
    class_ids = teacher_soft.get("class_ids")
    if class_ids is not None:
        class_ids_tensor = _as_tensor(class_ids, device)
        if class_ids_tensor is not None and int(class_ids_tensor.numel()) != int(student_logits.shape[1]):
            raise AssertionError(
                f"teacher class_ids count {int(class_ids_tensor.numel())} does not match student num_classes {student_logits.shape[1]}"
            )
    teacher_prob = _match_class_dim(teacher_prob, student_logits.shape[1])
    teacher_prob = teacher_prob / teacher_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)

    valid_mask = _as_tensor(teacher_soft.get("has_img_class_soft"), device)
    sample_weights = _build_sample_weights(teacher_soft, student_logits.shape[0], device, use_score_weight)

    temp = max(float(temperature), 1e-6)
    log_probs = F.log_softmax(student_logits / temp, dim=1)
    _assert_finite("student_logits.cls", student_logits)
    _assert_finite("teacher_prob.cls", teacher_prob)
    _assert_finite("student_log_probs.cls", log_probs)
    kl = F.kl_div(log_probs, teacher_prob, reduction="none").sum(dim=1) * (temp ** 2)
    return _reduce_loss(kl, sample_weights, valid_mask)
