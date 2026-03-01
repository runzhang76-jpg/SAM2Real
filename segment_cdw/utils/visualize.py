from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def _iter_masks_limited(
    masks: Iterable[np.ndarray],
    max_masks: int,
    downsample_factor: float,
    target_hw: tuple[int, int] | None = None,
):
    """逐个处理 mask，限制数量并支持下采样/还原尺寸，避免 np.stack 大数组。"""
    count = 0
    for m in masks:
        if count >= max_masks:
            break
        arr = np.asarray(m, dtype=bool)
        if target_hw is not None:
            h, w = target_hw
            arr = np.array(
                Image.fromarray(arr.astype(np.uint8) * 255).resize((w, h), resample=Image.NEAREST),
                copy=False,
            ) > 127
        if downsample_factor > 1.0:
            new_w = max(1, int(arr.shape[1] / downsample_factor))
            new_h = max(1, int(arr.shape[0] / downsample_factor))
            arr = np.array(
                Image.fromarray(arr.astype(np.uint8) * 255).resize((new_w, new_h), resample=Image.NEAREST),
                copy=False,
            ) > 127
        yield arr
        count += 1


def save_overlay(
    image_np: np.ndarray,
    pred_masks: Iterable[np.ndarray],
    gt_masks: Iterable[np.ndarray],
    save_path: Path,
    alpha: float = 0.4,
    orig_hw: tuple[int, int] | None = None,
    max_masks: int = 20,
    downsample_factor: float = 1.0,
) -> None:
    """
    将预测/GT 掩码覆盖到图像上，保存到文件。
    预测为红色，GT 为绿色。仅叠加前 max_masks 个掩码。
    """
    img = image_np.astype(np.uint8, copy=False)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)

    target_hw = orig_hw
    if orig_hw is not None:
        img = np.array(Image.fromarray(img).resize((orig_hw[1], orig_hw[0]), resample=Image.BILINEAR))

    overlay = np.zeros_like(img, dtype=np.float32)
    pred_union = None
    gt_union = None

    for arr in _iter_masks_limited(pred_masks, max_masks=max_masks, downsample_factor=downsample_factor, target_hw=target_hw):
        if pred_union is None:
            pred_union = arr.copy()
        else:
            pred_union |= arr
    for arr in _iter_masks_limited(gt_masks, max_masks=max_masks, downsample_factor=downsample_factor, target_hw=target_hw):
        if gt_union is None:
            gt_union = arr.copy()
        else:
            gt_union |= arr

    if pred_union is not None:
        overlay[..., 0] += pred_union * 255
    if gt_union is not None:
        overlay[..., 1] += gt_union * 255

    blended = (img.astype(np.float32) * (1 - alpha) + overlay * alpha).clip(0, 255)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blended.astype(np.uint8)).save(save_path)
