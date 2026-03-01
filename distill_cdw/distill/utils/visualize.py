"""可视化辅助。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from distill.utils.logging import get_logger

try:
    import torch
except Exception:  # pragma: no cover - 可选依赖
    torch = None  # type: ignore


def _to_numpy_image(image: Any) -> np.ndarray:
    if torch is not None and isinstance(image, torch.Tensor):
        img = image.detach().cpu()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)
        arr = img.numpy()
    else:
        arr = np.asarray(image)

    if arr.dtype.kind == "f":
        max_val = float(arr.max()) if arr.size else 1.0
        if max_val <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255)
    arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr


def _to_numpy_masks(masks: Any) -> List[np.ndarray]:
    if masks is None:
        return []
    if torch is not None and isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.asarray(masks)

    if masks_np.ndim == 2:
        return [masks_np.astype(bool)]
    if masks_np.ndim == 3:
        return [(masks_np[i] > 0.5) for i in range(masks_np.shape[0])]
    return []


def overlay_masks(
    image: Any,
    masks: Any,
    alpha: float = 0.5,
    colors: Optional[Iterable[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """将 mask 叠加到图像上，返回叠加后的 numpy 图像。"""

    img = _to_numpy_image(image)
    mask_list = _to_numpy_masks(masks)
    if not mask_list:
        return img

    rng = np.random.default_rng(42)
    color_list = list(colors) if colors is not None else []

    overlay = img.astype(np.float32)
    for idx, mask in enumerate(mask_list):
        if mask.shape[:2] != img.shape[:2]:
            continue
        if idx < len(color_list):
            color = np.array(color_list[idx], dtype=np.float32)
        else:
            color = rng.integers(0, 255, size=3).astype(np.float32)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_pr_curve(output_dir: str, metrics: Dict[str, Any]) -> None:
    """保存 PR 曲线图（若可用）。"""

    logger = get_logger("distill")
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    if precision is None or recall is None:
        logger.warning("未提供 precision/recall，跳过 PR 曲线绘制")
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - 可选依赖
        logger.warning("matplotlib 不可用，无法绘制 PR 曲线: %s", exc)
        return

    precision = np.asarray(precision)
    recall = np.asarray(recall)
    if precision.shape != recall.shape:
        logger.warning("precision/recall 形状不一致，跳过 PR 曲线绘制")
        return

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_path = f"{output_dir}/pr_curve.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("PR 曲线已保存: %s", out_path)
