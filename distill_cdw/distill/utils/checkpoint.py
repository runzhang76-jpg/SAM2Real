"""Checkpoint 保存/加载工具。"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

from distill.utils.logging import get_logger

try:
    import torch
except Exception:  # pragma: no cover - 可选依赖
    torch = None  # type: ignore


def save_checkpoint(
    output_dir: str,
    model: Any,
    optimizer: Any,
    tag: str,
    state: Optional[Any] = None,
) -> Path:
    """保存训练 checkpoint。"""

    logger = get_logger("distill")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_path / f"checkpoint_{tag}.pt"

    payload = {
        "model": getattr(model, "state_dict", lambda: {})(),
        "optimizer": getattr(optimizer, "state_dict", lambda: {})(),
        "state": state,
    }

    if torch is not None:
        torch.save(payload, ckpt_path)
    else:
        with ckpt_path.open("wb") as f:
            pickle.dump(payload, f)

    logger.info("checkpoint saved to %s", ckpt_path)
    return ckpt_path


def load_checkpoint(path: str, model: Any, optimizer: Any) -> None:
    """加载 checkpoint 到模型与优化器。"""

    logger = get_logger("distill")
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if torch is not None:
        payload = torch.load(ckpt_path, map_location="cpu")
    else:
        with ckpt_path.open("rb") as f:
            payload = pickle.load(f)

    model.load_state_dict(payload.get("model", {}), strict=False)
    optimizer.load_state_dict(payload.get("optimizer", {}))
    logger.info("checkpoint loaded from %s", ckpt_path)
