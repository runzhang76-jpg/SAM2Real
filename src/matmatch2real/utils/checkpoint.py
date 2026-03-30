"""Checkpoint /"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

from matmatch2real.utils.logging import get_logger

try:
    import torch
except Exception:  # pragma: no cover -
    torch = None  # type: ignore


def save_checkpoint(
    output_dir: str,
    model: Any,
    optimizer: Any,
    scheduler: Optional[Any],
    tag: str,
    state: Optional[Any] = None,
) -> Path:
    """ checkpoint"""

    logger = get_logger("distill")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_path / f"checkpoint_{tag}.pt"

    payload = {
        "model": getattr(model, "state_dict", lambda: {})(),
        "optimizer": getattr(optimizer, "state_dict", lambda: {})(),
        "scheduler": getattr(scheduler, "state_dict", lambda: {})() if scheduler is not None else {},
        "state": state,
    }

    if torch is not None:
        torch.save(payload, ckpt_path)
    else:
        with ckpt_path.open("wb") as f:
            pickle.dump(payload, f)

    logger.info("checkpoint saved to %s", ckpt_path)
    return ckpt_path


def load_checkpoint(path: str, model: Any, optimizer: Any, scheduler: Optional[Any] = None) -> Optional[Any]:
    """ checkpoint """

    logger = get_logger("distill")
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if torch is not None:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        with ckpt_path.open("rb") as f:
            payload = pickle.load(f)

    model.load_state_dict(payload.get("model", {}), strict=False)
    optimizer.load_state_dict(payload.get("optimizer", {}))
    if scheduler is not None:
        scheduler_state = payload.get("scheduler", {})
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)
    logger.info("checkpoint loaded from %s", ckpt_path)
    return payload.get("state")
