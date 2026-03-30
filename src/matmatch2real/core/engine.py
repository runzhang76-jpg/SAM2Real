""""""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from matmatch2real.utils.checkpoint import load_checkpoint, save_checkpoint
from matmatch2real.utils.logging import get_logger

try:
    import torch
    if hasattr(torch, "amp"):
        from torch.amp import GradScaler as AmpGradScaler, autocast as amp_autocast
    else:
        from torch.cuda.amp import GradScaler as AmpGradScaler, autocast as amp_autocast
except Exception:  # pragma: no cover - torch
    torch = None
    AmpGradScaler = None  # type: ignore
    amp_autocast = None  # type: ignore


@dataclass
class EngineState:
    """ Hook """

    epoch: int = 0
    step: int = 0
    global_step: int = 0
    optimizer_step: int = 0
    best_metric: Optional[float] = None


class DistillEngine:
    """-"""

    def __init__(
        self,
        model: Any,
        loss_fn: Any,
        optimizer: Any,
        dataloader: Any,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
        amp: bool = False,
        grad_accum: int = 1,
        clip_grad_norm: float = 0.0,
        teacher: Optional[Any] = None,
        teacher_mode: str = "offline",
        evaluator: Optional[Any] = None,
        visualizer: Optional[Any] = None,
        hooks: Optional[Any] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for DistillEngine.")
        self.logger = get_logger("distill")
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.amp = amp
        self.grad_accum = max(1, int(grad_accum))
        self.clip_grad_norm = clip_grad_norm
        self.teacher = teacher
        self.teacher_mode = teacher_mode
        self.evaluator = evaluator
        self.visualizer = visualizer
        self.hooks = hooks
        self.output_dir = output_dir
        self.state = EngineState()
        self.scaler = _build_grad_scaler(self.device.type, amp)
        self._online_cache: Dict[Any, Any] = {}

    @property
    def epoch(self) -> int:
        return self.state.epoch

    def load_checkpoint(self, path: str) -> None:
        """ checkpoint """
        loaded_state = load_checkpoint(path, self.model, self.optimizer, self.scheduler)
        if loaded_state is not None:
            self.state = loaded_state

    def save_checkpoint(self, tag: str = "latest") -> None:
        """ checkpoint """
        if not self.output_dir:
            return
        save_checkpoint(self.output_dir, self.model, self.optimizer, self.scheduler, tag, self.state)

    def train(self, max_epochs: int) -> None:
        """"""
        if torch is None:
            raise RuntimeError("PyTorch is required for training.")
        if self.hooks is not None:
            self.hooks.call("on_train_start", self)

        self.model.to(self.device)

        for epoch in range(1, max_epochs + 1):
            self.state.epoch = epoch
            if self.hooks is not None:
                self.hooks.call("on_epoch_start", self)

            self._train_one_epoch()

            if self.hooks is not None:
                self.hooks.call("on_epoch_end", self)

        if self.hooks is not None:
            self.hooks.call("on_train_end", self)

    def _train_one_epoch(self) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for training.")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.dataloader, start=1):
            self.state.step = step
            self.state.global_step += 1
            if self.hooks is not None:
                self.hooks.call("on_step_start", self, step)

            logs = self._train_step(batch, step)

            if self.hooks is not None:
                self.hooks.call("on_step_end", self, step, logs)

    def _train_step(self, batch: Dict[str, Any], step: int) -> Dict[str, Any]:
        if torch is None:
            raise RuntimeError("PyTorch is required for training.")
        images = batch["images"].to(self.device)
        instances = batch.get("instances", [])

        if self.teacher is not None and self.teacher_mode == "online":
            metas = batch.get("meta", [])
            image_ids = batch.get("image_ids", [])
            keys: List[Any] = []
            for i, meta in enumerate(metas):
                if i < len(image_ids):
                    keys.append(image_ids[i])
                else:
                    keys.append(meta.get("path", meta.get("image_id", i)))

            cached = [self._online_cache.get(key) for key in keys]
            missing_indices = [i for i, val in enumerate(cached) if val is None]
            if missing_indices:
                sub_images = images[missing_indices]
                sub_metas = [metas[i] for i in missing_indices]
                sub_ids = [image_ids[i] for i in missing_indices] if image_ids else None
                generated = self.teacher.generate(sub_images, sub_metas, image_ids=sub_ids)
                for idx, insts in zip(missing_indices, generated):
                    self._online_cache[keys[idx]] = insts
                    cached[idx] = insts
            instances = cached
            batch["instances"] = instances

        with _autocast_context(self.device.type, self.amp):
            outputs = self.model(images, targets=batch.get("instances"))
            loss_dict = self.loss_fn(outputs, batch)
            loss = loss_dict.get("total")

        if loss is None:
            raise RuntimeError("Loss function must return a 'total' key.")
        if isinstance(loss, torch.Tensor) and not loss.requires_grad:
            raise RuntimeError(
                "Loss tensor is detached before backward. This usually means the student stayed in eval/inference "
                "state after validation or the forward path entered no_grad unexpectedly."
            )

        if self.amp and self.scaler is not None:
            self.scaler.scale(loss / self.grad_accum).backward()
        else:
            (loss / self.grad_accum).backward()

        if step % self.grad_accum == 0:
            if self.clip_grad_norm and self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            if self.amp and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.state.optimizer_step += 1
            self.optimizer.zero_grad(set_to_none=True)

        logs = {k: (float(v.detach().cpu()) if hasattr(v, "detach") else v) for k, v in loss_dict.items()}
        logs["lr"] = float(self.optimizer.param_groups[0].get("lr", 0.0))
        return logs

    def evaluate(self) -> Dict[str, Any]:
        """"""
        if self.evaluator is None:
            return {}
        return self.evaluator.evaluate(self.model)


class _nullcontext:  # pragma: no cover -
    """ import  contextlib.nullcontext """

    def __enter__(self) -> None:
        return None

    def __exit__(self, *excinfo: Any) -> None:
        return None


def _build_grad_scaler(device_type: str, enabled: bool) -> Optional[Any]:
    """ AMP GradScaler"""
    if AmpGradScaler is None:
        return None
    try:
        return AmpGradScaler(device_type=device_type, enabled=enabled)
    except TypeError:
        return AmpGradScaler(enabled=enabled)


def _autocast_context(device_type: str, enabled: bool) -> Any:
    """ AMP autocast """
    if amp_autocast is None or not enabled:
        return _nullcontext()
    try:
        return amp_autocast(device_type=device_type, enabled=enabled)
    except TypeError:
        return amp_autocast(enabled=enabled)
