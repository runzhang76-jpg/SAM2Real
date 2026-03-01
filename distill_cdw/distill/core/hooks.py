"""用于日志、保存、评估与可视化的训练 Hook。"""

from __future__ import annotations

from dataclasses import dataclass
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

from distill.utils.logging import get_logger

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm 可选
    tqdm = None


class Hook:
    """基础 Hook，回调默认空实现。"""

    priority: int = 50

    def on_train_start(self, engine: "DistillEngine") -> None:  # noqa: F821
        pass

    def on_epoch_start(self, engine: "DistillEngine") -> None:  # noqa: F821
        pass

    def on_step_start(self, engine: "DistillEngine", step: int) -> None:  # noqa: F821
        pass

    def on_step_end(self, engine: "DistillEngine", step: int, logs: Dict[str, Any]) -> None:  # noqa: F821
        pass

    def on_epoch_end(self, engine: "DistillEngine") -> None:  # noqa: F821
        pass

    def on_train_end(self, engine: "DistillEngine") -> None:  # noqa: F821
        pass


class HookManager:
    """管理 Hook 的排序与分发。"""

    def __init__(self, hooks: Optional[Iterable[Hook]] = None) -> None:
        self.hooks: List[Hook] = sorted(list(hooks or []), key=lambda h: h.priority)

    def call(self, name: str, *args: Any, **kwargs: Any) -> None:
        for hook in self.hooks:
            callback = getattr(hook, name, None)
            if callable(callback):
                callback(*args, **kwargs)


class _SimpleProgress:
    """tqdm 不可用时的简易进度条。"""

    def __init__(self, total: Optional[int], desc: str, width: int = 28) -> None:
        self.total = total
        self.desc = desc
        self.width = width
        self.count = 0
        self.start = time.time()
        self.postfix: Dict[str, Any] = {}

    def update(self, n: int = 1) -> None:
        self.count += n
        self._render()

    def set_postfix(self, data: Dict[str, Any]) -> None:
        self.postfix = data
        self._render()

    def close(self) -> None:
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _render(self) -> None:
        elapsed = max(time.time() - self.start, 1e-6)
        rate = self.count / elapsed
        if self.total:
            pct = min(self.count / self.total, 1.0)
            filled = int(self.width * pct)
            bar = "#" * filled + "-" * (self.width - filled)
            main = f"{self.desc} [{bar}] {self.count}/{self.total} {pct*100:5.1f}% {rate:.1f}it/s"
        else:
            main = f"{self.desc} {self.count} {rate:.1f}it/s"
        postfix = " ".join(f"{k}={v}" for k, v in self.postfix.items())
        line = f"\r{main} {postfix}".rstrip()
        sys.stderr.write(line)
        sys.stderr.flush()


@dataclass
class ProgressBarHook(Hook):
    """训练进度条 Hook，动态展示训练状态。"""

    total_epochs: Optional[int] = None
    refresh_every: int = 1
    keys: Optional[List[str]] = None
    priority: int = 40

    def __post_init__(self) -> None:
        self.logger = get_logger("distill")
        self._pbar: Optional[Any] = None
        self._last_step: int = 0

    def on_epoch_start(self, engine: "DistillEngine") -> None:  # noqa: F821
        total = None
        try:
            total = len(engine.dataloader)
        except Exception:
            total = None
        if self.total_epochs:
            desc = f"Epoch {engine.epoch}/{self.total_epochs}"
        else:
            desc = f"Epoch {engine.epoch}"
        if tqdm is not None:
            self._pbar = tqdm(total=total, desc=desc, leave=False, dynamic_ncols=True)
        else:
            self._pbar = _SimpleProgress(total=total, desc=desc)
        self._last_step = 0

    def on_step_end(self, engine: "DistillEngine", step: int, logs: Dict[str, Any]) -> None:  # noqa: F821
        if self._pbar is None:
            return
        delta = step - self._last_step
        if delta <= 0:
            delta = 1
        self._pbar.update(delta)
        self._last_step = step
        if self.refresh_every > 1 and step % self.refresh_every != 0:
            return
        postfix = self._select_postfix(logs)
        if postfix:
            self._pbar.set_postfix(postfix)

    def on_epoch_end(self, engine: "DistillEngine") -> None:  # noqa: F821
        if self._pbar is None:
            return
        self._pbar.close()
        self._pbar = None

    def _select_postfix(self, logs: Dict[str, Any]) -> Dict[str, str]:
        preferred = self.keys or [
            "total",
            "loss_supervised",
            "loss_distill",
            "loss_mask",
            "loss_dice",
            "reliability_weight",
            "lr",
        ]
        postfix: Dict[str, str] = {}
        for key in preferred:
            value = logs.get(key)
            if isinstance(value, (int, float)):
                postfix[key] = self._format_value(key, float(value))
        if postfix:
            return postfix
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                postfix[key] = self._format_value(key, float(value))
            if len(postfix) >= 4:
                break
        return postfix

    def _format_value(self, key: str, value: float) -> str:
        if key == "lr":
            return f"{value:.3e}"
        return f"{value:.4f}"


@dataclass
class LoggingHook(Hook):
    """按固定间隔记录指标。"""

    log_every: int = 10

    def __post_init__(self) -> None:
        self.logger = get_logger("distill")

    def on_step_end(self, engine: "DistillEngine", step: int, logs: Dict[str, Any]) -> None:  # noqa: F821
        if step % self.log_every != 0:
            return
        log_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items())
        self.logger.info("epoch=%d step=%d %s", engine.epoch, step, log_str)


@dataclass
class TensorboardHook(Hook):
    """在可用时将指标写入 TensorBoard。"""

    log_dir: str
    log_every: int = 10
    priority: int = 55

    def __post_init__(self) -> None:
        self.logger = get_logger("distill")
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            self.writer = SummaryWriter(log_dir=self.log_dir)
        except Exception as exc:
            self.writer = None
            self.logger.warning("TensorBoard not available: %s", exc)

    def on_step_end(self, engine: "DistillEngine", step: int, logs: Dict[str, Any]) -> None:  # noqa: F821
        if self.writer is None or step % self.log_every != 0:
            return
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, engine.state.step)

    def on_train_end(self, engine: "DistillEngine") -> None:  # noqa: F821
        if self.writer is not None:
            self.writer.close()


@dataclass
class CheckpointHook(Hook):
    """按周期保存 checkpoint。"""

    save_every: int = 1
    priority: int = 60

    def __post_init__(self) -> None:
        self.logger = get_logger("distill")

    def on_epoch_end(self, engine: "DistillEngine") -> None:  # noqa: F821
        if engine.epoch % self.save_every != 0:
            return
        engine.save_checkpoint(tag=f"epoch_{engine.epoch}")
        self.logger.info("checkpoint saved: epoch %d", engine.epoch)


@dataclass
class EvalHook(Hook):
    """按周期执行评估。"""

    eval_every: int = 1
    priority: int = 70

    def __post_init__(self) -> None:
        self.logger = get_logger("distill")

    def on_epoch_end(self, engine: "DistillEngine") -> None:  # noqa: F821
        if engine.evaluator is None or engine.epoch % self.eval_every != 0:
            return
        metrics = engine.evaluate()
        if metrics:
            self.logger.info("eval metrics: %s", metrics)


@dataclass
class VisualizationHook(Hook):
    """可视化 Hook（mask 叠加、PR 曲线等）。"""

    priority: int = 80

    def __post_init__(self) -> None:
        self.logger = get_logger("distill")

    def on_epoch_end(self, engine: "DistillEngine") -> None:  # noqa: F821
        if engine.visualizer is None:
            return
        if callable(engine.visualizer):
            engine.visualizer(engine)
            return
        handler = getattr(engine.visualizer, "on_epoch_end", None)
        if callable(handler):
            handler(engine)
            return
        self.logger.warning("visualizer 未实现可调用接口或 on_epoch_end 方法")
