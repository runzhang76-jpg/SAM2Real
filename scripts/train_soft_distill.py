#!/usr/bin/env python
""""""
# test
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TextIO

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matmatch2real.core.hooks import (
    CheckpointHook,
    EvalHook,
    Hook,
    HookManager,
    LoggingHook,
    TensorboardHook,
    VisualizationHook,
)
from matmatch2real.data.remote import ensure_dataset_available
from matmatch2real.soft_distill import (
    CocoEvaluator,
    DistillEngine,
    build_eval_dataloader,
    build_train_dataloader,
    prepare_class_mappings,
)
from matmatch2real.config.loader import load_config, save_config
from matmatch2real.utils.logging import setup_logger
from matmatch2real.utils.seed import set_seed

try:
    import torch
except Exception:  # pragma: no cover - runtime dependency
    torch = None  # type: ignore


class _TeeStream:
    """Mirror stdout/stderr to both terminal and a log file."""

    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self.primary = primary
        self.secondary = secondary

    def write(self, data: str) -> int:
        self.primary.write(data)
        self.secondary.write(data)
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.primary, "isatty", lambda: False)())


class TrainStepLoggingHook(Hook):
    """"""

    priority = 40

    def __init__(self, log_every: int = 10) -> None:
        self.log_every = max(1, int(log_every))
        self.logger = setup_logger("distill")
        self._total_items: int | None = None
        self._batch_size: int = 1

    def on_epoch_start(self, engine: "DistillEngine") -> None:  # noqa: F821
        self._total_items = None
        self._batch_size = int(getattr(engine.dataloader, "batch_size", 1) or 1)
        try:
            dataset = getattr(engine.dataloader, "dataset", None)
            if dataset is not None:
                self._total_items = len(dataset)
        except Exception:
            self._total_items = None
        self.logger.info("epoch=%d start", engine.epoch)

    def on_step_end(self, engine: "DistillEngine", step: int, logs: Dict[str, Any]) -> None:  # noqa: F821
        if step % self.log_every != 0:
            return
        progress = ""
        if self._total_items is not None and self._total_items > 0:
            current_item = min(step * self._batch_size, self._total_items)
            remain = max(self._total_items - current_item, 0)
            progress = f" item={current_item}/{self._total_items} remain={remain}"
        log_str = ", ".join(self._format_log_item(k, v) for k, v in logs.items())
        self.logger.info("epoch=%d step=%d%s %s", engine.epoch, step, progress, log_str)

    def _format_log_item(self, key: str, value: Any) -> str:
        if not isinstance(value, float):
            return f"{key}={value}"
        if key == "lr" or (0.0 < abs(value) < 1e-3):
            return f"{key}={value:.3e}"
        return f"{key}={value:.4f}"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 distillation training")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "distill" / "distill_default.yaml"),
        help="Path to config YAML/JSON",
    )
    parser.add_argument("--output-dir", default="", help="Override output directory")
    parser.add_argument("--resume", default="", help="Checkpoint path to resume")
    parser.add_argument("--device", default="", help="Override device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=-1, help="Override random seed")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if torch is None:
        return "cpu"
    if not requested or requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def build_optimizer(cfg: Dict[str, Any], model: Any) -> Any:
    optim_cfg = cfg.get("optim", {})
    lr = float(optim_cfg.get("lr", 1e-4))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    optim_type = optim_cfg.get("type", "adamw").lower()
    if optim_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optim_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer type: {optim_type}")


def build_scheduler(
    cfg: Dict[str, Any],
    optimizer: Any,
    steps_per_epoch: int,
    max_epochs: int,
    grad_accum: int,
) -> Any:
    optim_cfg = cfg.get("optim", {})
    sched_cfg = optim_cfg.get("scheduler", {})
    if not bool(sched_cfg.get("enabled", False)):
        return None

    sched_type = str(sched_cfg.get("type", "cosine")).lower()
    grad_accum = max(1, int(grad_accum))
    optimizer_steps_per_epoch = max(1, math.ceil(max(1, int(steps_per_epoch)) / grad_accum))
    total_steps = max(1, optimizer_steps_per_epoch * max(1, int(max_epochs)))

    warmup_steps = int(sched_cfg.get("warmup_steps", 0) or 0)
    if warmup_steps <= 0:
        warmup_epochs = float(sched_cfg.get("warmup_epochs", 0.0) or 0.0)
        warmup_steps = int(round(warmup_epochs * optimizer_steps_per_epoch))
    warmup_steps = max(0, min(warmup_steps, max(total_steps - 1, 0)))

    warmup_start_factor = float(sched_cfg.get("warmup_start_factor", 0.1))
    warmup_start_factor = min(max(warmup_start_factor, 1e-6), 1.0)
    min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.1))
    min_lr_ratio = min(max(min_lr_ratio, 0.0), 1.0)

    step_size_epochs = int(sched_cfg.get("step_size_epochs", 4))
    step_size_steps = max(1, step_size_epochs * optimizer_steps_per_epoch)
    gamma = float(sched_cfg.get("gamma", 0.5))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            progress = float(current_step + 1) / float(warmup_steps)
            return warmup_start_factor + progress * (1.0 - warmup_start_factor)

        if sched_type == "constant":
            return 1.0

        if sched_type == "step":
            decay_count = (current_step - warmup_steps) // step_size_steps if current_step >= warmup_steps else 0
            return max(min_lr_ratio, gamma ** max(0, int(decay_count)))

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(max((current_step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        path = Path(args.output_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "work_dir" / timestamp


def main() -> None:
    from matmatch2real.soft_distill import DistillLoss, build_student
    from matmatch2real.teacher.matmatch_teacher import build_teacher

    args = parse_args()
    output_path = _resolve_output_dir(args)
    output_path.mkdir(parents=True, exist_ok=True)

    console_log = (output_path / "console.log").open("a", encoding="utf-8")
    sys.stdout = _TeeStream(sys.stdout, console_log)
    sys.stderr = _TeeStream(sys.stderr, console_log)

    cfg = load_config(args.config)

    exp_cfg = cfg.get("experiment", {})
    logging_cfg = cfg.get("logging", {})
    trainer_cfg = cfg.get("trainer", {})
    data_cfg = cfg.get("data", {})
    eval_cfg = data_cfg.get("eval", {})
    teacher_cfg = cfg.get("teacher", {})
    student_cfg = cfg.get("student", {})

    logger = setup_logger("distill", log_file=str(output_path / "train.log"), level=logging_cfg.get("level", "INFO"))

    device = resolve_device(args.device or exp_cfg.get("device", "auto"))
    seed = args.seed if args.seed >= 0 else int(exp_cfg.get("seed", 42))

    logger.info("run_dir=%s", output_path)
    logger.info("command=%s", " ".join(sys.argv))
    logger.info("config=%s", Path(args.config).resolve())
    logger.info("device=%s seed=%d", device, seed)
    set_seed(seed)

    #
    data_root = ensure_dataset_available(cfg, output_dir=output_path)
    cfg.setdefault("data", {}).setdefault("train", {})["resolved_root"] = str(data_root)
    prepare_class_mappings(cfg, data_root)

    #
    save_config(cfg, str(output_path / "config_snapshot.json"))

    config_src = Path(args.config)
    config_copy_path = output_path / f"config_source{config_src.suffix or '.yaml'}"
    config_copy_path.write_text(config_src.read_text(encoding="utf-8"), encoding="utf-8")

    dataloader = build_train_dataloader(cfg, data_root)

    teacher = build_teacher(teacher_cfg, device=device)

    student = build_student(student_cfg)
    loss_fn = DistillLoss(cfg.get("loss", {}))
    optimizer = build_optimizer(cfg, student)
    max_epochs = int(trainer_cfg.get("max_epochs", 1))
    scheduler = build_scheduler(
        cfg,
        optimizer,
        steps_per_epoch=len(dataloader),
        max_epochs=max_epochs,
        grad_accum=int(trainer_cfg.get("grad_accum", 1)),
    )
    if scheduler is None:
        logger.info("lr scheduler disabled")
    else:
        logger.info("lr scheduler enabled: %s", cfg.get("optim", {}).get("scheduler", {}))

    evaluator = None
    if eval_cfg.get("enabled", False):
        gt_json = eval_cfg.get("gt_json")
        iou_types = eval_cfg.get("coco", {}).get("iou_types", ["segm"])
        eval_loader = build_eval_dataloader(cfg, data_root)
        resolved_gt = getattr(eval_loader.dataset, "gt_json", gt_json)
        evaluator = CocoEvaluator(eval_loader, gt_json=resolved_gt, iou_types=iou_types)

    progress_cfg = logging_cfg.get("progress", {})
    progress_enabled = bool(progress_cfg.get("enabled", True))  #
    progress_keys = progress_cfg.get("keys")
    progress_every = int(progress_cfg.get("refresh_every", trainer_cfg.get("log_every", 10)))

    hook_list = []
    hook_list.append(TrainStepLoggingHook(log_every=int(trainer_cfg.get("log_every", 10))))
    hook_list.append(LoggingHook(log_every=10**9))
    if logging_cfg.get("tensorboard", {}).get("enabled", False):
        tb_dir = logging_cfg.get("tensorboard", {}).get("log_dir", str(output_path / "tb"))
        hook_list.append(TensorboardHook(log_dir=tb_dir, log_every=int(trainer_cfg.get("log_every", 10))))
    hook_list.extend(
        [
            CheckpointHook(save_every=int(trainer_cfg.get("save_every", 1))),
            EvalHook(eval_every=int(trainer_cfg.get("eval_every", 1))),
            VisualizationHook(),
        ]
    )
    hooks = HookManager(hook_list)

    engine = DistillEngine(
        model=student,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        device=device,
        amp=bool(exp_cfg.get("amp", False)),
        grad_accum=int(trainer_cfg.get("grad_accum", 1)),
        clip_grad_norm=float(trainer_cfg.get("clip_grad_norm", 0.0)),
        teacher=teacher,
        teacher_mode=str(teacher_cfg.get("mode", "offline")),
        evaluator=evaluator,
        visualizer=None,
        hooks=hooks,
        output_dir=str(output_path),
    )

    if args.resume:
        logger.info("resuming from checkpoint: %s", args.resume)
        engine.load_checkpoint(args.resume)

    engine.train(max_epochs=max_epochs)

    engine.save_checkpoint(tag="final")


if __name__ == "__main__":
    main()
