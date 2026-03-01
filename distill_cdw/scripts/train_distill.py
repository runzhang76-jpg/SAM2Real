#!/usr/bin/env python
"""蒸馏训练主入口。"""
# test
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PKG_ROOT = ROOT / "distill_cdw"
if PKG_ROOT.exists() and str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from distill.core.engine import DistillEngine
from distill.core.hooks import (
    CheckpointHook,
    EvalHook,
    HookManager,
    LoggingHook,
    ProgressBarHook,
    TensorboardHook,
    VisualizationHook,
)
from distill.data.datasets import build_train_dataloader, build_eval_dataloader
from distill.data.remote import ensure_dataset_available
from distill.evaluation.coco_eval import CocoEvaluator
from distill.student.losses import DistillLoss
from distill.student.models import build_student
from distill.teacher.sam2_teacher import build_teacher
from distill.utils.config import load_config, save_config
from distill.utils.logging import setup_logger
from distill.utils.seed import set_seed

import torch



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 distillation training")
    parser.add_argument("--config", default='distill_cdw/configs/distill_default.yaml', help="Path to config YAML/JSON")
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    exp_cfg = cfg.get("experiment", {})
    logging_cfg = cfg.get("logging", {})
    trainer_cfg = cfg.get("trainer", {})
    data_cfg = cfg.get("data", {})
    eval_cfg = data_cfg.get("eval", {})
    teacher_cfg = cfg.get("teacher", {})
    student_cfg = cfg.get("student", {})

    output_dir = args.output_dir or exp_cfg.get("output_dir", "outputs/distill")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("distill", log_file=str(output_path / "train.log"), level=logging_cfg.get("level", "INFO"))

    device = resolve_device(args.device or exp_cfg.get("device", "auto"))
    seed = args.seed if args.seed >= 0 else int(exp_cfg.get("seed", 42))

    logger.info("device=%s seed=%d", device, seed)
    set_seed(seed)

    # 远程数据集同步（可选）。
    data_root = ensure_dataset_available(cfg, output_dir=output_path)
    cfg.setdefault("data", {}).setdefault("train", {})["resolved_root"] = str(data_root)

    # 保存实验设置
    save_config(cfg, str(output_path / "config_snapshot.json"))

    dataloader = build_train_dataloader(cfg, data_root)

    teacher = build_teacher(teacher_cfg, device=device)

    student = build_student(student_cfg)
    loss_fn = DistillLoss(cfg.get("loss", {}))
    optimizer = build_optimizer(cfg, student)

    evaluator = None
    if eval_cfg.get("enabled", False):
        gt_json = eval_cfg.get("gt_json")
        iou_types = eval_cfg.get("coco", {}).get("iou_types", ["segm"])
        eval_loader = build_eval_dataloader(cfg, data_root)
        resolved_gt = getattr(eval_loader.dataset, "gt_json", gt_json)
        evaluator = CocoEvaluator(eval_loader, gt_json=resolved_gt, iou_types=iou_types)

    max_epochs = int(trainer_cfg.get("max_epochs", 1))
    progress_cfg = logging_cfg.get("progress", {})
    progress_enabled = bool(progress_cfg.get("enabled", True))  # 进度条开关
    progress_keys = progress_cfg.get("keys")
    progress_every = int(progress_cfg.get("refresh_every", trainer_cfg.get("log_every", 10)))

    hook_list = []
    if progress_enabled:
        hook_list.append(
            ProgressBarHook(
                total_epochs=max_epochs,
                refresh_every=max(1, progress_every),
                keys=progress_keys if isinstance(progress_keys, list) else None,
            )
        )
    hook_list.append(LoggingHook(log_every=int(trainer_cfg.get("log_every", 10))))
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
