#!/usr/bin/env python
"""蒸馏学生模型的评估脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PKG_ROOT = ROOT / "distill_cdw"
if PKG_ROOT.exists() and str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from distill.core.engine import DistillEngine
from distill.data.datasets import build_eval_dataloader
from distill.data.remote import ensure_dataset_available
from distill.evaluation.coco_eval import CocoEvaluator
from distill.student.losses import DistillLoss
from distill.student.models import build_student
from distill.utils.config import load_config
from distill.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate distilled student model")
    parser.add_argument("--config", required=True, help="Path to config")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("distill")

    try:
        import torch  # noqa: F401
    except Exception:
        logger.error("PyTorch is required for evaluation.")
        return

    data_root = ensure_dataset_available(cfg)
    dataloader = build_eval_dataloader(cfg, data_root)
    if dataloader is None:
        logger.error("评估数据未配置或加载失败，无法执行评估。")
        return

    student = build_student(cfg.get("student", {}))
    loss_fn = DistillLoss(cfg.get("loss", {}))

    import torch

    optimizer = torch.optim.SGD(student.parameters(), lr=0.0)

    engine = DistillEngine(
        model=student,
        loss_fn=loss_fn,
        optimizer=optimizer,
        dataloader=dataloader,
        device=cfg.get("experiment", {}).get("device", "cpu"),
    )

    engine.load_checkpoint(args.checkpoint)

    eval_cfg = cfg.get("data", {}).get("eval", {})
    iou_types = eval_cfg.get("coco", {}).get("iou_types", ["segm"])
    gt_json = getattr(dataloader.dataset, "gt_json", eval_cfg.get("gt_json"))
    evaluator = CocoEvaluator(dataloader, gt_json=gt_json, iou_types=iou_types)
    metrics = evaluator.evaluate(student)
    logger.info("evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()
