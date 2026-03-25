from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "hard_distill" / "default.yaml"


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Unable to parse boolean value: {value}")


def bool_or_str(value: str | bool) -> bool | str:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official Ultralytics YOLOv8-seg training.")
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CONFIG, help="Path to hard-distill config YAML.")
    parser.add_argument("--model", type=str, default=None, help="Model weights or model YAML.")
    parser.add_argument("--data", type=str, default=None, help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=None, help="Input image size.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size.")
    parser.add_argument("--device", type=str, default=None, help="Training device.")
    parser.add_argument("--project", type=str, default=None, help="Output project directory.")
    parser.add_argument("--name", type=str, default=None, help="Experiment name.")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers.")
    parser.add_argument("--patience", type=int, default=None, help="Early-stop patience.")
    parser.add_argument("--pretrained", nargs="?", const=True, default=None, type=str2bool, help="Use pretrained weights.")
    parser.add_argument("--resume", nargs="?", const=True, default=None, type=bool_or_str, help="Resume flag or checkpoint path.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer name.")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--close_mosaic", type=int, default=None, help="Epoch to disable mosaic.")
    parser.add_argument("--cache", nargs="?", const=True, default=None, type=bool_or_str, help="Cache mode.")
    parser.add_argument("--amp", nargs="?", const=True, default=None, type=str2bool, help="Enable AMP.")
    parser.add_argument("--single_cls", nargs="?", const=True, default=None, type=str2bool, help="Treat all classes as one.")
    parser.add_argument("--cos_lr", nargs="?", const=True, default=None, type=str2bool, help="Use cosine LR.")
    parser.add_argument("--exist_ok", nargs="?", const=True, default=None, type=str2bool, help="Allow existing output dir.")
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a mapping: {config_path}")
    return config


def merge_config_and_args(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = dict(config)
    for key, value in vars(args).items():
        if key == "cfg" or value is None:
            continue
        merged[key] = value
    return merged


def _resolve_project_path(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def build_train_kwargs(merged: dict[str, Any]) -> dict[str, Any]:
    train_keys = [
        "data",
        "epochs",
        "imgsz",
        "batch",
        "device",
        "project",
        "name",
        "workers",
        "patience",
        "pretrained",
        "resume",
        "seed",
        "optimizer",
        "lr0",
        "close_mosaic",
        "cache",
        "amp",
        "single_cls",
        "cos_lr",
        "exist_ok",
    ]
    return {key: merged[key] for key in train_keys if key in merged}


def normalize_runtime_config(merged: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(merged)
    for key in ("model", "data", "project"):
        if key in normalized:
            normalized[key] = _resolve_project_path(normalized[key])
    resume_value = normalized.get("resume")
    if isinstance(resume_value, str) and resume_value.strip() and resume_value.lower() not in {"true", "false"}:
        normalized["resume"] = _resolve_project_path(resume_value)
    return normalized


def run_yolo_training(cfg_path: Path, overrides: argparse.Namespace | None = None) -> Path:
    config = load_yaml_config(cfg_path)
    merged = merge_config_and_args(config, overrides or argparse.Namespace(cfg=cfg_path))
    merged = normalize_runtime_config(merged)
    if "model" not in merged:
        raise KeyError("Config must define `model`.")

    from ultralytics import YOLO

    model = YOLO(str(Path(merged["model"])))
    train_kwargs = build_train_kwargs(merged)
    model.train(**train_kwargs)
    return Path(model.trainer.save_dir)


def main() -> None:
    args = parse_args()
    save_dir = run_yolo_training(args.cfg, args)
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"

    print(f"Training output: {save_dir}")
    print(f"Best weights: {best_path}")
    print(f"Last weights: {last_path}")


if __name__ == "__main__":
    main()
