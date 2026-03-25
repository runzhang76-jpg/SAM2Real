from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ultralytics import YOLO

from sam2real.utils.paths import resolve_project_path


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Unable to parse boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a hard-distill YOLOv8 segmentation checkpoint.")
    parser.add_argument("--weights", "--model", dest="weights", type=str, required=True, help="Checkpoint path to validate.")
    parser.add_argument("--data", type=str, default="", help="Dataset YAML path. If omitted, infer from checkpoint name.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", type=str, default=None, help="Validation device, e.g. 0 or cpu.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to validate: val or test.")
    parser.add_argument("--project", type=str, default=None, help="Ultralytics output project directory.")
    parser.add_argument("--name", type=str, default=None, help="Ultralytics run name.")
    parser.add_argument("--save-json", nargs="?", const=True, default=False, type=str2bool, help="Export COCO JSON results.")
    parser.add_argument("--plots", nargs="?", const=True, default=True, type=str2bool, help="Save validation plots.")
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    return Path(resolve_project_path(value))


def infer_data_yaml(weights_path: Path) -> Path:
    run_name = weights_path.resolve().parents[1].name
    if run_name.startswith("shot_"):
        candidate = PROJECT_ROOT / "configs" / "hard_distill" / "generated" / f"{run_name}.yaml"
        if candidate.exists():
            return candidate.resolve()
    if run_name == "pseudo_pretrain":
        candidate = PROJECT_ROOT / "configs" / "hard_distill" / "cdw_pseudo.yaml"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Unable to infer data YAML for checkpoint: {weights_path}. Pass --data explicitly."
    )


def build_val_kwargs(args: argparse.Namespace, data_path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "data": str(data_path),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "split": str(args.split),
        "save_json": bool(args.save_json),
        "plots": bool(args.plots),
    }
    if args.device is not None:
        kwargs["device"] = args.device
    if args.project is not None:
        kwargs["project"] = str(_resolve_path(args.project))
    if args.name is not None:
        kwargs["name"] = args.name
    return kwargs


def _metric_group_dict(metrics_group: Any) -> dict[str, float] | None:
    if metrics_group is None:
        return None
    result: dict[str, float] = {}
    for src_key, dst_key in (
        ("mp", "precision"),
        ("mr", "recall"),
        ("map50", "map50"),
        ("map", "map50_95"),
    ):
        value = getattr(metrics_group, src_key, None)
        if value is not None:
            result[dst_key] = float(value)
    return result or None


def _format_metric_group(title: str, metrics_group: dict[str, float] | None) -> list[str]:
    if not metrics_group:
        return [f"{title}: unavailable"]
    return [
        f"{title}:",
        f"  Precision: {metrics_group.get('precision', float('nan')):.6f}",
        f"  Recall: {metrics_group.get('recall', float('nan')):.6f}",
        f"  mAP50: {metrics_group.get('map50', float('nan')):.6f}",
        f"  mAP50-95: {metrics_group.get('map50_95', float('nan')):.6f}",
    ]


def build_summary(metrics: Any, weights_path: Path, data_path: Path, split: str) -> dict[str, Any]:
    mask_group = _metric_group_dict(getattr(metrics, "seg", None) or getattr(metrics, "mask", None))
    box_group = _metric_group_dict(getattr(metrics, "box", None))
    save_dir = Path(getattr(metrics, "save_dir", weights_path.resolve().parents[1]))
    return {
        "weights": str(weights_path.resolve()),
        "data": str(data_path.resolve()),
        "split": split,
        "save_dir": str(save_dir.resolve()),
        "box": box_group,
        "mask": mask_group,
    }


def write_summary(summary: dict[str, Any]) -> tuple[Path, Path]:
    save_dir = Path(summary["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "evaluation_summary.json"
    txt_path = save_dir / "evaluation_summary.txt"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"Weights: {summary['weights']}",
        f"Data: {summary['data']}",
        f"Split: {summary['split']}",
        f"Save dir: {summary['save_dir']}",
        "",
    ]
    lines.extend(_format_metric_group("Box metrics", summary.get("box")))
    lines.append("")
    lines.extend(_format_metric_group("Mask metrics", summary.get("mask")))
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, txt_path


def main() -> None:
    args = parse_args()
    weights_path = _resolve_path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    data_path = _resolve_path(args.data) if str(args.data).strip() else infer_data_yaml(weights_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_path}")

    model = YOLO(str(weights_path))
    metrics = model.val(**build_val_kwargs(args, data_path=data_path))
    summary = build_summary(metrics, weights_path=weights_path, data_path=data_path, split=str(args.split))
    json_path, txt_path = write_summary(summary)

    for line in _format_metric_group("Box metrics", summary.get("box")):
        print(line)
    for line in _format_metric_group("Mask metrics", summary.get("mask")):
        print(line)
    print(f"Validation results dir: {summary['save_dir']}")
    print(f"Summary JSON: {json_path}")
    print(f"Summary TXT: {txt_path}")


if __name__ == "__main__":
    main()
