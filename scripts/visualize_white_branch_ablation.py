#!/usr/bin/env python
"""Visualize LabCC prompting with and without the white-object branch."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matmatch2real.config.loader import load_config
from matmatch2real.teacher.prompt_generators import LabCCBoxPromptGenerator


@dataclass
class SampleScore:
    image_id: int
    file_name: str
    image_path: Path
    score: float
    added_boxes: List[List[float]]
    without_boxes: np.ndarray
    with_boxes: np.ndarray
    without_mask: np.ndarray
    with_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a paper-style white-object branch ablation figure.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "teacher" / "teacher_default.yaml"))
    parser.add_argument("--gt-json", default="../data/cdw_classify/dataset_seg/annotations/instances_test.json")
    parser.add_argument("--images-root", default="../data/cdw_classify/dataset_seg/images/test")
    parser.add_argument("--prefer-name", default="", help="Optional image file name or substring to force/prefer.")
    parser.add_argument("--skip-top", type=int, default=0, help="Skip this many highest-scoring candidates.")
    parser.add_argument("--max-images", type=int, default=80, help="Maximum candidate images to scan.")
    parser.add_argument("--num-samples", type=int, default=1, choices=[1, 2], help="Use one or two selected images.")
    parser.add_argument("--added-iou-thr", type=float, default=0.2, help="IoU threshold below which a with-branch box is considered newly recovered.")
    parser.add_argument("--png-out", default="outputs/fig_4_3_1_white_branch_visualization.png")
    parser.add_argument("--tif-out", default="outputs/fig_4_3_1_white_branch_visualization.tif")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _load_labcc_cfg(config_path: str) -> Dict[str, Any]:
    cfg = load_config(config_path)
    lab_cfg = copy.deepcopy(cfg.get("teacher", {}).get("prompt_generator", {}).get("lab_cc_boxes", {}))
    if not lab_cfg:
        raise ValueError(f"No teacher.prompt_generator.lab_cc_boxes config found in {config_path}")
    return lab_cfg


def _branch_cfg(lab_cfg: Dict[str, Any], enabled: bool) -> Dict[str, Any]:
    out = copy.deepcopy(lab_cfg)
    out.setdefault("white_object_branch", {})["enabled"] = bool(enabled)
    out["save_debug"] = False
    return out


def _labcc_final_mask(image_rgb: np.ndarray, lab_cfg: Dict[str, Any], enabled: bool) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required to build LabCC masks.") from exc

    cfg = _branch_cfg(lab_cfg, enabled)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    colored_mask = ((b_ch > int(cfg.get("b_thresh", 133))) | (a_ch > int(cfg.get("a_thresh", 131)))) & (
        l_ch > int(cfg.get("l_thresh_min", 18))
    )

    white_mask = np.zeros_like(colored_mask, dtype=bool)
    white_cfg = cfg.get("white_object_branch", {})
    if bool(white_cfg.get("enabled", False)):
        a_centered = a_ch.astype(np.int16) - 128
        b_centered = b_ch.astype(np.int16) - 128
        chroma_sq = a_centered * a_centered + b_centered * b_centered
        chroma_max = float(white_cfg.get("chroma_max", 12.0))
        white_mask = (
            (l_ch >= int(white_cfg.get("l_min", 200)))
            & (l_ch <= int(white_cfg.get("l_max", 255)))
            & (chroma_sq <= chroma_max * chroma_max)
        )

    raw_mask = ((colored_mask | white_mask).astype(np.uint8)) * 255
    close_kernel = np.ones((max(1, int(cfg.get("close_kernel", 5))), max(1, int(cfg.get("close_kernel", 5)))), np.uint8)
    open_kernel = np.ones((max(1, int(cfg.get("open_kernel", 3))), max(1, int(cfg.get("open_kernel", 3)))), np.uint8)
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, open_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((clean_mask > 0).astype(np.uint8), connectivity=8)
    final_mask = np.zeros_like(clean_mask)
    min_cc_area = max(1, int(cfg.get("min_cc_area", 1500)))
    for i in range(1, num_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_cc_area:
            final_mask[labels == i] = 255
    return final_mask, white_mask.astype(np.uint8) * 255


def _generate_boxes(image_rgb: np.ndarray, lab_cfg: Dict[str, Any], enabled: bool, image_meta: Dict[str, Any]) -> np.ndarray:
    return LabCCBoxPromptGenerator(_branch_cfg(lab_cfg, enabled)).generate_boxes(image_rgb, image_meta=image_meta)


def _added_boxes(with_boxes: np.ndarray, without_boxes: np.ndarray, iou_thr: float) -> List[List[float]]:
    added: List[List[float]] = []
    for box in with_boxes.tolist():
        best_iou = 0.0
        for prev in without_boxes.tolist():
            best_iou = max(best_iou, _box_iou(box, prev))
        if best_iou < iou_thr:
            added.append([float(v) for v in box])
    return added


def _score_sample(
    image_rgb: np.ndarray,
    image_meta: Dict[str, Any],
    image_path: Path,
    lab_cfg: Dict[str, Any],
    added_iou_thr: float,
) -> Optional[SampleScore]:
    without_boxes = _generate_boxes(image_rgb, lab_cfg, False, image_meta)
    with_boxes = _generate_boxes(image_rgb, lab_cfg, True, image_meta)
    without_mask, _ = _labcc_final_mask(image_rgb, lab_cfg, False)
    with_mask, white_mask = _labcc_final_mask(image_rgb, lab_cfg, True)
    added = _added_boxes(with_boxes, without_boxes, added_iou_thr)
    if not added:
        return None

    score = 0.0
    height, width = image_rgb.shape[:2]
    for box in added:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        patch_white = white_mask[y0:y1, x0:x1] > 0
        patch_new = (with_mask[y0:y1, x0:x1] > 0) & ~(without_mask[y0:y1, x0:x1] > 0)
        area = float((x1 - x0) * (y1 - y0))
        white_frac = float(patch_white.mean()) if patch_white.size else 0.0
        new_frac = float(patch_new.mean()) if patch_new.size else 0.0
        score += math.sqrt(area) * (1.0 + 3.0 * white_frac + 2.0 * new_frac)

    if score <= 0.0:
        return None
    return SampleScore(
        image_id=int(image_meta.get("id", image_meta.get("image_id", -1))),
        file_name=str(image_meta.get("file_name", image_path.name)),
        image_path=image_path,
        score=score,
        added_boxes=added,
        without_boxes=without_boxes,
        with_boxes=with_boxes,
        without_mask=without_mask,
        with_mask=with_mask,
    )


def _candidate_images(gt: Dict[str, Any], images_root: Path, prefer_name: str, max_images: int) -> List[Dict[str, Any]]:
    images = list(gt.get("images", []))
    anns = gt.get("annotations", [])
    ceramic_ids = {int(cat["id"]) for cat in gt.get("categories", []) if "ceramic" in str(cat.get("name", "")).lower()}
    ceramic_image_ids = {
        int(ann.get("image_id", -1))
        for ann in anns
        if int(ann.get("category_id", -1)) in ceramic_ids
    }

    def priority(item: Dict[str, Any]) -> Tuple[int, int, str]:
        file_name = str(item.get("file_name", ""))
        image_id = int(item.get("id", -1))
        prefer_hit = bool(prefer_name) and prefer_name.lower() in file_name.lower()
        ceramic_hit = image_id in ceramic_image_ids or "ceramic" in file_name.lower()
        exists = (images_root / file_name).exists()
        return (0 if prefer_hit else 1, 0 if ceramic_hit else 1, "" if exists else "z", file_name)

    ordered = sorted(images, key=priority)
    selected: List[Dict[str, Any]] = []
    for item in ordered:
        if (images_root / str(item.get("file_name", ""))).exists():
            selected.append(item)
        if len(selected) >= max(1, int(max_images)):
            break
    return selected


def _font_properties() -> Any:
    import matplotlib.font_manager as fm

    candidates = [
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/timesbd.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return fm.FontProperties(fname=str(path))
    return fm.FontProperties(family="Times New Roman")


def _overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.32) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required to color foreground components.") from exc

    out = image.astype(np.float32).copy()
    active = (mask > 0).astype(np.uint8)
    if not np.any(active):
        return image

    palette = np.asarray(
        [
            (80, 150, 220),
            (80, 170, 120),
            (230, 170, 70),
            (170, 115, 200),
            (210, 110, 110),
            (70, 175, 175),
            (185, 170, 70),
            (135, 155, 215),
        ],
        dtype=np.float32,
    )
    num_labels, labels = cv2.connectedComponents(active, connectivity=8)
    for label in range(1, num_labels):
        component = labels == label
        color_arr = palette[(label - 1) % len(palette)]
        out[component] = (1.0 - alpha) * out[component] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_panel(
    ax: Any,
    image: np.ndarray,
    mask: Optional[np.ndarray],
    boxes: np.ndarray,
    added: List[List[float]],
    mode: str,
    font_prop: Any,
) -> None:
    from matplotlib.patches import Rectangle

    if mask is not None:
        image = _overlay_mask(image, mask, alpha=0.32)
    ax.imshow(image)
    ax.set_axis_off()

    for box in boxes.tolist():
        x0, y0, x1, y1 = [float(v) for v in box]
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="#c00000",
                linewidth=0.72,
            )
        )


def _render_figure(samples: List[SampleScore], png_out: Path, tif_out: Path, dpi: int) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    font_prop = _font_properties()
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["axes.linewidth"] = 0.6

    rows = len(samples)
    fig_w = 7.2
    fig_h = 2.75 * rows
    fig, axes = plt.subplots(rows, 3, figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    axes_arr = np.asarray(axes).reshape(rows, 3)

    titles = [
        "(a) Original image",
        "(b) Without white-object branch",
        "(c) With white-object branch",
    ]
    for row, sample in enumerate(samples):
        image = np.asarray(Image.open(sample.image_path).convert("RGB"))
        panels = [
            (image, None, np.zeros((0, 4), dtype=np.float32), [], "original"),
            (image, sample.without_mask, sample.without_boxes, [], "without"),
            (image, sample.with_mask, sample.with_boxes, sample.added_boxes, "with"),
        ]
        for col, (img, mask, boxes, added, mode) in enumerate(panels):
            ax = axes_arr[row, col]
            _draw_panel(ax, img, mask, boxes, added, mode, font_prop)
            ax.set_title(titles[col], fontproperties=font_prop, fontsize=10, pad=5)

    fig.subplots_adjust(left=0.015, right=0.985, top=0.92, bottom=0.035, wspace=0.035, hspace=0.12)
    png_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_out, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(tif_out, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    lab_cfg = _load_labcc_cfg(args.config)
    gt_path = Path(args.gt_json)
    images_root = Path(args.images_root)
    gt = json.loads(gt_path.read_text(encoding="utf-8"))

    scored: List[SampleScore] = []
    for image_meta in _candidate_images(gt, images_root, args.prefer_name, args.max_images):
        image_path = images_root / str(image_meta.get("file_name", ""))
        image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
        sample = _score_sample(image_rgb, image_meta, image_path, lab_cfg, args.added_iou_thr)
        if sample is not None:
            scored.append(sample)

    if not scored:
        raise RuntimeError("No image with a clear white-branch addition was found. Increase --max-images or relax --added-iou-thr.")

    scored.sort(key=lambda x: x.score, reverse=True)
    start = max(0, int(args.skip_top))
    selected = scored[start : start + args.num_samples]
    if not selected:
        raise RuntimeError(f"No selected image after --skip-top={args.skip_top}; reduce --skip-top or increase --max-images.")
    _render_figure(selected, Path(args.png_out), Path(args.tif_out), int(args.dpi))

    print(f"saved_png={Path(args.png_out).resolve()}")
    print(f"saved_tif={Path(args.tif_out).resolve()}")
    print("selected_images=" + ", ".join(s.file_name for s in selected))
    for idx, sample in enumerate(selected, start=1):
        print(
            f"sample_{idx}: image_id={sample.image_id} file={sample.file_name} "
            f"without_boxes={sample.without_boxes.shape[0]} with_boxes={sample.with_boxes.shape[0]} "
            f"new_recovered_boxes={len(sample.added_boxes)} score={sample.score:.2f}"
        )


if __name__ == "__main__":
    main()
