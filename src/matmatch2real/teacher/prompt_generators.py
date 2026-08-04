"""Prompt generators for SAM2 teacher."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from matmatch2real.utils.logging import get_logger


class PromptGenerator(Protocol):
    name: str

    def generate_boxes(self, image: np.ndarray, image_meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Return prompt boxes in XYXY format with shape [N, 4]."""


class SAM2AutoPromptGenerator:
    """Placeholder prompt generator for SAM2 auto mode."""

    name = "sam2_auto"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg or {}

    def generate_boxes(self, image: np.ndarray, image_meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
        return np.zeros((0, 4), dtype=np.float32)


class BoxPromptRefiner:
    """Apply common geometric refinement and limiting to generated box prompts."""

    def __init__(self, generator: PromptGenerator, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.generator = generator
        self.name = generator.name
        cfg = cfg or {}
        refinement_cfg = cfg.get("box_refinement", {}) or {}
        self.enabled = bool(refinement_cfg.get("enabled", False))
        self.x_negative_ratio = float(refinement_cfg.get("x_negative_ratio", 0.0))
        self.x_positive_ratio = float(refinement_cfg.get("x_positive_ratio", 0.0))
        self.y_negative_ratio = float(refinement_cfg.get("y_negative_ratio", 0.0))
        self.y_positive_ratio = float(refinement_cfg.get("y_positive_ratio", 0.0))
        self.max_boxes_per_image = int(cfg.get("max_boxes_per_image", 0))

    def generate_boxes(self, image: np.ndarray, image_meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
        boxes = np.asarray(self.generator.generate_boxes(image, image_meta), dtype=np.float32)
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"Prompt generator returned boxes with invalid shape: {boxes.shape}")

        if self.enabled:
            height, width = image.shape[:2]
            refined = boxes.copy()
            box_widths = boxes[:, 2] - boxes[:, 0]
            box_heights = boxes[:, 3] - boxes[:, 1]
            refined[:, 0] -= box_widths * self.x_negative_ratio
            refined[:, 2] += box_widths * self.x_positive_ratio
            refined[:, 1] -= box_heights * self.y_negative_ratio
            refined[:, 3] += box_heights * self.y_positive_ratio
            refined[:, [0, 2]] = np.clip(refined[:, [0, 2]], 0.0, float(width))
            refined[:, [1, 3]] = np.clip(refined[:, [1, 3]], 0.0, float(height))
            boxes = refined[(refined[:, 2] > refined[:, 0]) & (refined[:, 3] > refined[:, 1])]

        if self.max_boxes_per_image > 0:
            boxes = boxes[: self.max_boxes_per_image]
        return boxes.astype(np.float32, copy=False)


def _ensure_odd(value: int, fallback: int = 3) -> int:
    k = int(value)
    if k <= 0:
        k = int(fallback)
    if k % 2 == 0:
        k += 1
    return k


def _clip_box_xyxy(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x0 = int(max(0, min(x0, width - 1)))
    y0 = int(max(0, min(y0, height - 1)))
    x1 = int(max(x0 + 1, min(x1, width)))
    y1 = int(max(y0 + 1, min(y1, height)))
    return x0, y0, x1, y1


def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = a.tolist()
    bx0, by0, bx1, by1 = b.tolist()
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _nms_boxes_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    order = np.argsort(-scores)
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        remain = []
        for j in order[1:]:
            if _box_iou_xyxy(boxes[i], boxes[int(j)]) <= iou_thr:
                remain.append(int(j))
        order = np.asarray(remain, dtype=np.int64)
    return boxes[np.asarray(keep, dtype=np.int64)]


def _overlap_ratio_min_area(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap ratio = inter / min(area_a, area_b)."""
    ax0, ay0, ax1, ay1 = a.tolist()
    bx0, by0, bx1, by1 = b.tolist()
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = min(area_a, area_b)
    return float(inter / denom) if denom > 0.0 else 0.0


def _suppress_high_overlap_keep_large(boxes: np.ndarray, overlap_thr: float) -> np.ndarray:
    """
    If two boxes overlap too much, keep the larger one.
    Overlap metric uses inter/min(area_a, area_b), which is robust for nested boxes.
    """
    if boxes.size == 0:
        return boxes
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)  # large first
    keep: List[int] = []
    for idx in order.tolist():
        b = boxes[idx]
        drop = False
        for kept_idx in keep:
            if _overlap_ratio_min_area(b, boxes[kept_idx]) >= overlap_thr:
                drop = True
                break
        if not drop:
            keep.append(idx)
    return boxes[np.asarray(keep, dtype=np.int64)]


class CannyBoxPromptGenerator:
    """Canny -> contour -> boxes prompt generator."""

    name = "canny_boxes"

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.logger = get_logger("distill")

        self.blur_kernel = _ensure_odd(int(cfg.get("blur_kernel", 7)), fallback=7)
        self.canny_low = int(cfg.get("canny_low", 50))
        self.canny_high = int(cfg.get("canny_high", 150))
        self.dilate_kernel = max(1, int(cfg.get("dilate_kernel", 3)))
        self.dilate_iter = max(0, int(cfg.get("dilate_iter", 1)))
        self.close_kernel = max(1, int(cfg.get("close_kernel", 3)))
        self.min_box_w = max(1, int(cfg.get("min_box_w", 100)))
        self.min_box_h = max(1, int(cfg.get("min_box_h", 100)))
        self.force_square = bool(cfg.get("force_square", True))
        self.high_overlap_filter = bool(cfg.get("high_overlap_filter", False))
        self.high_overlap_thresh = float(cfg.get("high_overlap_thresh", 0.9))
        self.max_prompts_per_image = max(1, int(cfg.get("max_prompts_per_image", 200)))
        self.nms_thresh = float(cfg.get("nms_thresh", 0.7))
        self.save_debug = bool(cfg.get("save_debug", False))
        self.debug_dir = Path(str(cfg.get("debug_dir", "./debug/canny_prompts")))
        if self.save_debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def _format_debug_prefix(self, image_meta: Optional[Dict[str, Any]]) -> str:
        if not image_meta:
            return "img"
        image_id = image_meta.get("image_id", "x")
        file_name = str(image_meta.get("file_name", "")).strip()
        stem = Path(file_name).stem if file_name else f"img_{image_id}"
        return f"{stem}_{image_id}"

    def _save_debug_images(
        self,
        prefix: str,
        image_rgb: np.ndarray,
        edges: np.ndarray,
        closed: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> None:
        try:
            import cv2
        except Exception:
            self.logger.warning("OpenCV unavailable; skip canny debug image save")
            return

        cv2.imwrite(str(self.debug_dir / f"{prefix}_edges.png"), edges)
        cv2.imwrite(str(self.debug_dir / f"{prefix}_closed.png"), closed)

        overlay_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
        for box in boxes_xyxy:
            x0, y0, x1, y1 = [int(v) for v in box.tolist()]
            cv2.rectangle(overlay_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imwrite(str(self.debug_dir / f"{prefix}_boxes.png"), overlay_bgr)

    def generate_boxes(self, image: np.ndarray, image_meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("OpenCV is required for prompt_generator.type=canny_boxes") from exc

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("CannyBoxPromptGenerator expects RGB image in HWC format")
        height, width = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        dilate_kernel = np.ones((self.dilate_kernel, self.dilate_kernel), np.uint8)
        close_kernel = np.ones((self.close_kernel, self.close_kernel), np.uint8)
        dilated = cv2.dilate(edges, dilate_kernel, iterations=self.dilate_iter)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[List[float]] = []
        scores: List[float] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_box_w or h < self.min_box_h:
                continue

            if self.force_square:
                side = int(max(w, h))
                cx = x + w // 2
                cy = y + h // 2
                x0 = int(round(cx - side / 2.0))
                y0 = int(round(cy - side / 2.0))
                x1 = x0 + side
                y1 = y0 + side
            else:
                x0, y0, x1, y1 = x, y, x + w, y + h

            x0, y0, x1, y1 = _clip_box_xyxy(x0, y0, x1, y1, width=width, height=height)
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append([float(x0), float(y0), float(x1), float(y1)])
            scores.append(float((x1 - x0) * (y1 - y0)))

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        boxes_arr = np.asarray(boxes, dtype=np.float32)
        scores_arr = np.asarray(scores, dtype=np.float32)

        if self.high_overlap_filter and self.high_overlap_thresh > 0.0:
            boxes_arr = _suppress_high_overlap_keep_large(boxes_arr, overlap_thr=self.high_overlap_thresh)
            scores_arr = np.asarray([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes_arr], dtype=np.float32)

        if self.nms_thresh > 0.0:
            boxes_arr = _nms_boxes_xyxy(boxes_arr, scores_arr, iou_thr=self.nms_thresh)
            scores_arr = np.asarray([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes_arr], dtype=np.float32)

        order = np.argsort(-scores_arr)
        boxes_arr = boxes_arr[order]
        if boxes_arr.shape[0] > self.max_prompts_per_image:
            boxes_arr = boxes_arr[: self.max_prompts_per_image]

        if self.save_debug:
            self._save_debug_images(
                prefix=self._format_debug_prefix(image_meta),
                image_rgb=image,
                edges=edges,
                closed=closed,
                boxes_xyxy=boxes_arr,
            )

        return boxes_arr


class WatershedBoxPromptGenerator:
    """Watershed -> region contours -> boxes prompt generator."""

    name = "watershed_boxes"

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.logger = get_logger("distill")

        self.blur_kernel = _ensure_odd(int(cfg.get("blur_kernel", 7)), fallback=7)
        self.threshold_mode = str(cfg.get("threshold_mode", "otsu")).lower()
        self.binary_inv = bool(cfg.get("binary_inv", True))
        self.open_kernel = max(1, int(cfg.get("open_kernel", 3)))
        self.open_iter = max(0, int(cfg.get("open_iter", 1)))
        self.dilate_kernel = max(1, int(cfg.get("dilate_kernel", 3)))
        self.dilate_iter = max(1, int(cfg.get("dilate_iter", 2)))
        self.dist_thresh_ratio = float(cfg.get("dist_thresh_ratio", 0.4))
        self.min_box_w = max(1, int(cfg.get("min_box_w", 100)))
        self.min_box_h = max(1, int(cfg.get("min_box_h", 100)))
        self.force_square = bool(cfg.get("force_square", True))
        self.nms_thresh = float(cfg.get("nms_thresh", 0.7))
        self.max_prompts_per_image = max(1, int(cfg.get("max_prompts_per_image", 200)))
        self.save_debug = bool(cfg.get("save_debug", False))
        self.debug_dir = Path(str(cfg.get("debug_dir", "./debug/watershed_prompts")))
        if self.save_debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def _format_debug_prefix(self, image_meta: Optional[Dict[str, Any]]) -> str:
        if not image_meta:
            return "img"
        image_id = image_meta.get("image_id", "x")
        file_name = str(image_meta.get("file_name", "")).strip()
        stem = Path(file_name).stem if file_name else f"img_{image_id}"
        return f"{stem}_{image_id}"

    def _postprocess_boxes(self, boxes: List[List[float]], width: int, height: int) -> np.ndarray:
        scored_boxes: List[List[float]] = []
        scores: List[float] = []
        for b in boxes:
            x0, y0, x1, y1 = [int(v) for v in b]
            w = x1 - x0
            h = y1 - y0
            if w < self.min_box_w or h < self.min_box_h:
                continue
            if self.force_square:
                side = max(w, h)
                cx = x0 + w // 2
                cy = y0 + h // 2
                nx0 = int(round(cx - side / 2.0))
                ny0 = int(round(cy - side / 2.0))
                nx1 = nx0 + side
                ny1 = ny0 + side
                x0, y0, x1, y1 = _clip_box_xyxy(nx0, ny0, nx1, ny1, width=width, height=height)
            else:
                x0, y0, x1, y1 = _clip_box_xyxy(x0, y0, x1, y1, width=width, height=height)

            if x1 <= x0 or y1 <= y0:
                continue
            scored_boxes.append([float(x0), float(y0), float(x1), float(y1)])
            scores.append(float((x1 - x0) * (y1 - y0)))

        if len(scored_boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        boxes_arr = np.asarray(scored_boxes, dtype=np.float32)
        scores_arr = np.asarray(scores, dtype=np.float32)
        if self.nms_thresh > 0.0:
            boxes_arr = _nms_boxes_xyxy(boxes_arr, scores_arr, iou_thr=self.nms_thresh)
            scores_arr = np.asarray([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes_arr], dtype=np.float32)
        order = np.argsort(-scores_arr)
        boxes_arr = boxes_arr[order]
        if boxes_arr.shape[0] > self.max_prompts_per_image:
            boxes_arr = boxes_arr[: self.max_prompts_per_image]
        return boxes_arr

    def _save_debug_images(
        self,
        prefix: str,
        image_rgb: np.ndarray,
        binary: np.ndarray,
        markers_vis: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> None:
        try:
            import cv2
        except Exception:
            self.logger.warning("OpenCV unavailable; skip watershed debug image save")
            return

        cv2.imwrite(str(self.debug_dir / f"{prefix}_binary.png"), binary)
        cv2.imwrite(str(self.debug_dir / f"{prefix}_markers.png"), markers_vis)

        overlay_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
        for box in boxes_xyxy:
            x0, y0, x1, y1 = [int(v) for v in box.tolist()]
            cv2.rectangle(overlay_bgr, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.imwrite(str(self.debug_dir / f"{prefix}_boxes.png"), overlay_bgr)

    def generate_boxes(self, image: np.ndarray, image_meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("OpenCV is required for prompt_generator.type=watershed_boxes") from exc

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("WatershedBoxPromptGenerator expects RGB image in HWC format")

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        if self.threshold_mode == "otsu":
            thresh_flag = cv2.THRESH_BINARY_INV if self.binary_inv else cv2.THRESH_BINARY
            _, binary = cv2.threshold(blurred, 0, 255, thresh_flag + cv2.THRESH_OTSU)
        else:
            thresh_flag = cv2.THRESH_BINARY_INV if self.binary_inv else cv2.THRESH_BINARY
            _, binary = cv2.threshold(blurred, 127, 255, thresh_flag)

        open_kernel = np.ones((self.open_kernel, self.open_kernel), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=self.open_iter)

        dilate_kernel = np.ones((self.dilate_kernel, self.dilate_kernel), np.uint8)
        sure_bg = cv2.dilate(opening, dilate_kernel, iterations=self.dilate_iter)

        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        dist_thr = max(1e-6, self.dist_thresh_ratio * float(dist.max()))
        _, sure_fg = cv2.threshold(dist, dist_thr, 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        num_labels, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), markers)

        raw_boxes: List[List[float]] = []
        unique_labels = np.unique(markers)
        for label in unique_labels:
            if label <= 1:
                continue
            region = (markers == label).astype(np.uint8)
            if region.sum() == 0:
                continue
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                raw_boxes.append([float(x), float(y), float(x + w), float(y + h)])

        boxes_arr = self._postprocess_boxes(raw_boxes, width=width, height=height)

        if self.save_debug:
            markers_vis = np.zeros((height, width, 3), dtype=np.uint8)
            markers_vis[markers == -1] = (0, 0, 255)  # watershed boundaries
            markers_vis[markers > 1] = (0, 255, 0)
            self._save_debug_images(
                prefix=self._format_debug_prefix(image_meta),
                image_rgb=image,
                binary=binary,
                markers_vis=markers_vis,
                boxes_xyxy=boxes_arr,
            )

        return boxes_arr


class LabCCBoxPromptGenerator:
    """Lab threshold + connected components -> boxes prompt generator."""

    name = "lab_cc_boxes"

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.logger = get_logger("distill")

        # From debug/1.py logic:
        # mask = (((b > b_thresh) | (a > a_thresh)) & (L > l_thresh_min))
        self.l_thresh_min = int(cfg.get("l_thresh_min", 18))
        self.a_thresh = int(cfg.get("a_thresh", 131))
        self.b_thresh = int(cfg.get("b_thresh", 133))
        white_cfg = cfg.get("white_object_branch", {})
        self.white_branch_enabled = bool(white_cfg.get("enabled", False))
        self.white_l_min = int(white_cfg.get("l_min", 200))
        self.white_l_max = int(white_cfg.get("l_max", 255))
        self.white_chroma_max = float(white_cfg.get("chroma_max", 12.0))
        self.close_kernel = max(1, int(cfg.get("close_kernel", 5)))
        self.open_kernel = max(1, int(cfg.get("open_kernel", 3)))
        self.min_cc_area = max(1, int(cfg.get("min_cc_area", 1500)))

        # Keep consistency with canny_boxes
        self.min_box_w = max(1, int(cfg.get("min_box_w", 100)))
        self.min_box_h = max(1, int(cfg.get("min_box_h", 100)))
        self.force_square = bool(cfg.get("force_square", True))
        self.high_overlap_filter = bool(cfg.get("high_overlap_filter", False))
        self.high_overlap_thresh = float(cfg.get("high_overlap_thresh", 0.9))
        self.nms_thresh = float(cfg.get("nms_thresh", 0.7))
        self.max_prompts_per_image = max(1, int(cfg.get("max_prompts_per_image", 200)))
        self.save_debug = bool(cfg.get("save_debug", False))
        self.debug_dir = Path(str(cfg.get("debug_dir", "./debug/lab_cc_prompts")))
        if self.save_debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def _format_debug_prefix(self, image_meta: Optional[Dict[str, Any]]) -> str:
        if not image_meta:
            return "img"
        image_id = image_meta.get("image_id", "x")
        file_name = str(image_meta.get("file_name", "")).strip()
        stem = Path(file_name).stem if file_name else f"img_{image_id}"
        return f"{stem}_{image_id}"

    def _save_debug_images(
        self,
        prefix: str,
        image_rgb: np.ndarray,
        raw_mask: np.ndarray,
        clean_mask: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> None:
        try:
            import cv2
        except Exception:
            self.logger.warning("OpenCV unavailable; skip lab_cc debug image save")
            return
        cv2.imwrite(str(self.debug_dir / f"{prefix}_raw_mask.png"), raw_mask)
        cv2.imwrite(str(self.debug_dir / f"{prefix}_clean_mask.png"), clean_mask)

        overlay_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
        for box in boxes_xyxy:
            x0, y0, x1, y1 = [int(v) for v in box.tolist()]
            cv2.rectangle(overlay_bgr, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.imwrite(str(self.debug_dir / f"{prefix}_boxes.png"), overlay_bgr)

    def generate_boxes(self, image: np.ndarray, image_meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("OpenCV is required for prompt_generator.type=lab_cc_boxes") from exc

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("LabCCBoxPromptGenerator expects RGB image in HWC format")
        height, width = image.shape[:2]

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        colored_mask = ((b_ch > self.b_thresh) | (a_ch > self.a_thresh)) & (l_ch > self.l_thresh_min)

        if self.white_branch_enabled:
            a_centered = a_ch.astype(np.int16) - 128
            b_centered = b_ch.astype(np.int16) - 128
            chroma_sq = a_centered * a_centered + b_centered * b_centered
            white_mask = (
                (l_ch >= self.white_l_min)
                & (l_ch <= self.white_l_max)
                & (chroma_sq <= float(self.white_chroma_max * self.white_chroma_max))
            )
            raw_mask_bool = colored_mask | white_mask
        else:
            raw_mask_bool = colored_mask

        raw_mask = raw_mask_bool.astype(np.uint8) * 255

        close_kernel = np.ones((self.close_kernel, self.close_kernel), np.uint8)
        open_kernel = np.ones((self.open_kernel, self.open_kernel), np.uint8)
        clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, open_kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (clean_mask > 0).astype(np.uint8),
            connectivity=8,
        )
        final_mask = np.zeros_like(clean_mask)
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= self.min_cc_area:
                final_mask[labels == i] = 255

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[List[float]] = []
        scores: List[float] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < self.min_box_w or h < self.min_box_h:
                continue

            if self.force_square:
                side = int(max(w, h))
                cx = x + w // 2
                cy = y + h // 2
                x0 = int(round(cx - side / 2.0))
                y0 = int(round(cy - side / 2.0))
                x1 = x0 + side
                y1 = y0 + side
            else:
                x0, y0, x1, y1 = x, y, x + w, y + h

            x0, y0, x1, y1 = _clip_box_xyxy(x0, y0, x1, y1, width=width, height=height)
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append([float(x0), float(y0), float(x1), float(y1)])
            scores.append(float((x1 - x0) * (y1 - y0)))

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        boxes_arr = np.asarray(boxes, dtype=np.float32)
        scores_arr = np.asarray(scores, dtype=np.float32)

        if self.high_overlap_filter and self.high_overlap_thresh > 0.0:
            boxes_arr = _suppress_high_overlap_keep_large(boxes_arr, overlap_thr=self.high_overlap_thresh)
            scores_arr = np.asarray([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes_arr], dtype=np.float32)

        if self.nms_thresh > 0.0:
            boxes_arr = _nms_boxes_xyxy(boxes_arr, scores_arr, iou_thr=self.nms_thresh)
            scores_arr = np.asarray([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes_arr], dtype=np.float32)

        order = np.argsort(-scores_arr)
        boxes_arr = boxes_arr[order]
        if boxes_arr.shape[0] > self.max_prompts_per_image:
            boxes_arr = boxes_arr[: self.max_prompts_per_image]

        if self.save_debug:
            self._save_debug_images(
                prefix=self._format_debug_prefix(image_meta),
                image_rgb=image,
                raw_mask=raw_mask,
                clean_mask=final_mask,
                boxes_xyxy=boxes_arr,
            )

        return boxes_arr


def build_prompt_generator(cfg: Optional[Dict[str, Any]] = None) -> PromptGenerator:
    cfg = cfg or {}
    prompt_type = str(cfg.get("type", "sam2_auto")).lower()
    if prompt_type == "sam2_auto":
        generator: PromptGenerator = SAM2AutoPromptGenerator(cfg.get("sam2_auto", {}))
    elif prompt_type == "canny_boxes":
        generator = CannyBoxPromptGenerator(cfg.get("canny_boxes", {}))
    elif prompt_type == "watershed_boxes":
        generator = WatershedBoxPromptGenerator(cfg.get("watershed_boxes", {}))
    elif prompt_type == "lab_cc_boxes":
        generator = LabCCBoxPromptGenerator(cfg.get("lab_cc_boxes", {}))
    else:
        raise ValueError(f"Unknown prompt_generator.type: {prompt_type}")
    return BoxPromptRefiner(generator, cfg)
