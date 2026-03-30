"""SAM2  segment_cdw """

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import time

from matmatch2real.core.structures import InstancePrediction, PseudoLabelInstance
from matmatch2real.teacher.classifier_adapter import ClassifierAdapter
from matmatch2real.teacher.postprocess import (
    IdentityPostProcess,
    PostProcessPipeline,
    RawMaskPrediction,
    convert_instances,
)
from matmatch2real.teacher.prompt_generators import build_prompt_generator
from matmatch2real.teacher.reliability import compute_reliability
from matmatch2real.utils.logging import get_logger
from matmatch2real.utils.paths import PROJECT_ROOT, resolve_project_path


def _ensure_sam2_repo_on_path() -> None:
    """Ensure the upstream SAM2 repo root is importable as `sam2`."""

    sam2_repo = PROJECT_ROOT / "external" / "sam2"
    if sam2_repo.exists():
        sam2_repo_str = str(sam2_repo)
        if sam2_repo_str not in sys.path:
            sys.path.insert(0, sam2_repo_str)


def _load_sam2_api() -> tuple[Any, Any, Any]:
    """Load upstream SAM2 APIs lazily after resolving the repo path."""

    _ensure_sam2_repo_on_path()
    amg_mod = importlib.import_module("sam2.automatic_mask_generator")
    build_mod = importlib.import_module("sam2.build_sam")
    predictor_mod = importlib.import_module("sam2.sam2_image_predictor")
    return (
        getattr(amg_mod, "SAM2AutomaticMaskGenerator"),
        getattr(build_mod, "build_sam2"),
        getattr(predictor_mod, "SAM2ImagePredictor"),
    )

import torch


def _tensor_to_numpy(image: "torch.Tensor") -> np.ndarray:
    """ (C,H,W)  (H,W,C)  torch  uint8 numpy """

    if torch is None:
        raise RuntimeError("PyTorch ")
    img = image.detach().cpu()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)
    arr = img.numpy()
    if arr.dtype.kind == "f":
        max_val = float(arr.max()) if arr.size else 1.0
        if max_val <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255)
    arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr


def _to_numpy_batch(images: Any) -> List[np.ndarray]:
    """ HWC  uint8 numpy """

    if torch is not None and isinstance(images, torch.Tensor):
        return [_tensor_to_numpy(images[i]) for i in range(images.shape[0])]
    if isinstance(images, Sequence):
        batch = []
        for img in images:
            if torch is not None and isinstance(img, torch.Tensor):
                batch.append(_tensor_to_numpy(img))
            else:
                arr = np.asarray(img)
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                batch.append(arr)
        return batch
    raise TypeError("")


def _resize_image(img: np.ndarray, size: Tuple[int, int], resample: str) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover -
        raise RuntimeError("PIL ") from exc

    resample_map = {
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
    }
    return np.array(Image.fromarray(img).resize(size, resample=resample_map[resample]))


class SegmentCDWAdapter:
    """



      - images: List[np.ndarray]HWC uint8
      - metas: List[Dict[str, Any]]


      - List[List[RawMaskPrediction]]
    """

    def __init__(self, module: str, callable_name: str, kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.module = module
        self.callable_name = callable_name
        self.kwargs = kwargs or {}
        self.logger = get_logger("distill")
        self._callable = self._resolve_callable()

    def _resolve_callable(self) -> Optional[Any]:
        try:
            module = importlib.import_module(self.module)
            return getattr(module, self.callable_name)
        except Exception as exc:
            self.logger.warning("segment_cdw  %s:%s (%s)", self.module, self.callable_name, exc)
            return None

    def run(self, images: List[np.ndarray], metas: List[Dict[str, Any]]) -> List[List[RawMaskPrediction]]:
        if self._callable is None:
            self.logger.warning("segment_cdw ")
            return [[] for _ in range(len(metas))]
        outputs = self._callable(images=images, metas=metas, **self.kwargs)
        return outputs


class SegmentCDWRunEvalAdapter:
    """ SAM2 """

    def __init__(self, cfg: Dict[str, Any], device: str, prompt_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg
        self.device = cfg.get('device')
        self.logger = get_logger("distill")
        self.sam_model = None
        self.mask_gen = None
        self.mask_predictor = None
        self.img_downsample = bool(cfg.get("img_downsample", True))
        self.img_downsample_factor = float(cfg.get("img_downsample_factor", 4.0))
        self.prompt_cfg = prompt_cfg or {"type": "sam2_auto"}
        self.prompt_generator = build_prompt_generator(self.prompt_cfg)
        self.prompt_type = str(self.prompt_cfg.get("type", "sam2_auto")).lower()
        self._build_generator()

    def _build_generator(self) -> None:
        config_file = resolve_project_path(self.cfg.get("config_file"))
        ckpt_path = resolve_project_path(self.cfg.get("ckpt_path"))
        if not config_file or not ckpt_path:
            raise ValueError("SAM2  config_file  checkpoint")

        sam2_auto_cls, build_sam2_fn, sam2_predictor_cls = _load_sam2_api()

        self.sam_model = build_sam2_fn(
            config_file=config_file,
            ckpt_path=ckpt_path,
            device=self.device,
        )

        if self.prompt_type == "sam2_auto":
            self.mask_gen = sam2_auto_cls(
                self.sam_model,
                points_per_side=int(self.cfg.get("points_per_side", 24)),
                points_per_batch=int(self.cfg.get("points_per_batch", 16)),
                pred_iou_thresh=float(self.cfg.get("pred_iou_thresh", 0.6)),
                stability_score_thresh=float(self.cfg.get("stability_thresh", 0.7)),
                stability_score_offset=float(self.cfg.get("stability_score_offset", 1.0)),
                mask_threshold=float(self.cfg.get("mask_threshold", 0.0)),
                box_nms_thresh=float(self.cfg.get("box_nms_thresh", 0.7)),
                crop_n_layers=int(self.cfg.get("crop_n_layers", 0)),
                crop_nms_thresh=float(self.cfg.get("crop_nms_thresh", 0.7)),
                crop_overlap_ratio=float(self.cfg.get("crop_overlap_ratio", 0.34)),
                crop_n_points_downscale_factor=int(self.cfg.get("crop_n_points_downscale_factor", 1)),
                min_mask_region_area=int(self.cfg.get("min_mask_region_area", 0)),
                output_mode=str(self.cfg.get("output_mode", "binary_mask")),
                multimask_output=bool(self.cfg.get("multimask_output", True)),
            )
            self.logger.info("SAM2 built with prompt_generator=sam2_auto: %s", config_file)
        elif self.prompt_type in {"canny_boxes", "watershed_boxes", "lab_cc_boxes"}:
            self.mask_predictor = sam2_predictor_cls(
                self.sam_model,
                mask_threshold=float(self.cfg.get("mask_threshold", 0.0)),
            )
            if self.img_downsample and self.img_downsample_factor > 1.0:
                self.logger.warning(
                    "prompt_generator=canny_boxes currently runs on original image scale; "
                    "sam2.img_downsample is ignored in this mode."
                )
            if hasattr(self.prompt_generator, "max_prompts_per_image"):
                self.logger.info(
                    "SAM2 built with prompt_generator=%s (max_prompts_per_image=%d)",
                    self.prompt_type,
                    int(getattr(self.prompt_generator, "max_prompts_per_image", -1)),
                )
            else:
                self.logger.info("SAM2 built with prompt_generator=%s", self.prompt_type)
        else:
            raise ValueError(f"Unsupported prompt_generator.type: {self.prompt_type}")

    @staticmethod
    def _xyxy_to_xywh(box_xyxy: np.ndarray) -> List[float]:
        x0, y0, x1, y1 = [float(v) for v in box_xyxy.tolist()]
        return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]

    def _predict_from_boxes(self, img_np: np.ndarray, meta: Dict[str, Any]) -> List[RawMaskPrediction]:
        if self.mask_predictor is None:
            raise RuntimeError("SAM2 image predictor is not initialized")
        boxes_xyxy = self.prompt_generator.generate_boxes(img_np, image_meta=meta)
        if boxes_xyxy.shape[0] == 0:
            return []

        self.mask_predictor.set_image(img_np)
        multimask_output = bool(self.cfg.get("box_prompt_multimask_output", False))
        masks, ious, _ = self.mask_predictor.predict(
            box=boxes_xyxy,
            multimask_output=multimask_output,
            normalize_coords=True,
        )

        pred_items: List[RawMaskPrediction] = []
        masks_arr = np.asarray(masks)
        ious_arr = np.asarray(ious)

        if masks_arr.ndim == 4:
            # BxCxHxW, keep highest IoU candidate per prompt box.
            for bi in range(masks_arr.shape[0]):
                iou_vec = ious_arr[bi] if ious_arr.ndim == 2 else ious_arr
                best_ci = int(np.argmax(iou_vec))
                mask = masks_arr[bi, best_ci]
                score = float(iou_vec[best_ci])
                pred_items.append(
                    {
                        "segmentation": mask.astype(bool),
                        "predicted_iou": score,
                        "stability_score": 1.0,
                        "bbox": self._xyxy_to_xywh(boxes_xyxy[bi]),
                        "prompt_box": boxes_xyxy[bi].tolist(),
                        "area": int(np.asarray(mask, dtype=np.uint8).sum()),
                    }
                )
        elif masks_arr.ndim == 3:
            # CxHxW for single prompt; keep highest IoU candidate.
            iou_vec = ious_arr.reshape(-1)
            best_ci = int(np.argmax(iou_vec))
            mask = masks_arr[best_ci]
            score = float(iou_vec[best_ci])
            pred_items.append(
                {
                    "segmentation": mask.astype(bool),
                    "predicted_iou": score,
                    "stability_score": 1.0,
                    "bbox": self._xyxy_to_xywh(boxes_xyxy[0]),
                    "prompt_box": boxes_xyxy[0].tolist(),
                    "area": int(np.asarray(mask, dtype=np.uint8).sum()),
                }
            )
        else:
            self.logger.warning("Unexpected mask shape from SAM2 predictor: %s", str(masks_arr.shape))
            return []

        return pred_items

    def generate(self, images: List[np.ndarray], metas: List[Dict[str, Any]]) -> List[List[RawMaskPrediction]]:
        if self.prompt_type == "sam2_auto" and self.mask_gen is None:
            raise RuntimeError("SAM2 mask generator is not initialized")
        if self.prompt_type in {"canny_boxes", "watershed_boxes", "lab_cc_boxes"} and self.mask_predictor is None:
            raise RuntimeError("SAM2 image predictor is not initialized")

        outputs: List[List[RawMaskPrediction]] = []
        for img_np, meta in zip(images, metas):
            orig_h, orig_w = img_np.shape[:2]
            if self.prompt_type == "sam2_auto":
                sam_img = img_np
                if self.img_downsample and self.img_downsample_factor > 1.0:
                    new_w = max(1, int(orig_w / self.img_downsample_factor))
                    new_h = max(1, int(orig_h / self.img_downsample_factor))
                    sam_img = _resize_image(img_np, (new_w, new_h), resample="bilinear")

                pred_items = self.mask_gen.generate(sam_img)

                if self.img_downsample and self.img_downsample_factor > 1.0:
                    resized_items: List[RawMaskPrediction] = []
                    for item in pred_items:
                        seg = item.get("segmentation", item.get("mask", None))
                        if seg is None:
                            resized_items.append(item)
                            continue
                        seg_img = _resize_image(
                            (np.asarray(seg, dtype=np.uint8) * 255),
                            (orig_w, orig_h),
                            resample="nearest",
                        )
                        new_item = dict(item)
                        new_item["segmentation"] = np.array(seg_img, copy=False) > 127
                        resized_items.append(new_item)
                    pred_items = resized_items
            else:
                pred_items = self._predict_from_boxes(img_np, meta)

            outputs.append(pred_items)

        return outputs


class SAM2Teacher:
    """SAM2 """

    def __init__(self, cfg: Dict[str, Any], device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        self.logger = get_logger("distill")
        self.adapter: Optional[Any] = None
        self.classifier: Optional[ClassifierAdapter] = None
        self.post_cfg = cfg.get("sam2", {}).get("postprocess", {})
        seg_cfg = cfg.get("segment_cdw", {})
        sam_cfg = cfg.get("sam2", {})
        prompt_cfg = cfg.get("prompt_generator", {"type": "sam2_auto"})

        # SAM2
        if seg_cfg.get("enabled", False):
            module = str(seg_cfg.get("module", ""))
            callable_name = str(seg_cfg.get("callable", ""))
            if module and callable_name:
                adapter = SegmentCDWAdapter(module=module, callable_name=callable_name, kwargs=seg_cfg.get("kwargs", {}))
                if adapter._callable is None:
                    self.logger.warning("segment_cdw callable  run_eval ")
                    self.adapter = SegmentCDWRunEvalAdapter(cfg=sam_cfg, device=self.device, prompt_cfg=prompt_cfg)
                else:
                    self.adapter = adapter
            else:
                self.adapter = SegmentCDWRunEvalAdapter(cfg=sam_cfg, device=self.device, prompt_cfg=prompt_cfg)
        elif sam_cfg:
            self.adapter = SegmentCDWRunEvalAdapter(cfg=sam_cfg, device=self.device, prompt_cfg=prompt_cfg)

        #
        pp_cfg = cfg.get("postprocess", {})
        if pp_cfg.get("enabled", False):
            self.postprocess = PostProcessPipeline(pp_cfg)
        else:
            self.postprocess = IdentityPostProcess()

        #
        cls_cfg = cfg.get("classifier", {})
        if cls_cfg.get("enabled", False):
            merged_cls_cfg = dict(cls_cfg)
            crop_cfg = pp_cfg.get("instance_crop", {})
            if crop_cfg and "crop" not in merged_cls_cfg:
                merged_cls_cfg["crop"] = crop_cfg
            self.classifier = ClassifierAdapter(merged_cls_cfg)

    def generate(
        self,
        images: Any,
        metas: List[Dict[str, Any]],
        image_ids: Optional[List[int]] = None,
    ) -> List[List[PseudoLabelInstance]]:
        """


        : List[Dict] SAM2AutomaticMaskGenerator
        """

        if self.adapter is None:
            self.logger.info("SAM2Teacher ")
            return [[] for _ in range(len(metas))]

        images_np = _to_numpy_batch(images)

        #
        raw_preds = self.adapter.run(images_np, metas) if isinstance(self.adapter, SegmentCDWAdapter) else self.adapter.generate(images_np, metas)

        pseudo_labels: List[List[PseudoLabelInstance]] = []
        for idx, raw in enumerate(raw_preds):
            meta = metas[idx] if idx < len(metas) else {}
            image_id = int(meta.get("image_id", image_ids[idx] if image_ids and idx < len(image_ids) else idx))
            height = int(meta.get("height", images_np[idx].shape[0]))
            width = int(meta.get("width", images_np[idx].shape[1]))

            #
            if raw and isinstance(raw[0], InstancePrediction):
                processed = list(raw)
            else:
                raw = self.postprocess(list(raw), meta)
                cfg_for_convert = self.post_cfg
                processed = convert_instances(
                    raw,
                    image_hw=(height, width),
                    image_id=image_id,
                    class_id=0,
                    cfg=cfg_for_convert,
                    encode_rle=bool(self.post_cfg.get("encode_rle", False)),
                )

            #
            if self.classifier is not None:
                processed = self.classifier.classify(processed, image_np=images_np[idx])

            #
            for inst in processed:
                inst.reliability = compute_reliability(inst)

            pseudo_labels.append(
                [
                    PseudoLabelInstance(
                        image_id=inst.image_id,
                        bbox=inst.bbox,
                        class_id=inst.class_id,
                        score=inst.score,
                        reliability=inst.reliability,
                        mask=inst.mask,
                        rle=inst.rle,
                        meta=inst.meta,
                    )
                    for inst in processed
                ]
            )

        return pseudo_labels


def build_teacher(cfg: Dict[str, Any], device: str = "cpu") -> Optional[SAM2Teacher]:
    """"""

    if cfg.get("mode", "offline") == "offline":
        return None
    return SAM2Teacher(cfg, device=device)
