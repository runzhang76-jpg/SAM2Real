from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO


class CocoSparseDataset:
    """
    标准 COCO 格式数据集读取器（稀疏验证用）。
    """

    def __init__(
        self,
        data_root: Path | str ,
        annotation_file: Optional[Path | str] = None,
        image_dir: str = "images",
        min_area: int = 1,
        include_crowd: bool = False,
    ) -> None:
        self.data_root = Path(data_root)  # base path
        self.image_dir = self.data_root / image_dir  # image path
        self.annotation_file = Path(annotation_file)
        if self.annotation_file == None:
            self.annotation_file = "data/sparse_coco_segment_cdw/annotation/instances.json"

        self.min_area = min_area
        self.include_crowd = include_crowd

        # 使用 COCO API 读标注
        self.coco = COCO(str(self.annotation_file))
        self.image_ids = sorted(self.coco.getImgIds())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict]:
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = self.image_dir / img_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(
            imgIds=[img_id],
            iscrowd=None if self.include_crowd else 0,
        )
        anns = self.coco.loadAnns(ann_ids)

        masks: List[np.ndarray] = []
        boxes: List[np.ndarray] = []
        categories: List[int] = []

        for ann in anns:
            if ann.get("area", 0) < self.min_area:
                continue
            mask = self._ann_to_mask(ann, img_info["height"], img_info["width"])
            boxes.append(self._xywh_to_xyxy(ann["bbox"]))
            masks.append(mask.astype(bool))
            categories.append(int(ann.get("category_id", -1)))

        target = {
            "image_id": img_id,
            "file_name": img_info["file_name"],
            "bboxes": np.stack(boxes).astype(np.float32)
            if boxes
            else np.zeros((0, 4), dtype=np.float32),
            "masks": np.stack(masks) if masks else np.zeros((0, image.height, image.width), dtype=bool),
            "categories": categories,
            "height": img_info["height"],
            "width": img_info["width"],
        }
        return image, target

    @staticmethod
    def _xywh_to_xyxy(bbox: List[float]) -> np.ndarray:
        x, y, w, h = bbox
        return np.array([x, y, x + w, y + h], dtype=np.float32)

    @staticmethod
    def _ann_to_mask(ann: Dict, height: int, width: int) -> np.ndarray:
        segmentation = ann.get("segmentation", [])
        if isinstance(segmentation, list):
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(segmentation, dict) and "counts" in segmentation:
            rle = segmentation
        else:
            raise ValueError(f"Unsupported segmentation format: {type(segmentation)}")
        return mask_utils.decode(rle).astype(bool)
