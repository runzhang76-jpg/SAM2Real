"""Standalone image-patch classifier backed by a CNN model."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from sam2real.teacher.classier_network.cnns import seresnext50_32x4d
from sam2real.utils.paths import resolve_project_path

from PIL import Image



class ClassifierPatchTransform:
    def __init__(
        self,
        patch_size: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ) -> None:
        self.patch_size = patch_size
        self.norm = transforms.Normalize(mean=mean, std=std)

    def __call__(self, patch: np.ndarray) -> torch.Tensor:
        if Image is None:
            raise RuntimeError("PIL is required for patch resize")
        patch_img = Image.fromarray(patch).resize((self.patch_size, self.patch_size), resample=Image.BILINEAR)
        tensor = torch.from_numpy(np.array(patch_img)).permute(2, 0, 1).float() / 255.0
        return self.norm(tensor)


class ModelClassifier:
    """Patch classifier using a CNN checkpoint."""

    def __init__(self, cfg: Dict[str, Any], logger: Any) -> None:
        self.cfg = cfg
        self.logger = logger
        self.num_classes = int(cfg.get("num_classes", 1))
        self.device = str(cfg.get("device", "auto"))
        self.patch_size = int(cfg.get("input_size", cfg.get("patch_size", 224)))
        self.batch_size = int(cfg.get("batch_size", 32))
        self.normalize_mean = tuple(cfg.get("normalize_mean", cfg.get("mean", (0.485, 0.456, 0.406))))
        self.normalize_std = tuple(cfg.get("normalize_std", cfg.get("std", (0.229, 0.224, 0.225))))
        self.patch_transform = ClassifierPatchTransform(
            patch_size=self.patch_size,
            mean=self.normalize_mean,  # type: ignore[arg-type]
            std=self.normalize_std,  # type: ignore[arg-type]
        )
        self.model = None
        self._build_model()

    def _build_model(self) -> None:
        ckpt_path = resolve_project_path(self.cfg.get("checkpoint", ""))
        if not ckpt_path:
            self.logger.warning("classifier.model.checkpoint is empty; model backend disabled")
            return
        model = seresnext50_32x4d(pretrained=False, out_features=self.num_classes)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.logger.info("Classifier(model) loaded: %s (device=%s)", ckpt_path, self.device)

    def predict_patches(self, patches: List[np.ndarray]) -> List[Dict[str, Any]]:
        if self.model is None or len(patches) == 0:
            return []
        patch_tensors = [self.patch_transform(patch) for patch in patches]
        outputs: List[Dict[str, Any]] = []
        with torch.no_grad():
            for i in range(0, len(patch_tensors), self.batch_size):
                batch = torch.stack(patch_tensors[i : i + self.batch_size]).to(self.device)
                logits = self.model(batch)
                probs = F.softmax(logits, dim=1)
                conf, cls = torch.max(probs, dim=1)
                outputs.extend(
                    {
                        "category_id": int(class_id) + 1,
                        "category_score": float(score),
                    }
                    for class_id, score in zip(
                        cls.detach().cpu().tolist(),
                        conf.detach().cpu().tolist(),
                    )
                )
        return outputs
