"""DINOv3 + class-prototype similarity classifier."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

from distill.teacher.dinov3_knn_classifier import load_model


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return (x / denom).astype(np.float32, copy=False)


def _safe_float(value: Any, row_idx: int, col_name: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Non-numeric feature at row={row_idx}, col={col_name}: {value}") from exc


class DinoV3PrototypeClassifier:
    """Classifier that predicts by cosine similarity to class prototypes."""

    def __init__(self, cfg: Dict[str, Any], logger: Any) -> None:
        self.cfg = cfg
        self.logger = logger

        self.repo_dir = str(cfg.get("repo_dir", "")).strip()
        self.weights = str(cfg.get("weights", "")).strip()
        self.model_name = str(cfg.get("model_name", "dinov3_vitb16")).strip()
        self.model_source = str(cfg.get("model_source", "local")).strip() or "local"
        self.device = str(cfg.get("device", "cuda")).strip()
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.input_size = int(cfg.get("input_size", 224))
        self.batch_size = int(cfg.get("batch_size", 32))
        self.feature_dim = int(cfg.get("feature_dim", 768))
        self.prototype_csv = str(cfg.get("prototype_csv", "")).strip()
        self.label_col = str(cfg.get("label_col", "label"))
        self.feature_prefix = str(cfg.get("feature_prefix", "feat"))
        self.score_mode = str(cfg.get("score_mode", "softmax")).strip().lower()
        self.softmax_temperature = float(cfg.get("softmax_temperature", 1.0))
        self.mean = tuple(cfg.get("normalize_mean", cfg.get("mean", (0.485, 0.456, 0.406))))
        self.std = tuple(cfg.get("normalize_std", cfg.get("std", (0.229, 0.224, 0.225))))

        if self.score_mode not in {"softmax", "cosine"}:
            raise ValueError(f"Unsupported score_mode={self.score_mode}. Use softmax or cosine.")
        if self.softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be > 0")
        if not self.repo_dir:
            raise ValueError("classifier.dinov3_prototype.repo_dir is required")
        if not self.weights:
            raise ValueError("classifier.dinov3_prototype.weights is required")
        if not self.prototype_csv:
            raise ValueError("classifier.dinov3_prototype.prototype_csv is required")

        self.model = load_model(
            repo_dir=self.repo_dir,
            weights=self.weights,
            model_name=self.model_name,
            device=self.device,
            source=self.model_source,
        )
        proto = self._load_prototype_csv(Path(self.prototype_csv))
        self.prototype_labels = proto["labels"]
        self.prototype_matrix = proto["features"]
        self.label_to_index = {label: idx for idx, label in enumerate(self.prototype_labels)}

        self.logger.info(
            (
                "Classifier(dinov3_prototype) model=%s repo_dir=%s weights=%s "
                "prototype_csv=%s num_classes=%d feature_dim=%d score_mode=%s"
            ),
            self.model_name,
            self.repo_dir,
            self.weights,
            self.prototype_csv,
            len(self.prototype_labels),
            self.prototype_matrix.shape[1],
            self.score_mode,
        )

    def _resolve_feature_columns(self, fieldnames: Sequence[str]) -> List[str]:
        parsed: List[Tuple[int, str]] = []
        pattern = re.compile(rf"^{re.escape(self.feature_prefix)}_?(\d+)$")
        for col in fieldnames:
            match = pattern.match(col)
            if match is None:
                continue
            parsed.append((int(match.group(1)), col))
        if not parsed:
            sample_cols = ", ".join(list(fieldnames)[:12])
            raise ValueError(
                f"No feature columns found with prefix='{self.feature_prefix}'. "
                f"Expected '{self.feature_prefix}0' or '{self.feature_prefix}_0'. "
                f"CSV head columns: [{sample_cols}]"
            )
        parsed.sort(key=lambda x: x[0])
        cols = [col for _, col in parsed]
        if len(cols) != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} feature cols, got {len(cols)}")
        return cols

    def _load_prototype_csv(self, csv_path: Path) -> Dict[str, Any]:
        if not csv_path.exists():
            raise FileNotFoundError(f"prototype_csv not found: {csv_path}")

        labels: List[int] = []
        feature_rows: List[List[float]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if self.label_col not in fieldnames:
                raise ValueError(f"CSV missing label_col='{self.label_col}'")
            feature_cols = self._resolve_feature_columns(fieldnames)
            for idx, row in enumerate(reader):
                label_raw = row.get(self.label_col, None)
                if label_raw is None or str(label_raw).strip() == "":
                    raise ValueError(f"CSV has empty label at row={idx}")
                try:
                    label = int(label_raw)
                except Exception as exc:
                    raise ValueError(f"CSV label at row={idx} is not int-like: {label_raw}") from exc
                labels.append(label)

                feat_row: List[float] = []
                for col in feature_cols:
                    value = row.get(col, None)
                    if value is None or str(value).strip() == "":
                        raise ValueError(f"CSV has empty feature at row={idx}, col={col}")
                    feat_row.append(_safe_float(value, idx, col))
                feature_rows.append(feat_row)

        if not feature_rows:
            raise ValueError(f"CSV has no rows: {csv_path}")

        features = np.asarray(feature_rows, dtype=np.float32)
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"feature dim mismatch: csv={features.shape[1]} vs config={self.feature_dim}"
            )
        return {"labels": labels, "features": _l2_normalize(features)}

    def _preprocess_patches(self, patches: List[np.ndarray]) -> torch.Tensor:
        if Image is None:
            raise RuntimeError("PIL is required for DINOv3 patch preprocessing")
        mean_t = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        std_t = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)
        tensors = []
        for patch in patches:
            img = Image.fromarray(patch).resize((self.input_size, self.input_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)
            t = (t - mean_t) / std_t
            tensors.append(t)
        return torch.stack(tensors, dim=0)

    def _extract_features(self, patches: List[np.ndarray]) -> np.ndarray:
        if len(patches) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        feats: List[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                x = self._preprocess_patches(patches[i : i + self.batch_size]).to(self.device)
                f = self.model.forward_features(x)
                if isinstance(f, dict):
                    if "x_norm_clstoken" in f:
                        f = f["x_norm_clstoken"]
                    elif "x_norm_cls_token" in f:
                        f = f["x_norm_cls_token"]
                    else:
                        raise KeyError("forward_features output dict missing x_norm_clstoken")
                if not isinstance(f, torch.Tensor):
                    raise TypeError("forward_features must return Tensor or dict of Tensor")
                if f.ndim == 3:
                    f = f[:, 0, :]
                feats.append(f.detach().cpu().to(torch.float32).numpy())
        out = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
        if out.shape[1] != self.feature_dim:
            raise ValueError(f"DINO feature dim mismatch: model={out.shape[1]} config={self.feature_dim}")
        return _l2_normalize(out)

    def predict_patches(self, patches: List[np.ndarray]) -> List[Dict[str, Any]]:
        feats = self._extract_features(patches)
        if feats.shape[0] == 0:
            return []

        similarities = feats @ self.prototype_matrix.T
        pred_indices = np.argmax(similarities, axis=1)
        max_sims = similarities[np.arange(similarities.shape[0]), pred_indices]

        if self.score_mode == "cosine":
            scores = max_sims.astype(np.float32, copy=False)
        else:
            scaled = similarities / self.softmax_temperature
            scaled = scaled - np.max(scaled, axis=1, keepdims=True)
            probs = np.exp(scaled)
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            scores = probs[np.arange(probs.shape[0]), pred_indices].astype(np.float32, copy=False)

        outputs: List[Dict[str, Any]] = []
        for row_idx, proto_idx in enumerate(pred_indices.tolist()):
            outputs.append(
                {
                    "category_id": int(self.prototype_labels[int(proto_idx)]),
                    "category_score": float(scores[row_idx]),
                    "prototype_similarity": float(max_sims[row_idx]),
                    "prototype_index": int(proto_idx),
                }
            )
        return outputs
