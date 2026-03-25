"""Unified DINOv3 classifier with shared feature extraction."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from PIL import Image
from sklearn.neighbors import NearestNeighbors

from sam2real.utils.paths import resolve_project_path

def load_model(
    repo_dir: str,
    weights: str,
    model_name: str,
    device: str,
    source: str = "local",
) -> torch.nn.Module:
    model = torch.hub.load(
        repo_dir,
        model_name,
        source=source,
        weights=weights,
    )
    return model.to(device).eval()


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


class DinoV3FeatureExtractor:
    """Shared DINOv3 feature extractor for patch-based classifiers."""

    def __init__(self, cfg: Dict[str, Any], logger: Any) -> None:
        self.logger = logger
        self.model_name = str(cfg.get("model_name", "dinov3_vitb16")).strip()
        self.model_source = str(cfg.get("model_source", "local")).strip() or "local"
        repo_dir = str(cfg.get("repo_dir", "")).strip()
        self.repo_dir = resolve_project_path(repo_dir) if self.model_source == "local" else repo_dir
        self.weights = resolve_project_path(cfg.get("weights", ""))
        self.device = str(cfg.get("device", "cuda")).strip()
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.input_size = int(cfg.get("input_size", 224))
        self.batch_size = int(cfg.get("batch_size", 32))
        self.feature_dim = int(cfg.get("feature_dim", 768))
        self.mean = tuple(cfg.get("normalize_mean", cfg.get("mean", (0.485, 0.456, 0.406))))
        self.std = tuple(cfg.get("normalize_std", cfg.get("std", (0.229, 0.224, 0.225))))

        if not self.repo_dir:
            raise ValueError("DINOv3 repo_dir is required")
        if not self.weights:
            raise ValueError("DINOv3 weights are required")

        self.model = load_model(
            repo_dir=self.repo_dir,
            weights=self.weights,
            model_name=self.model_name,
            device=self.device,
            source=self.model_source,
        )

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
            tensors.append((t - mean_t) / std_t)
        return torch.stack(tensors, dim=0)

    def extract_features(self, patches: List[np.ndarray], normalize: bool = True) -> np.ndarray:
        if len(patches) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        feats: List[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                batch_np = patches[i : i + self.batch_size]
                x = self._preprocess_patches(batch_np).to(self.device)
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
        return _l2_normalize(out) if normalize else out


class PrototypeHead:
    """Prototype-similarity head on top of extracted DINOv3 features."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.feature_dim = int(cfg.get("feature_dim", 768))
        self.prototype_path = resolve_project_path(cfg.get("prototype_pth", cfg.get("prototype_csv", "")))
        self.score_mode = str(cfg.get("score_mode", "softmax")).strip().lower()
        self.softmax_temperature = float(cfg.get("softmax_temperature", 1.0))

        if self.score_mode not in {"softmax", "cosine"}:
            raise ValueError(f"Unsupported score_mode={self.score_mode}. Use softmax or cosine.")
        if self.softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be > 0")
        if not self.prototype_path:
            raise ValueError("classifier.dinov3.prototype.prototype_pth is required")

        proto = self._load_prototype_db(Path(self.prototype_path))
        self.prototype_labels = proto["labels"]
        self.prototype_matrix = proto["features"]

    def _load_prototype_db(self, db_path: Path) -> Dict[str, Any]:
        if not db_path.exists():
            raise FileNotFoundError(f"prototype database not found: {db_path}")
        if db_path.suffix.lower() != ".pth":
            raise ValueError(f"Unsupported prototype database format: {db_path.suffix}")
        payload = torch.load(db_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError("Prototype database must be a dict")
        labels_raw = payload.get("labels")
        prototypes_raw = payload.get("prototypes")
        if labels_raw is None or prototypes_raw is None:
            raise KeyError("Prototype database must contain 'labels' and 'prototypes'")
        if isinstance(labels_raw, torch.Tensor):
            labels = labels_raw.detach().cpu().to(torch.int64).tolist()
        else:
            labels = [int(v) for v in labels_raw]
        if isinstance(prototypes_raw, torch.Tensor):
            features = prototypes_raw.detach().cpu().to(torch.float32).numpy()
        else:
            features = np.asarray(prototypes_raw, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"Prototype matrix must be 2D, got shape={features.shape}")
        if len(labels) != features.shape[0]:
            raise ValueError(f"labels/prototypes size mismatch: labels={len(labels)} rows={features.shape[0]}")
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"feature dim mismatch: db={features.shape[1]} vs config={self.feature_dim}")
        return {"labels": labels, "features": _l2_normalize(features)}

    def predict(self, feats: np.ndarray) -> List[Dict[str, Any]]:
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


class KNNHead:
    """k-NN head on top of extracted DINOv3 features."""

    def __init__(self, cfg: Dict[str, Any], logger: Any) -> None:
        self.logger = logger
        self.metric = str(cfg.get("metric", "cosine")).lower()
        self.weights_mode = str(cfg.get("weights_mode", "distance")).lower()
        self.database_csv = resolve_project_path(cfg.get("database_csv", ""))
        self.image_col = str(cfg.get("image_col", "image"))
        self.path_col = str(cfg.get("path_col", "path"))
        self.label_col = str(cfg.get("label_col", "label"))
        self.feature_prefix = str(cfg.get("feature_prefix", "feature"))
        self.feature_dim = int(cfg.get("feature_dim", 768))
        self.k = int(cfg.get("k", 5))

        if self.metric not in {"cosine", "euclidean"}:
            raise ValueError(f"Unsupported metric={self.metric}. Use cosine or euclidean.")
        if self.weights_mode not in {"distance", "uniform"}:
            raise ValueError(f"Unsupported weights_mode={self.weights_mode}. Use distance or uniform.")
        if self.k <= 0:
            raise ValueError("k must be >= 1")
        if not self.database_csv:
            raise ValueError("classifier.dinov3.knn.database_csv is required")

        db = self._load_database_csv(Path(self.database_csv))
        self.db_features = db["features"]
        self.db_labels = db["labels"]
        if self.metric == "cosine":
            self.db_features = _l2_normalize(self.db_features)

        self.k = min(self.k, len(self.db_labels))
        self.nn_index = None
        if NearestNeighbors is not None:
            self.nn_index = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
            self.nn_index.fit(self.db_features)
        else:
            self.logger.warning("scikit-learn not found; fallback to numpy k-NN")

    def _resolve_feature_columns(self, fieldnames: Sequence[str]) -> List[str]:
        parsed: List[Tuple[int, str]] = []
        pattern = re.compile(rf"^{re.escape(self.feature_prefix)}_?(\d+)$")
        for col in fieldnames:
            match = pattern.match(col)
            if match is not None:
                parsed.append((int(match.group(1)), col))
        if not parsed:
            sample_cols = ", ".join(list(fieldnames)[:12])
            raise ValueError(
                f"No feature columns found with prefix='{self.feature_prefix}'. CSV head columns: [{sample_cols}]"
            )
        parsed.sort(key=lambda x: x[0])
        cols = [col for _, col in parsed]
        if len(cols) != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} feature cols, got {len(cols)}")
        return cols

    def _load_database_csv(self, csv_path: Path) -> Dict[str, Any]:
        if not csv_path.exists():
            raise FileNotFoundError(f"database_csv not found: {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if self.label_col not in fieldnames:
                raise ValueError(f"CSV missing label_col='{self.label_col}'")
            feature_cols = self._resolve_feature_columns(fieldnames)
            labels: List[Any] = []
            feature_rows: List[List[float]] = []
            for idx, row in enumerate(reader):
                label_val = row.get(self.label_col, None)
                if label_val is None or str(label_val).strip() == "":
                    raise ValueError(f"CSV has empty label at row={idx}")
                labels.append(label_val)
                fvec = []
                for col in feature_cols:
                    value = row.get(col, None)
                    if value is None or str(value).strip() == "":
                        raise ValueError(f"CSV has empty feature at row={idx}, col={col}")
                    fvec.append(_safe_float(value, idx, col))
                feature_rows.append(fvec)
        features = np.asarray(feature_rows, dtype=np.float32)
        return {"features": features, "labels": labels}

    def _knn_query_numpy(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.metric == "cosine":
            sims = feats @ self.db_features.T
            topk_idx = np.argpartition(-sims, kth=self.k - 1, axis=1)[:, : self.k]
            topk_sims = np.take_along_axis(sims, topk_idx, axis=1)
            order = np.argsort(-topk_sims, axis=1)
            idx = np.take_along_axis(topk_idx, order, axis=1)
            distances = 1.0 - np.take_along_axis(topk_sims, order, axis=1)
            return distances.astype(np.float32), idx.astype(np.int64)
        d2 = ((feats[:, None, :] - self.db_features[None, :, :]) ** 2).sum(axis=2)
        topk_idx = np.argpartition(d2, kth=self.k - 1, axis=1)[:, : self.k]
        topk_d2 = np.take_along_axis(d2, topk_idx, axis=1)
        order = np.argsort(topk_d2, axis=1)
        idx = np.take_along_axis(topk_idx, order, axis=1)
        distances = np.sqrt(np.take_along_axis(topk_d2, order, axis=1))
        return distances.astype(np.float32), idx.astype(np.int64)

    def _knn_query(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.nn_index is not None:
            distances, indices = self.nn_index.kneighbors(feats, n_neighbors=self.k, return_distance=True)
            return distances.astype(np.float32), indices.astype(np.int64)
        return self._knn_query_numpy(feats)

    def predict(self, feats: np.ndarray) -> List[Dict[str, Any]]:
        if feats.shape[0] == 0:
            return []
        distances, indices = self._knn_query(feats)
        outputs: List[Dict[str, Any]] = []
        eps = 1e-8
        for row_d, row_i in zip(distances, indices):
            labels = [self.db_labels[int(j)] for j in row_i.tolist()]
            if self.metric == "cosine":
                sims = 1.0 - row_d
                weights = np.clip(sims, a_min=0.0, a_max=None) if self.weights_mode == "distance" else np.ones_like(sims)
            else:
                weights = 1.0 / (row_d + eps) if self.weights_mode == "distance" else np.ones_like(row_d)

            total = float(np.sum(weights)) + eps
            score_by_label: Dict[Any, float] = {}
            for lb, weight in zip(labels, weights.tolist()):
                score_by_label[lb] = score_by_label.get(lb, 0.0) + float(weight)

            pred_label = max(score_by_label.items(), key=lambda x: x[1])[0]
            pred_score = float(score_by_label[pred_label] / total)
            try:
                pred_label = int(pred_label)
            except Exception:
                pass
            outputs.append(
                {
                    "category_id": pred_label,
                    "category_score": pred_score,
                    "neighbors": [
                        {
                            "label": labels[i],
                            "distance": float(row_d[i]),
                            "weight": float(weights[i]),
                        }
                        for i in range(len(labels))
                    ],
                }
            )
        return outputs


class DinoV3Classifier:
    """Unified DINOv3 classifier with pluggable decision head."""

    def __init__(self, cfg: Dict[str, Any], logger: Any) -> None:
        self.cfg = cfg
        self.logger = logger
        self.mode = str(cfg.get("mode", "prototype")).strip().lower()
        if self.mode not in {"prototype", "knn"}:
            raise ValueError(f"Unsupported DINOv3 mode={self.mode}")

        self.extractor = DinoV3FeatureExtractor(cfg, logger=logger)
        if self.mode == "prototype":
            self.head = PrototypeHead(cfg)
        else:
            self.head = KNNHead(cfg, logger=logger)

        self.logger.info(
            "Classifier(dinov3) mode=%s model=%s repo_dir=%s weights=%s",
            self.mode,
            self.extractor.model_name,
            self.extractor.repo_dir,
            self.extractor.weights,
        )

    def predict_patches(self, patches: List[np.ndarray]) -> List[Dict[str, Any]]:
        feats = self.extractor.extract_features(patches, normalize=True)
        return self.head.predict(feats)
