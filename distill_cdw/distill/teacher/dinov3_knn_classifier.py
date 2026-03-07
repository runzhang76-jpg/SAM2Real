"""DINOv3 + k-NN classifier using pre-extracted CSV feature database."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except Exception:  # pragma: no cover
    NearestNeighbors = None  # type: ignore


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
    model = model.to(device).eval()
    return model


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _safe_float(value: Any, row_idx: int, col_name: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Non-numeric feature at row={row_idx}, col={col_name}: {value}") from exc


class DINOv3KNNClassifier:
    """Classifier that extracts DINOv3 global feature and predicts by k-NN."""

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
        self.k = int(cfg.get("k", 5))
        self.metric = str(cfg.get("metric", "cosine")).lower()
        self.weights_mode = str(cfg.get("weights_mode", "distance")).lower()
        self.database_csv = str(cfg.get("database_csv", "")).strip()
        self.image_col = str(cfg.get("image_col", "image"))
        self.path_col = str(cfg.get("path_col", "path"))
        self.label_col = str(cfg.get("label_col", "label"))
        self.feature_prefix = str(cfg.get("feature_prefix", "feature"))
        self.feature_dim = int(cfg.get("feature_dim", 768))
        self.mean = tuple(cfg.get("normalize_mean", cfg.get("mean", (0.485, 0.456, 0.406))))
        self.std = tuple(cfg.get("normalize_std", cfg.get("std", (0.229, 0.224, 0.225))))

        if self.metric not in {"cosine", "euclidean"}:
            raise ValueError(f"Unsupported metric={self.metric}. Use cosine or euclidean.")
        if self.weights_mode not in {"distance", "uniform"}:
            raise ValueError(f"Unsupported weights_mode={self.weights_mode}. Use distance or uniform.")
        if self.k <= 0:
            raise ValueError("k must be >= 1")
        if not self.repo_dir:
            raise ValueError("classifier.dinov3_knn.repo_dir is required")
        if not self.weights:
            raise ValueError("classifier.dinov3_knn.weights is required")
        if not self.database_csv:
            raise ValueError("classifier.dinov3_knn.database_csv is required")

        self.model = load_model(
            self.repo_dir,
            self.weights,
            self.model_name,
            self.device,
            source=self.model_source,
        )

        db = self._load_database_csv(Path(self.database_csv))
        self.db_features = db["features"]
        self.db_labels = db["labels"]
        self.db_meta = db["meta"]

        if self.metric == "cosine":
            self.db_features = _l2_normalize(self.db_features)

        self.k = min(self.k, len(self.db_labels))
        if self.k < int(cfg.get("k", 5)):
            self.logger.warning("k is larger than DB size; adjusted to %d", self.k)

        self.nn_index = None
        if NearestNeighbors is not None:
            self.nn_index = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
            self.nn_index.fit(self.db_features)
        else:
            self.logger.warning("scikit-learn not found; fallback to numpy k-NN")

        self.logger.info(
            (
                "Classifier(dinov3_knn) model=%s repo_dir=%s weights=%s db_csv=%s "
                "db_samples=%d feature_dim=%d k=%d metric=%s weights_mode=%s"
            ),
            self.model_name,
            self.repo_dir,
            self.weights,
            self.database_csv,
            len(self.db_labels),
            self.db_features.shape[1],
            self.k,
            self.metric,
            self.weights_mode,
        )

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
            metas: List[Dict[str, Any]] = []
            feature_rows: List[List[float]] = []

            for idx, row in enumerate(reader):
                label_val = row.get(self.label_col, None)
                if label_val is None or str(label_val).strip() == "":
                    raise ValueError(f"CSV has empty label at row={idx}")
                labels.append(label_val)

                metas.append(
                    {
                        "image": row.get(self.image_col, ""),
                        "path": row.get(self.path_col, ""),
                    }
                )

                fvec = []
                for col in feature_cols:
                    value = row.get(col, None)
                    if value is None or str(value).strip() == "":
                        raise ValueError(f"CSV has empty feature at row={idx}, col={col}")
                    fvec.append(_safe_float(value, idx, col))
                feature_rows.append(fvec)

        if not feature_rows:
            raise ValueError(f"CSV has no rows: {csv_path}")

        features = np.asarray(feature_rows, dtype=np.float32)
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"feature dim mismatch: csv={features.shape[1]} vs config={self.feature_dim}"
            )

        self.logger.info("Loaded feature DB from CSV: %s (rows=%d)", csv_path, features.shape[0])
        return {"features": features, "labels": labels, "meta": metas}

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
        idxs = [i for i, _ in parsed]
        if idxs != list(range(idxs[0], idxs[0] + len(idxs))):
            raise ValueError("Feature columns are not contiguous by index")
        cols = [col for _, col in parsed]
        if len(cols) != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} feature cols, got {len(cols)}")
        return cols

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
                    # Fallback for [B, tokens, C]: use CLS token.
                    f = f[:, 0, :]
                feats.append(f.detach().cpu().numpy().astype(np.float32))
        out = np.concatenate(feats, axis=0)
        if out.shape[1] != self.feature_dim:
            raise ValueError(f"DINO feature dim mismatch: model={out.shape[1]} config={self.feature_dim}")
        return out

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

    def _vote(self, distances: np.ndarray, indices: np.ndarray) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        eps = 1e-8
        for row_d, row_i in zip(distances, indices):
            labels = [self.db_labels[int(j)] for j in row_i.tolist()]
            if self.metric == "cosine":
                sims = 1.0 - row_d
                if self.weights_mode == "distance":
                    weights = np.clip(sims, a_min=0.0, a_max=None)
                else:
                    weights = np.ones_like(sims)
            else:
                if self.weights_mode == "distance":
                    weights = 1.0 / (row_d + eps)
                else:
                    weights = np.ones_like(row_d)

            total = float(np.sum(weights)) + eps
            score_by_label: Dict[Any, float] = {}
            for lb, w in zip(labels, weights.tolist()):
                score_by_label[lb] = score_by_label.get(lb, 0.0) + float(w)

            pred_label = max(score_by_label.items(), key=lambda x: x[1])[0]
            pred_score = float(score_by_label[pred_label] / total)
            pred_int = None
            try:
                pred_int = int(pred_label)
            except Exception:
                pred_int = None

            neighbors = [
                {
                    "label": labels[i],
                    "distance": float(row_d[i]),
                    "weight": float(weights[i]),
                }
                for i in range(len(labels))
            ]
            outputs.append(
                {
                    "category_id": pred_int if pred_int is not None else pred_label,
                    "category_score": pred_score,
                    "neighbors": neighbors,
                }
            )
        return outputs

    def predict_patches(self, patches: List[np.ndarray]) -> List[Dict[str, Any]]:
        feats = self._extract_features(patches)
        if self.metric == "cosine":
            feats = _l2_normalize(feats)
        distances, indices = self._knn_query(feats)
        return self._vote(distances, indices)
