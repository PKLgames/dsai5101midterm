"""Fit-only photo-level RBF Nyström preference scores."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy import linalg, sparse
from scipy.spatial.distance import cdist, pdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.development import feature_differences  # noqa: E402
from src.estimator import fit_logistic  # noqa: E402
from src.preprocessing import fit_preprocessing, transform_photos  # noqa: E402


def load_features() -> list[dict[str, Any]]:
    rows = pq.read_table(ROOT / "features" / "C.parquet").to_pylist()
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate CNN photo keys.")
    return rows


def fit_map(rows: list[dict[str, Any]], fit_ids: set[str], multiplier: float,
            landmark_limit: int = 256) -> dict[str, Any]:
    if multiplier <= 0 or landmark_limit < 1:
        raise ValueError("Invalid kernel settings.")
    index = {r["image_id"]: r for r in rows}
    if len(index) != len(rows) or not fit_ids <= set(index) or len(fit_ids) < 2:
        raise ValueError("Insufficient unique fitting photo features.")
    ids = sorted(fit_ids)
    spec = fit_preprocessing("C", rows, fit_ids)
    values = transform_photos(spec, [index[key] for key in ids])
    distances = pdist(values, metric="sqeuclidean")
    positive = distances[distances > 1e-12]
    if not len(positive):
        raise ValueError("CNN fitting photos are kernel-distance degenerate.")
    median = float(np.median(positive))
    gamma = multiplier / median
    landmark_ids = sorted(ids, key=lambda x: (
        hashlib.sha256(("kernel-landmark-20261028:" + x).encode("utf-8")).hexdigest(), x))[:landmark_limit]
    lookup = {name: i for i, name in enumerate(ids)}
    landmarks = values[[lookup[name] for name in landmark_ids]]
    gram = np.exp(-gamma * cdist(landmarks, landmarks, metric="sqeuclidean"))
    eigenvalues, eigenvectors = linalg.eigh(gram, check_finite=True)
    if eigenvalues[-1] <= 0 or eigenvalues[0] < -1e-9 * eigenvalues[-1]:
        raise ValueError("Kernel landmark Gram matrix is not positive semidefinite.")
    floor = eigenvalues[-1] * 1e-12
    inverse = (eigenvectors * (1 / np.sqrt(np.maximum(eigenvalues, floor)))) @ eigenvectors.T
    projected = np.exp(-gamma * cdist(values, landmarks, metric="sqeuclidean")) @ inverse
    center = projected.mean(axis=0)
    rms = float(np.sqrt(np.mean(np.sum((projected-center)**2, axis=1))))
    if not np.isfinite(rms) or rms < 1e-12:
        raise ValueError("Fitting Nyström features are degenerate.")
    return {"cnn_preprocessing": spec, "fitting_photos": ids, "landmark_ids": landmark_ids,
            "landmark_cnn_vectors": landmarks.tolist(), "normalizer": inverse.tolist(),
            "center": center.tolist(), "rms": rms, "gamma": gamma,
            "distance_median_squared": median, "multiplier": multiplier,
            "eigenvalue_floor": float(floor), "eigenvalues_clipped": int(np.sum(eigenvalues < floor))}


def transform(mapping: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    x = transform_photos(mapping["cnn_preprocessing"], rows)
    projected = np.exp(-mapping["gamma"] * cdist(
        x, np.asarray(mapping["landmark_cnn_vectors"]), metric="sqeuclidean")) @ np.asarray(mapping["normalizer"])
    result = (projected - np.asarray(mapping["center"])) / mapping["rms"]
    if not np.isfinite(result).all():
        raise ValueError("Nyström transformation is nonfinite.")
    return result


def designs(train: list[dict[str, Any]], val: list[dict[str, Any]], features: list[dict[str, Any]],
            mapping: dict[str, Any]) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    ids = sorted({r[f"{side}_image"] for r in train + val for side in ("a", "b")})
    index = {r["image_id"]: r for r in features}
    if not set(ids) <= set(index) or any(index[key]["extraction_status"] != "ok" for key in ids):
        raise ValueError("Required photo feature is unavailable.")
    values = transform(mapping, [index[name] for name in ids])
    dx, dv = (feature_differences(data, ids, values) for data in (train, val))
    return (sparse.hstack([np.ones((len(train), 1)), dx], format="csr"),
            sparse.hstack([np.ones((len(val), 1)), dv], format="csr"))
