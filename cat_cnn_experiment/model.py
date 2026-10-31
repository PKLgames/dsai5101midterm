"""Fit-only CNN preprocessing and additive cat + CNN design."""

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.development import feature_differences, identity_matrix  # noqa: E402
from src.preprocessing import fit_preprocessing, transform_photos  # noqa: E402


def load_features():
    rows = pq.read_table(ROOT/"features"/"C.parquet").to_pylist()
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate CNN photo keys.")
    return rows


def fit_spec(rows, features):
    images = {r[f"{side}_image"] for r in rows for side in ("a", "b")}
    cats = sorted({r[f"{side}_cat"] for r in rows for side in ("a", "b")})
    return {"preprocessing": fit_preprocessing("C", features, images), "cats": cats, "fitting_photos": sorted(images)}


def design(rows, features, spec):
    cats = identity_matrix(rows, spec["cats"], "cat", side=False)
    index = {r["image_id"]: r for r in features}
    ids = sorted({r[f"{side}_image"] for r in rows for side in ("a", "b")})
    if len(index) != len(features) or not set(ids) <= set(index) or any(index[key]["extraction_status"] != "ok" for key in ids):
        raise ValueError("Missing or unusable CNN photo features.")
    values = transform_photos(spec["preprocessing"], [index[key] for key in ids])
    diff = feature_differences(rows, ids, values)
    return sparse.hstack([np.ones((len(rows), 1)), cats, diff], format="csr")


def penalty(cat_count, cnn_dim, cat_ridge, cnn_ridge):
    if cat_count < 1 or cnn_dim < 1 or cat_ridge <= 0 or cnn_ridge <= 0:
        raise ValueError("Invalid hybrid penalty configuration.")
    return np.r_[[1e-6], np.full(cat_count, cat_ridge), np.full(cnn_dim, cnn_ridge)]
