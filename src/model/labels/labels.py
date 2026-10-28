"""Readers and encoders for the YOLO + CLIP photo labels.

Provenance
----------
The label *values* are produced by this repo's ``photo-labelling/*.py``
(YOLO11n-seg mask -> fur-colour shares inside the mask, plus CLIP zero-shot
attributes).  This module only reads that table and turns it into pairwise
design blocks; the encoders mirror ``numeric_diff`` / ``one_hot_diff`` /
``with_side`` so scores match the original notebooks.

Nothing here is used by this project's default feature pipeline: the labels are
an *optional* input of the hybrid model (see ``model.hybrid``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths

#: The five fur-colour shares (share_white is omitted because
#: the six shares sum to one).
COLOUR_SHARES = ["share_black", "share_gray", "share_orange", "share_cream", "share_brown"]

#: Categorical label groups available in the table.
LABEL_GROUPS = {
    "colour_primary": ["colour_primary"],
    "colour_pattern": ["colour_pattern"],
    "coat_labels": ["colour_primary", "colour_pattern"],
    "pose": ["framing", "view", "eyes", "eyes_colour"],
    "breed": ["breed"],
    "shot": ["bbox_area_frac"],
}

NUMERIC_COLUMNS = ["share_black", "share_gray", "share_orange", "share_cream",
                   "share_brown", "share_white", "bbox_area_frac", "tabby_score", "pointed_score"]


def load_photo_labels(repo_dir=None) -> pd.DataFrame:
    """Read the photo-label table (indexed by image path).

    Uses ``repo_dir/artifacts/labels/photo_labels.csv`` when ``repo_dir`` is
    set, otherwise ``artifacts/labels/photo_labels.csv`` in this repo.
    """
    if repo_dir:
        csv = Path(repo_dir) / "artifacts" / "labels" / "photo_labels.csv"
    else:
        csv = paths.PHOTO_LABELS_CSV
    if not csv.exists():
        raise FileNotFoundError(
            "photo_labels.csv not found. Expected "
            f"{csv}. Regenerate it with photo-labelling/label_photos.py."
        )
    df = pd.read_csv(csv)
    df["image"] = df["image"].astype(str)
    return df.drop_duplicates("image").set_index("image")


def available_columns(labels: pd.DataFrame) -> dict[str, list[str]]:
    return {g: [c for c in cols if c in labels.columns] for g, cols in LABEL_GROUPS.items()}


def numeric_diff(df: pd.DataFrame, labels: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """``label_A - label_B`` for numeric label columns."""
    a = labels.reindex(df["a_image"].astype(str).to_numpy())[columns].to_numpy(dtype=float)
    b = labels.reindex(df["b_image"].astype(str).to_numpy())[columns].to_numpy(dtype=float)
    return np.nan_to_num(a - b, nan=0.0)


def one_hot(df: pd.DataFrame, labels: pd.DataFrame, column: str) -> np.ndarray:
    """One-hot difference for a categorical label column.

    Levels come from the label table (fixed, so train and test share columns);
    missing values get their own ``"missing"`` level.
    """
    values = labels[column].astype("string").fillna("missing")
    levels = sorted(values.unique())
    a = labels.reindex(df["a_image"].astype(str).to_numpy())[column].astype("string").fillna("missing")
    b = labels.reindex(df["b_image"].astype(str).to_numpy())[column].astype("string").fillna("missing")
    Xa = np.stack([(a == lv).to_numpy() for lv in levels], axis=1).astype(float)
    Xb = np.stack([(b == lv).to_numpy() for lv in levels], axis=1).astype(float)
    return Xa - Xb


def one_hot_names(labels: pd.DataFrame, column: str) -> list[str]:
    values = labels[column].astype("string").fillna("missing")
    return [f"{column}={lv}" for lv in sorted(values.unique())]


def label_block(df: pd.DataFrame, labels: pd.DataFrame, groups: list[str]) -> tuple[np.ndarray, list[str]]:
    """Concatenate the requested label groups into one design block."""
    blocks, names = [], []
    avail = available_columns(labels)
    for group in groups:
        if group in avail:
            cols = avail[group]
            if group == "shot":
                blocks.append(numeric_diff(df, labels, cols))
                names += [f"label_{c}" for c in cols]
                continue
            for col in cols:
                if col in NUMERIC_COLUMNS:
                    blocks.append(numeric_diff(df, labels, [col]))
                    names.append(f"label_{col}")
                else:
                    blocks.append(one_hot(df, labels, col))
                    names += one_hot_names(labels, col)
        elif group == "colour_shares":
            blocks.append(numeric_diff(df, labels, COLOUR_SHARES))
            names += [f"label_{c}" for c in COLOUR_SHARES]
        else:
            raise KeyError(f"unknown label group {group!r}; have {sorted(avail) + ['colour_shares']}")
    if not blocks:
        return np.zeros((len(df), 0)), []
    return np.hstack(blocks), names


def with_side(X: np.ndarray) -> np.ndarray:
    """Append the constant side column (``with_side``)."""
    return np.hstack([X, np.ones((len(X), 1))]) if X.shape[1] else np.ones((len(X), 1))
