"""Per-item strengths + side term (Bradley-Terry fit).

Adapted from the cat-strength fit (``fit_bt`` / ``predict_bt`` / ``pair_masks``).
Only the reusable pieces are kept.  Signatures were made explicit (item id
columns and penalty are parameters).

Model
-----
    P(choose A) = logistic(strength(item_A) - strength(item_B) + side)

Strengths are L2-penalised logistic-regression coefficients on a difference-
coded one-hot design (``+1`` for A's item, ``-1`` for B's), fitted with
``fit_intercept=False`` plus an explicit side column, then centred to mean 0.
An item absent from the fit gets strength 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

#: The comparison project fixes this penalty and shows nearby values agree.
DEFAULT_C = 1.0


def ensure_pair_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the two derived columns the strength fit relies on, if missing.

    ``unordered_pair_id`` is the two image paths sorted and joined by ``||``
    (so both orientations of a pair share an id); ``chose_a`` is 1 when the
    annotator picked side A.
    """
    out = df
    if "unordered_pair_id" not in out.columns:
        out = out.copy()
        out["unordered_pair_id"] = ["||".join(sorted(p)) for p in zip(out["a_image"], out["b_image"])]
    if "chose_a" not in out.columns:
        out = out.copy()
        out["chose_a"] = (out["choice"].astype("string").str.upper() == "A").astype(int)
    return out


def logistic(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def fit_strength(
    df: pd.DataFrame,
    id_a: str = "a_cat",
    id_b: str = "b_cat",
    label_col: str = "chose_a",
    C: float = DEFAULT_C,
) -> dict:
    """Fit one strength per item id plus the side term.

    Returns ``{"strength": {id: value}, "delta": side, "ids": array, "index": {}}``.
    """
    ids = pd.unique(np.concatenate([df[id_a].to_numpy(), df[id_b].to_numpy()]))
    index = {item: k for k, item in enumerate(ids)}
    n = len(df)
    X = np.zeros((n, len(ids) + 1))
    rows = np.arange(n)
    a = df[id_a].map(index).to_numpy()
    b = df[id_b].map(index).to_numpy()
    X[rows, a] += 1.0
    X[rows, b] -= 1.0
    X[:, -1] = 1.0  # side term

    clf = LogisticRegression(C=C, fit_intercept=False, solver="lbfgs", max_iter=500)
    clf.fit(X, df[label_col].to_numpy())

    coef = clf.coef_.ravel()
    strength = coef[:-1] - coef[:-1].mean()
    return {
        "strength": dict(zip(ids, strength)),
        "delta": float(coef[-1]),
        "ids": ids,
        "index": index,
    }


def predict_strength(model: dict, df: pd.DataFrame, id_a: str = "a_cat", id_b: str = "b_cat") -> np.ndarray:
    """Probabilities for ``df``; ids absent from the fit contribute strength 0."""
    sa = df[id_a].map(model["strength"]).fillna(0).to_numpy(dtype=float)
    sb = df[id_b].map(model["strength"]).fillna(0).to_numpy(dtype=float)
    return logistic(sa - sb + model["delta"])


def pair_masks(df: pd.DataFrame, k: int = 5, seed: int = 21, pair_col: str = "unordered_pair_id",
               rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """Group-aware K folds: both orientations of a pair stay in the same fold."""
    generator = rng if rng is not None else np.random.default_rng(seed)
    pairs = generator.permutation(df[pair_col].drop_duplicates().to_numpy())
    return [df[pair_col].isin(set(fold)).to_numpy() for fold in np.array_split(pairs, k)]
