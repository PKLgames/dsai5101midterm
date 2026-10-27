"""Hybrid preference model: per-cat memory + shared appearance features.

Why a hybrid
------------
The two compared projects fail in opposite directions:

* per-cat strengths (approach A) are the best predictor for cats seen in the fit
  but carry no information about a cat that never appeared (strength = 0);
* a shared embedding (this project) transfers to unseen images/cats but is a
  weaker fit for cats seen ~200 times.

One L2 logistic regression over both blocks gets the best of the two, and can
optionally add approach A's YOLO/CLIP label block (fur-colour shares, pose, ...).

Design blocks (difference-coded: ``+1`` for the A side, ``-1`` for the B side)

    cat_strength : one-hot over the cats **seen in the fit**      (from approach A)
    embedding    : frozen ImageNet pooled features, 576-d         (this project)
    hand         : 20 low-level statistics, z-scored on fit rows  (this project)
    labels       : approach A's photo-label block                 (from approach A)
    side         : constant column (the shared A/B offset)

Everything is fitted jointly with ``fit_intercept=False`` plus the side column,
so the model stays antisymmetric in the images once the side term is removed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .approach_a import labels as a_labels


@dataclass
class HybridSpec:
    """Which blocks to include, and the penalty."""

    cat: bool = True
    embedding: bool = True
    hand: bool = False
    label_groups: tuple[str, ...] = ()   # e.g. ("colour_shares",) or ("coat_labels",)
    C: float = 1.0
    name: str = "hybrid"

    def describe(self) -> str:
        blocks = []
        if self.cat:
            blocks.append("cat_strength")
        if self.embedding:
            blocks.append("embedding")
        if self.hand:
            blocks.append("hand")
        if self.label_groups:
            blocks.append("labels:" + "+".join(self.label_groups))
        return " + ".join(blocks) or "side only"

    @property
    def uses_labels(self) -> bool:
        return bool(self.label_groups)


class HybridModel:
    """Joint model over item memory and shared appearance features."""

    def __init__(self, spec: HybridSpec, feats, photo_labels: pd.DataFrame | None = None):
        self.spec = spec
        self.feats = feats
        self.photo_labels = photo_labels
        if spec.uses_labels and photo_labels is None:
            raise ValueError("label_groups requested but photo_labels is None")

        self.clf: LogisticRegression | None = None
        self.cat_index: dict[str, int] = {}
        self._img_row = {p: i for i, p in enumerate(feats.paths)}
        self._hand_mu: np.ndarray | None = None
        self._hand_sd: np.ndarray | None = None
        self.names: list[str] = []

    # ---------------------------------------------------------------- blocks
    def _cat_block(self, df: pd.DataFrame) -> np.ndarray:
        n_cats = len(self.cat_index)
        if n_cats == 0:
            return np.zeros((len(df), 0))
        X = np.zeros((len(df), n_cats))
        rows = np.arange(len(df))
        for col, sign in (("a_cat", 1.0), ("b_cat", -1.0)):
            idx = df[col].map(self.cat_index)
            m = idx.notna().to_numpy()
            X[rows[m], idx[m].astype(int).to_numpy()] += sign
        return X

    def _appearance(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Deep embedding difference and hand-feature difference (z-scored)."""
        ia = df["a_image"].map(self._img_row).to_numpy()
        ib = df["b_image"].map(self._img_row).to_numpy()
        deep_diff = self.feats.deep[ia] - self.feats.deep[ib]
        hand = self.feats.hand
        if self._hand_mu is not None:
            hand = (hand - self._hand_mu) / self._hand_sd
        hand_diff = hand[ia] - hand[ib]
        return deep_diff, hand_diff

    def design(self, df: pd.DataFrame) -> np.ndarray:
        """Design matrix for ``df`` using the fitted cat vocabulary / scaler."""
        blocks = []
        if self.spec.cat:
            blocks.append(self._cat_block(df))
        if self.spec.embedding or self.spec.hand:
            deep_diff, hand_diff = self._appearance(df)
            if self.spec.embedding:
                blocks.append(deep_diff)
            if self.spec.hand:
                blocks.append(hand_diff)
        if self.spec.uses_labels:
            Xl, _ = a_labels.label_block(df, self.photo_labels, list(self.spec.label_groups))
            if Xl.shape[1]:
                blocks.append(Xl)
        X = np.hstack(blocks) if blocks else np.zeros((len(df), 0))
        return np.hstack([X, np.ones((len(df), 1))]).astype(np.float64)

    def column_names(self, df: pd.DataFrame) -> list[str]:
        names: list[str] = []
        if self.spec.cat:
            names += [f"cat={c}" for c in self.cat_index]
        if self.spec.embedding:
            names += [f"deep_{i:04d}" for i in range(self.feats.deep.shape[1])]
        if self.spec.hand:
            names += [f"hand_{n}" for n in self._hand_names()]
        if self.spec.uses_labels:
            _, lab_names = a_labels.label_block(df.head(5), self.photo_labels, list(self.spec.label_groups))
            names += lab_names
        names.append("side")
        return names

    def _hand_names(self) -> list[str]:
        return list(self.feats.meta.get("hand_features", [])) or [
            f"hand_{i}" for i in range(self.feats.hand.shape[1])
        ]

    # ------------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame, y) -> "HybridModel":
        if self.spec.cat:
            seen = sorted(set(df["a_cat"].astype(str)) | set(df["b_cat"].astype(str)))
            self.cat_index = {c: i for i, c in enumerate(seen)}
        if self.spec.hand:
            used = sorted(set(df["a_image"]) | set(df["b_image"]))
            rows = [self._img_row[p] for p in used if p in self._img_row]
            hand = self.feats.hand[rows]
            self._hand_mu = hand.mean(0, keepdims=True)
            sd = hand.std(0, keepdims=True)
            sd[sd < 1e-8] = 1.0
            self._hand_sd = sd

        self.names = self.column_names(df)
        X = self.design(df)
        self.clf = LogisticRegression(
            C=self.spec.C, fit_intercept=False, solver="lbfgs", max_iter=800
        ).fit(X, np.asarray(y))
        return self

    # --------------------------------------------------------------- predict
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("call fit() first")
        return self.clf.predict_proba(self.design(df))[:, 1]

    def coefficients(self) -> dict[str, float]:
        if self.clf is None:
            raise RuntimeError("call fit() first")
        return dict(zip(self.names, self.clf.coef_.ravel()))


def default_specs(C: float = 1.0) -> dict[str, HybridSpec]:
    """The model grid used by the comparison and submission scripts."""
    return {
        "A: cat strength": HybridSpec(cat=True, embedding=False, C=C, name="A: cat strength"),
        "B: embedding": HybridSpec(cat=False, embedding=True, C=C, name="B: embedding"),
        "B: embedding+hand": HybridSpec(cat=False, embedding=True, hand=True, C=C,
                                        name="B: embedding+hand"),
        "D: cat + embedding": HybridSpec(cat=True, embedding=True, C=C,
                                         name="D: cat + embedding"),
        "D: cat + embedding + hand": HybridSpec(cat=True, embedding=True, hand=True, C=C,
                                                name="D: cat + embedding + hand"),
        "D: cat + embedding + fur colour": HybridSpec(cat=True, embedding=True,
                                                      label_groups=("colour_shares",), C=C,
                                                      name="D: cat + embedding + fur colour"),
        "D: cat + embedding + all labels": HybridSpec(
            cat=True, embedding=True,
            label_groups=("colour_shares", "coat_labels", "pose", "breed", "shot"),
            C=C, name="D: cat + embedding + all labels",
        ),
    }
