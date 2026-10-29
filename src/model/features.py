"""Load the cached image features and turn them into pairwise design matrices.

The downstream task is: given two images (A, B) and the annotator's profile,
predict P(annotator prefers A).  Because the test pairs never appear in train
and ~19% of test images are never seen, the design deliberately contains **no
image identity**: only functions of the two feature vectors.

Two encodings are supported:

``diff``
    ``f_a - f_b``.  Linear models on this are exactly the Bradley-Terry /
    Thurstone pairwise-comparison model with a learned image "strength"
    ``s = w @ f``: ``P(A) = sigmoid(w @ (f_a - f_b))``.  It is antisymmetric by
    construction, so it cannot learn a position bias that does not exist.
``diff_absdiff``
    ``[f_a - f_b, |f_a - f_b|]``.  The absolute term lets a model express
    "large differences matter more", at 2x the width.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths


@dataclass
class Features:
    paths: np.ndarray
    deep: np.ndarray
    hand: np.ndarray
    ok: np.ndarray
    meta: dict
    _row: dict[str, int] | None = None

    @property
    def n(self) -> int:
        return len(self.paths)

    def row(self, image_path: str) -> int:
        if self._row is None:
            self._row = {p: i for i, p in enumerate(self.paths)}
        return self._row[image_path]

    def lookup(self, image_paths) -> tuple[np.ndarray, np.ndarray]:
        """Stack feature rows; unknown/unreadable images -> zero row + mask False."""
        d = self.deep.shape[1]
        out = np.zeros((len(image_paths), d), dtype=np.float32)
        present = np.zeros(len(image_paths), dtype=bool)
        for i, p in enumerate(image_paths):
            j = self._row.get(p) if self._row else None
            if self._row is None:
                self._row = {q: k for k, q in enumerate(self.paths)}
                j = self._row.get(p)
            if j is not None and self.ok[j]:
                out[i] = self.deep[j]
                present[i] = True
        return out, present


def available_tags() -> list[str]:
    return sorted(p.stem for p in paths.FEATURES_DIR.glob("*.npz"))


def combine_caches(tags: list[str], out_tag: str, hand_from: str | None = None) -> str:
    """Stack the deep embeddings of several caches into one derived cache.

    Different backbones see the same photos through different inductive biases; a
    concatenated embedding is the cheapest way to add diversity without a new
    forward pass at modelling time.  The hand block is taken from ``hand_from``
    (or the first tag) and copied unchanged.  All caches must share ``paths``.
    """
    if len(tags) < 2:
        raise ValueError("need at least two caches to combine")
    loaded = [np.load(paths.FEATURES_DIR / f"{t}.npz", allow_pickle=False) for t in tags]
    base = loaded[0]["paths"].astype(str)
    for t, z in zip(tags[1:], loaded[1:]):
        if not np.array_equal(z["paths"].astype(str), base):
            raise ValueError(f"cache {t} has a different image order than {tags[0]}")
    deep = np.hstack([z["deep"].astype(np.float32) for z in loaded])
    src = tags.index(hand_from) if hand_from in tags else 0
    hand = loaded[src]["hand"].astype(np.float32)
    ok = np.logical_and.reduce([z["ok"].astype(bool) for z in loaded])

    dest = paths.FEATURES_DIR / f"{out_tag}.npz"
    np.savez_compressed(
        dest,
        paths=base,
        deep=deep,
        hand=hand,
        ok=ok,
        hand_names=loaded[src]["hand_names"],
    )
    metas = []
    for t in tags:
        mp = paths.FEATURES_DIR / f"{t}.meta.json"
        metas.append(json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {"tag": t})
    meta = dict(metas[0])
    meta.update(
        {
            "tag": out_tag,
            "feature_dim_deep": int(deep.shape[1]),
            "combined_from": tags,
            "components": [
                {"tag": m.get("tag"), "backbone": m.get("backbone"),
                 "dim": m.get("feature_dim_deep"), "elapsed_sec": m.get("elapsed_sec")}
                for m in metas
            ],
            "n_images": int(len(base)),
            "n_ok": int(ok.sum()),
            "l2_normalized": True,
            "note": "concatenation of L2-normalised per-backbone embeddings; the "
                    "concatenated vector is not re-normalised",
        }
    )
    (paths.FEATURES_DIR / f"{out_tag}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_tag


def load_features(tag: str | None = None, kind: str = "auto") -> Features:
    """Load ``artifacts/features/<tag>.npz``.

    ``tag=None`` picks the most recently written cache.  ``kind`` selects which
    embedding goes into ``deep``: ``deep`` (CNN), ``hand`` (low-level), ``both``
    (concatenated) or ``auto`` (CNN only).
    """
    tags = available_tags()
    if not tags:
        raise FileNotFoundError(
            f"no feature cache in {paths.FEATURES_DIR} - run scripts/03_extract_features.py"
        )
    if tag is None:
        tag = max(tags, key=lambda t: (paths.FEATURES_DIR / f"{t}.npz").stat().st_mtime)
    if tag not in tags:
        raise KeyError(f"unknown tag {tag!r}; available: {tags}")

    z = np.load(paths.FEATURES_DIR / f"{tag}.npz", allow_pickle=False)
    meta_path = paths.FEATURES_DIR / f"{tag}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    cnn = z["deep"].astype(np.float32)
    hand = z["hand"].astype(np.float32)

    if kind == "deep" or kind == "auto":
        deep = cnn
    elif kind == "hand":
        deep = _standardize(hand)
    elif kind == "both":
        deep = np.hstack([cnn, _standardize(hand)])
    else:
        raise ValueError(f"unknown kind {kind!r}")

    return Features(
        paths=z["paths"].astype(str),
        deep=deep,
        hand=hand,
        ok=z["ok"].astype(bool),
        meta=meta,
    )


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return ((x - mu) / sd).astype(np.float32)


def demo_onehot(df: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
    """One-hot the annotator columns, using an explicit ``missing`` level."""
    cols = columns or paths.DEMO_COLS
    blocks = []
    for c in cols:
        levels = paths.CATEGORICAL_LEVELS[c]
        vals = df[c].astype("string").fillna("missing") if c in df.columns else pd.Series(["missing"] * len(df))
        vals = vals.where(vals.isin(levels), "missing")
        blocks.append(np.stack([(vals == lv).to_numpy() for lv in levels], axis=1))
    return np.hstack(blocks).astype(np.float32)


def pair_matrix(
    df: pd.DataFrame,
    feats: Features,
    mode: str = "diff",
    with_demo: bool = False,
    with_comparison: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(X, present_a, present_b)`` for the rows of ``df``.

    ``present_*`` is False when an image has no usable feature row (missing
    file / decode error); the caller can fall back to a prior for those rows.
    """
    fa, pa = feats.lookup(df["a_image"].astype(str).tolist())
    fb, pb = feats.lookup(df["b_image"].astype(str).tolist())
    diff = fa - fb
    if mode == "diff":
        X = diff
    elif mode == "diff_absdiff":
        X = np.hstack([diff, np.abs(diff)])
    else:
        raise ValueError(f"unknown mode {mode!r}")

    extra = []
    if with_demo:
        extra.append(demo_onehot(df))
    if with_comparison:
        ct = df["comparison_type"].astype("string").fillna("missing")
        extra.append(
            np.stack(
                [(ct == lv).to_numpy() for lv in paths.CATEGORICAL_LEVELS["comparison_type"]],
                axis=1,
            ).astype(np.float32)
        )
    if extra:
        X = np.hstack([X] + extra)
    return X.astype(np.float32), pa, pb


def label_a(df: pd.DataFrame) -> np.ndarray:
    """1 when the annotator chose image A."""
    return (df["choice"].astype("string").str.upper() == "A").to_numpy(dtype=np.int8)
