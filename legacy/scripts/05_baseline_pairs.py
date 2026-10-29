#!/usr/bin/env python3
"""Downstream baseline: predict which image an annotator picked, from features.

This is deliberately a *baseline* whose job is to prove that the cached features
carry preference signal and that the interface is usable; a later, stronger
model can replace it without touching the feature layer.

Model
-----
Bradley-Terry / Thurstone style pairwise logistic regression on
``f_a - f_b`` (so P(A) + P(B) = 1 by construction), optionally concatenated
with annotator demographics and the given ``comparison_type`` flag.

Two leak-aware validation protocols
-----------------------------------
P1  pair-grouped 5-fold
    All rows of an image pair stay in one fold (the same pair is answered by up
    to 2 annotators).  Measures generalisation to new people / new pairs.
P2  image-holdout 3-fold
    A third of the *images* is removed from training; we then predict only the
    rows where **both** images are held out.  This mimics the test set, where
    25% of its rows contain two images that never appear in train.  P2 is the
    honest number for this competition.

Outputs
-------
    artifacts/features/baseline_metrics.{json,csv}
    artifacts/submissions/pred_test.csv     (choice + prob_a filled, test.csv order)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss  # noqa: E402
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from model import features, index, paths  # noqa: E402

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
VARIANTS = ["const", "demo", "deep", "deep_demo", "deep_hand"]


# --------------------------------------------------------------------------- #
# design matrices
# --------------------------------------------------------------------------- #
def _comparison_onehot(df: pd.DataFrame) -> np.ndarray:
    ct = df["comparison_type"].astype("string").fillna("missing")
    levels = paths.CATEGORICAL_LEVELS["comparison_type"]
    return np.stack([(ct == lv).to_numpy() for lv in levels], axis=1).astype(np.float32)


def build_X(df: pd.DataFrame, variant: str, feats_deep, feats_both) -> np.ndarray:
    if variant == "const":
        return np.zeros((len(df), 0), dtype=np.float32)
    if variant == "demo":
        return np.hstack([features.demo_onehot(df), _comparison_onehot(df)]).astype(np.float32)
    if variant == "deep":
        X, _, _ = features.pair_matrix(df, feats_deep, mode="diff", with_demo=False)
        return X
    if variant == "deep_demo":
        X, _, _ = features.pair_matrix(df, feats_deep, mode="diff", with_demo=True)
        return X
    if variant == "deep_hand":
        X, _, _ = features.pair_matrix(df, feats_both, mode="diff", with_demo=False)
        return X
    raise ValueError(variant)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def fit_predict(Xtr, ytr, Xte, C: float, prior: float) -> np.ndarray:
    if Xtr.shape[1] == 0:  # constant baseline
        return np.full(len(Xte), prior)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=400, tol=1e-4, solver="lbfgs")
    clf.fit(scaler.transform(Xtr), ytr)
    return clf.predict_proba(scaler.transform(Xte))[:, 1]


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "logloss": round(float(log_loss(y, p)), 5),
        "accuracy": round(float(accuracy_score(y, p >= 0.5)), 5),
        "brier": round(float(np.mean((p - y) ** 2)), 5),
        "base_rate": round(float(np.mean(y)), 4),
    }


# --------------------------------------------------------------------------- #
# protocols
# --------------------------------------------------------------------------- #
def choose_C(train, feats_deep, feats_both, groups, seed: int = 0):
    """Pick C on a held-out 25% of *pairs* (never a random row split)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr_idx, va_idx = next(gss.split(train, groups=groups))
    sub_tr, sub_va = train.iloc[tr_idx], train.iloc[va_idx]
    Xtr = build_X(sub_tr, "deep", feats_deep, feats_both)
    ytr = features.label_a(sub_tr)
    Xva = build_X(sub_va, "deep", feats_deep, feats_both)
    yva = features.label_a(sub_va)

    rows, best_C, best_ll = [], None, np.inf
    for C in C_GRID:
        p = fit_predict(Xtr, ytr, Xva, C, float(ytr.mean()))
        ll = float(log_loss(yva, np.clip(p, 1e-6, 1 - 1e-6)))
        rows.append({"C": C, "val_logloss": round(ll, 5)})
        if ll < best_ll:
            best_C, best_ll = C, ll
    print("  C sweep (deep, 25% pair-grouped holdout):")
    for r in rows:
        print(f"    C={r['C']:<6} logloss={r['val_logloss']:.5f}")
    return float(best_C), rows


def protocol_p1(train, feats_deep, feats_both, C: float, groups) -> dict:
    """Pair-grouped 5-fold cross-validation."""
    preds = {v: np.zeros(len(train)) for v in VARIANTS}
    for f, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=5).split(train, groups=groups)):
        sub_tr, sub_va = train.iloc[tr_idx], train.iloc[va_idx]
        ytr = features.label_a(sub_tr)
        prior = float(ytr.mean())
        for v in VARIANTS:
            preds[v][va_idx] = fit_predict(
                build_X(sub_tr, v, feats_deep, feats_both),
                ytr,
                build_X(sub_va, v, feats_deep, feats_both),
                C,
                prior,
            )
        print(f"    fold {f + 1}/5 done", flush=True)
    y = features.label_a(train)
    return {v: metrics(y, preds[v]) for v in VARIANTS}


def protocol_p2(train, feats_deep, feats_both, C: float, seed: int = 0, n_folds: int = 3) -> dict:
    """Image-holdout: predict only rows whose *both* images are unseen."""
    imgs = np.array(sorted(set(train["a_image"]) | set(train["b_image"])))
    store = {v: {"y": [], "p": [], "ct": []} for v in VARIANTS}
    for f, (_, held_idx) in enumerate(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(imgs)):
        held = set(imgs[held_idx])
        a_held = train["a_image"].isin(held).to_numpy()
        b_held = train["b_image"].isin(held).to_numpy()
        ev, tr = a_held & b_held, ~a_held & ~b_held
        sub_tr, sub_ev = train[tr], train[ev]
        ytr = features.label_a(sub_tr)
        prior = float(ytr.mean())
        for v in VARIANTS:
            store[v]["y"].append(features.label_a(sub_ev))
            store[v]["p"].append(
                fit_predict(
                    build_X(sub_tr, v, feats_deep, feats_both),
                    ytr,
                    build_X(sub_ev, v, feats_deep, feats_both),
                    C,
                    prior,
                )
            )
            store[v]["ct"].append(sub_ev["comparison_type"].astype(str).to_numpy())
        print(f"    fold {f + 1}/{n_folds}: train {len(sub_tr)} rows, eval {len(sub_ev)} rows", flush=True)

    out = {}
    for v in VARIANTS:
        y = np.concatenate(store[v]["y"])
        p = np.concatenate(store[v]["p"])
        ct = np.concatenate(store[v]["ct"])
        m = metrics(y, p)
        m["by_comparison_type"] = {k: metrics(y[ct == k], p[ct == k]) for k in np.unique(ct)}
        out[v] = m
    return out


# --------------------------------------------------------------------------- #
# submission
# --------------------------------------------------------------------------- #
def write_submission(test: pd.DataFrame, p: np.ndarray, out: Path) -> pd.DataFrame:
    out.parent.mkdir(parents=True, exist_ok=True)
    sub = test.copy()
    sub["prob_a"] = np.round(np.clip(p, 0.0, 1.0), 4)
    sub["choice"] = np.where(sub["prob_a"] >= 0.5, "A", "B")
    sub.to_csv(out, index=False)

    ex = pd.read_csv(paths.TEST_EXAMPLE_CSV, encoding="utf-8-sig")
    assert list(sub.columns) == list(test.columns) == list(ex.columns), "column mismatch"
    assert len(sub) == len(ex) == 6224, f"row count {len(sub)} != 6224"
    assert (sub["test_row_id"].to_numpy() == ex["test_row_id"].to_numpy()).all(), "row id order"
    assert sub["prob_a"].between(0, 1).all(), "prob_a out of range"
    assert ((sub["prob_a"] >= 0.5) == (sub["choice"] == "A")).all(), "choice/prob_a inconsistent"
    assert sub["prob_a"].notna().all() and sub["choice"].isin(["A", "B"]).all()
    return sub


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None, help="feature cache (default: most recent)")
    ap.add_argument("--C", type=float, default=None, help="skip the C sweep and use this value")
    ap.add_argument("--p2-folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-submission", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    train, test = index.load_train(), index.load_test()
    feats_deep = features.load_features(args.tag, kind="deep")
    tag = feats_deep.meta.get("tag", args.tag)
    feats_both = features.load_features(tag, kind="both")
    y = features.label_a(train)
    groups = index.pair_key(train).to_numpy()
    print(f"tag={tag}  train rows={len(train)}  pairs={len(set(groups))}  features={feats_deep.deep.shape[1]}-d"
          f" (+{feats_both.deep.shape[1] - feats_deep.deep.shape[1]} hand)")

    C = args.C
    sweep: list[dict] = []
    if C is None:
        C, sweep = choose_C(train, feats_deep, feats_both, groups, seed=args.seed)
    print(f"  using C={C}\n")

    results: dict = {"tag": tag, "C": C, "seed": args.seed, "c_sweep": sweep, "protocols": {}}

    print("P1: pair-grouped 5-fold CV")
    results["protocols"]["P1_pair_grouped_5fold"] = protocol_p1(train, feats_deep, feats_both, C, groups)
    print("\nP2: image-holdout 3-fold (both images unseen, mimics the test set)")
    results["protocols"]["P2_image_holdout_3fold"] = protocol_p2(
        train, feats_deep, feats_both, C, seed=args.seed, n_folds=args.p2_folds
    )

    rows = [
        {"protocol": proto, "variant": v, "n": m["n"], "logloss": m["logloss"],
         "accuracy": m["accuracy"], "brier": m["brier"]}
        for proto, per_variant in results["protocols"].items()
        for v, m in per_variant.items()
    ]
    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))
    table.to_csv(paths.FEATURES_DIR / "baseline_metrics.csv", index=False)

    p2 = results["protocols"]["P2_image_holdout_3fold"]
    best = min(VARIANTS, key=lambda v: p2[v]["logloss"])
    print(f"\nbest variant by P2 logloss: {best}")
    for k, m in p2[best]["by_comparison_type"].items():
        print(f"  {k:<13} n={m['n']:<6} logloss={m['logloss']:.4f} acc={m['accuracy']:.4f}")
    results["best_variant_p2"] = best

    (paths.FEATURES_DIR / "baseline_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nmetrics -> {paths.FEATURES_DIR / 'baseline_metrics.json'}")

    if not args.no_submission:
        print(f"\nfitting final '{best}' model on all training rows ...")
        Xtr = build_X(train, best, feats_deep, feats_both)
        Xte = build_X(test, best, feats_deep, feats_both)
        p_te = fit_predict(Xtr, y, Xte, C, float(y.mean()))
        out = paths.SUBMISSIONS_DIR / "pred_test.csv"
        sub = write_submission(test, p_te, out)
        print(f"submission -> {out}  (checks passed, mean prob_a={sub['prob_a'].mean():.3f}, "
              f"A share={(sub['choice'] == 'A').mean():.3f})")


if __name__ == "__main__":
    main()
