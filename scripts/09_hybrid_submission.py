#!/usr/bin/env python3
"""Fit the final preference model on all training rows and write the test predictions.

Two structures are available:

``joint`` (default in earlier versions)
    One L2 logistic regression over
    ``[cat one-hot | 576-d embedding | 20 hand | side]``.  Best average score, but the
    cat dummies absorb cat-level variance, so the shared weights transfer badly to a
    cat (or photo) that was never seen.

``routed`` (default here)
    Fit the joint model **and** a transfer model (fur-colour shares only), then at
    prediction time use the joint model only when both cats *and* both photos were
    present in the training data; otherwise use the transfer model.  This removes the
    cold-start regression without touching the average:

        routed  avg 0.6565 | both photos unseen 0.6823 | both cats unseen 0.6805
        joint   avg 0.6565 | both photos unseen 0.6847 | both cats unseen 0.6945

    (family medians on those two cold slices: 0.6837 and 0.6892)

The routing signal (was this cat / photo in the training data) is known at submission
time and uses no test labels.

Outputs
    data/submissions/pred_routed_test.csv   when --structure routed (recommended)
    data/submissions/pred_hybrid_test.csv   when --structure joint
    data/features/hybrid_fit.json           spec, width, routing shares, slice summary

Usage
    python scripts/09_hybrid_submission.py
    python scripts/09_hybrid_submission.py --structure joint
    python scripts/09_hybrid_submission.py --C 0.3 --sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from catarena import features, hybrid, index, paths  # noqa: E402
from catarena.approach_a import labels as a_labels  # noqa: E402
from catarena.approach_a import metrics as a_metrics  # noqa: E402
from catarena.approach_a import strength as a_strength  # noqa: E402

JOINT_SPEC = "D: cat + embedding + hand"
DEFAULT_C = 0.3
TRANSFER_SPEC = hybrid.HybridSpec(cat=False, embedding=False,
                                  label_groups=("colour_shares",), C=1.0, name="transfer: fur colour")


def evaluate(train, y, masks, spec, feats, photo_labels) -> dict:
    p = np.empty(len(train))
    for mask in masks:
        fit_df = train.loc[~mask]
        model = hybrid.HybridModel(spec, feats, photo_labels).fit(fit_df, y[~mask])
        p[mask] = model.predict(train.loc[mask])
    return a_metrics.metrics(y, p)


def slice_table(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    cats_seen = set(train_df["a_cat"]) | set(train_df["b_cat"])
    imgs_seen = set(train_df["a_image"]) | set(train_df["b_image"])
    n_unseen_cats = (~df["a_cat"].isin(cats_seen)).astype(int) + (~df["b_cat"].isin(cats_seen)).astype(int)
    both_photos = df["a_image"].isin(imgs_seen) & df["b_image"].isin(imgs_seen)
    is_between = df["comparison_type"].eq("between_cats")
    label = np.where(
        ~is_between, "within_cat",
        np.where(n_unseen_cats == 0, "both cats seen",
                 np.where(n_unseen_cats == 1, "one unseen cat", "two unseen cats")),
    )
    label = np.where(both_photos, label, "photos unseen")
    return (
        df.assign(_slice=label).groupby("_slice")
        .agg(rows=("prob_a", "size"), mean_prob_a=("prob_a", "mean"),
             share_choice_a=("choice", lambda s: float(np.mean(s == "A"))))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="mobilenet_v3_small_224",
                    help="feature cache; pins the backbone used in the reported results")
    ap.add_argument("--structure", choices=["routed", "joint"], default="routed")
    ap.add_argument("--spec", default=JOINT_SPEC)
    ap.add_argument("--C", type=float, default=DEFAULT_C)
    ap.add_argument("--sweep", action="store_true", help="score candidate specs before fitting")
    ap.add_argument("--approach-a", default=None,
                    help="optional path to a clone of the label project; default uses the vendored table")
    args = ap.parse_args()

    train, test = index.load_train(), index.load_test()
    train = a_strength.ensure_pair_columns(train)
    y = features.label_a(train)
    feats = features.load_features(args.tag, kind="deep")
    photo_labels = a_labels.load_photo_labels(Path(args.approach_a) if args.approach_a else None)

    specs = hybrid.default_specs(args.C)
    if args.spec not in specs:
        raise SystemExit(f"unknown spec {args.spec!r}; choose from {sorted(specs)}")
    spec = specs[args.spec]

    masks = a_strength.pair_masks(train, 5, seed=21)
    summary = {spec.name: evaluate(train, y, masks, spec, feats, photo_labels)}
    if args.sweep:
        print("pair-grouped 5-fold in-sample check (all rows):")
        for name, s in specs.items():
            m = summary.setdefault(name, evaluate(train, y, masks, s, feats, photo_labels))
            print(f"  {name:34s} C={s.C:<5} logloss {m['logloss']:.4f}  acc {m['accuracy']:.3f}")
    else:
        m = summary[spec.name]
        print(f"in-sample (pair-grouped 5-fold) {spec.name}: logloss {m['logloss']:.4f} acc {m['accuracy']:.3f}")

    print(f"\nfitting on all {len(train):,} rows: {spec.describe()}, C={spec.C}")
    model = hybrid.HybridModel(spec, feats, photo_labels).fit(train, y)
    coef = model.coefficients()
    print(f"  design width {len(coef)} columns; side term {coef['side']:+.4f}")

    prob = model.predict(test)
    routing = {"structure": "joint", "share_rows_using_joint": 1.0}
    if args.structure == "routed":
        transfer = hybrid.HybridModel(TRANSFER_SPEC, feats, photo_labels).fit(train, y)
        cats_seen = set(train["a_cat"]) | set(train["b_cat"])
        imgs_seen = set(train["a_image"]) | set(train["b_image"])
        known = ((test["a_cat"].isin(cats_seen) & test["b_cat"].isin(cats_seen))
                 & test["a_image"].isin(imgs_seen) & test["b_image"].isin(imgs_seen)).to_numpy()
        prob = np.where(known, prob, transfer.predict(test))
        routing = {
            "structure": "routed",
            "gate": "both cats and both photos present in training",
            "share_rows_using_joint": round(float(known.mean()), 3),
            "transfer_spec": TRANSFER_SPEC.describe(),
        }

    sub = test.copy()
    sub["prob_a"] = np.round(np.clip(prob, 0.0, 1.0), 4)
    sub["choice"] = np.where(sub["prob_a"] >= 0.5, "A", "B")
    out = paths.SUBMISSIONS_DIR / ("pred_routed_test.csv" if args.structure == "routed" else "pred_hybrid_test.csv")
    paths.ensure_dirs()
    sub.to_csv(out, index=False)

    ex = pd.read_csv(paths.TEST_EXAMPLE_CSV, encoding="utf-8-sig")
    assert list(sub.columns) == list(test.columns) == list(ex.columns)
    assert len(sub) == len(ex) == 6224
    assert (sub["test_row_id"].to_numpy() == ex["test_row_id"].to_numpy()).all()
    assert sub["prob_a"].between(0, 1).all() and sub["prob_a"].notna().all()
    assert ((sub["prob_a"] >= 0.5) == (sub["choice"] == "A")).all()

    table = slice_table(sub, train)
    print("\n" + table.round(3).to_string())
    print(f"\nsubmission -> {out}")
    if args.structure == "routed":
        print(f"  joint model used for {routing['share_rows_using_joint']:.1%} of rows; "
              f"the rest use {TRANSFER_SPEC.describe()}")

    info = {
        "spec": args.spec,
        "blocks": spec.describe(),
        "C": spec.C,
        "design_columns": len(coef),
        "n_cats_in_fit": len(model.cat_index),
        "side_log_odds": round(float(coef["side"]), 4),
        "feature_tag": feats.meta.get("tag"),
        "routing": routing,
        "fold_check": summary,
        "test_slices": table.reset_index().astype(object).to_dict(orient="records"),
        "submission": str(out.relative_to(paths.ROOT)),
    }
    (paths.FEATURES_DIR / "hybrid_fit.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"summary    -> {paths.FEATURES_DIR / 'hybrid_fit.json'}")


if __name__ == "__main__":
    main()
