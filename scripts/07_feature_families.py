#!/usr/bin/env python3
"""Which image-feature family actually helps predict different observers' choices?

Motivation
----------
The end goal is not "describe a cat photo" but "predict whether a given observer will
choose A or B".  A feature family is therefore judged on preference metrics, not on how
rich its description looks:

* population log loss / Brier / accuracy (probability quality matters, not just the sign);
* **per-observer-type** log loss (the 70 demographic profiles) — a family that only works
  for some observer types is not a good general feature;
* **cross-observer consistency**: on pairs answered twice by different people, how often
  do the answers agree (an upper bound on what any pair-level model can learn), and does
  the model pick the side the two observers agreed on;
* **calibration** (ECE, recalibration slope) — downstream decisions use `prob_a`, not just
  the argmax;
* **cold start** (both photos unseen) — 25 % of the test set looks like this.

Feature families compared (all on the same folds, same estimator family)
------------------------------------------------------------------------
    observer only   26 demographic one-hot columns
    labels (colour) approach A's fur-colour shares (copied reader)
    labels (all)    approach A's full label block (colour, coat, pose, breed, shot)
    hand            20 low-level statistics (this project)
    embedding       576-d frozen ImageNet embedding (this project)
    embedding+hand  the two appearance blocks
    cat memory      per-cat strengths (copied from approach A)
    cat + embedding / + hand / + labels   hybrids

Outputs
    data/features/feature_families.json
    data/features/feature_families.md
    data/features/feature_families.png

Usage
    python scripts/07_feature_families.py
    python scripts/07_feature_families.py --no-coldstart    # skip the image-holdout protocol
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
from sklearn.model_selection import KFold  # noqa: E402

from catarena import features, hybrid, index, paths  # noqa: E402
from catarena.approach_a import labels as a_labels  # noqa: E402
from catarena.approach_a import metrics as a_metrics  # noqa: E402
from catarena.approach_a import strength as a_strength  # noqa: E402

PROFILE_COLS = ["gender", "pet_history", "pet_care", "pet_contact", "future_pet", "mbti_ie"]
FAMILIES = {
    "observer only": None,                       # handled separately (row-level demographics)
    "labels (colour)": hybrid.HybridSpec(cat=False, embedding=False,
                                         label_groups=("colour_shares",), C=1.0),
    "labels (all)": hybrid.HybridSpec(cat=False, embedding=False,
                                      label_groups=("colour_shares", "coat_labels", "pose", "breed", "shot"),
                                      C=1.0),
    "hand": hybrid.HybridSpec(cat=False, embedding=False, hand=True, C=1.0),
    "embedding": hybrid.HybridSpec(cat=False, embedding=True, C=1.0),
    "embedding+hand": hybrid.HybridSpec(cat=False, embedding=True, hand=True, C=1.0),
    "cat memory": hybrid.HybridSpec(cat=True, embedding=False, C=1.0),
    "cat + embedding": hybrid.HybridSpec(cat=True, embedding=True, C=1.0),
    "cat + embedding + hand": hybrid.HybridSpec(cat=True, embedding=True, hand=True, C=1.0),
    "cat + embedding + labels": hybrid.HybridSpec(cat=True, embedding=True,
                                                  label_groups=("colour_shares",), C=1.0),
}


# --------------------------------------------------------------------------- #
# fitting helpers
# --------------------------------------------------------------------------- #
def side(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))]) if X.shape[1] else np.ones((len(X), 1))


def fit_predict(name: str, feats, photo_labels, fit_df, y_fit, pred_df) -> np.ndarray:
    if name == "observer only":
        X_fit = side(features.demo_onehot(fit_df))
        X_pred = side(features.demo_onehot(pred_df))
        clf = LogisticRegression(C=1.0, fit_intercept=False, solver="lbfgs", max_iter=800)
        clf.fit(X_fit, y_fit)
        return clf.predict_proba(X_pred)[:, list(clf.classes_).index(1)]

    spec = FAMILIES[name]
    model = hybrid.HybridModel(spec, feats, photo_labels).fit(fit_df, y_fit)
    return model.predict(pred_df)


def oof(df, y, masks, feats, photo_labels, names) -> dict[str, np.ndarray]:
    preds = {n: np.empty(len(df)) for n in names}
    for i, mask in enumerate(masks):
        fit_df = df.loc[~mask]
        y_fit = y[~mask]
        for n in names:
            preds[n][mask] = fit_predict(n, feats, photo_labels, fit_df, y_fit, df.loc[mask])
        print(f"    fold {i + 1}/{len(masks)}", flush=True)
    return preds


# --------------------------------------------------------------------------- #
# observer-oriented metrics
# --------------------------------------------------------------------------- #
def profile_key(df: pd.DataFrame) -> np.ndarray:
    cols = [df[c].astype("string").fillna("missing") if c in df else pd.Series(["missing"] * len(df))
            for c in PROFILE_COLS]
    return pd.DataFrame({c: v for c, v in zip(PROFILE_COLS, cols)}).agg("|".join, axis=1).to_numpy()


def per_profile(y, p, profiles) -> dict:
    rows = []
    for prof, idx in pd.Series(range(len(y))).groupby(profiles):
        ii = idx.to_numpy()
        if len(ii) < 40:
            continue
        base = float(y[ii].mean())
        rows.append({"profile": prof, "n": int(len(ii)),
                     "logloss": a_metrics.log_loss(y[ii], p[ii]),
                     "baseline": a_metrics.log_loss(y[ii], np.full(len(ii), base)),
                     "accuracy": a_metrics.accuracy(y[ii], p[ii])})
    t = pd.DataFrame(rows).sort_values("logloss")
    # gain over the profile's own constant predictor: removes the part of the
    # spread that comes from some observer types simply being near 50/50
    t["gain"] = t["baseline"] - t["logloss"]
    return {
        "n_profiles": int(len(t)),
        "mean": round(float(t["logloss"].mean()), 4),
        "sd": round(float(t["logloss"].std()), 4),
        "p10": round(float(t["logloss"].quantile(0.10)), 4),
        "p90": round(float(t["logloss"].quantile(0.90)), 4),
        "best": {"profile": t.iloc[0]["profile"], "logloss": round(float(t.iloc[0]["logloss"]), 4)},
        "worst": {"profile": t.iloc[-1]["profile"], "logloss": round(float(t.iloc[-1]["logloss"]), 4)},
        "spread": round(float(t.iloc[-1]["logloss"] - t.iloc[0]["logloss"]), 4),
        "gain_mean": round(float(t["gain"].mean()), 4),
        "gain_sd": round(float(t["gain"].std()), 4),
        "gain_min": round(float(t["gain"].min()), 4),
        "share_profiles_helped": round(float((t["gain"] > 0).mean()), 3),
    }


def calibration(y, p, bins: int = 10) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(acc - conf)
        rows.append({"bin": b, "n": int(m.sum()), "mean_p": round(conf, 3), "frac_a": round(acc, 3)})
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit, y)
    return {"ece": round(float(ece), 4),
            "slope": round(float(clf.coef_[0, 0]), 3),
            "intercept": round(float(clf.intercept_[0]), 3),
            "bins": rows}


def consensus(df: pd.DataFrame, y, p) -> dict:
    """Cross-observer check on ordered pairs answered twice by different people."""
    pair = (df["a_image"].astype(str) + "||" + df["b_image"].astype(str)).to_numpy()
    tbl = pd.DataFrame({"pair": pair, "y": y, "p": p})
    counts = tbl.groupby("pair")["y"].size()
    repeated = tbl[tbl["pair"].isin(counts[counts == 2].index)]
    if repeated.empty:
        return {}
    agree_mask = repeated.groupby("pair")["y"].nunique().eq(1)
    agreed_pairs = set(agree_mask[agree_mask].index)
    disagree_pairs = set(agree_mask[~agree_mask].index)

    def acc_for(pairs):
        sub = repeated[repeated["pair"].isin(pairs)]
        if sub.empty:
            return None
        # both rows of a pair share the features, so one prediction per pair
        first = sub.groupby("pair").first()
        return float(((first["p"] >= 0.5).astype(int) == first["y"]).mean())

    agree = acc_for(agreed_pairs)
    disagree = acc_for(disagree_pairs)
    return {
        "repeated_pairs": int(len(agree_mask)),
        "agreement_rate": round(float(len(agreed_pairs) / max(len(agree_mask), 1)), 4),
        "accuracy_when_observers_agree": None if agree is None else round(agree, 4),
        "accuracy_when_observers_disagree": None if disagree is None else round(disagree, 4),
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def to_markdown(res: dict) -> str:
    order = list(FAMILIES)
    lines = ["# Which feature family helps predict different observers?", "",
             f"Pair-grouped 5-fold over all {res['setup']['n_rows']:,} rows; "
             f"L2 logistic regression with a side column, `C = 1`; "
             f"{res['setup']['n_profiles']} observer profiles "
             f"(6 demographic fields, `missing` as its own level).", "",
             "## Population-level and per-observer-type scores", "",
             "| feature family | log loss | Brier | accuracy | per-profile improvement over that profile's constant (mean ± sd) | profiles helped | ECE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in order:
        m = res["pair_folds"][name]
        pf = res["per_profile"][name]
        lines.append(
            f"| {name} | {m['logloss']:.4f} | {m['brier']:.4f} | {m['accuracy']:.3f} "
            f"| +{pf['gain_mean']:.4f} ± {pf['gain_sd']:.3f} "
            f"| {pf['share_profiles_helped']:.0%} "
            f"| {res['calibration'][name]['ece']:.3f} |"
        )
    lines += ["", "## Cross-observer consistency (ordered pairs answered twice)", "",
              "| feature family | agreement between the two observers | accuracy on agreed pairs | accuracy on disputed pairs |",
              "|---|---:|---:|---:|"]
    for name in order:
        c = res["consensus"][name]
        if not c:
            continue
        lines.append(f"| {name} | {c['agreement_rate']:.3f} | "
                     f"{c['accuracy_when_observers_agree']} | {c['accuracy_when_observers_disagree']} |")
    for key, title in (("image_holdout", "Cold start 1: both photos unseen"),
                       ("cat_holdout", "Cold start 2: both cats unseen")):
        if res.get(key):
            lines += ["", f"## {title}", "", "| feature family | log loss | accuracy |", "|---|---:|---:|"]
            for name in order:
                m = res[key][name]
                lines.append(f"| {name} | {m['logloss']:.4f} | {m['accuracy']:.3f} |")
    return "\n".join(lines) + "\n"


def plot(res: dict, dest: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(FAMILIES, key=lambda n: res["pair_folds"][n]["logloss"])
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0),
                             gridspec_kw={"width_ratios": [1.3, 1, 1]})

    ax = axes[0]
    vals = [res["pair_folds"][n]["logloss"] for n in order]
    colors = ["#4c78a8" if "cat" not in n else "#d62728" for n in order]
    ax.barh(np.arange(len(order)), vals, color=colors)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.axvline(0.6925, ls="--", lw=1, color="k", alpha=0.7)
    ax.set_xlim(0.64, 0.70)
    ax.invert_yaxis()
    ax.set_xlabel("log loss (lower is better)")
    ax.set_title("Preference prediction, pair-grouped 5-fold")

    ax = axes[1]
    means = [res["per_profile"][n]["mean"] for n in order]
    sds = [res["per_profile"][n]["sd"] for n in order]
    ax.barh(np.arange(len(order)), means, xerr=sds, color="#7f7f7f", capsize=2)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(0.64, 0.70)
    ax.set_xlabel("mean log loss across 70 observer profiles")
    ax.set_title("Does it work for every observer type?")

    ax = axes[2]
    ypos = np.arange(len(order))
    if res.get("image_holdout"):
        ax.barh(ypos - 0.2, [res["image_holdout"][n]["logloss"] for n in order],
                0.4, color="#2ca02c", label="both photos unseen")
    if res.get("cat_holdout"):
        ax.barh(ypos + 0.2, [res["cat_holdout"][n]["logloss"] for n in order],
                0.4, color="#9467bd", label="both cats unseen")
    ax.set_yticks(ypos)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.axvline(0.6925, ls="--", lw=1, color="k", alpha=0.7)
    ax.set_xlim(0.64, 0.72)
    ax.set_xlabel("log loss under cold start")
    ax.set_title("Cold start (two kinds)")
    ax.legend(fontsize=7, frameon=True, framealpha=0.95, loc="lower right")

    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--p2-folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-coldstart", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw the figure from the saved JSON only")
    args = ap.parse_args()

    if args.replot:
        dest = paths.FEATURES_DIR / "feature_families.png"
        plot(json.loads((paths.FEATURES_DIR / "feature_families.json").read_text(encoding="utf-8")), dest)
        print(f"redrawn -> {dest}")
        return

    train, _ = index.load_train(), index.load_test()
    train = a_strength.ensure_pair_columns(train)
    y = features.label_a(train)
    feats = features.load_features(args.tag, kind="deep")
    photo_labels = a_labels.load_photo_labels()
    names = list(FAMILIES)
    profiles = profile_key(train)
    print(f"rows {len(train):,} | pairs {train['unordered_pair_id'].nunique():,} | "
          f"observer profiles {len(set(profiles))} | features {feats.meta.get('tag')}")

    masks = a_strength.pair_masks(train, args.folds, seed=args.seed)
    print("\n[pair-grouped folds]")
    preds = oof(train, y, masks, feats, photo_labels, names)

    res = {
        "setup": {
            "n_rows": int(len(train)),
            "n_pairs": int(train["unordered_pair_id"].nunique()),
            "n_profiles": int(len(set(profiles))),
            "folds": args.folds,
            "seed": args.seed,
            "feature_tag": feats.meta.get("tag"),
            "estimator": "LogisticRegression(C=1, fit_intercept=False, side column)",
            "label_source": "approach A's prepared/photo_labels.csv (YOLO11n-seg + CLIP)",
        },
        "pair_folds": {},
        "per_profile": {},
        "calibration": {},
        "consensus": {},
    }
    for n in names:
        res["pair_folds"][n] = a_metrics.metrics(y, preds[n])
        res["per_profile"][n] = per_profile(y, preds[n], profiles)
        res["calibration"][n] = calibration(y, preds[n])
        res["consensus"][n] = consensus(train, y, preds[n])
        pf = res["per_profile"][n]
        print(f"  {n:26s} logloss {res['pair_folds'][n]['logloss']:.4f} "
              f"| per-profile {pf['mean']:.4f}±{pf['sd']:.3f} "
              f"| ECE {res['calibration'][n]['ece']:.3f} "
              f"| consensus acc {res['consensus'][n].get('accuracy_when_observers_agree')}")

    if not args.no_coldstart:
        # --- cold start 1: both photos unseen (cats usually still known) ----- #
        imgs = np.array(sorted(set(train["a_image"]) | set(train["b_image"])))
        res["image_holdout"] = {}
        preds_cold = {n: np.full(len(train), np.nan) for n in names}
        print("\n[image-holdout folds: both photos unseen]")
        for f, (_, held_idx) in enumerate(KFold(n_splits=args.p2_folds, shuffle=True,
                                                random_state=args.seed).split(imgs)):
            held = set(imgs[held_idx])
            a_held = train["a_image"].isin(held).to_numpy()
            b_held = train["b_image"].isin(held).to_numpy()
            ev, fit = a_held & b_held, ~a_held & ~b_held
            fit_df, ev_df = train[fit], train[ev]
            y_fit, y_ev = y[fit], y[ev]
            for n in names:
                p = fit_predict(n, feats, photo_labels, fit_df, y_fit, ev_df)
                preds_cold[n][ev] = p
            print(f"    fold {f + 1}/{args.p2_folds}: fit {fit.sum():,}, eval {ev.sum():,}", flush=True)
        for n in names:
            m = np.isfinite(preds_cold[n])
            res["image_holdout"][n] = a_metrics.metrics(y[m], preds_cold[n][m])

        # --- cold start 2: both cats unseen (the hardest slice) ------------- #
        cats = np.array(sorted(set(train["a_cat"]) | set(train["b_cat"])))
        res["cat_holdout"] = {}
        preds_newcat = {n: np.full(len(train), np.nan) for n in names}
        between = train["comparison_type"].eq("between_cats").to_numpy()
        print("\n[cat-holdout folds: both cats unseen]")
        for f, (_, held_idx) in enumerate(KFold(n_splits=args.p2_folds, shuffle=True,
                                                random_state=args.seed + 1).split(cats)):
            held = set(cats[held_idx])
            a_new = train["a_cat"].isin(held).to_numpy()
            b_new = train["b_cat"].isin(held).to_numpy()
            ev = between & a_new & b_new
            fit = ~a_new & ~b_new
            fit_df, ev_df = train[fit], train[ev]
            y_fit = y[fit]
            for n in names:
                preds_newcat[n][ev] = fit_predict(n, feats, photo_labels, fit_df, y_fit, ev_df)
            print(f"    fold {f + 1}/{args.p2_folds}: fit {fit.sum():,}, eval {ev.sum():,}", flush=True)
        for n in names:
            m = np.isfinite(preds_newcat[n])
            res["cat_holdout"][n] = a_metrics.metrics(y[m], preds_newcat[n][m])

    dest = paths.FEATURES_DIR / "feature_families.json"
    dest.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    md = paths.FEATURES_DIR / "feature_families.md"
    md.write_text(to_markdown(res), encoding="utf-8")
    plot(res, paths.FEATURES_DIR / "feature_families.png")
    print(f"\nwritten -> {dest}\n         -> {md}\n         -> {paths.FEATURES_DIR / 'feature_families.png'}")


if __name__ == "__main__":
    main()
