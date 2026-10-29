#!/usr/bin/env python3
"""Close the hybrid's cold-start gap: better features vs a routing structure.

Problem from `scripts/07_feature_families.py`:

    cat memory + embedding + hand  ->  best average 0.6565
                                       both-cats-unseen 0.6945
    median of all families there                    0.6892

Diagnosis: in a single joint fit the cat dummies absorb cat-level variance, so the
appearance weights only explain the residual and transfer badly to a cat (or photo)
that was never seen. Two remedies are tested here, one per axis:

1. **Features** — richer appearance blocks:
     emb                576-d MobileNetV3-Small
     emb+hand+colour    20 statistics + fur-colour shares
     multi-backbone     MobileNetV3-Small (+) ResNet18 concatenation (1088-d)
2. **Structure** — do not force one model to do both jobs. Keep the strong joint model
   for rows whose cat *and* photos were seen, and fall back to a transfer model
   (coarse labels, or appearance-only) for genuinely new items.

Acceptance: average stays at the current best (<= 0.6565) **and** both-cats-unseen
drops to the family median or below (<= 0.6892).

Outputs
    artifacts/features/cold_start_fix.json
    artifacts/features/cold_start_fix.md
    artifacts/submissions/pred_routed_test.csv   (winner fitted on all rows)

Usage
    python scripts/08_cold_start_fix.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from model import features, hybrid, index, paths  # noqa: E402
from model.labels import labels as label_mod  # noqa: E402
from model.labels import metrics as pair_metrics  # noqa: E402
from model.labels import strength as item_strength  # noqa: E402

BASELINE_AVG = 0.65652        # cat + embedding + hand, pair-grouped 5-fold (07)
BASELINE_CAT_COLD = 0.69446   # same model, both cats unseen (07)
MEDIAN_CAT_COLD = 0.68922     # median of the ten families on that slice (07)
MEDIAN_PHOTO_COLD = 0.68370   # median of the ten families, both photos unseen (07)


# --------------------------------------------------------------------------- #
# configurations
# --------------------------------------------------------------------------- #
def configs(feats_single, feats_multi):
    """name -> (spec, feature cache)."""
    return {
        "joint: cat+emb+hand": (hybrid.HybridSpec(cat=True, embedding=True, hand=True, C=1.0), feats_single),
        "joint: cat+multi+hand+colour": (hybrid.HybridSpec(
            cat=True, embedding=True, hand=True, label_groups=("colour_shares",), C=1.0), feats_multi),
        "labels colour": (hybrid.HybridSpec(cat=False, embedding=False,
                                            label_groups=("colour_shares",), C=1.0), feats_single),
        "app: emb+hand+colour": (hybrid.HybridSpec(
            cat=False, embedding=True, hand=True, label_groups=("colour_shares",), C=1.0), feats_single),
        "app: multi+hand+colour": (hybrid.HybridSpec(
            cat=False, embedding=True, hand=True, label_groups=("colour_shares",), C=1.0), feats_multi),
    }


def fold_masks(train, protocol: str, k: int, seed: int):
    if protocol == "pair":
        for m in item_strength.pair_masks(train, k, seed=seed):
            yield ~m, m
    elif protocol == "photo":
        imgs = np.array(sorted(set(train["a_image"]) | set(train["b_image"])))
        for _, held in KFold(n_splits=k, shuffle=True, random_state=seed).split(imgs):
            held = set(imgs[held])
            a, b = train["a_image"].isin(held).to_numpy(), train["b_image"].isin(held).to_numpy()
            yield ~a & ~b, a & b
    elif protocol == "cat":
        cats = np.array(sorted(set(train["a_cat"]) | set(train["b_cat"])))
        between = train["comparison_type"].eq("between_cats").to_numpy()
        for _, held in KFold(n_splits=k, shuffle=True, random_state=seed + 1).split(cats):
            held = set(cats[held])
            a, b = train["a_cat"].isin(held).to_numpy(), train["b_cat"].isin(held).to_numpy()
            yield ~a & ~b, between & a & b
    else:
        raise ValueError(protocol)


def run_protocol(train, y, protocol, k, seed, cfgs, photo_labels) -> dict:
    """Fit every config on every fold; also record what was known at prediction time."""
    n = len(train)
    preds = {name: np.full(n, np.nan) for name in cfgs}
    known_cat = np.zeros(n, dtype=bool)
    known_photo = np.zeros(n, dtype=bool)

    for fit_mask, ev_mask in fold_masks(train, protocol, k, seed):
        fit_df, ev_df = train[fit_mask], train[ev_mask]
        y_fit = y[fit_mask]
        cats_seen = set(fit_df["a_cat"]) | set(fit_df["b_cat"])
        imgs_seen = set(fit_df["a_image"]) | set(fit_df["b_image"])
        known_cat[ev_mask] = (ev_df["a_cat"].isin(cats_seen) & ev_df["b_cat"].isin(cats_seen)).to_numpy()
        known_photo[ev_mask] = (ev_df["a_image"].isin(imgs_seen) & ev_df["b_image"].isin(imgs_seen)).to_numpy()
        for name, (spec, feats) in cfgs.items():
            model = hybrid.HybridModel(spec, feats, photo_labels).fit(fit_df, y_fit)
            preds[name][ev_mask] = model.predict(ev_df)
        print(f"    {protocol} fold done", flush=True)
    return {"preds": preds, "known_cat": known_cat, "known_photo": known_photo}


def evaluate(y, preds) -> dict:
    """Score only the rows this protocol actually predicts (the rest stay NaN)."""
    m = np.isfinite(preds)
    return pair_metrics.metrics(y[m], preds[m])


# --------------------------------------------------------------------------- #
def pick_winner(routes):
    """Among routes that meet both criteria, prefer avg, then photo-cold."""
    def meets(m):
        return (m["pair"]["logloss"] <= BASELINE_AVG
                and m["cat"]["logloss"] <= MEDIAN_CAT_COLD)

    winners = {n: m for n, m in routes.items() if meets(m)}
    if not winners:
        return None
    return min(winners, key=lambda n: (winners[n]["pair"]["logloss"],
                                       winners[n]["photo"]["logloss"]))


def build_submission(winner, train, y, cfgs, photo_labels) -> dict:
    fb = winner.split(" else ")[1].rstrip("]")
    gate = "photo" if "cat+photo" in winner else "cat"
    best_joint = winner.split(" if ")[0].replace("route[", "")
    m_joint = hybrid.HybridModel(cfgs[best_joint][0], cfgs[best_joint][1], photo_labels).fit(train, y)
    m_fb = hybrid.HybridModel(cfgs[fb][0], cfgs[fb][1], photo_labels).fit(train, y)
    test = index.load_test()
    cats_seen = set(train["a_cat"]) | set(train["b_cat"])
    imgs_seen = set(train["a_image"]) | set(train["b_image"])
    known_cat = (test["a_cat"].isin(cats_seen) & test["b_cat"].isin(cats_seen)).to_numpy()
    known_photo = (test["a_image"].isin(imgs_seen) & test["b_image"].isin(imgs_seen)).to_numpy()
    known = known_photo if gate == "photo" else known_cat
    p = np.where(known, m_joint.predict(test), m_fb.predict(test))

    sub = test.copy()
    sub["prob_a"] = np.round(np.clip(p, 0, 1), 4)
    sub["choice"] = np.where(sub["prob_a"] >= 0.5, "A", "B")
    out = paths.SUBMISSIONS_DIR / "pred_routed_test.csv"
    sub.to_csv(out, index=False)
    ex = pd.read_csv(paths.TEST_EXAMPLE_CSV, encoding="utf-8-sig")
    assert list(sub.columns) == list(ex.columns) and len(sub) == len(ex) == 6224
    assert sub["prob_a"].between(0, 1).all()
    return {"submission": str(out.relative_to(paths.ROOT)),
            "share_rows_using_joint": round(float(known.mean()), 3)}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cold-folds", type=int, default=3)
    ap.add_argument("--reuse", action="store_true",
                    help="skip the experiment, re-emit the submission for the recorded winner")
    args = ap.parse_args()

    train = item_strength.ensure_pair_columns(index.load_train())
    y = features.label_a(train)
    photo_labels = label_mod.load_photo_labels()

    combined = "mnv3s_resnet18"
    if combined not in features.available_tags():
        features.combine_caches(["mobilenet_v3_small_224", "resnet18_224"], combined)
        print(f"built combined cache {combined}")
    feats_single = features.load_features("mobilenet_v3_small_224")
    feats_multi = features.load_features(combined)
    cfgs = configs(feats_single, feats_multi)

    if args.reuse:
        src = paths.FEATURES_DIR / "cold_start_fix.json"
        result = json.loads(src.read_text(encoding="utf-8"))
        winner = pick_winner(result["routes"])
        info = build_submission(winner, train, y, cfgs, photo_labels)
        result["winner"] = winner
        result.update(info)
        src.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"winner: {winner}")
        print(f"submission -> {info['submission']} "
              f"(joint for {info['share_rows_using_joint']:.1%} of test rows)")
        return

    runs = {}
    for protocol, k in (("pair", args.folds), ("photo", args.cold_folds), ("cat", args.cold_folds)):
        print(f"\n[{protocol} protocol]")
        runs[protocol] = run_protocol(train, y, protocol, k, args.seed, cfgs, photo_labels)

    single = {}
    for name in cfgs:
        single[name] = {
            "pair": evaluate(y, runs["pair"]["preds"][name]),
            "photo_cold": evaluate(y, runs["photo"]["preds"][name]),
            "cat_cold": evaluate(y, runs["cat"]["preds"][name]),
        }
        print(f"  {name:30s} pair {single[name]['pair']['logloss']:.4f} "
              f"| photo-cold {single[name]['photo_cold']['logloss']:.4f} "
              f"| cat-cold {single[name]['cat_cold']['logloss']:.4f}")

    best_joint = min((n for n in cfgs if n.startswith("joint")),
                     key=lambda n: single[n]["pair"]["logloss"])
    fallbacks = [n for n in cfgs if not n.startswith("joint") and n in single]
    routes = {}
    for fb in fallbacks:
        for gate, key in (("cat", "known_cat"), ("cat+photo", "known_photo")):
            name = f"route[{best_joint} if {gate} known else {fb}]"
            routes[name] = {}
            for protocol in ("pair", "photo", "cat"):
                p = np.where(runs[protocol][key], runs[protocol]["preds"][best_joint],
                             runs[protocol]["preds"][fb])
                routes[name][protocol] = evaluate(y, p)

    def meets(m):
        return (m["pair"]["logloss"] <= BASELINE_AVG
                and m["cat"]["logloss"] <= MEDIAN_CAT_COLD)

    print(f"\n  best joint: {best_joint}")
    for name, m in sorted(routes.items(), key=lambda kv: kv[1]["pair"]["logloss"]):
        flag = "  <-- meets both criteria" if meets(m) else ""
        print(f"  {name}\n      pair {m['pair']['logloss']:.4f} | photo-cold {m['photo']['logloss']:.4f} "
              f"| cat-cold {m['cat']['logloss']:.4f}{flag}")

    winner = pick_winner(routes)
    print(f"\n  winner: {winner}")

    result = {
        "baseline": {"avg": BASELINE_AVG, "cat_cold": BASELINE_CAT_COLD},
        "targets": {"median_cat_cold": MEDIAN_CAT_COLD, "median_photo_cold": MEDIAN_PHOTO_COLD},
        "single_models": single,
        "routes": routes,
        "best_joint": best_joint,
        "winner": winner,
        "seed": args.seed,
    }

    if winner:
        info = build_submission(winner, train, y, cfgs, photo_labels)
        result.update(info)
        print(f"\nsubmission -> {info['submission']} "
              f"(joint for {info['share_rows_using_joint']:.1%} of test rows)")

    (paths.FEATURES_DIR / "cold_start_fix.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Closing the hybrid's cold-start gap", "",
             f"Criteria: average ≤ **{BASELINE_AVG:.4f}** and both-cats-unseen ≤ **{MEDIAN_CAT_COLD:.4f}** "
             f"(the family median).", "",
             "## Single models", "",
             "| model | avg | both photos unseen | both cats unseen |", "|---|---:|---:|---:|"]
    for name, m in single.items():
        lines.append(f"| {name} | {m['pair']['logloss']:.4f} | {m['photo_cold']['logloss']:.4f} "
                     f"| {m['cat_cold']['logloss']:.4f} |")
    lines += ["", "## Routed models", "",
              "| routing | avg | both photos unseen | both cats unseen | meets both |", "|---|---:|---:|---:|:--:|"]
    for name, m in routes.items():
        lines.append(f"| {name} | {m['pair']['logloss']:.4f} | {m['photo']['logloss']:.4f} "
                     f"| {m['cat']['logloss']:.4f} | {'yes' if meets(m) else 'no'} |")
    (paths.FEATURES_DIR / "cold_start_fix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"json -> {paths.FEATURES_DIR / 'cold_start_fix.json'}")
    print(f"md   -> {paths.FEATURES_DIR / 'cold_start_fix.md'}")


if __name__ == "__main__":
    main()
