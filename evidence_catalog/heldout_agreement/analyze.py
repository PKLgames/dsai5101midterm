"""Post-test repeated-pair agreement and assumption-dependent oracle comparisons."""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from evidence_catalog.photo_demographic_support.analyze import digest, write_csv, write_json  # noqa: E402

SOURCES = {
    "G2": [("original", "artifacts/test_predictions.parquet"), ("demographic", "demographic_experiment/test_predictions.parquet"),
           ("kernel", "kernel_experiment/g2_predictions.parquet"), ("cat_cnn", "cat_cnn_experiment/g2_predictions.parquet")],
    "G3": [("original", "g3_experiment/test_predictions.parquet"), ("demographic", "demographic_experiment/diagnostics_extension/g3_predictions.parquet"),
           ("kernel", "kernel_experiment/g3_predictions.parquet"), ("cat_cnn", "cat_cnn_experiment/g3_predictions.parquet")],
    "G4": [("original", "g4_experiment/test_predictions.parquet"), ("demographic", "demographic_experiment/diagnostics_extension/g4_predictions.parquet"),
           ("kernel", "kernel_experiment/g4_predictions.parquet")],
}


def snapshot():
    result = {}
    for directory, subdirs, names in os.walk(ROOT):
        base = Path(directory)
        if base == ROOT:
            subdirs[:] = [n for n in subdirs if n not in {".git", ".venv", ".cache"}]
        subdirs[:] = [n for n in subdirs if n not in {"__pycache__", "matplotlib_config"} and base/n != HERE]
        for name in names:
            path = base/name
            if path.is_file():
                result[str(path.relative_to(ROOT))] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return result


def group_data(frame, level):
    if level == "photo_pair":
        part = frame.copy()
        part["group_key"] = part["pair_id"]
        part["canonical_y"] = np.where(part.orientation == 1, part.outcome, 1-part.outcome)
    elif level == "cat_pair":
        part = frame.loc[frame.a_cat != frame.b_cat].copy()
        part["group_key"] = [json.dumps(sorted([a, b]), separators=(",", ":")) for a, b in zip(part.a_cat, part.b_cat)]
        part["canonical_y"] = np.where(part.a_cat < part.b_cat, part.outcome, 1-part.outcome)
    else:
        raise ValueError(level)
    aggregate = part.groupby("group_key", sort=True)["canonical_y"].agg(["size", "sum"])
    aggregate.columns = ["n", "k"]
    aggregate = aggregate.loc[aggregate.n >= 2].copy()
    if aggregate.empty:
        raise ValueError("No repeated groups.")
    aggregate["disagreement"] = 2*aggregate.k*(aggregate.n-aggregate.k)/(aggregate.n*(aggregate.n-1))
    aggregate["noise"] = aggregate.disagreement/2
    aggregate["loo_brier"] = aggregate.n/(aggregate.n-1)*aggregate.noise
    aggregate["in_sample_brier"] = aggregate.k*(aggregate.n-aggregate.k)/aggregate.n**2
    aggregate["response_agree"] = 1-aggregate.disagreement
    aggregate["unordered_comparisons"] = aggregate.n*(aggregate.n-1)/2
    return part, aggregate


def describe_groups(part, group):
    n = group.n.to_numpy(dtype=float)
    d = group.disagreement.to_numpy(dtype=float)
    noise = np.average(d/2, weights=n)
    return {"scope_responses": len(part), "scope_groups": int(part.group_key.nunique()),
            "repeated_responses": int(n.sum()), "repeated_groups": len(group),
            "repeat_response_coverage": float(n.sum()/len(part)),
            "median_responses_per_repeated_group": float(np.median(n)), "maximum_responses_per_group": int(n.max()),
            "agreement_response_weighted": float(np.average(1-d, weights=n)),
            "agreement_group_weighted": float(np.mean(1-d)),
            "agreement_response_pairs_weighted": float(np.average(1-d, weights=group.unordered_comparisons)),
            "estimated_oracle_brier": float(noise), "loo_brier": float(np.average(group.loo_brier, weights=n)),
            "in_sample_brier_optimistic": float(np.average(group.in_sample_brier, weights=n)),
            "plugin_accuracy_lower_bound": float(1-2*noise),
            "plugin_accuracy_upper_bound": float(.5+np.sqrt(max(0, .25-noise))) if noise <= .25 else None}


def bootstrap(group, errors, reps=3000, seed=23801):
    """Cluster bootstrap: sample observed photo-pair or cat-pair groups with replacement."""
    rng = np.random.default_rng(seed)
    n = group.n.to_numpy(dtype=float)
    noise_sums = n*group.noise.to_numpy()
    groups = len(n)
    result_noise, result_gap = [], []
    for start in range(0, reps, 150):
        size = min(150, reps-start)
        samples = rng.integers(0, groups, size=(size, groups))
        denom = n[samples].sum(axis=1)
        noise = noise_sums[samples].sum(axis=1)/denom
        result_noise.extend(noise)
        if errors is not None:
            result_gap.extend((errors[samples].sum(axis=1)-noise_sums[samples].sum(axis=1))/denom)
    interval = lambda x: [float(v) for v in np.quantile(x, [.025, .975])]
    return interval(result_noise), interval(result_gap) if errors is not None else None


def load():
    structure = pq.read_table(ROOT/"artifacts"/"rows_canonical.parquet",
                              columns=["train_row_id", "pair_id", "orientation", "a_cat", "b_cat"]).to_pandas()
    if structure.train_row_id.duplicated().any() or not structure.orientation.isin([1, -1]).all():
        raise ValueError("Invalid structural identities or orientation.")
    cohorts = {}
    for cohort, sources in SOURCES.items():
        result, reference = [], None
        for source, path in sources:
            table = pq.read_table(ROOT.joinpath(*path.split("/"))).to_pandas()
            table = table.rename(columns={"model_id": "model", "outcome_a": "outcome", "prob_a": "probability"})
            for model, rows in table.groupby("model", sort=True):
                rows = rows[["train_row_id", "pair_id", "outcome", "probability"]].copy()
                if rows.train_row_id.duplicated().any() or not rows.outcome.isin([0, 1]).all():
                    raise ValueError(f"Bad saved predictions: {cohort} {source} {model}")
                if not np.isfinite(rows.probability).all() or not rows.probability.between(0, 1).all():
                    raise ValueError(f"Bad probability: {cohort} {source} {model}")
                rows = rows.merge(structure, on="train_row_id", how="left", validate="one_to_one", suffixes=("", "_structure"))
                if rows.orientation.isna().any() or not rows.pair_id.equals(rows.pair_id_structure):
                    raise ValueError(f"Structural pair mismatch: {cohort} {source} {model}")
                rows = rows.sort_values("train_row_id").reset_index(drop=True)
                identity = rows[["train_row_id", "pair_id", "outcome"]]
                if reference is None:
                    reference = identity
                elif not identity.equals(reference):
                    raise ValueError(f"Unmatched rows/outcomes: {cohort} {source} {model}")
                result.append((source, model, rows))
        cohorts[cohort] = result
    return cohorts


def compute(cohorts, reps=3000):
    agreement, models = [], []
    for cohort, predictions in cohorts.items():
        base = predictions[0][2]
        for level in ("photo_pair", "cat_pair"):
            part, group = group_data(base, level)
            stats = describe_groups(part, group)
            noise_ci, _ = bootstrap(group, None, reps)
            agreement.append({"cohort": cohort, "level": level, **stats,
                              "oracle_brier_ci_low": noise_ci[0], "oracle_brier_ci_high": noise_ci[1]})
            selected = set(part.loc[part.group_key.isin(group.index), "train_row_id"])
            for source, model, frame in predictions:
                scoped = frame if level == "photo_pair" else frame.loc[frame.a_cat != frame.b_cat]
                mp, _ = group_data(frame, level)
                mp = mp.loc[mp.train_row_id.isin(selected)].copy()
                mp["sq_error"] = (mp.probability-mp.outcome)**2
                mp["correct"] = ((mp.probability >= .5) == (mp.outcome == 1)).astype(int)
                grouped_error = mp.groupby("group_key")["sq_error"].sum().reindex(group.index)
                if grouped_error.isna().any() or len(mp) != stats["repeated_responses"]:
                    raise ValueError("Model repeated subset does not match oracle groups.")
                _, ci = bootstrap(group, grouped_error.to_numpy(), reps)
                brier = float(mp.sq_error.mean())
                models.append({"cohort": cohort, "level": level, "source": source, "model": model,
                               "full_scope_responses": len(scoped), "full_scope_brier": float(((scoped.probability-scoped.outcome)**2).mean()),
                               "repeated_responses": len(mp), "repeated_groups": len(group), "repeated_brier": brier,
                               "repeated_accuracy": float(mp.correct.mean()), "estimated_oracle_brier": stats["estimated_oracle_brier"],
                               "model_minus_oracle": brier-stats["estimated_oracle_brier"],
                               "difference_ci_low": ci[0], "difference_ci_high": ci[1]})
    return agreement, models


def plot(agreement, models):
    with tempfile.TemporaryDirectory(prefix="agreement_mpl_") as config:
        os.environ["MPLCONFIGDIR"] = config
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(HERE/"presentation_heldout_agreement.pdf") as pdf:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            for i, level in enumerate(("photo_pair", "cat_pair")):
                rows = [r for r in agreement if r["level"] == level]
                x = np.arange(len(rows))
                axes[i].bar(x, [r["agreement_response_weighted"] for r in rows], color="#326d80")
                axes[i].set_xticks(x, [r["cohort"] for r in rows])
                axes[i].set_ylim(0, 1)
                axes[i].set_title(level.replace("_", " ") + " repeat agreement")
                axes[i].set_ylabel("Response-weighted agreement")
                for j, r in enumerate(rows):
                    axes[i].text(j, r["agreement_response_weighted"]+.02, f"n={r['repeated_responses']}", ha="center", fontsize=9)
            fig.suptitle("Post-test: repeat agreement among observed held-out responses")
            fig.text(.04, .02, "Different cohort/group populations; agreement is not a universal accuracy ceiling.", fontsize=9)
            fig.tight_layout(rect=(0, .06, 1, .95))
            pdf.savefig(fig)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(12, 6))
            rows = [r for r in models if r["cohort"] == "G2" and r["level"] == "photo_pair" and
                    (r["source"], r["model"]) in {("original", "common:M0"), ("original", "common:M1"),
                                                  ("original", "M3-C"), ("original", "M4-C"),
                                                  ("demographic", "Cat+D"), ("kernel", "kernel_M3-C"), ("cat_cnn", "Cat+CNN")}]
            labels = [r["source"] + ": " + r["model"] for r in rows]
            x = np.arange(len(rows))
            ax.bar(x, [r["repeated_brier"] for r in rows], color="#326d80")
            ax.axhline(rows[0]["estimated_oracle_brier"], color="#b65040", linestyle="--", label="Estimated pair-only oracle Brier")
            ax.set_xticks(x, labels, rotation=25, ha="right")
            ax.set_ylabel("Brier score (lower is better)")
            ax.set_title("G2: same repeated-photo-pair subset")
            ax.legend()
            fig.text(.04, .02, "Oracle estimate assumes exchangeable independent repeats; not a universal ceiling or a deployable model.", fontsize=9)
            fig.tight_layout(rect=(0, .07, 1, 1))
            pdf.savefig(fig)
            plt.close(fig)


def run():
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    names = ["original_hashes_before.json", "agreement_summary.csv", "model_comparisons.csv",
             "presentation_heldout_agreement.pdf", "analysis_provenance.json"]
    if any((HERE/name).exists() for name in names):
        raise FileExistsError("Output already exists.")
    before = snapshot()
    write_json(HERE/names[0], before)
    input_paths = ["artifacts/rows_canonical.parquet"] + [path for paths in SOURCES.values() for _, path in paths]
    input_hashes = {path: digest(ROOT.joinpath(*path.split("/"))) for path in input_paths}
    cohorts = load()
    agreement, models = compute(cohorts)
    write_csv(HERE/names[1], agreement, list(agreement[0]))
    write_csv(HERE/names[2], models, list(models[0]))
    plot(agreement, models)
    if snapshot() != before:
        raise RuntimeError("Pre-existing project file changed during analysis.")
    write_json(HERE/names[4], {"status": "post_test_exploratory_complete", "input_sha256": input_hashes,
               "output_sha256": {name: digest(HERE/name) for name in names[1:4]},
               "original_snapshot_sha256": digest(HERE/names[0]), "original_files_unchanged": len(before),
               "bootstrap": {"replicates": 3000, "seed": 23801, "sampling_unit": "photo pair or cat pair, according to level"},
               "models_fitted": False, "heldout_saved_prediction_outcomes_read": True, "raw_outcome_source_read": False,
               "caveats": "Post-test, conditional repeated-group population, dependent respondents unknown; pair-only oracle is not a universal ceiling."})
    print(json.dumps({"agreement": agreement, "model_rows": len(models)}, indent=2))


if __name__ == "__main__":
    run()
