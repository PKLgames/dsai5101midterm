"""Descriptive within-cat analysis of saved, frozen G3 test predictions."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit

HERE = Path(__file__).resolve().parent
G3 = HERE.parent
ROOT = G3.parent
sys.path.insert(0, str(G3))

from common import cluster_bootstrap, digest, load_structure, write_csv, write_json  # noqa: E402
from evaluate import verify_freeze  # noqa: E402

MODELS = ("M0", "M1", "M3-H", "M3-C", "M3-Y", "M4-H", "M4-C", "M4-Y")
EXCLUDED_ROOT = {".git", ".venv", ".cache", "Presentation_Evidence"}


def snapshot(root: Path, excluded_root: set[str], excluded_subtree: Path | None = None) -> dict[str, dict[str, Any]]:
    entries = {}
    for directory, subdirs, files in os.walk(root):
        base = Path(directory)
        if base == root:
            subdirs[:] = [name for name in subdirs if name not in excluded_root]
        if excluded_subtree is not None:
            subdirs[:] = [name for name in subdirs if base / name != excluded_subtree]
        for name in files:
            path = base / name
            if path.is_file() and (base != root or name not in excluded_root):
                entries[str(path.relative_to(root))] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return entries


def verify_inputs() -> dict[str, Any]:
    freeze = verify_freeze()
    provenance = json.loads((G3 / "evaluation_provenance.json").read_text(encoding="utf-8"))
    if provenance["status"] != "completed_once_post_freeze_retrospective_exploratory":
        raise ValueError("G3 evaluation provenance is not complete.")
    if provenance["freeze_sha256"] != digest(G3 / "freeze.json"):
        raise ValueError("G3 freeze does not match the evaluation provenance.")
    if digest(G3 / "evaluation_started.json") != provenance["test_access_record_sha256"]:
        raise ValueError("G3 test access record hash mismatch.")
    for name, expected in provenance["output_sha256"].items():
        if digest(G3 / name) != expected:
            raise ValueError(f"G3 output hash mismatch: {name}")
    return freeze


def align_predictions(structure: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index = {row["train_row_id"]: row for row in structure}
    if len(index) != len(structure) or not index:
        raise ValueError("G3 structural response IDs must be unique and nonempty.")
    by_model = {name: {} for name in MODELS}
    for row in records:
        name, key = row["model"], row["train_row_id"]
        if name not in by_model or key not in index or key in by_model[name]:
            raise ValueError("Unexpected or duplicate saved G3 prediction.")
        expected = index[key]
        if row["pair_id"] != expected["pair_id"] or row["held_out_photo"] != expected["held_out_photo"]:
            raise ValueError("Saved prediction does not match the structural row.")
        if row["outcome_a"] not in (0, 1):
            raise ValueError("Invalid saved binary outcome.")
        logit = float(row["logit"])
        p, loss = float(row["prob_a"]), float(row["loss"])
        recomputed = row["outcome_a"] * np.logaddexp(0., -logit) + (1-row["outcome_a"]) * np.logaddexp(0., logit)
        if not np.isfinite([logit, p, loss]).all() or abs(p - expit(logit)) > 1e-12 or abs(loss - recomputed) > 1e-12:
            raise ValueError("Saved prediction, logit, and loss do not reconcile.")
        by_model[name][key] = row
    order = [row["train_row_id"] for row in structure]
    aligned = {}
    for name in MODELS:
        if set(by_model[name]) != set(order):
            raise ValueError(f"Missing predictions for {name}.")
        aligned[name] = [by_model[name][key] for key in order]
    reference = [row["outcome_a"] for row in aligned["M0"]]
    if any([row["outcome_a"] for row in aligned[name]] != reference for name in MODELS):
        raise ValueError("Saved outcomes differ across matched models.")
    return aligned


def calculate(structure: list[dict[str, Any]], aligned: dict[str, list[dict[str, Any]]],
              selected: str, m1_intercept: float, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    for row in structure:
        if (row["a_cat"] == row["b_cat"]) != (row["comparison_type"] == "within_cat"):
            raise ValueError("Comparison type and cat IDs disagree.")
    indices = [i for i, row in enumerate(structure) if row["comparison_type"] == "within_cat"]
    subset = [structure[i] for i in indices]
    if not subset:
        raise ValueError("No within-cat G3 test responses.")
    pairs = len({row["pair_id"] for row in subset})
    photos = {row["held_out_photo"] for row in subset}
    support = protocol["operational_support"]
    if len(subset) < support["minimum_responses"] or pairs < support["minimum_pairs"]:
        raise ValueError("Within-cat subgroup is below the registered reporting support threshold.")
    if selected not in ("H", "C", "Y"):
        raise ValueError("Invalid validation-selected G3 representation.")
    subset_rows = {name: [aligned[name][i] for i in indices] for name in MODELS}
    if not np.allclose([r["logit"] for r in subset_rows["M1"]], m1_intercept, rtol=0, atol=1e-12):
        raise ValueError("M1 within-cat predictions do not equal the displayed-side intercept.")
    losses = {name: np.asarray([r["loss"] for r in subset_rows[name]]) for name in MODELS}
    clusters = [row["held_out_photo"] for row in subset]
    settings = protocol["uncertainty"]
    samples = cluster_bootstrap(clusters, losses, settings["replicates"], settings["seed"])
    results = []
    for name in MODELS:
        model_rows = subset_rows[name]
        y = np.asarray([r["outcome_a"] for r in model_rows])
        p = np.asarray([r["prob_a"] for r in model_rows])
        low, high = np.quantile(samples[name], [.025, .975])
        results.append({"model": name, "responses": len(subset), "pairs": pairs, "held_out_photos": len(photos),
                        "log_loss": float(losses[name].mean()), "ci_low": float(low), "ci_high": float(high),
                        "brier": float(np.mean((p-y)**2)), "accuracy": float(np.mean((p >= .5) == y))})
    contrasts = []
    for a, b in (("M0", "M1"), ("M1", "M3-" + selected),
                 ("M3-H", "M4-H"), ("M3-C", "M4-C"), ("M3-Y", "M4-Y")):
        difference = samples[a] - samples[b]
        low, high = np.quantile(difference, [.025, .975])
        contrasts.append({"reference": a, "candidate": b,
                          "reference_minus_candidate": float(np.mean(losses[a]-losses[b])),
                          "ci_low": float(low), "ci_high": float(high),
                          "bootstrap_se": float(np.std(difference, ddof=1)),
                          "responses": len(subset), "pairs": pairs, "held_out_photos": len(photos)})
    summary = {"status": "post_test_retrospective_exploratory", "responses": len(subset), "pairs": pairs,
               "held_out_photos": len(photos), "cats": len({r["a_cat"] for r in subset}),
               "response_fraction_of_g3_test": len(subset) / len(structure),
               "g2_test_partition_overlap_responses": sum(r["partition"] == "test" for r in subset),
               "selected_representation_before_subgroup": selected, "operational_support_met": True,
               "m1_within_cat_constant_logit": m1_intercept,
               "interpretation": protocol["interpretation"], "prior_exposure": protocol["prior_exposure"],
               "uncertainty": settings["limitations"]}
    return results, contrasts, summary


def plot(results: list[dict[str, Any]], contrasts: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="mpl_", dir=HERE) as config:
        os.environ["MPLCONFIGDIR"] = config
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.0))
        fig.suptitle("G3 within-cat subgroup: unseen photos of familiar cats", fontsize=15, weight="bold")
        fig.text(.5, .91, f"{summary['responses']} responses | {summary['pairs']} pairs | "
                 f"{summary['held_out_photos']} held-out photos | {summary['cats']} cats", ha="center", fontsize=10)
        y = np.arange(len(results))
        for i, row in enumerate(results):
            axes[0].hlines(i, row["ci_low"], row["ci_high"], color="#326d80", linewidth=1.5)
            axes[0].plot(row["log_loss"], i, "o", color="#326d80")
        axes[0].set_yticks(y, [r["model"] for r in results])
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Response-weighted log loss (lower is better)")
        axes[0].set_title("Frozen model performance")
        cy = np.arange(len(contrasts))
        axes[1].axvline(0, color="#555555", linewidth=1)
        for i, row in enumerate(contrasts):
            axes[1].hlines(i, row["ci_low"], row["ci_high"], color="#326d80", linewidth=1.5)
            axes[1].plot(row["reference_minus_candidate"], i, "o", color="#326d80")
        axes[1].set_yticks(cy, [f"{r['reference']} minus {r['candidate']}" for r in contrasts])
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Log-loss difference (positive favors candidate)")
        axes[1].set_title("Paired comparisons")
        fig.text(.06, .095, "M1 cat scores cancel within a cat; its prediction is a constant displayed-side intercept.", fontsize=9)
        fig.text(.06, .065, "Post-test, retrospective, not independent or causal. Intervals: 95% pointwise held-out-photo bootstrap.", fontsize=9)
        fig.text(.06, .035, "Shared cats, known photos, and respondents may remain dependent; intervals are not multiplicity-adjusted.", fontsize=9)
        fig.subplots_adjust(left=.1, right=.97, top=.85, bottom=.20, wspace=.52)
        fig.savefig(HERE / "presentation_within_cat_g3.pdf")
        plt.close(fig)


def run() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    names = ("original_hashes_before.json", "evidence_hashes_before.json", "within_cat_test_results.csv",
             "within_cat_contrasts.csv", "within_cat_summary.json", "presentation_within_cat_g3.pdf",
             "analysis_provenance.json")
    if any((HERE / name).exists() for name in names):
        raise FileExistsError("Within-cat analysis already exists; refusing to overwrite.")
    originals = snapshot(ROOT, EXCLUDED_ROOT, HERE)
    evidence = snapshot(ROOT / "Presentation_Evidence", set())
    write_json(HERE / "original_hashes_before.json", originals)
    write_json(HERE / "evidence_hashes_before.json", evidence)
    verify_inputs()
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    models = json.loads((G3 / "selected_models.json").read_text(encoding="utf-8"))
    structure = load_structure("test")
    records = pq.read_table(G3 / "test_predictions.parquet").to_pylist()
    aligned = align_predictions(structure, records)
    results, contrasts, summary = calculate(structure, aligned, models["selected_representation"],
                                            models["selected"]["M1"]["coefficients"][0], protocol)
    write_csv(HERE / "within_cat_test_results.csv", results, list(results[0]))
    write_csv(HERE / "within_cat_contrasts.csv", contrasts, list(contrasts[0]))
    write_json(HERE / "within_cat_summary.json", summary)
    plot(results, contrasts, summary)
    if snapshot(ROOT, EXCLUDED_ROOT, HERE) != originals or snapshot(ROOT / "Presentation_Evidence", set()) != evidence:
        raise RuntimeError("Original project or presentation evidence changed during subgroup analysis.")
    outputs = names[2:6]
    inputs = [G3 / "freeze.json", G3 / "freeze.sha256", G3 / "evaluation_provenance.json",
              G3 / "test_predictions.parquet", G3 / "selected_models.json", G3 / "row_allocation.csv",
              ROOT / "artifacts" / "rows_canonical.parquet", HERE / "protocol.json", HERE / "analyze.py"]
    provenance = {
        "status": "completed_post_test_retrospective_exploratory",
        "source_usage": "Saved G3 prediction outcomes and structural data only; no raw test outcomes parsed, no refit or test re-evaluation.",
        "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        "output_sha256": {name: digest(HERE / name) for name in outputs},
        "original_hash_snapshot_sha256": digest(HERE / "original_hashes_before.json"),
        "prior_evidence_snapshot_sha256": digest(HERE / "evidence_hashes_before.json"),
        "original_project_files_unchanged": len(originals), "previous_evidence_files_unchanged": len(evidence),
        "frozen_g2_and_g3_unchanged": True, "models_refit": False, "threshold_or_model_selected": False,
        "limitations": protocol["prior_exposure"] + " " + protocol["uncertainty"]["limitations"]
    }
    write_json(HERE / "analysis_provenance.json", provenance)
    print(json.dumps({"summary": summary, "results": results, "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    run()
