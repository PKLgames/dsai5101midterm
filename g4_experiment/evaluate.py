"""Evaluate the frozen retrospective G4 models once."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit

from shared import HERE, ROOT, digest, load_outcomes, load_structure, predict, write_csv, write_json
from src.evaluate import pair_bootstrap


def verify_freeze() -> dict:
    freeze_path = HERE / "freeze.json"
    if digest(freeze_path) != (HERE / "freeze.sha256").read_text(encoding="ascii").strip():
        raise ValueError("G4 freeze seal mismatch.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["status"] != "frozen_before_g4_test_access" or freeze["g4_test_outcomes_accessed"]:
        raise ValueError("Invalid G4 freeze state.")
    for path, expected in freeze["experiment_sha256"].items():
        if digest(HERE / path) != expected:
            raise ValueError(f"G4 frozen experiment artifact changed: {path}")
    for path, expected in freeze["root_input_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"G4 frozen root input changed: {path}")
    return freeze


def evaluate() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    names = ["evaluation_started.json", "test_predictions.parquet", "test_results.csv",
             "test_contrasts.csv", "presentation_g4_results.pdf", "evaluation_provenance.json"]
    if any((HERE / name).exists() for name in names):
        raise FileExistsError("G4 evaluation has already started; refusing to reaccess test outcomes.")
    freeze = verify_freeze()
    models = json.loads((HERE / "selected_models.json").read_text(encoding="utf-8"))
    rows = load_structure("test")
    coverage = json.loads((HERE / "coverage.json").read_text(encoding="utf-8"))
    if len(rows) != coverage["g4"]["test"]["responses"]:
        raise ValueError("G4 test population differs from the pre-outcome coverage record.")
    logits = predict(models, rows)
    if set(logits) != {"M0", "M1-zero", "M3-H", "M3-C", "M3-Y", "M4-H", "M4-C", "M4-Y"}:
        raise ValueError("Incomplete frozen G4 prediction set.")
    with (ROOT / "g3_experiment" / "row_allocation.csv").open(encoding="utf-8", newline="") as stream:
        g3 = {r["train_row_id"]: r["allocation"] for r in csv.DictReader(stream)}
    g2_overlap = dict(sorted(Counter(r["partition"] for r in rows).items()))
    g3_overlap = dict(sorted(Counter(g3[r["train_row_id"]] for r in rows).items()))
    write_json(HERE / "evaluation_started.json", {
        "status": "g4_test_access_started", "freeze_sha256": digest(HERE / "freeze.json"),
        "previous_g2_partition_overlap": g2_overlap, "previous_g3_allocation_overlap": g3_overlap,
        "limitation": "Retrospective test outcomes include outcomes previously exposed in the completed G2/G3 analyses.",
    })
    y = load_outcomes(rows, freeze["root_input_sha256"]["dataset\\train.csv"])
    losses = {name: y*np.logaddexp(0., -value) + (1.-y)*np.logaddexp(0., value)
              for name, value in logits.items()}
    clusters = [row["held_out_cat"] for row in rows]
    samples, _ = pair_bootstrap(clusters, losses, seed=20261101, replicates=5000)
    results, predictions = [], []
    for name in ("M0", "M1-zero", "M3-H", "M3-C", "M3-Y", "M4-H", "M4-C", "M4-Y"):
        p = expit(logits[name])
        low, high = np.quantile(samples[name], [.025, .975])
        results.append({"model": name, "responses": len(rows), "pairs": len({r["pair_id"] for r in rows}),
                        "held_out_cats": len(set(clusters)), "log_loss": float(losses[name].mean()),
                        "log_loss_ci_low": float(low), "log_loss_ci_high": float(high),
                        "brier": float(np.mean((p-y)**2)), "accuracy": float(np.mean((p >= .5) == y)),
                        "pooled": models["selected"][name]["pooled"]})
        for i, row in enumerate(rows):
            predictions.append({"model": name, "train_row_id": row["train_row_id"], "pair_id": row["pair_id"],
                                "held_out_cat": clusters[i], "outcome_a": int(y[i]),
                                "logit": float(logits[name][i]), "prob_a": float(p[i]),
                                "loss": float(losses[name][i])})
    selected = models["selected_representation"]
    contrasts = []
    for a, b in (("M0", "M1-zero"), ("M1-zero", "M3-" + selected), ("M3-" + selected, "M4-" + selected)):
        difference = samples[a] - samples[b]
        low, high = np.quantile(difference, [.025, .975])
        contrasts.append({"reference": a, "candidate": b,
                          "reference_minus_candidate": float(np.mean(losses[a]-losses[b])),
                          "ci_low": float(low), "ci_high": float(high),
                          "bootstrap_se": float(np.std(difference, ddof=1)),
                          "responses": len(rows), "pairs": len({r["pair_id"] for r in rows}),
                          "held_out_cats": len(set(clusters))})
    pq.write_table(pa.Table.from_pylist(predictions), HERE / "test_predictions.parquet", compression="zstd")
    write_csv(HERE / "test_results.csv", results, list(results[0]))
    write_csv(HERE / "test_contrasts.csv", contrasts, list(contrasts[0]))
    plot(results, contrasts, selected)
    write_json(HERE / "evaluation_provenance.json", {
        "status": "completed_once_post_freeze_retrospective_exploratory",
        "freeze_sha256": digest(HERE / "freeze.json"),
        "test_access_record_sha256": digest(HERE / "evaluation_started.json"),
        "output_sha256": {name: digest(HERE / name) for name in names[1:-1]},
        "responses": len(rows), "pairs": len({r["pair_id"] for r in rows}),
        "held_out_cats": len(set(clusters)), "bootstrap_replicates": 5000,
        "bootstrap_unit": "held-out cat", "bootstrap_seed": 20261101,
        "previous_g2_partition_overlap": g2_overlap, "previous_g3_allocation_overlap": g3_overlap,
        "interpretation": "Post-test retrospective exploration, not an independent or confirmatory holdout. Intervals are pointwise and conditional on fitted models and selection; held-out-cat resampling does not account for all shared known-cat, photo or respondent dependence.",
        "m1_zero_convention": models["m1_zero_convention"],
        "deployment_or_frozen_g2_g3_changed": False,
    })
    print(json.dumps({"selected_representation": selected, "test_results": results, "contrasts": contrasts}, indent=2))


def plot(results: list[dict], contrasts: list[dict], selected: str) -> None:
    with tempfile.TemporaryDirectory(prefix="mpl_", dir=HERE) as config:
        os.environ["MPLCONFIGDIR"] = config
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.0))
        fig.suptitle("Retrospective G4: an unseen cat versus a known cat", fontsize=15, weight="bold")
        first = results[0]
        fig.text(.5, .91, f"{first['responses']} responses | {first['pairs']} pairs | "
                 f"{first['held_out_cats']} held-out cats", ha="center", fontsize=10)
        y = np.arange(len(results))
        for i, row in enumerate(results):
            axes[0].hlines(i, row["log_loss_ci_low"], row["log_loss_ci_high"], color="#326d80", linewidth=1.5)
            axes[0].plot(row["log_loss"], i, "o", color="#326d80")
        axes[0].set_yticks(y, [r["model"] for r in results])
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Response-weighted log loss (lower is better)")
        axes[0].set_title("Frozen G4 test results")
        cy = np.arange(len(contrasts))
        axes[1].axvline(0, color="#555555", linewidth=1)
        for i, row in enumerate(contrasts):
            axes[1].hlines(i, row["ci_low"], row["ci_high"], color="#326d80", linewidth=1.5)
            axes[1].plot(row["reference_minus_candidate"], i, "o", color="#326d80")
        axes[1].set_yticks(cy, [f"{r['reference']} minus {r['candidate']}" for r in contrasts])
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Log-loss difference (positive favors candidate)")
        axes[1].set_title(f"Validation-selected representation: {selected}")
        fig.text(.06, .095, "M1-zero assigns the unseen cat a fixed zero score; it is not an estimated unseen-cat score.", fontsize=9)
        fig.text(.06, .065, "Retrospective and exploratory: previously exposed outcomes; not an independent or causal test.", fontsize=9)
        fig.text(.06, .035, "95% pointwise held-out-cat bootstrap; shared known cats/photos and respondents may remain dependent.", fontsize=9)
        fig.subplots_adjust(left=.1, right=.97, top=.85, bottom=.20, wspace=.52)
        fig.savefig(HERE / "presentation_g4_results.pdf")
        plt.close(fig)


if __name__ == "__main__":
    evaluate()
