"""Evaluate the frozen retrospective G3 experiment once."""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit

from common import EXPERIMENT, ROOT, cluster_bootstrap, digest, load_outcomes, load_structure, predict, write_csv, write_json


def verify_freeze() -> dict:
    freeze_path = EXPERIMENT / "freeze.json"
    seal = (EXPERIMENT / "freeze.sha256").read_text(encoding="ascii").strip()
    if digest(freeze_path) != seal:
        raise ValueError("G3 freeze seal mismatch.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["status"] != "frozen_before_g3_test_access" or freeze["g3_test_outcomes_accessed"]:
        raise ValueError("G3 freeze state is invalid.")
    for path, expected in freeze["experiment_sha256"].items():
        if digest(EXPERIMENT / path) != expected:
            raise ValueError(f"G3 frozen experiment artifact changed: {path}")
    for path, expected in freeze["root_input_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Frozen root input changed: {path}")
    return freeze


def evaluate() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    names = ["evaluation_started.json", "test_predictions.parquet", "test_results.csv",
             "test_contrasts.csv", "presentation_g3_results.pdf", "evaluation_provenance.json"]
    if any((EXPERIMENT / name).exists() for name in names):
        raise FileExistsError("G3 evaluation has already started; refusing to access test outcomes again.")
    freeze = verify_freeze()
    models = json.loads((EXPERIMENT / "selected_models.json").read_text(encoding="utf-8"))
    rows = load_structure("test")
    coverage = json.loads((EXPERIMENT / "coverage.json").read_text(encoding="utf-8"))
    if len(rows) != coverage["g3"]["test"]["responses"]:
        raise ValueError("Test population differs from the pre-outcome coverage record.")
    logits = predict(models, rows)
    if set(logits) != {"M0", "M1", "M3-H", "M3-C", "M3-Y", "M4-H", "M4-C", "M4-Y"}:
        raise ValueError("Incomplete frozen prediction set.")
    write_json(EXPERIMENT / "evaluation_started.json", {
        "status": "g3_test_access_started", "freeze_sha256": digest(EXPERIMENT / "freeze.json"),
        "prior_original_g2_partition_overlap": dict(sorted(Counter(r["partition"] for r in rows).items())),
        "limitation": "Retrospective test outcomes include outcomes previously exposed in the completed G2 analysis.",
    })
    y = load_outcomes(rows, freeze["root_input_sha256"]["dataset\\train.csv"])
    losses = {name: y * np.logaddexp(0., -value) + (1. - y) * np.logaddexp(0., value)
              for name, value in logits.items()}
    clusters = [row["held_out_photo"] for row in rows]
    samples = cluster_bootstrap(clusters, losses, 5000, 20261101)
    results = []
    predictions = []
    for name in ("M0", "M1", "M3-H", "M3-C", "M3-Y", "M4-H", "M4-C", "M4-Y"):
        p = expit(logits[name])
        low, high = np.quantile(samples[name], [0.025, 0.975])
        results.append({"model": name, "responses": len(rows), "pairs": len({r["pair_id"] for r in rows}),
                        "held_out_photos": len(set(clusters)), "log_loss": float(losses[name].mean()),
                        "log_loss_ci_low": float(low), "log_loss_ci_high": float(high),
                        "brier": float(np.mean((p-y)**2)), "accuracy": float(np.mean((p >= .5) == y)),
                        "pooled": models["selected"][name]["pooled"]})
        for i, row in enumerate(rows):
            predictions.append({"model": name, "train_row_id": row["train_row_id"], "pair_id": row["pair_id"],
                                "held_out_photo": clusters[i], "outcome_a": int(y[i]),
                                "logit": float(logits[name][i]), "prob_a": float(p[i]),
                                "loss": float(losses[name][i])})
    selected = models["selected_representation"]
    contrasts = []
    for a, b in (("M0", "M1"), ("M1", "M3-" + selected), ("M3-" + selected, "M4-" + selected)):
        values = samples[a] - samples[b]
        low, high = np.quantile(values, [0.025, 0.975])
        contrasts.append({"reference": a, "candidate": b,
                          "reference_minus_candidate": float(np.mean(losses[a] - losses[b])),
                          "ci_low": float(low), "ci_high": float(high),
                          "bootstrap_se": float(np.std(values, ddof=1)),
                          "responses": len(rows), "held_out_photos": len(set(clusters))})
    pq.write_table(pa.Table.from_pylist(predictions), EXPERIMENT / "test_predictions.parquet",
                   compression="zstd")
    write_csv(EXPERIMENT / "test_results.csv", results, list(results[0]))
    write_csv(EXPERIMENT / "test_contrasts.csv", contrasts, list(contrasts[0]))
    plot(results, contrasts, selected)
    outputs = names[1:-1]
    write_json(EXPERIMENT / "evaluation_provenance.json", {
        "status": "completed_once_post_freeze_retrospective_exploratory",
        "freeze_sha256": digest(EXPERIMENT / "freeze.json"),
        "test_access_record_sha256": digest(EXPERIMENT / "evaluation_started.json"),
        "output_sha256": {name: digest(EXPERIMENT / name) for name in outputs},
        "responses": len(rows), "pairs": len({r["pair_id"] for r in rows}),
        "held_out_photos": len(set(clusters)), "bootstrap_replicates": 5000,
        "bootstrap_unit": "held-out photo", "bootstrap_seed": 20261101,
        "original_g2_test_responses_in_g3_test": sum(r["partition"] == "test" for r in rows),
        "interpretation": "Post-test retrospective exploration, not a new independent or confirmatory holdout. Intervals are pointwise and conditional on model selection; held-out-photo resampling does not account for all shared known-photo or respondent dependence.",
        "no_g2_pipeline_changes": True, "deployment_threshold_or_model_changed": False,
    })
    print(json.dumps({"selected_representation": selected, "test_results": results,
                      "contrasts": contrasts}, indent=2))


def plot(results: list[dict], contrasts: list[dict], selected: str) -> None:
    os.environ["MPLCONFIGDIR"] = str(EXPERIMENT / "matplotlib_config")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.8))
    fig.suptitle("Retrospective G3: prediction for unseen photos of familiar cats", fontsize=15, weight="bold")
    names = [r["model"] for r in results]
    y = np.arange(len(names))
    axes[0].barh(y, [r["log_loss"] for r in results], color="#5b8fa8")
    for i, row in enumerate(results):
        axes[0].hlines(i, row["log_loss_ci_low"], row["log_loss_ci_high"], color="#222222", linewidth=1.5)
        axes[0].plot(row["log_loss"], i, "o", color="#222222", markersize=3)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Response-weighted log loss (lower is better)")
    axes[0].set_title("Frozen G3 test results")
    axes[0].set_xlim(left=0)
    cy = np.arange(len(contrasts))
    axes[1].axvline(0, color="#555555", linewidth=1)
    for i, row in enumerate(contrasts):
        axes[1].hlines(i, row["ci_low"], row["ci_high"], color="#225f70", linewidth=2)
        axes[1].plot(row["reference_minus_candidate"], i, "o", color="#225f70")
    axes[1].set_yticks(cy, [f"{r['reference']} minus {r['candidate']}" for r in contrasts])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Log-loss difference (positive favors candidate)")
    axes[1].set_title(f"Validation-selected representation: {selected}")
    fig.text(0.06, 0.06,
             "Exploratory and retrospective: outcomes were previously exposed; not an independent test. "
             "Intervals: 95% pointwise held-out-photo bootstrap.",
             fontsize=9, wrap=True)
    fig.text(0.06, 0.025,
             "Association and predictive performance, not causal effects. Shared known photos and respondents may remain dependent.",
             fontsize=9)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.17, wspace=0.45)
    fig.savefig(EXPERIMENT / "presentation_g3_results.pdf")
    plt.close(fig)


if __name__ == "__main__":
    evaluate()
