"""Validation-fixed temperature and exploratory abstention on saved G2 predictions."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit

from common import EXPERIMENT, HERE, ROOT, digest, verify_inputs, write_csv, write_json
from experiment import REQUESTED, load_rows, predict, score
from src.abstention_utility import calculate
from src.evaluate import pair_bootstrap


def select_temperature() -> None:
    verify_inputs()
    if any((HERE / name).exists() for name in ("temperature_candidates.csv", "temperature_freeze.json", "temperature_freeze.sha256")):
        raise FileExistsError("Temperature selection already exists.")
    models = json.loads((EXPERIMENT / "selected_models.json").read_text(encoding="utf-8"))
    val = load_rows("validation")
    y = np.asarray([row["y_displayed"] for row in val], dtype=float)
    logits = predict(val, models)
    grid = [.25, .5, .75, 1., 1.5, 2., 4.]
    candidates = []
    chosen = {}
    for model in REQUESTED:
        options = [(temperature, score(y, logits[model]/temperature)[0]) for temperature in grid]
        minimum = min(value for _, value in options)
        winner = min(((temperature, value) for temperature, value in options if value <= minimum + 1e-6),
                     key=lambda item: (abs(item[0]-1), item[0]))[0]
        chosen[model] = winner
        candidates.extend({"model": model, "temperature": temperature, "validation_loss": loss,
                           "selected": temperature == winner} for temperature, loss in options)
    write_csv(HERE / "temperature_candidates.csv", candidates, list(candidates[0]))
    freeze = {"status": "validation_selected_before_temperature_test_reanalysis",
              "temperature": chosen, "selection_grid": grid,
              "source_sha256": digest(EXPERIMENT / "selected_models.json"),
              "validation_candidates_sha256": digest(HERE / "temperature_candidates.csv"),
              "demographic_freeze_sha256": digest(EXPERIMENT / "freeze.json"),
              "prior_test_exposure": "Original G2 and demographic model test outcomes were already exposed; post-test diagnostic."}
    write_json(HERE / "temperature_freeze.json", freeze)
    (HERE / "temperature_freeze.sha256").write_text(digest(HERE / "temperature_freeze.json") + "\n", encoding="ascii")
    print(json.dumps(freeze, indent=2))


def evaluate_saved() -> None:
    verify_inputs()
    names = ("saved_analysis_started.json", "temperature_results.csv", "abstention_utility.csv",
             "presentation_fixed_diagnostics.pdf", "fixed_diagnostics_provenance.json")
    if any((HERE / name).exists() for name in names):
        raise FileExistsError("Saved-test diagnostic analysis already started.")
    path = HERE / "temperature_freeze.json"
    if digest(path) != (HERE / "temperature_freeze.sha256").read_text(encoding="ascii").strip():
        raise ValueError("Temperature selection seal mismatch.")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if (freeze["source_sha256"] != digest(EXPERIMENT / "selected_models.json")
            or freeze["validation_candidates_sha256"] != digest(HERE / "temperature_candidates.csv")
            or freeze["demographic_freeze_sha256"] != digest(EXPERIMENT / "freeze.json")):
        raise ValueError("Temperature source mismatch.")
    write_json(HERE / "saved_analysis_started.json", {
        "status": "previously_saved_g2_outcome_analysis_started",
        "temperature_freeze_sha256": digest(path), "raw_outcomes_reopened": False,
        "limitations": "Post-test, retrospective, noncausal; no model/temperature/threshold promotion."})
    records = pq.read_table(EXPERIMENT / "test_predictions.parquet",
                            filters=[("model", "in", list(REQUESTED) + ["M4-C"])]).to_pylist()
    models = {name: sorted([r for r in records if r["model"] == name], key=lambda r: r["train_row_id"])
              for name in (*REQUESTED, "M4-C")}
    reference = [(r["train_row_id"], r["pair_id"], r["outcome"]) for r in models["M4-C"]]
    if not reference or len(set(r[0] for r in reference)) != len(reference):
        raise ValueError("Missing or duplicate saved response IDs.")
    for name, rows in models.items():
        if [(r["train_row_id"], r["pair_id"], r["outcome"]) for r in rows] != reference:
            raise ValueError(f"Unmatched saved prediction population: {name}")
        y = np.asarray([r["outcome"] for r in rows])
        z = np.asarray([r["logit"] for r in rows])
        np.testing.assert_allclose(expit(z), [r["probability"] for r in rows], rtol=0, atol=1e-12)
        np.testing.assert_allclose(y*np.logaddexp(0., -z)+(1-y)*np.logaddexp(0., z),
                                   [r["loss"] for r in rows], rtol=0, atol=1e-12)
    pair_ids = [r[1] for r in reference]
    outcomes = np.asarray([r[2] for r in reference], dtype=float)
    losses = {}
    for name in REQUESTED:
        z = np.asarray([r["logit"] for r in models[name]])
        for label, logits in ((name, z), (name + "-temperature", z / freeze["temperature"][name])):
            losses[label] = outcomes*np.logaddexp(0., -logits)+(1-outcomes)*np.logaddexp(0., logits)
    samples, paired = pair_bootstrap(pair_ids, losses, seed=20261101, replicates=5000)
    temperature_results = []
    for name in REQUESTED:
        adjusted = name + "-temperature"
        delta = samples[name]-samples[adjusted]
        pair_delta = paired[name]-paired[adjusted]
        low, high = np.quantile(delta, [.025, .975])
        plow, phigh = np.quantile(pair_delta, [.025, .975])
        temperature_results.append({
            "model": name, "temperature": freeze["temperature"][name],
            "unscaled_loss": float(losses[name].mean()), "temperature_loss": float(losses[adjusted].mean()),
            "difference_unscaled_minus_temperature": float(np.mean(losses[name]-losses[adjusted])),
            "ci95_low": float(low), "ci95_high": float(high), "bootstrap_se": float(np.std(delta, ddof=1)),
            "pair_ci95_low": float(plow), "pair_ci95_high": float(phigh), "responses": len(reference),
            "pairs": len(set(pair_ids)), "interpretation": "Post-test exploratory; validation-fixed temperature; no promotion."})
    write_csv(HERE / "temperature_results.csv", temperature_results, list(temperature_results[0]))
    abstention = calculate(models, [round(.5+.025*i, 3) for i in range(21)], 20261101, 5000)
    write_csv(HERE / "abstention_utility.csv", abstention, list(abstention[0]))
    plot(abstention, temperature_results)
    outputs = names[1:-1]
    write_json(HERE / "fixed_diagnostics_provenance.json", {
        "status": "completed_post_test_exploratory", "temperature_freeze_sha256": digest(path),
        "source_demographic_predictions_sha256": digest(EXPERIMENT / "test_predictions.parquet"),
        "output_sha256": {name: digest(HERE / name) for name in outputs},
        "test_responses": len(reference), "pairs": len(set(pair_ids)),
        "temperature_rows": len(temperature_results), "abstention_rows": len(abstention),
        "models": list(models), "bootstrap_replicates": 5000, "bootstrap_unit": "unordered pair",
        "raw_outcomes_reopened": False, "threshold_optimized": False, "deployment_changed": False,
        "interpretation": "Descriptive pointwise intervals are unadjusted; existing test outcomes were previously exposed."})
    print(f"Completed {len(temperature_results)} temperature comparisons and {len(abstention)} abstention cells.")


def plot(rows: list[dict], temperatures: list[dict]) -> None:
    with tempfile.TemporaryDirectory(prefix="mpl_", dir=HERE) as config:
        os.environ["MPLCONFIGDIR"] = config
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(HERE / "presentation_fixed_diagnostics.pdf") as pdf:
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.5))
            for name in (*REQUESTED, "M4-C"):
                subset = [r for r in rows if r["model"] == name]
                axes[0].plot([r["threshold"] for r in subset], [r["coverage"] for r in subset], label=name)
                axes[1].plot([r["threshold"] for r in subset], [r["utility_per_original_response"] for r in subset], label=name)
            axes[0].set(title="Coverage", xlabel="Minimum confidence", ylabel="Retained / all responses")
            axes[1].set(title="Realized utility: +1 / -1 / 0", xlabel="Minimum confidence", ylabel="Utility / original response")
            for ax in axes:
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            fig.suptitle("Post-test exploratory abstention; no cutoff selected", fontsize=15)
            fig.text(.05, .035, "Existing frozen outcomes | 95% pointwise pair bootstrap in table | Not a deployment policy", fontsize=9)
            fig.tight_layout(rect=(0, .07, 1, .95))
            pdf.savefig(fig)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.axvline(0, color="gray")
            for i, row in enumerate(temperatures):
                ax.hlines(i, row["ci95_low"], row["ci95_high"], color="#28677c")
                ax.plot(row["difference_unscaled_minus_temperature"], i, "o", color="#28677c")
            ax.set_yticks(range(len(temperatures)), [f"{r['model']} (T={r['temperature']:g})" for r in temperatures])
            ax.invert_yaxis()
            ax.set(xlabel="Unscaled minus temperature-scaled log loss (positive favors scaling)",
                   title="Validation-selected temperatures: exploratory saved-test contrasts")
            fig.text(.05, .04, "Pointwise 95% intervals are not multiplicity adjusted. Scaling was not promoted.", fontsize=9)
            fig.tight_layout(rect=(0, .07, 1, 1))
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("select", "saved"))
    {"select": select_temperature, "saved": evaluate_saved}[parser.parse_args().stage]()
