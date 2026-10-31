"""Reconcile saved diagnostic artifacts without refitting or reading raw labels."""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
from pypdf import PdfReader
from scipy.special import expit

from common import EXPERIMENT, HERE, ROOT, digest, read_csv, verify_inputs, write_json
from experiment import REQUESTED
from src.abstention_utility import point_metrics
from transfer import predict_transfer, source_rows


def verify_transfer(target: str) -> dict:
    base = target.lower()
    parent = ROOT / (base + "_experiment")
    proof = json.loads((HERE / f"{base}_provenance.json").read_text(encoding="utf-8"))
    if proof["freeze_sha256"] != digest(HERE / f"{base}_freeze.json"):
        raise ValueError(f"{target} transfer freeze mismatch.")
    if proof["source_predictions_sha256"] != digest(parent / "test_predictions.parquet"):
        raise ValueError(f"{target} original saved predictions changed.")
    for name, expected in proof["outputs_sha256"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"{target} saved output mismatch: {name}")
    model = json.loads((HERE / f"{base}_models.json").read_text(encoding="utf-8"))
    structure = source_rows(target, "test", False)
    reconstructed = predict_transfer(structure, model, target == "G4")
    records = pq.read_table(HERE / f"{base}_predictions.parquet").to_pylist()
    original = pq.read_table(parent / "test_predictions.parquet",
                             filters=[("model", "=", "M0")],
                             columns=["train_row_id", "outcome_a", "pair_id"]).to_pylist()
    original_by_id = {r["train_row_id"]: r for r in original}
    results = {r["model"]: r for r in read_csv(HERE / f"{base}_results.csv")}
    contrasts = read_csv(HERE / f"{base}_contrasts.csv")
    ids = [r["train_row_id"] for r in structure]
    if len(original_by_id) != len(original) or set(original_by_id) != set(ids):
        raise ValueError("Original reserved outcomes do not cover the same IDs.")
    losses = {}
    for name, result in results.items():
        rows = [r for r in records if r["model"] == name]
        if [r["train_row_id"] for r in rows] != ids:
            raise ValueError(f"Transfer prediction row order differs: {name}")
        y = np.asarray([r["outcome"] for r in rows])
        if y.tolist() != [original_by_id[key]["outcome_a"] for key in ids]:
            raise ValueError("Transfer outcome alignment differs.")
        z = np.asarray([r["logit"] for r in rows])
        if name in reconstructed:
            np.testing.assert_allclose(z, reconstructed[name], rtol=0, atol=1e-13)
        p = expit(z)
        ll = y*np.logaddexp(0., -z)+(1-y)*np.logaddexp(0., z)
        np.testing.assert_allclose(p, [r["probability"] for r in rows], rtol=0, atol=1e-13)
        np.testing.assert_allclose(ll, [r["loss"] for r in rows], rtol=0, atol=1e-13)
        np.testing.assert_allclose([ll.mean(), np.mean((p-y)**2), np.mean((p >= .5) == y)],
                                   [float(result["log_loss"]), float(result["brier"]), float(result["accuracy"])],
                                   rtol=0, atol=1e-12)
        losses[name] = ll
    cluster_field = "held_out_photo" if target == "G3" else "held_out_cat"
    clusters = [row[cluster_field] for row in structure]
    unique = sorted(set(clusters))
    mapping = {key: i for i, key in enumerate(unique)}
    codes = np.asarray([mapping[key] for key in clusters])
    counts = np.bincount(codes)
    names = list(results)
    sums = np.stack([np.bincount(codes, weights=losses[name]) for name in names])
    rng = np.random.default_rng(20261101)
    bootstrap = np.empty((5000, len(names)))
    for i in range(5000):
        draw = rng.integers(0, len(unique), len(unique))
        weights = np.bincount(draw, minlength=len(unique))
        bootstrap[i] = sums @ weights / (counts @ weights)
    for i, name in enumerate(names):
        np.testing.assert_allclose(np.quantile(bootstrap[:, i], [.025, .975]),
                                   [float(results[name]["ci_low"]), float(results[name]["ci_high"])],
                                   rtol=0, atol=1e-12)
    for row in contrasts:
        ai, bi = names.index(row["reference"]), names.index(row["candidate"])
        values = bootstrap[:, ai]-bootstrap[:, bi]
        np.testing.assert_allclose([np.mean(losses[row["reference"]]-losses[row["candidate"]]),
                                    *np.quantile(values, [.025, .975]), np.std(values, ddof=1)],
                                   [float(row["difference"]), float(row["ci_low"]), float(row["ci_high"]),
                                    float(row["bootstrap_se"])], rtol=0, atol=1e-12)
    return {"responses": len(ids), "models": len(results), "contrasts": len(contrasts),
            "clusters": len(unique), "bootstrap_replicates": 5000}


def run() -> None:
    verify_inputs()
    path = HERE / "verification.json"
    if path.exists():
        raise FileExistsError("Diagnostic verification already exists.")
    summary = json.loads((HERE / "diagnostic_summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "all_requested_additional_diagnostics_available":
        raise ValueError("Diagnostics are incomplete.")
    for name, expected in summary["outputs_sha256"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"Diagnostic summary output mismatch: {name}")
    temperature_freeze = json.loads((HERE / "temperature_freeze.json").read_text(encoding="utf-8"))
    if digest(HERE / "temperature_freeze.json") != (HERE / "temperature_freeze.sha256").read_text(encoding="ascii").strip():
        raise ValueError("Temperature selection freeze mismatch.")
    for proof_name in ("fixed_diagnostics_provenance.json", "subgroup_calibration_provenance.json",
                       "test_support_slices_provenance.json"):
        proof = json.loads((HERE / proof_name).read_text(encoding="utf-8"))
        outputs = proof.get("output_sha256", {})
        if isinstance(outputs, dict):
            for name, expected in outputs.items():
                if digest(HERE / name) != expected:
                    raise ValueError(f"Saved diagnostic output hash mismatch: {name}")
        elif proof_name == "subgroup_calibration_provenance.json":
            if digest(HERE / "subgroup_calibration.csv") != outputs:
                raise ValueError("Subgroup calibration hash mismatch.")
    predictions = pq.read_table(EXPERIMENT / "test_predictions.parquet",
                                filters=[("model", "in", list(REQUESTED) + ["M4-C"])]).to_pylist()
    abstention = read_csv(HERE / "abstention_utility.csv")
    if len(abstention) != 105:
        raise ValueError("Incomplete fixed-threshold abstention table.")
    for row in abstention:
        subset = sorted([r for r in predictions if r["model"] == row["model"]], key=lambda r: r["train_row_id"])
        point, _, _, _ = point_metrics([r["probability"] for r in subset], [r["outcome"] for r in subset],
                                       [r["loss"] for r in subset], [r["pair_id"] for r in subset], float(row["threshold"]))
        for key in ("original_responses", "retained_responses", "retained_pairs", "correct_retained",
                    "incorrect_retained", "coverage", "utility_per_original_response"):
            if abs(float(row[key])-float(point[key])) > 1e-12:
                raise ValueError(f"Abstention point metric mismatch: {row['model']} {row['threshold']} {key}")
    for row in read_csv(HERE / "temperature_results.csv"):
        subset = [r for r in predictions if r["model"] == row["model"]]
        y = np.asarray([r["outcome"] for r in subset])
        z = np.asarray([r["logit"] for r in subset]) / temperature_freeze["temperature"][row["model"]]
        loss = y*np.logaddexp(0., -z)+(1-y)*np.logaddexp(0., z)
        if abs(float(loss.mean())-float(row["temperature_loss"])) > 1e-12:
            raise ValueError("Temperature point metric mismatch.")
    g3, g4 = verify_transfer("G3"), verify_transfer("G4")
    abstention_proof = json.loads((HERE / "abstention_figure_provenance.json").read_text(encoding="utf-8"))
    if (abstention_proof["input_sha256"] != digest(HERE / "abstention_utility.csv") or
            abstention_proof["output_sha256"] != digest(HERE / "presentation_abstention_full.pdf")):
        raise ValueError("Abstention dashboard provenance mismatch.")
    for name, pages in (("presentation_fixed_diagnostics.pdf", 2), ("presentation_abstention_full.pdf", 4),
                        ("presentation_learning_penalty_transfer.pdf", 3)):
        if len(PdfReader(HERE / name).pages) != pages:
            raise ValueError(f"Unexpected PDF pages: {name}")
    write_json(path, {"status": "independent_replay_passed", "g3": g3, "g4": g4,
                      "abstention_cells_point_checked": len(abstention),
                      "temperature_test_point_metrics_checked": 4,
                      "learning_rows": summary["learning_curve_rows"], "penalty_refits": summary["penalty_refits"],
                      "summary_sha256": digest(HERE / "diagnostic_summary.json"),
                      "raw_outcomes_reopened": False, "limitations": "Existing outcomes previously exposed; not independent."})
    print(json.dumps({"g3": g3, "g4": g4, "abstention_cells": len(abstention)}, indent=2))


if __name__ == "__main__":
    run()
