"""Reconcile saved G4 results without rereading raw test outcomes."""

from __future__ import annotations

import csv
import json

import numpy as np
import pyarrow.parquet as pq
from pypdf import PdfReader
from scipy.special import expit

from evaluate import evaluate, verify_freeze
from shared import HERE, digest, load_structure, predict


def verify() -> None:
    freeze = verify_freeze()
    models = json.loads((HERE / "selected_models.json").read_text(encoding="utf-8"))
    provenance = json.loads((HERE / "evaluation_provenance.json").read_text(encoding="utf-8"))
    if provenance["freeze_sha256"] != digest(HERE / "freeze.json"):
        raise ValueError("G4 evaluation provenance freeze mismatch.")
    if provenance["test_access_record_sha256"] != digest(HERE / "evaluation_started.json"):
        raise ValueError("G4 evaluation access-record hash mismatch.")
    for name, expected in provenance["output_sha256"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"G4 evaluation output hash mismatch: {name}")
    with (HERE / "test_results.csv").open(encoding="utf-8", newline="") as stream:
        results = {r["model"]: r for r in csv.DictReader(stream)}
    with (HERE / "test_contrasts.csv").open(encoding="utf-8", newline="") as stream:
        contrasts = list(csv.DictReader(stream))
    records = pq.read_table(HERE / "test_predictions.parquet").to_pylist()
    structure = load_structure("test")
    reconstruction = predict(models, structure)
    if set(results) != set(reconstruction) or len(contrasts) != 3:
        raise ValueError("G4 results do not match frozen model set.")
    ids = [row["train_row_id"] for row in structure]
    reference = None
    losses = {}
    for name in sorted(results):
        selected = [r for r in records if r["model"] == name]
        if [r["train_row_id"] for r in selected] != ids:
            raise ValueError(f"Saved G4 prediction IDs differ for {name}.")
        alignment = [(r["train_row_id"], r["outcome_a"], r["held_out_cat"], r["pair_id"]) for r in selected]
        if reference is not None and alignment != reference:
            raise ValueError(f"G4 matched populations differ for {name}.")
        reference = alignment
        y = np.asarray([r["outcome_a"] for r in selected], dtype=np.float64)
        logits = np.asarray([r["logit"] for r in selected])
        np.testing.assert_allclose(logits, reconstruction[name], rtol=0, atol=1e-14)
        p = expit(logits)
        np.testing.assert_allclose(p, [r["prob_a"] for r in selected], rtol=0, atol=1e-14)
        loss = y*np.logaddexp(0., -logits) + (1.-y)*np.logaddexp(0., logits)
        np.testing.assert_allclose(loss, [r["loss"] for r in selected], rtol=0, atol=1e-14)
        expected = results[name]
        np.testing.assert_allclose(
            [loss.mean(), np.mean((p-y)**2), np.mean((p >= .5) == y)],
            [float(expected["log_loss"]), float(expected["brier"]), float(expected["accuracy"])],
            rtol=0, atol=1e-14)
        losses[name] = loss
    if reference is None or len(records) != len(results) * len(structure):
        raise ValueError("Unexpected saved G4 prediction count.")
    for block in ("H", "C", "Y"):
        if models["selected"]["M4-" + block]["pooled"]:
            np.testing.assert_array_equal(reconstruction["M3-" + block], reconstruction["M4-" + block])
    clusters = [row[2] for row in reference]
    unique = sorted(set(clusters))
    lookup = {cat: i for i, cat in enumerate(unique)}
    rng = np.random.default_rng(20261101)
    draws = {name: [] for name in losses}
    for _ in range(5000):
        sampled = rng.integers(0, len(unique), len(unique))
        multiplicity = np.bincount(sampled, minlength=len(unique))
        weights = np.asarray([multiplicity[lookup[cat]] for cat in clusters])
        for name, loss in losses.items():
            draws[name].append(float(weights @ loss / weights.sum()))
    for name in losses:
        low, high = np.quantile(draws[name], [.025, .975])
        np.testing.assert_allclose([low, high], [float(results[name]["log_loss_ci_low"]),
                                                 float(results[name]["log_loss_ci_high"])],
                                   rtol=0, atol=1e-12)
    for row in contrasts:
        a, b = row["reference"], row["candidate"]
        differences = np.asarray(draws[a]) - np.asarray(draws[b])
        low, high = np.quantile(differences, [.025, .975])
        np.testing.assert_allclose(
            [np.mean(losses[a]-losses[b]), low, high, np.std(differences, ddof=1)],
            [float(row["reference_minus_candidate"]), float(row["ci_low"]),
             float(row["ci_high"]), float(row["bootstrap_se"])], rtol=0, atol=1e-12)
    try:
        evaluate()
    except FileExistsError:
        pass
    else:
        raise ValueError("G4 single-evaluation guard did not trigger.")
    pages = PdfReader(HERE / "presentation_g4_results.pdf").pages
    if len(pages) != 1 or "not an independent or causal test" not in pages[0].extract_text():
        raise ValueError("G4 presentation is missing its exploratory limitation.")
    print(f"Verified {len(records)} predictions, {len(results)} models, {len(contrasts)} contrasts, "
          f"5000 independently replayed held-out-cat resamples, freeze {freeze['status']}, "
          "and the single-evaluation guard.")


if __name__ == "__main__":
    verify()
