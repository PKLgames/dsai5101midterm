"""Reconcile saved G3 predictions without rereading raw test outcomes."""

from __future__ import annotations

import csv
import json

import numpy as np
import pyarrow.parquet as pq
from pypdf import PdfReader
from scipy.special import expit

from common import EXPERIMENT, digest, load_structure, predict
from evaluate import evaluate, verify_freeze


def verify() -> None:
    freeze = verify_freeze()
    models = json.loads((EXPERIMENT / "selected_models.json").read_text(encoding="utf-8"))
    provenance = json.loads((EXPERIMENT / "evaluation_provenance.json").read_text(encoding="utf-8"))
    if provenance["freeze_sha256"] != digest(EXPERIMENT / "freeze.json"):
        raise ValueError("Evaluation provenance freeze mismatch.")
    for name, expected in provenance["output_sha256"].items():
        if digest(EXPERIMENT / name) != expected:
            raise ValueError(f"Evaluation output hash mismatch: {name}")
    with (EXPERIMENT / "test_results.csv").open(encoding="utf-8", newline="") as stream:
        results = {r["model"]: r for r in csv.DictReader(stream)}
    with (EXPERIMENT / "test_contrasts.csv").open(encoding="utf-8", newline="") as stream:
        contrasts = list(csv.DictReader(stream))
    records = pq.read_table(EXPERIMENT / "test_predictions.parquet").to_pylist()
    structure = load_structure("test")
    reconstruction = predict(models, structure)
    if set(results) != set(reconstruction):
        raise ValueError("Result models differ from frozen selected models.")
    ordered_ids = [r["train_row_id"] for r in structure]
    reference = None
    losses = {}
    for name in sorted(results):
        selected = [r for r in records if r["model"] == name]
        if [r["train_row_id"] for r in selected] != ordered_ids:
            raise ValueError(f"Saved prediction IDs differ for {name}.")
        alignment = [(r["train_row_id"], r["outcome_a"], r["held_out_photo"], r["pair_id"])
                     for r in selected]
        if reference is not None and alignment != reference:
            raise ValueError(f"Matched model populations differ for {name}.")
        reference = alignment
        y = np.asarray([r["outcome_a"] for r in selected], dtype=np.float64)
        logits = np.asarray([r["logit"] for r in selected])
        np.testing.assert_allclose(logits, reconstruction[name], rtol=0, atol=1e-14)
        probability = expit(logits)
        np.testing.assert_allclose(probability, [r["prob_a"] for r in selected], rtol=0, atol=1e-14)
        loss = y * np.logaddexp(0., -logits) + (1.-y) * np.logaddexp(0., logits)
        np.testing.assert_allclose(loss, [r["loss"] for r in selected], rtol=0, atol=1e-14)
        expected = results[name]
        actual = (loss.mean(), np.mean((probability-y)**2), np.mean((probability >= .5) == y))
        np.testing.assert_allclose(actual, [float(expected["log_loss"]), float(expected["brier"]),
                                            float(expected["accuracy"])], rtol=0, atol=1e-14)
        losses[name] = loss
    if reference is None or len(records) != len(results) * len(structure):
        raise ValueError("Unexpected saved prediction count.")
    clusters = [row[2] for row in reference]
    unique = sorted(set(clusters))
    lookup = {photo: i for i, photo in enumerate(unique)}
    rng = np.random.default_rng(20261101)
    draws = {name: [] for name in losses}
    for _ in range(5000):
        sampled = rng.integers(0, len(unique), len(unique))
        multiplicity = np.bincount(sampled, minlength=len(unique))
        weights = np.asarray([multiplicity[lookup[photo]] for photo in clusters])
        for name, loss in losses.items():
            draws[name].append(float(weights @ loss / weights.sum()))
    for name in losses:
        low, high = np.quantile(draws[name], [0.025, 0.975])
        np.testing.assert_allclose([low, high], [float(results[name]["log_loss_ci_low"]),
                                                 float(results[name]["log_loss_ci_high"])],
                                   rtol=0, atol=1e-12)
    for row in contrasts:
        a, b = row["reference"], row["candidate"]
        differences = np.asarray(draws[a]) - np.asarray(draws[b])
        low, high = np.quantile(differences, [0.025, 0.975])
        np.testing.assert_allclose(
            [np.mean(losses[a]-losses[b]), low, high, np.std(differences, ddof=1)],
            [float(row["reference_minus_candidate"]), float(row["ci_low"]),
             float(row["ci_high"]), float(row["bootstrap_se"])], rtol=0, atol=1e-12)
    try:
        evaluate()
    except FileExistsError:
        pass
    else:
        raise ValueError("Single-evaluation guard did not trigger.")
    pdf = PdfReader(EXPERIMENT / "presentation_g3_results.pdf")
    if len(pdf.pages) != 1 or "not an independent test" not in pdf.pages[0].extract_text():
        raise ValueError("Presentation PDF is missing its exploratory limitation.")
    print(f"Verified {len(records)} predictions, {len(results)} model results, "
          f"{len(contrasts)} paired contrasts, 5000 independent bootstrap replays, "
          f"freeze {freeze['status']}, and the single-evaluation guard.")


if __name__ == "__main__":
    verify()
