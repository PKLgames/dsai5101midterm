"""Independently replay within-cat results from saved G3 predictions."""

from __future__ import annotations

import csv
import json

import numpy as np
import pyarrow.parquet as pq
from pypdf import PdfReader
from scipy.special import expit

from analyze import G3, HERE, MODELS, ROOT, digest, load_structure, verify_inputs


def verify() -> None:
    verify_inputs()
    provenance = json.loads((HERE / "analysis_provenance.json").read_text(encoding="utf-8"))
    for name, expected in provenance["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Subgroup source hash mismatch: {name}")
    for name, expected in provenance["output_sha256"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"Subgroup output hash mismatch: {name}")
    structure = load_structure("test")
    subgroup = {r["train_row_id"]: r for r in structure if r["a_cat"] == r["b_cat"]}
    if any((r["a_cat"] == r["b_cat"]) != (r["comparison_type"] == "within_cat") for r in structure):
        raise ValueError("Comparison type and cat IDs disagree.")
    records = pq.read_table(G3 / "test_predictions.parquet").to_pylist()
    with (HERE / "within_cat_test_results.csv").open(encoding="utf-8", newline="") as stream:
        results = {r["model"]: r for r in csv.DictReader(stream)}
    with (HERE / "within_cat_contrasts.csv").open(encoding="utf-8", newline="") as stream:
        contrasts = list(csv.DictReader(stream))
    summary = json.loads((HERE / "within_cat_summary.json").read_text(encoding="utf-8"))
    if set(results) != set(MODELS) or len(contrasts) != 5:
        raise ValueError("Missing or unexpected model results or comparisons.")
    ids = sorted(subgroup)
    heldout = [subgroup[key]["held_out_photo"] for key in ids]
    if (len(ids), len({r["pair_id"] for r in subgroup.values()}), len(set(heldout))) != (
            summary["responses"], summary["pairs"], summary["held_out_photos"]):
        raise ValueError("Subgroup counts do not match structural data.")
    arrays = {}
    outcomes = None
    for name in MODELS:
        rows = [r for r in records if r["model"] == name and r["train_row_id"] in subgroup]
        index = {r["train_row_id"]: r for r in rows}
        if len(rows) != len(ids) or set(index) != set(ids):
            raise ValueError(f"Missing, duplicate, or extra within-cat prediction for {name}.")
        ordered = [index[key] for key in ids]
        y = np.asarray([r["outcome_a"] for r in ordered])
        if outcomes is not None and not np.array_equal(y, outcomes):
            raise ValueError("Outcomes differ across model predictions.")
        outcomes = y
        z = np.asarray([r["logit"] for r in ordered])
        p = expit(z)
        loss = y * np.logaddexp(0., -z) + (1.-y) * np.logaddexp(0., z)
        np.testing.assert_allclose(p, [r["prob_a"] for r in ordered], rtol=0, atol=1e-13)
        np.testing.assert_allclose(loss, [r["loss"] for r in ordered], rtol=0, atol=1e-13)
        if name == "M1":
            np.testing.assert_allclose(z, summary["m1_within_cat_constant_logit"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            [loss.mean(), np.mean((p-y)**2), np.mean((p >= .5) == y)],
            [float(results[name]["log_loss"]), float(results[name]["brier"]), float(results[name]["accuracy"])],
            rtol=0, atol=1e-13)
        arrays[name] = loss
    clusters = sorted(set(heldout))
    lookup = {photo: i for i, photo in enumerate(clusters)}
    rng = np.random.default_rng(20261101)
    draws = rng.integers(0, len(clusters), size=(5000, len(clusters)))
    multiplicities = np.zeros((5000, len(clusters)), dtype=int)
    np.add.at(multiplicities, (np.arange(5000)[:, None], draws), 1)
    row_weights = multiplicities[:, [lookup[p] for p in heldout]]
    denominators = row_weights.sum(axis=1)
    if np.any(denominators == 0):
        raise ValueError("Bootstrap draw has no responses.")
    samples = {name: row_weights @ loss / denominators for name, loss in arrays.items()}
    for name in MODELS:
        np.testing.assert_allclose(np.quantile(samples[name], [.025, .975]),
                                   [float(results[name]["ci_low"]), float(results[name]["ci_high"])],
                                   rtol=0, atol=1e-12)
    for row in contrasts:
        a, b = row["reference"], row["candidate"]
        differences = samples[a] - samples[b]
        low, high = np.quantile(differences, [.025, .975])
        np.testing.assert_allclose(
            [np.mean(arrays[a]-arrays[b]), low, high, np.std(differences, ddof=1)],
            [float(row["reference_minus_candidate"]), float(row["ci_low"]),
             float(row["ci_high"]), float(row["bootstrap_se"])], rtol=0, atol=1e-12)
    pages = PdfReader(HERE / "presentation_within_cat_g3.pdf").pages
    if len(pages) != 1 or "not independent or causal" not in pages[0].extract_text():
        raise ValueError("Missing presentation limitation.")
    print(f"Verified {len(ids)} matched responses for {len(MODELS)} models, "
          f"{len(contrasts)} paired comparisons and 5000 independently replayed cluster resamples.")


if __name__ == "__main__":
    verify()
