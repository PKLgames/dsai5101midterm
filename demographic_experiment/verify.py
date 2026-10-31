"""Independent replay of saved exploratory results, not a new evaluation."""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import pyarrow.parquet as pq
from pypdf import PdfReader
from scipy.special import expit

from experiment import (EXCLUDED, FROZEN_MODELS, HERE, ROOT, digest, evaluate, load_rows,
                        predict, score, snapshot, verify_freeze, write_json)


def verify() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    if (HERE / "verification.json").exists():
        raise FileExistsError("Verification already exists.")
    verify_freeze()
    provenance = json.loads((HERE / "evaluation_provenance.json").read_text(encoding="utf-8"))
    if provenance["freeze_sha256"] != digest(HERE / "freeze.json"):
        raise ValueError("Evaluation freeze mismatch.")
    for name, expected in provenance["outputs_sha256"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"Output hash mismatch: {name}")
    models = json.loads((HERE / "selected_models.json").read_text(encoding="utf-8"))
    val = load_rows("validation")
    yval = np.asarray([r["y_displayed"] for r in val], dtype=float)
    validation_predictions = predict(val, models)
    for name, logits in validation_predictions.items():
        if abs(score(yval, logits)[0] - models["selected"][name]["validation_loss"]) > 1e-12:
            raise ValueError("Validation model replay mismatch.")
    test = load_rows("test")
    reconstruction = predict(test, models)
    saved = pq.read_table(HERE / "test_predictions.parquet").to_pylist()
    with (HERE / "test_results.csv").open(encoding="utf-8", newline="") as stream:
        results = {r["model"]: r for r in csv.DictReader(stream)}
    with (HERE / "test_contrasts.csv").open(encoding="utf-8", newline="") as stream:
        contrasts = list(csv.DictReader(stream))
    if len(saved) != len(test)*len(results) or set(results) != set(reconstruction) | set(FROZEN_MODELS):
        raise ValueError("Result/prediction population shape mismatch.")
    ids = [r["train_row_id"] for r in test]
    loss_by_model = {}
    outcomes = None
    for name, row in results.items():
        subset = [r for r in saved if r["model"] == name]
        if [r["train_row_id"] for r in subset] != ids:
            raise ValueError("Matched IDs are incomplete or reordered.")
        yy = np.asarray([r["outcome"] for r in subset])
        if outcomes is not None and not np.array_equal(outcomes, yy):
            raise ValueError("Matched outcomes differ.")
        outcomes = yy
        zz = np.asarray([r["logit"] for r in subset])
        if name in reconstruction:
            np.testing.assert_allclose(zz, reconstruction[name], rtol=0, atol=1e-13)
        pp = expit(zz)
        losses = yy*np.logaddexp(0., -zz)+(1.-yy)*np.logaddexp(0., zz)
        np.testing.assert_allclose(losses, [r["loss"] for r in subset], rtol=0, atol=1e-13)
        np.testing.assert_allclose(pp, [r["probability"] for r in subset], rtol=0, atol=1e-13)
        np.testing.assert_allclose([losses.mean(), np.mean((pp-yy)**2), np.mean((pp >= .5) == yy)],
                                   [float(row["log_loss"]), float(row["brier"]), float(row["accuracy"])],
                                   rtol=0, atol=1e-12)
        loss_by_model[name] = losses
    pairs = sorted({r["pair_id"] for r in test})
    mapping = {pair: i for i, pair in enumerate(pairs)}
    codes = np.asarray([mapping[r["pair_id"]] for r in test])
    counts = np.bincount(codes)
    names = list(results)
    sums = np.stack([np.bincount(codes, weights=loss_by_model[name]) for name in names])
    means = sums / counts
    rng = np.random.default_rng(20261101)
    response_draws, pair_draws = [], []
    for _ in range(5000):
        chosen = rng.integers(0, len(pairs), len(pairs))
        weights = np.bincount(chosen, minlength=len(pairs))
        response_draws.append(sums @ weights / (counts @ weights))
        pair_draws.append(means @ weights / len(pairs))
    draws, pd = np.asarray(response_draws), np.asarray(pair_draws)
    for i, name in enumerate(names):
        row = results[name]
        np.testing.assert_allclose(np.quantile(draws[:, i], [.025, .975]),
                                   [float(row["ci_low"]), float(row["ci_high"])], rtol=0, atol=1e-12)
        np.testing.assert_allclose([means[i].mean(), *np.quantile(pd[:, i], [.025, .975])],
                                   [float(row["pair_log_loss"]), float(row["pair_ci_low"]), float(row["pair_ci_high"])],
                                   rtol=0, atol=1e-12)
    for row in contrasts:
        ai, bi = names.index(row["reference"]), names.index(row["candidate"])
        difference = draws[:, ai]-draws[:, bi]
        np.testing.assert_allclose([*np.quantile(difference, [.025, .975]), np.std(difference, ddof=1)],
                                   [float(row["ci_low"]), float(row["ci_high"]), float(row["bootstrap_se"])],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.quantile(pd[:, ai]-pd[:, bi], [.025, .975]),
                                   [float(row["pair_ci_low"]), float(row["pair_ci_high"])], rtol=0, atol=1e-12)
    for branch in ("Cat", "Y", "C"):
        if models["selected"][branch+"+D"]["pooled"]:
            np.testing.assert_array_equal(reconstruction[branch+"+D"], reconstruction[branch+"+side"])
    try:
        evaluate()
    except FileExistsError:
        pass
    else:
        raise ValueError("One-evaluation guard did not trigger.")
    pdf = PdfReader(HERE / "presentation_demographic_models.pdf")
    if len(pdf.pages) != 2 or not all("Not Causal Effect" in page.extract_text() for page in pdf.pages):
        raise ValueError("Presentation layout/limitations missing.")
    before = json.loads((HERE / "original_hashes_before.json").read_text(encoding="utf-8"))
    if snapshot(ROOT, EXCLUDED) != before:
        raise RuntimeError("Original artifacts changed during verification.")
    report = {"status": "independent_replay_passed", "matched_test_responses": len(test),
              "models_verified": len(results), "selected_new_models": len(reconstruction),
              "paired_contrasts_verified": len(contrasts), "bootstrap_replicates_replayed": 5000,
              "exact_pooled_fallbacks_verified": True, "validation_predictions_replayed": True,
              "single_evaluation_guard_verified": True, "original_files_unchanged": len(before),
              "pdf_pages": len(pdf.pages), "raw_test_outcomes_parsed": False,
              "prediction_sha256": digest(HERE / "test_predictions.parquet")}
    write_json(HERE / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    verify()
