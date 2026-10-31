"""Quarantined G3/G4 refits and comparisons on existing reserved populations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit

from common import EXPERIMENT, HERE, ROOT, digest, verify_inputs, write_csv, write_json
from experiment import DEMOGRAPHICS, REQUESTED, category, design, side_matrix
from src.development import feature_differences
from src.evaluate import pair_bootstrap
from src.preprocessing import transform_photos
from trainer import cat_differences, train_models


def source_rows(target: str, allocation: str, with_outcomes: bool) -> list[dict[str, Any]]:
    source = ROOT / (target.lower() + "_experiment")
    with (source / "row_allocation.csv").open(encoding="utf-8", newline="") as stream:
        selected = {}
        for entry in csv.DictReader(stream):
            if entry["allocation"] == allocation:
                if entry["train_row_id"] in selected:
                    raise ValueError("Duplicate reserved allocation ID.")
                selected[entry["train_row_id"]] = entry
    fields = ["train_row_id", "partition", "pair_id", "a_image", "b_image", "a_cat", "b_cat",
              "comparison_type", *DEMOGRAPHICS]
    wanted = list(selected)
    if not wanted:
        raise ValueError("Reserved allocation is empty.")
    rows = pq.read_table(ROOT / "artifacts" / "rows_canonical.parquet", columns=fields,
                         filters=[("train_row_id", "in", wanted)]).to_pylist()
    if len(rows) != len(wanted) or len({r["train_row_id"] for r in rows}) != len(wanted):
        raise ValueError("Reserved allocation has missing/duplicate structural IDs.")
    for row in rows:
        allocation_row = selected[row["train_row_id"]]
        if row["pair_id"] != allocation_row["pair_id"]:
            raise ValueError("Reserved allocation pair mismatch.")
        for name in DEMOGRAPHICS:
            category(row, name)
        row["held_out_photo"] = allocation_row.get("held_out_photo", "")
        row["held_out_cat"] = allocation_row.get("held_out_cat", "")
    if with_outcomes:
        dev_ids = [r["train_row_id"] for r in rows if r["partition"] in ("fit", "validation")]
        g2_ids = [r["train_row_id"] for r in rows if r["partition"] == "test"]
        outcomes = {}
        if dev_ids:
            development = pq.read_table(ROOT / "artifacts" / "rows_canonical.parquet",
                                        columns=["train_row_id", "y_displayed"],
                                        filters=[("train_row_id", "in", dev_ids)]).to_pylist()
            outcomes.update({r["train_row_id"]: r["y_displayed"] for r in development})
        if g2_ids:
            exposed = pq.read_table(ROOT / "artifacts" / "test_predictions.parquet",
                                    columns=["train_row_id", "outcome"],
                                    filters=[("population", "=", "common"), ("model_id", "=", "common:M0"),
                                             ("train_row_id", "in", g2_ids)]).to_pylist()
            outcomes.update({r["train_row_id"]: r["outcome"] for r in exposed})
        if set(outcomes) != set(wanted) or any(value not in (0, 1) for value in outcomes.values()):
            raise ValueError("Previously saved fitting/validation outcomes are incomplete or invalid.")
        for row in rows:
            row["y_displayed"] = outcomes[row["train_row_id"]]
    return rows


def predict_transfer(rows: list[dict[str, Any]], models: dict[str, Any], g4: bool) -> dict[str, np.ndarray]:
    sides = side_matrix(rows, models["encoding"])
    differences = {"Cat": cat_differences(rows, models["cat_keys"], g4)}
    images = sorted({r[f"{side}_image"] for r in rows for side in ("a", "b")})
    for block in ("Y", "C"):
        features = pq.read_table(ROOT / "features" / f"{block}.parquet").to_pylist()
        index = {r["image_id"]: r for r in features}
        if len(index) != len(features) or not set(images) <= set(index):
            raise ValueError("Transfer features missing or duplicate.")
        vectors = transform_photos(models["preprocessing"][block], [index[image] for image in images])
        differences[block] = feature_differences(rows, images, vectors)
    logits = {}
    for name, model in models["selected"].items():
        if model["pooled"]:
            logits[name] = logits[model["pooled_from"]].copy()
        else:
            branch = model["branch"]
            x = design(sides, None if branch == "D" else differences[branch], model["interacting"])
            logits[name] = np.asarray(x @ np.asarray(model["coefficients"])).ravel()
        if logits[name].shape != (len(rows),) or not np.isfinite(logits[name]).all():
            raise ValueError(f"Transfer predictions missing: {name}")
    return logits


def develop(target: str) -> None:
    verify_inputs()
    base = target.lower()
    candidates, selected = HERE / f"{base}_attempts.csv", HERE / f"{base}_models.json"
    freeze_path = HERE / f"{base}_freeze.json"
    if any(p.exists() for p in (candidates, selected, freeze_path, HERE / f"{base}_freeze.sha256")):
        raise FileExistsError("Transfer development already exists.")
    parent = ROOT / (base + "_experiment")
    parent_freeze = json.loads((parent / "freeze.json").read_text(encoding="utf-8"))
    if digest(parent / "freeze.json") != (parent / "freeze.sha256").read_text(encoding="ascii").strip():
        raise ValueError("Reserved experiment seal mismatch.")
    for name, expected in parent_freeze["experiment_sha256"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"Reserved experiment source changed: {name}")
    fit, val = source_rows(target, "core_fit", True), source_rows(target, "validation", True)
    model = train_models(fit, val, candidates, g4=(target == "G4"))
    with candidates.open(encoding="utf-8", newline="") as stream:
        attempts = list(csv.DictReader(stream))
    if any(r["accepted"] != "True" for r in attempts):
        raise RuntimeError("Transfer fit failed numerical acceptance; inspect attempt log.")
    model.update({"target": target, "previous_g2_test_responses_in_fit": sum(r["partition"] == "test" for r in fit),
                  "previous_g2_test_responses_in_validation": sum(r["partition"] == "test" for r in val),
                  "reserved_test_outcomes_read_during_development": False,
                  "candidate_log_sha256": digest(candidates)})
    write_json(selected, model)
    frozen = {"status": "frozen_before_reserved_test_saved_outcome_reanalysis", "target": target,
              "selected_models_sha256": digest(selected), "attempts_sha256": digest(candidates),
              "source_code_sha256": {p.name: digest(p) for p in (HERE/"trainer.py", HERE/"transfer.py")},
              "source_reservation_freeze_sha256": digest(parent/"freeze.json"),
              "original_demographic_freeze_sha256": digest(EXPERIMENT/"freeze.json"),
              "prior_exposure": "Reserved outcomes were previously exposed in G2/G3/G4; not an independent test."}
    write_json(freeze_path, frozen)
    (HERE / f"{base}_freeze.sha256").write_text(digest(freeze_path) + "\n", encoding="ascii")
    print(f"{target}: separate fit/validation complete; freeze sealed.", flush=True)


def evaluate(target: str) -> None:
    verify_inputs()
    base = target.lower()
    parent = ROOT / (base + "_experiment")
    freeze_path = HERE / f"{base}_freeze.json"
    if any((HERE / f"{base}_{name}").exists() for name in ("evaluation_started.json", "results.csv", "contrasts.csv", "predictions.parquet", "provenance.json")):
        raise FileExistsError("Transfer test analysis already started.")
    if digest(freeze_path) != (HERE / f"{base}_freeze.sha256").read_text(encoding="ascii").strip():
        raise ValueError("Transfer freeze seal mismatch.")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    for name, expected in frozen["source_code_sha256"].items():
        if digest(HERE / name) != expected:
            raise ValueError("Frozen transfer source changed.")
    if (digest(HERE / f"{base}_models.json") != frozen["selected_models_sha256"]
            or digest(HERE / f"{base}_attempts.csv") != frozen["attempts_sha256"]
            or digest(parent / "freeze.json") != frozen["source_reservation_freeze_sha256"]
            or digest(EXPERIMENT / "freeze.json") != frozen["original_demographic_freeze_sha256"]):
        raise ValueError("Transfer freeze source mismatch.")
    provenance = json.loads((parent / "evaluation_provenance.json").read_text(encoding="utf-8"))
    prediction_path = parent / "test_predictions.parquet"
    if digest(prediction_path) != provenance["output_sha256"]["test_predictions.parquet"]:
        raise ValueError("Original reserved predictions changed.")
    rows = source_rows(target, "test", False)
    models = json.loads((HERE / f"{base}_models.json").read_text(encoding="utf-8"))
    logits = predict_transfer(rows, models, target == "G4")
    write_json(HERE / f"{base}_evaluation_started.json", {
        "status": "existing_reserved_test_outcomes_reanalysis_started", "freeze_sha256": digest(freeze_path),
        "raw_test_outcomes_read": False, "limitation": "Retrospective; outcome access occurred in earlier experiments."})
    frozen_models = ("M0", "M1", "M3-Y", "M3-C", "M4-C") if target == "G3" else (
        "M0", "M1-zero", "M3-Y", "M3-C", "M4-C")
    saved = pq.read_table(prediction_path, filters=[("model", "in", list(frozen_models))]).to_pylist()
    lookup = {name: {} for name in frozen_models}
    for row in saved:
        name, key = row["model"], row["train_row_id"]
        if key in lookup[name]:
            raise ValueError("Duplicate original transfer prediction.")
        lookup[name][key] = row
    ids = [row["train_row_id"] for row in rows]
    for name in frozen_models:
        if set(lookup[name]) != set(ids):
            raise ValueError(f"Original transfer comparator has mismatched rows: {name}")
        if any(lookup[name][key]["pair_id"] != row["pair_id"] for key, row in zip(ids, rows)):
            raise ValueError("Transfer pair mismatch.")
        logits["original:" + name] = np.asarray([lookup[name][key]["logit"] for key in ids])
    y = np.asarray([lookup["M0"][key]["outcome_a"] for key in ids], dtype=float)
    if any([lookup[name][key]["outcome_a"] for key in ids] != y.tolist() for name in frozen_models):
        raise ValueError("Saved reserved-test outcomes differ among original models.")
    losses = {name: y*np.logaddexp(0., -z)+(1-y)*np.logaddexp(0., z) for name, z in logits.items()}
    for name in frozen_models:
        np.testing.assert_allclose(losses["original:" + name], [lookup[name][key]["loss"] for key in ids], rtol=0, atol=1e-12)
    cluster_field = "held_out_photo" if target == "G3" else "held_out_cat"
    clusters = [row[cluster_field] for row in rows]
    if not all(clusters):
        raise ValueError("Reserved bootstrap cluster missing.")
    samples, cluster_weighted = pair_bootstrap(clusters, losses, seed=20261101, replicates=5000)
    results, predictions = [], []
    for name, z in logits.items():
        p = expit(z)
        low, high = np.quantile(samples[name], [.025, .975])
        results.append({"model": name, "responses": len(rows), "pairs": len({r["pair_id"] for r in rows}),
                        "held_out_clusters": len(set(clusters)), "cluster_unit": cluster_field,
                        "log_loss": float(losses[name].mean()), "ci_low": float(low), "ci_high": float(high),
                        "brier": float(np.mean((p-y)**2)), "accuracy": float(np.mean((p >= .5) == y))})
        for i, row in enumerate(rows):
            predictions.append({"model": name, "train_row_id": row["train_row_id"], "pair_id": row["pair_id"],
                                "cluster": clusters[i], "outcome": int(y[i]), "logit": float(z[i]),
                                "probability": float(p[i]), "loss": float(losses[name][i])})
    cat_reference = "original:M1" if target == "G3" else "original:M1-zero"
    comparisons = [("original:M0", "D"), (cat_reference, "Cat+D"), ("original:M3-Y", "Y+D"),
                   ("original:M3-C", "C+D"), ("original:M4-C", "C+D")]
    comparisons += [(branch + "+side", branch + "+D") for branch in ("Cat", "Y", "C")]
    contrasts = []
    for a, b in comparisons:
        difference = samples[a]-samples[b]
        low, high = np.quantile(difference, [.025, .975])
        contrasts.append({"reference": a, "candidate": b, "difference": float(np.mean(losses[a]-losses[b])),
                          "ci_low": float(low), "ci_high": float(high), "bootstrap_se": float(np.std(difference, ddof=1)),
                          "cluster_unit": cluster_field, "held_out_clusters": len(set(clusters))})
    results_path, contrasts_path = HERE / f"{base}_results.csv", HERE / f"{base}_contrasts.csv"
    prediction_output = HERE / f"{base}_predictions.parquet"
    write_csv(results_path, results, list(results[0]))
    write_csv(contrasts_path, contrasts, list(contrasts[0]))
    pq.write_table(pa.Table.from_pylist(predictions), prediction_output, compression="zstd")
    write_json(HERE / f"{base}_provenance.json", {
        "status": "retrospective_transfer_completed", "freeze_sha256": digest(freeze_path),
        "source_predictions_sha256": digest(prediction_path), "raw_test_outcomes_read": False,
        "outputs_sha256": {p.name: digest(p) for p in (results_path, contrasts_path, prediction_output)},
        "responses": len(rows), "pairs": len({r["pair_id"] for r in rows}),
        "clusters": len(set(clusters)), "cluster_unit": cluster_field, "bootstrap_replicates": 5000,
        "limitations": "Post-test retrospective; held-out photo/cat bootstrap does not account for all dependence; 95% pointwise unadjusted intervals.",
        "g4_unseen_cat_zero": target == "G4", "deployment_changed": False})
    print(f"{target}: saved-test reanalysis complete, {len(rows)} responses, {len(set(clusters))} clusters.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=("G3", "G4"))
    parser.add_argument("stage", choices=("develop", "evaluate"))
    args = parser.parse_args()
    {"develop": develop, "evaluate": evaluate}[args.stage](args.target)
