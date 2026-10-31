"""Quarantined all-demographic predictive associations on original G2."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from scipy.special import expit
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from g3_experiment.prepare import digest, snapshot, write_csv, write_json  # noqa: E402
from src.audit import CONTEXT  # noqa: E402
from src.development import choose, feature_differences, identity_matrix, score  # noqa: E402
from src.estimator import context_mapping, fit_logistic  # noqa: E402
from src.evaluate import pair_bootstrap  # noqa: E402
from src.freeze import validate_freeze  # noqa: E402
from src.preprocessing import fit_preprocessing, transform_photos  # noqa: E402

DEMOGRAPHICS = ("gender", "pet_history", "pet_care", "pet_contact", "future_pet", "mbti_ie")
EXCLUDED = {".git", ".venv", ".cache", "Presentation_Evidence", "demographic_experiment"}
FROZEN_MODELS = ("common:M0", "common:M1", "M3-Y", "M3-C", "M4-C", "M3-side", "M4-side")
REQUESTED = ("D", "Cat+D", "Y+D", "C+D")


def verify_original() -> None:
    validate_freeze(ROOT)
    for directory in ("g3_experiment", "g4_experiment"):
        folder = ROOT / directory
        freeze = json.loads((folder / "freeze.json").read_text(encoding="utf-8"))
        if digest(folder / "freeze.json") != (folder / "freeze.sha256").read_text(encoding="ascii").strip():
            raise ValueError(f"{directory} freeze seal mismatch.")
        for name, expected in freeze["experiment_sha256"].items():
            if digest(folder / name) != expected:
                raise ValueError(f"{directory} frozen file changed: {name}")
    evaluation = json.loads((ROOT / "artifacts" / "evaluation_provenance.json").read_text(encoding="utf-8"))
    if evaluation["freeze_sha256"] != digest(ROOT / "artifacts" / "model_freeze.json"):
        raise ValueError("Original evaluation/freeze mismatch.")
    for name, expected in evaluation["output_sha256"].items():
        if digest(ROOT.joinpath(*name.split("/"))) != expected:
            raise ValueError(f"Original evaluation output changed: {name}")


def category(row: dict[str, Any], name: str) -> str:
    value = row[name]
    if not isinstance(value, str):
        raise ValueError(f"Demographic must be a string: {name}")
    if value == "":
        return "<missing>"
    if value == "<missing>" or (name in CONTEXT and value not in CONTEXT[name]):
        raise ValueError(f"Invalid demographic code: {name}={value}")
    return value


def load_rows(partition: str) -> list[dict[str, Any]]:
    if partition not in {"fit", "validation", "test"}:
        raise ValueError("Invalid original partition.")
    fields = ["train_row_id", "partition", "pair_id", "a_image", "b_image", "a_cat", "b_cat",
              "comparison_type", *DEMOGRAPHICS]
    if partition != "test":
        fields.append("y_displayed")
    rows = pq.read_table(ROOT / "artifacts" / "rows_canonical.parquet", columns=fields,
                         filters=[("partition", "=", partition)]).to_pylist()
    masks = pq.read_table(ROOT / "artifacts" / "g2_eligibility.parquet",
                          filters=[("partition", "=", partition)]).to_pylist()
    index = {r["train_row_id"]: r for r in masks}
    if len(index) != len(masks) or len(rows) != len(index) or len({r["train_row_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate or incomplete original partition masks.")
    selected = []
    key = "common_visual_fit" if partition == "fit" else "common_strict_g2"
    for row in rows:
        if row["train_row_id"] not in index or index[row["train_row_id"]]["partition"] != partition:
            raise ValueError("Response/partition mask mismatch.")
        for name in DEMOGRAPHICS:
            category(row, name)
        if partition != "test" and row["y_displayed"] not in (0, 1):
            raise ValueError("Development outcomes are invalid.")
        if index[row["train_row_id"]][key]:
            selected.append(row)
    if not selected:
        raise ValueError("Empty matched original population.")
    return selected


def fit_encoding(rows: list[dict[str, Any]]) -> dict[str, Any]:
    specification = {}
    pairs = [r["pair_id"] for r in rows]
    for name in DEMOGRAPHICS:
        values = [category(r, name) for r in rows]
        reference, deviations = context_mapping(values, pairs)
        counts = Counter(values)
        specification[name] = {"reference": reference, "deviations": list(deviations),
                               "fitting_counts": dict(sorted(counts.items())),
                               "fitting_pairs": {value: len({r["pair_id"] for r in rows if category(r, name) == value})
                                                 for value in sorted(counts)}}
    return specification


def side_matrix(rows: list[dict[str, Any]], encoding: dict[str, Any]) -> sparse.csr_matrix:
    columns = []
    for name in DEMOGRAPHICS:
        values = np.asarray([category(r, name) for r in rows])
        columns.extend((values == level).astype(float) for level in encoding[name]["deviations"])
    return sparse.csr_matrix(np.column_stack(columns)) if columns else sparse.csr_matrix((len(rows), 0))


def design(side: sparse.csr_matrix, differences: sparse.csr_matrix | None, interacting: bool) -> sparse.csr_matrix:
    blocks = [sparse.csr_matrix(np.ones((side.shape[0], 1))), side]
    if differences is not None:
        if side.shape[0] != differences.shape[0]:
            raise ValueError("Demographic/image design rows mismatch.")
        blocks.append(differences)
        if interacting:
            blocks.extend(sparse.diags(side[:, i].toarray().ravel()) @ differences for i in range(side.shape[1]))
    elif interacting:
        raise ValueError("Demographics-only cannot have image score interactions.")
    return sparse.hstack(blocks, format="csr")


def penalties(side_count: int, dimensions: int, interacting: bool, ridge: float, rho: float) -> np.ndarray:
    return np.r_[[1e-6], np.full(side_count + dimensions, ridge),
                 np.full(side_count * dimensions if interacting else 0, ridge * rho)]


def side_names(encoding: dict[str, Any]) -> list[str]:
    return [f"{name}={value}" for name in DEMOGRAPHICS for value in encoding[name]["deviations"]]


def get_differences(rows: list[dict[str, Any]], models: dict[str, Any]) -> dict[str, sparse.csr_matrix]:
    result = {"Cat": identity_matrix(rows, models["cat_keys"], "cat", side=False)}
    ids = sorted({r[f"{s}_image"] for r in rows for s in ("a", "b")})
    for block in ("Y", "C"):
        records = pq.read_table(ROOT / "features" / f"{block}.parquet").to_pylist()
        by_id = {r["image_id"]: r for r in records}
        if len(by_id) != len(records) or not set(ids) <= set(by_id):
            raise ValueError(f"{block} features incomplete or duplicated.")
        if any(by_id[image]["extraction_status"] != "ok" for image in ids):
            raise ValueError(f"{block} unavailable on the matched population.")
        values = transform_photos(models["preprocessing"][block], [by_id[image] for image in ids])
        result[block] = feature_differences(rows, ids, values)
    return result


def predict(rows: list[dict[str, Any]], models: dict[str, Any]) -> dict[str, np.ndarray]:
    side = side_matrix(rows, models["encoding"])
    diffs = get_differences(rows, models)
    result = {}
    for name, candidate in models["selected"].items():
        if candidate["pooled"]:
            source = candidate["pooled_from"]
            if source not in result:
                raise ValueError("Pooled source must be reconstructed first.")
            result[name] = result[source].copy()
        else:
            branch = candidate["branch"]
            x = design(side, None if branch == "D" else diffs[branch], candidate["interacting"])
            result[name] = np.asarray(x @ np.asarray(candidate["coefficients"])).ravel()
        if result[name].shape != (len(rows),) or not np.isfinite(result[name]).all():
            raise ValueError(f"Incomplete predictions: {name}")
    return result


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def support_table(partitions: dict[str, list[dict[str, Any]]], encoding: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for partition, rows in partitions.items():
        for name in DEMOGRAPHICS:
            values = sorted({category(r, name) for r in rows})
            for value in values:
                subset = [r for r in rows if category(r, name) == value]
                spec = encoding[name]
                result.append({"partition": partition, "demographic": name, "category": value,
                               "responses": len(subset), "pairs": len({r["pair_id"] for r in subset}),
                               "reference": spec["reference"], "estimated_deviation": value in spec["deviations"],
                               "status": "reference" if value == spec["reference"] else
                               "estimated" if value in spec["deviations"] else "unsupported_zero_fallback"})
    return result


def develop() -> None:
    if any((HERE / n).exists() for n in ("original_hashes_before.json", "selected_models.json", "all_fit_attempts.csv", "freeze.json")):
        raise FileExistsError("Development already started; refusing to overwrite.")
    before = snapshot(ROOT, EXCLUDED)
    evidence = snapshot(ROOT / "Presentation_Evidence", set())
    write_json(HERE / "original_hashes_before.json", before)
    write_json(HERE / "evidence_hashes_before.json", evidence)
    verify_original()
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    fit, val = load_rows("fit"), load_rows("validation")
    encoding = fit_encoding(fit)
    images = {r[f"{s}_image"] for r in fit for s in ("a", "b")}
    models = {"encoding": encoding, "preprocessing": {},
              "cat_keys": sorted({r[f"{s}_cat"] for r in fit for s in ("a", "b")}), "selected": {}}
    for block in ("Y", "C"):
        records = pq.read_table(ROOT / "features" / f"{block}.parquet").to_pylist()
        spec = fit_preprocessing(block, records, images)
        old = json.loads((ROOT / "artifacts" / "preprocessing" / f"{block}.json").read_text(encoding="utf-8"))
        if spec != old:
            raise ValueError(f"{block} fit-only preprocessing differs from the original.")
        models["preprocessing"][block] = spec
    write_json(HERE / "development_access.json", {"status": "original_fit_validation_only",
               "fitting_responses": len(fit), "validation_responses": len(val), "test_outcomes_read": False,
               "raw_training_outcomes_parsed": False, "note": "Retrospective analysis; earlier outcome access already occurred."})
    yy = np.asarray([r["y_displayed"] for r in fit], dtype=np.float64)
    vy = np.asarray([r["y_displayed"] for r in val], dtype=np.float64)
    sx, sv = side_matrix(fit, encoding), side_matrix(val, encoding)
    dx, dv = get_differences(fit, models), get_differences(val, models)
    selection = protocol["selection"]
    fields = ["model", "branch", "ridge", "rho", "pooled", "pooled_from", "interacting", "accepted",
              "validation_loss", "train_loss", "brier", "accuracy", "objective", "gradient_max",
              "coefficient_count", "iterations", "seconds", "fit_attempts"]
    with (HERE / "all_fit_attempts.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()

        def log(candidate: dict[str, Any]) -> None:
            row = {k: safe(candidate.get(k)) for k in fields}
            row["fit_attempts"] = json.dumps(safe(candidate["fit_attempts"]), allow_nan=False)
            writer.writerow(row)
            stream.flush()

        def sweep(name: str, branch: str, interacting: bool, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
            candidates = []
            dimensions = 0 if branch == "D" else dx[branch].shape[1]
            x = design(sx, None if branch == "D" else dx[branch], interacting)
            vx = design(sv, None if branch == "D" else dv[branch], interacting)
            if fallback is not None:
                pooled = {**fallback, "model": name, "pooled": True, "pooled_from": fallback["model"],
                          "rho": 1e99, "interacting": False, "seconds": 0., "fit_attempts": []}
                candidates.append(pooled)
                log(pooled)
            for ridge in selection["ridge_grid"]:
                for rho in selection["interaction_multipliers"] if interacting else [1]:
                    started = time.monotonic()
                    fitted = fit_logistic(x, yy, penalties(sx.shape[1], dimensions, interacting, ridge, rho))
                    logits = np.asarray(vx @ fitted.coefficients).ravel() if fitted.accepted else None
                    valid = logits is not None and np.isfinite(logits).all()
                    loss, brier, accuracy = score(vy, logits) if valid else (None, None, None)
                    candidate = {"model": name, "branch": branch, "ridge": ridge, "rho": rho,
                                 "pooled": False, "pooled_from": "", "interacting": interacting,
                                 "accepted": bool(fitted.accepted and valid), "validation_loss": loss,
                                 "train_loss": score(yy, np.asarray(x @ fitted.coefficients).ravel())[0] if fitted.accepted else None,
                                 "brier": brier, "accuracy": accuracy, "objective": safe(fitted.objective),
                                 "gradient_max": safe(fitted.gradient_max), "coefficient_count": x.shape[1],
                                 "iterations": fitted.iterations, "seconds": time.monotonic()-started,
                                 "fit_attempts": list(fitted.attempts), "coefficients": fitted.coefficients.tolist()}
                    candidates.append(candidate)
                    log(candidate)
            winner = choose(candidates, contextual=fallback is not None)
            print(f"{name} validation={winner['validation_loss']:.9f} ridge={winner['ridge']} "
                  f"rho={winner['rho']} pooled={winner['pooled']}", flush=True)
            return winner

        with threadpool_limits(limits=1):
            models["selected"]["D"] = sweep("D", "D", False)
            for branch in ("Cat", "Y", "C"):
                baseline = branch + "+side"
                models["selected"][baseline] = sweep(baseline, branch, False)
                models["selected"][branch + "+D"] = sweep(branch + "+D", branch, True, models["selected"][baseline])
    write_json(HERE / "selected_models.json", models)
    predictions = predict(val, models)
    for name, logits in predictions.items():
        if abs(score(vy, logits)[0] - models["selected"][name]["validation_loss"]) > 1e-12:
            raise ValueError("Selected validation reconstruction failed.")
    rows = [{k: candidate[k] for k in fields if k not in {"fit_attempts"}} for candidate in models["selected"].values()]
    write_csv(HERE / "validation_results.csv", rows, list(rows[0]))
    support = support_table({"fit": fit, "validation": val}, encoding)
    write_csv(HERE / "development_support.csv", support, list(support[0]))
    sources = {p.name: digest(p) for p in HERE.glob("*.py")}
    sources.update({name: digest(HERE / name) for name in ("protocol.json", "selected_models.json",
                   "validation_results.csv", "all_fit_attempts.csv", "development_support.csv", "development_access.json")})
    inputs = {name: record["sha256"] for name, record in before.items() if name.startswith(
        ("src\\", "features\\", "artifacts\\model_freeze", "artifacts\\rows_canonical", "artifacts\\g2_eligibility",
         "artifacts\\preprocessing", "artifacts\\test_predictions", "artifacts\\evaluation_provenance"))}
    freeze = {"status": "frozen_before_saved_test_outcome_analysis", "test_outcomes_read_during_development": False,
              "exploratory": True, "source_output_sha256": sources, "root_input_sha256": inputs}
    write_json(HERE / "freeze.json", freeze)
    (HERE / "freeze.sha256").write_text(digest(HERE / "freeze.json") + "\n", encoding="ascii")


def verify_freeze() -> dict[str, Any]:
    if digest(HERE / "freeze.json") != (HERE / "freeze.sha256").read_text(encoding="ascii").strip():
        raise ValueError("Experiment seal mismatch.")
    frozen = json.loads((HERE / "freeze.json").read_text(encoding="utf-8"))
    if frozen["status"] != "frozen_before_saved_test_outcome_analysis" or frozen["test_outcomes_read_during_development"]:
        raise ValueError("Experiment was not frozen before this test analysis.")
    for name, expected in frozen["source_output_sha256"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"Experiment frozen file changed: {name}")
    for name, expected in frozen["root_input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Original frozen input changed: {name}")
    return frozen


def evaluate() -> None:
    outputs = ("evaluation_started.json", "test_results.csv", "test_contrasts.csv", "test_predictions.parquet",
               "subgroups.csv", "calibration.csv", "coefficients.csv", "test_support.csv", "presentation_demographic_models.pdf",
               "evaluation_provenance.json")
    if any((HERE / name).exists() for name in outputs):
        raise FileExistsError("Test analysis already started; refusing to overwrite.")
    verify_freeze()
    verify_original()
    rows = load_rows("test")
    models = json.loads((HERE / "selected_models.json").read_text(encoding="utf-8"))
    logits = predict(rows, models)
    write_json(HERE / "evaluation_started.json", {"status": "saved_g2_test_outcome_reanalysis_started",
               "freeze_sha256": digest(HERE / "freeze.json"), "raw_test_outcomes_read": False,
               "note": "Existing G2 test outcomes were previously exposed; this is post-test exploratory."})
    saved = pq.read_table(ROOT / "artifacts" / "test_predictions.parquet",
                          filters=[("population", "=", "common"), ("model_id", "in", list(FROZEN_MODELS))]).to_pylist()
    index = {name: {} for name in FROZEN_MODELS}
    for r in saved:
        name, key = r["model_id"], r["train_row_id"]
        if key in index[name]:
            raise ValueError("Duplicate frozen predictions.")
        index[name][key] = r
    ids = [r["train_row_id"] for r in rows]
    for name in FROZEN_MODELS:
        if set(index[name]) != set(ids):
            raise ValueError("Incomplete matched frozen comparison.")
        ordered = [index[name][key] for key in ids]
        if any(r["pair_id"] != source["pair_id"] for r, source in zip(ordered, rows)):
            raise ValueError("Frozen pair IDs disagree.")
        logits[name] = np.asarray([r["logit"] for r in ordered])
    y = np.asarray([index["common:M0"][key]["outcome"] for key in ids], dtype=float)
    if np.any((y != 0) & (y != 1)) or any(
        [index[name][key]["outcome"] for key in ids] != y.tolist() for name in FROZEN_MODELS):
        raise ValueError("Matched frozen outcomes differ.")
    losses = {name: y*np.logaddexp(0., -value)+(1.-y)*np.logaddexp(0., value) for name, value in logits.items()}
    for name in FROZEN_MODELS:
        np.testing.assert_allclose(losses[name], [index[name][key]["loss"] for key in ids], rtol=0, atol=1e-13)
    samples, pair_samples = pair_bootstrap([r["pair_id"] for r in rows], losses, 20261101, 5000)
    pairs = sorted({r["pair_id"] for r in rows})
    pair_index = {key: i for i, key in enumerate(pairs)}
    codes = np.asarray([pair_index[r["pair_id"]] for r in rows])
    counts = np.bincount(codes)
    results, predictions = [], []
    for name, value in logits.items():
        p = expit(value)
        low, high = np.quantile(samples[name], [.025, .975])
        plow, phigh = np.quantile(pair_samples[name], [.025, .975])
        pair_loss = np.mean(np.bincount(codes, weights=losses[name]) / counts)
        candidate = models["selected"].get(name)
        results.append({"model": name, "responses": len(rows), "pairs": len(pairs),
                        "log_loss": float(losses[name].mean()), "ci_low": float(low), "ci_high": float(high),
                        "pair_log_loss": float(pair_loss), "pair_ci_low": float(plow), "pair_ci_high": float(phigh),
                        "brier": float(np.mean((p-y)**2)), "accuracy": float(np.mean((p >= .5) == y)),
                        "validation_loss": candidate["validation_loss"] if candidate else "",
                        "pooled_no_interactions": candidate["pooled"] if candidate else "",
                        "coefficient_count": candidate["coefficient_count"] if candidate else "",
                        "source": "exploratory_new_fit" if candidate else "original_frozen_comparator"})
        for i, row in enumerate(rows):
            predictions.append({"model": name, "train_row_id": row["train_row_id"], "pair_id": row["pair_id"],
                                "outcome": int(y[i]), "logit": float(value[i]), "probability": float(p[i]),
                                "loss": float(losses[name][i])})
    comparisons = [("common:M0", "D"), ("common:M1", "Cat+D"), ("M3-Y", "Y+D"),
                   ("M3-C", "C+D"), ("M4-C", "C+D"), ("M4-side", "C+D")]
    comparisons += [(branch + "+side", branch + "+D") for branch in ("Cat", "Y", "C")]
    contrasts = []
    for a, b in comparisons:
        difference, pdifference = samples[a]-samples[b], pair_samples[a]-pair_samples[b]
        low, high = np.quantile(difference, [.025, .975])
        plow, phigh = np.quantile(pdifference, [.025, .975])
        contrasts.append({"reference": a, "candidate": b, "difference": float(np.mean(losses[a]-losses[b])),
                          "ci_low": float(low), "ci_high": float(high), "bootstrap_se": float(np.std(difference, ddof=1)),
                          "pair_difference": float(results[list(logits).index(a)]["pair_log_loss"]-
                                                   results[list(logits).index(b)]["pair_log_loss"]),
                          "pair_ci_low": float(plow), "pair_ci_high": float(phigh)})
    subgroup_rows, calibration = [], []
    for name in REQUESTED:
        p = expit(logits[name])
        for dimension in ("comparison_type", *DEMOGRAPHICS):
            values = [r["comparison_type"] if dimension == "comparison_type" else category(r, dimension) for r in rows]
            for group in sorted(set(values)):
                take = np.asarray([v == group for v in values])
                pair_count = len({r["pair_id"] for r, yes in zip(rows, take) if yes})
                supported = take.sum() >= 100 and pair_count >= 30
                subgroup_rows.append({"model": name, "dimension": dimension, "group": group,
                                      "responses": int(take.sum()), "pairs": pair_count,
                                      "support_status": "supported" if supported else "low_support",
                                      "log_loss": float(losses[name][take].mean()) if supported else "",
                                      "brier": float(np.mean((p[take]-y[take])**2)) if supported else "",
                                      "accuracy": float(np.mean((p[take] >= .5) == y[take])) if supported else ""})
        bins = np.minimum((p*10).astype(int), 9)
        for b in range(10):
            take = bins == b
            pair_count = len({r["pair_id"] for r, yes in zip(rows, take) if yes})
            calibration.append({"model": name, "bin": b, "responses": int(take.sum()), "pairs": pair_count,
                                "support_status": "supported" if take.sum() >= 30 and pair_count >= 10 else "low_support",
                                "mean_probability": float(p[take].mean()) if take.any() else "",
                                "observed_a_rate": float(y[take].mean()) if take.any() else ""})
    coefficient_rows = []
    sides = side_names(models["encoding"])
    for name in REQUESTED:
        candidate = models["selected"][name]
        branch = candidate["branch"]
        base_names = [] if branch == "D" else models["cat_keys"] if branch == "Cat" else models["preprocessing"][branch]["final_names"]
        labels = ["intercept", *["side:" + label for label in sides], *["score:" + label for label in base_names]]
        if candidate["interacting"]:
            labels.extend(f"interaction:{group}:{feature}" for group in sides for feature in base_names)
        if len(labels) != len(candidate["coefficients"]):
            raise ValueError("Coefficient label length mismatch.")
        coefficient_rows.extend({"model": name, "term": label, "coefficient": value}
                                for label, value in zip(labels, candidate["coefficients"]))
    write_csv(HERE / "test_results.csv", results, list(results[0]))
    write_csv(HERE / "test_contrasts.csv", contrasts, list(contrasts[0]))
    pq.write_table(pa.Table.from_pylist(predictions), HERE / "test_predictions.parquet", compression="zstd")
    write_csv(HERE / "subgroups.csv", subgroup_rows, list(subgroup_rows[0]))
    write_csv(HERE / "calibration.csv", calibration, list(calibration[0]))
    write_csv(HERE / "coefficients.csv", coefficient_rows, list(coefficient_rows[0]))
    support = support_table({"test": rows}, models["encoding"])
    write_csv(HERE / "test_support.csv", support, list(support[0]))
    plot(results, contrasts, calibration)
    before = json.loads((HERE / "original_hashes_before.json").read_text(encoding="utf-8"))
    evidence = json.loads((HERE / "evidence_hashes_before.json").read_text(encoding="utf-8"))
    if snapshot(ROOT, EXCLUDED) != before or snapshot(ROOT / "Presentation_Evidence", set()) != evidence:
        raise RuntimeError("Original artifacts changed during the exploratory experiment.")
    write_json(HERE / "evaluation_provenance.json", {
        "status": "exploratory_saved_test_analysis_completed_once", "freeze_sha256": digest(HERE / "freeze.json"),
        "original_prediction_sha256": digest(ROOT / "artifacts" / "test_predictions.parquet"),
        "source_outcomes": "Saved frozen G2 predictions only", "raw_test_outcomes_read": False,
        "outputs_sha256": {name: digest(HERE / name) for name in outputs[1:-1]},
        "models_refit_on_original_fit_only": True, "models_tuned_on_validation_only": True,
        "test_responses": len(rows), "test_pairs": len(pairs), "bootstrap_replicates": 5000,
        "bootstrap_unit": "unordered photo pair", "bootstrap_seed": 20261101,
        "multiplicity": "95% pointwise exploratory; not adjusted", "original_files_unchanged": len(before),
        "previous_evidence_files_unchanged": len(evidence), "deployment_changed": False,
        "limits": json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))["limitations"]})
    print(json.dumps({"results": results, "contrasts": contrasts}, indent=2))


def plot(results: list[dict], contrasts: list[dict], calibration: list[dict]) -> None:
    with tempfile.TemporaryDirectory(prefix="mpl_", dir=HERE) as config:
        os.environ["MPLCONFIGDIR"] = config
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(HERE / "presentation_demographic_models.pdf") as pdf:
            fig, axes = plt.subplots(1, 2, figsize=(14, 8))
            wanted = ["common:M0", "D", "common:M1", "Cat+D", "M3-Y", "Y+D", "M4-C", "C+D"]
            subset = [next(r for r in results if r["model"] == name) for name in wanted]
            for i, r in enumerate(subset):
                axes[0].hlines(i, r["ci_low"], r["ci_high"], color="#28677c")
                axes[0].plot(r["log_loss"], i, "o", color="#28677c")
            axes[0].set_yticks(range(len(wanted)), wanted)
            axes[0].invert_yaxis()
            axes[0].set_xlabel("Response log loss (lower is better)")
            axes[0].set_title("Original G2 rows: frozen and exploratory models")
            selected = contrasts[:5]
            axes[1].axvline(0, color="gray")
            for i, r in enumerate(selected):
                axes[1].hlines(i, r["ci_low"], r["ci_high"], color="#28677c")
                axes[1].plot(r["difference"], i, "o", color="#28677c")
            axes[1].set_yticks(range(len(selected)), [f"{r['reference']} minus {r['candidate']}" for r in selected])
            axes[1].invert_yaxis()
            axes[1].set_xlabel("Positive favors candidate")
            axes[1].set_title("95% pointwise paired intervals")
            fig.suptitle("All-demographic models: retrospective predictive comparison", fontsize=15)
            fig.text(.05, .06, "Predictive Association, Not Causal Effect | 3,733 responses / 1,207 pairs | Post-test exploratory, not independent", fontsize=10)
            fig.text(.05, .03, "D predicts displayed-A tendencies only; A is not established to mean left. Intervals are not multiplicity adjusted.", fontsize=10)
            fig.subplots_adjust(left=.10, right=.97, bottom=.16, top=.9, wspace=.50)
            pdf.savefig(fig)
            plt.close(fig)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for ax, name in zip(axes.ravel(), REQUESTED):
                bins = [r for r in calibration if r["model"] == name and r["responses"]]
                supported = [r for r in bins if r["support_status"] == "supported"]
                ax.plot([0, 1], [0, 1], "--", color="gray")
                ax.scatter([r["mean_probability"] for r in bins], [r["observed_a_rate"] for r in bins],
                           facecolors="none", edgecolors="#28677c")
                ax.scatter([r["mean_probability"] for r in supported], [r["observed_a_rate"] for r in supported], color="#28677c")
                ax.set(title=name, xlabel="Mean predicted A probability", ylabel="Observed A frequency", xlim=(0, 1), ylim=(0, 1))
            fig.suptitle("Exploratory test calibration: filled bins meet reporting support", fontsize=14)
            fig.text(.05, .02, "Predictive Association, Not Causal Effect | Sparse bins hollow; calibration does not establish complete information.", fontsize=10)
            fig.tight_layout(rect=(0, .06, 1, .94))
            pdf.savefig(fig)
            plt.close(fig)


def publish() -> None:
    evidence = ROOT / "Presentation_Evidence"
    stage = Path("By_Stage") / "14_Retrospective_All_Demographics"
    flow = Path("By_Presentation_Flow") / "03_Diagnostics" / "08_Retrospective_All_Demographics"
    manifest = evidence / "demographic_selection_manifest.csv"
    verification = evidence / "demographic_verification.json"
    if any(p.exists() for p in (evidence/stage, evidence/flow, manifest, verification, HERE/"publication_verification.json")):
        raise FileExistsError("Publication already started.")
    verify_freeze()
    verify_original()
    before = json.loads((HERE / "original_hashes_before.json").read_text(encoding="utf-8"))
    old_evidence = json.loads((HERE / "evidence_hashes_before.json").read_text(encoding="utf-8"))
    if snapshot(ROOT, EXCLUDED) != before or snapshot(evidence, set()) != old_evidence:
        raise RuntimeError("Original artifacts changed before publication.")
    provenance = json.loads((HERE / "evaluation_provenance.json").read_text(encoding="utf-8"))
    for name, expected in provenance["outputs_sha256"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"Exploratory output hash mismatch: {name}")
    import shutil
    sources = ("presentation_demographic_models.pdf", "test_results.csv", "test_contrasts.csv", "validation_results.csv",
               "subgroups.csv", "calibration.csv", "development_support.csv", "protocol.json", "evaluation_provenance.json")
    hashes = {name: digest(HERE/name) for name in sources}
    for directory in (stage, flow):
        (evidence/directory).mkdir(parents=True, exist_ok=False)
    entries = []
    for name in sources:
        for directory in (stage, flow):
            shutil.copy2(HERE/name, evidence/directory/name)
            if digest(evidence/directory/name) != hashes[name]:
                raise RuntimeError("Copy hash mismatch.")
        entries.append({"source": str(Path("demographic_experiment")/name), "by_stage_copy": str(stage/name),
                        "by_flow_copy": str(flow/name), "bytes": (HERE/name).stat().st_size, "sha256": hashes[name],
                        "presentation_role": "Exploratory all-demographic predictive comparison; not causal or independently confirmed."})
    if snapshot(ROOT, EXCLUDED) != before:
        raise RuntimeError("Original files changed during copying.")
    for name, record in old_evidence.items():
        if digest(evidence/name) != record["sha256"]:
            raise RuntimeError("Previous presentation evidence changed during copying.")
    if any(digest(HERE/name) != hashes[name] for name in sources):
        raise RuntimeError("Source changed during copying.")
    write_csv(manifest, entries, list(entries[0]))
    report = {"status": "all_checks_passed", "original_files_unchanged": len(before),
              "previous_evidence_files_unchanged_before_tree_refresh": len(old_evidence),
              "new_sources": len(sources), "new_copies_verified": 2*len(sources),
              "manifest_sha256": digest(manifest), "experiment_freeze_sha256": digest(HERE/"freeze.json"),
              "no_original_claim_or_official_prediction_update": True,
              "note": "Historical manifests unchanged; only folder_tree.txt is refreshed after verification."}
    write_json(verification, report)
    write_json(HERE/"publication_verification.json", report)

    def tree(directory: Path, prefix: str = "") -> list[str]:
        children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        for i, p in enumerate(children):
            last = i == len(children)-1
            lines.append(prefix+("`-- " if last else "|-- ")+p.name+("\\" if p.is_dir() else ""))
            if p.is_dir():
                lines.extend(tree(p, prefix+("    " if last else "|   ")))
        return lines
    (evidence/"folder_tree.txt").write_text("Presentation_Evidence\\\n"+"\n".join(tree(evidence))+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1 to preserve the original tree.")
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("develop", "evaluate", "publish"))
    arguments = parser.parse_args()
    {"develop": develop, "evaluate": evaluate, "publish": publish}[arguments.stage]()
