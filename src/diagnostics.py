"""Development-only diagnostic data and fair-coin controls."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .development import _hash, feature_differences, group_name, identity_matrix, load_development, score, visual_design
from .estimator import contextual_penalties, fit_logistic
from .preprocessing import transform_photos


@dataclass
class DevelopmentData:
    fit: list[dict[str, Any]]
    val: list[dict[str, Any]]
    features: dict[str, list[dict[str, Any]]]
    selection: dict[str, Any]
    protocol: dict[str, Any]


def load_data(root: Path) -> DevelopmentData:
    verification = json.loads((root / "artifacts/stage7_verification.json").read_text(encoding="utf-8"))
    for name, digest in verification["output_sha256"].items():
        path = root.joinpath(*name.split("/"))
        if not path.is_file() or _hash(path) != digest:
            raise ValueError(f"Stage 7 output changed: {name}")
    if _hash(root / "src/verify_development.py") != verification["verification_source_sha256"]:
        raise ValueError("Stage 7 verification source changed.")
    from .development import verify_inputs
    verify_inputs(root)
    selection = json.loads((root / "artifacts/model_selection.json").read_text(encoding="utf-8"))
    for name, digest in selection["input_sha256"].items():
        if _hash(root.joinpath(*name.split("/"))) != digest:
            raise ValueError(f"Stage 7 input changed: {name}")
    protocol = json.loads((root / "artifacts/stage8_protocol.json").read_text(encoding="utf-8"))
    fit, val = load_development(root)
    fit = [row for row in fit if row["common_visual_fit"]]
    val = [row for row in val if row["common_strict_g2"]]
    if len(fit) != selection["fitting_responses"] or len(val) != selection["validation_responses"]:
        raise ValueError("Matched development population differs from Stage 7.")
    features = {block: pq.read_table(root / "features" / f"{block}.parquet").to_pylist()
                for block in ("H", "C", "Y")}
    return DevelopmentData(fit, val, features, selection, protocol)


def differences(
    rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]], spec: dict[str, Any]
) -> sparse.csr_matrix:
    records = {row["image_id"]: row for row in feature_rows}
    if len(records) != len(feature_rows):
        raise ValueError("Duplicate image features.")
    ids = sorted({row[f"{side}_image"] for row in rows for side in ("a", "b")})
    if not set(ids) <= set(records) or any(records[image]["extraction_status"] != "ok" for image in ids):
        raise ValueError("Features unavailable for development rows.")
    matrix = transform_photos(spec, [records[image] for image in ids])
    return feature_differences(rows, ids, matrix)


def load_predictions(root: Path, data: DevelopmentData) -> dict[str, np.ndarray]:
    with (root / "tables/validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    expected = [row["train_row_id"] for row in data.val]
    predictions: dict[str, np.ndarray] = {}
    for name in ("common:M0", "common:M1", "common:M2", "M3-H", "M3-C", "M3-Y",
                 "M4-H", "M4-C", "M4-Y", "M3-side", "M4-side"):
        entries = grouped[name]
        if [row["train_row_id"] for row in entries] != expected:
            raise ValueError(f"Validation ID mismatch for {name}.")
        logits = np.asarray([float(row["logit"]) for row in entries])
        labels = np.asarray([row["y_displayed"] for row in data.val])
        model = data.selection["selected"][name]
        if abs(score(labels, logits)[0] - model["validation_loss"]) > 1e-12:
            raise ValueError(f"Validation loss mismatch for {name}.")
        predictions[name] = logits
    return predictions


def baseline_diagnostics(root: Path, data: DevelopmentData) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    logits = load_predictions(root, data)
    y = np.asarray([row["y_displayed"] for row in data.val], dtype=np.float64)
    nodes = pq.read_table(root / "artifacts/graph_components.parquet").to_pylist()
    opponents = {row["node_id"]: row["distinct_opponents"] for row in nodes
                 if row["population"] == "common" and row["entity"] == "photo"}
    y_features = {row["image_id"]: row for row in data.features["Y"]}
    slices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(data.val):
        slices[("overall", "all")].append(i)
        slices[("comparison_type", row["comparison_type"])].append(i)
        slices[("pet_care", group_name(row))].append(i)
        support = min(opponents[row["a_image"]], opponents[row["b_image"]])
        band = "1" if support <= 1 else "2-4" if support <= 4 else "5-9" if support <= 9 else "10+"
        slices[("minimum_endpoint_opponents", band)].append(i)
        endpoints = [y_features[row["a_image"]], y_features[row["b_image"]]]
        for label, condition in (
            ("no_detection", any(x["detection_status"] == "no_detection" for x in endpoints)),
            ("dog_fallback", any(x["detection_status"] == "dog_fallback" for x in endpoints)),
            ("multiple_cats", any(x["candidate_cat_count"] > 1 for x in endpoints)),
            ("both_cat_detected", all(x["detection_status"] == "cat_detected" for x in endpoints)),
        ):
            if condition:
                slices[("feature_quality_overlapping", label)].append(i)
    metrics, calibration = [], []
    for name, values in logits.items():
        probs = expit(values)
        for (dimension, label), indices in sorted(slices.items()):
            ids = np.asarray(indices)
            pair_count = len({data.val[i]["pair_id"] for i in indices})
            sufficient = len(indices) >= 100 and pair_count >= 30
            loss, brier, accuracy = score(y[ids], values[ids])
            metrics.append({
                "partition": "common_strict_validation", "model": name, "dimension": dimension,
                "group": label, "responses": len(ids), "pairs": pair_count,
                "support_status": "supported" if sufficient else "low_support",
                "log_loss": loss, "brier": brier, "accuracy": accuracy,
                "mean_probability": float(probs[ids].mean()),
                "mean_abs_logit": float(np.mean(np.abs(values[ids]))),
            })
        for (dimension, label), indices in sorted(slices.items()):
            if len(indices) < 100 or len({data.val[i]["pair_id"] for i in indices}) < 30:
                continue
            subset = np.asarray(indices)
            for bin_id in range(10):
                local = probs[subset]
                ids = subset[(local >= bin_id / 10) &
                             ((local < (bin_id + 1) / 10) if bin_id < 9 else (local <= 1))]
                pairs = len({data.val[i]["pair_id"] for i in ids})
                calibration.append({
                    "partition": "common_strict_validation", "model": name,
                    "dimension": dimension, "group": label, "bin": bin_id,
                    "lower": bin_id / 10, "upper": (bin_id + 1) / 10,
                    "responses": len(ids), "pairs": pairs,
                    "support_status": "supported" if len(ids) >= 30 and pairs >= 10 else "low_support",
                    "mean_probability": float(probs[ids].mean()) if len(ids) else "",
                    "observed_a_rate": float(y[ids].mean()) if len(ids) else "",
                })
    return metrics, calibration


def control_designs(root: Path, data: DevelopmentData) -> dict[str, tuple[Any, Any, np.ndarray]]:
    selected = data.selection["selected"]
    block = data.selection["selected_representation"]
    spec = json.loads((root / "artifacts/preprocessing" / f"{block}.json").read_text(encoding="utf-8"))
    dx, dv = differences(data.fit, data.features[block], spec), differences(data.val, data.features[block], spec)
    groups_fit = [group_name(row) for row in data.fit]
    groups_val = [group_name(row) for row in data.val]
    interactions = tuple(data.selection["interaction_contexts"])
    m2 = selected["common:M2"]
    result: dict[str, tuple[Any, Any, np.ndarray]] = {
        "M2": (identity_matrix(data.fit, m2["identity_keys"], "image"),
               identity_matrix(data.val, m2["identity_keys"], "image"),
               np.r_[[1e-6], np.full(len(m2["identity_keys"]), m2["ridge"])]),
    }
    m3 = selected["M3*"]
    dim = dx.shape[1]
    result["M3*"] = (
        visual_design(dx, groups_fit, ()), visual_design(dv, groups_val, ()),
        np.r_[[1e-6], np.full(dim, m3["ridge"])],
    )
    m4 = selected["M4*"]
    if m4.get("unsupported"):
        return result
    if m4["pooled"]:
        result["M4*"] = result["M3*"]
    else:
        result["M4*"] = (
            visual_design(dx, groups_fit, interactions), visual_design(dv, groups_val, interactions),
            contextual_penalties(dim, len(interactions), m4["ridge"], m4["rho"]),
        )
    return result


def fair_coin_control(data: DevelopmentData, designs: dict[str, tuple[Any, Any, np.ndarray]]) -> list[dict[str, Any]]:
    cfg = data.protocol["synthetic_label_control"]
    rng = np.random.default_rng(cfg["seed"])
    results = []
    with threadpool_limits(limits=1):
        for rep in range(cfg["repetitions"]):
            yfit = rng.integers(0, 2, size=len(data.fit)).astype(float)
            yval = rng.integers(0, 2, size=len(data.val)).astype(float)
            prior: dict[str, tuple[Any, float | None]] = {}
            for name in ("M2", "M3*", "M4*"):
                if name not in designs:
                    continue
                if name == "M4*" and data.selection["selected"]["M4*"]["pooled"]:
                    fitted, loss = prior["M3*"]
                else:
                    x, vx, penalty = designs[name]
                    fitted = fit_logistic(x, yfit, penalty)
                    loss = score(yval, np.asarray(vx @ fitted.coefficients).reshape(-1))[0] if fitted.accepted else None
                prior[name] = fitted, loss
                results.append({
                    "repetition": rep, "seed": cfg["seed"], "model": name,
                    "accepted": fitted.accepted, "log_loss": loss,
                    "baseline_loss": math.log(2),
                    "improvement": math.log(2) - loss if loss is not None else None,
                    "gradient_max": fitted.gradient_max if np.isfinite(fitted.gradient_max) else None,
                    "iterations": fitted.iterations, "solver_status": fitted.status,
                    "solver_message": fitted.message, "fit_attempts": json.dumps(fitted.attempts),
                })
            print(f"Fair-coin control {rep+1}/{cfg['repetitions']}", flush=True)
    return results


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {path}")
    if not rows and not fields:
        raise ValueError(f"Cannot infer empty diagnostic table columns: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_core(root: Path) -> dict[str, Any]:
    data = load_data(root)
    outputs = [root / "tables/diagnostic_results.csv", root / "tables/diagnostic_calibration.csv",
               root / "tables/synthetic_label_control.csv", root / "artifacts/stage8_core.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Stage 8 core outputs already exist.")
    metrics, calibration = baseline_diagnostics(root, data)
    designs = control_designs(root, data)
    control = fair_coin_control(data, designs)
    failures = sum(not row["accepted"] for row in control)
    improved = Counter(row["model"] for row in control if row["improvement"] is not None and row["improvement"] > 0)
    flags = {name: improved[name] >= 16 for name in designs}
    write_csv(outputs[0], metrics)
    write_csv(outputs[1], calibration)
    write_csv(outputs[2], control)
    report = {
        "validation_responses": len(data.val), "fitting_responses": len(data.fit),
        "control_repetitions": data.protocol["synthetic_label_control"]["repetitions"],
        "failed_control_fits": failures, "improved_repetitions": dict(improved),
        "investigation_triggered": flags,
        "status": "blocked_investigation_required" if failures or any(flags.values()) else "passed_operational_control",
        "test_outcomes_read": False,
        "note": "The control cannot certify leakage freedom; subgroup performance may be low support.",
        "input_sha256": {name: _hash(root.joinpath(*name.split("/"))) for name in (
            "artifacts/stage7_verification.json", "artifacts/stage8_protocol.json", "src/diagnostics.py",
        )},
        "output_sha256": {path.relative_to(root).as_posix(): _hash(path) for path in outputs[:-1]},
    }
    outputs[-1].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
