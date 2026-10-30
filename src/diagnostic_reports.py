"""Validation confidence, optional temperature, and score-recovery probes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .development import score
from .diagnostics import DevelopmentData, load_data, load_predictions, write_csv
from .preprocessing import transform_photos


def confidence_and_temperature(
    data: DevelopmentData, predictions: dict[str, np.ndarray], protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    block = data.selection["selected_representation"]
    aliases = {"M2": "common:M2", "M3*": f"M3-{block}", "M4*": f"M4-{block}"}
    y = np.asarray([row["y_displayed"] for row in data.val], dtype=np.float64)
    temperatures = data.protocol["optional_diagnostics"]["temperature_grid"]
    audit, summary, temps = [], [], []
    for label, model in aliases.items():
        eta = predictions[model]
        p = expit(eta)
        losses = y * np.logaddexp(0., -eta) + (1.-y)*np.logaddexp(0., eta)
        count = protocol["worst_loss_audit"]["count_per_model"]
        order = sorted(range(len(y)), key=lambda i: (-losses[i], data.val[i]["train_row_id"]))[:count]
        for rank, i in enumerate(order, 1):
            audit.append({
                "partition": "common_strict_validation", "model": label, "rank": rank,
                "train_row_id": data.val[i]["train_row_id"], "pair_id": data.val[i]["pair_id"],
                "comparison_type": data.val[i]["comparison_type"], "pet_care": data.val[i]["pet_care"] or "missing",
                "outcome": int(y[i]), "logit": float(eta[i]), "probability": float(p[i]), "loss": float(losses[i]),
            })
        summary.append({
            "partition": "common_strict_validation", "model": label,
            "probability_p05": float(np.quantile(p, .05)), "probability_median": float(np.median(p)),
            "probability_p95": float(np.quantile(p, .95)),
            "absolute_logit_p95": float(np.quantile(np.abs(eta), .95)),
            "maximum_loss": float(losses.max()),
        })
        candidate = []
        for temperature in temperatures:
            loss, brier, accuracy = score(y, eta/temperature)
            candidate.append({"model": label, "temperature": temperature, "validation_loss": loss,
                              "brier": brier, "accuracy": accuracy})
        best = min(candidate, key=lambda x: (x["validation_loss"], abs(x["temperature"]-1), x["temperature"]))
        for item in candidate:
            temps.append({**item, "selected": item is best, "partition": "common_strict_validation",
                          "interpretation": "secondary validation-only confidence diagnostic"})
    return audit, summary, temps


def reconstruction(data: DevelopmentData, root: Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    photo_model = data.selection["selected"]["common:M2"]
    target = dict(zip(photo_model["identity_keys"], photo_model["coefficients"][1:]))
    ids = sorted({row[f"{side}_image"] for row in data.fit for side in ("a", "b")})
    if set(ids) != set(target):
        raise ValueError("M2 photo scores do not match common fitting-photo identities.")
    components = pq.read_table(root / "artifacts/graph_components.parquet").to_pylist()
    component = {row["node_id"]: row["component_id"] for row in components
                 if row["population"] == "common" and row["entity"] == "photo"}
    if not set(ids) <= set(component):
        raise ValueError("Missing fitting-photo graph component.")
    groups: dict[int, list[int]] = {}
    for i, image in enumerate(ids):
        groups.setdefault(component[image], []).append(i)
    target_array = np.array([target[image] for image in ids], dtype=np.float64)
    centered_y = target_array.copy()
    for indices in groups.values():
        centered_y[indices] -= centered_y[indices].mean()
    ridge = protocol["representation_recovery"]["ridge"]
    if ridge <= 0:
        raise ValueError("Representation-recovery ridge must be positive.")
    results = []
    with threadpool_limits(limits=1):
        for block in ("H", "C", "Y"):
            records = {row["image_id"]: row for row in data.features[block]}
            spec = json.loads((root / "artifacts/preprocessing" / f"{block}.json").read_text(encoding="utf-8"))
            raw = transform_photos(spec, [records[image] for image in ids])
            centered = raw.copy()
            for indices in groups.values():
                centered[indices] -= centered[indices].mean(axis=0)
            lhs = centered.T @ centered / len(ids) + ridge*np.eye(centered.shape[1])
            coefficients = np.linalg.solve(lhs, centered.T @ centered_y / len(ids))
            prediction = centered @ coefficients
            rmse = float(np.sqrt(np.mean((centered_y-prediction)**2)))
            correlation = float(np.corrcoef(centered_y, prediction)[0, 1]) if np.std(prediction) > 0 else None
            results.append({
                "partition": "common_fit_photos", "representation": block, "photos": len(ids),
                "components": len(groups), "dimensions": centered.shape[1], "ridge": ridge,
                "rmse": rmse, "correlation": correlation,
                "interpretation": "in-sample reconstruction of estimated M2 scores, not an independent preference test",
            })
    return results


def reliability_figure(root: Path) -> None:
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (root / "tables/diagnostic_calibration.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(6, 5))
    selection = json.loads((root / "artifacts/model_selection.json").read_text(encoding="utf-8"))
    block = selection["selected_representation"]
    for model in ("common:M2", f"M3-{block}", f"M4-{block}"):
        subset = [r for r in rows if r["dimension"] == "overall" and r["model"] == model
                  and r["support_status"] == "supported"]
        if subset:
            ax.plot([float(r["mean_probability"]) for r in subset],
                    [float(r["observed_a_rate"]) for r in subset], marker="o", label=model)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="Mean predicted P(A)", ylabel="Observed A frequency",
           title="Common strict validation; supported bins only", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "figures/diagnostic_calibration.pdf")
    plt.close(fig)


def run_reports(root: Path) -> dict[str, Any]:
    outputs = [root / "tables/worst_validation_losses.csv", root / "tables/diagnostic_confidence.csv",
               root / "tables/temperature_diagnostic.csv", root / "tables/representation_recovery.csv",
               root / "figures/diagnostic_calibration.pdf", root / "artifacts/diagnostic_report_status.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Diagnostic report outputs already exist.")
    core = json.loads((root / "artifacts/stage8_core.json").read_text(encoding="utf-8"))
    if core["status"] != "passed_operational_control":
        raise ValueError("Fair-coin control has not passed its operational gate.")
    data = load_data(root)
    protocol = json.loads((root / "artifacts/stage8_probe_protocol.json").read_text(encoding="utf-8"))
    audit, summary, temperatures = confidence_and_temperature(data, load_predictions(root, data), protocol)
    recovery = reconstruction(data, root, protocol)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in zip(outputs[:4], (audit, summary, temperatures, recovery)):
        write_csv(path, rows)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    reliability_figure(root)
    result = {"status": "completed", "worst_losses": len(audit), "temperature_candidates": len(temperatures),
              "representations": len(recovery), "test_outcomes_read": False}
    outputs[-1].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
