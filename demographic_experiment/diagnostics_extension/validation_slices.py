"""Reproduce the original supported-slice development diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit

from common import EXPERIMENT, HERE, ROOT, digest, verify_inputs, write_csv, write_json
from experiment import REQUESTED, load_rows, predict


def run() -> None:
    verify_inputs()
    outputs = [HERE / "validation_support_slices.csv", HERE / "validation_support_calibration.csv",
               HERE / "validation_support_provenance.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Validation support diagnostics already exist.")
    rows = load_rows("validation")
    models = json.loads((EXPERIMENT / "selected_models.json").read_text(encoding="utf-8"))
    logits = predict(rows, models)
    graph = pq.read_table(ROOT / "artifacts" / "graph_components.parquet").to_pylist()
    opponents = {r["node_id"]: r["distinct_opponents"] for r in graph
                 if r["population"] == "common" and r["entity"] == "photo"}
    features = pq.read_table(ROOT / "features" / "Y.parquet",
                             columns=["image_id", "detection_status", "candidate_cat_count"]).to_pylist()
    index = {r["image_id"]: r for r in features}
    if len(index) != len(features):
        raise ValueError("Duplicate Y feature IDs.")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        try:
            support = min(opponents[row["a_image"]], opponents[row["b_image"]])
            endpoints = [index[row["a_image"]], index[row["b_image"]]]
        except KeyError as exc:
            raise ValueError(f"Original validation support unavailable: {exc}") from exc
        band = "1" if support <= 1 else "2-4" if support <= 4 else "5-9" if support <= 9 else "10+"
        groups[("overall", "all")].append(i)
        groups[("comparison_type", row["comparison_type"])].append(i)
        groups[("pet_care", row["pet_care"] if row["pet_care"] else "missing")].append(i)
        groups[("minimum_endpoint_opponents", band)].append(i)
        for label, yes in (
            ("no_detection", any(r["detection_status"] == "no_detection" for r in endpoints)),
            ("dog_fallback", any(r["detection_status"] == "dog_fallback" for r in endpoints)),
            ("multiple_cats", any(r["candidate_cat_count"] is not None and r["candidate_cat_count"] > 1 for r in endpoints)),
            ("both_cat_detected", all(r["detection_status"] == "cat_detected" for r in endpoints)),
        ):
            if yes:
                groups[("feature_quality_overlapping", label)].append(i)
    y = np.asarray([r["y_displayed"] for r in rows])
    metrics, calibration = [], []
    for name in REQUESTED:
        z, p = logits[name], expit(logits[name])
        losses = y*np.logaddexp(0., -z)+(1-y)*np.logaddexp(0., z)
        for (dimension, group), members in sorted(groups.items()):
            ids = np.asarray(members, dtype=int)
            n, pairs = len(ids), len({rows[i]["pair_id"] for i in ids})
            supported = n >= 100 and pairs >= 30
            metrics.append({"model": name, "dimension": dimension, "group": group,
                            "responses": n, "pairs": pairs, "support_status": "supported" if supported else "low_support",
                            "log_loss": float(losses[ids].mean()), "brier": float(np.mean((p[ids]-y[ids])**2)),
                            "accuracy": float(np.mean((p[ids] >= .5) == y[ids])),
                            "mean_probability": float(p[ids].mean()), "mean_abs_logit": float(np.mean(np.abs(z[ids])))})
            if not supported:
                continue
            bins = np.minimum((p[ids]*10).astype(int), 9)
            for b in range(10):
                selected = ids[bins == b]
                count, pair_count = len(selected), len({rows[i]["pair_id"] for i in selected})
                calibration.append({"model": name, "dimension": dimension, "group": group, "bin": b,
                                    "lower": b/10, "upper": (b+1)/10, "responses": count, "pairs": pair_count,
                                    "support_status": "supported" if count >= 30 and pair_count >= 10 else "low_support",
                                    "mean_probability": float(p[selected].mean()) if count else "",
                                    "observed_a_rate": float(y[selected].mean()) if count else ""})
    write_csv(outputs[0], metrics, list(metrics[0]))
    write_csv(outputs[1], calibration, list(calibration[0]))
    write_json(outputs[2], {"status": "completed", "partition": "original_common_strict_validation",
                           "responses": len(rows), "models": list(REQUESTED), "subgroup_rows": len(metrics),
                           "calibration_rows": len(calibration),
                           "model_sha256": digest(EXPERIMENT/"selected_models.json"),
                           "outputs_sha256": {p.name: digest(p) for p in outputs[:2]},
                           "test_outcomes_read": False, "limitations": "Post-test motivated development diagnostics; not independent."})
    print(json.dumps({"subgroup_rows": len(metrics), "calibration_rows": len(calibration)}, indent=2))


if __name__ == "__main__":
    run()
