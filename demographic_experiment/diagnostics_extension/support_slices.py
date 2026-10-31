"""Reproduce the original fitting-support and feature-quality test slice definitions."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

from common import EXPERIMENT, HERE, ROOT, digest, verify_inputs, write_csv, write_json
from experiment import REQUESTED, load_rows
from src.evaluate import subgroup_labels


def run() -> None:
    verify_inputs()
    outputs = [HERE / "test_support_slices.csv", HERE / "test_support_calibration.csv",
               HERE / "test_support_slices_provenance.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Original-style test support slices already exist.")
    rows = load_rows("test")
    ids = [r["train_row_id"] for r in rows]
    original = pq.read_table(ROOT / "artifacts" / "test_predictions.parquet",
                             filters=[("population", "=", "common"), ("model_id", "=", "common:M0")]).to_pylist()
    flags = {r["train_row_id"]: r for r in original}
    if len(flags) != len(original) or set(flags) != set(ids):
        raise ValueError("Original frozen quality/support flags have missing or duplicate IDs.")
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        flag = flags[row["train_row_id"]]
        if flag["pair_id"] != row["pair_id"]:
            raise ValueError("Frozen support and structural pair mismatch.")
        for key in subgroup_labels(flag):
            grouped[key].append(i)
    predictions = pq.read_table(EXPERIMENT / "test_predictions.parquet",
                                filters=[("model", "in", list(REQUESTED))]).to_pylist()
    by_model = {name: {r["train_row_id"]: r for r in predictions if r["model"] == name} for name in REQUESTED}
    results, calibration = [], []
    for name in REQUESTED:
        if set(by_model[name]) != set(ids) or len(by_model[name]) != len(ids):
            raise ValueError(f"Saved demographic predictions incomplete: {name}")
        ordered = [by_model[name][key] for key in ids]
        y = np.asarray([r["outcome"] for r in ordered])
        p = np.asarray([r["probability"] for r in ordered])
        loss = np.asarray([r["loss"] for r in ordered])
        if any(ordered[i]["outcome"] != flags[ids[i]]["outcome"] for i in range(len(ids))):
            raise ValueError("Frozen demographic outcome alignment mismatch.")
        bins = np.minimum((p*10).astype(int), 9)
        for (dimension, group), members in sorted(grouped.items()):
            index = np.asarray(members, dtype=int)
            n, pair_count = len(index), len({rows[i]["pair_id"] for i in index})
            results.append({"model": name, "dimension": dimension, "group": group,
                            "responses": n, "pairs": pair_count,
                            "support_status": "supported" if n >= 100 and pair_count >= 30 else "low_support",
                            "log_loss": float(loss[index].mean()), "brier": float(np.mean((p[index]-y[index])**2)),
                            "accuracy": float(np.mean((p[index] >= .5) == y[index]))})
            for b in range(10):
                selected = index[bins[index] == b]
                count, pairs = len(selected), len({rows[i]["pair_id"] for i in selected})
                calibration.append({"model": name, "dimension": dimension, "group": group, "bin": b,
                                    "lower": b/10, "upper": (b+1)/10, "responses": count, "pairs": pairs,
                                    "support_status": "supported" if count >= 30 and pairs >= 10 else "low_support",
                                    "mean_probability": float(p[selected].mean()) if count else "",
                                    "observed_a_rate": float(y[selected].mean()) if count else ""})
    write_csv(outputs[0], results, list(results[0]))
    write_csv(outputs[1], calibration, list(calibration[0]))
    write_json(outputs[2], {"status": "completed", "subgroup_rows": len(results), "calibration_rows": len(calibration),
                           "original_support_flags_sha256": digest(ROOT/"artifacts"/"test_predictions.parquet"),
                           "demographic_predictions_sha256": digest(EXPERIMENT/"test_predictions.parquet"),
                           "output_sha256": {p.name: digest(p) for p in outputs[:2]}, "raw_outcomes_reopened": False,
                           "interpretation": "Post-test descriptive slices; quality flags overlap; low-support values flagged."})
    print(json.dumps({"subgroup_rows": len(results), "calibration_rows": len(calibration)}, indent=2))


if __name__ == "__main__":
    run()
