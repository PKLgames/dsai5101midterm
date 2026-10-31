"""Original-style development diagnostic slices without opening test outcomes."""

import json
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit

from study import HERE, ROOT, digest, load_frozen, load_rows, logits, write_csv, write_json


def run() -> None:
    if any((HERE/name).exists() for name in ("validation_slices.csv", "validation_calibration.csv", "validation_provenance.json")):
        raise FileExistsError("Validation diagnostics already exist.")
    model, rows = load_frozen("g2"), load_rows("validation")
    eta = logits(model, rows)
    y = np.asarray([r["y_displayed"] for r in rows])
    p = expit(eta)
    loss = y*np.logaddexp(0., -eta)+(1-y)*np.logaddexp(0., eta)
    graph = pq.read_table(ROOT/"artifacts"/"graph_components.parquet").to_pylist()
    opponents = {r["node_id"]: r["distinct_opponents"] for r in graph if r["population"] == "common" and r["entity"] == "photo"}
    features = pq.read_table(ROOT/"features"/"Y.parquet",
                             columns=["image_id", "detection_status", "candidate_cat_count"]).to_pylist()
    index = {r["image_id"]: r for r in features}
    if len(index) != len(features):
        raise ValueError("Duplicate Y photo keys.")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        a, b = row["a_image"], row["b_image"]
        if a not in index or b not in index or a not in opponents or b not in opponents:
            raise ValueError("Missing validation support or feature record.")
        support = min(opponents[a], opponents[b])
        band = "1" if support <= 1 else "2-4" if support <= 4 else "5-9" if support <= 9 else "10+"
        groups[("overall", "all")].append(i)
        groups[("comparison_type", row["comparison_type"])].append(i)
        groups[("pet_care", row["pet_care"] if row["pet_care"] else "missing")].append(i)
        groups[("minimum_endpoint_opponents", band)].append(i)
        endpoints = [index[a], index[b]]
        for label, yes in (
            ("no_detection", any(r["detection_status"] == "no_detection" for r in endpoints)),
            ("dog_fallback", any(r["detection_status"] == "dog_fallback" for r in endpoints)),
            ("multiple_cats", any(r["candidate_cat_count"] is not None and r["candidate_cat_count"] > 1 for r in endpoints)),
            ("both_cat_detected", all(r["detection_status"] == "cat_detected" for r in endpoints)),
        ):
            if yes:
                groups[("feature_quality_overlapping", label)].append(i)
    results, calibration = [], []
    for (dimension, group), members in sorted(groups.items()):
        indices = np.asarray(members, dtype=int)
        n, pairs = len(indices), len({rows[i]["pair_id"] for i in indices})
        supported = n >= 100 and pairs >= 30
        results.append({"dimension": dimension, "group": group, "responses": n, "pairs": pairs,
                        "support": "supported" if supported else "low_support",
                        "log_loss": float(loss[indices].mean()), "brier": float(np.mean((p[indices]-y[indices])**2)),
                        "accuracy": float(np.mean((p[indices] >= .5) == y[indices]))})
        if not supported:
            continue
        bins = np.minimum((p[indices]*10).astype(int), 9)
        for b in range(10):
            selected = indices[bins == b]
            count, pair_count = len(selected), len({rows[i]["pair_id"] for i in selected})
            calibration.append({"dimension": dimension, "group": group, "bin": b, "responses": count,
                                "pairs": pair_count, "support": "supported" if count >= 30 and pair_count >= 10 else "low_support",
                                "mean_probability": float(p[selected].mean()) if count else "",
                                "observed_a_rate": float(y[selected].mean()) if count else ""})
    write_csv(HERE/"validation_slices.csv", results, list(results[0]))
    write_csv(HERE/"validation_calibration.csv", calibration, list(calibration[0]))
    write_json(HERE/"validation_provenance.json", {"status": "completed", "responses": len(rows),
               "slices": len(results), "bins": len(calibration), "model_freeze_sha256": digest(HERE/"g2_freeze.json"),
               "outputs_sha256": {name: digest(HERE/name) for name in ("validation_slices.csv", "validation_calibration.csv")},
               "test_outcomes_read": False, "limitations": "Post-test motivated development diagnostics; not independent."})
    print(f"Saved {len(results)} validation slices and {len(calibration)} reliability bins.")


if __name__ == "__main__":
    run()
