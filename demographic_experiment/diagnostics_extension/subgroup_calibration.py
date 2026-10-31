"""Reproduce the original supported-subgroup reliability-bin evidence."""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from common import EXPERIMENT, HERE, digest, verify_inputs, write_csv, write_json
from experiment import DEMOGRAPHICS, REQUESTED, category, load_rows


def run() -> None:
    verify_inputs()
    table, metadata = HERE / "subgroup_calibration.csv", HERE / "subgroup_calibration_provenance.json"
    if table.exists() or metadata.exists():
        raise FileExistsError("Subgroup calibration output already exists.")
    rows = load_rows("test")
    saved = pq.read_table(EXPERIMENT / "test_predictions.parquet",
                          filters=[("model", "in", list(REQUESTED))]).to_pylist()
    by_model = {name: {r["train_row_id"]: r for r in saved if r["model"] == name} for name in REQUESTED}
    ids = [r["train_row_id"] for r in rows]
    if any(len(by_model[name]) != len(ids) or set(by_model[name]) != set(ids) for name in REQUESTED):
        raise ValueError("Saved model rows are incomplete or duplicated.")
    reference = [by_model["D"][key]["outcome"] for key in ids]
    if any([by_model[name][key]["outcome"] for key in ids] != reference for name in REQUESTED):
        raise ValueError("Saved model outcomes do not align.")
    output = []
    for model in REQUESTED:
        p = np.asarray([by_model[model][key]["probability"] for key in ids])
        y = np.asarray(reference)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid saved probabilities.")
        bins = np.minimum((p*10).astype(int), 9)
        for dimension in ("overall", "comparison_type", *DEMOGRAPHICS):
            values = ["all" if dimension == "overall" else r["comparison_type"]
                      if dimension in ("overall", "comparison_type") else category(r, dimension) for r in rows]
            for group in sorted(set(values)):
                group_mask = np.asarray([value == group for value in values])
                for b in range(10):
                    mask = group_mask & (bins == b)
                    n = int(mask.sum())
                    pairs = len({r["pair_id"] for r, keep in zip(rows, mask) if keep})
                    output.append({"model": model, "dimension": dimension, "group": group,
                                   "bin": b, "lower": b/10, "upper": (b+1)/10,
                                   "responses": n, "pairs": pairs,
                                   "support_status": "supported" if n >= 30 and pairs >= 10 else "low_support",
                                   "mean_probability": float(p[mask].mean()) if n else "",
                                   "observed_a_rate": float(y[mask].mean()) if n else ""})
    write_csv(table, output, list(output[0]))
    write_json(metadata, {"status": "completed", "rows": len(output), "test_responses": len(rows),
                          "models": list(REQUESTED), "bins": 10, "minimum_bin_responses": 30,
                          "minimum_bin_pairs": 10, "source_saved_predictions_sha256": digest(EXPERIMENT/"test_predictions.parquet"),
                          "output_sha256": digest(table), "raw_outcomes_reopened": False,
                          "interpretation": "Post-test descriptive reliability only; low-support bins retained and flagged."})
    print(json.dumps({"bins": len(output), "supported": sum(r["support_status"] == "supported" for r in output)}, indent=2))


if __name__ == "__main__":
    run()
