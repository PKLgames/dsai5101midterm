"""Saved-prediction G2 support, feature-quality, demographic and calibration slices."""

import json
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

from study import HERE, ROOT, digest, load_rows, load_frozen, write_csv, write_json
from src.evaluate import subgroup_labels


def run() -> None:
    if any((HERE/name).exists() for name in ("g2_slices.csv", "g2_slice_calibration.csv", "g2_slice_provenance.json")):
        raise FileExistsError("Slice diagnostics already exist.")
    load_frozen("g2")
    provenance = json.loads((HERE/"g2_evaluation_provenance.json").read_text(encoding="utf-8"))
    if digest(HERE/"g2_predictions.parquet") != provenance["outputs_sha256"]["g2_predictions.parquet"]:
        raise ValueError("Kernel saved predictions changed.")
    rows = load_rows("test")
    ids = [r["train_row_id"] for r in rows]
    kernel = pq.read_table(HERE/"g2_predictions.parquet", filters=[("model", "=", "kernel_M3-C")]).to_pylist()
    old = pq.read_table(ROOT/"artifacts"/"test_predictions.parquet",
                        filters=[("population", "=", "common"), ("model_id", "=", "common:M0")]).to_pylist()
    flags = {r["train_row_id"]: r for r in old}
    pred = {r["train_row_id"]: r for r in kernel}
    if len(flags) != len(old) or len(pred) != len(kernel) or set(flags) != set(ids) or set(pred) != set(ids):
        raise ValueError("Support flag/prediction populations do not align.")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        flag = flags[ids[i]]
        if pred[ids[i]]["outcome"] != flag["outcome"]:
            raise ValueError("Outcome mismatch.")
        for key in subgroup_labels(flag):
            groups[key].append(i)
    y = np.asarray([pred[key]["outcome"] for key in ids])
    p = np.asarray([pred[key]["probability"] for key in ids])
    loss = np.asarray([pred[key]["loss"] for key in ids])
    bins = np.minimum((p*10).astype(int), 9)
    results, calibration = [], []
    for (dimension, group), members in sorted(groups.items()):
        indices = np.asarray(members, dtype=int)
        n, pairs = len(indices), len({rows[i]["pair_id"] for i in indices})
        results.append({"dimension": dimension, "group": group, "responses": n, "pairs": pairs,
                        "support": "supported" if n >= 100 and pairs >= 30 else "low_support",
                        "log_loss": float(loss[indices].mean()), "brier": float(np.mean((p[indices]-y[indices])**2)),
                        "accuracy": float(np.mean((p[indices] >= .5) == y[indices]))})
        for b in range(10):
            subset = indices[bins[indices] == b]
            count, pair_count = len(subset), len({rows[i]["pair_id"] for i in subset})
            calibration.append({"dimension": dimension, "group": group, "bin": b, "responses": count,
                                "pairs": pair_count, "support": "supported" if count >= 30 and pair_count >= 10 else "low_support",
                                "mean_probability": float(p[subset].mean()) if count else "",
                                "observed_a_rate": float(y[subset].mean()) if count else ""})
    write_csv(HERE/"g2_slices.csv", results, list(results[0]))
    write_csv(HERE/"g2_slice_calibration.csv", calibration, list(calibration[0]))
    write_json(HERE/"g2_slice_provenance.json", {"status": "completed", "slice_rows": len(results),
                "calibration_rows": len(calibration), "source_kernel_predictions_sha256": digest(HERE/"g2_predictions.parquet"),
                "source_original_predictions_sha256": digest(ROOT/"artifacts"/"test_predictions.parquet"),
                "output_sha256": {name: digest(HERE/name) for name in ("g2_slices.csv", "g2_slice_calibration.csv")},
                "feature_quality_flags_overlap": True, "raw_test_outcomes_read": False,
                "limitation": "Retrospective descriptive test slices; flags and low support prevent independent subgroup claims."})
    print(f"Saved {len(results)} original-style test slices and {len(calibration)} reliability bins.")


if __name__ == "__main__":
    run()
