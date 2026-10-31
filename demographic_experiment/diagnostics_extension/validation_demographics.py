"""Development-side demographic subgroup metrics and calibration."""

from __future__ import annotations

import json

import numpy as np
from scipy.special import expit

from common import EXPERIMENT, HERE, digest, verify_inputs, write_csv, write_json
from experiment import DEMOGRAPHICS, REQUESTED, category, load_rows, predict


def run() -> None:
    verify_inputs()
    outputs = [HERE / "validation_demographic_slices.csv", HERE / "validation_demographic_calibration.csv",
               HERE / "validation_demographic_provenance.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Validation demographic diagnostics already exist.")
    rows = load_rows("validation")
    models = json.loads((EXPERIMENT / "selected_models.json").read_text(encoding="utf-8"))
    logits = predict(rows, models)
    y = np.asarray([r["y_displayed"] for r in rows])
    metrics, calibration = [], []
    for model in REQUESTED:
        z, p = logits[model], expit(logits[model])
        losses = y*np.logaddexp(0., -z)+(1-y)*np.logaddexp(0., z)
        for dimension in DEMOGRAPHICS:
            values = [category(row, dimension) for row in rows]
            for group in sorted(set(values)):
                index = np.flatnonzero(np.asarray(values) == group)
                n, pairs = len(index), len({rows[i]["pair_id"] for i in index})
                supported = n >= 100 and pairs >= 30
                metrics.append({"model": model, "dimension": dimension, "group": group,
                                "responses": n, "pairs": pairs, "support_status": "supported" if supported else "low_support",
                                "log_loss": float(losses[index].mean()), "brier": float(np.mean((p[index]-y[index])**2)),
                                "accuracy": float(np.mean((p[index] >= .5) == y[index]))})
                if not supported:
                    continue
                bins = np.minimum((p[index]*10).astype(int), 9)
                for b in range(10):
                    selected = index[bins == b]
                    count, distinct_pairs = len(selected), len({rows[i]["pair_id"] for i in selected})
                    calibration.append({"model": model, "dimension": dimension, "group": group,
                                        "bin": b, "lower": b/10, "upper": (b+1)/10,
                                        "responses": count, "pairs": distinct_pairs,
                                        "support_status": "supported" if count >= 30 and distinct_pairs >= 10 else "low_support",
                                        "mean_probability": float(p[selected].mean()) if count else "",
                                        "observed_a_rate": float(y[selected].mean()) if count else ""})
    write_csv(outputs[0], metrics, list(metrics[0]))
    write_csv(outputs[1], calibration, list(calibration[0]))
    write_json(outputs[2], {"status": "completed", "partition": "original_common_strict_validation",
                           "responses": len(rows), "models": list(REQUESTED), "subgroup_rows": len(metrics),
                           "calibration_rows": len(calibration), "model_sha256": digest(EXPERIMENT/"selected_models.json"),
                           "outputs_sha256": {p.name: digest(p) for p in outputs[:2]},
                           "test_outcomes_read": False, "limitations": "Post-test motivated development diagnostics; not independent."})
    print(json.dumps({"subgroup_rows": len(metrics), "calibration_rows": len(calibration)}, indent=2))


if __name__ == "__main__":
    run()
