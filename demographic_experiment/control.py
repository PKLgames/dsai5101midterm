"""Fixed-setting development-only fair-coin controls for the four procedures."""

from __future__ import annotations

import csv
import json
import math
import os
import time

import numpy as np
from threadpoolctl import threadpool_limits

from experiment import (HERE, REQUESTED, design, digest, get_differences, load_rows, penalties,
                        safe, score, side_matrix, verify_freeze, write_json)
from src.estimator import fit_logistic


def run() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    if any((HERE / name).exists() for name in ("control_results.csv", "control_provenance.json", "evaluation_started.json")):
        raise FileExistsError("Control already started or saved test outcome analysis already occurred.")
    verify_freeze()
    models = json.loads((HERE / "selected_models.json").read_text(encoding="utf-8"))
    fit, val = load_rows("fit"), load_rows("validation")
    side_fit, side_val = side_matrix(fit, models["encoding"]), side_matrix(val, models["encoding"])
    diffs_fit, diffs_val = get_differences(fit, models), get_differences(val, models)
    designs = {}
    for name in REQUESTED:
        selected = models["selected"][name]
        branch, interacting = selected["branch"], selected["interacting"]
        dimension = 0 if branch == "D" else diffs_fit[branch].shape[1]
        x = design(side_fit, None if branch == "D" else diffs_fit[branch], interacting)
        vx = design(side_val, None if branch == "D" else diffs_val[branch], interacting)
        designs[name] = (x, vx, penalties(side_fit.shape[1], dimension, interacting, selected["ridge"],
                                         selected["rho"] if interacting else 1))
    generator = np.random.default_rng(20261102)
    results = []
    fields = ["repetition", "model", "accepted", "validation_loss", "improvement_over_log2",
              "gradient_max", "seconds", "attempts"]
    with (HERE / "control_results.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        with threadpool_limits(limits=1):
            for repetition in range(20):
                yy = generator.integers(0, 2, len(fit)).astype(float)
                vy = generator.integers(0, 2, len(val)).astype(float)
                for name, (x, vx, penalty) in designs.items():
                    started = time.monotonic()
                    fitted = fit_logistic(x, yy, penalty)
                    loss = score(vy, np.asarray(vx @ fitted.coefficients).ravel())[0] if fitted.accepted else None
                    row = {"repetition": repetition+1, "model": name, "accepted": fitted.accepted,
                           "validation_loss": loss, "improvement_over_log2": math.log(2)-loss if loss is not None else None,
                           "gradient_max": safe(fitted.gradient_max), "seconds": time.monotonic()-started,
                           "attempts": json.dumps(safe(fitted.attempts), allow_nan=False)}
                    writer.writerow(row)
                    stream.flush()
                    results.append(row)
                print(f"Fair-coin controls {repetition+1}/20", flush=True)
    failures = sum(not r["accepted"] for r in results)
    counts = {name: sum(r["model"] == name and r["improvement_over_log2"] is not None and
                        r["improvement_over_log2"] > 0 for r in results) for name in REQUESTED}
    triggered = {name: value >= 16 for name, value in counts.items()}
    report = {"status": "passed_operational_control" if not failures and not any(triggered.values()) else
              "blocked_investigation_required", "repetitions": 20, "fits": len(results), "failed_fits": failures,
              "seed": 20261102, "baseline": math.log(2), "improved_repetitions": counts,
              "mean_validation_loss": {name: float(np.mean([r["validation_loss"] for r in results
                                          if r["model"] == name and r["accepted"]])) for name in REQUESTED
                                       if any(r["model"] == name and r["accepted"] for r in results)},
              "investigation_trigger": "Existing repository rule: >=16 of 20 fair-coin validation improvements or any failed fits.",
              "investigation_triggered": triggered, "test_outcomes_read": False,
              "fixed_settings_not_retuned": True, "source_sha256": digest(HERE/"control.py"),
              "freeze_sha256": digest(HERE/"freeze.json"), "results_sha256": digest(HERE/"control_results.csv"),
              "note": "Operational leakage/recovery probe only, not proof of leakage freedom or model generalization."}
    write_json(HERE/"control_provenance.json", report)
    if report["status"] != "passed_operational_control":
        raise RuntimeError("Fair-coin controls require investigation; do not proceed to saved test outcomes.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
