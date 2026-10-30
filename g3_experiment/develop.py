"""Fit and select retrospective G3 models on separate fitting/validation data."""

from __future__ import annotations

import json
import os
import time
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from common import EXPERIMENT, ROOT, digest, differences, load_outcomes, load_structure, write_csv, write_json
from src.development import choose, choose_representation, group_name, identity_matrix, score, visual_design
from src.estimator import context_mapping, contextual_penalties, fit_logistic
from src.preprocessing import fit_preprocessing


def run() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    output_names = ["development_access.json", "all_fit_attempts.csv", "validation_results.csv",
                    "selected_models.json", "freeze.json", "freeze.sha256"]
    if any((EXPERIMENT / name).exists() for name in output_names):
        raise FileExistsError("Development or freeze already started; refusing to overwrite.")
    protocol = json.loads((EXPERIMENT / "protocol.json").read_text(encoding="utf-8"))
    coverage = json.loads((EXPERIMENT / "coverage.json").read_text(encoding="utf-8"))
    if coverage["status"] != "coverage_passed" or coverage["protocol_sha256"] != digest(EXPERIMENT / "protocol.json"):
        raise ValueError("Outcome-blind G3 coverage gate or protocol hash failed.")
    before = json.loads((EXPERIMENT / "root_hashes_before.json").read_text(encoding="utf-8"))
    protected = ["dataset\\train.csv", "artifacts\\rows_canonical.parquet",
                 "features\\H.parquet", "features\\C.parquet", "features\\Y.parquet",
                 "src\\development.py", "src\\estimator.py", "src\\preprocessing.py",
                 "src\\features.py", "src\\feature_y.py", "src\\split.py", "src\\canonicalize.py"]
    for name in protected:
        if digest(ROOT / name) != before[name]["sha256"]:
            raise ValueError(f"Pre-experiment input changed: {name}")
    fit, validation = load_structure("core_fit"), load_structure("validation")
    if len(fit) != coverage["core_fitting_responses"] or len(validation) != coverage["g3"]["validation"]["responses"]:
        raise ValueError("Population differs from outcome-blind coverage record.")
    overlap = {name: dict(sorted(Counter(row["partition"] for row in rows).items()))
               for name, rows in (("separate_core_fit", fit), ("g3_validation", validation))}
    write_json(EXPERIMENT / "development_access.json", {
        "status": "development_outcome_access_started",
        "outcomes_requested": ["separate core fitting", "G3 validation"],
        "g3_test_outcomes_requested": False,
        "original_g2_partition_overlap": overlap,
        "warning": "Original G2 test outcomes occur in the separate fitting data. This retrospective analysis is not an independent G2 or G3 confirmation.",
    })
    expected_hash = before["dataset\\train.csv"]["sha256"]
    yfit, yval = load_outcomes(fit, expected_hash), load_outcomes(validation, expected_hash)
    groups_fit, groups_val = [group_name(r) for r in fit], [group_name(r) for r in validation]
    reference, interactions = context_mapping(groups_fit, [r["pair_id"] for r in fit])
    cat_keys = sorted({r[f"{side}_cat"] for r in fit for side in ("a", "b")})
    fit_photos = {r[f"{side}_image"] for r in fit for side in ("a", "b")}
    selected, preprocessing, attempts = {}, {}, []

    def sweep(name, x, vx, penalty_fn, rhos, fallback=None):
        candidates = []
        if fallback is not None:
            pooled = {**fallback, "model": name, "pooled": True, "rho": 1e99,
                      "pooled_from": fallback["model"], "fit_attempts": [], "seconds": 0.0}
            candidates.append(pooled)
            attempts.append({k: v for k, v in pooled.items() if k != "coefficients"})
        for ridge in protocol["model_selection"]["ridge_grid"]:
            for rho in rhos:
                started = time.monotonic()
                fitted = fit_logistic(x, yfit, penalty_fn(ridge, rho))
                logits = np.asarray(vx @ fitted.coefficients).reshape(-1) if fitted.accepted else None
                valid = logits is not None and np.isfinite(logits).all()
                loss, brier, accuracy = score(yval, logits) if valid else (None, None, None)
                candidate = {
                    "model": name, "ridge": ridge, "rho": rho, "pooled": False,
                    "pooled_from": "", "accepted": bool(fitted.accepted and valid),
                    "validation_loss": loss, "brier": brier, "accuracy": accuracy,
                    "objective": fitted.objective if np.isfinite(fitted.objective) else None,
                    "gradient_max": fitted.gradient_max if np.isfinite(fitted.gradient_max) else None,
                    "coefficient_count": x.shape[1], "iterations": fitted.iterations,
                    "seconds": time.monotonic() - started, "fit_attempts": list(fitted.attempts),
                    "coefficients": fitted.coefficients.tolist(),
                }
                candidates.append(candidate)
                attempts.append({k: v for k, v in candidate.items() if k != "coefficients"})
        winner = choose(candidates, contextual=fallback is not None)
        print(f"{name}: validation log loss {winner['validation_loss']:.6f}; "
              f"ridge={winner['ridge']}, pooled={winner['pooled']}", flush=True)
        return winner

    with threadpool_limits(limits=1):
        p = (float(yfit.sum()) + 0.5) / (len(yfit) + 1)
        beta = float(np.log(p / (1 - p)))
        loss, brier, accuracy = score(yval, np.full(len(yval), beta))
        selected["M0"] = {"model": "M0", "coefficients": [beta], "accepted": True, "ridge": 0,
                          "rho": 0, "pooled": False, "validation_loss": loss, "brier": brier,
                          "accuracy": accuracy}
        x, vx = identity_matrix(fit, cat_keys, "cat"), identity_matrix(validation, cat_keys, "cat")
        selected["M1"] = sweep("M1", x, vx, lambda ridge, rho: np.r_[[1e-6], np.full(x.shape[1]-1, ridge)], [0])
        for block in ("H", "C", "Y"):
            records = pq.read_table(ROOT / "features" / f"{block}.parquet").to_pylist()
            spec = fit_preprocessing(block, records, fit_photos)
            preprocessing[block] = spec
            dx, dv = differences(block, spec, fit), differences(block, spec, validation)
            x, vx = visual_design(dx, groups_fit, ()), visual_design(dv, groups_val, ())
            dim = dx.shape[1]
            m3 = sweep("M3-" + block, x, vx,
                       lambda ridge, rho: np.r_[[1e-6], np.full(dim, ridge)], [0])
            selected["M3-" + block] = m3
            if interactions:
                x, vx = visual_design(dx, groups_fit, interactions), visual_design(dv, groups_val, interactions)
                m4 = sweep("M4-" + block, x, vx,
                           lambda ridge, rho: contextual_penalties(dim, len(interactions), ridge, rho),
                           protocol["model_selection"]["context_multipliers"], m3)
            else:
                m4 = {**m3, "model": "M4-" + block, "pooled": True, "rho": 1e99,
                      "pooled_from": m3["model"]}
                attempts.append({k: v for k, v in m4.items() if k != "coefficients"})
            selected["M4-" + block] = m4
    representation = choose_representation({b: selected["M3-" + b] for b in ("H", "C", "Y")})
    attempts_rows = [{**item, "fit_attempts": json.dumps(item.get("fit_attempts", []), allow_nan=False)}
                     for item in attempts]
    fields = ["model", "ridge", "rho", "pooled", "pooled_from", "accepted", "validation_loss", "brier",
              "accuracy", "objective", "gradient_max", "coefficient_count", "iterations", "seconds", "fit_attempts"]
    write_csv(EXPERIMENT / "all_fit_attempts.csv", attempts_rows, fields)
    validation_rows = [{k: model.get(k) for k in ("model", "ridge", "rho", "pooled", "validation_loss", "brier", "accuracy")}
                       for model in selected.values()]
    write_csv(EXPERIMENT / "validation_results.csv", validation_rows, list(validation_rows[0]))
    models = {"status": "validation_selected_retrospective", "selected_representation": representation,
              "selected": selected, "preprocessing": preprocessing, "cat_keys": cat_keys,
              "context": {"reference": reference, "interactions": list(interactions)},
              "fit_responses": len(fit), "validation_responses": len(validation)}
    write_json(EXPERIMENT / "selected_models.json", models)
    frozen_files = ["protocol.json", "prepare.py", "common.py", "develop.py", "evaluate.py",
                    "tests.py", "photo_reservation.csv", "core_split.csv", "row_allocation.csv",
                    "coverage.json", "development_access.json", "all_fit_attempts.csv",
                    "validation_results.csv", "selected_models.json"]
    freeze = {"status": "frozen_before_g3_test_access", "exploratory": True,
              "g3_test_outcomes_accessed": False, "selected_representation": representation,
              "experiment_sha256": {name: digest(EXPERIMENT / name) for name in frozen_files},
              "root_input_sha256": {name: before[name]["sha256"] for name in protected}}
    write_json(EXPERIMENT / "freeze.json", freeze)
    (EXPERIMENT / "freeze.sha256").write_text(digest(EXPERIMENT / "freeze.json") + "\n", encoding="ascii")
    print(f"Frozen G3 representation: {representation}", flush=True)


if __name__ == "__main__":
    run()
