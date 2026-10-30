"""Fit and freeze separate retrospective G4 models using G4 validation."""

from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter

import networkx as nx
import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from shared import HERE, ROOT, digest, differences, identity_design, load_outcomes, load_structure, write_csv, write_json
from src.development import choose, choose_representation, group_name, score, visual_design
from src.estimator import context_mapping, contextual_penalties, fit_logistic
from src.preprocessing import fit_preprocessing


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def run() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    outputs = ["development_access.json", "all_fit_attempts.csv", "validation_results.csv",
               "selected_models.json", "freeze.json", "freeze.sha256"]
    if any((HERE / name).exists() for name in outputs):
        raise FileExistsError("G4 development or freeze already started; refusing to overwrite.")
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    coverage = json.loads((HERE / "coverage.json").read_text(encoding="utf-8"))
    if coverage["status"] != "coverage_passed" or coverage["protocol_sha256"] != digest(HERE / "protocol.json"):
        raise ValueError("G4 outcome-blind coverage gate or protocol hash failed.")
    before = json.loads((HERE / "root_hashes_before.json").read_text(encoding="utf-8"))
    protected = ["dataset\\train.csv", "artifacts\\rows_canonical.parquet", "artifacts\\raw_hashes.json",
                 "features\\H.parquet", "features\\C.parquet", "features\\Y.parquet",
                 "src\\development.py", "src\\estimator.py", "src\\preprocessing.py",
                 "src\\features.py", "src\\feature_y.py", "src\\split.py", "src\\canonicalize.py",
                 "src\\evaluate.py", "src\\freeze.py", "g3_experiment\\prepare.py"]
    for name in protected:
        if digest(ROOT / name) != before[name]["sha256"]:
            raise ValueError(f"G4 pre-experiment input changed: {name}")
    fit, validation = load_structure("core_fit"), load_structure("validation")
    if len(fit) != coverage["core_fitting_responses"] or len(validation) != coverage["g4"]["validation"]["responses"]:
        raise ValueError("G4 population differs from the outcome-blind coverage record.")
    cats = sorted({row[f"{side}_cat"] for row in fit for side in ("a", "b")})
    graph = nx.Graph()
    graph.add_nodes_from(cats)
    graph.add_edges_from((r["a_cat"], r["b_cat"]) for r in fit if r["a_cat"] != r["b_cat"])
    if not nx.is_connected(graph):
        raise ValueError("Separate fitting cat graph is disconnected; cold-zero identity comparisons are not comparable.")
    fit_x = identity_design(fit, cats, cold_start=False)
    val_x = identity_design(validation, cats, cold_start=True)
    with (ROOT / "g3_experiment" / "row_allocation.csv").open(encoding="utf-8", newline="") as stream:
        g3 = {r["train_row_id"]: r["allocation"] for r in csv.DictReader(stream)}
    overlap = {}
    for name, rows in (("separate_core_fit", fit), ("g4_validation", validation)):
        overlap[name] = {"original_g2_partitions": dict(sorted(Counter(r["partition"] for r in rows).items())),
                         "original_g3_allocations": dict(sorted(Counter(g3[r["train_row_id"]] for r in rows).items()))}
    write_json(HERE / "development_access.json", {
        "status": "development_outcome_access_started",
        "outcomes_requested": ["separate core fitting", "G4 validation"],
        "g4_test_outcomes_requested": False, "previous_partition_overlap": overlap,
        "warning": "Some fitting and validation outcomes were previously exposed during G2/G3; G4 is retrospective.",
        "fitting_cat_graph_connected": True, "fitting_cats": len(cats),
    })
    yfit = load_outcomes(fit, before["dataset\\train.csv"]["sha256"])
    yval = load_outcomes(validation, before["dataset\\train.csv"]["sha256"])
    fit_groups = [group_name(r) for r in fit]
    val_groups = [group_name(r) for r in validation]
    reference, interactions = context_mapping(fit_groups, [r["pair_id"] for r in fit])
    fit_photos = {r[f"{side}_image"] for r in fit for side in ("a", "b")}
    selected, preprocessing = {}, {}
    fields = ["model", "ridge", "rho", "pooled", "pooled_from", "accepted", "validation_loss", "brier",
              "accuracy", "objective", "gradient_max", "coefficient_count", "iterations", "seconds", "fit_attempts"]
    with (HERE / "all_fit_attempts.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        stream.flush()

        def log(candidate):
            row = {key: candidate.get(key) for key in fields}
            row["fit_attempts"] = json.dumps(json_safe(candidate.get("fit_attempts", [])), allow_nan=False)
            writer.writerow(row)
            stream.flush()

        def sweep(name, x, vx, penalty_fn, rhos, fallback=None):
            candidates = []
            if fallback is not None:
                pooled = {**fallback, "model": name, "pooled": True, "rho": 1e99,
                          "pooled_from": fallback["model"], "fit_attempts": [], "seconds": 0.}
                candidates.append(pooled)
                log(pooled)
            for ridge in protocol["models"]["ridge_grid"]:
                for rho in rhos:
                    started = time.monotonic()
                    fitted = fit_logistic(x, yfit, penalty_fn(ridge, rho))
                    logits = np.asarray(vx @ fitted.coefficients).reshape(-1) if fitted.accepted else None
                    valid = logits is not None and np.isfinite(logits).all()
                    loss, brier, accuracy = score(yval, logits) if valid else (None, None, None)
                    candidate = {"model": name, "ridge": ridge, "rho": rho, "pooled": False,
                                 "pooled_from": "", "accepted": bool(fitted.accepted and valid),
                                 "validation_loss": loss, "brier": brier, "accuracy": accuracy,
                                 "objective": fitted.objective if np.isfinite(fitted.objective) else None,
                                 "gradient_max": fitted.gradient_max if np.isfinite(fitted.gradient_max) else None,
                                 "coefficient_count": x.shape[1], "iterations": fitted.iterations,
                                 "seconds": time.monotonic()-started, "fit_attempts": list(fitted.attempts),
                                 "coefficients": fitted.coefficients.tolist()}
                    candidates.append(candidate)
                    log(candidate)
            winner = choose(candidates, contextual=fallback is not None)
            print(f"{name}: validation log loss {winner['validation_loss']:.6f}; "
                  f"ridge={winner['ridge']}, pooled={winner['pooled']}", flush=True)
            return winner

        with threadpool_limits(limits=1):
            p = (float(yfit.sum()) + .5) / (len(yfit) + 1)
            beta = float(np.log(p / (1-p)))
            loss, brier, accuracy = score(yval, np.full(len(yval), beta))
            selected["M0"] = {"model": "M0", "coefficients": [beta], "accepted": True, "ridge": 0,
                              "rho": 0, "pooled": False, "validation_loss": loss, "brier": brier,
                              "accuracy": accuracy, "coefficient_count": 1}
            log(selected["M0"])
            width = fit_x.shape[1]
            selected["M1-zero"] = sweep("M1-zero", fit_x, val_x,
                                        lambda ridge, rho: np.r_[[1e-6], np.full(width-1, ridge)], [0])
            if abs(float(np.mean(selected["M1-zero"]["coefficients"][1:]))) > 1e-8:
                raise ValueError("Selected fitting-cat scores are not centered as expected under ridge.")
            for block in ("H", "C", "Y"):
                records = pq.read_table(ROOT / "features" / f"{block}.parquet").to_pylist()
                spec = fit_preprocessing(block, records, fit_photos)
                preprocessing[block] = spec
                dx, dv = differences(block, spec, fit), differences(block, spec, validation)
                x, vx = visual_design(dx, fit_groups, ()), visual_design(dv, val_groups, ())
                dimension = dx.shape[1]
                m3 = sweep("M3-" + block, x, vx,
                           lambda ridge, rho: np.r_[[1e-6], np.full(dimension, ridge)], [0])
                selected["M3-" + block] = m3
                if interactions:
                    x = visual_design(dx, fit_groups, interactions)
                    vx = visual_design(dv, val_groups, interactions)
                    m4 = sweep("M4-" + block, x, vx,
                               lambda ridge, rho: contextual_penalties(dimension, len(interactions), ridge, rho),
                               protocol["models"]["context_multipliers"], m3)
                else:
                    m4 = {**m3, "model": "M4-" + block, "pooled": True, "rho": 1e99,
                          "pooled_from": m3["model"]}
                    log(m4)
                selected["M4-" + block] = m4
    representation = choose_representation({b: selected["M3-" + b] for b in ("H", "C", "Y")})
    validation_rows = [{k: model.get(k) for k in ("model", "ridge", "rho", "pooled", "validation_loss", "brier", "accuracy")}
                       for model in selected.values()]
    write_csv(HERE / "validation_results.csv", validation_rows, list(validation_rows[0]))
    models = {"status": "validation_selected_retrospective", "selected_representation": representation,
              "selected": selected, "preprocessing": preprocessing, "cat_keys": cats,
              "context": {"reference": reference, "interactions": list(interactions)},
              "fit_responses": len(fit), "validation_responses": len(validation),
              "m1_zero_convention": protocol["models"]["M1-zero"]}
    write_json(HERE / "selected_models.json", models)
    files = ["protocol.json", "reserve.py", "shared.py", "develop.py", "evaluate.py", "tests.py",
             "cat_reservation.csv", "core_split.csv", "row_allocation.csv", "coverage.json",
             "development_access.json", "all_fit_attempts.csv", "validation_results.csv", "selected_models.json"]
    freeze = {"status": "frozen_before_g4_test_access", "exploratory": True,
              "g4_test_outcomes_accessed": False, "selected_representation": representation,
              "experiment_sha256": {name: digest(HERE / name) for name in files},
              "root_input_sha256": {name: before[name]["sha256"] for name in protected}}
    write_json(HERE / "freeze.json", freeze)
    (HERE / "freeze.sha256").write_text(digest(HERE / "freeze.json") + "\n", encoding="ascii")
    print(f"Frozen G4 representation: {representation}", flush=True)


if __name__ == "__main__":
    run()
