"""Nested-pair learning curves with fitting-only feature preprocessing."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from .development import choose, group_name, identity_matrix, score, visual_design
from .diagnostics import DevelopmentData, differences, load_data, write_csv
from .estimator import context_mapping, contextual_penalties, fit_logistic
from .feature_quality import Components
from .preprocessing import fit_preprocessing


def nested_pair_sets(fit: list[dict[str, Any]], seed: int, sizes: list[float]) -> dict[float, set[str]]:
    pairs = sorted({row["pair_id"] for row in fit})
    random.Random(seed).shuffle(pairs)
    if not pairs or any(size <= 0 or size > 1 for size in sizes):
        raise ValueError("Invalid nested learning-curve sizes or fitting pairs.")
    return {size: set(pairs[:max(1, int(len(pairs) * size))]) for size in sizes}


def common_validation_indices(
    fit: list[dict[str, Any]], val: list[dict[str, Any]], seeds: list[int], sizes: list[float]
) -> list[int]:
    eligible = set(range(len(val)))
    for seed in seeds:
        nested = nested_pair_sets(fit, seed, sizes)
        for size in sizes:
            graph = Components()
            for row in fit:
                if row["pair_id"] in nested[size]:
                    graph.add_pair(row["a_image"], row["b_image"])
            eligible &= {i for i, row in enumerate(val) if graph.supports(row["a_image"], row["b_image"])}
    return sorted(eligible)


def curve_sweep(
    model: str, train_x: Any, val_x: Any, y: np.ndarray, vy: np.ndarray,
    penalty: Any, grid: list[float], rhos: list[float], fallback: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    if fallback is not None:
        rows.append({
            **fallback, "model": model, "pooled": True, "rho": float("inf"),
            "fit_attempts": [], "seconds": 0.0,
        })
    for ridge in grid:
        for rho in rhos:
            start = time.monotonic()
            result = fit_logistic(train_x, y, penalty(ridge, rho))
            loss = score(vy, np.asarray(val_x @ result.coefficients).reshape(-1))[0] if result.accepted else None
            rows.append({
                "model": model, "ridge": ridge, "rho": rho, "pooled": False,
                "accepted": result.accepted, "validation_loss": loss,
                "gradient_max": result.gradient_max if np.isfinite(result.gradient_max) else None,
                "fit_attempts": list(result.attempts), "seconds": time.monotonic() - start,
            })
    return choose(rows, contextual=fallback is not None), rows


def calculate_curves(data: DevelopmentData, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sizes = list(data.protocol["learning_curve_sizes"])
    seeds = data.protocol["learning_curve_seeds"]
    threshold = data.protocol["learning_curve_validation_minimum"]
    indices = common_validation_indices(data.fit, data.val, seeds, sizes)
    pairs = len({data.val[i]["pair_id"] for i in indices})
    if len(indices) < threshold["responses"] or pairs < threshold["distinct_pairs"]:
        sizes = list(data.protocol["learning_curve_fallback_sizes"])
        indices = common_validation_indices(data.fit, data.val, seeds, sizes)
        pairs = len({data.val[i]["pair_id"] for i in indices})
    if len(indices) < threshold["responses"] or pairs < threshold["distinct_pairs"]:
        return [], [], {"status": "omitted_inadequate_common_validation", "responses": len(indices),
                        "pairs": pairs, "sizes": sizes}
    val = [data.val[i] for i in indices]
    vy = np.asarray([row["y_displayed"] for row in val], dtype=np.float64)
    block = data.selection["selected_representation"]
    grid = config["candidate_registry"]["base_ridge_grid"]
    rhos = config["candidate_registry"]["context_penalty_multipliers"]
    results: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for seed in seeds:
            pair_sets = nested_pair_sets(data.fit, seed, sizes)
            for size in sizes:
                train = [row for row in data.fit if row["pair_id"] in pair_sets[size]]
                y = np.asarray([row["y_displayed"] for row in train], dtype=np.float64)
                images = {row[f"{side}_image"] for row in train for side in ("a", "b")}
                reference, interactions = context_mapping(
                    [group_name(row) for row in train], [row["pair_id"] for row in train]
                )
                groups, vg = [group_name(row) for row in train], [group_name(row) for row in val]
                spec = fit_preprocessing(block, data.features[block], images)
                dx = differences(train, data.features[block], spec)
                dv = differences(val, data.features[block], spec)
                keys = sorted(images)
                x2, v2 = identity_matrix(train, keys, "image"), identity_matrix(val, keys, "image")
                x3, v3 = visual_design(dx, groups, ()), visual_design(dv, vg, ())
                dim = dx.shape[1]
                winner2, all2 = curve_sweep(
                    "M2", x2, v2, y, vy, lambda ridge, _rho: np.r_[[1e-6], np.full(len(keys), ridge)],
                    grid, [0.],
                )
                winner3, all3 = curve_sweep(
                    "M3*", x3, v3, y, vy, lambda ridge, _rho: np.r_[[1e-6], np.full(dim, ridge)],
                    grid, [0.],
                )
                model_groups: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = [
                    ("M2", winner2, all2), ("M3*", winner3, all3),
                ]
                if interactions:
                    x4 = visual_design(dx, groups, interactions)
                    v4 = visual_design(dv, vg, interactions)
                    winner4, all4 = curve_sweep(
                        "M4*", x4, v4, y, vy,
                        lambda ridge, rho: contextual_penalties(dim, len(interactions), ridge, rho),
                        grid, rhos, winner3,
                    )
                    model_groups.append(("M4*", winner4, all4))
                else:
                    results.append({
                        "seed": seed, "fraction": size, "model": "M4*", "status": "unsupported",
                        "fitting_responses": len(train), "fitting_pairs": len(pair_sets[size]),
                        "fitting_photos": len(images), "validation_responses": len(val),
                        "validation_pairs": pairs, "reference_context": reference or "",
                        "interaction_contexts": json.dumps(list(interactions)), "ridge": "",
                        "rho": "", "pooled": True, "validation_loss": "",
                    })
                for model, winner, candidates in model_groups:
                    row = {
                        "seed": seed, "fraction": size, "model": model, "status": "ok",
                        "fitting_responses": len(train), "fitting_pairs": len(pair_sets[size]),
                        "fitting_photos": len(images), "validation_responses": len(val),
                        "validation_pairs": pairs, "reference_context": reference or "",
                        "interaction_contexts": json.dumps(list(interactions)),
                        "ridge": winner["ridge"], "rho": "exact_pooled" if winner["pooled"] else winner["rho"],
                        "pooled": winner["pooled"], "validation_loss": winner["validation_loss"],
                    }
                    results.append(row)
                    for candidate in candidates:
                        attempts.append({
                            "seed": seed, "fraction": size, "model": model,
                            "ridge": candidate["ridge"], "rho": "exact_pooled" if candidate["pooled"] else candidate["rho"],
                            "pooled": candidate["pooled"], "accepted": candidate["accepted"],
                            "selected": candidate is winner, "validation_loss": candidate["validation_loss"],
                            "gradient_max": candidate["gradient_max"],
                            "fit_attempts": json.dumps(candidate["fit_attempts"]), "seconds": candidate["seconds"],
                        })
                print(f"Learning curve seed={seed} size={size:.2f} responses={len(train)}", flush=True)
    return results, attempts, {"status": "completed", "responses": len(indices), "pairs": pairs, "sizes": sizes}


def run_curves(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    outputs = [root / "tables/learning_curves.csv", root / "tables/learning_curve_candidates.csv",
               root / "artifacts/learning_curve_status.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Learning-curve outputs already exist.")
    core = json.loads((root / "artifacts/stage8_core.json").read_text(encoding="utf-8"))
    if core["status"] != "passed_operational_control":
        raise ValueError("Fair-coin control has not passed its operational gate.")
    data = load_data(root)
    results, attempts, status = calculate_curves(data, config)
    result_fields = ["seed", "fraction", "model", "status", "fitting_responses", "fitting_pairs",
                     "fitting_photos", "validation_responses", "validation_pairs", "reference_context",
                     "interaction_contexts", "ridge", "rho", "pooled", "validation_loss"]
    attempt_fields = ["seed", "fraction", "model", "ridge", "rho", "pooled", "accepted", "selected",
                      "validation_loss", "gradient_max", "fit_attempts", "seconds"]
    write_csv(outputs[0], results, result_fields)
    write_csv(outputs[1], attempts, attempt_fields)
    outputs[2].write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status
