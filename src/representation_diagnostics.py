"""Reproduce regularization paths and run registered representation probes."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

from .development import choose, group_name, score, visual_design
from .diagnostics import DevelopmentData, differences, load_data, write_csv
from .estimator import contextual_penalties, fit_logistic
from .preprocessing import transform_photos


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _refit(
    name: str, x: Any, vx: Any, y: np.ndarray, vy: np.ndarray,
    penalty: np.ndarray, ridge: float, rho: float,
) -> dict[str, Any]:
    start = time.monotonic()
    result = fit_logistic(x, y, penalty)
    if not result.accepted:
        raise RuntimeError(f"Unaccepted diagnostic fit: {name}, lambda={ridge}, rho={rho}: {result.message}")
    training = score(y, np.asarray(x @ result.coefficients).reshape(-1))[0]
    validation = score(vy, np.asarray(vx @ result.coefficients).reshape(-1))[0]
    return {
        "model": name, "ridge": ridge, "rho": rho, "accepted": True,
        "train_loss": training, "validation_loss": validation,
        "coefficient_norm": float(np.linalg.norm(result.coefficients[1:])),
        "gradient_max": result.gradient_max, "objective": result.objective,
        "iterations": result.iterations, "fit_attempts": json.dumps(result.attempts),
        "seconds": time.monotonic() - start,
    }


def _prepare_block(root: Path, data: DevelopmentData, block: str) -> tuple[Any, Any, dict[str, Any]]:
    spec = json.loads((root / "artifacts/preprocessing" / f"{block}.json").read_text(encoding="utf-8"))
    return differences(data.fit, data.features[block], spec), differences(data.val, data.features[block], spec), spec


def fit_paths(root: Path, data: DevelopmentData) -> list[dict[str, Any]]:
    originals = _read_csv(root / "tables/all_fit_attempts.csv")
    y = np.asarray([row["y_displayed"] for row in data.fit], dtype=np.float64)
    vy = np.asarray([row["y_displayed"] for row in data.val], dtype=np.float64)
    groups, val_groups = [group_name(row) for row in data.fit], [group_name(row) for row in data.val]
    interactions = tuple(data.selection["interaction_contexts"])
    records = []
    with threadpool_limits(limits=1):
        for block in ("H", "C", "Y"):
            dx, dv, _ = _prepare_block(root, data, block)
            for model, context in ((f"common:M3-{block}", False), (f"common:M4-{block}", True)):
                x = visual_design(dx, groups, interactions if context else ())
                vx = visual_design(dv, val_groups, interactions if context else ())
                relevant = [row for row in originals if row["model"] == model and row["pooled"] == "False"]
                for original in relevant:
                    ridge, rho = float(original["ridge"]), float(original["rho"])
                    penalty = contextual_penalties(dx.shape[1], len(interactions), ridge, rho) if context else \
                              np.r_[[1e-6], np.full(dx.shape[1], ridge)]
                    entry = _refit(model, x, vx, y, vy, penalty, ridge, rho)
                    entry["stage7_validation_loss"] = float(original["validation_loss"])
                    entry["reproduction_delta"] = entry["validation_loss"] - entry["stage7_validation_loss"]
                    if abs(entry["reproduction_delta"]) > 1e-6:
                        raise RuntimeError(f"Regularization-path refit differs from Stage 7: {model}/{ridge}/{rho}")
                    records.append(entry)
                print(f"Penalty path reproduced: {model}", flush=True)
    return records


def _group_name(feature_name: str) -> str:
    return feature_name.split("=", 1)[0].removesuffix("__missing")


def fit_ablations(root: Path, data: DevelopmentData, grid: list[float]) -> list[dict[str, Any]]:
    y = np.asarray([row["y_displayed"] for row in data.fit], dtype=np.float64)
    vy = np.asarray([row["y_displayed"] for row in data.val], dtype=np.float64)
    groups, val_groups = [group_name(row) for row in data.fit], [group_name(row) for row in data.val]
    fit_ids = sorted({row[f"{side}_image"] for row in data.fit for side in ("a", "b")})
    entries = []
    with threadpool_limits(limits=1):
        for block in ("H", "Y"):
            spec = json.loads((root / "artifacts/preprocessing" / f"{block}.json").read_text(encoding="utf-8"))
            by_id = {row["image_id"]: row for row in data.features[block]}
            vectors = transform_photos(spec, [by_id[image] for image in fit_ids])
            dx, dv, _ = _prepare_block(root, data, block)
            for group, members in data.protocol["ablation_groups"][block].items():
                indices = [i for i, name in enumerate(spec["final_names"]) if _group_name(name) not in members]
                dropped = [name for name in spec["final_names"] if _group_name(name) in members]
                if not dropped:
                    entries.append({"block": block, "group": group, "status": "no_retained_dimensions",
                                    "dropped_dimensions": 0, "selected": False, "ridge": "",
                                    "validation_loss": "", "baseline_validation_loss": data.selection["selected"][f"M3-{block}"]["validation_loss"],
                                    "coefficient_norm": "", "gradient_max": "", "fit_attempts": ""})
                    continue
                rms = float(np.sqrt(np.mean(np.sum(vectors[:, indices] ** 2, axis=1))))
                if not np.isfinite(rms) or rms < 1e-12:
                    raise ValueError(f"Ablated {block}/{group} has degenerate fitting RMS.")
                x = visual_design(dx[:, indices] / rms, groups, ())
                vx = visual_design(dv[:, indices] / rms, val_groups, ())
                candidates = []
                for ridge in grid:
                    row = _refit(f"ablation:{block}:{group}", x, vx, y, vy,
                                 np.r_[[1e-6], np.full(len(indices), ridge)], ridge, 0)
                    row.update({"pooled": False})
                    candidates.append(row)
                winner = choose(candidates)
                for row in candidates:
                    entries.append({
                        "block": block, "group": group, "status": "ok",
                        "dropped_dimensions": len(dropped), "selected": row is winner,
                        "ridge": row["ridge"], "validation_loss": row["validation_loss"],
                        "baseline_validation_loss": data.selection["selected"][f"M3-{block}"]["validation_loss"],
                        "coefficient_norm": row["coefficient_norm"], "gradient_max": row["gradient_max"],
                        "fit_attempts": row["fit_attempts"],
                    })
                print(f"Grouped ablation: {block}/{group}", flush=True)
    return entries


def fit_pca(data: DevelopmentData, grid: list[float]) -> list[dict[str, Any]]:
    block = data.selection["selected_representation"]
    if block != "C":
        raise ValueError("Registered PCA diagnostic requires the C block; do not silently substitute another.")
    records = {row["image_id"]: row for row in data.features["C"]}
    fit_ids = sorted({row[f"{side}_image"] for row in data.fit for side in ("a", "b")})
    all_ids = sorted({row[f"{side}_image"] for row in data.fit + data.val for side in ("a", "b")})
    columns = [f"cnn_{i:03d}" for i in range(576)]
    raw = np.asarray([[records[image][name] for name in columns] for image in fit_ids], dtype=np.float64)
    mean = raw.mean(axis=0)
    k = min(64, len(fit_ids)-1, raw.shape[1])
    if k < 1:
        raise ValueError("Not enough fitting photos for PCA.")
    with threadpool_limits(limits=1):
        _, _, vt = linalg.svd(raw - mean, full_matrices=False, check_finite=True)
    projected = (np.asarray([[records[image][name] for name in columns] for image in all_ids]) - mean) @ vt[:k].T
    index = {image: i for i, image in enumerate(all_ids)}
    train_z = projected[[index[image] for image in fit_ids]]
    rms = float(np.sqrt(np.mean(np.sum(train_z**2, axis=1))))
    if not np.isfinite(rms) or rms < 1e-12:
        raise ValueError("PCA diagnostic representation is degenerate.")
    projected /= rms
    from .development import feature_differences
    dx, dv = feature_differences(data.fit, all_ids, projected), feature_differences(data.val, all_ids, projected)
    x = visual_design(dx, [group_name(row) for row in data.fit], ())
    vx = visual_design(dv, [group_name(row) for row in data.val], ())
    y = np.asarray([row["y_displayed"] for row in data.fit], dtype=np.float64)
    vy = np.asarray([row["y_displayed"] for row in data.val], dtype=np.float64)
    with threadpool_limits(limits=1):
        candidates = [
            {**_refit("PCA-C", x, vx, y, vy, np.r_[[1e-6], np.full(k, ridge)], ridge, 0), "pooled": False}
            for ridge in grid
        ]
    winner = choose(candidates)
    return [{
        "model": "PCA-C", "dimensions": k, "whitening": False,
        "ridge": row["ridge"], "selected": row is winner, "validation_loss": row["validation_loss"],
        "baseline_validation_loss": data.selection["selected"]["M3-C"]["validation_loss"],
        "gradient_max": row["gradient_max"], "fit_attempts": row["fit_attempts"],
    } for row in candidates]


def run_representation_diagnostics(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    outputs = [root / "tables/penalty_path_refits.csv", root / "tables/grouped_ablations.csv",
               root / "tables/cnn_pca_diagnostic.csv", root / "artifacts/representation_diagnostic_status.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Representation diagnostic outputs already exist.")
    core = json.loads((root / "artifacts/stage8_core.json").read_text(encoding="utf-8"))
    if core["status"] != "passed_operational_control":
        raise ValueError("Fair-coin control has not passed its operational gate.")
    data = load_data(root)
    paths = fit_paths(root, data)
    write_csv(outputs[0], paths)
    ablations = fit_ablations(root, data, config["candidate_registry"]["base_ridge_grid"])
    write_csv(outputs[1], ablations)
    pca = fit_pca(data, config["candidate_registry"]["base_ridge_grid"])
    write_csv(outputs[2], pca)
    report = {"penalty_refits": len(paths), "ablation_fits": len(ablations),
              "pca_fits": len(pca), "status": "completed", "internal_test_outcomes_read": False}
    outputs[3].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
