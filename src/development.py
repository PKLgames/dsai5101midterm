"""Fit registered development models without reading internal-test outcomes."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .estimator import context_mapping, contextual_penalties, fit_logistic
from .preprocessing import fit_preprocessing, transform_photos


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def verify_inputs(root: Path) -> None:
    for path in ("artifacts/graph_provenance.json", "artifacts/simulation_provenance.json",
                 "artifacts/stage6_provenance.json"):
        report = json.loads((root / path).read_text(encoding="utf-8"))
        if path.endswith("graph_provenance.json") and report.get("status") != "graphs_and_masks_created":
            raise ValueError("Stage 4 graph gate is not satisfied.")
        if path.endswith("simulation_provenance.json") and report.get("failed_fits") != 0:
            raise ValueError("Stage 5 synthetic estimator gate is not satisfied.")
        for group in ("input_sha256", "inputs_sha256", "source_sha256", "output_sha256", "outputs_sha256"):
            for name, digest in report.get(group, {}).items():
                candidate = root.joinpath(*name.split("/"))
                if not candidate.is_file() or _hash(candidate) != digest:
                    raise ValueError(f"{path} has stale provenance: {name}")


def load_development(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = [
        "train_row_id", "partition", "pair_id", "a_image", "b_image", "a_cat", "b_cat",
        "comparison_type", "pet_care", "y_displayed",
    ]
    rows = pq.read_table(
        root / "artifacts/rows_canonical.parquet", columns=fields,
        filters=[("partition", "in", ["fit", "validation"])],
    ).to_pylist()
    masks = pq.read_table(
        root / "artifacts/g2_eligibility.parquet",
        columns=["train_row_id", "partition", "full_strict_g2", "common_strict_g2", "common_visual_fit"],
        filters=[("partition", "in", ["fit", "validation"])],
    ).to_pylist()
    index = {row["train_row_id"]: row for row in masks}
    if len(index) != len(masks) or len(index) != len(rows):
        raise ValueError("Development masks have duplicate or missing response IDs.")
    for row in rows:
        if row["partition"] not in {"fit", "validation"} or row["y_displayed"] not in (0, 1):
            raise ValueError("Invalid or held-out outcome encountered in development.")
        mask = index[row["train_row_id"]]
        if mask["partition"] != row["partition"]:
            raise ValueError("Population mask partition mismatch.")
        row.update({name: mask[name] for name in ("full_strict_g2", "common_strict_g2", "common_visual_fit")})
    return ([row for row in rows if row["partition"] == "fit"],
            [row for row in rows if row["partition"] == "validation"])


def identity_matrix(rows: list[dict[str, Any]], keys: list[str], entity: str, side: bool = True) -> sparse.csr_matrix:
    if entity not in {"cat", "image"}:
        raise ValueError("Unknown identity model.")
    index = {key: i for i, key in enumerate(keys)}
    if len(index) != len(keys):
        raise ValueError("Identity keys must be unique.")
    n = len(rows)
    rr, cc, values = [], [], []
    for i, row in enumerate(rows):
        for prefix, sign in (("a", 1.), ("b", -1.)):
            key = row[f"{prefix}_{entity}"]
            if key not in index:
                raise ValueError(f"Unknown fitting {entity} in matched validation: {key}")
            rr.append(i)
            cc.append(index[key])
            values.append(sign)
    block = sparse.coo_matrix((values, (rr, cc)), shape=(n, len(keys))).tocsr()
    block.eliminate_zeros()
    return sparse.hstack([np.ones((n, 1)), block], format="csr") if side else block


def feature_differences(rows: list[dict[str, Any]], photo_ids: list[str], values: np.ndarray) -> sparse.csr_matrix:
    index = {image: i for i, image in enumerate(photo_ids)}
    if len(index) != len(photo_ids) or values.shape[0] != len(photo_ids):
        raise ValueError("Feature lookup must have one row per image ID.")
    try:
        a = np.asarray([index[row["a_image"]] for row in rows])
        b = np.asarray([index[row["b_image"]] for row in rows])
    except KeyError as exc:
        raise ValueError(f"Feature unavailable for matched response: {exc}") from exc
    return sparse.csr_matrix(values[a] - values[b])


def group_name(row: dict[str, Any]) -> str:
    value = row["pet_care"]
    if value == "":
        return "missing"
    if value not in {"never", "occasionally", "past_regular", "current_regular"}:
        raise ValueError(f"Unexpected pet_care category: {value!r}")
    return value


def visual_design(diff: sparse.csr_matrix, groups: list[str], interactions: tuple[str, ...],
                  side_groups: tuple[str, ...] = ()) -> sparse.csr_matrix:
    if diff.shape[0] != len(groups):
        raise ValueError("Context vector length does not match design.")
    categories = np.asarray(groups)
    blocks: list[Any] = [np.ones((len(groups), 1))]
    blocks.extend(sparse.csr_matrix((categories == group).astype(float)[:, None]) for group in side_groups)
    blocks.append(diff)
    blocks.extend(sparse.diags((categories == group).astype(float)) @ diff for group in interactions)
    return sparse.hstack(blocks, format="csr")


def score(y: np.ndarray, logits: np.ndarray) -> tuple[float, float, float]:
    if logits.shape != y.shape or not np.isfinite(logits).all():
        raise ValueError("Missing or nonfinite validation logits.")
    losses = y * np.logaddexp(0., -logits) + (1. - y) * np.logaddexp(0., logits)
    p = expit(logits)
    return float(losses.mean()), float(np.mean((p-y)**2)), float(np.mean((p >= .5) == y))


def choose(candidates: list[dict[str, Any]], contextual: bool = False) -> dict[str, Any]:
    available = [row for row in candidates if row["accepted"] and row["validation_loss"] is not None]
    if not available:
        raise ValueError("No numerically accepted candidate has complete validation predictions.")
    minimum = min(row["validation_loss"] for row in available)
    tied = [row for row in available if row["validation_loss"] <= minimum + 1e-6]
    if contextual:
        return max(tied, key=lambda row: (row["pooled"], row["ridge"], row["rho"]))
    return max(tied, key=lambda row: row["ridge"])


def choose_representation(candidates: dict[str, dict[str, Any]]) -> str:
    if not candidates:
        raise ValueError("No accepted visual representation.")
    minimum = min(row["validation_loss"] for row in candidates.values())
    return next(block for block in ("H", "C", "Y")
                if block in candidates and candidates[block]["validation_loss"] <= minimum + 1e-6)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_development(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    verify_inputs(root)
    for test_file in ("test_objective.py", "test_recovery.py", "test_orientation.py",
                      "test_preprocessing.py", "test_development.py"):
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", test_file],
            cwd=root, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Required Stage 7 test gate failed: {test_file}")
    adequacy_path = root / "artifacts/stage7_adequacy.json"
    adequacy = json.loads(adequacy_path.read_text(encoding="utf-8"))
    if adequacy["decision"] != "accept_common_population_for_registered_comparison":
        raise ValueError("Stage 7 common-population adequacy has not been accepted.")
    destinations = [root / "models", root / "artifacts/preprocessing",
                    root / "tables/all_fit_attempts.csv", root / "tables/validation_results.csv",
                    root / "artifacts/model_selection.json"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Stage 7 outputs already exist; refusing to overwrite.")
    fit, validation = load_development(root)
    full_val = [row for row in validation if row["full_strict_g2"]]
    common_fit = [row for row in fit if row["common_visual_fit"]]
    common_val = [row for row in validation if row["common_strict_g2"]]
    if not full_val or not common_fit or not common_val:
        raise ValueError("Required fitting or strict-validation population is empty.")
    if (len(common_fit) != adequacy["fitting_responses"] or
        len(common_val) != adequacy["strict_validation_responses"] or
        len({row["pair_id"] for row in common_val}) != adequacy["strict_validation_pairs"]):
        raise ValueError("Common population differs from the accepted adequacy record.")
    registry = json.loads((root / "artifacts/common_population_report.json").read_text(encoding="utf-8"))
    blocks = registry["admitted_blocks"]
    if blocks != adequacy["feature_blocks"] or blocks != ["H", "C", "Y"]:
        raise ValueError("Feature admissions differ from the accepted common population.")
    ridge_grid = config["candidate_registry"]["base_ridge_grid"]
    rho_grid = config["candidate_registry"]["context_penalty_multipliers"]
    if ridge_grid != [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1] or rho_grid != [1, 10, 100]:
        raise ValueError("Unexpected registered regularization grid.")
    for path in destinations[:2]:
        path.mkdir(parents=True)
    yfit = np.asarray([row["y_displayed"] for row in common_fit], dtype=np.float64)
    yval = np.asarray([row["y_displayed"] for row in common_val], dtype=np.float64)
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    fields = ("model", "ridge", "rho", "pooled", "accepted", "validation_loss", "brier", "accuracy",
              "train_loss", "objective", "gradient_max", "coefficient_count", "pooled_from",
              "iterations", "seconds", "fit_attempts")
    log_path = root / "tables/all_fit_attempts.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=fields, lineterminator="\n").writeheader()

    def log_candidate(candidate: dict[str, Any]) -> None:
        attempts.append(candidate)
        record = {**candidate, "rho": "exact_pooled" if candidate["pooled"] else candidate["rho"],
                  "fit_attempts": json.dumps(candidate["fit_attempts"])}
        with log_path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n").writerow(record)

    def constant(name: str, train: list[dict[str, Any]], val: list[dict[str, Any]]) -> dict[str, Any]:
        yy = np.asarray([row["y_displayed"] for row in train], dtype=np.float64)
        y = np.asarray([row["y_displayed"] for row in val], dtype=np.float64)
        p = (float(yy.sum()) + .5) / (len(yy) + 1)
        beta = float(np.log(p / (1 - p)))
        loss, brier, accuracy = score(y, np.full(len(y), beta))
        candidate = {"model": name, "ridge": 0., "rho": 0., "pooled": False,
                     "accepted": True, "validation_loss": loss, "brier": brier,
                     "accuracy": accuracy, "objective": None, "gradient_max": None,
                     "train_loss": score(yy, np.full(len(yy), beta))[0],
                     "coefficient_count": 1, "pooled_from": "",
                     "iterations": 0, "seconds": 0., "fit_attempts": [], "coefficients": [beta],
                     "_validation_logits": np.full(len(y), beta)}
        log_candidate(candidate)
        return candidate

    def sweep(name: str, train_x: sparse.csr_matrix, train_y: np.ndarray,
              val_x: sparse.csr_matrix, val_y: np.ndarray, penalty_fn: Any,
              rhos: list[float], fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        if fallback is not None:
            pooled = {**fallback, "model": name, "pooled": True, "rho": float("inf"),
                      "pooled_from": fallback["model"], "fit_attempts": [], "seconds": 0.}
            candidates.append(pooled)
            log_candidate(pooled)
        for ridge in ridge_grid:
            for rho in rhos:
                start = time.monotonic()
                result = fit_logistic(train_x, train_y, penalty_fn(ridge, rho))
                logits = np.asarray(val_x @ result.coefficients).reshape(-1) if result.accepted else None
                valid = logits is not None and np.isfinite(logits).all()
                loss, brier, accuracy = score(val_y, logits) if valid else (None, None, None)
                train_logits = np.asarray(train_x @ result.coefficients).reshape(-1) if result.accepted else None
                train_loss = score(train_y, train_logits)[0] if train_logits is not None else None
                candidate = {
                    "model": name, "ridge": ridge, "rho": rho, "pooled": False,
                    "accepted": bool(result.accepted and valid), "validation_loss": loss,
                    "brier": brier, "accuracy": accuracy,
                    "objective": result.objective if np.isfinite(result.objective) else None,
                    "gradient_max": result.gradient_max if np.isfinite(result.gradient_max) else None,
                    "train_loss": train_loss, "coefficient_count": train_x.shape[1], "pooled_from": "",
                    "iterations": result.iterations, "seconds": time.monotonic() - start,
                    "fit_attempts": list(result.attempts), "coefficients": result.coefficients.tolist(),
                    "_validation_logits": logits,
                }
                candidates.append(candidate)
                log_candidate(candidate)
                print(f"{name} lambda={ridge:g} rho={rho:g} accepted={candidate['accepted']} "
                      f"validation={loss}", flush=True)
        return choose(candidates, contextual=fallback is not None)

    with threadpool_limits(limits=1):
        for population, train, val in (("full", fit, full_val), ("common", common_fit, common_val)):
            selected[f"{population}:M0"] = constant(f"{population}:M0", train, val)
            for entity, name, with_side in (("cat", "M1", True), ("image", "M2", True),
                                            ("image", "M2-no-position", False)):
                keys = sorted({row[f"{side}_{entity}"] for row in train for side in ("a", "b")})
                train_x = identity_matrix(train, keys, entity, with_side)
                val_x = identity_matrix(val, keys, entity, with_side)
                yy = np.asarray([row["y_displayed"] for row in train], dtype=np.float64)
                vy = np.asarray([row["y_displayed"] for row in val], dtype=np.float64)
                def penalties(ridge: float, rho: float, width: int = train_x.shape[1], side: bool = with_side) -> np.ndarray:
                    return np.r_[[1e-6], np.full(width-1, ridge)] if side else np.full(width, ridge)
                winner = sweep(f"{population}:{name}", train_x, yy, val_x, vy, penalties, [0.])
                winner["identity_keys"] = keys
                selected[f"{population}:{name}"] = winner

        fit_groups = [group_name(row) for row in common_fit]
        val_groups = [group_name(row) for row in common_val]
        reference, interactions = context_mapping(fit_groups, [row["pair_id"] for row in common_fit])
        selected["context"] = {"reference": reference, "interactions": list(interactions),
                               "unsupported_fallback": "zero deviation"}
        fit_images = {row[f"{side}_image"] for row in common_fit for side in ("a", "b")}
        required_images = {row[f"{side}_image"] for row in common_fit + common_val for side in ("a", "b")}
        representations: dict[str, dict[str, Any]] = {}
        blocks_data: dict[str, tuple[sparse.csr_matrix, sparse.csr_matrix]] = {}
        for block in blocks:
            all_rows = pq.read_table(root / "features" / f"{block}.parquet").to_pylist()
            by_id = {row["image_id"]: row for row in all_rows}
            if len(by_id) != len(all_rows) or not required_images <= set(by_id):
                raise ValueError(f"{block} features do not cover the matched population.")
            if any(by_id[image]["extraction_status"] != "ok" for image in required_images):
                raise ValueError(f"{block} has unusable features in the matched population.")
            spec = fit_preprocessing(block, all_rows, fit_images)
            _write_json(root / "artifacts/preprocessing" / f"{block}.json", spec)
            image_ids = sorted(required_images)
            vectors = transform_photos(spec, [by_id[image] for image in image_ids])
            train_diff = feature_differences(common_fit, image_ids, vectors)
            val_diff = feature_differences(common_val, image_ids, vectors)
            blocks_data[block] = (train_diff, val_diff)
            x = visual_design(train_diff, fit_groups, ())
            vx = visual_design(val_diff, val_groups, ())
            dim = train_diff.shape[1]
            penalties = lambda ridge, rho, dim=dim: np.r_[[1e-6], np.full(dim, ridge)]
            m3 = sweep(f"common:M3-{block}", x, yfit, vx, yval, penalties, [0.])
            representations[block] = m3
            selected[f"M3-{block}"] = m3
            if interactions:
                cx = visual_design(train_diff, fit_groups, interactions)
                cvx = visual_design(val_diff, val_groups, interactions)
                cp = lambda ridge, rho, dim=dim: contextual_penalties(dim, len(interactions), ridge, rho)
                m4 = sweep(f"common:M4-{block}", cx, yfit, cvx, yval, cp, rho_grid, fallback=m3)
                selected[f"M4-{block}"] = m4
            else:
                selected[f"M4-{block}"] = {"unsupported": True, "reason": "fewer than two eligible contexts"}

        winner = choose_representation(representations)
        selected["selected_representation"] = winner
        selected["M3*"] = selected[f"M3-{winner}"]
        selected["M4*"] = selected[f"M4-{winner}"]
        td, vd = blocks_data[winner]
        side_groups = interactions
        sx, sv = visual_design(td, fit_groups, (), side_groups), visual_design(vd, val_groups, (), side_groups)
        dim = td.shape[1]
        side_penalty = lambda ridge, rho: np.r_[[1e-6], np.full(len(side_groups)+dim, ridge)]
        side_m3 = sweep("common:M3-side", sx, yfit, sv, yval, side_penalty, [0.])
        selected["M3-side"] = side_m3
        if interactions:
            ix, iv = visual_design(td, fit_groups, interactions, side_groups), visual_design(vd, val_groups, interactions, side_groups)
            side_m4_penalty = lambda ridge, rho: np.r_[
                [1e-6], np.full(len(side_groups), ridge), np.full(dim, ridge),
                np.full(dim * len(interactions), ridge * rho),
            ]
            selected["M4-side"] = sweep("common:M4-side", ix, yfit, iv, yval,
                                        side_m4_penalty, rho_grid, fallback=side_m3)
        else:
            selected["M4-side"] = {"unsupported": True, "reason": "fewer than two eligible contexts"}

    # Infinity is a tie-order sentinel; serialize pooled rho as an explicit string.
    for candidate in attempts:
        if candidate["pooled"]:
            candidate["rho"] = "exact_pooled"
    results = [value for key, value in selected.items()
               if isinstance(value, dict) and "validation_loss" in value and key not in {"M3*", "M4*"}]
    with (root / "tables/validation_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for candidate in results:
            writer.writerow({**candidate, "fit_attempts": json.dumps(candidate["fit_attempts"])})
    predictions_path = root / "tables/validation_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "train_row_id", "logit", "probability"],
                                lineterminator="\n")
        writer.writeheader()
        for key, value in selected.items():
            if not isinstance(value, dict) or "_validation_logits" not in value or key in {"M3*", "M4*"}:
                continue
            val_rows = full_val if key.startswith("full:") else common_val
            logits = value["_validation_logits"]
            if len(logits) != len(val_rows):
                raise ValueError(f"Selected model has missing validation predictions: {key}")
            for row, logit in zip(val_rows, logits):
                writer.writerow({"model": key, "train_row_id": row["train_row_id"],
                                 "logit": float(logit), "probability": float(expit(logit))})
    for candidate in attempts:
        candidate.pop("_validation_logits", None)
    for key, value in selected.items():
        if isinstance(value, dict) and "coefficients" in value and key not in {"M3*", "M4*"}:
            _write_json(root / "models" / (key.replace(":", "_") + ".json"), value)
    _write_json(root / "artifacts/model_selection.json", {
        "selected_representation": winner, "selected": selected,
        "fitting_responses": len(common_fit), "validation_responses": len(common_val),
        "validation_pair_count": len({row["pair_id"] for row in common_val}),
        "reference_context": reference, "interaction_contexts": list(interactions),
        "internal_test_outcomes_read": False,
        "input_sha256": {name: _hash(root.joinpath(*name.split("/"))) for name in (
            "artifacts/rows_canonical.parquet", "artifacts/g2_eligibility.parquet",
            "artifacts/stage7_adequacy.json", "artifacts/graph_provenance.json",
            "artifacts/stage6_provenance.json", "artifacts/simulation_provenance.json",
            "src/estimator.py", "src/preprocessing.py", "src/development.py", "decisions.yaml",
        )},
        "synthetic_label_control": "pending Stage 8 after fixed development selection",
    })
    return {"selected_representation": winner, "candidate_count": len(attempts)}
