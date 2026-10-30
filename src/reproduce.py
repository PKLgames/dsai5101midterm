"""Fresh-process reconstruction checks; never overwrite frozen analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .development import _hash, feature_differences, group_name, identity_matrix, load_development, score, visual_design
from .estimator import fit_logistic
from .evaluate import load_test_structure, pair_bootstrap, interval
from .freeze import validate_freeze
from .preprocessing import fit_preprocessing, transform_photos


def verify_history(root: Path) -> dict[str, Any]:
    frozen = validate_freeze(root)
    for name in ("artifacts/graph_provenance.json", "artifacts/stage6_provenance.json",
                 "artifacts/stage7_verification.json", "artifacts/stage8_core.json",
                 "artifacts/stage8_provenance.json", "artifacts/evaluation_provenance.json",
                 "artifacts/presentation_feature_provenance.json", "artifacts/stage10_5_provenance.json"):
        report = json.loads((root / name).read_text(encoding="utf-8"))
        for key in ("input_sha256", "inputs_sha256", "source_sha256", "output_sha256", "outputs_sha256"):
            for source, expected in report.get(key, {}).items():
                candidate = root.joinpath(*source.split("/"))
                if not candidate.is_file() or _hash(candidate) != expected:
                    raise ValueError(f"History check failed: {source}")
    stage10 = json.loads((root / "artifacts/stage10_provenance.json").read_text(encoding="utf-8"))
    for name, digest in stage10["input_sha256"].items():
        if _hash(root / name) != digest:
            raise ValueError(f"Stage 10 source changed: {name}")
    append = json.loads((root / "artifacts/stage10_5_provenance.json").read_text(encoding="utf-8"))
    for name, lineage in append["append_lineage"].items():
        raw = (root / name).read_bytes()
        if (_hash(root / name) != lineage["current_sha256"] or
            hashlib.sha256(raw[:lineage["parent_bytes"]]).hexdigest() != lineage["parent_sha256"] or
            lineage["parent_sha256"] != stage10["output_sha256"][name]):
            raise ValueError(f"Synthesis append lineage changed: {name}")
    evaluation = json.loads((root / "artifacts/evaluation_provenance.json").read_text(encoding="utf-8"))
    if evaluation["freeze_sha256"] != _hash(root / "artifacts/model_freeze.json"):
        raise ValueError("Evaluation freeze does not match the saved record.")
    return frozen


def _visual_differences(
    rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]], spec: dict[str, Any],
) -> Any:
    by_id = {row["image_id"]: row for row in feature_rows}
    ids = sorted({row[f"{side}_image"] for row in rows for side in ("a", "b")})
    if not set(ids) <= set(by_id) or any(by_id[image]["extraction_status"] != "ok" for image in ids):
        raise ValueError("Cached feature reproduction has unavailable endpoint features.")
    return feature_differences(rows, ids, transform_photos(spec, [by_id[image] for image in ids]))


def reproduce_statistics(root: Path, tolerance: float = 1e-6) -> dict[str, Any]:
    output = root / "artifacts/reproduction_statistical.json"
    if output.exists():
        raise FileExistsError("Statistical reproduction log already exists.")
    frozen = verify_history(root)
    started = time.monotonic()
    selection = json.loads((root / "artifacts/model_selection.json").read_text(encoding="utf-8"))
    chosen = selection["selected"]
    fit, val = load_development(root)
    common_fit = [row for row in fit if row["common_visual_fit"]]
    fits = {"full": fit, "common": common_fit}
    vals = {"full": [row for row in val if row["full_strict_g2"]],
            "common": [row for row in val if row["common_strict_g2"]]}
    test = load_test_structure(root)
    tests = {"full": [row for row in test if row["full_strict_g2"]],
             "common": [row for row in test if row["common_strict_g2"]]}
    cached = pq.read_table(root / "artifacts/test_predictions.parquet").to_pylist()
    saved_test: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in cached:
        saved_test.setdefault((row["population"], row["model_id"]), []).append(row)
    features, specs = {}, {}
    images = {row[f"{side}_image"] for row in common_fit for side in ("a", "b")}
    for block in ("H", "C", "Y"):
        features[block] = pq.read_table(root / "features" / f"{block}.parquet").to_pylist()
        saved = json.loads((root / "artifacts/preprocessing" / f"{block}.json").read_text(encoding="utf-8"))
        recomputed = fit_preprocessing(block, features[block], images)
        if saved != recomputed:
            raise ValueError(f"Unique-fitting-photo preprocessing differs: {block}")
        specs[block] = recomputed
    with (root / "tables/main_results.csv").open(encoding="utf-8", newline="") as stream:
        expected = {(row["population"], row["model_id"]): row for row in csv.DictReader(stream)}
    validation_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    records = []
    interactions = tuple(selection["interaction_contexts"])
    matrices = {
        block: tuple(_visual_differences(rows, features[block], specs[block])
                     for rows in (common_fit, vals["common"], tests["common"]))
        for block in ("H", "C", "Y")
    }
    with threadpool_limits(limits=1):
        for key in frozen["selected_model_keys"]:
            model = chosen[key]
            population = "full" if key.startswith("full:") else "common"
            training, validation, heldout = fits[population], vals[population], tests[population]
            yy = np.asarray([row["y_displayed"] for row in training], dtype=np.float64)
            vy = np.asarray([row["y_displayed"] for row in validation], dtype=np.float64)
            result = None
            if model["pooled"]:
                source = model["pooled_from"].removeprefix("common:")
                vl = validation_predictions[source].copy()
                tl = test_predictions[source].copy()
            elif key.endswith("M0"):
                p = (yy.sum()+.5)/(len(yy)+1)
                beta = float(np.log(p/(1-p)))
                vl, tl = np.full(len(validation), beta), np.full(len(heldout), beta)
            else:
                if key.endswith("M1") or key.endswith("M2") or key.endswith("M2-no-position"):
                    entity = "cat" if key.endswith("M1") else "image"
                    side = not key.endswith("M2-no-position")
                    x, vx, tx = [identity_matrix(rows, model["identity_keys"], entity, side)
                                 for rows in (training, validation, heldout)]
                    penalty = (np.r_[[1e-6], np.full(x.shape[1]-1, model["ridge"])]
                               if side else np.full(x.shape[1], model["ridge"]))
                else:
                    block = frozen["selected_representation"] if "side" in key else key[-1]
                    contextual = key.startswith("M4")
                    side_groups = interactions if "side" in key else ()
                    x, vx, tx = [
                        visual_design(diff, [group_name(row) for row in rows],
                                      interactions if contextual else (), side_groups)
                        for diff, rows in zip(matrices[block], (training, validation, heldout))
                    ]
                    dim = matrices[block][0].shape[1]
                    penalty = np.r_[[1e-6], np.full(len(side_groups), model["ridge"]),
                                    np.full(dim, model["ridge"]),
                                    np.full(dim*len(interactions) if contextual else 0,
                                            model["ridge"] * (model["rho"] if contextual else 1))]
                result = fit_logistic(x, yy, penalty)
                if not result.accepted:
                    raise ValueError(f"Reproduction fit not numerically accepted: {key}")
                vl = np.asarray(vx @ result.coefficients).reshape(-1)
                tl = np.asarray(tx @ result.coefficients).reshape(-1)
            validation_predictions[key], test_predictions[key] = vl, tl
            validation_delta = abs(score(vy, vl)[0]-model["validation_loss"])
            original = saved_test[(population, key)]
            if [row["train_row_id"] for row in heldout] != [row["train_row_id"] for row in original]:
                raise ValueError(f"Reproduction test row alignment differs: {key}")
            # Only already-saved evaluation outcomes are used here, never raw withheld choices.
            ytest = np.asarray([row["outcome"] for row in original], dtype=np.float64)
            metrics = score(ytest, tl)
            metric_delta = max(abs(value-float(expected[(population, key)][name]))
                               for value, name in zip(metrics, ("response_log_loss", "brier", "accuracy")))
            probability_delta = float(np.max(np.abs(expit(tl)-np.asarray([row["probability"] for row in original]))))
            if max(validation_delta, metric_delta, probability_delta) > tolerance:
                raise ValueError(f"Statistical reproduction exceeds tolerance: {key}")
            records.append({
                "model": key, "validation_loss_abs_difference": validation_delta,
                "test_metric_max_abs_difference": metric_delta,
                "test_probability_max_abs_difference": probability_delta,
                "gradient_max": result.gradient_max if result is not None else None,
                "solver_attempts": list(result.attempts) if result is not None else [],
            })
            print(f"Reproduced selected fit: {key}", flush=True)
    common = saved_test[("common", "common:M0")]
    pair_ids = [row["pair_id"] for row in common]
    loss_by_model = {}
    for key in {item[name] for item in frozen["primary_contrasts"] for name in ("reference", "candidate")}:
        entries = saved_test[("common", key)]
        if [row["train_row_id"] for row in entries] != [row["train_row_id"] for row in common]:
            raise ValueError("Saved paired test predictions do not align.")
        loss_by_model[key] = np.asarray([row["loss"] for row in entries])
    samples, psamples = pair_bootstrap(pair_ids, loss_by_model,
                                       frozen["bootstrap"]["seed"], frozen["bootstrap"]["replicates"])
    with (root / "tables/primary_contrasts.csv").open(encoding="utf-8", newline="") as stream:
        contrasts = list(csv.DictReader(stream))
    for item in contrasts:
        low, high, _ = interval(samples[item["reference"]]-samples[item["candidate"]], .9875)
        plow, phigh, _ = interval(psamples[item["reference"]]-psamples[item["candidate"]], .9875)
        if max(abs(low-float(item["response_interval_low"])), abs(high-float(item["response_interval_high"])),
               abs(plow-float(item["pair_interval_low"])), abs(phigh-float(item["pair_interval_high"]))) > tolerance:
            raise ValueError("Saved paired bootstrap intervals do not reproduce.")
    verify_history(root)
    report = {
        "status": "passed", "fresh_process": True, "python": sys.version, "platform": platform.platform(),
        "tolerance": tolerance, "selected_models_reproduced": len(records), "records": records,
        "elapsed_seconds": time.monotonic()-started,
        "max_test_metric_abs_difference": max(row["test_metric_max_abs_difference"] for row in records),
        "max_test_probability_abs_difference": max(row["test_probability_max_abs_difference"] for row in records),
        "paired_bootstrap_reproduced": True, "unique_fitting_photo_preprocessing_recomputed": True,
        "source_sha256": _hash(root / "src/reproduce.py"),
        "model_freeze_sha256": _hash(root / "artifacts/model_freeze.json"),
        "raw_internal_test_outcomes_read": False, "frozen_artifacts_modified": False,
        "scope": "Refit selected models at frozen settings in memory; no retuning or new evaluation. "
                 "Test metrics are reconstruction checks using already-saved evaluation outcomes.",
        "not_checked": "This check is not a full raw-image-to-report run or a fresh model-selection search.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        report = reproduce_statistics(args.root.resolve())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Reproduction failed: {exc}", file=sys.stderr)
        return 2
    print(f"Fresh-process reconstruction passed for {report['selected_models_reproduced']} selected models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
