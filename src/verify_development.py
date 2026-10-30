"""Verify persisted Stage 7 models against their validation predictions."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit

from .development import feature_differences, group_name, identity_matrix, load_development, score, visual_design
from .preprocessing import fit_preprocessing, transform_photos


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def verify(root: Path) -> dict[str, Any]:
    path = root / "artifacts/stage7_verification.json"
    if path.exists():
        raise FileExistsError("Stage 7 verification already exists.")
    selection = json.loads((root / "artifacts/model_selection.json").read_text(encoding="utf-8"))
    for name, digest in selection["input_sha256"].items():
        if _hash(root.joinpath(*name.split("/"))) != digest:
            raise ValueError(f"Stage 7 source or input hash mismatch: {name}")
    fit, val = load_development(root)
    common_fit = [row for row in fit if row["common_visual_fit"]]
    populations = {
        "full": [row for row in val if row["full_strict_g2"]],
        "common": [row for row in val if row["common_strict_g2"]],
    }
    image_ids = sorted({row[f"{side}_image"] for row in common_fit + populations["common"]
                        for side in ("a", "b")})
    fitting_images = {row[f"{side}_image"] for row in common_fit for side in ("a", "b")}
    groups = [group_name(row) for row in populations["common"]]
    interactions = tuple(selection["interaction_contexts"])
    matrices = {}
    for block in ("H", "C", "Y"):
        all_rows = pq.read_table(root / "features" / f"{block}.parquet").to_pylist()
        spec = json.loads((root / "artifacts/preprocessing" / f"{block}.json").read_text(encoding="utf-8"))
        if spec != fit_preprocessing(block, all_rows, fitting_images):
            raise ValueError(f"Persisted {block} preprocessing is not reproducible from fitting photos.")
        by_id = {row["image_id"]: row for row in all_rows}
        vectors = transform_photos(spec, [by_id[image] for image in image_ids])
        matrices[block] = feature_differences(populations["common"], image_ids, vectors)
    with (root / "tables/validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
        saved = list(csv.DictReader(stream))
    by_model: dict[str, list[dict[str, str]]] = {}
    for row in saved:
        by_model.setdefault(row["model"], []).append(row)
    selected = selection["selected"]
    expected_keys = {key for key, value in selected.items()
                     if isinstance(value, dict) and "coefficients" in value and key not in {"M3*", "M4*"}}
    if set(by_model) != expected_keys:
        raise ValueError("Selected models and saved validation-prediction groups differ.")
    for key in sorted(expected_keys):
        model = selected[key]
        model_path = root / "models" / (key.replace(":", "_") + ".json")
        if json.loads(model_path.read_text(encoding="utf-8")) != model:
            raise ValueError(f"Stored model differs from selection record: {key}")
        population = "full" if key.startswith("full:") else "common"
        rows = populations[population]
        saved_rows = by_model[key]
        if [row["train_row_id"] for row in rows] != [row["train_row_id"] for row in saved_rows]:
            raise ValueError(f"Validation ID order mismatch: {key}")
        if key.endswith("M0"):
            logits = np.full(len(rows), model["coefficients"][0], dtype=np.float64)
        elif key.endswith("M1") or key.endswith("M2") or key.endswith("M2-no-position"):
            entity = "cat" if key.endswith("M1") else "image"
            design = identity_matrix(rows, model["identity_keys"], entity, not key.endswith("M2-no-position"))
            logits = np.asarray(design @ np.asarray(model["coefficients"])).reshape(-1)
        else:
            base_key = model["pooled_from"].removeprefix("common:") if model["pooled"] else key
            base = selected[base_key]
            block = selection["selected_representation"] if "side" in key else key[-1]
            contextual = key.startswith("M4") and not model["pooled"]
            side = "side" in key
            design = visual_design(
                matrices[block], groups, interactions if contextual else (),
                interactions if side else (),
            )
            logits = np.asarray(design @ np.asarray(base["coefficients"])).reshape(-1)
        stored_logits = np.array([float(row["logit"]) for row in saved_rows])
        stored_probs = np.array([float(row["probability"]) for row in saved_rows])
        if not np.allclose(logits, stored_logits, rtol=0, atol=1e-12):
            raise ValueError(f"Reconstructed validation logits do not match: {key}")
        if not np.allclose(expit(logits), stored_probs, rtol=0, atol=1e-12):
            raise ValueError(f"Reconstructed validation probabilities do not match: {key}")
        labels = np.array([row["y_displayed"] for row in rows], dtype=np.float64)
        if abs(score(labels, logits)[0] - model["validation_loss"]) > 1e-12:
            raise ValueError(f"Validation loss does not match reconstructed predictions: {key}")
    output_paths = [
        root / "artifacts/model_selection.json", root / "tables/all_fit_attempts.csv",
        root / "tables/validation_results.csv", root / "tables/validation_predictions.csv",
        *sorted((root / "models").glob("*.json")), *sorted((root / "artifacts/preprocessing").glob("*.json")),
    ]
    report = {
        "verified_selected_models": len(expected_keys),
        "validation_responses": len(populations["common"]),
        "preprocessing_recomputed_from_unique_fitting_photos": True,
        "probability_reconstruction_tolerance": 1e-12,
        "internal_test_outcomes_read": False,
        "output_sha256": {file.relative_to(root).as_posix(): _hash(file) for file in output_paths},
        "verification_source_sha256": _hash(root / "src/verify_development.py"),
    }
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
