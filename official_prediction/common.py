"""Shared deployment-only helpers; no frozen evaluation files are changed."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from g3_experiment.prepare import digest, snapshot, write_csv, write_json  # noqa: E402
from src.freeze import validate_freeze  # noqa: E402

EXCLUDED = {".git", ".venv", ".cache", "official_prediction"}


def verify_experiment(name: str) -> dict[str, Any]:
    folder = ROOT / name
    freeze_path = folder / "freeze.json"
    if digest(freeze_path) != (folder / "freeze.sha256").read_text(encoding="ascii").strip():
        raise ValueError(f"{name} freeze seal mismatch.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for path, expected in freeze["experiment_sha256"].items():
        if digest(folder / path) != expected:
            raise ValueError(f"{name} frozen artifact changed: {path}")
    for path, expected in freeze["root_input_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"{name} frozen input changed: {path}")
    evaluation = json.loads((folder / "evaluation_provenance.json").read_text(encoding="utf-8"))
    if evaluation["freeze_sha256"] != digest(freeze_path):
        raise ValueError(f"{name} evaluation freeze mismatch.")
    if evaluation["test_access_record_sha256"] != digest(folder / "evaluation_started.json"):
        raise ValueError(f"{name} evaluation access record mismatch.")
    for path, expected in evaluation["output_sha256"].items():
        if digest(folder / path) != expected:
            raise ValueError(f"{name} evaluation output changed: {path}")
    return freeze


def verify_history() -> dict[str, Any]:
    g2 = validate_freeze(ROOT)
    evaluation = json.loads((ROOT / "artifacts" / "evaluation_provenance.json").read_text(encoding="utf-8"))
    if evaluation["freeze_sha256"] != digest(ROOT / "artifacts" / "model_freeze.json"):
        raise ValueError("G2 evaluation provenance freeze mismatch.")
    for name, expected in evaluation.get("output_sha256", {}).items():
        if digest(ROOT.joinpath(*name.split("/"))) != expected:
            raise ValueError(f"G2 evaluation output changed: {name}")
    g3, g4 = verify_experiment("g3_experiment"), verify_experiment("g4_experiment")
    return {"g2": g2, "g3": g3, "g4": g4}


def load_labeled(expected_hash: str) -> list[dict[str, Any]]:
    source = ROOT / "dataset" / "train.csv"
    if digest(source) != expected_hash:
        raise ValueError("Labeled source differs from the pre-deployment audit.")
    fields = ["train_row_id", "a_cat", "b_cat", "a_image", "b_image", "comparison_type", "pet_care"]
    structure = pq.read_table(ROOT / "artifacts" / "rows_canonical.parquet", columns=fields).to_pylist()
    index = {row["train_row_id"]: row for row in structure}
    if len(index) != len(structure):
        raise ValueError("Duplicate canonical training IDs.")
    rows = []
    seen = set()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(fields + ["choice"]) <= set(reader.fieldnames):
            raise ValueError("Labeled source is missing required fields.")
        for raw in reader:
            key = raw["train_row_id"]
            if key in seen or key not in index or raw["choice"] not in {"A", "B"}:
                raise ValueError(f"Unknown, duplicate or invalid labeled response: {key}")
            canonical = index[key]
            if any(raw[field] != canonical[field] for field in fields):
                raise ValueError(f"Labeled response differs from the audited structure: {key}")
            seen.add(key)
            rows.append({**canonical, "y_displayed": int(raw["choice"] == "A")})
    if seen != set(index) or digest(source) != expected_hash:
        raise ValueError("Labeled source is incomplete or changed while reading.")
    return rows


def classify_route(row: dict[str, Any], photo_components: dict[str, int],
                   cat_components: dict[str, int], usable: set[str]) -> tuple[str, str]:
    a, b = row["a_image"], row["b_image"]
    ca, cb = row["a_cat"], row["b_cat"]
    known_cats = ca in cat_components and cb in cat_components and cat_components[ca] == cat_components[cb]
    known_photos = a in photo_components and b in photo_components and photo_components[a] == photo_components[b]
    if known_photos and known_cats:
        return "known_photo_m1", "supported_known_photos"
    if a in usable and b in usable:
        return "shared_visual", "new_or_structurally_unsupported_photos"
    if known_cats:
        return "cat_fallback_m1", "visual_unavailable_known_cats"
    return "constant_fallback_m0", "visual_or_identity_unsupported"


def probabilities_to_choices(probabilities: np.ndarray) -> list[str]:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Predictions must be finite probabilities in [0,1].")
    return ["A" if value >= .5 else "B" for value in p]
