"""Shared, isolated helpers for the retrospective G3 experiment."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from prepare import digest, write_csv, write_json  # noqa: E402
from src.development import feature_differences, group_name, identity_matrix, visual_design  # noqa: E402
from src.preprocessing import transform_photos  # noqa: E402


def load_structure(allocation: str) -> list[dict[str, Any]]:
    with (EXPERIMENT / "row_allocation.csv").open(encoding="utf-8", newline="") as stream:
        selected = {row["train_row_id"]: row for row in csv.DictReader(stream)
                    if row["allocation"] == allocation}
    fields = ["train_row_id", "pair_id", "a_image", "b_image", "a_cat", "b_cat",
              "comparison_type", "pet_care", "partition"]
    all_rows = pq.read_table(ROOT / "artifacts" / "rows_canonical.parquet", columns=fields).to_pylist()
    result = []
    for row in all_rows:
        allocation_row = selected.get(row["train_row_id"])
        if allocation_row is not None:
            if row["pair_id"] != allocation_row["pair_id"]:
                raise ValueError("Pair allocation disagrees with structural data.")
            row["held_out_photo"] = allocation_row["held_out_photo"]
            result.append(row)
    if not result or len(result) != len(selected):
        raise ValueError(f"Missing, duplicate, or empty allocation: {allocation}")
    return result


def load_outcomes(rows: list[dict[str, Any]], expected_hash: str) -> np.ndarray:
    source = ROOT / "dataset" / "train.csv"
    if digest(source) != expected_hash:
        raise ValueError("Raw outcome source does not match the pre-experiment snapshot.")
    wanted = {row["train_row_id"] for row in rows}
    if len(wanted) != len(rows):
        raise ValueError("Duplicate requested outcome ID.")
    outcomes = {}
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["train_row_id"]
            if key in wanted:
                if key in outcomes or row["choice"] not in {"A", "B"}:
                    raise ValueError(f"Invalid or duplicate requested outcome: {key}")
                outcomes[key] = int(row["choice"] == "A")
    if set(outcomes) != wanted or digest(source) != expected_hash:
        raise ValueError("Requested outcomes are incomplete or source changed while reading.")
    return np.asarray([outcomes[row["train_row_id"]] for row in rows], dtype=np.float64)


def differences(block: str, spec: dict[str, Any], rows: list[dict[str, Any]]) -> Any:
    records = pq.read_table(ROOT / "features" / f"{block}.parquet").to_pylist()
    index = {row["image_id"]: row for row in records}
    ids = sorted({row[f"{side}_image"] for row in rows for side in ("a", "b")})
    if len(index) != len(records) or not set(ids) <= set(index):
        raise ValueError(f"Missing or duplicate {block} feature IDs.")
    if any(index[image]["extraction_status"] != "ok" for image in ids):
        raise ValueError(f"Unusable {block} features in a matched population.")
    return feature_differences(rows, ids, transform_photos(spec, [index[image] for image in ids]))


def predict(models: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    groups = [group_name(row) for row in rows]
    context = tuple(models["context"]["interactions"])
    selected = models["selected"]
    matrices = {block: differences(block, models["preprocessing"][block], rows)
                for block in ("H", "C", "Y")}
    result = {"M0": np.full(len(rows), selected["M0"]["coefficients"][0], dtype=np.float64)}
    x = identity_matrix(rows, models["cat_keys"], "cat")
    result["M1"] = np.asarray(x @ np.asarray(selected["M1"]["coefficients"])).reshape(-1)
    for block in ("H", "C", "Y"):
        m3, m4 = "M3-" + block, "M4-" + block
        x3 = visual_design(matrices[block], groups, ())
        result[m3] = np.asarray(x3 @ np.asarray(selected[m3]["coefficients"])).reshape(-1)
        if selected[m4]["pooled"]:
            result[m4] = result[m3].copy()
        else:
            x4 = visual_design(matrices[block], groups, context)
            result[m4] = np.asarray(x4 @ np.asarray(selected[m4]["coefficients"])).reshape(-1)
    for name, logits in result.items():
        if logits.shape != (len(rows),) or not np.isfinite(logits).all():
            raise ValueError(f"Missing or nonfinite predictions for {name}.")
    return result


def cluster_bootstrap(
    clusters: list[str], losses: dict[str, np.ndarray], replicates: int, seed: int,
) -> dict[str, np.ndarray]:
    unique = sorted(set(clusters))
    if not unique or any(not item for item in clusters) or replicates < 2:
        raise ValueError("Bootstrap requires nonempty held-out-photo clusters and at least two draws.")
    mapping = {key: i for i, key in enumerate(unique)}
    codes = np.asarray([mapping[key] for key in clusters], dtype=int)
    counts = np.bincount(codes, minlength=len(unique))
    names = sorted(losses)
    totals = []
    for name in names:
        values = np.asarray(losses[name], dtype=np.float64)
        if values.shape != codes.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid bootstrap losses: {name}")
        totals.append(np.bincount(codes, weights=values, minlength=len(unique)))
    totals_array = np.stack(totals)
    samples = np.empty((replicates, len(names)))
    rng = np.random.default_rng(seed)
    for i in range(replicates):
        multiplicity = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        samples[i] = totals_array @ multiplicity / (counts @ multiplicity)
    return {name: samples[:, index] for index, name in enumerate(names)}
