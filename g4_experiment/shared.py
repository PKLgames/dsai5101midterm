"""Shared helpers for the quarantined retrospective G4 experiment."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from g3_experiment.prepare import digest, snapshot, write_csv, write_json  # noqa: E402
from src.development import feature_differences, group_name, visual_design  # noqa: E402
from src.preprocessing import transform_photos  # noqa: E402


def load_structure(allocation: str) -> list[dict[str, Any]]:
    with (HERE / "row_allocation.csv").open(encoding="utf-8", newline="") as stream:
        selected = {}
        for row in csv.DictReader(stream):
            if row["allocation"] == allocation:
                if row["train_row_id"] in selected:
                    raise ValueError("Duplicate G4 allocation ID.")
                selected[row["train_row_id"]] = row
    fields = ["train_row_id", "pair_id", "a_image", "b_image", "a_cat", "b_cat",
              "comparison_type", "pet_care", "partition"]
    all_rows = pq.read_table(ROOT / "artifacts" / "rows_canonical.parquet", columns=fields).to_pylist()
    result = []
    for row in all_rows:
        match = selected.get(row["train_row_id"])
        if match is not None:
            if row["pair_id"] != match["pair_id"]:
                raise ValueError("G4 pair allocation disagrees with structural data.")
            row["held_out_cat"] = match["held_out_cat"]
            result.append(row)
    if not result or len(result) != len(selected):
        raise ValueError(f"G4 allocation is empty, duplicate or incomplete: {allocation}")
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


def identity_design(rows: list[dict[str, Any]], cats: list[str], cold_start: bool) -> sparse.csr_matrix:
    index = {cat: i for i, cat in enumerate(cats)}
    if len(index) != len(cats):
        raise ValueError("Fitting cat IDs must be unique.")
    rr, cc, values = [], [], []
    for i, row in enumerate(rows):
        unknown = []
        for side, sign in (("a", 1.), ("b", -1.)):
            cat = row[f"{side}_cat"]
            if cat in index:
                rr.append(i)
                cc.append(index[cat])
                values.append(sign)
            else:
                unknown.append(cat)
        if cold_start:
            if len(unknown) != 1 or unknown[0] != row["held_out_cat"]:
                raise ValueError("G4 identity prediction requires exactly the reserved cat to be unseen.")
        elif unknown:
            raise ValueError("Unknown cat in fitting identity design.")
    block = sparse.coo_matrix((values, (rr, cc)), shape=(len(rows), len(cats))).tocsr()
    block.eliminate_zeros()
    return sparse.hstack([np.ones((len(rows), 1)), block], format="csr")


def differences(block: str, spec: dict[str, Any], rows: list[dict[str, Any]]) -> sparse.csr_matrix:
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
    interactions = tuple(models["context"]["interactions"])
    selected = models["selected"]
    matrices = {block: differences(block, models["preprocessing"][block], rows) for block in ("H", "C", "Y")}
    result = {"M0": np.full(len(rows), selected["M0"]["coefficients"][0], dtype=np.float64)}
    identity = identity_design(rows, models["cat_keys"], cold_start=True)
    result["M1-zero"] = np.asarray(identity @ np.asarray(selected["M1-zero"]["coefficients"])).reshape(-1)
    for block in ("H", "C", "Y"):
        m3, m4 = "M3-" + block, "M4-" + block
        x3 = visual_design(matrices[block], groups, ())
        result[m3] = np.asarray(x3 @ np.asarray(selected[m3]["coefficients"])).reshape(-1)
        if selected[m4]["pooled"]:
            result[m4] = result[m3].copy()
        else:
            x4 = visual_design(matrices[block], groups, interactions)
            result[m4] = np.asarray(x4 @ np.asarray(selected[m4]["coefficients"])).reshape(-1)
    for name, logits in result.items():
        if logits.shape != (len(rows),) or not np.isfinite(logits).all():
            raise ValueError(f"Missing or nonfinite predictions for {name}.")
    return result
