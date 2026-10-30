"""Outcome-independent selection of validation examples for model-logit contributions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .diagnostics import differences, load_data, write_csv
from .development import group_name
from .representation_diagnostics import _group_name


def grouped_contributions(
    differences_: np.ndarray, coefficients: np.ndarray, names: list[str], group_map: dict[str, list[str]],
) -> dict[str, float]:
    if differences_.shape != coefficients.shape or differences_.shape != (len(names),):
        raise ValueError("Feature contributions do not match coefficient dimensions.")
    reverse = {}
    for group, members in group_map.items():
        for member in members:
            if member in reverse:
                raise ValueError(f"Feature belongs to more than one contribution group: {member}")
            reverse[member] = group
    result: dict[str, float] = {}
    for name, value in zip(names, differences_ * coefficients):
        group = reverse.get(_group_name(name), "other")
        result[group] = result.get(group, 0.) + float(value)
    return result


def run_contributions(root: Path) -> int:
    output = root / "tables/feature_contributions.csv"
    if output.exists():
        raise FileExistsError("Feature contribution output already exists.")
    core = json.loads((root / "artifacts/stage8_core.json").read_text(encoding="utf-8"))
    if core["status"] != "passed_operational_control":
        raise ValueError("Fair-coin control has not passed its operational gate.")
    data = load_data(root)
    protocol = json.loads((root / "artifacts/stage8_contribution_protocol.json").read_text(encoding="utf-8"))
    examples = sorted(data.val, key=lambda row: row["train_row_id"])[:5]
    with (root / "tables/validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
        recorded = {(row["model"], row["train_row_id"]): float(row["logit"])
                    for row in csv.DictReader(stream)}
    records: list[dict[str, Any]] = []
    interactions = data.selection["interaction_contexts"]
    for block in ("H", "C", "Y"):
        spec = json.loads((root / "artifacts/preprocessing" / f"{block}.json").read_text(encoding="utf-8"))
        diffs = differences(examples, data.features[block], spec).toarray()
        names = spec["final_names"]
        for kind in ("M3", "M4"):
            key = f"{kind}-{block}"
            model = data.selection["selected"][key]
            theta = np.asarray(model["coefficients"], dtype=np.float64)
            for i, row in enumerate(examples):
                base = theta[1:1+len(names)]
                if kind == "M4" and not model["pooled"] and group_name(row) in interactions:
                    group_index = interactions.index(group_name(row))
                    start = 1 + len(names) * (1 + group_index)
                    base = base + theta[start:start+len(names)]
                if block == "C":
                    pieces = {"whole_embedding": float(diffs[i] @ base)}
                else:
                    pieces = grouped_contributions(diffs[i], base, names, data.protocol["ablation_groups"][block])
                summed = float(theta[0]) + sum(pieces.values())
                original = recorded[(key, row["train_row_id"])]
                if not np.isclose(summed, original, rtol=0, atol=1e-10):
                    raise ValueError(f"Contributions do not reconstruct {key}/{row['train_row_id']}.")
                for group, contribution in sorted(pieces.items()):
                    records.append({
                        "partition": "common_strict_validation", "model": key,
                        "train_row_id": row["train_row_id"], "pair_id": row["pair_id"],
                        "pet_care": group_name(row), "group": group, "contribution_logit": contribution,
                        "intercept": float(theta[0]), "reconstructed_logit": summed,
                        "interpretation": protocol["interpretation"],
                    })
    write_csv(output, records)
    return len(records)
