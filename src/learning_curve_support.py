"""Persist fitting-only preprocessing and context support for learning curves."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .development import group_name
from .diagnostics import load_data, write_csv
from .estimator import context_mapping
from .learning_curves import nested_pair_sets
from .preprocessing import fit_preprocessing


def run_curve_support(root: Path) -> dict[str, Any]:
    tables = root / "tables/learning_curve_context_support.csv"
    specs_path = root / "artifacts/learning_curve_preprocessing.json"
    if tables.exists() or specs_path.exists():
        raise FileExistsError("Learning-curve support artifacts already exist.")
    status = json.loads((root / "artifacts/learning_curve_status.json").read_text(encoding="utf-8"))
    data = load_data(root)
    if status["status"] not in {"completed", "omitted_inadequate_common_validation"}:
        raise ValueError("Learning-curve status is invalid.")
    records: list[dict[str, Any]] = []
    specs: dict[str, Any] = {}
    if status["status"] == "completed":
        block = data.selection["selected_representation"]
        full_groups = sorted({group_name(row) for row in data.fit})
        for seed in data.protocol["learning_curve_seeds"]:
            pair_sets = nested_pair_sets(data.fit, seed, status["sizes"])
            for size in status["sizes"]:
                fit = [row for row in data.fit if row["pair_id"] in pair_sets[size]]
                groups = [group_name(row) for row in fit]
                reference, interactions = context_mapping(groups, [row["pair_id"] for row in fit])
                counts = Counter(groups)
                pairs: dict[str, set[str]] = defaultdict(set)
                for row in fit:
                    pairs[group_name(row)].add(row["pair_id"])
                images = {row[f"{side}_image"] for row in fit for side in ("a", "b")}
                specs[f"{seed}:{size}"] = fit_preprocessing(block, data.features[block], images)
                for group in full_groups:
                    eligible = counts[group] >= 100 and len(pairs[group]) >= 30
                    records.append({
                        "seed": seed, "fraction": size, "pet_care": group,
                        "responses": counts[group], "distinct_pairs": len(pairs[group]),
                        "eligible": eligible, "reference": group == reference,
                        "interaction": group in interactions,
                        "behavior": "reference_rule" if group == reference else
                                    "estimated_deviation" if group in interactions else "reference_fallback",
                    })
    fields = ["seed", "fraction", "pet_care", "responses", "distinct_pairs", "eligible",
              "reference", "interaction", "behavior"]
    write_csv(tables, records, fields)
    specs_path.write_text(json.dumps({
        "status": status["status"], "representation": data.selection["selected_representation"],
        "fitting_only_preprocessing": specs,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": status["status"], "context_rows": len(records), "preprocessing_specs": len(specs)}
