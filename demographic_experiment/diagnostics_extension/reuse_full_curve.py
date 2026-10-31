"""Reuse one full-size curve fit only after proving both seeds give identical inputs."""

from __future__ import annotations

import csv
import json
import shutil

from common import EXPERIMENT, HERE, digest, verify_inputs, write_json
from experiment import load_rows
from src.learning_curves import common_validation_indices, nested_pair_sets


def run() -> None:
    verify_inputs()
    source_models, source_attempts = HERE/"curve_20261031_100_models.json", HERE/"curve_20261031_100_attempts.csv"
    target_models, target_attempts = HERE/"curve_20261030_100_models.json", HERE/"curve_20261030_100_attempts.csv"
    if target_models.exists() or target_attempts.exists() or (HERE/"full_size_reuse_provenance.json").exists():
        raise FileExistsError("Full-size reuse output already exists.")
    fit, val = load_rows("fit"), load_rows("validation")
    first = nested_pair_sets(fit, 20261030, [1.])[1.]
    second = nested_pair_sets(fit, 20261031, [1.])[1.]
    first_ids = [r["train_row_id"] for r in fit if r["pair_id"] in first]
    second_ids = [r["train_row_id"] for r in fit if r["pair_id"] in second]
    if first != second or first_ids != second_ids or len(first_ids) != len(fit):
        raise ValueError("Full-size fitting sets or row order differ between seeds.")
    indices = common_validation_indices(fit, val, [20261030, 20261031], [.25, .5, .75, 1.])
    model = json.loads(source_models.read_text(encoding="utf-8"))
    if (model["seed"] != 20261031 or model["fraction"] != 1.0 or
            model["candidate_log_sha256"] != digest(source_attempts) or
            model["demographic_freeze_sha256"] != digest(EXPERIMENT/"freeze.json") or
            model["validation_responses"] != len(indices)):
        raise ValueError("Full-size source model or common validation identity does not verify.")
    with source_attempts.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 94 or any(row["accepted"] != "True" for row in rows):
        raise ValueError("Full-size source candidate log is incomplete.")
    shutil.copy2(source_attempts, target_attempts)
    if digest(target_attempts) != digest(source_attempts):
        raise ValueError("Reused candidate log copy differs.")
    model["seed"] = 20261030
    model["reused_from_seed"] = 20261031
    model["reuse_reason"] = "At full size both seeds include exactly the same fitting pairs and row order and use the same common validation rows."
    write_json(target_models, model)
    write_json(HERE/"full_size_reuse_provenance.json", {
        "status": "identical_full_size_inputs_verified", "source_seed": 20261031, "reported_seed": 20261030,
        "fitting_responses": len(first_ids), "fitting_pairs": len(first), "validation_responses": len(indices),
        "candidate_log_sha256": digest(source_attempts), "source_models_sha256": digest(source_models),
        "derived_models_sha256": digest(target_models),
        "interpretation": "One full-size fit reused for two identical seeds; not independent replications.",
        "reason": "Reduce memory pressure while preserving the exact full-size candidate procedure."})
    print("Verified identical full-size fitting rows/order and reused the completed full-size candidate results.")


if __name__ == "__main__":
    run()
