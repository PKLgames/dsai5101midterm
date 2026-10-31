"""Nested-pair learning curves with fit-only preprocessing and full retuning."""

from __future__ import annotations

import argparse
import csv
import json

from common import EXPERIMENT, HERE, digest, verify_inputs, write_json
from experiment import load_rows
from src.learning_curves import common_validation_indices, nested_pair_sets
from trainer import train_models


def run(seed: int) -> None:
    verify_inputs()
    if seed not in (20261030, 20261031):
        raise ValueError("Unexpected learning-curve seed.")
    fit, validation = load_rows("fit"), load_rows("validation")
    seeds, sizes = [20261030, 20261031], [.25, .5, .75, 1.]
    indices = common_validation_indices(fit, validation, seeds, sizes)
    if len(indices) < 100 or len({validation[i]["pair_id"] for i in indices}) < 30:
        sizes = [.5, .75, 1.]
        indices = common_validation_indices(fit, validation, seeds, sizes)
    val = [validation[i] for i in indices]
    if len(val) < 100 or len({r["pair_id"] for r in val}) < 30:
        raise ValueError("Inadequate common validation intersection after fallback.")
    old_status = json.loads((HERE.parent.parent / "artifacts" / "learning_curve_status.json").read_text(encoding="utf-8"))
    if (len(val), len({r["pair_id"] for r in val})) != (old_status["responses"], old_status["pairs"]):
        raise ValueError("Common intersection differs from original learning-curve support.")
    pair_sets = nested_pair_sets(fit, seed, sizes)
    for size in sizes:
        label = f"{seed}_{int(size*100):03d}"
        selected_path, attempts_path = HERE / f"curve_{label}_models.json", HERE / f"curve_{label}_attempts.csv"
        if selected_path.exists() and attempts_path.exists():
            saved = json.loads(selected_path.read_text(encoding="utf-8"))
            if (saved["seed"] != seed or saved["fraction"] != size or
                    saved["candidate_log_sha256"] != digest(attempts_path) or
                    saved["demographic_freeze_sha256"] != digest(EXPERIMENT / "freeze.json")):
                raise ValueError(f"Existing learning-curve result does not verify: {label}")
            print(f"Verified and retained completed learning-curve subset {label}.", flush=True)
            continue
        if selected_path.exists() or attempts_path.exists():
            raise FileExistsError(f"Incomplete learning-curve output must be preserved before retry: {label}")
        train = [row for row in fit if row["pair_id"] in pair_sets[size]]
        result = train_models(train, val, attempts_path)
        with attempts_path.open(encoding="utf-8", newline="") as stream:
            attempts = list(csv.DictReader(stream))
        if any(row["accepted"] != "True" for row in attempts):
            raise RuntimeError(f"At least one curve fit failed; investigate {attempts_path}.")
        result.update({"seed": seed, "fraction": size, "fitting_pairs": len(pair_sets[size]),
                       "validation_pairs": len({r["pair_id"] for r in val}),
                       "candidate_log_sha256": digest(attempts_path),
                       "demographic_freeze_sha256": digest(EXPERIMENT / "freeze.json"),
                       "test_outcomes_read": False, "post_test_development_diagnostic": True})
        write_json(selected_path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    run(parser.parse_args().seed)
