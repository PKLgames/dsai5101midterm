"""Reproduce the original all-demographic development penalty paths."""

from __future__ import annotations

import csv
import json

from common import EXPERIMENT, HERE, digest, verify_inputs, write_csv, write_json
from experiment import load_rows
from trainer import train_models


def key(row: dict) -> tuple:
    return (row["model"], float(row["ridge"]), float(row["rho"]), str(row["pooled"]))


def run() -> None:
    verify_inputs()
    attempts = HERE / "penalty_refit_attempts.csv"
    output = HERE / "penalty_path_refits.csv"
    metadata = HERE / "penalty_path_provenance.json"
    if any(p.exists() for p in (attempts, output, metadata)):
        raise FileExistsError("Penalty-path refit already started.")
    fit, val = load_rows("fit"), load_rows("validation")
    models = train_models(fit, val, attempts)
    with attempts.open(encoding="utf-8", newline="") as stream:
        refit = list(csv.DictReader(stream))
    with (EXPERIMENT / "all_fit_attempts.csv").open(encoding="utf-8", newline="") as stream:
        original = list(csv.DictReader(stream))
    source = {key(row): row for row in original}
    if len(source) != len(original) or len(refit) != len(original):
        raise ValueError("Penalty-path candidate sets differ.")
    records = []
    for row in refit:
        match = source.get(key(row))
        if match is None or row["accepted"] != "True" or match["accepted"] != "True":
            raise ValueError(f"Penalty-path fit missing or failed: {key(row)}")
        delta = float(row["validation_loss"]) - float(match["validation_loss"])
        if abs(delta) > 1e-6:
            raise ValueError(f"Penalty-path validation reproduction failed: {key(row)} {delta}")
        records.append({"model": row["model"], "ridge": row["ridge"], "rho": row["rho"],
                        "pooled": row["pooled"], "training_loss": row["training_loss"],
                        "validation_loss": row["validation_loss"], "coefficient_norm": row["coefficient_norm"],
                        "gradient_max": row["gradient_max"], "validation_reproduction_delta": delta,
                        "original_validation_loss": match["validation_loss"]})
    write_csv(output, records, list(records[0]))
    write_json(metadata, {"status": "all_candidates_reproduced", "candidate_records": len(records),
                          "numerical_refits": sum(row["pooled"] == "False" for row in records),
                          "exact_fallback_copies": sum(row["pooled"] == "True" for row in records),
                          "maximum_abs_validation_difference": max(abs(r["validation_reproduction_delta"]) for r in records),
                          "selected_model_names": list(models["selected"]),
                          "source_attempts_sha256": digest(EXPERIMENT / "all_fit_attempts.csv"),
                          "refit_attempts_sha256": digest(attempts), "output_sha256": digest(output),
                          "test_outcomes_read": False, "post_test_development_diagnostic": True})
    print(f"Reproduced {len(records)} penalty candidates.")


if __name__ == "__main__":
    run()
