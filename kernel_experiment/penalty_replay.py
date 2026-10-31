"""Reproduce every original kernel G2 penalty candidate from zero initialization."""

import csv
import json

import numpy as np
from threadpoolctl import threadpool_limits

from kernel import designs, fit_map, load_features
from study import HERE, GRID, WIDTHS, digest, load_rows, load_frozen, score, write_csv, write_json
from src.estimator import fit_logistic


def run():
    if (HERE/"penalty_refits.csv").exists() or (HERE/"penalty_provenance.json").exists():
        raise FileExistsError("Penalty refit output exists.")
    load_frozen("g2")
    with (HERE/"g2_attempts.csv").open(encoding="utf-8", newline="") as stream:
        original = {(float(r["width_multiplier"]), float(r["ridge"])): r for r in csv.DictReader(stream)}
    train, val = load_rows("fit"), load_rows("validation")
    y = np.asarray([r["y_displayed"] for r in train])
    vy = np.asarray([r["y_displayed"] for r in val])
    photos = {r[f"{side}_image"] for r in train for side in ("a", "b")}
    features, rows = load_features(), []
    with threadpool_limits(limits=1):
        for width in WIDTHS:
            mapping = fit_map(features, photos, width)
            x, vx = designs(train, val, features, mapping)
            for ridge in GRID:
                fit = fit_logistic(x, y, np.r_[[1e-6], np.full(x.shape[1]-1, ridge)])
                if not fit.accepted:
                    raise ValueError(f"Kernel penalty refit failed: {width}/{ridge}: {fit.attempts}")
                loss = score(vy, np.asarray(vx @ fit.coefficients).ravel())[0]
                source = original[(width, ridge)]
                delta = loss-float(source["validation_loss"])
                if abs(delta) > 1e-6:
                    raise ValueError(f"Kernel penalty validation mismatch: {width}/{ridge}: {delta}")
                rows.append({"width_multiplier": width, "ridge": ridge, "accepted": True,
                             "training_loss": score(y, np.asarray(x @ fit.coefficients).ravel())[0],
                             "validation_loss": loss, "coefficient_norm": float(np.linalg.norm(fit.coefficients[1:])),
                             "gradient_max": fit.gradient_max, "validation_reproduction_delta": delta})
    write_csv(HERE/"penalty_refits.csv", rows, list(rows[0]))
    write_json(HERE/"penalty_provenance.json", {"status": "all_candidates_reproduced", "fits": len(rows),
               "maximum_validation_difference": max(abs(r["validation_reproduction_delta"]) for r in rows),
               "original_attempts_sha256": digest(HERE/"g2_attempts.csv"),
               "output_sha256": digest(HERE/"penalty_refits.csv"), "test_outcomes_read": False,
               "interpretation": "Post-test motivated development reconstruction; not independent."})
    print(json.dumps({"fits": len(rows), "max_delta": max(abs(r["validation_reproduction_delta"]) for r in rows)}))


if __name__ == "__main__":
    run()
