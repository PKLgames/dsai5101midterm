"""Verify additional matched baselines, slices and penalty reproduction."""

import csv
import json

import numpy as np
import pyarrow.parquet as pq

from study import HERE, ROOT, digest, write_json


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def run():
    if (HERE/"additional_verification.json").exists():
        raise FileExistsError("Additional verification exists.")
    proof = json.loads((HERE/"penalty_provenance.json").read_text(encoding="utf-8"))
    if proof["status"] != "all_candidates_reproduced" or proof["output_sha256"] != digest(HERE/"penalty_refits.csv"):
        raise ValueError("Penalty refit provenance mismatch.")
    refits = read(HERE/"penalty_refits.csv")
    if len(refits) != 21 or any(r["accepted"] != "True" or abs(float(r["validation_reproduction_delta"])) > 1e-6 for r in refits):
        raise ValueError("Penalty refit grid incomplete.")
    results = read(HERE/"additional_paired_contrasts.csv")
    for label in ("g2", "g3", "g4"):
        rows = pq.read_table(HERE/f"{label}_predictions.parquet", filters=[("model", "=", "kernel_M3-C")]).to_pylist()
        source = (ROOT/"demographic_experiment"/"test_predictions.parquet" if label == "g2" else
                  ROOT/"demographic_experiment"/"diagnostics_extension"/f"{label}_predictions.parquet")
        comp = pq.read_table(source, filters=[("model", "=", "C+D")]).to_pylist()
        index = {r["train_row_id"]: r for r in comp}
        if len(index) != len(comp) or set(index) != {r["train_row_id"] for r in rows}:
            raise ValueError("Additional comparator rows differ.")
        clusters = [r["cluster"] for r in rows]
        if any(index[r["train_row_id"]]["outcome"] != r["outcome"] for r in rows):
            raise ValueError("Additional comparator outcomes differ.")
        a = np.asarray([index[r["train_row_id"]]["loss"] for r in rows])
        b = np.asarray([r["loss"] for r in rows])
        unique = sorted(set(clusters))
        lookup = {name: i for i, name in enumerate(unique)}
        codes = np.asarray([lookup[name] for name in clusters])
        counts = np.bincount(codes)
        sums = np.bincount(codes, weights=a-b)
        rng = np.random.default_rng(20261101)
        samples = []
        for _ in range(5000):
            weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
            samples.append(float(sums @ weights / (counts @ weights)))
        row = next(r for r in results if r["target"] == label)
        np.testing.assert_allclose([np.mean(a-b), *np.quantile(samples, [.025,.975])],
                                   [float(row["difference"]), float(row["ci_low"]), float(row["ci_high"])],
                                   rtol=0, atol=1e-12)
    old = read(ROOT/"tables"/"subgroup_results.csv")
    base = {(r["dimension"], r["group"]): r for r in old if r["population"] == "common" and r["model_id"] == "M3-C"}
    slices = read(HERE/"g2_slices.csv")
    for r in slices:
        original = base[(r["dimension"], r["group"])]
        if (r["responses"], r["pairs"], r["support"]) != (original["responses"], original["pairs"], original["support_status"]):
            raise ValueError("Kernel subgroup support does not match original M3-C.")
    vbase = {(r["dimension"], r["group"]): r for r in read(ROOT/"tables"/"diagnostic_results.csv") if r["model"] == "M3-C"}
    for row in read(HERE/"validation_slices.csv"):
        expected = vbase[(row["dimension"], row["group"])]
        if (row["responses"], row["pairs"], row["support"]) != (expected["responses"], expected["pairs"], expected["support_status"]):
            raise ValueError("Kernel validation subgroup support mismatch.")
    figure = json.loads((HERE/"abstention_figure_provenance.json").read_text(encoding="utf-8"))
    if figure["input_sha256"] != digest(HERE/"g2_abstention_utility.csv") or figure["output_sha256"] != digest(HERE/"presentation_kernel_abstention.pdf"):
        raise ValueError("Abstention figure provenance mismatch.")
    report = {"status": "passed", "penalty_candidates_replayed": len(refits),
              "extra_comparators": 3, "bootstrap_replicates_per_target": 5000,
              "test_support_groups_matched": len(slices),
              "validation_support_groups_matched": len(read(HERE/"validation_slices.csv")),
              "raw_test_outcomes_read": False, "limitations": "Previously exposed outcomes remain exploratory."}
    write_json(HERE/"additional_verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
