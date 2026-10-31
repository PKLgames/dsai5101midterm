"""Matched within-cat G3 comparisons using saved held-out-photo predictions."""

import json

import numpy as np
import pyarrow.parquet as pq

from study import HERE, ROOT, digest, source_rows, write_csv, write_json
from src.evaluate import pair_bootstrap


def run():
    if any((HERE/name).exists() for name in ("g3_within_results.csv", "g3_within_contrasts.csv", "g3_within_provenance.json")):
        raise FileExistsError("Within-cat G3 outputs exist.")
    source = json.loads((HERE/"g3_evaluation_provenance.json").read_text(encoding="utf-8"))
    if digest(HERE/"g3_predictions.parquet") != source["outputs_sha256"]["g3_predictions.parquet"]:
        raise ValueError("G3 prediction provenance mismatch.")
    structure = source_rows("G3", "test", False)
    if any((r["a_cat"] == r["b_cat"]) != (r["comparison_type"] == "within_cat") for r in structure):
        raise ValueError("G3 comparison type does not match cat IDs.")
    ids = {r["train_row_id"] for r in structure if r["comparison_type"] == "within_cat"}
    predictions = pq.read_table(HERE/"g3_predictions.parquet").to_pylist()
    names = ("kernel_M3-C", "original:M0", "original:M1", "original:M3-C", "original:M4-C")
    models = {name: [r for r in predictions if r["model"] == name and r["train_row_id"] in ids] for name in names}
    reference = [(r["train_row_id"], r["pair_id"], r["cluster"], r["outcome"]) for r in models[names[0]]]
    if not reference or any([(r["train_row_id"], r["pair_id"], r["cluster"], r["outcome"]) for r in models[name]] != reference for name in names):
        raise ValueError("Within-cat model populations do not align.")
    clusters = [r[2] for r in reference]
    losses = {name: np.asarray([r["loss"] for r in models[name]]) for name in names}
    samples, _ = pair_bootstrap(clusters, losses, 20261101, 5000)
    results = []
    for name in names:
        rows = models[name]
        p = np.asarray([r["probability"] for r in rows])
        y = np.asarray([r["outcome"] for r in rows])
        low, high = np.quantile(samples[name], [.025, .975])
        results.append({"model": name, "responses": len(rows), "pairs": len({r["pair_id"] for r in rows}),
                        "held_out_photos": len(set(clusters)), "log_loss": float(losses[name].mean()),
                        "ci_low": float(low), "ci_high": float(high), "brier": float(np.mean((p-y)**2)),
                        "accuracy": float(np.mean((p >= .5) == y))})
    contrasts = []
    for reference_name in ("original:M0", "original:M1", "original:M3-C", "original:M4-C"):
        diff = samples[reference_name]-samples["kernel_M3-C"]
        low, high = np.quantile(diff, [.025,.975])
        contrasts.append({"reference": reference_name, "candidate": "kernel_M3-C",
                          "difference": float(np.mean(losses[reference_name]-losses["kernel_M3-C"])),
                          "ci_low": float(low), "ci_high": float(high), "bootstrap_se": float(np.std(diff, ddof=1))})
    write_csv(HERE/"g3_within_results.csv", results, list(results[0]))
    write_csv(HERE/"g3_within_contrasts.csv", contrasts, list(contrasts[0]))
    write_json(HERE/"g3_within_provenance.json", {"status": "completed_post_test_exploratory",
               "input_sha256": digest(HERE/"g3_predictions.parquet"),
               "outputs_sha256": {name: digest(HERE/name) for name in ("g3_within_results.csv", "g3_within_contrasts.csv")},
               "responses": len(reference), "pairs": len({r[1] for r in reference}), "held_out_photos": len(set(clusters)),
               "bootstrap_replicates": 5000, "seed": 20261101, "raw_test_outcomes_read": False,
               "limitations": "Post-test pointwise, unadjusted, held-out-photo bootstrap; not independent."})
    print(json.dumps({"results": results, "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    run()
