"""Pair-weighted G2 and matched side comparisons using saved predictions."""

import csv
import json

import numpy as np
import pyarrow.parquet as pq

from study import HERE, ROOT, digest, write_csv, write_json
from src.evaluate import pair_bootstrap


def run() -> None:
    if any((HERE/name).exists() for name in ("additional_paired_contrasts.csv", "g2_pair_weighted.csv", "paired_provenance.json")):
        raise FileExistsError("Additional paired evidence already exists.")
    records, pair_results = [], []
    for label in ("g2", "g3", "g4"):
        proof = json.loads((HERE/f"{label}_evaluation_provenance.json").read_text(encoding="utf-8"))
        if digest(HERE/f"{label}_predictions.parquet") != proof["outputs_sha256"][f"{label}_predictions.parquet"]:
            raise ValueError(f"{label} kernel prediction source changed.")
        if label == "g2":
            source = ROOT/"demographic_experiment"/"test_predictions.parquet"
            provenance = json.loads((ROOT/"demographic_experiment"/"evaluation_provenance.json").read_text(encoding="utf-8"))
            expected = provenance["outputs_sha256"]["test_predictions.parquet"]
        else:
            base = ROOT/"demographic_experiment"/"diagnostics_extension"
            source = base/f"{label}_predictions.parquet"
            provenance = json.loads((base/f"{label}_provenance.json").read_text(encoding="utf-8"))
            expected = provenance["outputs_sha256"][f"{label}_predictions.parquet"]
        if digest(source) != expected:
            raise ValueError(f"{label} demographic source changed.")
        kernel = pq.read_table(HERE/f"{label}_predictions.parquet").to_pylist()
        demographic = pq.read_table(source, filters=[("model", "=", "C+D")]).to_pylist()
        kernel_rows = [r for r in kernel if r["model"] == "kernel_M3-C"]
        index = {r["train_row_id"]: r for r in demographic}
        if len(index) != len(demographic) or set(index) != {r["train_row_id"] for r in kernel_rows}:
            raise ValueError(f"{label} demographic rows differ.")
        if any(index[r["train_row_id"]]["pair_id"] != r["pair_id"] or
               index[r["train_row_id"]]["outcome"] != r["outcome"] for r in kernel_rows):
            raise ValueError(f"{label} demographic outcomes or pairs differ.")
        losses = {"kernel_M3-C": np.asarray([r["loss"] for r in kernel_rows]),
                  "C+D": np.asarray([index[r["train_row_id"]]["loss"] for r in kernel_rows])}
        cluster = [r["cluster"] for r in kernel_rows]
        if label != "g2" and any(index[r["train_row_id"]]["cluster"] != r["cluster"] for r in kernel_rows):
            raise ValueError(f"{label} held-out cluster IDs differ.")
        response, weighted = pair_bootstrap(cluster, losses, seed=20261101, replicates=5000)
        delta = response["C+D"]-response["kernel_M3-C"]
        lo, hi = np.quantile(delta, [.025, .975])
        records.append({"target": label, "reference": "C+D", "candidate": "kernel_M3-C",
                        "difference": float(np.mean(losses["C+D"]-losses["kernel_M3-C"])),
                        "ci_low": float(lo), "ci_high": float(hi),
                        "responses": len(kernel_rows), "clusters": len(set(cluster)),
                        "interpretation": "Post-test exploratory; additional demographic inputs and tuning differ."})
        if label == "g2":
            saved = {name: [r for r in kernel if r["model"] == name] for name in
                     ("kernel_M3-C", "original:M3-C", "original:M4-C")}
            ids = [r["train_row_id"] for r in kernel_rows]
            for name, rows in saved.items():
                if [r["train_row_id"] for r in rows] != ids:
                    raise ValueError("G2 pair-weighted model rows do not align.")
            all_losses = {name: np.asarray([r["loss"] for r in rows]) for name, rows in saved.items()}
            _, boot = pair_bootstrap(cluster, all_losses, 20261101, 5000)
            unique = sorted(set(cluster))
            code = np.asarray([{v:i for i,v in enumerate(unique)}[p] for p in cluster])
            count = np.bincount(code)
            for name, values in all_losses.items():
                low, high = np.quantile(boot[name], [.025, .975])
                pair_results.append({"model": name, "pairs": len(unique),
                                     "pair_weighted_loss": float(np.mean(np.bincount(code, weights=values)/count)),
                                     "ci_low": float(low), "ci_high": float(high)})
    write_csv(HERE/"additional_paired_contrasts.csv", records, list(records[0]))
    write_csv(HERE/"g2_pair_weighted.csv", pair_results, list(pair_results[0]))
    write_json(HERE/"paired_provenance.json", {"status": "completed", "matched_pairing_and_clusters_verified": True,
               "bootstrap_replicates": 5000, "source_sha256": {
                   "demographic_g2": digest(ROOT/"demographic_experiment"/"test_predictions.parquet"),
                   "demographic_g3": digest(ROOT/"demographic_experiment"/"diagnostics_extension"/"g3_predictions.parquet"),
                   "demographic_g4": digest(ROOT/"demographic_experiment"/"diagnostics_extension"/"g4_predictions.parquet")},
               "outputs_sha256": {name: digest(HERE/name) for name in ("additional_paired_contrasts.csv", "g2_pair_weighted.csv")},
               "interpretation": "Post-test exploratory; C+D uses demographics while kernel is pooled."})
    print(json.dumps({"cross_model": records, "pair_weighted": pair_results}, indent=2))


if __name__ == "__main__":
    run()
