"""Independently replay G2/G3 subgroup point differences and bootstrap intervals."""

import csv

import numpy as np
import pyarrow.parquet as pq

from study import HERE, digest, load_rows, source_rows, write_json


def read(path):
    with path.open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))


def run():
    if (HERE/"subgroup_verification.json").exists():raise FileExistsError("Subgroup verification exists.")
    count=0
    for label in ("g2","g3"):
        structure=load_rows("test") if label=="g2" else source_rows("G3","test",False)
        if any((r["a_cat"]==r["b_cat"])!=(r["comparison_type"]=="within_cat") for r in structure):
            raise ValueError("Comparison type conflicts with cat identity.")
        types={r["train_row_id"]:r["comparison_type"] for r in structure}
        predictions=pq.read_table(HERE/f"{label}_predictions.parquet").to_pylist()
        contrasts=read(HERE/("g2_subgroup_contrasts.csv" if label=="g2" else "g3_within_contrasts.csv"))
        groups=("within_cat","between_cats") if label=="g2" else ("within_cat",)
        for group in groups:
            subset=[r for r in contrasts if label=="g3" or r["comparison_type"]==group]
            names=["Cat+CNN"]+[r["reference"] for r in subset]
            arrays={name:[r for r in predictions if r["model"]==name and types[r["train_row_id"]]==group] for name in names}
            alignment=[(r["train_row_id"],r["pair_id"],r["cluster"],r["outcome"]) for r in arrays["Cat+CNN"]]
            if not alignment or any([(r["train_row_id"],r["pair_id"],r["cluster"],r["outcome"]) for r in arrays[name]]!=alignment for name in names):
                raise ValueError("Subgroup rows or outcomes do not align.")
            clusters=[r[2] for r in alignment];unique=sorted(set(clusters));index={k:i for i,k in enumerate(unique)}
            codes=np.asarray([index[c] for c in clusters]);counts=np.bincount(codes)
            losses={name:np.asarray([r["loss"] for r in arrays[name]]) for name in names}
            sums=np.stack([np.bincount(codes,weights=losses[name]) for name in names])
            rng=np.random.default_rng(20261101);samples=np.empty((5000,len(names)))
            for i in range(5000):
                w=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
                samples[i]=sums@w/(counts@w)
            for row in subset:
                pos=names.index(row["reference"]);delta=samples[:,pos]-samples[:,0]
                np.testing.assert_allclose([np.mean(losses[row["reference"]]-losses["Cat+CNN"]),*np.quantile(delta,[.025,.975])],
                    [float(row["difference"]),float(row["ci_low"]),float(row["ci_high"])],rtol=0,atol=1e-12)
                count+=1
    write_json(HERE/"subgroup_verification.json",{"status":"passed","contrasts_replayed":count,
               "bootstrap_replicates_per_group":5000,"g2_prediction_sha256":digest(HERE/"g2_predictions.parquet"),
               "g3_prediction_sha256":digest(HERE/"g3_predictions.parquet"),"raw_test_outcomes_read":False,
               "limitations":"Post-test subgroup intervals are unadjusted and not independent."})
    print(f"Replayed {count} matched G2/G3 subgroup comparisons and intervals.")


if __name__=="__main__":
    run()
