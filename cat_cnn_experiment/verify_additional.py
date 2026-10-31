"""Reconcile additional matched comparisons, slices and G3 within-cat intervals."""

import csv
import json

import numpy as np
import pyarrow.parquet as pq

from study import HERE, ROOT, digest, write_json


def read(path):
    with path.open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))


def bootstrap_difference(a,b,clusters):
    keys=sorted(set(clusters));lookup={k:i for i,k in enumerate(keys)}
    codes=np.asarray([lookup[k] for k in clusters]);counts=np.bincount(codes)
    sums=np.bincount(codes,weights=a-b);generator=np.random.default_rng(20261101);samples=[]
    for _ in range(5000):
        weights=np.bincount(generator.integers(0,len(keys),len(keys)),minlength=len(keys))
        samples.append(float(sums@weights/(counts@weights)))
    return [float(np.mean(a-b)),*np.quantile(samples,[.025,.975])]


def run():
    if (HERE/"additional_verification.json").exists():raise FileExistsError("Additional verification exists.")
    penalty=json.loads((HERE/"penalty_provenance.json").read_text(encoding="utf-8"))
    if penalty["status"]!="all_candidates_reproduced" or penalty["output_sha256"]!=digest(HERE/"penalty_refits.csv"):
        raise ValueError("Penalty refit provenance mismatch.")
    fits=read(HERE/"penalty_refits.csv")
    if len(fits)!=49 or any(r["accepted"]!="True" or abs(float(r["validation_reproduction_delta"]))>1e-6 for r in fits):
        raise ValueError("Penalty refit grid incomplete.")
    contrasts=read(HERE/"additional_paired_contrasts.csv")
    for target in ("g2","g3"):
        hybrid=pq.read_table(HERE/f"{target}_predictions.parquet",filters=[("model","=","Cat+CNN")]).to_pylist()
        if target=="g2":demo=ROOT/"demographic_experiment"/"test_predictions.parquet"
        else:demo=ROOT/"demographic_experiment"/"diagnostics_extension"/"g3_predictions.parquet"
        kernel=ROOT/"kernel_experiment"/f"{target}_predictions.parquet"
        records=pq.read_table(demo,filters=[("model","in",["Cat+D","C+D"])]).to_pylist()
        records+=pq.read_table(kernel,filters=[("model","=","kernel_M3-C")]).to_pylist()
        ids=[r["train_row_id"] for r in hybrid];clusters=[r["cluster"] for r in hybrid]
        b=np.asarray([r["loss"] for r in hybrid])
        for name in ("Cat+D","C+D","kernel_M3-C"):
            subset=[r for r in records if r["model"]==name];lookup={r["train_row_id"]:r for r in subset}
            if len(lookup)!=len(subset) or set(lookup)!=set(ids):raise ValueError("Additional comparator mismatch.")
            if any(lookup[r["train_row_id"]]["outcome"]!=r["outcome"] or lookup[r["train_row_id"]]["pair_id"]!=r["pair_id"] for r in hybrid):
                raise ValueError("Additional comparator outcomes/pairs differ.")
            a=np.asarray([lookup[key]["loss"] for key in ids])
            row=next(r for r in contrasts if r["target"]==target and r["reference"]==name)
            np.testing.assert_allclose(bootstrap_difference(a,b,clusters),
                                       [float(row["difference"]),float(row["ci_low"]),float(row["ci_high"])],rtol=0,atol=1e-12)
    for partition,oldfile,modelcol,model in (
        ("validation",ROOT/"tables"/"diagnostic_results.csv","model","M3-C"),
        ("test",ROOT/"tables"/"subgroup_results.csv","model_id","M3-C")):
        old=read(oldfile)
        reference={(r["dimension"],r["group"]):r for r in old if r[modelcol]==model and
                   (partition!="test" or r["population"]=="common")}
        rows=read(HERE/f"{partition}_support_slices.csv")
        for r in rows:
            baseline=reference[(r["dimension"],r["group"])]
            if (r["responses"],r["pairs"],r["support"])!=(baseline["responses"],baseline["pairs"],baseline["support_status"]):
                raise ValueError("Original-style subgroup support differs.")
    figure=json.loads((HERE/"abstention_figure_provenance.json").read_text(encoding="utf-8"))
    if figure["input_sha256"]!=digest(HERE/"g2_abstention_utility.csv") or figure["output_sha256"]!=digest(HERE/"presentation_abstention.pdf"):
        raise ValueError("Abstention figure provenance mismatch.")
    write_json(HERE/"additional_verification.json",{"status":"passed","penalty_candidates_replayed":len(fits),
               "extra_comparisons_replayed":len(contrasts),"bootstrap_replicates_per_target":5000,
               "validation_and_test_support_groups_matched":16,"raw_test_outcomes_read":False,
               "limitations":"Previously exposed outcomes remain exploratory."})
    print("Verified 49 penalty refits, six additional paired comparisons, and original subgroup support.")


if __name__=="__main__":
    run()
