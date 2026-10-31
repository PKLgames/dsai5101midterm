"""Exploratory paired within/between G2 comparisons from saved predictions."""

import json

import numpy as np
import pyarrow.parquet as pq

from study import HERE, digest, load_rows, write_csv, write_json
from src.evaluate import pair_bootstrap


def run():
    if any((HERE/name).exists() for name in ("g2_subgroup_contrasts.csv", "g2_subgroup_contrasts_provenance.json")):
        raise FileExistsError("G2 subgroup contrast outputs already exist.")
    proof=json.loads((HERE/"g2_evaluation_provenance.json").read_text(encoding="utf-8"))
    if digest(HERE/"g2_predictions.parquet")!=proof["outputs_sha256"]["g2_predictions.parquet"]:
        raise ValueError("G2 saved predictions changed.")
    structure=load_rows("test")
    predictions=pq.read_table(HERE/"g2_predictions.parquet").to_pylist()
    names=("Cat+CNN","original:common:M0","original:common:M1","original:common:M2","original:M3-C","original:M4-C")
    results=[]
    for group in ("within_cat","between_cats"):
        ids={r["train_row_id"] for r in structure if r["comparison_type"]==group}
        models={name:[r for r in predictions if r["model"]==name and r["train_row_id"] in ids] for name in names}
        identity=[(r["train_row_id"],r["pair_id"],r["outcome"]) for r in models[names[0]]]
        if not identity or any([(r["train_row_id"],r["pair_id"],r["outcome"]) for r in models[name]]!=identity for name in names):
            raise ValueError("Subgroup matched identities differ.")
        pairs=[r[1] for r in identity];losses={n:np.asarray([r["loss"] for r in models[n]]) for n in names}
        samples,_=pair_bootstrap(pairs,losses,20261101,5000)
        for name in names[1:]:
            delta=samples[name]-samples["Cat+CNN"];lo,hi=np.quantile(delta,[.025,.975])
            results.append({"comparison_type":group,"reference":name,"candidate":"Cat+CNN",
                            "responses":len(identity),"pairs":len(set(pairs)),
                            "difference":float(np.mean(losses[name]-losses["Cat+CNN"])),
                            "ci_low":float(lo),"ci_high":float(hi),"bootstrap_se":float(np.std(delta,ddof=1)),
                            "interpretation":"Post-test 95% pointwise unadjusted subgroup comparison."})
    write_csv(HERE/"g2_subgroup_contrasts.csv",results,list(results[0]))
    write_json(HERE/"g2_subgroup_contrasts_provenance.json",{"status":"completed_post_test_exploratory",
               "input_sha256":digest(HERE/"g2_predictions.parquet"),"output_sha256":digest(HERE/"g2_subgroup_contrasts.csv"),
               "bootstrap_replicates":5000,"seed":20261101,"cluster":"unordered pair","raw_test_outcomes_read":False,
               "limitations":"Subgroup comparisons are post-test, unadjusted and not independent."})
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    run()
