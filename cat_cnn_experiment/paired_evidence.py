"""Matched Cat + CNN versus kernel and demographic saved predictions."""

import json

import numpy as np
import pyarrow.parquet as pq

from study import HERE, ROOT, digest, write_csv, write_json
from src.evaluate import pair_bootstrap


def run():
    if (HERE/"additional_paired_contrasts.csv").exists() or (HERE/"paired_provenance.json").exists():
        raise FileExistsError("Additional paired evidence exists.")
    records=[]
    sources={}
    for label in ("g2","g3"):
        proof=json.loads((HERE/f"{label}_evaluation_provenance.json").read_text(encoding="utf-8"))
        target=HERE/f"{label}_predictions.parquet"
        if digest(target)!=proof["outputs_sha256"][target.name]:raise ValueError("Hybrid prediction source changed.")
        hybrid=pq.read_table(target,filters=[("model","=","Cat+CNN")]).to_pylist()
        if label=="g2":
            demo=ROOT/"demographic_experiment"/"test_predictions.parquet"
            demo_proof=json.loads((ROOT/"demographic_experiment"/"evaluation_provenance.json").read_text(encoding="utf-8"))
            expected=demo_proof["outputs_sha256"]["test_predictions.parquet"]
        else:
            base=ROOT/"demographic_experiment"/"diagnostics_extension"
            demo=base/"g3_predictions.parquet";demo_proof=json.loads((base/"g3_provenance.json").read_text(encoding="utf-8"))
            expected=demo_proof["outputs_sha256"]["g3_predictions.parquet"]
        if digest(demo)!=expected:raise ValueError("Demographic prediction source changed.")
        kernel=ROOT/"kernel_experiment"/f"{label}_predictions.parquet"
        kernel_proof=json.loads((ROOT/"kernel_experiment"/f"{label}_evaluation_provenance.json").read_text(encoding="utf-8"))
        if digest(kernel)!=kernel_proof["outputs_sha256"][kernel.name]:raise ValueError("Kernel prediction source changed.")
        data=pq.read_table(demo,filters=[("model","in",["Cat+D","C+D"])]).to_pylist()
        data+=pq.read_table(kernel,filters=[("model","=","kernel_M3-C")]).to_pylist()
        ids=[r["train_row_id"] for r in hybrid];clusters=[r["cluster"] for r in hybrid]
        losses={"Cat+CNN":np.asarray([r["loss"] for r in hybrid])}
        for name in ("Cat+D","C+D","kernel_M3-C"):
            subset=[r for r in data if r["model"]==name];index={r["train_row_id"]:r for r in subset}
            if len(index)!=len(subset) or set(index)!=set(ids):raise ValueError("Matched comparator IDs differ.")
            if any(index[r["train_row_id"]]["pair_id"]!=r["pair_id"] or index[r["train_row_id"]]["outcome"]!=r["outcome"] for r in hybrid):
                raise ValueError("Matched comparator outcomes or pairs differ.")
            if label=="g3" and any(index[r["train_row_id"]]["cluster"]!=r["cluster"] for r in hybrid):
                raise ValueError("G3 bootstrap clusters differ.")
            losses[name]=np.asarray([index[key]["loss"] for key in ids])
        samples,_=pair_bootstrap(clusters,losses,20261101,5000)
        for name in ("Cat+D","C+D","kernel_M3-C"):
            d=samples[name]-samples["Cat+CNN"];lo,hi=np.quantile(d,[.025,.975])
            records.append({"target":label,"reference":name,"candidate":"Cat+CNN",
                            "difference":float(np.mean(losses[name]-losses["Cat+CNN"])),
                            "ci_low":float(lo),"ci_high":float(hi),"bootstrap_se":float(np.std(d,ddof=1)),
                            "responses":len(ids),"clusters":len(set(clusters)),
                            "interpretation":"Post-test exploratory; comparator inputs and tuning differ."})
        sources[label]={"hybrid":digest(target),"demographic":digest(demo),"kernel":digest(kernel)}
    write_csv(HERE/"additional_paired_contrasts.csv",records,list(records[0]))
    write_json(HERE/"paired_provenance.json",{"status":"completed","matched_ids_pairs_outcomes_clusters_verified":True,
               "bootstrap_replicates":5000,"source_sha256":sources,"output_sha256":digest(HERE/"additional_paired_contrasts.csv"),
               "limitations":"Post-test exploratory, comparisons do not isolate causal components or equal model inputs."})
    print(json.dumps(records,indent=2))


if __name__=="__main__":
    run()
