"""Reconstruct selected hybrid fits and replay saved matched intervals."""

import csv
import json

import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from model import HERE, ROOT, design, fit_spec, load_features, penalty
from study import GRID, digest, load_frozen, load_rows, predict, source_rows, verify_base, write_json
from src.abstention_utility import calculate
from src.development import score
from src.estimator import objective_gradient
from src.learning_curves import common_validation_indices, nested_pair_sets


def read(path):
    with path.open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))


def check_fit(label,train,val,features):
    model=load_frozen(label)
    if model["fit_row_ids"]!=[r["train_row_id"] for r in train] or model["validation_row_ids"]!=[r["train_row_id"] for r in val]:
        raise ValueError(f"{label} fit/validation identity differs.")
    spec=fit_spec(train,features)
    if spec!=model["spec"]:raise ValueError(f"{label} fitting-only preprocessing/cat keys differ.")
    x,vx=design(train,features,spec),design(val,features,spec)
    theta=np.asarray(model["selected"]["coefficients"]); y=np.asarray([r["y_displayed"] for r in train]);vy=np.asarray([r["y_displayed"] for r in val])
    p=penalty(len(spec["cats"]),len(spec["preprocessing"]["final_names"]),
              model["selected"]["cat_ridge"],model["selected"]["cnn_ridge"])
    obj,gradient=objective_gradient(theta,x,y,np.ones(len(y)),p)
    loss=score(vy,np.asarray(vx@theta).ravel())[0]
    np.testing.assert_allclose([obj,loss],[model["selected"]["objective"],model["selected"]["validation_loss"]],rtol=0,atol=1e-12)
    if np.max(np.abs(gradient))>=1e-5:raise ValueError(f"{label} gradient unacceptable.")
    candidates=read(HERE/f"{label}_attempts.csv")
    if len(candidates)!=49 or any(r["accepted"]!="True" for r in candidates):
        raise ValueError(f"{label} candidate grid incomplete.")
    if {(float(r["cat_ridge"]),float(r["cnn_ridge"])) for r in candidates}!={(a,b) for a in GRID for b in GRID}:
        raise ValueError(f"{label} candidate grid differs.")
    minimum=min(float(r["validation_loss"]) for r in candidates)
    winner=max((r for r in candidates if float(r["validation_loss"])<=minimum+1e-6),
               key=lambda r:(float(r["cat_ridge"]),float(r["cnn_ridge"])))
    if (float(winner["cat_ridge"]),float(winner["cnn_ridge"]))!=(model["selected"]["cat_ridge"],model["selected"]["cnn_ridge"]):
        raise ValueError(f"{label} selected penalties differ.")
    return {"model":label,"fit_rows":len(train),"validation_rows":len(val),
            "gradient_max":float(np.max(np.abs(gradient))),"validation_delta":abs(loss-model["selected"]["validation_loss"])}


def check_test(label):
    proof=json.loads((HERE/f"{label}_evaluation_provenance.json").read_text(encoding="utf-8"))
    for name,value in proof["outputs_sha256"].items():
        if digest(HERE/name)!=value:raise ValueError("Saved evaluation output changed.")
    rows=load_rows("test") if label=="g2" else source_rows("G3","test",False)
    reconstructed=predict(load_frozen(label),rows)
    predictions=pq.read_table(HERE/f"{label}_predictions.parquet").to_pylist()
    result={r["model"]:r for r in read(HERE/f"{label}_results.csv")}
    contrasts=read(HERE/f"{label}_contrasts.csv")
    ids=[r["train_row_id"] for r in rows]; names=list(result); losses={}; identity=None
    for name in names:
        data=[r for r in predictions if r["model"]==name]
        if [r["train_row_id"] for r in data]!=ids:raise ValueError("Prediction IDs differ.")
        values=[(r["outcome"],r["cluster"],r["pair_id"]) for r in data]
        if identity is not None and values!=identity:raise ValueError("Matched outcomes/clusters differ.")
        identity=values
        y=np.asarray([r["outcome"] for r in data]);z=np.asarray([r["logit"] for r in data])
        if name=="Cat+CNN":np.testing.assert_allclose(z,reconstructed,rtol=0,atol=1e-12)
        ll=y*np.logaddexp(0.,-z)+(1-y)*np.logaddexp(0.,z)
        np.testing.assert_allclose(ll,[r["loss"] for r in data],rtol=0,atol=1e-12)
        np.testing.assert_allclose(ll.mean(),float(result[name]["log_loss"]),rtol=0,atol=1e-12)
        losses[name]=ll
    clusters=[r["cluster"] for r in predictions if r["model"]=="Cat+CNN"]
    unique=sorted(set(clusters));lookup={k:i for i,k in enumerate(unique)};codes=np.asarray([lookup[v] for v in clusters])
    counts=np.bincount(codes);sums=np.stack([np.bincount(codes,weights=losses[n]) for n in names])
    rng=np.random.default_rng(20261101);samples=np.empty((5000,len(names)))
    for i in range(5000):
        w=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
        samples[i]=sums@w/(counts@w)
    for i,name in enumerate(names):
        np.testing.assert_allclose(np.quantile(samples[:,i],[.025,.975]),[float(result[name]["ci_low"]),float(result[name]["ci_high"])],rtol=0,atol=1e-12)
    for row in contrasts:
        a,b=names.index(row["reference"]),names.index(row["candidate"]);delta=samples[:,a]-samples[:,b]
        np.testing.assert_allclose([np.mean(losses[names[a]]-losses[names[b]]),*np.quantile(delta,[.025,.975])],
                                   [float(row["difference"]),float(row["ci_low"]),float(row["ci_high"])],rtol=0,atol=1e-12)
    return {"target":label,"responses":len(ids),"models":len(names),"contrasts":len(contrasts),"clusters":len(unique),"draws":5000}


def run():
    if (HERE/"verification.json").exists():raise FileExistsError("Verification exists.")
    verify_base();features=load_features();fit,val=load_rows("fit"),load_rows("validation");checks=[]
    with threadpool_limits(limits=1):
        checks.append(check_fit("g2",fit,val,features))
        checks.append(check_fit("g3",source_rows("G3","core_fit",True),source_rows("G3","validation",True),features))
        indices=common_validation_indices(fit,val,[20261030,20261031],[.25,.5,.75,1.]);common=[val[i] for i in indices]
        for seed in (20261030,20261031):
            subsets=nested_pair_sets(fit,seed,[.25,.5,.75,1.])
            for fraction in (.25,.5,.75,1.):
                checks.append(check_fit(f"curve_{seed}_{int(fraction*100):03d}",
                                        [r for r in fit if r["pair_id"] in subsets[fraction]],common,features))
        tests=[check_test(name) for name in ("g2","g3")]
        data=pq.read_table(HERE/"g2_predictions.parquet").to_pylist()
        names=("Cat+CNN","original:common:M1","original:M3-C","original:M4-C")
        models={n:sorted([r for r in data if r["model"]==n],key=lambda r:r["train_row_id"]) for n in names}
        calculated=calculate(models,[round(.5+.025*i,3) for i in range(21)],20261101,5000)
        saved=read(HERE/"g2_abstention_utility.csv")
        if len(saved)!=len(calculated):raise ValueError("Abstention grid incomplete.")
        keys=("coverage","utility_per_original_response","utility_ci95_low","utility_ci95_high","coverage_ci95_low","coverage_ci95_high")
        for a,b in zip(saved,calculated):
            if a["model"]!=b["model"] or float(a["threshold"])!=b["threshold"]:raise ValueError("Utility grid misaligned.")
            np.testing.assert_allclose([float(a[k]) for k in keys],[float(b[k]) for k in keys],rtol=0,atol=1e-12)
    write_json(HERE/"verification.json",{"status":"passed","selected_fits":checks,"test_replays":tests,
               "abstention_cells_replayed":len(calculated),"raw_test_outcomes_read":False,
               "g4":"unsupported by user choice; no predictions generated",
               "limitations":"Previously exposed outcomes remain retrospective and not independently confirmatory."})
    print(json.dumps({"selected_fits":len(checks),"test_replays":tests,"abstention_cells":len(calculated)},indent=2))


if __name__=="__main__":
    run()
