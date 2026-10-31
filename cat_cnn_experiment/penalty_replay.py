"""Reproduce all 49 G2 cat/CNN penalty candidates independently."""

import csv

import numpy as np
from threadpoolctl import threadpool_limits

from model import HERE, design, fit_spec, load_features, penalty
from study import GRID, digest, load_frozen, load_rows, score, write_csv, write_json
from src.estimator import fit_logistic


def run():
    if (HERE/"penalty_refits.csv").exists() or (HERE/"penalty_provenance.json").exists():
        raise FileExistsError("Penalty refit output exists.")
    load_frozen("g2")
    with (HERE/"g2_attempts.csv").open(encoding="utf-8",newline="") as stream:
        original={(float(r["cat_ridge"]),float(r["cnn_ridge"])):r for r in csv.DictReader(stream)}
    train,val=load_rows("fit"),load_rows("validation");features=load_features();spec=fit_spec(train,features)
    y=np.asarray([r["y_displayed"] for r in train]);vy=np.asarray([r["y_displayed"] for r in val])
    rows=[];nc=len(spec["cats"]);dim=len(spec["preprocessing"]["final_names"])
    with threadpool_limits(limits=1):
        x,vx=design(train,features,spec),design(val,features,spec)
        for cr in GRID:
            for vr in GRID:
                fit=fit_logistic(x,y,penalty(nc,dim,cr,vr))
                if not fit.accepted:raise ValueError(f"Penalty refit failed {cr}/{vr}: {fit.attempts}")
                loss=score(vy,np.asarray(vx@fit.coefficients).ravel())[0]
                delta=loss-float(original[(cr,vr)]["validation_loss"])
                if abs(delta)>1e-6:raise ValueError(f"Penalty validation mismatch {cr}/{vr}: {delta}")
                rows.append({"cat_ridge":cr,"cnn_ridge":vr,"accepted":True,
                             "training_loss":score(y,np.asarray(x@fit.coefficients).ravel())[0],
                             "validation_loss":loss,"gradient_max":fit.gradient_max,
                             "cat_coefficient_norm":float(np.linalg.norm(fit.coefficients[1:1+nc])),
                             "cnn_coefficient_norm":float(np.linalg.norm(fit.coefficients[1+nc:])),
                             "validation_reproduction_delta":delta})
            print(f"Penalty replay cat ridge {cr:g} completed",flush=True)
    write_csv(HERE/"penalty_refits.csv",rows,list(rows[0]))
    write_json(HERE/"penalty_provenance.json",{"status":"all_candidates_reproduced","fits":len(rows),
               "maximum_validation_difference":max(abs(r["validation_reproduction_delta"]) for r in rows),
               "original_attempts_sha256":digest(HERE/"g2_attempts.csv"),"output_sha256":digest(HERE/"penalty_refits.csv"),
               "test_outcomes_read":False,"interpretation":"Post-test motivated validation reconstruction."})
    print(f"Reproduced {len(rows)} penalty candidates.")


if __name__=="__main__":
    run()
