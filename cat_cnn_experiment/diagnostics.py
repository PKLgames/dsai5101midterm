"""Original-style validation and test support/calibration slices."""

import argparse
import json
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit

from study import HERE, ROOT, digest, load_frozen, load_rows, predict, write_csv, write_json
from src.evaluate import subgroup_labels


def run(partition):
    outputs = [HERE/f"{partition}_support_slices.csv", HERE/f"{partition}_support_calibration.csv",
               HERE/f"{partition}_support_provenance.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Support diagnostics already exist.")
    rows = load_rows(partition)
    if partition == "test":
        proof = json.loads((HERE/"g2_evaluation_provenance.json").read_text(encoding="utf-8"))
        if digest(HERE/"g2_predictions.parquet") != proof["outputs_sha256"]["g2_predictions.parquet"]:
            raise ValueError("Saved hybrid predictions changed.")
        saved = pq.read_table(HERE/"g2_predictions.parquet", filters=[("model", "=", "Cat+CNN")]).to_pylist()
        index = {r["train_row_id"]: r for r in saved}
        if len(index) != len(saved) or set(index) != {r["train_row_id"] for r in rows}:
            raise ValueError("Saved hybrid predictions incomplete.")
        z=np.asarray([index[r["train_row_id"]]["logit"] for r in rows])
        y=np.asarray([index[r["train_row_id"]]["outcome"] for r in rows])
        original=pq.read_table(ROOT/"artifacts"/"test_predictions.parquet",
                               filters=[("population","=","common"),("model_id","=","common:M0")]).to_pylist()
        flags={r["train_row_id"]:r for r in original}
        if len(flags)!=len(original) or set(flags)!=set(index):
            raise ValueError("Original support flags incomplete.")
        groups=defaultdict(list)
        for i,row in enumerate(rows):
            flag=flags[row["train_row_id"]]
            if flag["pair_id"]!=row["pair_id"] or flag["outcome"]!=y[i]:
                raise ValueError("Matched support flags disagree.")
            for key in subgroup_labels(flag): groups[key].append(i)
    else:
        z=predict(load_frozen("g2"),rows)
        y=np.asarray([r["y_displayed"] for r in rows])
        graph=pq.read_table(ROOT/"artifacts"/"graph_components.parquet").to_pylist()
        opponents={r["node_id"]:r["distinct_opponents"] for r in graph if r["population"]=="common" and r["entity"]=="photo"}
        features=pq.read_table(ROOT/"features"/"Y.parquet",columns=["image_id","detection_status","candidate_cat_count"]).to_pylist()
        index={r["image_id"]:r for r in features}
        if len(index)!=len(features): raise ValueError("Duplicate Y photo keys.")
        groups=defaultdict(list)
        for i,row in enumerate(rows):
            a,b=row["a_image"],row["b_image"]
            if a not in index or b not in index or a not in opponents or b not in opponents:
                raise ValueError("Validation support/feature records missing.")
            support=min(opponents[a],opponents[b]); band="1" if support<=1 else "2-4" if support<=4 else "5-9" if support<=9 else "10+"
            for key in (("overall","all"),("comparison_type",row["comparison_type"]),
                        ("pet_care",row["pet_care"] if row["pet_care"] else "missing"),
                        ("minimum_endpoint_opponents",band)): groups[key].append(i)
            endpoints=[index[a],index[b]]
            for label,yes in (("no_detection",any(r["detection_status"]=="no_detection" for r in endpoints)),
                              ("dog_fallback",any(r["detection_status"]=="dog_fallback" for r in endpoints)),
                              ("multiple_cats",any(r["candidate_cat_count"] is not None and r["candidate_cat_count"]>1 for r in endpoints)),
                              ("both_cat_detected",all(r["detection_status"]=="cat_detected" for r in endpoints))):
                if yes:groups[("feature_quality_overlapping",label)].append(i)
    p=expit(z); loss=y*np.logaddexp(0.,-z)+(1-y)*np.logaddexp(0.,z); bins=np.minimum((p*10).astype(int),9)
    slices=[];calibration=[]
    for (dimension,group),members in sorted(groups.items()):
        indices=np.asarray(members,dtype=int); n=len(indices); pairs=len({rows[i]["pair_id"] for i in indices})
        supported=n>=100 and pairs>=30
        slices.append({"dimension":dimension,"group":group,"responses":n,"pairs":pairs,
                       "support":"supported" if supported else "low_support","log_loss":float(loss[indices].mean()),
                       "brier":float(np.mean((p[indices]-y[indices])**2)),"accuracy":float(np.mean((p[indices]>=.5)==y[indices]))})
        if partition=="validation" and not supported:continue
        for b in range(10):
            sel=indices[bins[indices]==b]; count=len(sel); pc=len({rows[i]["pair_id"] for i in sel})
            calibration.append({"dimension":dimension,"group":group,"bin":b,"responses":count,"pairs":pc,
                                "support":"supported" if count>=30 and pc>=10 else "low_support",
                                "mean_probability":float(p[sel].mean()) if count else "",
                                "observed_a_rate":float(y[sel].mean()) if count else ""})
    write_csv(outputs[0],slices,list(slices[0]));write_csv(outputs[1],calibration,list(calibration[0]))
    write_json(outputs[2],{"status":"completed","partition":partition,"slices":len(slices),"bins":len(calibration),
                           "model_freeze_sha256":digest(HERE/"g2_freeze.json"),
                           "outputs_sha256":{p.name:digest(p) for p in outputs[:2]},
                           "raw_test_outcomes_read":False,"limitation":"Retrospective descriptive; quality flags overlap."})
    print(f"{partition}: saved {len(slices)} support slices and {len(calibration)} reliability bins.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("partition",choices=("validation","test"))
    run(parser.parse_args().partition)
