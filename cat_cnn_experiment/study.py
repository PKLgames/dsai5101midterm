"""Quarantined Cat + CNN development, control, transfer and saved-outcome analysis."""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit
from threadpoolctl import threadpool_limits

from model import HERE, ROOT, design, fit_spec, load_features, penalty
sys.path.insert(0, str(ROOT/"demographic_experiment"/"diagnostics_extension"))
from transfer import source_rows  # noqa: E402
from g3_experiment.prepare import digest, snapshot, write_csv, write_json  # noqa: E402
from src.abstention_utility import calculate  # noqa: E402
from src.development import score  # noqa: E402
from src.estimator import fit_logistic  # noqa: E402
from src.evaluate import pair_bootstrap  # noqa: E402
from src.freeze import validate_freeze  # noqa: E402
from src.learning_curves import common_validation_indices, nested_pair_sets  # noqa: E402
from experiment import load_rows  # noqa: E402

EXCLUDED = {".git", ".venv", ".cache", "Presentation_Evidence", "cat_cnn_experiment"}
GRID = (1e-6, 1e-5, 1e-4, .001, .01, .1, 1.)


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def verify_base():
    validate_freeze(ROOT)
    for folder in ("g3_experiment", "g4_experiment", "demographic_experiment"):
        root = ROOT/folder
        path = root/"freeze.json"
        if digest(path) != (root/"freeze.sha256").read_text(encoding="ascii").strip():
            raise ValueError(f"{folder} freeze mismatch.")
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, value in data.get("experiment_sha256", data.get("source_output_sha256", {})).items():
            if digest(root/name) != value:
                raise ValueError(f"{folder} frozen artifact changed: {name}")
    path = ROOT/"kernel_experiment"/"g2_freeze.json"
    if digest(path) != (path.parent/"g2_freeze.sha256").read_text(encoding="ascii").strip():
        raise ValueError("Kernel freeze mismatch.")
    evaluation = json.loads((ROOT/"artifacts"/"evaluation_provenance.json").read_text(encoding="utf-8"))
    if evaluation["freeze_sha256"] != digest(ROOT/"artifacts"/"model_freeze.json"):
        raise ValueError("Original evaluation freeze mismatch.")
    for name, value in evaluation["output_sha256"].items():
        if digest(ROOT.joinpath(*name.split("/"))) != value:
            raise ValueError(f"Original evaluation output changed: {name}")


def prepare():
    if (HERE/"original_hashes_before.json").exists():
        raise FileExistsError("Study snapshot already exists.")
    verify_base()
    before = snapshot(ROOT, EXCLUDED)
    evidence = snapshot(ROOT/"Presentation_Evidence", set())
    write_json(HERE/"original_hashes_before.json", before)
    write_json(HERE/"evidence_hashes_before.json", evidence)
    write_json(HERE/"g4_unsupported.json", {"status": "unsupported_by_user_choice",
               "reason": "The user explicitly declined zero-score cold-start for an unseen cat.",
               "predictions_generated": False, "test_outcomes_read": False,
               "note": "The CNN block could score the photo, but the specified joint model has no fitted cat identity score."})
    print(f"Protected {len(before)} project files and {len(evidence)} prior evidence files.")


def train(label, train_rows, val_rows):
    attempts, model_path, freeze_path = (HERE/f"{label}_{suffix}" for suffix in
                                          ("attempts.csv", "model.json", "freeze.json"))
    if any(p.exists() for p in (attempts, model_path, freeze_path, HERE/f"{label}_freeze.sha256")):
        raise FileExistsError(f"Development outputs already exist: {label}")
    verify_base()
    if any(r["y_displayed"] not in (0, 1) for r in train_rows+val_rows):
        raise ValueError("Invalid fitting/validation outcomes.")
    features = load_features()
    spec = fit_spec(train_rows, features)
    with threadpool_limits(limits=1):
        x, vx = design(train_rows, features, spec), design(val_rows, features, spec)
    y = np.asarray([r["y_displayed"] for r in train_rows])
    vy = np.asarray([r["y_displayed"] for r in val_rows])
    nc, dimension = len(spec["cats"]), len(spec["preprocessing"]["final_names"])
    fields = ("cat_ridge", "cnn_ridge", "accepted", "validation_loss", "training_loss",
              "objective", "gradient_max", "cat_coefficient_norm", "cnn_coefficient_norm",
              "coefficient_count", "iterations", "seconds", "attempts")
    candidates = []
    with attempts.open("x", encoding="utf-8", newline="") as stream, threadpool_limits(limits=1):
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        stream.flush()
        for cat_ridge in GRID:
            for cnn_ridge in GRID:
                started = time.monotonic()
                fitted = fit_logistic(x, y, penalty(nc, dimension, cat_ridge, cnn_ridge))
                v = np.asarray(vx @ fitted.coefficients).ravel() if fitted.accepted else None
                valid = v is not None and np.isfinite(v).all()
                entry = {"cat_ridge": cat_ridge, "cnn_ridge": cnn_ridge,
                         "accepted": bool(fitted.accepted and valid),
                         "validation_loss": score(vy, v)[0] if valid else None,
                         "training_loss": score(y, np.asarray(x @ fitted.coefficients).ravel())[0] if valid else None,
                         "objective": safe(fitted.objective), "gradient_max": safe(fitted.gradient_max),
                         "cat_coefficient_norm": float(np.linalg.norm(fitted.coefficients[1:1+nc])) if valid else None,
                         "cnn_coefficient_norm": float(np.linalg.norm(fitted.coefficients[1+nc:])) if valid else None,
                         "coefficient_count": x.shape[1], "iterations": fitted.iterations,
                         "seconds": time.monotonic()-started, "attempts": json.dumps(safe(fitted.attempts)),
                         "coefficients": fitted.coefficients.tolist()}
                writer.writerow({name: entry[name] for name in fields})
                stream.flush()
                candidates.append(entry)
            print(f"{label}: completed cat ridge {cat_ridge:g}", flush=True)
    failures = sum(not r["accepted"] for r in candidates)
    if failures:
        raise RuntimeError(f"{label} has {failures} failed candidates; inspect log.")
    minimum = min(r["validation_loss"] for r in candidates)
    selected = max((r for r in candidates if r["validation_loss"] <= minimum+1e-6),
                   key=lambda r: (r["cat_ridge"], r["cnn_ridge"]))
    model = {"label": label, "selected": selected, "spec": spec,
             "fit_responses": len(train_rows), "fit_pairs": len({r["pair_id"] for r in train_rows}),
             "fit_photos": len(spec["fitting_photos"]), "fit_cats": nc,
             "validation_responses": len(val_rows), "validation_pairs": len({r["pair_id"] for r in val_rows}),
             "fit_row_ids": [r["train_row_id"] for r in train_rows],
             "validation_row_ids": [r["train_row_id"] for r in val_rows]}
    write_json(model_path, model)
    frozen = {"status": "frozen_before_saved_test_reanalysis", "label": label,
              "protocol_sha256": digest(HERE/"protocol.json"), "model_sha256": digest(model_path),
              "attempts_sha256": digest(attempts), "source_sha256": {name: digest(HERE/name) for name in ("model.py", "study.py")},
              "g2_freeze_sha256": digest(ROOT/"artifacts"/"model_freeze.json"),
              "cnn_features_sha256": digest(ROOT/"features"/"C.parquet"), "prior_outcome_exposure": True}
    write_json(freeze_path, frozen)
    (HERE/f"{label}_freeze.sha256").write_text(digest(freeze_path)+"\n", encoding="ascii")
    print(f"{label} selected cat={selected['cat_ridge']} cnn={selected['cnn_ridge']} val={selected['validation_loss']:.9f}", flush=True)


def load_frozen(label):
    path = HERE/f"{label}_freeze.json"
    if digest(path) != (HERE/f"{label}_freeze.sha256").read_text(encoding="ascii").strip():
        raise ValueError(f"{label} freeze seal mismatch.")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    for file, key in ((HERE/"protocol.json", "protocol_sha256"), (HERE/f"{label}_model.json", "model_sha256"),
                      (HERE/f"{label}_attempts.csv", "attempts_sha256"),
                      (ROOT/"artifacts"/"model_freeze.json", "g2_freeze_sha256"),
                      (ROOT/"features"/"C.parquet", "cnn_features_sha256")):
        if digest(file) != freeze[key]:
            raise ValueError(f"{label} frozen input changed: {file.name}")
    for name, value in freeze["source_sha256"].items():
        if digest(HERE/name) != value:
            raise ValueError(f"{label} frozen code changed: {name}")
    return json.loads((HERE/f"{label}_model.json").read_text(encoding="utf-8"))


def predict(model, rows):
    with threadpool_limits(limits=1):
        x = design(rows, load_features(), model["spec"])
        z = np.asarray(x @ np.asarray(model["selected"]["coefficients"])).ravel()
    if z.shape != (len(rows),) or not np.isfinite(z).all():
        raise ValueError("Hybrid predictions are incomplete or nonfinite.")
    return z


def control():
    out, proof = HERE/"fair_coin_control.csv", HERE/"fair_coin_provenance.json"
    if out.exists() or proof.exists():
        raise FileExistsError("Fair-coin control already exists.")
    model = load_frozen("g2")
    train_rows, val_rows = load_rows("fit"), load_rows("validation")
    features = load_features()
    with threadpool_limits(limits=1):
        x, vx = design(train_rows, features, model["spec"]), design(val_rows, features, model["spec"])
    p = penalty(len(model["spec"]["cats"]), len(model["spec"]["preprocessing"]["final_names"]),
                model["selected"]["cat_ridge"], model["selected"]["cnn_ridge"])
    generator = np.random.default_rng(20261102)
    records = []
    with out.open("x", encoding="utf-8", newline="") as stream, threadpool_limits(limits=1):
        writer = csv.DictWriter(stream, fieldnames=["rep", "accepted", "validation_loss", "improvement", "gradient_max", "attempts"])
        writer.writeheader()
        for rep in range(20):
            y = generator.integers(0, 2, len(train_rows)).astype(float)
            vy = generator.integers(0, 2, len(val_rows)).astype(float)
            fitted = fit_logistic(x, y, p)
            loss = score(vy, np.asarray(vx @ fitted.coefficients).ravel())[0] if fitted.accepted else None
            row = {"rep": rep+1, "accepted": fitted.accepted, "validation_loss": loss,
                   "improvement": float(np.log(2)-loss) if loss is not None else None,
                   "gradient_max": safe(fitted.gradient_max), "attempts": json.dumps(safe(fitted.attempts))}
            writer.writerow(row)
            stream.flush()
            records.append(row)
    fails = sum(not r["accepted"] for r in records)
    improved = sum(r["improvement"] is not None and r["improvement"] > 0 for r in records)
    status = "passed_operational_control" if fails == 0 and improved < 16 else "blocked_investigation_required"
    write_json(proof, {"status": status, "seed": 20261102, "repetitions": 20, "failed_fits": fails,
                       "improvements": improved, "mean_loss": float(np.mean([r["validation_loss"] for r in records if r["accepted"]])),
                       "g2_freeze_sha256": digest(HERE/"g2_freeze.json"), "results_sha256": digest(out),
                       "limitations": "Operational check, not a leakage-free certificate."})
    if status != "passed_operational_control":
        raise RuntimeError("Fair-coin control requires investigation.")
    print(f"Control: {improved}/20 improvements, {fails} failed fits.")


def temperature():
    if any((HERE/name).exists() for name in ("temperature_candidates.csv", "temperature_freeze.json", "temperature_freeze.sha256")):
        raise FileExistsError("Temperature selection already exists.")
    model, val = load_frozen("g2"), load_rows("validation")
    z = predict(model, val)
    y = np.asarray([r["y_displayed"] for r in val])
    grid = (.25, .5, .75, 1., 1.5, 2., 4.)
    values = [(t, score(y, z/t)) for t in grid]
    chosen = min(values, key=lambda x: (x[1][0], abs(x[0]-1), x[0]))[0]
    rows = [{"temperature": t, "validation_loss": s[0], "brier": s[1], "accuracy": s[2],
             "selected": t == chosen} for t, s in values]
    write_csv(HERE/"temperature_candidates.csv", rows, list(rows[0]))
    write_json(HERE/"temperature_freeze.json", {"temperature": chosen, "selection": "exact minimum; tie nearest 1 then smaller temperature",
               "g2_freeze_sha256": digest(HERE/"g2_freeze.json"),
               "candidates_sha256": digest(HERE/"temperature_candidates.csv"), "previous_test_exposure": True})
    (HERE/"temperature_freeze.sha256").write_text(digest(HERE/"temperature_freeze.json")+"\n", encoding="ascii")
    print(f"Validation-selected temperature: {chosen}")


def evaluate(label):
    if label not in ("g2", "g3"):
        raise ValueError("G4 is unsupported by explicit user choice.")
    model = load_frozen(label)
    guard = HERE/f"{label}_evaluation_started.json"
    if guard.exists() or (HERE/f"{label}_results.csv").exists():
        raise FileExistsError("Saved-test reanalysis already started.")
    if label == "g2":
        control_proof = json.loads((HERE/"fair_coin_provenance.json").read_text(encoding="utf-8"))
        if control_proof["status"] != "passed_operational_control" or control_proof["results_sha256"] != digest(HERE/"fair_coin_control.csv"):
            raise ValueError("Fair-coin control failed or changed.")
        tf = json.loads((HERE/"temperature_freeze.json").read_text(encoding="utf-8"))
        if (digest(HERE/"temperature_freeze.json") != (HERE/"temperature_freeze.sha256").read_text().strip() or
                tf["g2_freeze_sha256"] != digest(HERE/"g2_freeze.json") or
                tf["candidates_sha256"] != digest(HERE/"temperature_candidates.csv")):
            raise ValueError("Temperature freeze mismatch.")
        rows = load_rows("test")
        cluster_field = "pair_id"
        original_path = ROOT/"artifacts"/"test_predictions.parquet"
        raw = pq.read_table(original_path, filters=[("population", "=", "common"),
                    ("model_id", "in", ["common:M0", "common:M1", "common:M2", "M3-C", "M4-C"])]).to_pylist()
        sources = {}
        for r in raw:
            sources.setdefault(r["model_id"], {})[r["train_row_id"]] = r
        yfield = "outcome"
    else:
        rows = source_rows("G3", "test", False)
        cluster_field = "held_out_photo"
        original_path = ROOT/"g3_experiment"/"test_predictions.parquet"
        provenance = json.loads((ROOT/"g3_experiment"/"evaluation_provenance.json").read_text(encoding="utf-8"))
        if digest(original_path) != provenance["output_sha256"]["test_predictions.parquet"]:
            raise ValueError("G3 saved predictions changed.")
        raw = pq.read_table(original_path).to_pylist()
        sources = {}
        for r in raw:
            if r["model"] in ("M0", "M1", "M3-C", "M4-C"):
                sources.setdefault(r["model"], {})[r["train_row_id"]] = r
        yfield = "outcome_a"
    z = predict(model, rows)
    write_json(guard, {"status": "saved_test_reanalysis_started", "source_freeze_sha256": digest(HERE/f"{label}_freeze.json"),
                       "raw_test_labels_parsed": False, "previously_exposed": True})
    ids = [r["train_row_id"] for r in rows]
    for name, index in sources.items():
        if len(index) != len(ids) or set(index) != set(ids):
            raise ValueError(f"Matched comparator incomplete: {name}")
        if any(index[key]["pair_id"] != row["pair_id"] for key, row in zip(ids, rows)):
            raise ValueError("Matched pair ID differs.")
    y = np.asarray([sources[next(iter(sources))][key][yfield] for key in ids], dtype=float)
    if not np.isin(y, [0,1]).all() or any([sources[name][key][yfield] for key in ids] != y.tolist() for name in sources):
        raise ValueError("Matched frozen outcomes differ.")
    all_logits = {"Cat+CNN": z, **{"original:"+name: np.asarray([index[key]["logit"] for key in ids])
                                   for name,index in sources.items()}}
    if label == "g2":
        all_logits["Cat+CNN-temperature"] = z/tf["temperature"]
    losses = {name: y*np.logaddexp(0., -v)+(1-y)*np.logaddexp(0.,v) for name,v in all_logits.items()}
    for name in sources:
        np.testing.assert_allclose(losses["original:"+name], [sources[name][key]["loss"] for key in ids], rtol=0, atol=1e-12)
    clusters = [row[cluster_field] for row in rows]
    if not all(clusters):
        raise ValueError("Bootstrap cluster missing.")
    samples, cluster_weighted = pair_bootstrap(clusters, losses, seed=20261101, replicates=5000)
    pair_ids = [r["pair_id"] for r in rows]
    pairs = sorted(set(pair_ids))
    mapping = {pair:i for i,pair in enumerate(pairs)}
    codes = np.asarray([mapping[p] for p in pair_ids])
    counts = np.bincount(codes)
    results, predictions = [], []
    for name,value in all_logits.items():
        p = expit(value)
        low, high = np.quantile(samples[name], [.025,.975])
        pair_loss = float(np.mean(np.bincount(codes, weights=losses[name])/counts))
        results.append({"model":name, "responses":len(rows), "pairs":len(pairs), "cluster_unit":cluster_field,
                        "clusters":len(set(clusters)), "log_loss":float(losses[name].mean()), "ci_low":float(low),
                        "ci_high":float(high), "pair_weighted_log_loss":pair_loss,
                        "brier":float(np.mean((p-y)**2)), "accuracy":float(np.mean((p>=.5)==y))})
        predictions.extend({"model":name, "train_row_id":ids[i], "pair_id":pair_ids[i], "cluster":clusters[i],
                            "outcome":int(y[i]), "logit":float(value[i]), "probability":float(p[i]),
                            "loss":float(losses[name][i])} for i in range(len(rows)))
    comparisons = [(name, "Cat+CNN") for name in all_logits if name.startswith("original:")]
    if label == "g2":
        comparisons.append(("Cat+CNN", "Cat+CNN-temperature"))
    contrasts=[]
    for a,b in comparisons:
        d=samples[a]-samples[b]
        lo,hi=np.quantile(d,[.025,.975])
        contrasts.append({"reference":a,"candidate":b,"difference":float(np.mean(losses[a]-losses[b])),
                          "ci_low":float(lo),"ci_high":float(hi),"bootstrap_se":float(np.std(d,ddof=1)),
                          "cluster_unit":cluster_field,"clusters":len(set(clusters))})
    write_csv(HERE/f"{label}_results.csv",results,list(results[0]))
    write_csv(HERE/f"{label}_contrasts.csv",contrasts,list(contrasts[0]))
    pq.write_table(pa.Table.from_pylist(predictions),HERE/f"{label}_predictions.parquet",compression="zstd")
    subgroup,calibration=[],[]
    p=expit(z)
    for group in ("within_cat","between_cats"):
        mask=np.asarray([r["comparison_type"]==group for r in rows])
        if mask.any():
            n=int(mask.sum()); pairs_count=len({r["pair_id"] for r,keep in zip(rows,mask) if keep})
            subgroup.append({"comparison_type":group,"responses":n,"pairs":pairs_count,
                             "support":"supported" if n>=100 and pairs_count>=30 else "low_support",
                             "log_loss":float(losses["Cat+CNN"][mask].mean()),
                             "brier":float(np.mean((p[mask]-y[mask])**2)),"accuracy":float(np.mean((p[mask]>=.5)==y[mask]))})
    bins=np.minimum((p*10).astype(int),9)
    for b in range(10):
        mask=bins==b; n=int(mask.sum()); pc=len({r["pair_id"] for r,keep in zip(rows,mask) if keep})
        calibration.append({"bin":b,"lower":b/10,"upper":(b+1)/10,"responses":n,"pairs":pc,
                            "support":"supported" if n>=30 and pc>=10 else "low_support",
                            "mean_probability":float(p[mask].mean()) if n else "",
                            "observed_a_rate":float(y[mask].mean()) if n else ""})
    write_csv(HERE/f"{label}_subgroups.csv",subgroup,list(subgroup[0]))
    write_csv(HERE/f"{label}_calibration.csv",calibration,list(calibration[0]))
    names=[f"{label}_{s}" for s in ("results.csv","contrasts.csv","predictions.parquet","subgroups.csv","calibration.csv")]
    if label=="g2":
        utils={}
        for name in ("Cat+CNN","original:common:M1","original:M3-C","original:M4-C"):
            utils[name]=sorted([r for r in predictions if r["model"]==name],key=lambda r:r["train_row_id"])
        utility=calculate(utils,[round(.5+.025*i,3) for i in range(21)],20261101,5000)
        write_csv(HERE/"g2_abstention_utility.csv",utility,list(utility[0]))
        names.append("g2_abstention_utility.csv")
    write_json(HERE/f"{label}_evaluation_provenance.json",{"status":"completed_retrospective",
               "model_freeze_sha256":digest(HERE/f"{label}_freeze.json"),"original_predictions_sha256":digest(original_path),
               "outputs_sha256":{name:digest(HERE/name) for name in names},"responses":len(rows),"pairs":len(pairs),
               "cluster_unit":cluster_field,"clusters":len(set(clusters)),"bootstrap_replicates":5000,
               "raw_test_outcomes_read":False,"prior_test_exposure":True,
               "limitations":"Post-test 95% pointwise unadjusted; not independently confirmatory or causal."})
    print(json.dumps({"target":label,"results":results,"contrasts":contrasts},indent=2))


def curves(seed):
    fit,val=load_rows("fit"),load_rows("validation")
    fractions=[.25,.5,.75,1.]
    indices=common_validation_indices(fit,val,[20261030,20261031],fractions)
    common=[val[i] for i in indices]
    if (len(common),len({r["pair_id"] for r in common})) != (1836,589):
        raise ValueError("Original common learning-curve population mismatch.")
    subsets=nested_pair_sets(fit,seed,fractions)
    for fraction in fractions:
        label=f"curve_{seed}_{int(fraction*100):03d}"
        if (HERE/f"{label}_freeze.json").exists():
            load_frozen(label)
            continue
        train(label,[r for r in fit if r["pair_id"] in subsets[fraction]],common)


if __name__ == "__main__":
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","g2","control","temperature","evaluate","g3","curve"))
    parser.add_argument("--target",choices=("g2","g3"))
    parser.add_argument("--seed",type=int,choices=(20261030,20261031))
    args=parser.parse_args()
    if args.stage=="prepare": prepare()
    elif args.stage=="g2": train("g2",load_rows("fit"),load_rows("validation"))
    elif args.stage=="control": control()
    elif args.stage=="temperature": temperature()
    elif args.stage=="evaluate":
        if not args.target: parser.error("--target required")
        evaluate(args.target)
    elif args.stage=="g3": train("g3",source_rows("G3","core_fit",True),source_rows("G3","validation",True))
    else:
        if not args.seed: parser.error("--seed required")
        curves(args.seed)
