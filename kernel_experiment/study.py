"""Quarantined development, controls, transfer and saved-outcome evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit
from threadpoolctl import threadpool_limits

from kernel import HERE, ROOT, designs, fit_map, load_features
sys.path.insert(0, str(ROOT / "demographic_experiment" / "diagnostics_extension"))
from transfer import source_rows  # noqa: E402
from g3_experiment.prepare import digest, snapshot, write_csv, write_json  # noqa: E402
from src.abstention_utility import calculate  # noqa: E402
from src.development import score  # noqa: E402
from src.estimator import fit_logistic  # noqa: E402
from src.evaluate import pair_bootstrap  # noqa: E402
from src.freeze import validate_freeze  # noqa: E402
from src.learning_curves import common_validation_indices, nested_pair_sets  # noqa: E402
from experiment import load_rows  # noqa: E402

EXCLUDED = {".git", ".venv", ".cache", "Presentation_Evidence", "kernel_experiment"}
GRID = (1e-6, 1e-5, 1e-4, .001, .01, .1, 1.)
WIDTHS = (.25, 1., 4.)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def verify_base() -> None:
    validate_freeze(ROOT)
    for folder in ("g3_experiment", "g4_experiment", "demographic_experiment"):
        root = ROOT / folder
        path = root / "freeze.json"
        if digest(path) != (root / "freeze.sha256").read_text(encoding="ascii").strip():
            raise ValueError(f"{folder} freeze mismatch.")
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, value in data.get("experiment_sha256", data.get("source_output_sha256", {})).items():
            if digest(root / name) != value:
                raise ValueError(f"{folder} frozen artifact changed: {name}")
    eval_data = json.loads((ROOT / "artifacts" / "evaluation_provenance.json").read_text(encoding="utf-8"))
    if eval_data["freeze_sha256"] != digest(ROOT / "artifacts" / "model_freeze.json"):
        raise ValueError("G2 evaluation freeze mismatch.")
    for name, value in eval_data["output_sha256"].items():
        if digest(ROOT.joinpath(*name.split("/"))) != value:
            raise ValueError(f"G2 evaluation output changed: {name}")


def prepare() -> None:
    if (HERE / "original_hashes_before.json").exists():
        raise FileExistsError("Study snapshot already exists.")
    verify_base()
    before = snapshot(ROOT, EXCLUDED)
    evidence = snapshot(ROOT / "Presentation_Evidence", set())
    write_json(HERE / "original_hashes_before.json", before)
    write_json(HERE / "evidence_hashes_before.json", evidence)
    print(f"Snapshot: {len(before)} protected project files and {len(evidence)} evidence files.")


def train(label: str, train_rows: list[dict], val_rows: list[dict]) -> None:
    log_path, model_path, freeze_path = (HERE / f"{label}_{suffix}" for suffix in
                                          ("attempts.csv", "model.json", "freeze.json"))
    if any(p.exists() for p in (log_path, model_path, freeze_path, HERE / f"{label}_freeze.sha256")):
        raise FileExistsError(f"Existing {label} development artifacts.")
    verify_base()
    if any(r["y_displayed"] not in (0, 1) for r in train_rows+val_rows):
        raise ValueError("Invalid fitting/validation outcomes.")
    y = np.asarray([r["y_displayed"] for r in train_rows], dtype=float)
    vy = np.asarray([r["y_displayed"] for r in val_rows], dtype=float)
    fit_ids = {r[f"{side}_image"] for r in train_rows for side in ("a", "b")}
    features = load_features()
    candidates = []
    fields = ("width_multiplier", "ridge", "accepted", "validation_loss", "training_loss", "objective",
              "gradient_max", "coefficient_norm", "coefficient_count", "iterations", "seconds", "attempts")
    maps = {}
    with log_path.open("x", encoding="utf-8", newline="") as stream, threadpool_limits(limits=1):
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        stream.flush()
        for width in WIDTHS:
            mapping = fit_map(features, fit_ids, width)
            maps[width] = mapping
            x, vx = designs(train_rows, val_rows, features, mapping)
            for ridge in GRID:
                start = time.monotonic()
                fit = fit_logistic(x, y, np.r_[[1e-6], np.full(x.shape[1]-1, ridge)])
                logits = np.asarray(vx @ fit.coefficients).ravel() if fit.accepted else None
                valid = logits is not None and np.isfinite(logits).all()
                candidate = {"width_multiplier": width, "ridge": ridge, "accepted": bool(fit.accepted and valid),
                             "validation_loss": score(vy, logits)[0] if valid else None,
                             "training_loss": score(y, np.asarray(x @ fit.coefficients).ravel())[0] if valid else None,
                             "objective": json_safe(fit.objective), "gradient_max": json_safe(fit.gradient_max),
                             "coefficient_norm": float(np.linalg.norm(fit.coefficients[1:])) if valid else None,
                             "coefficient_count": x.shape[1], "iterations": fit.iterations,
                             "seconds": time.monotonic()-start, "attempts": json.dumps(json_safe(fit.attempts)),
                             "coefficients": fit.coefficients.tolist()}
                writer.writerow({k: candidate[k] for k in fields})
                stream.flush()
                candidates.append(candidate)
            print(f"{label}: completed kernel width {width:g}", flush=True)
    failed = [r for r in candidates if not r["accepted"]]
    if failed:
        raise RuntimeError(f"{label}: {len(failed)} failed fits; inspect the saved log.")
    best = min(r["validation_loss"] for r in candidates)
    winner = min((r for r in candidates if r["validation_loss"] <= best+1e-6),
                 key=lambda r: (-r["ridge"], r["width_multiplier"]))
    model = {"label": label, "selected": winner, "mapping": maps[winner["width_multiplier"]],
             "fit_responses": len(train_rows), "fit_pairs": len({r["pair_id"] for r in train_rows}),
             "fit_photos": len(fit_ids), "validation_responses": len(val_rows),
             "validation_pairs": len({r["pair_id"] for r in val_rows}),
             "validation_row_ids": [r["train_row_id"] for r in val_rows],
             "fit_row_ids": [r["train_row_id"] for r in train_rows]}
    write_json(model_path, model)
    freeze = {"status": "frozen_before_saved_test_reanalysis", "label": label,
              "protocol_sha256": digest(HERE / "protocol.json"), "model_sha256": digest(model_path),
              "attempts_sha256": digest(log_path), "source_sha256": {
                  "kernel.py": digest(HERE / "kernel.py"), "study.py": digest(HERE / "study.py")},
              "g2_freeze_sha256": digest(ROOT / "artifacts" / "model_freeze.json"),
              "source_C_sha256": digest(ROOT / "features" / "C.parquet"),
              "prior_outcome_exposure": True}
    write_json(freeze_path, freeze)
    (HERE / f"{label}_freeze.sha256").write_text(digest(freeze_path) + "\n", encoding="ascii")
    print(f"{label}: selected width={winner['width_multiplier']} ridge={winner['ridge']} validation={best:.9f}", flush=True)


def load_frozen(label: str) -> dict:
    path = HERE / f"{label}_freeze.json"
    if digest(path) != (HERE / f"{label}_freeze.sha256").read_text(encoding="ascii").strip():
        raise ValueError(f"{label} freeze seal mismatch.")
    record = json.loads(path.read_text(encoding="utf-8"))
    for path, key in ((HERE / "protocol.json", "protocol_sha256"),
                      (HERE / f"{label}_model.json", "model_sha256"),
                      (HERE / f"{label}_attempts.csv", "attempts_sha256"),
                      (ROOT / "features" / "C.parquet", "source_C_sha256"),
                      (ROOT / "artifacts" / "model_freeze.json", "g2_freeze_sha256")):
        if digest(path) != record[key]:
            raise ValueError(f"{label} frozen source changed: {path.name}")
    for name, value in record["source_sha256"].items():
        if digest(HERE / name) != value:
            raise ValueError(f"{label} code changed: {name}")
    return json.loads((HERE / f"{label}_model.json").read_text(encoding="utf-8"))


def logits(model: dict, rows: list[dict]) -> np.ndarray:
    features = load_features()
    fit_stub = [{"a_image": model["mapping"]["fitting_photos"][0],
                 "b_image": model["mapping"]["fitting_photos"][1]}]
    _, x = designs(fit_stub, rows, features, model["mapping"])
    result = np.asarray(x @ np.asarray(model["selected"]["coefficients"])).ravel()
    if result.shape != (len(rows),) or not np.isfinite(result).all():
        raise ValueError("Incomplete kernel predictions.")
    return result


def run_control() -> None:
    target, proof = HERE / "fair_coin_control.csv", HERE / "fair_coin_provenance.json"
    if target.exists() or proof.exists():
        raise FileExistsError("Control already exists.")
    model = load_frozen("g2")
    train_rows, val_rows = load_rows("fit"), load_rows("validation")
    x, vx = designs(train_rows, val_rows, load_features(), model["mapping"])
    ridge = model["selected"]["ridge"]
    penalty = np.r_[[1e-6], np.full(x.shape[1]-1, ridge)]
    rng = np.random.default_rng(20261102)
    results = []
    with target.open("x", encoding="utf-8", newline="") as stream, threadpool_limits(limits=1):
        writer = csv.DictWriter(stream, fieldnames=["rep", "accepted", "validation_loss", "improvement", "gradient_max", "attempts"])
        writer.writeheader()
        for rep in range(20):
            y = rng.integers(0, 2, len(train_rows)).astype(float)
            vy = rng.integers(0, 2, len(val_rows)).astype(float)
            fitted = fit_logistic(x, y, penalty)
            loss = score(vy, np.asarray(vx @ fitted.coefficients).ravel())[0] if fitted.accepted else None
            row = {"rep": rep+1, "accepted": fitted.accepted, "validation_loss": loss,
                   "improvement": float(np.log(2)-loss) if loss is not None else None,
                   "gradient_max": json_safe(fitted.gradient_max), "attempts": json.dumps(json_safe(fitted.attempts))}
            writer.writerow(row)
            stream.flush()
            results.append(row)
    failures = sum(not r["accepted"] for r in results)
    improvements = sum(r["improvement"] is not None and r["improvement"] > 0 for r in results)
    status = "passed_operational_control" if failures == 0 and improvements < 16 else "blocked_investigation_required"
    write_json(proof, {"status": status, "seed": 20261102, "repetitions": 20, "failed_fits": failures,
                       "improvements": improvements, "mean_loss": float(np.mean([r["validation_loss"] for r in results if r["accepted"]])),
                       "selected_fixed_width_and_penalty": True, "source_freeze_sha256": digest(HERE/"g2_freeze.json"),
                       "results_sha256": digest(target), "note": "Operational control, not a leakage-free certificate."})
    if status != "passed_operational_control":
        raise RuntimeError("Fair-coin control requires investigation.")
    print(f"Fair-coin: {improvements}/20 improvements, {failures} failures.")


def temperature_select() -> None:
    path, seal = HERE/"temperature_freeze.json", HERE/"temperature_freeze.sha256"
    if path.exists() or seal.exists() or (HERE/"temperature_candidates.csv").exists():
        raise FileExistsError("Temperature selection already exists.")
    model, val = load_frozen("g2"), load_rows("validation")
    z = logits(model, val)
    y = np.asarray([r["y_displayed"] for r in val], dtype=float)
    grid = (.25, .5, .75, 1., 1.5, 2., 4.)
    choices = [(t, score(y, z/t)[0]) for t in grid]
    chosen = min(choices, key=lambda x: (x[1], abs(x[0]-1), x[0]))[0]
    records = [{"temperature": t, "validation_loss": loss, "selected": t == chosen} for t, loss in choices]
    write_csv(HERE/"temperature_candidates.csv", records, list(records[0]))
    write_json(path, {"temperature": chosen, "selection": "exact minimum; tie nearest 1 then smaller temperature",
                      "source_freeze_sha256": digest(HERE/"g2_freeze.json"),
                      "candidates_sha256": digest(HERE/"temperature_candidates.csv"),
                      "previous_test_exposure": True})
    seal.write_text(digest(path)+"\n", encoding="ascii")
    print(f"Validation-selected temperature: {chosen}")


def evaluate(label: str) -> None:
    model = load_frozen(label)
    guard = HERE/f"{label}_evaluation_started.json"
    if guard.exists() or (HERE/f"{label}_results.csv").exists():
        raise FileExistsError(f"{label} evaluation already started.")
    if label == "g2":
        control = json.loads((HERE/"fair_coin_provenance.json").read_text(encoding="utf-8"))
        if control["status"] != "passed_operational_control" or control["results_sha256"] != digest(HERE/"fair_coin_control.csv"):
            raise ValueError("Fair-coin control not passed or changed.")
        temperature = json.loads((HERE/"temperature_freeze.json").read_text(encoding="utf-8"))
        if digest(HERE/"temperature_freeze.json") != (HERE/"temperature_freeze.sha256").read_text().strip():
            raise ValueError("Temperature freeze mismatch.")
        if temperature["source_freeze_sha256"] != digest(HERE/"g2_freeze.json") or temperature["candidates_sha256"] != digest(HERE/"temperature_candidates.csv"):
            raise ValueError("Temperature frozen input mismatch.")
        rows = load_rows("test")
        cluster_field = "pair_id"
        original_path = ROOT/"artifacts"/"test_predictions.parquet"
        raw = pq.read_table(original_path, filters=[("population", "=", "common"),
                           ("model_id", "in", ["common:M0", "common:M1", "common:M2", "M3-C", "M4-C"])]).to_pylist()
        old = {}
        for r in raw:
            old.setdefault(r["model_id"], {})[r["train_row_id"]] = r
        yfield, zfield = "outcome", "logit"
    else:
        parent = ROOT/(label+"_experiment")
        rows = source_rows(label.upper(), "test", False)
        cluster_field = "held_out_photo" if label == "g3" else "held_out_cat"
        original_path = parent/"test_predictions.parquet"
        provenance = json.loads((parent/"evaluation_provenance.json").read_text(encoding="utf-8"))
        if digest(original_path) != provenance["output_sha256"]["test_predictions.parquet"]:
            raise ValueError("Original transfer saved predictions changed.")
        raw = pq.read_table(original_path).to_pylist()
        old = {}
        keep = ("M0", "M1", "M3-C", "M4-C") if label == "g3" else ("M0", "M1-zero", "M3-C", "M4-C")
        for r in raw:
            if r["model"] in keep:
                old.setdefault(r["model"], {})[r["train_row_id"]] = r
        yfield, zfield = "outcome_a", "logit"
    z = logits(model, rows)
    write_json(guard, {"status": "saved_test_reanalysis_started", "source_freeze_sha256": digest(HERE/f"{label}_freeze.json"),
                       "raw_test_labels_parsed": False, "previously_exposed": True})
    ids = [r["train_row_id"] for r in rows]
    for name, index in old.items():
        if len(index) != len(ids) or set(index) != set(ids):
            raise ValueError(f"Original matched comparator missing IDs: {name}")
        if any(index[key]["pair_id"] != row["pair_id"] for key, row in zip(ids, rows)):
            raise ValueError(f"Original matched comparator pairs differ: {name}")
    first = next(iter(old))
    y = np.asarray([old[first][key][yfield] for key in ids], dtype=float)
    if not np.isin(y, [0, 1]).all() or any(
        [old[name][key][yfield] for key in ids] != y.tolist() for name in old):
        raise ValueError("Saved comparator outcomes differ.")
    all_logits = {"kernel_M3-C": z}
    all_logits.update({f"original:{name}": np.asarray([idx[key][zfield] for key in ids]) for name, idx in old.items()})
    if label == "g2":
        all_logits["kernel_temperature"] = z / temperature["temperature"]
    losses = {name: y*np.logaddexp(0., -value)+(1-y)*np.logaddexp(0., value) for name, value in all_logits.items()}
    for name in old:
        np.testing.assert_allclose(losses["original:"+name], [old[name][key]["loss"] for key in ids], rtol=0, atol=1e-12)
    clusters = [row[cluster_field] for row in rows]
    if not all(clusters):
        raise ValueError("Missing bootstrap cluster.")
    samples, cluster_weighted = pair_bootstrap(clusters, losses, seed=20261101, replicates=5000)
    result, prediction = [], []
    for name, value in all_logits.items():
        probability = expit(value)
        low, high = np.quantile(samples[name], [.025, .975])
        result.append({"model": name, "responses": len(rows), "pairs": len({r["pair_id"] for r in rows}),
                       "clusters": len(set(clusters)), "cluster_unit": cluster_field,
                       "log_loss": float(losses[name].mean()), "ci_low": float(low), "ci_high": float(high),
                       "brier": float(np.mean((probability-y)**2)), "accuracy": float(np.mean((probability >= .5) == y))})
        prediction.extend({"model": name, "train_row_id": ids[i], "pair_id": rows[i]["pair_id"], "cluster": clusters[i],
                           "outcome": int(y[i]), "logit": float(value[i]), "probability": float(probability[i]),
                           "loss": float(losses[name][i])} for i in range(len(rows)))
    comparisons = [(name, "kernel_M3-C") for name in all_logits if name.startswith("original:")]
    if label == "g2":
        comparisons.append(("kernel_M3-C", "kernel_temperature"))
    contrasts = []
    for a, b in comparisons:
        delta = samples[a] - samples[b]
        low, high = np.quantile(delta, [.025, .975])
        contrasts.append({"reference": a, "candidate": b, "difference": float(np.mean(losses[a]-losses[b])),
                          "ci_low": float(low), "ci_high": float(high), "bootstrap_se": float(np.std(delta, ddof=1)),
                          "clusters": len(set(clusters)), "cluster_unit": cluster_field})
    write_csv(HERE/f"{label}_results.csv", result, list(result[0]))
    write_csv(HERE/f"{label}_contrasts.csv", contrasts, list(contrasts[0]))
    pq.write_table(pa.Table.from_pylist(prediction), HERE/f"{label}_predictions.parquet", compression="zstd")
    subgroups, bins = [], []
    for group in ("within_cat", "between_cats"):
        mask = np.asarray([r["comparison_type"] == group for r in rows])
        n, pairs = int(mask.sum()), len({r["pair_id"] for r, keep in zip(rows, mask) if keep})
        if n:
            p = expit(z[mask])
            subgroups.append({"comparison_type": group, "responses": n, "pairs": pairs,
                              "support": "supported" if n >= 100 and pairs >= 30 else "low_support",
                              "log_loss": float(losses["kernel_M3-C"][mask].mean()),
                              "brier": float(np.mean((p-y[mask])**2)), "accuracy": float(np.mean((p >= .5) == y[mask]))})
    probability = expit(z)
    bins_idx = np.minimum((probability*10).astype(int), 9)
    for b in range(10):
        mask = bins_idx == b
        n, pairs = int(mask.sum()), len({r["pair_id"] for r, keep in zip(rows, mask) if keep})
        bins.append({"bin": b, "lower": b/10, "upper": (b+1)/10, "responses": n, "pairs": pairs,
                     "support": "supported" if n >= 30 and pairs >= 10 else "low_support",
                     "mean_probability": float(probability[mask].mean()) if n else "",
                     "observed_a_rate": float(y[mask].mean()) if n else ""})
    write_csv(HERE/f"{label}_subgroups.csv", subgroups, list(subgroups[0]))
    write_csv(HERE/f"{label}_calibration.csv", bins, list(bins[0]))
    if label == "g2":
        inputs = {}
        for name in ("kernel_M3-C", "original:M3-C", "original:M4-C"):
            inputs[name] = sorted([r for r in prediction if r["model"] == name], key=lambda r: r["train_row_id"])
        utility = calculate(inputs, [round(.5+.025*i, 3) for i in range(21)], 20261101, 5000)
        write_csv(HERE/"g2_abstention_utility.csv", utility, list(utility[0]))
    names = [f"{label}_{suffix}" for suffix in ("results.csv", "contrasts.csv", "predictions.parquet", "subgroups.csv", "calibration.csv")]
    if label == "g2":
        names.append("g2_abstention_utility.csv")
    write_json(HERE/f"{label}_evaluation_provenance.json", {
        "status": "completed_retrospective", "model_freeze_sha256": digest(HERE/f"{label}_freeze.json"),
        "original_saved_predictions_sha256": digest(original_path), "outputs_sha256": {name: digest(HERE/name) for name in names},
        "responses": len(rows), "pairs": len({r["pair_id"] for r in rows}), "cluster_unit": cluster_field,
        "clusters": len(set(clusters)), "bootstrap_replicates": 5000,
        "raw_test_outcomes_read": False, "prior_test_exposure": True,
        "interpretation": "Post-test, 95% pointwise unadjusted; not independently confirmatory or causal."})
    print(json.dumps({"target": label, "results": result, "contrasts": contrasts}, indent=2))


def curves(seed: int) -> None:
    fit, val = load_rows("fit"), load_rows("validation")
    fractions = [.25, .5, .75, 1.]
    indices = common_validation_indices(fit, val, [20261030, 20261031], fractions)
    subset = [val[i] for i in indices]
    if (len(subset), len({r["pair_id"] for r in subset})) != (1836, 589):
        raise ValueError("Learning-curve validation population differs from original.")
    sets = nested_pair_sets(fit, seed, fractions)
    for fraction in fractions:
        label = f"curve_{seed}_{int(fraction*100):03d}"
        if (HERE/f"{label}_freeze.json").exists():
            load_frozen(label)
            continue
        fitting = [r for r in fit if r["pair_id"] in sets[fraction]]
        train(label, fitting, subset)


if __name__ == "__main__":
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "g2", "control", "temperature", "evaluate", "transfer", "curve"))
    parser.add_argument("--target", choices=("g2", "g3", "g4"))
    parser.add_argument("--seed", type=int, choices=(20261030, 20261031))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "g2":
        train("g2", load_rows("fit"), load_rows("validation"))
    elif args.stage == "control":
        run_control()
    elif args.stage == "temperature":
        temperature_select()
    elif args.stage == "evaluate":
        if not args.target:
            parser.error("--target is required")
        evaluate(args.target)
    elif args.stage == "transfer":
        if args.target not in ("g3", "g4"):
            parser.error("--target g3 or g4 is required")
        train(args.target, source_rows(args.target.upper(), "core_fit", True),
              source_rows(args.target.upper(), "validation", True))
    else:
        if not args.seed:
            parser.error("--seed is required")
        curves(args.seed)
