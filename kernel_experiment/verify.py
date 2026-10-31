"""Reconstruct selected kernel fits and independently replay saved-test intervals."""

import csv
import json

import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from kernel import designs, fit_map, load_features
from study import HERE, ROOT, GRID, WIDTHS, digest, load_frozen, load_rows, logits, source_rows, verify_base, write_json
from src.abstention_utility import calculate
from src.development import score
from src.estimator import objective_gradient
from src.learning_curves import common_validation_indices, nested_pair_sets


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check_fit(label, train, val, features):
    model = load_frozen(label)
    if model["fit_row_ids"] != [r["train_row_id"] for r in train] or model["validation_row_ids"] != [r["train_row_id"] for r in val]:
        raise ValueError(f"{label} fit/validation identity differs.")
    fit_ids = {r[f"{side}_image"] for r in train for side in ("a", "b")}
    if set(model["mapping"]["landmark_ids"]) - fit_ids:
        raise ValueError(f"{label} includes held-out landmarks.")
    mapping = fit_map(features, fit_ids, model["selected"]["width_multiplier"])
    existing = model["mapping"]
    for key in ("cnn_preprocessing", "fitting_photos", "landmark_ids", "gamma", "distance_median_squared",
                "multiplier", "eigenvalues_clipped"):
        if mapping[key] != existing[key]:
            raise ValueError(f"{label} kernel fitting-only mapping mismatch: {key}")
    for key in ("landmark_cnn_vectors", "normalizer", "center"):
        np.testing.assert_allclose(mapping[key], existing[key], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(mapping["rms"], existing["rms"], rtol=1e-9, atol=1e-12)
    x, vx = designs(train, val, features, existing)
    theta = np.asarray(model["selected"]["coefficients"])
    y = np.asarray([r["y_displayed"] for r in train])
    vy = np.asarray([r["y_displayed"] for r in val])
    objective, gradient = objective_gradient(theta, x, y, np.ones(len(y)),
                                              np.r_[[1e-6], np.full(x.shape[1]-1, model["selected"]["ridge"])])
    val_loss = score(vy, np.asarray(vx @ theta).ravel())[0]
    np.testing.assert_allclose([objective, val_loss], [model["selected"]["objective"], model["selected"]["validation_loss"]],
                               rtol=0, atol=1e-12)
    if np.max(np.abs(gradient)) >= 1e-5:
        raise ValueError(f"{label} selected fit failed gradient acceptance.")
    rows = read(HERE/f"{label}_attempts.csv")
    if len(rows) != 21 or any(r["accepted"] != "True" for r in rows):
        raise ValueError(f"{label} candidate log incomplete.")
    if {(float(r["width_multiplier"]), float(r["ridge"])) for r in rows} != {(w, l) for w in WIDTHS for l in GRID}:
        raise ValueError(f"{label} candidate grid mismatch.")
    minimum = min(float(r["validation_loss"]) for r in rows)
    winner = min((r for r in rows if float(r["validation_loss"]) <= minimum+1e-6),
                 key=lambda r: (-float(r["ridge"]), float(r["width_multiplier"])))
    if (float(winner["ridge"]), float(winner["width_multiplier"])) != (
            model["selected"]["ridge"], model["selected"]["width_multiplier"]):
        raise ValueError(f"{label} validation selection differs.")
    return {"model": label, "fit_rows": len(train), "validation_rows": len(val),
            "gradient_max": float(np.max(np.abs(gradient))), "validation_replay_delta": abs(val_loss-model["selected"]["validation_loss"])}


def check_test(label):
    proof = json.loads((HERE/f"{label}_evaluation_provenance.json").read_text(encoding="utf-8"))
    for name, value in proof["outputs_sha256"].items():
        if digest(HERE/name) != value:
            raise ValueError(f"{label} saved output mismatch: {name}")
    rows = load_rows("test") if label == "g2" else source_rows(label.upper(), "test", False)
    model = load_frozen(label)
    reconstructed = logits(model, rows)
    predictions = pq.read_table(HERE/f"{label}_predictions.parquet").to_pylist()
    result = {r["model"]: r for r in read(HERE/f"{label}_results.csv")}
    contrasts = read(HERE/f"{label}_contrasts.csv")
    ids = [r["train_row_id"] for r in rows]
    names = list(result)
    losses, identity = {}, None
    for name in names:
        subset = [r for r in predictions if r["model"] == name]
        if [r["train_row_id"] for r in subset] != ids:
            raise ValueError(f"{label} matched IDs differ.")
        values = [(r["outcome"], r["cluster"], r["pair_id"]) for r in subset]
        if identity is not None and values != identity:
            raise ValueError("Matched outcomes/clusters differ.")
        identity = values
        y = np.asarray([r["outcome"] for r in subset])
        z = np.asarray([r["logit"] for r in subset])
        if name == "kernel_M3-C":
            np.testing.assert_allclose(z, reconstructed, atol=1e-12, rtol=0)
        loss = y*np.logaddexp(0., -z)+(1-y)*np.logaddexp(0., z)
        np.testing.assert_allclose(loss, [r["loss"] for r in subset], atol=1e-12, rtol=0)
        np.testing.assert_allclose(loss.mean(), float(result[name]["log_loss"]), atol=1e-12, rtol=0)
        losses[name] = loss
    clusters = [r["cluster"] for r in predictions if r["model"] == "kernel_M3-C"]
    unique = sorted(set(clusters))
    mapping = {key: i for i, key in enumerate(unique)}
    codes = np.asarray([mapping[v] for v in clusters])
    counts = np.bincount(codes)
    sums = np.stack([np.bincount(codes, weights=losses[name]) for name in names])
    rng = np.random.default_rng(20261101)
    samples = np.empty((5000, len(names)))
    for i in range(5000):
        weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        samples[i] = sums @ weights / (counts @ weights)
    for i, name in enumerate(names):
        np.testing.assert_allclose(np.quantile(samples[:, i], [.025, .975]),
                                   [float(result[name]["ci_low"]), float(result[name]["ci_high"])], atol=1e-12, rtol=0)
    for row in contrasts:
        a, b = names.index(row["reference"]), names.index(row["candidate"])
        delta = samples[:, a]-samples[:, b]
        np.testing.assert_allclose([np.mean(losses[names[a]]-losses[names[b]]), *np.quantile(delta, [.025, .975])],
                                   [float(row["difference"]), float(row["ci_low"]), float(row["ci_high"])], atol=1e-12, rtol=0)
    return {"target": label, "responses": len(ids), "models": len(names), "contrasts": len(contrasts),
            "clusters": len(unique), "bootstrap_replicates": 5000}


def main():
    if (HERE/"verification.json").exists():
        raise FileExistsError("Verification already exists.")
    verify_base()
    fit, val = load_rows("fit"), load_rows("validation")
    features = load_features()
    checks = []
    with threadpool_limits(limits=1):
        checks.append(check_fit("g2", fit, val, features))
        for label in ("g3", "g4"):
            checks.append(check_fit(label, source_rows(label.upper(), "core_fit", True),
                                    source_rows(label.upper(), "validation", True), features))
        indices = common_validation_indices(fit, val, [20261030, 20261031], [.25, .5, .75, 1.])
        common_val = [val[i] for i in indices]
        for seed in (20261030, 20261031):
            subsets = nested_pair_sets(fit, seed, [.25, .5, .75, 1.])
            for fraction in (.25, .5, .75, 1.):
                train = [r for r in fit if r["pair_id"] in subsets[fraction]]
                checks.append(check_fit(f"curve_{seed}_{int(fraction*100):03d}", train, common_val, features))
        tests = [check_test(label) for label in ("g2", "g3", "g4")]
        table = pq.read_table(HERE/"g2_predictions.parquet").to_pylist()
        names = ("kernel_M3-C", "original:M3-C", "original:M4-C")
        input_models = {name: sorted([r for r in table if r["model"] == name], key=lambda r: r["train_row_id"]) for name in names}
        calculated = calculate(input_models, [round(.5+.025*i, 3) for i in range(21)], 20261101, 5000)
        saved = read(HERE/"g2_abstention_utility.csv")
        if len(saved) != len(calculated):
            raise ValueError("Abstention grid incomplete.")
        keys = ("coverage", "utility_per_original_response", "utility_ci95_low", "utility_ci95_high",
                "coverage_ci95_low", "coverage_ci95_high", "undefined_selective_accuracy_bootstrap_replicates")
        for expected, actual in zip(saved, calculated):
            if expected["model"] != actual["model"] or float(expected["threshold"]) != actual["threshold"]:
                raise ValueError("Abstention grid alignment differs.")
            np.testing.assert_allclose([float(expected[k]) for k in keys], [float(actual[k]) for k in keys],
                                       rtol=0, atol=1e-12)
    report = {"status": "passed", "selected_fits": checks, "test_replays": tests,
              "abstention_cells_replayed": len(calculated), "raw_test_outcomes_read": False,
              "note": "Recorded previously exposed outcomes remain exploratory; checks are numerical, not independent validation."}
    write_json(HERE/"verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
