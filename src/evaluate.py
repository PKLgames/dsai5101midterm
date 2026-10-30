"""Frozen Stage 9 internal-test evaluation and pair-cluster uncertainty."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit

from .development import _hash, feature_differences, group_name, identity_matrix, score, visual_design
from .freeze import validate_freeze
from .preprocessing import transform_photos


def load_test_structure(root: Path) -> list[dict[str, Any]]:
    fields = ["train_row_id", "partition", "pair_id", "a_image", "b_image", "a_cat", "b_cat",
              "comparison_type", "pet_care", "orientation"]
    rows = pq.read_table(root / "artifacts/rows_canonical.parquet", columns=fields,
                         filters=[("partition", "=", "test")]).to_pylist()
    masks = pq.read_table(root / "artifacts/g2_eligibility.parquet",
                          columns=["train_row_id", "partition", "full_strict_g2", "common_strict_g2"],
                          filters=[("partition", "=", "test")]).to_pylist()
    index = {row["train_row_id"]: row for row in masks}
    if not rows or len(index) != len(masks) or len(index) != len(rows):
        raise ValueError("Internal-test structure or masks are empty, duplicate, or inconsistent.")
    for row in rows:
        if row["partition"] != "test" or row["train_row_id"] not in index:
            raise ValueError("Invalid internal-test structural row.")
        mask = index[row["train_row_id"]]
        if mask["partition"] != "test":
            raise ValueError("Internal-test mask partition mismatch.")
        row["full_strict_g2"] = mask["full_strict_g2"]
        row["common_strict_g2"] = mask["common_strict_g2"]
    return rows


def _record_test_access(root: Path) -> None:
    path = root / "artifacts/outcome_access_log.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if fields is None or any(row["entry_id"] == "A14" for row in rows):
        raise ValueError("Stage 9 test-outcome access is already recorded; do not evaluate twice.")
    entry = {
        "entry_id": "A14", "date": datetime.now().date().isoformat(), "source": "Stage 9 frozen internal test",
        "partition_or_material": "internal-test training outcomes",
        "access_or_evidence": "Loaded held-out outcomes by response ID after verified model freeze",
        "actor": "assistant", "status": "documented",
        "limitations": "Single frozen internal holdout; previous analysis exposure and dependence remain limitations",
    }
    with path.open("a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=fields, lineterminator="\n").writerow(entry)
        stream.flush()


def load_outcomes(root: Path, rows: list[dict[str, Any]], expected_hash: str) -> dict[str, int]:
    path = root / "dataset/train.csv"
    if _hash(path) != expected_hash:
        raise ValueError("Raw outcome source differs from the frozen hash.")
    wanted = {row["train_row_id"] for row in rows}
    if len(wanted) != len(rows):
        raise ValueError("Duplicate internal-test IDs.")
    outcomes: dict[str, int] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["train_row_id"]
            if key in wanted:
                if key in outcomes or row["choice"] not in {"A", "B"}:
                    raise ValueError(f"Duplicate or invalid internal-test outcome: {key}")
                outcomes[key] = int(row["choice"] == "A")
    if set(outcomes) != wanted or _hash(path) != expected_hash:
        raise ValueError("Internal-test outcomes are incomplete or source changed while reading.")
    return outcomes


def test_logits(
    root: Path, frozen: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[tuple[str, str], tuple[list[dict[str, Any]], np.ndarray]]:
    selection = json.loads((root / "artifacts/model_selection.json").read_text(encoding="utf-8"))
    selected = selection["selected"]
    populations = {
        "full": [row for row in rows if row["full_strict_g2"]],
        "common": [row for row in rows if row["common_strict_g2"]],
    }
    if not populations["full"] or not populations["common"]:
        raise ValueError("Full or common strict-G2 test population is empty.")
    groups = [group_name(row) for row in populations["common"]]
    interactions = tuple(frozen["context_interactions"])
    matrices = {}
    for block in ("H", "C", "Y"):
        all_rows = pq.read_table(root / "features" / f"{block}.parquet").to_pylist()
        by_id = {row["image_id"]: row for row in all_rows}
        if len(by_id) != len(all_rows):
            raise ValueError(f"Duplicate {block} feature IDs.")
        ids = sorted({row[f"{side}_image"] for row in populations["common"] for side in ("a", "b")})
        if not set(ids) <= set(by_id) or any(by_id[image]["extraction_status"] != "ok" for image in ids):
            raise ValueError(f"{block} features are missing on the frozen common test set.")
        spec = json.loads((root / "artifacts/preprocessing" / f"{block}.json").read_text(encoding="utf-8"))
        vectors = transform_photos(spec, [by_id[image] for image in ids])
        matrices[block] = feature_differences(populations["common"], ids, vectors)
    result: dict[tuple[str, str], tuple[list[dict[str, Any]], np.ndarray]] = {}
    for key in frozen["selected_model_keys"]:
        model = selected[key]
        population = "full" if key.startswith("full:") else "common"
        subset = populations[population]
        theta = np.asarray(model["coefficients"], dtype=np.float64)
        if not model["accepted"] or not np.isfinite(theta).all():
            raise ValueError(f"Frozen model is not numerically accepted: {key}")
        if key.endswith("M0"):
            logits = np.full(len(subset), theta[0])
        elif key.endswith("M1") or key.endswith("M2") or key.endswith("M2-no-position"):
            entity = "cat" if key.endswith("M1") else "image"
            design = identity_matrix(subset, model["identity_keys"], entity, not key.endswith("M2-no-position"))
            logits = np.asarray(design @ theta).reshape(-1)
        else:
            if model["pooled"]:
                source = model["pooled_from"].removeprefix("common:")
                if (population, source) not in result:
                    raise ValueError(f"Exact pooled source must be evaluated first: {key}")
                logits = result[(population, source)][1].copy()
            else:
                block = frozen["selected_representation"] if "side" in key else key[-1]
                contextual = key.startswith("M4")
                design = visual_design(matrices[block], groups, interactions if contextual else (),
                                       interactions if "side" in key else ())
                logits = np.asarray(design @ theta).reshape(-1)
        if logits.shape != (len(subset),) or not np.isfinite(logits).all():
            raise ValueError(f"Missing or nonfinite frozen predictions: {key}")
        result[(population, key)] = (subset, logits)
    for label, model in (("M2", "common:M2"), ("M3*", f"M3-{frozen['selected_representation']}"),
                         ("M4*", f"M4-{frozen['selected_representation']}")):
        rows_, logits = result[("common", model)]
        temperature = frozen["temperature_diagnostics"][label]
        if temperature <= 0 or not math.isfinite(temperature):
            raise ValueError(f"Invalid pre-frozen temperature: {label}")
        result[("common", f"{label}-temperature")] = (rows_, logits / temperature)
    return result


def pair_bootstrap(
    pair_ids: list[str], losses: dict[str, np.ndarray], seed: int, replicates: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not pair_ids or replicates < 2:
        raise ValueError("Bootstrap requires pairs and at least two resamples.")
    pairs = sorted(set(pair_ids))
    mapping = {pair: index for index, pair in enumerate(pairs)}
    codes = np.asarray([mapping[pair] for pair in pair_ids], dtype=np.int64)
    counts = np.bincount(codes, minlength=len(pairs)).astype(np.float64)
    models = sorted(losses)
    sums = []
    for model in models:
        values = np.asarray(losses[model], dtype=np.float64)
        if values.shape != codes.shape or not np.isfinite(values).all():
            raise ValueError(f"Bootstrap losses are incomplete or nonfinite: {model}")
        sums.append(np.bincount(codes, weights=values, minlength=len(pairs)))
    sums_array = np.stack(sums)
    means = sums_array / counts
    response = np.empty((replicates, len(models)))
    pair_weighted = np.empty_like(response)
    rng = np.random.default_rng(seed)
    for i in range(replicates):
        sample = rng.integers(0, len(pairs), size=len(pairs))
        multiplicity = np.bincount(sample, minlength=len(pairs))
        response[i] = sums_array @ multiplicity / (counts @ multiplicity)
        pair_weighted[i] = means @ multiplicity / len(pairs)
    return ({model: response[:, i] for i, model in enumerate(models)},
            {model: pair_weighted[:, i] for i, model in enumerate(models)})


def interval(values: np.ndarray, confidence: float) -> tuple[float, float, float]:
    if not 0 < confidence < 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Invalid bootstrap interval inputs.")
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(values, [alpha, 1-alpha])
    return float(lower), float(upper), float(np.std(values, ddof=1))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty evaluation table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _feature_flags(root: Path) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, Any]]]:
    nodes = pq.read_table(root / "artifacts/graph_components.parquet").to_pylist()
    opponent = {
        population: {row["node_id"]: row["distinct_opponents"] for row in nodes
                     if row["population"] == population and row["entity"] == "photo"}
        for population in ("full", "common")
    }
    y = pq.read_table(root / "features/Y.parquet",
                      columns=["image_id", "extraction_status", "detection_status", "candidate_cat_count"]).to_pylist()
    quality = {row["image_id"]: row for row in y}
    if len(quality) != len(y):
        raise ValueError("Duplicate Y feature-quality IDs.")
    return opponent, quality


def subgroup_labels(row: dict[str, Any]) -> list[tuple[str, str]]:
    support = row["min_endpoint_opponents"]
    band = "not_available" if support < 0 else "1" if support <= 1 else "2-4" if support <= 4 else "5-9" if support <= 9 else "10+"
    labels = [("overall", "all"), ("comparison_type", row["comparison_type"]),
              ("pet_care", row["pet_care"]), ("minimum_endpoint_opponents", band)]
    for name in ("any_no_detection", "any_dog_fallback", "any_multiple_cats", "both_cat_detected"):
        if row[name]:
            labels.append(("feature_quality_overlapping", name))
    return labels


def evaluate(root: Path) -> dict[str, Any]:
    frozen = validate_freeze(root)
    outputs = [root / name for name in (
        "artifacts/test_predictions.parquet", "tables/main_results.csv", "tables/primary_contrasts.csv",
        "tables/secondary_results.csv", "tables/subgroup_results.csv", "tables/test_calibration.csv",
        "figures/calibration.pdf", "artifacts/evaluation_provenance.json",
    )]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Internal-test outputs already exist; refusing to re-evaluate.")
    rows = load_test_structure(root)
    logits_by_model = test_logits(root, frozen, rows)
    needed = {("common", item[name]) for item in frozen["primary_contrasts"] for name in ("reference", "candidate")}
    needed |= {("common", f"M{level}-{block}") for level in (3, 4) for block in ("H", "C", "Y")}
    needed |= {("common", name) for name in ("common:M2-no-position", "M3-side", "M4-side",
                                             "M2-temperature", "M3*-temperature", "M4*-temperature")}
    if not needed <= set(logits_by_model):
        raise ValueError("A required frozen comparison is missing before test outcome access.")
    opponents, quality = _feature_flags(root)
    _record_test_access(root)
    outcomes = load_outcomes(root, rows, frozen["raw_training_sha256"])
    selection = json.loads((root / "artifacts/model_selection.json").read_text(encoding="utf-8"))
    with (root / "tables/model_resources.csv").open(encoding="utf-8", newline="") as stream:
        resources = {row["model"]: row for row in csv.DictReader(stream)}
    predictions: list[dict[str, Any]] = []
    losses_by_population: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    population_rows: dict[str, list[dict[str, Any]]] = {}
    for (population, model), (subset, logits) in logits_by_model.items():
        if population in population_rows and [r["train_row_id"] for r in population_rows[population]] != [
            r["train_row_id"] for r in subset
        ]:
            raise ValueError(f"Matched models have inconsistent test row IDs: {model}")
        population_rows[population] = subset
        y = np.asarray([outcomes[row["train_row_id"]] for row in subset], dtype=np.float64)
        probabilities = expit(logits)
        losses = y * np.logaddexp(0., -logits) + (1.-y)*np.logaddexp(0., logits)
        if not np.isfinite(losses).all():
            raise ValueError(f"Nonfinite test losses: {model}")
        losses_by_population[population][model] = losses
        for row, outcome, logit, probability, loss in zip(subset, y, logits, probabilities, losses):
            a, b = row["a_image"], row["b_image"]
            # Full identity rows can lack common support in future datasets.
            support = (min(opponents[population][a], opponents[population][b])
                       if a in opponents[population] and b in opponents[population] else -1)
            qa, qb = quality.get(a), quality.get(b)
            detection = [entry["detection_status"] for entry in (qa, qb)
                         if entry is not None and entry["extraction_status"] == "ok"]
            predictions.append({
                "population": population, "model_id": model, "train_row_id": row["train_row_id"],
                "pair_id": row["pair_id"], "outcome": int(outcome), "logit": float(logit),
                "probability": float(probability), "loss": float(loss),
                "comparison_type": row["comparison_type"], "pet_care": group_name(row),
                "min_endpoint_opponents": support,
                "any_no_detection": "no_detection" in detection,
                "any_dog_fallback": "dog_fallback" in detection,
                "any_multiple_cats": any(entry is not None and entry["extraction_status"] == "ok"
                                         and entry["candidate_cat_count"] is not None
                                         and entry["candidate_cat_count"] > 1 for entry in (qa, qb)),
                "both_cat_detected": len(detection) == 2 and all(value == "cat_detected" for value in detection),
                "feature_quality_available": len(detection) == 2,
            })
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["population"], row["model_id"])].append(row)
    bootstrap_response, bootstrap_pair = {}, {}
    for population, losses in losses_by_population.items():
        subset = population_rows[population]
        samples, pairs = pair_bootstrap([row["pair_id"] for row in subset], losses,
                                        frozen["bootstrap"]["seed"], frozen["bootstrap"]["replicates"])
        bootstrap_response[population], bootstrap_pair[population] = samples, pairs
    main, subgroup, calibration = [], [], []
    for (population, model), entries in sorted(grouped.items()):
        losses = np.asarray([entry["loss"] for entry in entries])
        y = np.asarray([entry["outcome"] for entry in entries])
        logits = np.asarray([entry["logit"] for entry in entries])
        loss, brier, accuracy = score(y, logits)
        if not np.isclose(loss, losses.mean(), atol=1e-14):
            raise ValueError(f"Row-level loss reconciliation failed: {model}")
        pair_losses: dict[str, list[float]] = defaultdict(list)
        for entry in entries:
            pair_losses[entry["pair_id"]].append(entry["loss"])
        pair_loss = float(np.mean([np.mean(values) for values in pair_losses.values()]))
        low, high, se = interval(bootstrap_response[population][model], .95)
        plow, phigh, pse = interval(bootstrap_pair[population][model], .95)
        source = {
            "M2-temperature": "common:M2",
            "M3*-temperature": f"M3-{frozen['selected_representation']}",
            "M4*-temperature": f"M4-{frozen['selected_representation']}",
        }.get(model, model)
        metadata = selection["selected"][source]
        resource = resources[source]
        if int(resource["stored_trainable_coefficients"]) != metadata["coefficient_count"]:
            raise ValueError(f"Model parameter count does not reconcile for {model}.")
        probabilities = expit(logits)
        main.append({
            "population": population, "model_id": model, "responses": len(entries),
            "pairs": len(pair_losses), "response_log_loss": loss, "brier": brier, "accuracy": accuracy,
            "pair_log_loss": pair_loss, "response_ci95_low": low, "response_ci95_high": high,
            "response_bootstrap_se": se, "pair_ci95_low": plow, "pair_ci95_high": phigh,
            "pair_bootstrap_se": pse,
            "mean_probability": float(probabilities.mean()), "probability_p05": float(np.quantile(probabilities, .05)),
            "probability_median": float(np.median(probabilities)), "probability_p95": float(np.quantile(probabilities, .95)),
            "fitting_loss": metadata["train_loss"], "penalized_objective": metadata["objective"],
            "validation_loss": metadata["validation_loss"], "ridge": metadata["ridge"], "rho": metadata["rho"],
            "gradient_max": metadata["gradient_max"], "coefficient_count": metadata["coefficient_count"],
            "retained_feature_dimensions": int(resource["retained_feature_dimensions"]),
            "fitting_seconds": metadata["seconds"], "inference_seconds": float(resource["mean_inference_seconds_3_runs"]),
            "inference_timing_scope": resource["inference_scope"], "test_response_coverage": len(entries)/len(rows),
        })
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, entry in enumerate(entries):
            for label in subgroup_labels(entry):
                groups[label].append(i)
        for (dimension, label), indices in sorted(groups.items()):
            ids = np.asarray(indices)
            pairs = len({entries[i]["pair_id"] for i in indices})
            sufficient = len(ids) >= 100 and pairs >= 30
            gl, gb, ga = score(y[ids], logits[ids])
            subgroup.append({
                "population": population, "model_id": model, "dimension": dimension, "group": label,
                "responses": len(ids), "pairs": pairs, "support_status": "supported" if sufficient else "low_support",
                "response_log_loss": gl, "brier": gb, "accuracy": ga,
            })
            if not sufficient:
                continue
            probs = expit(logits[ids])
            for bin_id in range(10):
                ix = ids[(probs >= bin_id/10) & ((probs < (bin_id+1)/10) if bin_id < 9 else (probs <= 1))]
                bpairs = len({entries[i]["pair_id"] for i in ix})
                calibration.append({
                    "population": population, "model_id": model, "dimension": dimension, "group": label,
                    "bin": bin_id, "lower": bin_id/10, "upper": (bin_id+1)/10,
                    "responses": len(ix), "pairs": bpairs,
                    "support_status": "supported" if len(ix) >= 30 and bpairs >= 10 else "low_support",
                    "mean_probability": float(expit(logits[ix]).mean()) if len(ix) else "",
                    "observed_a_rate": float(y[ix].mean()) if len(ix) else "",
                })
        type_groups = [row for row in subgroup if row["population"] == population and row["model_id"] == model
                       and row["dimension"] == "comparison_type"]
        weighted = sum(row["response_log_loss"] * row["responses"] for row in type_groups) / len(entries)
        if not np.isclose(weighted, loss, atol=1e-12):
            raise ValueError(f"Subgroup metrics do not reconcile for {model}.")

    def contrast(label: str, reference: str, candidate: str, primary: bool) -> dict[str, Any]:
        r, c = grouped[("common", reference)], grouped[("common", candidate)]
        if [x["train_row_id"] for x in r] != [x["train_row_id"] for x in c]:
            raise ValueError(f"Paired contrast IDs differ: {label}")
        rloss, closs = np.asarray([x["loss"] for x in r]), np.asarray([x["loss"] for x in c])
        samples = bootstrap_response["common"][reference] - bootstrap_response["common"][candidate]
        psamples = bootstrap_pair["common"][reference] - bootstrap_pair["common"][candidate]
        low95, high95, se = interval(samples, .95)
        low, high, _ = interval(samples, .9875 if primary else .95)
        plow, phigh, pse = interval(psamples, .9875 if primary else .95)
        by_pair: dict[str, list[float]] = defaultdict(list)
        for row, value in zip(r, rloss-closs):
            by_pair[row["pair_id"]].append(float(value))
        return {
            "contrast": label, "reference": reference, "candidate": candidate,
            "responses": len(r), "pairs": len(by_pair),
            "response_loss_difference": float(np.mean(rloss-closs)),
            "pair_loss_difference": float(np.mean([np.mean(v) for v in by_pair.values()])),
            "interval_level": .9875 if primary else .95, "response_interval_low": low,
            "response_interval_high": high, "response_ci95_low": low95, "response_ci95_high": high95,
            "response_bootstrap_se": se, "pair_interval_low": plow, "pair_interval_high": phigh,
            "pair_bootstrap_se": pse,
            "interpretation": "positive favors candidate; conditional on fitted models and development choices",
            "multiplicity": "four primary contrasts: approximate 98.75% pointwise intervals" if primary else
                            "secondary descriptive interval; multiplicity not adjusted",
        }
    primary = [contrast(item["label"], item["reference"], item["candidate"], True)
               for item in frozen["primary_contrasts"]]
    block = frozen["selected_representation"]
    secondary_specs = [(f"{b}_context", f"M3-{b}", f"M4-{b}") for b in ("H", "C", "Y")]
    secondary_specs += [
        ("position_effect", "common:M2-no-position", "common:M2"),
        ("context_with_group_side", "M3-side", "M4-side"),
        ("temperature_M2", "common:M2", "M2-temperature"),
        ("temperature_M3", f"M3-{block}", "M3*-temperature"),
        ("temperature_M4", f"M4-{block}", "M4*-temperature"),
    ]
    secondary = [contrast(label, ref, candidate, False) for label, ref, candidate in secondary_specs]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(predictions), outputs[0])
    for path, table in zip(outputs[1:6], (main, primary, secondary, subgroup, calibration)):
        _write_csv(path, table)
    _calibration_figure(outputs[6], calibration, block)
    provenance = {
        "status": "evaluated_once", "freeze_sha256": _hash(root / "artifacts/model_freeze.json"),
        "internal_test_outcomes_read": True, "bootstrap": frozen["bootstrap"],
        "limitations": frozen["limitations"],
        "output_sha256": {path.relative_to(root).as_posix(): _hash(path) for path in outputs[:-1]},
    }
    with outputs[-1].open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")
    return {"models": len(grouped), "common_responses": len(population_rows["common"]),
            "full_responses": len(population_rows["full"])}


def _calibration_figure(path: Path, calibration: list[dict[str, Any]], block: str) -> None:
    import os
    import matplotlib
    os.environ.setdefault("MPLCONFIGDIR", str(path.parents[1] / ".cache/matplotlib"))
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    for model in ("common:M0", "common:M1", "common:M2", f"M3-{block}", f"M4-{block}"):
        rows = [row for row in calibration if row["population"] == "common" and row["model_id"] == model
                and row["dimension"] == "overall" and row["support_status"] == "supported"]
        if rows:
            ax.plot([row["mean_probability"] for row in rows], [row["observed_a_rate"] for row in rows],
                    marker="o", label=model)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="Mean predicted P(A)", ylabel="Observed A frequency", xlim=(0, 1), ylim=(0, 1),
           title="Frozen common strict-G2 test; supported bins only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
