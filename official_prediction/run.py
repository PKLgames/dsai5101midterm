"""Fit validation-selected deployment models and write a separate official CSV."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter

import networkx as nx
import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit
from threadpoolctl import threadpool_limits

from common import (EXCLUDED, HERE, ROOT, classify_route, digest, load_labeled,
                    probabilities_to_choices, snapshot, verify_history, write_csv, write_json)
from src.development import feature_differences, group_name, identity_matrix, visual_design
from src.estimator import contextual_penalties, fit_logistic
from src.preprocessing import fit_preprocessing, transform_photos


def verify_policy() -> dict:
    path = HERE / "deployment_policy.json"
    if digest(path) != (HERE / "deployment_policy.sha256").read_text(encoding="ascii").strip():
        raise ValueError("Deployment policy seal mismatch.")
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy["status"] != "validation_only_deployment_rule_sealed_before_refit":
        raise ValueError("Invalid deployment policy state.")
    sources = {
        "g2_freeze": ROOT / "artifacts" / "model_freeze.json",
        "g2_selection": ROOT / "artifacts" / "model_selection.json",
        "g3_freeze": ROOT / "g3_experiment" / "freeze.json",
        "g3_selection": ROOT / "g3_experiment" / "selected_models.json",
        "g4_freeze": ROOT / "g4_experiment" / "freeze.json",
        "raw_training": ROOT / "dataset" / "train.csv",
        "official_test": ROOT / "dataset" / "test.csv",
    }
    for key, path in sources.items():
        if digest(path) != policy["source_sha256"][key]:
            raise ValueError(f"Sealed deployment source changed: {key}")
    return policy


def components(rows: list[dict], entity: str) -> dict[str, int]:
    graph = nx.Graph()
    for row in rows:
        graph.add_edge(row[f"a_{entity}"], row[f"b_{entity}"])
    groups = sorted(nx.connected_components(graph), key=lambda group: min(group))
    return {key: i for i, group in enumerate(groups) for key in group}


def read_official(expected_hash: str, known_owners: dict[str, str]) -> tuple[list[str], list[dict]]:
    path = ROOT / "dataset" / "test.csv"
    if digest(path) != expected_hash:
        raise ValueError("Official test file differs from its sealed hash.")
    required = {"test_row_id", "a_cat", "b_cat", "a_image", "b_image", "choice",
                "prob_a", "comparison_type", "pet_care"}
    owners = dict(known_owners)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if fields is None or len(fields) != len(set(fields)) or not required <= set(fields):
            raise ValueError("Official test schema has missing or duplicate columns.")
        rows = list(reader)
    ids = set()
    for row in rows:
        key = row["test_row_id"]
        if None in row or not key or key in ids or row["choice"] != "" or row["prob_a"] != "":
            raise ValueError(f"Official test row is malformed, duplicated or already filled: {key}")
        ids.add(key)
        if row["a_image"] == row["b_image"] or (
            (row["a_cat"] == row["b_cat"]) != (row["comparison_type"] == "within_cat")
        ) or row["comparison_type"] not in {"within_cat", "between_cats"}:
            raise ValueError(f"Official row has invalid comparison structure: {key}")
        group_name(row)
        for side in ("a", "b"):
            cat, image = row[f"{side}_cat"], row[f"{side}_image"]
            parts = image.split("/")
            if not cat or len(parts) < 2 or parts[0] != cat or any(p in {"", ".", ".."} for p in parts) or "\\" in image:
                raise ValueError(f"Official row has an invalid image/cat path: {key}")
            if image in owners and owners[image] != cat:
                raise ValueError(f"Image has inconsistent cat identity: {image}")
            owners[image] = cat
    if not rows or digest(path) != expected_hash:
        raise ValueError("Official test file is empty or changed while reading.")
    return fields, rows


def run() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    names = ("deployment_started.json", "deployment_models.json", "official_predictions.csv",
             "routing.csv", "deployment_provenance.json")
    if any((HERE / name).exists() for name in names):
        raise FileExistsError("Deployment refit already started; refusing to overwrite.")
    before = json.loads((HERE / "original_hashes_before.json").read_text(encoding="utf-8"))
    if snapshot(ROOT, EXCLUDED) != before:
        raise RuntimeError("Original project files changed after the deployment policy was sealed.")
    verify_history()
    policy = verify_policy()
    if policy["known_photo_model"]["model"] != "common:M1" or policy["new_or_unsupported_photo_model"]["model"] != "M4-C":
        raise ValueError("Deployment implementation requires the sealed M1 and M4-C selections.")
    write_json(HERE / "deployment_started.json", {
        "status": "all_labeled_deployment_refit_started",
        "policy_sha256": digest(HERE / "deployment_policy.json"),
        "test_labels": "Official test choice and prob_a are blank; no official labels were used.",
        "training_scope": "All audited labeled responses, including former internal-test responses.",
    })
    train = load_labeled(policy["source_sha256"]["raw_training"])
    owners = {}
    for row in train:
        for side in ("a", "b"):
            photo, cat = row[f"{side}_image"], row[f"{side}_cat"]
            if photo in owners and owners[photo] != cat:
                raise ValueError(f"Training photo has inconsistent cat IDs: {photo}")
            owners[photo] = cat
    fields, test = read_official(policy["source_sha256"]["official_test"], owners)
    photo_components, cat_components = components(train, "image"), components(train, "cat")
    fitting_photos = set(owners)
    fitting_cats = sorted(cat_components)
    records = pq.read_table(ROOT / "features" / "C.parquet").to_pylist()
    by_id = {row["image_id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate CNN feature IDs.")
    audited = json.loads((ROOT / "artifacts" / "raw_hashes.json").read_text(encoding="utf-8"))["files"]
    required_images = fitting_photos | {row[f"{side}_image"] for row in test for side in ("a", "b")}
    usable = set()
    for image in required_images:
        record = by_id.get(image)
        source = audited.get("dataset/images/" + image)
        if record is None or source is None or record["extraction_status"] != "ok":
            continue
        if record["source_sha256"] != source["sha256"]:
            raise ValueError(f"CNN source image hash mismatch: {image}")
        usable.add(image)
    if not fitting_photos <= usable:
        raise ValueError("The chosen visual model cannot refit on all labeled photos.")
    routes = [classify_route(row, photo_components, cat_components, usable) for row in test]
    context = policy["new_or_unsupported_photo_model"]
    interactions = tuple(context["context_interactions"])
    fit_groups = [group_name(row) for row in train]
    y = np.asarray([row["y_displayed"] for row in train], dtype=np.float64)
    spec = fit_preprocessing("C", records, fitting_photos)
    selected_images = sorted(fitting_photos | {
        row[f"{side}_image"] for row, (route, _) in zip(test, routes) if route == "shared_visual"
        for side in ("a", "b")
    })
    vectors = transform_photos(spec, [by_id[image] for image in selected_images])
    train_diff = feature_differences(train, selected_images, vectors)
    cat_x = identity_matrix(train, fitting_cats, "cat")
    visual_x = visual_design(train_diff, fit_groups, interactions)
    ridge = float(policy["known_photo_model"]["ridge"])
    visual_ridge, rho = float(context["ridge"]), float(context["rho"])
    with threadpool_limits(limits=1):
        cat_fit = fit_logistic(cat_x, y, np.r_[[1e-6], np.full(len(fitting_cats), ridge)])
        if not cat_fit.accepted:
            raise RuntimeError(f"M1 deployment refit failed numerical acceptance: {cat_fit.attempts}")
        visual_fit = fit_logistic(visual_x, y,
                                  contextual_penalties(train_diff.shape[1], len(interactions), visual_ridge, rho))
        if not visual_fit.accepted:
            raise RuntimeError(f"M4-C deployment refit failed numerical acceptance: {visual_fit.attempts}")
    p0 = (float(y.sum()) + .5) / (len(y) + 1)
    m0_logit = float(np.log(p0 / (1.-p0)))
    logits = np.full(len(test), np.nan)
    cat_indices = [i for i, (route, _) in enumerate(routes) if route in {"known_photo_m1", "cat_fallback_m1"}]
    if cat_indices:
        sub = [test[i] for i in cat_indices]
        logits[cat_indices] = np.asarray(identity_matrix(sub, fitting_cats, "cat") @ cat_fit.coefficients).reshape(-1)
    visual_indices = [i for i, (route, _) in enumerate(routes) if route == "shared_visual"]
    if visual_indices:
        sub = [test[i] for i in visual_indices]
        diff = feature_differences(sub, selected_images, vectors)
        design = visual_design(diff, [group_name(row) for row in sub], interactions)
        logits[visual_indices] = np.asarray(design @ visual_fit.coefficients).reshape(-1)
    for i, (route, _) in enumerate(routes):
        if route == "constant_fallback_m0":
            logits[i] = m0_logit
    if not np.isfinite(logits).all():
        raise ValueError("At least one official row lacks a finite routed prediction.")
    probs = expit(logits)
    choices = probabilities_to_choices(probs)
    prediction_rows, routing = [], []
    for i, row in enumerate(test):
        probability = format(float(probs[i]), ".17g")
        if ("A" if float(probability) >= .5 else "B") != choices[i]:
            raise ValueError("Serialized probability disagrees with the choice threshold.")
        prediction_rows.append({**row, "choice": choices[i], "prob_a": probability})
        unseen_cats = sum(row[f"{s}_cat"] not in cat_components for s in ("a", "b"))
        unseen_photos = sum(row[f"{s}_image"] not in photo_components for s in ("a", "b"))
        routing.append({"test_row_id": row["test_row_id"], "route": routes[i][0], "reason": routes[i][1],
                        "unseen_cats": unseen_cats, "unseen_photos": unseen_photos,
                        "pet_care_group": group_name(row), "logit": float(logits[i]), "prob_a": probability})
    models = {
        "status": "separate_deployment_refit_not_evaluation",
        "training_responses": len(train), "unique_training_photos": len(fitting_photos),
        "training_cats": len(fitting_cats), "photo_components": len(set(photo_components.values())),
        "cat_components": len(set(cat_components.values())),
        "M0": {"logit": m0_logit, "probability": p0},
        "M1": {"ridge": ridge, "cat_keys": fitting_cats, "coefficients": cat_fit.coefficients.tolist(),
               "gradient_max": cat_fit.gradient_max, "objective": cat_fit.objective, "attempts": list(cat_fit.attempts)},
        "M4-C": {"ridge": visual_ridge, "rho": rho, "context_reference": context["context_reference"],
                 "interactions": list(interactions), "preprocessing": spec,
                 "coefficients": visual_fit.coefficients.tolist(), "gradient_max": visual_fit.gradient_max,
                 "objective": visual_fit.objective, "attempts": list(visual_fit.attempts)},
    }
    write_json(HERE / "deployment_models.json", models)
    write_csv(HERE / "official_predictions.csv", prediction_rows, fields)
    write_csv(HERE / "routing.csv", routing, list(routing[0]))
    with (HERE / "official_predictions.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError("Written prediction schema does not match the official test file.")
        written = list(reader)
    if len(written) != len(test):
        raise ValueError("Written official prediction row count differs from the original test file.")
    for source, output in zip(test, written):
        if any(source[field] != output[field] for field in fields if field not in {"choice", "prob_a"}):
            raise ValueError(f"Official row changed outside prediction fields: {source['test_row_id']}")
        p = float(output["prob_a"])
        if not np.isfinite(p) or not 0 <= p <= 1 or output["choice"] != ("A" if p >= .5 else "B"):
            raise ValueError(f"Invalid serialized official prediction: {source['test_row_id']}")
    if snapshot(ROOT, EXCLUDED) != before:
        raise RuntimeError("Original project files changed during the deployment refit.")
    verify_history()
    verify_policy()
    route_counts = dict(sorted(Counter(row["route"] for row in routing).items()))
    unseen_counts = dict(sorted(Counter(str(row["unseen_cats"]) for row in routing).items()))
    provenance = {
        "status": "official_predictions_generated_not_submitted",
        "original_project_files_unchanged": len(before), "original_scope_exclusions": sorted(EXCLUDED),
        "policy_sha256": digest(HERE / "deployment_policy.json"),
        "original_hash_snapshot_sha256": digest(HERE / "original_hashes_before.json"),
        "input_sha256": policy["source_sha256"],
        "output_sha256": {name: digest(HERE / name) for name in names[1:4]},
        "deployment_started_sha256": digest(HERE / "deployment_started.json"),
        "training_responses": len(train), "official_responses": len(test),
        "route_counts": route_counts, "unseen_cat_endpoint_counts": unseen_counts,
        "choice_counts": dict(sorted(Counter(choices).items())),
        "all_original_columns_and_row_order_preserved": True,
        "all_probabilities_finite_and_valid": True, "choice_threshold_verified": True,
        "models_refit_on_all_labeled_responses": True, "internal_test_metrics_used_for_selection": False,
        "official_test_labels_available": False, "upload_performed": False,
        "limitations": policy["limitations"],
    }
    write_json(HERE / "deployment_provenance.json", provenance)
    print(json.dumps({key: provenance[key] for key in ("training_responses", "official_responses", "route_counts",
                                                       "unseen_cat_endpoint_counts", "choice_counts",
                                                       "original_project_files_unchanged")}, indent=2))


if __name__ == "__main__":
    run()
