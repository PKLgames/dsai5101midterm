"""Independently reconstruct and validate the official prediction artifact."""

from __future__ import annotations

import csv
import json

import networkx as nx
import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit
from threadpoolctl import threadpool_limits

from common import EXCLUDED, HERE, ROOT, digest, load_labeled, snapshot, verify_history, write_json
from run import read_official, verify_policy
from src.development import feature_differences, group_name, identity_matrix, visual_design
from src.estimator import contextual_penalties, objective_gradient
from src.preprocessing import fit_preprocessing, transform_photos


def verify() -> None:
    if (HERE / "verification.json").exists():
        raise FileExistsError("Deployment verification already exists.")
    policy = verify_policy()
    verify_history()
    before = json.loads((HERE / "original_hashes_before.json").read_text(encoding="utf-8"))
    if snapshot(ROOT, EXCLUDED) != before:
        raise RuntimeError("Original project files differ from the pre-deployment snapshot.")
    provenance = json.loads((HERE / "deployment_provenance.json").read_text(encoding="utf-8"))
    if provenance["policy_sha256"] != digest(HERE / "deployment_policy.json"):
        raise ValueError("Deployment provenance policy mismatch.")
    if provenance["deployment_started_sha256"] != digest(HERE / "deployment_started.json"):
        raise ValueError("Deployment access record hash mismatch.")
    for name, expected in provenance["output_sha256"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"Deployment output hash mismatch: {name}")
    model = json.loads((HERE / "deployment_models.json").read_text(encoding="utf-8"))
    g2 = json.loads((ROOT / "artifacts" / "model_selection.json").read_text(encoding="utf-8"))["selected"]
    g3 = json.loads((ROOT / "g3_experiment" / "selected_models.json").read_text(encoding="utf-8"))
    if (model["M1"]["ridge"] != g2["common:M1"]["ridge"] or
        model["M4-C"]["ridge"] != g3["selected"]["M4-C"]["ridge"] or
        model["M4-C"]["rho"] != g3["selected"]["M4-C"]["rho"] or
        model["M4-C"]["context_reference"] != g3["context"]["reference"] or
        model["M4-C"]["interactions"] != g3["context"]["interactions"]):
        raise ValueError("Deployment parameters do not match validation-selected models.")
    train = load_labeled(policy["source_sha256"]["raw_training"])
    owners = {r[f"{side}_image"]: r[f"{side}_cat"] for r in train for side in ("a", "b")}
    fields, test = read_official(policy["source_sha256"]["official_test"], owners)
    with (HERE / "official_predictions.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError("Official prediction columns differ from the source.")
        output = list(reader)
    with (HERE / "routing.csv").open(encoding="utf-8", newline="") as stream:
        routing = list(csv.DictReader(stream))
    if len(train) != model["training_responses"] or len(output) != len(test) or len(routing) != len(test):
        raise ValueError("Training, prediction or route counts differ.")
    graph_photo, graph_cat = nx.Graph(), nx.Graph()
    graph_photo.add_edges_from((r["a_image"], r["b_image"]) for r in train)
    graph_cat.add_edges_from((r["a_cat"], r["b_cat"]) for r in train)
    photo_comp = {node: i for i, group in enumerate(nx.connected_components(graph_photo)) for node in group}
    cat_comp = {node: i for i, group in enumerate(nx.connected_components(graph_cat)) for node in group}
    records = pq.read_table(ROOT / "features" / "C.parquet").to_pylist()
    by_id = {r["image_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate cached CNN features.")
    fit_photos = set(owners)
    spec = fit_preprocessing("C", records, fit_photos)
    if spec != model["M4-C"]["preprocessing"]:
        raise ValueError("Full-training-photo CNN preprocessing does not reproduce.")
    y = np.asarray([r["y_displayed"] for r in train], dtype=np.float64)
    cats = sorted(cat_comp)
    if cats != model["M1"]["cat_keys"]:
        raise ValueError("Deployment cat keys do not match all labeled data.")
    p0 = (float(y.sum()) + .5) / (len(y) + 1)
    np.testing.assert_allclose(p0, model["M0"]["probability"], rtol=0, atol=1e-15)
    usable = {image for image, record in by_id.items() if record["extraction_status"] == "ok"}
    expected_routes = []
    for source, prediction, route in zip(test, output, routing):
        if source["test_row_id"] != prediction["test_row_id"] or source["test_row_id"] != route["test_row_id"]:
            raise ValueError("Official IDs or row order do not match.")
        if any(source[field] != prediction[field] for field in fields if field not in {"choice", "prob_a"}):
            raise ValueError("A nonprediction official field changed.")
        a, b = source["a_image"], source["b_image"]
        ca, cb = source["a_cat"], source["b_cat"]
        known_photos = a in photo_comp and b in photo_comp and photo_comp[a] == photo_comp[b]
        known_cats = ca in cat_comp and cb in cat_comp and cat_comp[ca] == cat_comp[cb]
        if known_photos and known_cats:
            expected_route = "known_photo_m1"
        elif a in usable and b in usable:
            expected_route = "shared_visual"
        elif known_cats:
            expected_route = "cat_fallback_m1"
        else:
            expected_route = "constant_fallback_m0"
        if route["route"] != expected_route:
            raise ValueError(f"Wrong route for {source['test_row_id']}.")
        expected_routes.append(expected_route)
        probability = float(prediction["prob_a"])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid serialized official probability.")
        if prediction["choice"] != ("A" if probability >= .5 else "B"):
            raise ValueError("Official choice differs from the serialized probability threshold.")
    needed = sorted(fit_photos | {r[f"{side}_image"] for r, route in zip(test, expected_routes)
                                  if route == "shared_visual" for side in ("a", "b")})
    vectors = transform_photos(spec, [by_id[image] for image in needed])
    cat_x = identity_matrix(train, cats, "cat")
    interactions = tuple(model["M4-C"]["interactions"])
    train_diff = feature_differences(train, needed, vectors)
    visual_x = visual_design(train_diff, [group_name(r) for r in train], interactions)
    cat_theta = np.asarray(model["M1"]["coefficients"])
    visual_theta = np.asarray(model["M4-C"]["coefficients"])
    weights = np.ones(len(train))
    with threadpool_limits(limits=1):
        cat_objective, cat_gradient = objective_gradient(
            cat_theta, cat_x, y, weights, np.r_[[1e-6], np.full(len(cats), model["M1"]["ridge"])])
        visual_objective, visual_gradient = objective_gradient(
            visual_theta, visual_x, y, weights, contextual_penalties(
                train_diff.shape[1], len(interactions), model["M4-C"]["ridge"], model["M4-C"]["rho"]))
    for name, objective, gradient in (("M1", cat_objective, cat_gradient), ("M4-C", visual_objective, visual_gradient)):
        maximum = float(np.max(np.abs(gradient)))
        if maximum >= 1e-5:
            raise ValueError(f"{name} deployment gradient fails acceptance.")
        np.testing.assert_allclose([objective, maximum], [model[name]["objective"], model[name]["gradient_max"]],
                                   rtol=0, atol=1e-12)
    reconstructed = np.full(len(test), model["M0"]["logit"], dtype=np.float64)
    cat_indices = [i for i, route in enumerate(expected_routes) if route in {"known_photo_m1", "cat_fallback_m1"}]
    if cat_indices:
        reconstructed[cat_indices] = np.asarray(identity_matrix([test[i] for i in cat_indices], cats, "cat") @ cat_theta).reshape(-1)
    visual_indices = [i for i, route in enumerate(expected_routes) if route == "shared_visual"]
    if visual_indices:
        subset = [test[i] for i in visual_indices]
        diff = feature_differences(subset, needed, vectors)
        design = visual_design(diff, [group_name(r) for r in subset], interactions)
        reconstructed[visual_indices] = np.asarray(design @ visual_theta).reshape(-1)
    probs = expit(reconstructed)
    np.testing.assert_allclose(probs, [float(r["prob_a"]) for r in output], rtol=0, atol=1e-14)
    np.testing.assert_allclose(probs, [float(r["prob_a"]) for r in routing], rtol=0, atol=1e-14)
    np.testing.assert_allclose(reconstructed, [float(r["logit"]) for r in routing], rtol=0, atol=1e-14)
    if snapshot(ROOT, EXCLUDED) != before:
        raise RuntimeError("Original project files changed during independent verification.")
    report = {
        "status": "independent_reconstruction_passed", "official_rows_verified": len(output),
        "labeled_training_rows_verified": len(train), "prediction_columns_and_order_preserved": True,
        "all_probabilities_reconstructed": True, "all_choice_thresholds_verified": True,
        "full_training_photo_preprocessing_reproduced": True, "selected_parameters_verified": True,
        "m1_gradient_max": float(np.max(np.abs(cat_gradient))),
        "m4_c_gradient_max": float(np.max(np.abs(visual_gradient))),
        "g2_g3_g4_frozen_history_verified": True, "original_project_files_unchanged": len(before),
        "prediction_sha256": digest(HERE / "official_predictions.csv"),
        "deployment_provenance_sha256": digest(HERE / "deployment_provenance.json"),
        "official_labels_available": False, "upload_performed": False,
    }
    write_json(HERE / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    verify()
