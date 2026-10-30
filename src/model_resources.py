"""Outcome-blind resource and identifiability report for selected Stage 7 fits."""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .development import feature_differences, group_name, identity_matrix, load_development, visual_design
from .preprocessing import transform_photos


def report_resources(root: Path) -> int:
    output = root / "tables/model_resources.csv"
    if output.exists():
        raise FileExistsError("Model resource report already exists.")
    selection = json.loads((root / "artifacts/model_selection.json").read_text(encoding="utf-8"))
    fit, val = load_development(root)
    common_fit = [row for row in fit if row["common_visual_fit"]]
    populations = {
        "full": [row for row in val if row["full_strict_g2"]],
        "common": [row for row in val if row["common_strict_g2"]],
    }
    image_ids = sorted({row[f"{side}_image"] for row in common_fit + populations["common"]
                        for side in ("a", "b")})
    groups = [group_name(row) for row in populations["common"]]
    interactions = tuple(selection["interaction_contexts"])
    matrices, dimensions = {}, {}
    for block in ("H", "C", "Y"):
        all_rows = pq.read_table(root / "features" / f"{block}.parquet").to_pylist()
        spec = json.loads((root / "artifacts/preprocessing" / f"{block}.json").read_text(encoding="utf-8"))
        by_id = {row["image_id"]: row for row in all_rows}
        vectors = transform_photos(spec, [by_id[image] for image in image_ids])
        matrices[block] = feature_differences(populations["common"], image_ids, vectors)
        dimensions[block] = len(spec["final_names"])
    components = pq.read_table(
        root / "artifacts/graph_components.parquet",
        columns=["population", "entity", "node_id", "component_id"],
    ).to_pylist()
    graph_summaries = list(csv.DictReader((root / "tables/graph_summary.csv").open(encoding="utf-8", newline="")))

    os.environ.setdefault("TORCH_HOME", str(root / ".cache/torch"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".cache/ultralytics"))
    import open_clip
    from ultralytics import YOLO
    from legacy.src.model.backbones import build_backbone

    backbone, _ = build_backbone("mobilenet_v3_small", pretrained=True)
    manifest = json.loads((root / "features/manifests/Y.json").read_text(encoding="utf-8"))
    yolo = YOLO(manifest["checkpoint"]["detector"])
    clip, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True,
        cache_dir=str(root / ".cache/open_clip"), device="cpu",
    )
    backbone_counts = {
        "C": sum(parameter.numel() for parameter in backbone.parameters()),
        "Y_detector": sum(parameter.numel() for parameter in yolo.model.parameters()),
        "Y_clip": sum(parameter.numel() for parameter in clip.parameters()),
    }
    records = []
    selected = selection["selected"]
    with threadpool_limits(limits=1):
        for key, model in selected.items():
            if not isinstance(model, dict) or "coefficients" not in model or key in {"M3*", "M4*"}:
                continue
            population = "full" if key.startswith("full:") else "common"
            rows = populations[population]
            block = ""
            entity = ""
            max_mean: float | str = ""
            if key.endswith("M0"):
                design = None
            elif key.endswith("M1") or key.endswith("M2") or key.endswith("M2-no-position"):
                entity = "cat" if key.endswith("M1") else "photo"
                side = not key.endswith("M2-no-position")
                design = identity_matrix(rows, model["identity_keys"], "cat" if entity == "cat" else "image", side)
                index = {x["node_id"]: x["component_id"] for x in components
                         if x["population"] == population and x["entity"] == entity}
                values: dict[int, list[float]] = defaultdict(list)
                for node, value in zip(model["identity_keys"], model["coefficients"][int(side):]):
                    values[index[node]].append(value)
                max_mean = max(abs(float(np.mean(values_))) for values_ in values.values())
            else:
                block = selection["selected_representation"] if "side" in key else key[-1]
                contextual = key.startswith("M4") and not model["pooled"]
                design = visual_design(matrices[block], groups, interactions if contextual else (),
                                       interactions if "side" in key else ())
            coefficients = np.asarray(model["coefficients"], dtype=np.float64)
            samples = []
            for _ in range(3):
                start = time.perf_counter()
                logits = np.full(len(rows), coefficients[0]) if design is None else np.asarray(design @ coefficients)
                probabilities = expit(logits)
                samples.append(time.perf_counter() - start)
                if not np.isfinite(probabilities).all():
                    raise ValueError(f"Nonfinite prediction during timing: {key}")
            photo = next(x for x in graph_summaries if x["population"] == population and x["entity"] == "photo")
            cat = next(x for x in graph_summaries if x["population"] == population and x["entity"] == "cat")
            records.append({
                "model": key, "population": population, "retained_feature_dimensions": dimensions.get(block, 0),
                "stored_trainable_coefficients": len(coefficients),
                "estimated_interaction_groups": len(interactions) if key.startswith("M4") and not model["pooled"] else 0,
                "fitted_photos": int(photo["nodes"]), "fitted_cats": int(cat["nodes"]),
                "photo_components": int(photo["undirected_components"]), "cat_components": int(cat["undirected_components"]),
                "identity_max_abs_component_mean": max_mean,
                "selected_fit_seconds": model["seconds"], "validation_responses": len(rows),
                "mean_inference_seconds_3_runs": float(np.mean(samples)),
                "inference_scope": "existing design matrix to probability; excludes preprocessing and IO",
                "frozen_cnn_parameters": backbone_counts["C"] if block == "C" else 0,
                "frozen_yolo_parameters": backbone_counts["Y_detector"] if block == "Y" else 0,
                "frozen_clip_parameters": backbone_counts["Y_clip"] if block == "Y" else 0,
                "pooled_fallback": model["pooled"],
            })
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return len(records)
