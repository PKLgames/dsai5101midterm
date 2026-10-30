"""Outcome-blind structural feature QA and human-review queue."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .feature_y import CATEGORICAL_FEATURES, NUMERIC_FEATURES, decode_mask
from .features import H_FEATURES, _file_hash, _input_inventory

IMMUTABLE_REVIEW_FIELDS = (
    "image_id", "image_path", "review_panel_path", "selection", "flags",
    "y_detection_status", "y_mask_source", "y_attributes", "reviewer",
)


def review_fingerprint(rows: list[dict[str, str]]) -> str:
    payload = sorted(
        ({key: row[key] for key in IMMUTABLE_REVIEW_FIELDS} for row in rows),
        key=lambda row: (row["image_id"], row["reviewer"]),
    )
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class Components:
    def __init__(self) -> None:
        self.parents: dict[str, str] = {}

    def find(self, item: str) -> str:
        if item not in self.parents:
            raise KeyError(item)
        while self.parents[item] != item:
            self.parents[item] = self.parents[self.parents[item]]
            item = self.parents[item]
        return item

    def add_pair(self, a: str, b: str) -> None:
        self.parents.setdefault(a, a)
        self.parents.setdefault(b, b)
        self.parents[self.find(a)] = self.find(b)

    def supports(self, a: str, b: str) -> bool:
        return a in self.parents and b in self.parents and self.find(a) == self.find(b)


def build_components(manifest: list[dict[str, Any]], usable: set[str] | None = None) -> Components:
    graph = Components()
    for pair in manifest:
        a, b = pair["image_i"], pair["image_j"]
        if pair["partition"] == "fit" and (usable is None or (a in usable and b in usable)):
            graph.add_pair(a, b)
    return graph


def rank_image(image_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).hexdigest()


def audit_block(
    root: Path, block: str, expected_images: dict[str, str], validation: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = root / "features" / f"{block}.parquet"
    manifest_path = root / "features" / "manifests" / f"{block}.json"
    report: dict[str, Any] = {
        "block": block, "status": "unavailable", "automated_checks": "not_run",
        "successful_images": 0, "failed_images": 0, "strict_validation_responses": len(validation),
        "covered_validation_responses": 0, "validation_coverage": 0.0, "details": "",
    }
    if not path.is_file() or not manifest_path.is_file():
        report["details"] = "Feature table or provenance manifest is missing."
        return report, {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        table = pq.read_table(path)
    except (OSError, ValueError) as exc:
        report.update({
            "status": "quarantined", "automated_checks": "failed",
            "details": f"Cannot read feature artifact: {type(exc).__name__}: {exc}",
        })
        return report, {}
    rows = table.to_pylist()
    errors = []
    required = {"image_id", "source_sha256", "extraction_status", "failure_reason"}
    if block == "Y":
        required |= {"detection_status", "mask_source", "candidate_cat_count", "mask_fraction", "breed_margin",
                     "mask_height", "mask_width", "mask_bitmap_zlib"}
    if not required.issubset(table.column_names):
        report.update({
            "status": "quarantined", "automated_checks": "failed",
            "details": "Feature table lacks required identity, status, or diagnostic columns.",
        })
        return report, {}
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_images):
        errors.append("image IDs are duplicated or do not match the Stage 1 image inventory")
    if manifest.get("feature_file_sha256") != _file_hash(path):
        errors.append("feature file hash does not match the manifest")
    source_hashes = manifest.get("source_sha256", {})
    if "src/features.py" not in source_hashes:
        errors.append("feature adapter source hash is missing")
    for name, expected in source_hashes.items():
        source = root.joinpath(*name.split("/"))
        if not source.is_file() or _file_hash(source) != expected:
            errors.append(f"source hash mismatch: {name}")
    expected_names = (
        list(H_FEATURES) if block == "H" else
        [f"cnn_{index:03d}" for index in range(576)] if block == "C" else
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )
    if manifest.get("feature_names") != expected_names or manifest.get("feature_dimension") != len(expected_names):
        errors.append("feature names, order, or dimension differ from the registered schema")
    if any(name not in table.column_names for name in expected_names):
        errors.append("feature table is missing registered columns")
    else:
        positions = [table.column_names.index(name) for name in expected_names]
        if positions != sorted(positions):
            errors.append("feature table column order does not match the manifest")
        for name in expected_names:
            expected_type = pa.string() if block == "Y" and name in CATEGORICAL_FEATURES else pa.float32()
            if table.schema.field(name).type != expected_type:
                errors.append(f"unexpected feature dtype: {name}")
    if block == "C":
        checkpoint_value = manifest.get("checkpoint")
        checkpoint = Path(checkpoint_value) if isinstance(checkpoint_value, str) else None
        if checkpoint is None or not checkpoint.is_file() or _file_hash(checkpoint) != manifest.get("weight_sha256"):
            errors.append("CNN checkpoint is missing or has the wrong hash")
    if block == "Y":
        checkpoints = manifest.get("checkpoint", {})
        hashes = manifest.get("weight_sha256", {})
        for kind in ("detector", "clip"):
            value = checkpoints.get(kind) if isinstance(checkpoints, dict) else None
            checkpoint = Path(value) if isinstance(value, str) else None
            if checkpoint is None or not checkpoint.is_file() or _file_hash(checkpoint) != hashes.get(kind):
                errors.append(f"{kind} checkpoint is missing or has the wrong hash")

    by_id = {row["image_id"]: row for row in rows}
    usable: set[str] = set()
    failures: Counter[str] = Counter()
    if all(name in table.column_names for name in expected_names):
        for row in rows:
            image_id = row["image_id"]
            if row["source_sha256"] != expected_images.get(image_id):
                failures["source_image_hash_mismatch"] += 1
                continue
            status, reason = row["extraction_status"], row["failure_reason"]
            if status == "failed":
                if not reason:
                    failures["failure_without_reason"] += 1
                continue
            if status != "ok" or reason:
                failures["invalid_status_or_failure_reason"] += 1
                continue
            if block in {"H", "C"}:
                values = [row[name] for name in expected_names]
                if any(value is None or not math.isfinite(value) for value in values):
                    failures["non_finite_or_missing_successful_features"] += 1
                    continue
                if block == "C" and not math.isclose(math.sqrt(sum(v * v for v in values)), 1.0, abs_tol=1e-5):
                    failures["cnn_norm_not_one"] += 1
                    continue
                if block == "H":
                    bounded = (
                        "brightness", "contrast", "saturation_mean", "saturation_std",
                        "edge_density", "warm_pixel_ratio", "hue_mean", "hue_concentration",
                        "mean_r", "mean_g", "mean_b", "overexposed_frac", "underexposed_frac",
                    )
                    if any(not -1e-6 <= row[name] <= 1.000001 for name in bounded):
                        failures["handcrafted_value_out_of_range"] += 1
                        continue
                    if row["is_effectively_gray"] not in {0.0, 1.0} or row["aspect_ratio"] <= 0:
                        failures["invalid_handcrafted_geometry_or_gray_flag"] += 1
                        continue
            else:
                numeric = [row[name] for name in NUMERIC_FEATURES]
                if any(value is not None and not math.isfinite(value) for value in numeric):
                    failures["non_finite_y_measurement"] += 1
                    continue
                if row["detection_status"] == "no_detection":
                    if any(row[name] is not None for name in NUMERIC_FEATURES):
                        failures["no_detection_has_fabricated_measurement"] += 1
                        continue
                    if row["mask_bitmap_zlib"] is not None:
                        failures["no_detection_has_mask"] += 1
                        continue
                    if row["colour_primary"] != "no_detection" or row["colour_pattern"] != "no_detection":
                        failures["no_detection_not_explicitly_encoded"] += 1
                        continue
                elif row["detection_status"] not in {"cat_detected", "dog_fallback"}:
                    failures["invalid_detection_status"] += 1
                    continue
                else:
                    try:
                        mask = decode_mask(row["mask_bitmap_zlib"], row["mask_height"], row["mask_width"])
                    except (TypeError, ValueError, zlib.error):
                        failures["invalid_detector_mask"] += 1
                        continue
                    if row["mask_fraction"] is None or not math.isclose(
                        float(mask.mean()), row["mask_fraction"], abs_tol=1e-5
                    ):
                        failures["inconsistent_detector_mask_fraction"] += 1
                        continue
            usable.add(image_id)
    if failures:
        errors.append("structural failures: " + json.dumps(dict(failures), sort_keys=True))
    covered = sum(row["a_image"] in usable and row["b_image"] in usable for row in validation)
    coverage = covered / len(validation) if validation else 0.0
    if not validation:
        errors.append("strict-validation population is empty")
    elif coverage < 0.95:
        errors.append("usable endpoint coverage is below 95%")
    report.update({
        "status": "quarantined" if errors else "pending_human_review",
        "automated_checks": "failed" if errors else "passed",
        "successful_images": len(usable), "failed_images": len(rows) - len(usable),
        "covered_validation_responses": covered, "validation_coverage": coverage,
        "details": "; ".join(errors),
    })
    return report, by_id


def select_review_images(
    fitting_images: set[str], flags: dict[str, set[str]], seed: int
) -> list[tuple[str, str, str]]:
    ordered = sorted(fitting_images, key=lambda image: (rank_image(image, seed), image))
    base = ordered[:48]
    remaining = [image for image in ordered if image not in set(base) and flags.get(image)]
    priorities = (
        "no_detection", "H_unavailable", "C_unavailable", "Y_unavailable",
        "multiple_cats", "small_or_bbox_mask", "dog_fallback",
        "low_clip_breed_margin", "extreme_H",
    )
    flagged: list[str] = []
    for family in priorities:
        candidate = next((
            image for image in remaining if image not in flagged
            and any(flag == family or flag.startswith(family + ":") for flag in flags[image])
        ), None)
        if candidate is not None:
            flagged.append(candidate)
    flagged.extend(image for image in remaining if image not in flagged)
    selected = [(image, "all_fitting", ";".join(sorted(flags.get(image, set())))) for image in base]
    selected.extend((image, "flagged", ";".join(sorted(flags[image]))) for image in flagged[:16])
    return selected


def run_quality(root: Path, config: dict[str, Any]) -> None:
    output = root / "tables" / "feature_quality.csv"
    review = root / "artifacts" / "feature_review.csv"
    decisions = root / "artifacts" / "feature_admission_decisions.csv"
    review_manifest = root / "artifacts" / "feature_review_manifest.json"
    if output.exists() or review.exists() or decisions.exists() or review_manifest.exists():
        raise FileExistsError("Feature QA or review queue already exists; refusing to overwrite it.")
    _input_inventory(root)
    manifest = pq.read_table(root / "artifacts" / "split_manifest.parquet").to_pylist()
    rows = pq.read_table(
        root / "artifacts" / "rows_canonical.parquet",
        columns=["train_row_id", "a_image", "b_image", "partition"],
    ).to_pylist()
    graph = build_components(manifest)
    fitting_images = set(graph.parents)
    validation = [row for row in rows if row["partition"] == "validation"
                  and graph.supports(row["a_image"], row["b_image"])]
    raw_hashes = json.loads((root / "artifacts" / "raw_hashes.json").read_text(encoding="utf-8"))
    prefix = "dataset/images/"
    images = {name[len(prefix):]: details["sha256"]
              for name, details in raw_hashes["files"].items() if name.startswith(prefix)}
    images.update({name: "" for name in raw_hashes["unreadable_images"]})
    reports = []
    blocks = {}
    for block in ("H", "C", "Y"):
        report, records = audit_block(root, block, images, validation)
        reports.append(report)
        blocks[block] = records
    flags: dict[str, set[str]] = {}
    for image in fitting_images:
        current: set[str] = set()
        for block, records in blocks.items():
            if image not in records or records[image]["extraction_status"] != "ok":
                current.add(f"{block}_unavailable")
        y = blocks["Y"].get(image)
        if y and y["extraction_status"] == "ok":
            if y["detection_status"] == "no_detection":
                current.add("no_detection")
            if y["detection_status"] == "dog_fallback":
                current.add("dog_fallback")
            if y["candidate_cat_count"] > 1:
                current.add("multiple_cats")
            if y["mask_source"] == "bbox" or (
                y["mask_fraction"] is not None and y["mask_fraction"] < 0.01
            ):
                current.add("small_or_bbox_mask")
            if y["breed_margin"] is not None and y["breed_margin"] < 0.42:
                current.add("low_clip_breed_margin")
        flags[image] = current
    for name in H_FEATURES:
        values = [
            blocks["H"][image][name] for image in fitting_images
            if image in blocks["H"] and blocks["H"][image]["extraction_status"] == "ok"
            and blocks["H"][image][name] is not None
            and math.isfinite(blocks["H"][image][name])
        ]
        if not values:
            continue
        low, high = np.quantile(values, [0.01, 0.99])
        for image in fitting_images:
            record = blocks["H"].get(image)
            if record and record["extraction_status"] == "ok" and record[name] is not None:
                if record[name] < low or record[name] > high:
                    flags[image].add(f"extreme_H:{name}")
    selected = select_review_images(fitting_images, flags, config["seeds"]["feature_audit"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        fields = ["block", "status", "automated_checks", "successful_images", "failed_images",
                  "strict_validation_responses", "covered_validation_responses",
                  "validation_coverage", "details"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reports)
    review.parent.mkdir(parents=True, exist_ok=True)
    review_records = []
    with review.open("w", encoding="utf-8", newline="") as stream:
        fields = ["image_id", "image_path", "review_panel_path", "selection", "flags", "y_detection_status",
                  "y_mask_source", "y_attributes", "reviewer", "review_status",
                  "orientation", "cnn_crop_retention", "detection_mask", "visible_attributes",
                  "no_detection_behavior", "notes"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for image, selection, reasons in selected:
            y = blocks["Y"].get(image)
            attributes = {name: y.get(name) for name in CATEGORICAL_FEATURES} if y else {}
            for reviewer in ("Lu Yuqing", "Yuen Lok Yan"):
                record = {
                    "image_id": image, "image_path": "dataset/images/" + image,
                    "review_panel_path": "artifacts/review_panels/" +
                                         hashlib.sha256(image.encode("utf-8")).hexdigest() + ".jpg",
                    "selection": selection, "flags": reasons,
                    "y_detection_status": y.get("detection_status", "") if y else "",
                    "y_mask_source": y.get("mask_source", "") if y else "",
                    "y_attributes": json.dumps(attributes, ensure_ascii=False, sort_keys=True),
                    "reviewer": reviewer, "review_status": "pending",
                }
                writer.writerow(record)
                review_records.append(record)
    with decisions.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "block", "decision", "reviewer", "systematic_bugs_resolved", "reason",
        ])
        writer.writeheader()
        for block, reviewer in (("H", "Yuen Lok Yan"), ("C", "Yuen Lok Yan"), ("Y", "Lu Yuqing")):
            writer.writerow({
                "block": block, "decision": "pending", "reviewer": reviewer,
                "systematic_bugs_resolved": "pending", "reason": "",
            })
    metadata = {
        "audit_seed": config["seeds"]["feature_audit"],
        "image_count": len(selected),
        "review_rows": len(review_records),
        "reviewer_names": ["Lu Yuqing", "Yuen Lok Yan"],
        "selection_rule": "48 hash-ranked fitting images; up to 16 flagged fitting images selected by fixed flag-family priority then hash order",
        "flag_rules": "no detection, dog fallback, multiple cats, bbox/small mask, breed margin < 0.42, H outside fitting-photo 1st/99th percentiles",
        "immutable_review_sha256": review_fingerprint(review_records),
        "split_manifest_sha256": _file_hash(root / "artifacts" / "split_manifest.parquet"),
        "quality_code_sha256": _file_hash(root / "src" / "feature_quality.py"),
        "feature_file_sha256": {
            block: _file_hash(root / "features" / f"{block}.parquet")
            for block in ("H", "C", "Y") if (root / "features" / f"{block}.parquet").is_file()
        },
    }
    review_manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
