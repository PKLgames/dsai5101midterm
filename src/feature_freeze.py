"""Freeze feature admission and matched populations after human review."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .feature_quality import audit_block, build_components, review_fingerprint
from .features import _file_hash, _input_inventory

CRITERIA = (
    "orientation", "cnn_crop_retention", "detection_mask", "visible_attributes",
    "no_detection_behavior",
)
REVIEW_VALUES = {"pass", "issue", "not_visually_verifiable", "not_applicable"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_reviews(root: Path, fitting_images: set[str]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    reviews = _read_csv(root / "artifacts" / "feature_review.csv")
    meta = json.loads((root / "artifacts" / "feature_review_manifest.json").read_text(encoding="utf-8"))
    if review_fingerprint(reviews) != meta["immutable_review_sha256"]:
        raise ValueError("The fixed review sample or reviewer assignments were changed.")
    if len(reviews) != meta["review_rows"] or len({row["image_id"] for row in reviews}) != meta["image_count"]:
        raise ValueError("Review rows or image count do not match the recorded selection.")
    if _file_hash(root / "artifacts" / "split_manifest.parquet") != meta["split_manifest_sha256"]:
        raise ValueError("The split changed after review selection.")
    if _file_hash(root / "src" / "feature_quality.py") != meta["quality_code_sha256"]:
        raise ValueError("The quality code changed after review selection.")
    for block, digest in meta["feature_file_sha256"].items():
        if _file_hash(root / "features" / f"{block}.parquet") != digest:
            raise ValueError(f"{block} features changed after review selection.")
    per_image: dict[str, set[str]] = defaultdict(set)
    for row in reviews:
        image, reviewer = row["image_id"], row["reviewer"]
        if image not in fitting_images:
            raise ValueError(f"Review sample contains a non-fitting image: {image}")
        if reviewer in per_image[image]:
            raise ValueError(f"Duplicate reviewer for image {image}: {reviewer}")
        per_image[image].add(reviewer)
        if row["review_status"] != "reviewed":
            raise ValueError(f"Human review is not complete for {image}: {reviewer}")
        if not (root.joinpath(*row["review_panel_path"].split("/"))).is_file():
            raise ValueError(f"Review panel is missing for {image}")
        if any(row[field] not in REVIEW_VALUES for field in CRITERIA):
            raise ValueError(f"All review criteria need explicit valid values for {image}: {reviewer}")
        if any(row[field] == "issue" for field in CRITERIA) and not row["notes"].strip():
            raise ValueError(f"Review issues need explanatory notes for {image}: {reviewer}")
    expected_reviewers = set(meta["reviewer_names"])
    if any(reviewers != expected_reviewers for reviewers in per_image.values()):
        raise ValueError("Each selected image requires both assigned independent reviewers.")
    return reviews, meta


def validate_admissions(
    decisions: list[dict[str, str]], reports: dict[str, dict[str, Any]]
) -> list[str]:
    if len(decisions) != 3 or {row["block"] for row in decisions} != {"H", "C", "Y"}:
        raise ValueError("Exactly one admission decision is required for each of H, C, and Y.")
    admitted = []
    for row in decisions:
        block, choice = row["block"], row["decision"]
        if choice not in {"admitted", "quarantined"}:
            raise ValueError(f"Admission decision for {block} is still pending or invalid.")
        if not row["reviewer"].strip() or not row["reason"].strip():
            raise ValueError(f"Admission decision for {block} needs a reviewer and rationale.")
        if choice == "admitted":
            if reports[block]["automated_checks"] != "passed":
                raise ValueError(f"{block} cannot be admitted because automated checks did not pass.")
            if row["systematic_bugs_resolved"] != "yes":
                raise ValueError(f"{block} cannot be admitted with unresolved systematic bugs.")
            admitted.append(block)
    return admitted


def compute_population(
    rows: list[dict[str, Any]], split: list[dict[str, Any]],
    usable: set[str], admitted: list[str],
) -> list[dict[str, Any]]:
    full_graph = build_components(split)
    matched_graph = build_components(split, usable)
    result = []
    for row in rows:
        a, b, partition = row["a_image"], row["b_image"], row["partition"]
        feature_usable = bool(admitted) and a in usable and b in usable
        is_heldout = partition in {"validation", "test"}
        result.append({
            "train_row_id": row["train_row_id"], "partition": partition,
            "full_strict_g2": is_heldout and full_graph.supports(a, b),
            "common_visual_fit": partition == "fit" and feature_usable,
            "common_visual_strict_g2": is_heldout and feature_usable and matched_graph.supports(a, b),
        })
    return result


def run_freeze(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifacts" / "common_population_manifest.parquet"
    report_path = root / "artifacts" / "common_population_report.json"
    if manifest_path.exists() or report_path.exists():
        raise FileExistsError("Common-population artifacts already exist; refusing to overwrite them.")
    _input_inventory(root)
    split = pq.read_table(root / "artifacts" / "split_manifest.parquet").to_pylist()
    rows = pq.read_table(
        root / "artifacts" / "rows_canonical.parquet",
        columns=["train_row_id", "a_image", "b_image", "partition"],
    ).to_pylist()
    full_graph = build_components(split)
    fitting_images = set(full_graph.parents)
    reviews, meta = validate_reviews(root, fitting_images)
    validation = [row for row in rows if row["partition"] == "validation"
                  and full_graph.supports(row["a_image"], row["b_image"])]
    raw_hashes = json.loads((root / "artifacts" / "raw_hashes.json").read_text(encoding="utf-8"))
    prefix = "dataset/images/"
    images = {name[len(prefix):]: details["sha256"]
              for name, details in raw_hashes["files"].items() if name.startswith(prefix)}
    images.update({name: "" for name in raw_hashes["unreadable_images"]})
    reports, records = {}, {}
    for block in ("H", "C", "Y"):
        reports[block], records[block] = audit_block(root, block, images, validation)
    decisions = _read_csv(root / "artifacts" / "feature_admission_decisions.csv")
    admitted = validate_admissions(decisions, reports)
    usable = set(images)
    if admitted:
        for block in admitted:
            usable &= {key for key, row in records[block].items() if row["extraction_status"] == "ok"}
    else:
        usable.clear()
    result = compute_population(rows, split, usable, admitted)
    counts = Counter((row["partition"], "fit" if row["common_visual_fit"] else
                      "strict" if row["common_visual_strict_g2"] else "excluded") for row in result)
    report = {
        "admitted_blocks": admitted,
        "status": "no_admitted_visual_features" if not admitted else (
            "insufficient_common_visual_population" if not counts[("validation", "strict")]
            or not counts[("test", "strict")] else "available_adequacy_review_required"
        ),
        "common_fitting_responses": counts[("fit", "fit")],
        "common_strict_validation_responses": counts[("validation", "strict")],
        "common_strict_test_responses": counts[("test", "strict")],
        "review_images": meta["image_count"],
        "review_rows": len(reviews),
        "review_issue_fields": sum(row[field] == "issue" for row in reviews for field in CRITERIA),
        "admission_decisions": decisions,
        "feature_reports": reports,
        "note": "Population membership is structural; no held-out outcomes were read. Review sample-size adequacy before model selection.",
    }
    schema = pa.schema([
        ("train_row_id", pa.string()), ("partition", pa.string()), ("full_strict_g2", pa.bool_()),
        ("common_visual_fit", pa.bool_()), ("common_visual_strict_g2", pa.bool_()),
    ])
    pq.write_table(pa.Table.from_pylist(result, schema=schema), manifest_path)
    report["population_manifest_sha256"] = _file_hash(manifest_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
