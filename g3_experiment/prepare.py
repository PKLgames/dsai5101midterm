"""Outcome-blind retrospective G3 reservation and coverage gate."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.split import allocate_pairs, collect_pairs  # noqa: E402


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def snapshot(root: Path, excluded: set[str]) -> dict[str, dict[str, Any]]:
    entries = {}
    for directory, subdirs, files in os.walk(root):
        base = Path(directory)
        if base == root:
            subdirs[:] = [name for name in subdirs if name not in excluded]
        for name in files:
            path = base / name
            relative = path.relative_to(root)
            if relative.parts[0] not in excluded and path.is_file():
                entries[str(relative)] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return entries


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1 to protect the G2 tree.")
    outputs = [
        "root_hashes_before.json", "evidence_hashes_before.json", "photo_reservation.csv",
        "core_split.csv", "row_allocation.csv", "coverage.json",
    ]
    if any((EXPERIMENT / name).exists() for name in outputs):
        raise FileExistsError("Preparation outputs already exist; refusing to replace them.")
    write_json(EXPERIMENT / "root_hashes_before.json", snapshot(
        ROOT, {".git", ".venv", ".cache", "Presentation_Evidence", "g3_experiment"}))
    write_json(EXPERIMENT / "evidence_hashes_before.json", snapshot(ROOT / "Presentation_Evidence", set()))
    protocol = json.loads((EXPERIMENT / "protocol.json").read_text(encoding="utf-8"))
    if protocol["status"] != "retrospective_exploratory_not_preregistered":
        raise ValueError("Unexpected experiment scope.")

    fields = ["train_row_id", "pair_id", "a_image", "b_image", "a_cat", "b_cat",
              "comparison_type", "pet_care"]
    rows = pq.read_table(ROOT / "artifacts" / "rows_canonical.parquet", columns=fields).to_pylist()
    if len({r["train_row_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate response IDs.")
    audited = json.loads((ROOT / "artifacts" / "raw_hashes.json").read_text(encoding="utf-8"))
    photos = sorted({row[f"{side}_image"] for row in rows for side in ("a", "b")})
    photo_hash = {photo: audited["files"]["dataset/images/" + photo]["sha256"] for photo in photos}
    groups = sorted(set(photo_hash.values()), key=lambda h: (
        hashlib.sha256(("retrospective-g3-20261028:" + h).encode("utf-8")).hexdigest(), h))
    n = max(1, math.floor(len(groups) * 0.05))
    if 2 * n >= len(groups):
        raise ValueError("Insufficient independent photo content groups.")
    group_partition = {group: ("test" if i < n else "validation" if i < 2*n else "core")
                       for i, group in enumerate(groups)}
    assignment = {photo: group_partition[photo_hash[photo]] for photo in photos}
    feature_index = {}
    for block in ("H", "C", "Y"):
        records = pq.read_table(ROOT / "features" / f"{block}.parquet",
                                columns=["image_id", "source_sha256", "extraction_status"]).to_pylist()
        index = {r["image_id"]: r for r in records}
        if len(index) != len(records) or not set(photos) <= set(index):
            raise ValueError(f"{block} feature IDs are duplicate or incomplete.")
        for photo in photos:
            if index[photo]["source_sha256"] != photo_hash[photo]:
                raise ValueError(f"{block} source image hash mismatch: {photo}")
        feature_index[block] = index

    def usable(photo: str) -> bool:
        return all(feature_index[b][photo]["extraction_status"] == "ok" for b in ("H", "C", "Y"))

    core = [r for r in rows if assignment[r["a_image"]] == assignment[r["b_image"]] == "core"
            and usable(r["a_image"]) and usable(r["b_image"])]
    core_ids = {row["train_row_id"] for row in core}
    pairs = collect_pairs(core)
    split = allocate_pairs(pairs, seed=20261028)
    pair_partition = {r["pair_id"]: r["partition"] for r in split}
    fit = [r for r in core if pair_partition[r["pair_id"]] == "fit"]
    fitting_photos = {r[f"{side}_image"] for r in fit for side in ("a", "b")}
    fitting_cats = {r[f"{side}_cat"] for r in fit for side in ("a", "b")}
    if any(assignment[p] != "core" for p in fitting_photos):
        raise ValueError("Reserved photo entered the separate fitting population.")
    by_id = {}
    for row in rows:
        a, b = assignment[row["a_image"]], assignment[row["b_image"]]
        heldout = ""
        if a != "core" and b != "core":
            category, reason = "quarantine", "both_endpoints_reserved"
        elif a == b == "core":
            category = "core_" + pair_partition[row["pair_id"]] if row["train_row_id"] in core_ids else "excluded"
            reason = "" if category != "excluded" else "unusable_feature"
        else:
            side = "a" if a != "core" else "b"
            other = "b" if side == "a" else "a"
            heldout = row[f"{side}_image"]
            category = assignment[heldout]
            failures = []
            if row[f"{other}_image"] not in fitting_photos:
                failures.append("nonreserved_photo_not_in_fit")
            if row[f"{side}_cat"] not in fitting_cats or row[f"{other}_cat"] not in fitting_cats:
                failures.append("cat_not_in_fit")
            if not usable(row["a_image"]) or not usable(row["b_image"]):
                failures.append("unusable_feature")
            reason = ";".join(failures)
            if failures:
                category = "excluded_" + category
        by_id[row["train_row_id"]] = {"train_row_id": row["train_row_id"], "pair_id": row["pair_id"],
                                      "allocation": category, "held_out_photo": heldout, "reason": reason}
    if len(by_id) != len(rows):
        raise ValueError("Allocation does not cover all responses.")
    summary = {}
    for name in ("validation", "test"):
        selected = [r for r in by_id.values() if r["allocation"] == name]
        counts = {"responses": len(selected), "pairs": len({r["pair_id"] for r in selected}),
                  "held_out_photos": len({r["held_out_photo"] for r in selected})}
        counts["coverage_passed"] = all(counts[k] >= v for k, v in
                                        protocol["eligibility"]["coverage_gate_per_partition"].items())
        summary[name] = counts
    coverage = {
        "status": "coverage_passed" if all(v["coverage_passed"] for v in summary.values()) else "insufficient_coverage",
        "outcomes_read": False, "training_photo_count": len(photos), "content_groups": len(groups),
        "reserved_groups_per_partition": n, "core_fitting_responses": len(fit),
        "core_fitting_pairs": len({r["pair_id"] for r in fit}), "core_fitting_photos": len(fitting_photos),
        "core_fitting_cats": len(fitting_cats), "allocation_counts": dict(sorted(Counter(
            r["allocation"] for r in by_id.values()).items())), "g3": summary,
        "protocol_sha256": digest(EXPERIMENT / "protocol.json"),
    }
    photo_rows = [{"image_id": p, "content_sha256": photo_hash[p], "reservation": assignment[p],
                   "feature_usable": usable(p)} for p in photos]
    write_csv(EXPERIMENT / "photo_reservation.csv", photo_rows, list(photo_rows[0]))
    write_csv(EXPERIMENT / "core_split.csv", split, list(split[0]))
    write_csv(EXPERIMENT / "row_allocation.csv", list(by_id.values()),
              ["train_row_id", "pair_id", "allocation", "held_out_photo", "reason"])
    write_json(EXPERIMENT / "coverage.json", coverage)
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    prepare()
