"""Outcome-blind cat-disjoint retrospective G4 reservation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from g3_experiment.prepare import digest, snapshot, write_csv, write_json  # noqa: E402
from src.split import allocate_pairs, collect_pairs  # noqa: E402


def cat_groups(rows: list[dict[str, Any]], hashes: dict[str, str]) -> list[tuple[str, ...]]:
    cats = {r[f"{side}_cat"] for r in rows for side in ("a", "b")}
    parent = {cat: cat for cat in cats}

    def find(cat: str) -> str:
        while parent[cat] != cat:
            parent[cat] = parent[parent[cat]]
            cat = parent[cat]
        return cat

    image_owner = {}
    content_owner = {}
    for row in rows:
        for side in ("a", "b"):
            cat, image = row[f"{side}_cat"], row[f"{side}_image"]
            if image in image_owner and image_owner[image] != cat:
                raise ValueError(f"Image has inconsistent cat IDs: {image}")
            image_owner[image] = cat
            content = hashes[image]
            if content in content_owner:
                a, b = find(cat), find(content_owner[content])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                content_owner[content] = cat
    groups: dict[str, list[str]] = {}
    for cat in sorted(cats):
        groups.setdefault(find(cat), []).append(cat)
    return [tuple(group) for group in groups.values()]


def reserve_groups(groups: list[tuple[str, ...]]) -> dict[str, str]:
    def encoded(group: tuple[str, ...]) -> str:
        return json.dumps(list(group), separators=(",", ":"), ensure_ascii=True)
    ordered = sorted(groups, key=lambda group: (
        hashlib.sha256(("retrospective-g4-20261028:" + encoded(group)).encode("utf-8")).hexdigest(),
        encoded(group)))
    size = max(1, math.floor(.10 * len(ordered)))
    if 2 * size >= len(ordered):
        raise ValueError("Too few independent cat groups.")
    return {cat: "test" if i < size else "validation" if i < 2*size else "core"
            for i, group in enumerate(ordered) for cat in group}


def run() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    names = ("root_hashes_before.json", "evidence_hashes_before.json", "cat_reservation.csv",
             "core_split.csv", "row_allocation.csv", "coverage.json")
    if any((HERE / name).exists() for name in names):
        raise FileExistsError("G4 reservation already started; refusing to overwrite.")
    original = snapshot(ROOT, {".git", ".venv", ".cache", "Presentation_Evidence", "g4_experiment"})
    evidence = snapshot(ROOT / "Presentation_Evidence", set())
    write_json(HERE / "root_hashes_before.json", original)
    write_json(HERE / "evidence_hashes_before.json", evidence)
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    if protocol["status"] != "retrospective_exploratory_not_preregistered":
        raise ValueError("Unexpected G4 protocol status.")
    g2 = ROOT / "artifacts" / "model_freeze.json"
    if digest(g2) != (ROOT / "artifacts" / "model_freeze.sha256").read_text(encoding="ascii").strip():
        raise ValueError("G2 freeze seal mismatch.")
    g3 = ROOT / "g3_experiment"
    if digest(g3 / "freeze.json") != (g3 / "freeze.sha256").read_text(encoding="ascii").strip():
        raise ValueError("G3 freeze seal mismatch.")
    g3_freeze = json.loads((g3 / "freeze.json").read_text(encoding="utf-8"))
    for name, expected in g3_freeze["experiment_sha256"].items():
        if digest(g3 / name) != expected:
            raise ValueError(f"G3 frozen artifact changed: {name}")
    for name, expected in g3_freeze["root_input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"G3 frozen input changed: {name}")
    g3_evaluation = json.loads((g3 / "evaluation_provenance.json").read_text(encoding="utf-8"))
    if g3_evaluation["freeze_sha256"] != digest(g3 / "freeze.json"):
        raise ValueError("G3 evaluation freeze mismatch.")
    for name, expected in g3_evaluation["output_sha256"].items():
        if digest(g3 / name) != expected:
            raise ValueError(f"G3 evaluation output changed: {name}")
    fields = ["train_row_id", "pair_id", "a_image", "b_image", "a_cat", "b_cat",
              "comparison_type", "pet_care", "partition"]
    rows = pq.read_table(ROOT / "artifacts" / "rows_canonical.parquet", columns=fields).to_pylist()
    if len({r["train_row_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate training response IDs.")
    if any((r["a_cat"] == r["b_cat"]) != (r["comparison_type"] == "within_cat") for r in rows):
        raise ValueError("Comparison type and cat IDs disagree.")
    photos = {r[f"{s}_image"] for r in rows for s in ("a", "b")}
    audited = json.loads((ROOT / "artifacts" / "raw_hashes.json").read_text(encoding="utf-8"))
    hashes = {p: audited["files"]["dataset/images/" + p]["sha256"] for p in photos}
    groups = cat_groups(rows, hashes)
    reservation = reserve_groups(groups)
    feature_index = {}
    for block in ("H", "C", "Y"):
        records = pq.read_table(ROOT / "features" / f"{block}.parquet",
                                columns=["image_id", "source_sha256", "extraction_status"]).to_pylist()
        index = {r["image_id"]: r for r in records}
        if len(index) != len(records) or not photos <= set(index):
            raise ValueError(f"{block} features have missing or duplicate photo IDs.")
        if any(index[p]["source_sha256"] != hashes[p] for p in photos):
            raise ValueError(f"{block} feature source hashes disagree with the image audit.")
        feature_index[block] = index

    def usable(photo: str) -> bool:
        return all(feature_index[b][photo]["extraction_status"] == "ok" for b in ("H", "C", "Y"))

    core = [r for r in rows if reservation[r["a_cat"]] == reservation[r["b_cat"]] == "core"
            and usable(r["a_image"]) and usable(r["b_image"])]
    core_ids = {row["train_row_id"] for row in core}
    pairs = collect_pairs(core)
    split = allocate_pairs(pairs, seed=20261028)
    assigned = {r["pair_id"]: r["partition"] for r in split}
    fit = [r for r in core if assigned[r["pair_id"]] == "fit"]
    fitting_cats = {r[f"{s}_cat"] for r in fit for s in ("a", "b")}
    fitting_photos = {r[f"{s}_image"] for r in fit for s in ("a", "b")}
    if any(reservation[c] != "core" for c in fitting_cats):
        raise ValueError("Held-out cat entered the separate fitting population.")
    allocation = []
    for row in rows:
        a, b = reservation[row["a_cat"]], reservation[row["b_cat"]]
        heldout = ""
        if a == b == "core":
            category = "core_" + assigned[row["pair_id"]] if row["train_row_id"] in core_ids else "excluded_core"
            reason = "" if category != "excluded_core" else "unusable_feature"
        elif a != "core" and b != "core":
            category, reason = "quarantine", "two_reserved_endpoints_or_same_reserved_cat"
        else:
            side = "a" if a != "core" else "b"
            other = "b" if side == "a" else "a"
            heldout = row[f"{side}_cat"]
            category = reservation[heldout]
            failures = []
            if row[f"{other}_cat"] not in fitting_cats:
                failures.append("known_cat_not_in_fit")
            if not usable(row["a_image"]) or not usable(row["b_image"]):
                failures.append("unusable_feature")
            reason = ";".join(failures)
            if failures:
                category = "excluded_" + category
        allocation.append({"train_row_id": row["train_row_id"], "pair_id": row["pair_id"],
                           "allocation": category, "held_out_cat": heldout, "reason": reason})
    summary = {}
    gate = protocol["coverage_gate_per_partition"]
    for name in ("validation", "test"):
        selected = [r for r in allocation if r["allocation"] == name]
        stats = {"responses": len(selected), "pairs": len({r["pair_id"] for r in selected}),
                 "held_out_cats": len({r["held_out_cat"] for r in selected})}
        stats["coverage_passed"] = all(stats[k] >= gate[k] for k in ("responses", "pairs", "held_out_cats"))
        summary[name] = stats
    coverage = {"status": "coverage_passed" if all(v["coverage_passed"] for v in summary.values()) else "insufficient_coverage",
                "outcomes_read": False, "training_cats": len(reservation), "duplicate_aware_cat_groups": len(groups),
                "core_fitting_responses": len(fit), "core_fitting_pairs": len({r["pair_id"] for r in fit}),
                "core_fitting_cats": len(fitting_cats), "core_fitting_photos": len(fitting_photos),
                "allocation_counts": dict(sorted(Counter(r["allocation"] for r in allocation).items())),
                "g4": summary, "protocol_sha256": digest(HERE / "protocol.json")}
    cat_rows = [{"cat_id": cat, "reservation": reservation[cat]} for cat in sorted(reservation)]
    write_csv(HERE / "cat_reservation.csv", cat_rows, ["cat_id", "reservation"])
    write_csv(HERE / "core_split.csv", split, list(split[0]))
    write_csv(HERE / "row_allocation.csv", allocation, ["train_row_id", "pair_id", "allocation", "held_out_cat", "reason"])
    write_json(HERE / "coverage.json", coverage)
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    run()
