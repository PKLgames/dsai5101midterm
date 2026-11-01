"""Outcome-blind per-photo demographic support in the original G2 development set."""

import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from evidence_catalog.cat_demographic_support.analyze import (  # noqa: E402
    DEMOGRAPHICS, category, digest, load_structure, write_csv, write_json,
)


def snapshot():
    result = {}
    for directory, subdirs, names in os.walk(ROOT):
        base = Path(directory)
        if base == ROOT:
            subdirs[:] = [n for n in subdirs if n not in {".git", ".venv", ".cache"}]
        subdirs[:] = [n for n in subdirs if n not in {"__pycache__", "matplotlib_config"} and base/n != HERE]
        for name in names:
            path = base/name
            if path.is_file():
                result[str(path.relative_to(ROOT))] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return result


def compute(scopes):
    if set(scopes) != {"fit", "development"}:
        raise ValueError("Expected fit and development scopes.")
    photo_cats = {}
    for row in scopes["development"]:
        for side in ("a", "b"):
            photo, cat = row[f"{side}_image"], row[f"{side}_cat"]
            if photo in photo_cats and photo_cats[photo] != cat:
                raise ValueError(f"Photo maps to multiple cats: {photo}")
            photo_cats[photo] = cat
    photos = sorted({row[f"{side}_image"] for row in scopes["fit"] for side in ("a", "b")})
    if not photos or set(photos) != set(photo_cats):
        raise ValueError("Development photo universe differs from fitting photo universe.")
    categories = {name: sorted({category(row, name) for row in scopes["development"]}) for name in DEMOGRAPHICS}
    detail, summary, overlap, totals = [], [], [], {}
    for scope, rows in scopes.items():
        ids = [r["train_row_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate response IDs: {scope}")
        cells = defaultdict(lambda: {"n": 0, "within": 0, "between": 0, "pairs": set(), "opponents": set()})
        group_responses = Counter()
        pair_counts = defaultdict(Counter)
        for row in rows:
            if row["a_image"] == row["b_image"]:
                raise ValueError("A response must contain two distinct photos.")
            if row["comparison_type"] not in ("within_cat", "between_cats"):
                raise ValueError("Invalid comparison type.")
            for name in DEMOGRAPHICS:
                group = category(row, name)
                group_responses[name, group] += 1
                pair_counts[name, group][row["pair_id"]] += 1
                for side, other in (("a", "b"), ("b", "a")):
                    item = cells[name, group, row[f"{side}_image"]]
                    item["n"] += 1
                    item["within" if row["comparison_type"] == "within_cat" else "between"] += 1
                    item["pairs"].add(row["pair_id"])
                    item["opponents"].add(row[f"{other}_image"])
        totals[scope] = {"responses": len(rows), "pairs": len({r["pair_id"] for r in rows}),
                         "fitting_photo_universe": len(photos), "photo_appearances_per_demographic": 2*len(rows)}
        for name in DEMOGRAPHICS:
            if sum(group_responses[name, g] for g in categories[name]) != len(rows):
                raise ValueError("Group counts do not partition responses.")
            if sum(cells[name, g, p]["n"] for g in categories[name] for p in photos) != 2*len(rows):
                raise ValueError("Photo appearance totals do not reconcile.")
            for group in categories[name]:
                records = []
                for photo in photos:
                    item = cells[name, group, photo]
                    record = {"partition": scope, "demographic": name, "group": group,
                              "photo_id": photo, "cat_id": photo_cats[photo], "responses_with_photo": item["n"],
                              "distinct_photo_pairs": len(item["pairs"]), "distinct_opponent_photos": len(item["opponents"]),
                              "within_cat_responses": item["within"], "between_cats_responses": item["between"]}
                    detail.append(record)
                    records.append(record)
                vals = np.array([r["responses_with_photo"] for r in records])
                summary.append({"partition": scope, "demographic": name, "group": group, "photos_in_universe": len(photos),
                                "group_response_rows": group_responses[name, group], "photo_response_appearances": int(vals.sum()),
                                "photos_with_zero": int((vals == 0).sum()), "minimum": int(vals.min()),
                                "q25": float(np.quantile(vals, .25)), "median": float(np.median(vals)),
                                "q75": float(np.quantile(vals, .75)), "maximum": int(vals.max()), "mean": float(vals.mean()),
                                "photos_under_5_responses": int((vals < 5).sum()), "photos_under_10_responses": int((vals < 10).sum()),
                                "photos_under_20_responses": int((vals < 20).sum()),
                                "median_distinct_photo_pairs": float(np.median([r["distinct_photo_pairs"] for r in records])),
                                "within_cat_median": float(np.median([r["within_cat_responses"] for r in records])),
                                "photos_zero_within_cat": sum(r["within_cat_responses"] == 0 for r in records)})
            for a, b in combinations(categories[name], 2):
                ca, cb = pair_counts[name, a], pair_counts[name, b]
                shared = ca.keys() & cb.keys()
                overlap.append({"partition": scope, "demographic": name, "group_a": a, "group_b": b,
                                "photos_in_universe": len(photos),
                                "photos_with_any_in_both_groups": sum(cells[name, a, p]["n"] >= 1 and cells[name, b, p]["n"] >= 1 for p in photos),
                                "photos_with_at_least_5_in_both_groups": sum(cells[name, a, p]["n"] >= 5 and cells[name, b, p]["n"] >= 5 for p in photos),
                                "photos_with_at_least_10_in_both_groups": sum(cells[name, a, p]["n"] >= 10 and cells[name, b, p]["n"] >= 10 for p in photos),
                                "shared_pairs": len(shared), "shared_pairs_with_2_in_each_group": sum(ca[p] >= 2 and cb[p] >= 2 for p in shared),
                                "shared_pairs_with_5_in_each_group": sum(ca[p] >= 5 and cb[p] >= 5 for p in shared)})
    return detail, summary, overlap, {"partitions": totals, "demographic_categories": categories,
                                     "photo_counting": "One appearance per photo in each comparison; each response has two distinct photos, including within-cat comparisons.",
                                     "zero_cells": "All fitting photos crossed with every development category.",
                                     "outcome_columns_read": False, "test_rows_read": False, "models_fitted": False}


def plot(summary, overlap):
    with tempfile.TemporaryDirectory(prefix="photo_mpl_") as config:
        os.environ["MPLCONFIGDIR"] = config
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(HERE/"presentation_photo_demographic_support.pdf") as pdf:
            for scope in ("fit", "development"):
                rows = [r for r in summary if r["partition"] == scope]
                fig, axes = plt.subplots(1, 2, figsize=(15, 11))
                y = np.arange(len(rows))
                labels = [r["demographic"] + ": " + r["group"] for r in rows]
                axes[0].barh(y, [r["median"] for r in rows], color="#326d80")
                axes[1].barh(y, [100*r["photos_under_10_responses"]/r["photos_in_universe"] for r in rows], color="#a95c38")
                for ax in axes:
                    ax.set_yticks(y, labels, fontsize=8)
                    ax.invert_yaxis()
                    ax.grid(axis="x", alpha=.2)
                axes[0].set(xlabel="Median response appearances per photo", title="Typical per-photo group support")
                axes[1].set(xlabel="Photos with fewer than 10 responses (%)", title="Sparse per-photo group cells", xlim=(0, 100))
                fig.suptitle(f"Per-photo demographic support: {scope}", fontsize=16)
                fig.text(.04, .025, "All fitting photos, including zero cells. Responses are not necessarily unique people; 10 is illustrative, not a power threshold.", fontsize=9)
                fig.subplots_adjust(left=.18, right=.97, top=.92, bottom=.08, wspace=.55)
                pdf.savefig(fig)
                plt.close(fig)
            fig, ax = plt.subplots(figsize=(13, 7))
            ax.axis("off")
            r = next(r for r in overlap if r["partition"] == "development" and r["demographic"] == "gender" and r["group_a"] == "man" and r["group_b"] == "woman")
            lines = [f"Development: {r['photos_with_any_in_both_groups']}/{r['photos_in_universe']} photos appear in both gender groups.",
                     f"Development: {r['photos_with_at_least_10_in_both_groups']} photos have at least 10 responses in each group.",
                     f"Development: {r['shared_pairs']} photo pairs have at least one response in each group.",
                     "Shared photos need not have the same opponents in each group.",
                     "Counts are response records, not independent respondents, effects or power.",
                     "Fitting/development structural columns only; no outcomes read or models fitted."]
            fig.suptitle("Shared photo support is not matched-pair support", fontsize=15)
            for i, line in enumerate(lines):
                ax.text(.03, .86-i*.12, "- " + line, fontsize=11, transform=ax.transAxes)
            pdf.savefig(fig)
            plt.close(fig)


def run():
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    outputs = ["original_hashes_before.json", "per_photo_demographic_counts.csv", "group_support_summary.csv",
               "matched_group_support.csv", "support_summary.json", "presentation_photo_demographic_support.pdf", "analysis_provenance.json"]
    if any((HERE/name).exists() for name in outputs):
        raise FileExistsError("Output already exists.")
    freeze_path = ROOT/"artifacts"/"model_freeze.json"
    if digest(freeze_path) != (ROOT/"artifacts"/"model_freeze.sha256").read_text(encoding="ascii").strip():
        raise ValueError("G2 freeze seal mismatch.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    inputs = ["artifacts/model_freeze.json", "artifacts/model_freeze.sha256",
              "artifacts/rows_canonical.parquet", "artifacts/g2_eligibility.parquet"]
    for name in inputs[2:]:
        if digest(ROOT.joinpath(*name.split("/"))) != freeze["artifacts_sha256"][name]:
            raise ValueError(f"Frozen structural input changed: {name}")
    before = snapshot()
    write_json(HERE/outputs[0], before)
    detail, summary, overlap, totals = compute(load_structure())
    for name, records in zip(outputs[1:4], (detail, summary, overlap)):
        write_csv(HERE/name, records, list(records[0]))
    write_json(HERE/outputs[4], totals)
    plot(summary, overlap)
    if snapshot() != before:
        raise RuntimeError("Pre-existing project files changed during analysis.")
    write_json(HERE/outputs[6], {"status": "completed_outcome_blind_descriptive",
                                "input_sha256": {name: digest(ROOT.joinpath(*name.split("/"))) for name in inputs},
                                "outputs_sha256": {name: digest(HERE/name) for name in outputs[1:6]},
                                "original_snapshot_sha256": digest(HERE/outputs[0]), "original_files_unchanged": len(before),
                                "per_photo_rows": len(detail), "group_summary_rows": len(summary), "matched_group_rows": len(overlap),
                                "outcome_columns_read": False, "test_rows_read": False, "models_fitted": False})
    print(json.dumps({"totals": totals, "per_photo_rows": len(detail), "summary_rows": len(summary),
                      "comparison_rows": len(overlap)}, indent=2))


if __name__ == "__main__":
    run()
