"""Outcome-blind per-cat demographic support from the original G2 development split."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from g3_experiment.prepare import digest, write_csv, write_json  # noqa: E402
from src.audit import CONTEXT  # noqa: E402

DEMOGRAPHICS = ("gender", "pet_history", "pet_care", "pet_contact", "future_pet", "mbti_ie")


def snapshots() -> dict[str, dict[str, Any]]:
    entries = {}
    for directory, subdirs, names in os.walk(ROOT):
        base = Path(directory)
        if base == ROOT:
            subdirs[:] = [name for name in subdirs if name not in {".git", ".venv", ".cache"}]
        subdirs[:] = [name for name in subdirs if name not in {"__pycache__", "matplotlib_config"}
                      and base / name != HERE]
        for name in names:
            path = base / name
            if path.is_file():
                entries[str(path.relative_to(ROOT))] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return entries


def category(row: dict[str, Any], name: str) -> str:
    value = row[name]
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string; received {value!r}.")
    if value == "":
        return "<missing>"
    if value == "<missing>" or (name in CONTEXT and value not in CONTEXT[name]):
        raise ValueError(f"Unexpected {name} category: {value!r}")
    return value


def load_structure() -> dict[str, list[dict[str, Any]]]:
    fields = ["train_row_id", "partition", "pair_id", "a_image", "b_image", "a_cat", "b_cat",
              "comparison_type", *DEMOGRAPHICS]
    rows = pq.read_table(ROOT / "artifacts" / "rows_canonical.parquet", columns=fields,
                         filters=[("partition", "in", ["fit", "validation"])]).to_pylist()
    masks = pq.read_table(ROOT / "artifacts" / "g2_eligibility.parquet",
                          columns=["train_row_id", "partition", "common_visual_fit", "common_strict_g2"],
                          filters=[("partition", "in", ["fit", "validation"])]).to_pylist()
    index = {r["train_row_id"]: r for r in masks}
    if len(index) != len(masks) or len(index) != len(rows) or len({r["train_row_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate or incomplete original development masks.")
    fit, val = [], []
    for row in rows:
        identity = row["train_row_id"]
        if identity not in index or index[identity]["partition"] != row["partition"]:
            raise ValueError(f"Development mask mismatch: {identity}")
        if not row["a_image"] or not row["b_image"] or row["a_image"] == row["b_image"]:
            raise ValueError(f"Invalid photo pair: {identity}")
        if row["comparison_type"] not in ("within_cat", "between_cats") or (
            (row["a_cat"] == row["b_cat"]) != (row["comparison_type"] == "within_cat")
        ):
            raise ValueError(f"Comparison type disagrees with cat IDs: {identity}")
        for name in DEMOGRAPHICS:
            category(row, name)
        if row["partition"] == "fit" and index[identity]["common_visual_fit"]:
            fit.append(row)
        elif row["partition"] == "validation" and index[identity]["common_strict_g2"]:
            val.append(row)
    if not fit or not val:
        raise ValueError("Matched fitting or validation structure is empty.")
    if {r["pair_id"] for r in fit} & {r["pair_id"] for r in val}:
        raise ValueError("Original fitting and validation pairs overlap.")
    return {"fit": fit, "development": fit+val}


def compute(scopes: dict[str, list[dict[str, Any]]]) -> tuple[list[dict], list[dict], list[dict], dict]:
    if set(scopes) != {"fit", "development"}:
        raise ValueError("Expected fitting and development scopes.")
    cats = sorted({row[f"{side}_cat"] for row in scopes["fit"] for side in ("a", "b")})
    dev_cats = {row[f"{side}_cat"] for row in scopes["development"] for side in ("a", "b")}
    if not cats or set(cats) != dev_cats:
        raise ValueError("Development cats differ from fitting cats; specify a new denominator before continuing.")
    categories = {name: sorted({category(row, name) for row in scopes["development"]}) for name in DEMOGRAPHICS}
    detail, summary, comparisons, totals = [], [], [], {}
    for partition, rows in scopes.items():
        ids = [r["train_row_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate response IDs in {partition}.")
        cells = defaultdict(lambda: {"responses": 0, "within_cat_responses": 0, "between_cats_responses": 0,
                                     "pairs": set(), "photos": set(), "opponent_cats": set()})
        group_responses = Counter()
        pair_counts = defaultdict(Counter)
        for row in rows:
            cats_in_row = set((row["a_cat"], row["b_cat"]))
            for name in DEMOGRAPHICS:
                group = category(row, name)
                group_responses[(name, group)] += 1
                pair_counts[(name, group)][row["pair_id"]] += 1
                for cat in cats_in_row:
                    item = cells[(name, group, cat)]
                    item["responses"] += 1
                    item[row["comparison_type"] + "_responses"] += 1
                    item["pairs"].add(row["pair_id"])
                    for side in ("a", "b"):
                        if row[f"{side}_cat"] == cat:
                            item["photos"].add(row[f"{side}_image"])
                        else:
                            item["opponent_cats"].add(row[f"{side}_cat"])
        within = sum(row["comparison_type"] == "within_cat" for row in rows)
        expected_appearances = 2*len(rows)-within
        totals[partition] = {"responses": len(rows), "pairs": len({r["pair_id"] for r in rows}),
                             "within_cat_responses": within, "between_cats_responses": len(rows)-within,
                             "fitting_cat_universe": len(cats),
                             "cat_response_appearances_per_demographic": expected_appearances}
        for name in DEMOGRAPHICS:
            if sum(group_responses[(name, value)] for value in categories[name]) != len(rows):
                raise ValueError(f"Demographic categories do not partition responses: {name}")
            if sum(cells[(name, group, cat)]["responses"] for group in categories[name] for cat in cats) != expected_appearances:
                raise ValueError(f"Cat-appearance totals do not reconcile: {partition}/{name}")
            for group in categories[name]:
                values = []
                group_detail = []
                for cat in cats:
                    item = cells[(name, group, cat)]
                    count = item["responses"]
                    values.append(count)
                    record = {"partition": partition, "demographic": name, "group": group, "cat_id": cat,
                              "responses_with_cat": count, "distinct_photo_pairs": len(item["pairs"]),
                              "distinct_photos_of_cat": len(item["photos"]),
                              "distinct_other_opponent_cats": len(item["opponent_cats"]),
                              "within_cat_responses": item["within_cat_responses"],
                              "between_cats_responses": item["between_cats_responses"]}
                    detail.append(record)
                    group_detail.append(record)
                arr = np.asarray(values)
                summary.append({"partition": partition, "demographic": name, "group": group,
                                "cats_in_universe": len(cats), "group_response_rows": group_responses[(name, group)],
                                "cat_response_appearances": int(arr.sum()),
                                "cats_with_zero": int(np.sum(arr == 0)), "cats_with_any": int(np.sum(arr > 0)),
                                "minimum": int(arr.min()), "q25": float(np.quantile(arr, .25)),
                                "median": float(np.quantile(arr, .5)), "q75": float(np.quantile(arr, .75)),
                                "maximum": int(arr.max()), "mean": float(arr.mean()),
                                "cats_under_5_responses": int(np.sum(arr < 5)),
                                "cats_under_10_responses": int(np.sum(arr < 10)),
                                "cats_under_20_responses": int(np.sum(arr < 20)),
                                "cats_at_least_5_responses": int(np.sum(arr >= 5)),
                                "median_distinct_photo_pairs": float(np.median([r["distinct_photo_pairs"] for r in group_detail])),
                                "median_distinct_photos_of_cat": float(np.median([r["distinct_photos_of_cat"] for r in group_detail]))})
            for a, b in combinations(categories[name], 2):
                pa, pb = pair_counts[(name, a)], pair_counts[(name, b)]
                shared = set(pa) & set(pb)
                comparisons.append({"partition": partition, "demographic": name, "group_a": a, "group_b": b,
                                    "cats_in_universe": len(cats),
                                    "cats_with_any_in_both_groups": sum(cells[(name, a, cat)]["responses"] > 0 and
                                                                          cells[(name, b, cat)]["responses"] > 0 for cat in cats),
                                    "cats_with_at_least_5_in_both_groups": sum(cells[(name, a, cat)]["responses"] >= 5 and
                                                                               cells[(name, b, cat)]["responses"] >= 5 for cat in cats),
                                    "cats_with_at_least_10_in_both_groups": sum(cells[(name, a, cat)]["responses"] >= 10 and
                                                                                cells[(name, b, cat)]["responses"] >= 10 for cat in cats),
                                    "pairs_with_any_in_group_a": len(pa), "pairs_with_any_in_group_b": len(pb),
                                    "shared_pairs": len(shared),
                                    "shared_pairs_with_2_responses_in_each_group": sum(pa[p] >= 2 and pb[p] >= 2 for p in shared),
                                    "shared_pairs_with_5_responses_in_each_group": sum(pa[p] >= 5 and pb[p] >= 5 for p in shared)})
    return detail, summary, comparisons, {"partitions": totals, "demographic_categories": categories,
                                          "response_counting": "One response per distinct cat appearing in it; within-cat responses count once for that cat; between-cat responses count once for each of two cats.",
                                          "count_units": "Response records, not unique respondents; pair counts use unordered photo pair IDs.",
                                          "zero_cells": "Every fitting cat crossed with every observed development category, separately for both scopes.",
                                          "threshold_interpretation": "5, 10 and 20 responses per cat/group are descriptive cutoffs, not power guarantees. Five responses per pair/group is the existing group-contrast operational rule.",
                                          "outcome_columns_read": False, "test_rows_read": False, "models_fitted": False}


def plot(summary: list[dict], comparisons: list[dict]) -> None:
    with tempfile.TemporaryDirectory(prefix="mpl_", dir=HERE) as config:
        os.environ["MPLCONFIGDIR"] = config
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(HERE / "presentation_cat_demographic_support.pdf") as pdf:
            for partition in ("fit", "development"):
                rows = [r for r in summary if r["partition"] == partition]
                fig, axes = plt.subplots(1, 2, figsize=(15, 11))
                y = np.arange(len(rows))
                labels = [r["demographic"] + ": " + r["group"] for r in rows]
                axes[0].barh(y, [r["median"] for r in rows], color="#326d80")
                axes[1].barh(y, [100*r["cats_under_10_responses"]/r["cats_in_universe"] for r in rows], color="#a95c38")
                for ax in axes:
                    ax.set_yticks(y, labels, fontsize=8)
                    ax.invert_yaxis()
                    ax.grid(axis="x", alpha=.2)
                axes[0].set(xlabel="Median response appearances per cat", title="Typical per-cat group support")
                axes[1].set(xlabel="Cats with fewer than 10 responses (%)", title="Sparse per-cat group cells", xlim=(0, 100))
                fig.suptitle(f"Per-cat demographic support: {partition}", fontsize=16)
                fig.text(.04, .025, "Each group uses the same fitting-cat universe; within-cat responses count once. 10 is illustrative, not a power threshold.", fontsize=9)
                fig.subplots_adjust(left=.18, right=.97, top=.92, bottom=.08, wspace=.55)
                pdf.savefig(fig)
                plt.close(fig)
            fig, ax = plt.subplots(figsize=(13, 7))
            ax.axis("off")
            gender = next(r for r in comparisons if r["partition"] == "development" and r["demographic"] == "gender"
                          and r["group_a"] == "man" and r["group_b"] == "woman")
            lines = [
                f"Development gender: {gender['cats_with_any_in_both_groups']}/{gender['cats_in_universe']} cats appear in both man and woman groups.",
                f"Development gender: {gender['shared_pairs']} photo pairs have at least one response from each group.",
                f"Development gender: {gender['shared_pairs_with_5_responses_in_each_group']} shared pairs have at least five responses in each group.",
                "Per-cat counts measure response records, not distinct respondents or independent observations.",
                "Cat counts aggregate different photo pairs; they do not establish matched semantic-feature preference.",
                "Missing categories are tracked separately and are not substantive demographic groups.",
                "Structural fit/development rows only; no outcomes were read and no models were fitted.",
            ]
            fig.suptitle("Interpretation: cat support is not matched-pair support", fontsize=15)
            for i, line in enumerate(lines):
                ax.text(.03, .86-i*.105, "- "+line, fontsize=11, transform=ax.transAxes)
            pdf.savefig(fig)
            plt.close(fig)


def run() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    outputs = ("original_hashes_before.json", "per_cat_demographic_counts.csv", "group_support_summary.csv",
               "matched_group_support.csv", "support_summary.json", "presentation_cat_demographic_support.pdf",
               "analysis_provenance.json")
    if any((HERE / name).exists() for name in outputs):
        raise FileExistsError("Support-analysis outputs already exist; refusing to overwrite.")
    freeze_path = ROOT / "artifacts" / "model_freeze.json"
    if digest(freeze_path) != (ROOT / "artifacts" / "model_freeze.sha256").read_text(encoding="ascii").strip():
        raise ValueError("Original G2 freeze seal mismatch.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for name in ("artifacts/rows_canonical.parquet", "artifacts/g2_eligibility.parquet"):
        if digest(ROOT.joinpath(*name.split("/"))) != freeze["artifacts_sha256"][name]:
            raise ValueError(f"Frozen structural source changed: {name}")
    before = snapshots()
    write_json(HERE / "original_hashes_before.json", before)
    scopes = load_structure()
    detail, summary, comparisons, totals = compute(scopes)
    write_csv(HERE / "per_cat_demographic_counts.csv", detail, list(detail[0]))
    write_csv(HERE / "group_support_summary.csv", summary, list(summary[0]))
    write_csv(HERE / "matched_group_support.csv", comparisons, list(comparisons[0]))
    write_json(HERE / "support_summary.json", totals)
    plot(summary, comparisons)
    if snapshots() != before:
        raise RuntimeError("A pre-existing project file changed during outcome-blind analysis.")
    names = ("per_cat_demographic_counts.csv", "group_support_summary.csv", "matched_group_support.csv",
             "support_summary.json", "presentation_cat_demographic_support.pdf")
    write_json(HERE / "analysis_provenance.json", {
        "status": "completed_outcome_blind_descriptive",
        "input_sha256": {name: digest(ROOT.joinpath(*name.split("/"))) for name in
                         ("artifacts/model_freeze.json", "artifacts/model_freeze.sha256",
                          "artifacts/rows_canonical.parquet", "artifacts/g2_eligibility.parquet")},
        "outputs_sha256": {name: digest(HERE / name) for name in names},
        "original_snapshot_sha256": digest(HERE / "original_hashes_before.json"),
        "original_files_unchanged": len(before), "per_cat_rows": len(detail), "group_summary_rows": len(summary),
        "matched_group_rows": len(comparisons), "outcome_columns_read": False, "test_rows_read": False,
        "models_fitted": False, "nonclaim": "No causal demographic preference, respondent independence or statistical power claim."})
    print(json.dumps({"totals": totals, "per_cat_rows": len(detail),
                      "summary_rows": len(summary), "comparison_rows": len(comparisons)}, indent=2))


if __name__ == "__main__":
    run()
