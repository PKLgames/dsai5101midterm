"""Independently verify per-cat counts using a pandas endpoint expansion."""

import csv
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from analyze import DEMOGRAPHICS, HERE, ROOT, digest, write_json


def read(name):
    with (HERE/name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def run():
    if (HERE/"verification.json").exists():
        raise FileExistsError("Verification already exists.")
    provenance = json.loads((HERE/"analysis_provenance.json").read_text(encoding="utf-8"))
    for name, value in provenance["input_sha256"].items():
        if digest(ROOT.joinpath(*name.split("/"))) != value:
            raise ValueError(f"Structural source changed: {name}")
    for name, value in provenance["outputs_sha256"].items():
        if digest(HERE/name) != value:
            raise ValueError(f"Count output changed: {name}")
    fields = ["train_row_id", "partition", "pair_id", "a_image", "b_image", "a_cat", "b_cat",
              "comparison_type", *DEMOGRAPHICS]
    raw = pq.read_table(ROOT/"artifacts"/"rows_canonical.parquet", columns=fields,
                        filters=[("partition", "in", ["fit", "validation"])]).to_pandas()
    masks = pq.read_table(ROOT/"artifacts"/"g2_eligibility.parquet",
                          columns=["train_row_id", "partition", "common_visual_fit", "common_strict_g2"],
                          filters=[("partition", "in", ["fit", "validation"])]).to_pandas()
    if raw["train_row_id"].duplicated().any() or masks["train_row_id"].duplicated().any():
        raise ValueError("Duplicate response IDs.")
    frame = raw.merge(masks, on=["train_row_id", "partition"], validate="one_to_one", how="inner")
    if len(frame) != len(raw) or len(raw) != len(masks):
        raise ValueError("Structural/mask row coverage mismatch.")
    fit = frame.loc[(frame.partition == "fit") & frame.common_visual_fit]
    validation = frame.loc[(frame.partition == "validation") & frame.common_strict_g2]
    scopes = {"fit": fit, "development": pd.concat([fit, validation], ignore_index=True)}
    details, summaries, comparisons = read("per_cat_demographic_counts.csv"), read("group_support_summary.csv"), read("matched_group_support.csv")
    expected_cells = 0
    for partition, data in scopes.items():
        cats = sorted(set(data.a_cat) | set(data.b_cat))
        within = int((data.comparison_type == "within_cat").sum())
        appearances = 2*len(data)-within
        left = data[["train_row_id", "pair_id", "comparison_type", "a_cat", "a_image", "b_cat", *DEMOGRAPHICS]].copy()
        left.columns = ["train_row_id", "pair_id", "comparison_type", "cat_id", "image_id", "other_cat", *DEMOGRAPHICS]
        right = data[["train_row_id", "pair_id", "comparison_type", "b_cat", "b_image", "a_cat", *DEMOGRAPHICS]].copy()
        right.columns = left.columns
        endpoint = pd.concat([left, right], ignore_index=True)
        appearances_df = endpoint.drop_duplicates(["train_row_id", "cat_id"])
        if len(appearances_df) != appearances:
            raise ValueError("Independent cat appearance total mismatch.")
        for demographic in DEMOGRAPHICS:
            observed = data.assign(group=data[demographic].replace("", "<missing>"))
            app = appearances_df.assign(group=appearances_df[demographic].replace("", "<missing>"))
            photo = endpoint.assign(group=endpoint[demographic].replace("", "<missing>"))
            grouped = app.groupby(["group", "cat_id"], dropna=False)
            actual = grouped.agg(responses=("train_row_id", "size"), pairs=("pair_id", "nunique"))
            photo_count = photo.groupby(["group", "cat_id"])["image_id"].nunique()
            opponents = (app.loc[app.cat_id != app.other_cat]
                         .groupby(["group", "cat_id"])["other_cat"].nunique())
            expected = [r for r in details if r["partition"] == partition and r["demographic"] == demographic]
            for record in expected:
                key = (record["group"], record["cat_id"])
                response = int(actual.loc[key, "responses"]) if key in actual.index else 0
                pairs = int(actual.loc[key, "pairs"]) if key in actual.index else 0
                photos = int(photo_count.loc[key]) if key in photo_count.index else 0
                opponent_count = int(opponents.loc[key]) if key in opponents.index else 0
                if (response, pairs, photos, opponent_count) != (
                    int(record["responses_with_cat"]), int(record["distinct_photo_pairs"]),
                    int(record["distinct_photos_of_cat"]), int(record["distinct_other_opponent_cats"])
                ):
                    raise ValueError(f"Independent per-cat mismatch: {partition} {demographic} {key}")
                expected_cells += 1
            for summary in [r for r in summaries if r["partition"] == partition and r["demographic"] == demographic]:
                subset = [r for r in expected if r["group"] == summary["group"]]
                values = np.asarray([int(r["responses_with_cat"]) for r in subset])
                calculations = [len(values), int(values.sum()), int((values == 0).sum()),
                                int((values < 5).sum()), int((values < 10).sum()), int((values < 20).sum())]
                listed = [int(summary[k]) for k in ("cats_in_universe", "cat_response_appearances", "cats_with_zero",
                                                   "cats_under_5_responses", "cats_under_10_responses", "cats_under_20_responses")]
                if calculations != listed:
                    raise ValueError(f"Summary mismatch: {partition} {demographic} {summary['group']}")
                np.testing.assert_allclose(np.quantile(values, [.25, .5, .75]),
                                           [float(summary["q25"]), float(summary["median"]), float(summary["q75"])],
                                           rtol=0, atol=1e-12)
                if int((observed.group == summary["group"]).sum()) != int(summary["group_response_rows"]):
                    raise ValueError("Group response denominator mismatch.")
            pair_counts = observed.groupby(["group", "pair_id"]).size()
            for comparison in [r for r in comparisons if r["partition"] == partition and r["demographic"] == demographic]:
                a, b = comparison["group_a"], comparison["group_b"]
                ca = pair_counts.loc[a] if a in pair_counts.index.get_level_values(0) else pd.Series(dtype=int)
                cb = pair_counts.loc[b] if b in pair_counts.index.get_level_values(0) else pd.Series(dtype=int)
                shared = set(ca.index) & set(cb.index)
                actual_pairs = (len(shared), sum(ca[p] >= 2 and cb[p] >= 2 for p in shared),
                                sum(ca[p] >= 5 and cb[p] >= 5 for p in shared))
                recorded = tuple(int(comparison[k]) for k in ("shared_pairs",
                                 "shared_pairs_with_2_responses_in_each_group", "shared_pairs_with_5_responses_in_each_group"))
                if actual_pairs != recorded:
                    raise ValueError(f"Shared-pair support mismatch: {partition} {demographic} {a} {b}")
    if expected_cells != len(details):
        raise ValueError("Independent verification did not cover every cat/group cell.")
    if len(details) != 12324 or len(summaries) != 52 or len(comparisons) != 92:
        raise ValueError("Unexpected output shape.")
    write_json(HERE/"verification.json", {
        "status": "independent_pandas_reconciliation_passed",
        "per_cat_group_cells_checked": expected_cells, "group_summary_rows_checked": len(summaries),
        "matched_group_rows_checked": len(comparisons), "fitting_and_development_verified": True,
        "within_cat_endpoints_deduplicated": True, "both_within_cat_photos_retained": True,
        "structural_columns_only": True, "test_rows_read": False, "models_fitted": False,
        "analysis_provenance_sha256": digest(HERE/"analysis_provenance.json"),
        "note": "Response records are not unique people; illustrative support cutoffs are not power guarantees."})
    print(f"Independently checked {expected_cells} cat/group cells, {len(summaries)} group summaries, and {len(comparisons)} matched-group rows.")


if __name__ == "__main__":
    run()
