"""Reconcile all photo-level cells with independently expanded pandas endpoints."""

import csv
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from analyze import DEMOGRAPHICS, HERE, ROOT, digest, write_json


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def run():
    if (HERE/"verification.json").exists():
        raise FileExistsError("Verification already exists.")
    provenance = json.loads((HERE/"analysis_provenance.json").read_text(encoding="utf-8"))
    for path, value in provenance["input_sha256"].items():
        if digest(ROOT.joinpath(*path.split("/"))) != value:
            raise ValueError(f"Input hash mismatch: {path}")
    for name, value in provenance["outputs_sha256"].items():
        if digest(HERE/name) != value:
            raise ValueError(f"Output hash mismatch: {name}")
    columns = ["train_row_id", "partition", "pair_id", "a_image", "b_image", "a_cat", "b_cat",
               "comparison_type", *DEMOGRAPHICS]
    raw = pq.read_table(ROOT/"artifacts"/"rows_canonical.parquet", columns=columns,
                        filters=[("partition", "in", ["fit", "validation"])]).to_pandas()
    mask = pq.read_table(ROOT/"artifacts"/"g2_eligibility.parquet",
                         columns=["train_row_id", "partition", "common_visual_fit", "common_strict_g2"],
                         filters=[("partition", "in", ["fit", "validation"])]).to_pandas()
    data = raw.merge(mask, on=["train_row_id", "partition"], how="inner", validate="one_to_one")
    if len(data) != len(raw) or len(raw) != len(mask):
        raise ValueError("Mask coverage mismatch.")
    fit = data.loc[(data.partition == "fit") & data.common_visual_fit]
    validation = data.loc[(data.partition == "validation") & data.common_strict_g2]
    scopes = {"fit": fit, "development": pd.concat([fit, validation], ignore_index=True)}
    details = read(HERE/"per_photo_demographic_counts.csv")
    summaries = read(HERE/"group_support_summary.csv")
    comparisons = read(HERE/"matched_group_support.csv")
    old = {(r["partition"], r["demographic"], r["group_a"], r["group_b"]): r
           for r in read(ROOT/"evidence_catalog"/"cat_demographic_support"/"matched_group_support.csv")}
    checked = 0
    for scope, rows in scopes.items():
        endpoints = []
        for side, other in (("a", "b"), ("b", "a")):
            part = rows[["train_row_id", "pair_id", "comparison_type", f"{side}_image", f"{side}_cat",
                         f"{other}_image", *DEMOGRAPHICS]].copy()
            part.columns = ["train_row_id", "pair_id", "comparison_type", "photo_id", "cat_id", "opponent", *DEMOGRAPHICS]
            endpoints.append(part)
        endpoint = pd.concat(endpoints, ignore_index=True)
        if len(endpoint) != 2*len(rows) or endpoint.duplicated(["train_row_id", "photo_id"]).any():
            raise ValueError("Response endpoints are not two distinct photos.")
        for demographic in DEMOGRAPHICS:
            long = endpoint.assign(group=endpoint[demographic].replace("", "<missing>"))
            grouped = long.groupby(["group", "photo_id"])
            aggregate = grouped.agg(responses=("train_row_id", "size"), pairs=("pair_id", "nunique"),
                                    opponents=("opponent", "nunique"),
                                    within=("comparison_type", lambda x: (x == "within_cat").sum()),
                                    between=("comparison_type", lambda x: (x == "between_cats").sum()))
            cats = long.groupby("photo_id")["cat_id"].first()
            subset = [r for r in details if r["partition"] == scope and r["demographic"] == demographic]
            for record in subset:
                key = (record["group"], record["photo_id"])
                values = tuple(int(v) for v in aggregate.loc[key]) if key in aggregate.index else (0, 0, 0, 0, 0)
                expected = tuple(int(record[k]) for k in ("responses_with_photo", "distinct_photo_pairs",
                                 "distinct_opponent_photos", "within_cat_responses", "between_cats_responses"))
                if values != expected or cats.loc[record["photo_id"]] != record["cat_id"]:
                    raise ValueError(f"Per-photo discrepancy: {scope} {demographic} {key}")
                checked += 1
            for r in [r for r in summaries if r["partition"] == scope and r["demographic"] == demographic]:
                items = [x for x in subset if x["group"] == r["group"]]
                x = np.array([int(t["responses_with_photo"]) for t in items])
                w = np.array([int(t["within_cat_responses"]) for t in items])
                values = {"photos_in_universe": len(x), "photo_response_appearances": x.sum(), "photos_with_zero": (x == 0).sum(),
                          "minimum": x.min(), "maximum": x.max(), "photos_under_5_responses": (x < 5).sum(),
                          "photos_under_10_responses": (x < 10).sum(), "photos_under_20_responses": (x < 20).sum(),
                          "photos_zero_within_cat": (w == 0).sum(),
                          "group_response_rows": (rows[demographic].replace("", "<missing>") == r["group"]).sum()}
                for key, value in values.items():
                    if int(r[key]) != int(value):
                        raise ValueError(f"Summary discrepancy: {scope} {demographic} {r['group']} {key}")
                floats = {"q25": np.quantile(x, .25), "median": np.median(x), "q75": np.quantile(x, .75), "mean": np.mean(x),
                          "within_cat_median": np.median(w),
                          "median_distinct_photo_pairs": np.median([int(t["distinct_photo_pairs"]) for t in items])}
                for key, value in floats.items():
                    if not np.isclose(float(r[key]), value, atol=1e-12, rtol=0):
                        raise ValueError(f"Summary discrepancy: {scope} {demographic} {r['group']} {key}")
            pair_counts = rows.assign(group=rows[demographic].replace("", "<missing>")).groupby(["group", "pair_id"]).size()
            for r in [r for r in comparisons if r["partition"] == scope and r["demographic"] == demographic]:
                a, b = r["group_a"], r["group_b"]
                aa = {x["photo_id"]: int(x["responses_with_photo"]) for x in subset if x["group"] == a}
                bb = {x["photo_id"]: int(x["responses_with_photo"]) for x in subset if x["group"] == b}
                if aa.keys() != bb.keys() or len(aa) != int(r["photos_in_universe"]):
                    raise ValueError("Photo overlap universe mismatch.")
                for threshold, key in [(1, "photos_with_any_in_both_groups"), (5, "photos_with_at_least_5_in_both_groups"),
                                       (10, "photos_with_at_least_10_in_both_groups")]:
                    if int(r[key]) != sum(aa[p] >= threshold and bb[p] >= threshold for p in aa):
                        raise ValueError("Photo overlap mismatch.")
                ca, cb = pair_counts.loc[a], pair_counts.loc[b]
                shared = set(ca.index) & set(cb.index)
                values = [len(shared), sum(ca[p] >= 2 and cb[p] >= 2 for p in shared),
                          sum(ca[p] >= 5 and cb[p] >= 5 for p in shared)]
                if values != [int(r[k]) for k in ("shared_pairs", "shared_pairs_with_2_in_each_group", "shared_pairs_with_5_in_each_group")]:
                    raise ValueError("Pair support mismatch.")
                prior = old[scope, demographic, a, b]
                if values != [int(prior[k]) for k in ("shared_pairs", "shared_pairs_with_2_responses_in_each_group",
                                                       "shared_pairs_with_5_responses_in_each_group")]:
                    raise ValueError("Disagreement with prior cat-level matched-pair evidence.")
    if (checked, len(summaries), len(comparisons)) != (67392, 52, 92):
        raise ValueError("Unexpected output coverage.")
    write_json(HERE/"verification.json", {"status": "independent_pandas_reconciliation_passed",
               "per_photo_group_cells_checked": checked, "summary_rows_checked": len(summaries),
               "matched_group_rows_checked": len(comparisons), "prior_matched_pair_rows_reconciled": len(comparisons),
               "analysis_provenance_sha256": digest(HERE/"analysis_provenance.json"),
               "structural_columns_only": True, "test_rows_read": False, "models_fitted": False})
    print(f"Verified {checked} photo cells, {len(summaries)} summary rows and {len(comparisons)} group overlaps.")


if __name__ == "__main__":
    run()
