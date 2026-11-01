"""Reconcile repeated-response statistics using explicit response combinations."""

import csv
import itertools
import json
from collections import defaultdict

import numpy as np

from analyze import HERE, ROOT, digest, load, write_json


def read(name):
    with (HERE/name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    if not np.isclose(float(actual), float(expected), rtol=0, atol=1e-12):
        raise ValueError(f"Mismatch: {actual} vs {expected}")


def run():
    if (HERE/"verification.json").exists():
        raise FileExistsError("Verification already exists.")
    provenance = json.loads((HERE/"analysis_provenance.json").read_text(encoding="utf-8"))
    for path, value in provenance["input_sha256"].items():
        if digest(ROOT.joinpath(*path.split("/"))) != value:
            raise ValueError(f"Input changed: {path}")
    for name, value in provenance["output_sha256"].items():
        if digest(HERE/name) != value:
            raise ValueError(f"Output changed: {name}")
    aggregates = {(r["cohort"], r["level"]): r for r in read("agreement_summary.csv")}
    comparisons = {(r["cohort"], r["level"], r["source"], r["model"]): r for r in read("model_comparisons.csv")}
    checked = 0
    for cohort, predictions in load().items():
        for level in ("photo_pair", "cat_pair"):
            base = predictions[0][2]
            grouped = defaultdict(list)
            for r in base.itertuples(index=False):
                if level == "cat_pair" and r.a_cat == r.b_cat:
                    continue
                key = r.pair_id if level == "photo_pair" else tuple(sorted((r.a_cat, r.b_cat)))
                flip = r.orientation == -1 if level == "photo_pair" else r.a_cat > r.b_cat
                grouped[key].append((r.train_row_id, int(1-r.outcome if flip else r.outcome)))
            repeated = {key: items for key, items in grouped.items() if len(items) >= 2}
            total = sum(len(items) for items in repeated.values())
            disagreement_sum, loo_sum, naive_sum, pair_agree, pair_n = 0, 0, 0, 0, 0
            selected = set()
            for items in repeated.values():
                ys = [y for _, y in items]
                selected.update(row for row, _ in items)
                pairs = list(itertools.combinations(ys, 2))
                disagreement = sum(a != b for a, b in pairs)/len(pairs)
                disagreement_sum += len(ys)*disagreement
                pair_agree += sum(a == b for a, b in pairs)
                pair_n += len(pairs)
                for i, y in enumerate(ys):
                    other_mean = sum(ys[:i]+ys[i+1:])/(len(ys)-1)
                    loo_sum += (other_mean-y)**2
                    naive_sum += (sum(ys)/len(ys)-y)**2
            r = aggregates[cohort, level]
            if int(r["repeated_responses"]) != total or int(r["repeated_groups"]) != len(repeated):
                raise ValueError("Repeated coverage mismatch.")
            close(r["agreement_response_weighted"], 1-disagreement_sum/total)
            close(r["agreement_response_pairs_weighted"], pair_agree/pair_n)
            close(r["estimated_oracle_brier"], disagreement_sum/(2*total))
            close(r["loo_brier"], loo_sum/total)
            close(r["in_sample_brier_optimistic"], naive_sum/total)
            for source, model, frame in predictions:
                scope = frame if level == "photo_pair" else frame.loc[frame.a_cat != frame.b_cat]
                matched = frame.loc[frame.train_row_id.isin(selected)]
                record = comparisons[cohort, level, source, model]
                close(record["full_scope_brier"], np.mean((scope.probability-scope.outcome)**2))
                brier = np.mean((matched.probability-matched.outcome)**2)
                close(record["repeated_brier"], brier)
                close(record["repeated_accuracy"], np.mean((matched.probability >= .5) == (matched.outcome == 1)))
                close(record["model_minus_oracle"], brier-disagreement_sum/(2*total))
                checked += 1
    if checked != len(comparisons) or len(aggregates) != 6:
        raise ValueError("Output coverage mismatch.")
    write_json(HERE/"verification.json", {"status": "independent_pair_enumeration_passed",
               "agreement_rows_verified": len(aggregates), "model_rows_verified": checked,
               "checks": ["explicit unordered response-pair agreement", "leave-one-out Brier",
                          "in-sample Brier", "repeated subset and all saved-model Brier", "orientation and cross-source response alignment"],
               "intervals": "Bootstrap intervals are not independently replayed.",
               "analysis_provenance_sha256": digest(HERE/"analysis_provenance.json")})
    print(f"Verified {len(aggregates)} agreement summaries and {checked} saved-model comparisons.")


if __name__ == "__main__":
    run()
