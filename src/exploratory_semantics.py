"""Exploratory T2 semantic choice associations from development responses only."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .audit import CONTEXT
from .descriptive import wilson
from .development import _hash
from .freeze import validate_freeze
from .presentation_features import feature_label

FIELDS = ("gender", "pet_history", "pet_care", "pet_contact", "future_pet", "mbti_ie")
HARD = ("colour_primary", "colour_secondary", "colour_pattern", "breed", "framing",
        "view", "eyes", "eyes_colour", "age", "posture")
NUMERIC = ("share_black", "share_white", "share_gray", "share_orange", "share_cream",
           "share_brown", "tabby_score", "pointed_score", "bbox_area_frac")
UNKNOWN_SEMANTICS = {"", "unknown", "<unknown>", "<other>", "no_detection"}
RATES = "tables/exploratory_demographic_semantic_rates.csv"
OVERLAP = "tables/exploratory_demographic_semantic_overlap.csv"
SUPPORT = "tables/exploratory_demographic_semantic_support.csv"
FIGURE = "figures/exploratory_demographic_semantics.pdf"
SUPPORT_FIGURE = "figures/exploratory_demographic_semantic_support.pdf"
PROVENANCE = "artifacts/stage10_5_provenance.json"
DISCLAIMER = "Exploratory Observational Association, Not Causal Effect"


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no headers: {path}")
        return list(reader.fieldnames), list(reader)


def load_development_rows(root: Path) -> list[dict[str, Any]]:
    columns = ["train_row_id", "partition", "pair_id", "image_i", "image_j",
               "y_canonical", "comparison_type", *FIELDS]
    return pq.read_table(
        root / "artifacts/rows_canonical.parquet", columns=columns,
        filters=[("partition", "in", ["fit", "validation"])],
    ).to_pylist()


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows or len({row["train_row_id"] for row in rows}) != len(rows):
        raise ValueError("Development responses are empty or have duplicate IDs.")
    endpoints: dict[str, tuple[str, str]] = {}
    for row in rows:
        if row["partition"] not in {"fit", "validation"} or row["y_canonical"] not in (0, 1):
            raise ValueError("Only labeled fitting and validation responses are permitted.")
        if row["image_i"] >= row["image_j"]:
            raise ValueError("Canonical semantic coordinates must have ordered distinct endpoints.")
        pair = (row["image_i"], row["image_j"])
        if endpoints.setdefault(row["pair_id"], pair) != pair:
            raise ValueError("A pair ID maps to inconsistent image endpoints.")
        if row["comparison_type"] not in {"within_cat", "between_cats"}:
            raise ValueError("Unexpected comparison type.")
        for field in FIELDS:
            if field in CONTEXT and row[field] and row[field] not in CONTEXT[field]:
                raise ValueError(f"Unexpected demographic code: {field}={row[field]!r}")
            if row[field] == "<missing>":
                raise ValueError("Raw demographic value collides with reserved missingness category.")


def semantic_variables(features: list[dict[str, Any]]) -> list[dict[str, str]]:
    variables = []
    for family in HARD:
        categories = sorted({str(row[family]) for row in features
                             if row.get(family) is not None and str(row[family]) not in UNKNOWN_SEMANTICS})
        for value in categories:
            variables.append({
                "kind": "category", "family": family, "feature": family + "=" + value,
                "category": value, "display_label": feature_label(family + "=" + value, "Y"),
            })
    for name in NUMERIC:
        variables.append({
            "kind": "numeric", "family": name, "feature": name, "category": "",
            "display_label": "Higher " + feature_label(name, "Y"),
        })
    return variables


def feature_events(
    rows: list[dict[str, Any]], by_id: dict[str, dict[str, Any]],
    variable: dict[str, str], tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return usability, feature discrimination, carrier wins, and measurement gaps."""
    usable = np.zeros(len(rows), dtype=bool)
    differs = np.zeros(len(rows), dtype=bool)
    wins = np.zeros(len(rows), dtype=bool)
    gaps = np.zeros(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        first, second = by_id.get(row["image_i"]), by_id.get(row["image_j"])
        if first is None or second is None:
            raise ValueError("A development endpoint has no Y record.")
        if any(record["extraction_status"] != "ok" or record["detection_status"] == "no_detection"
               for record in (first, second)):
            continue
        a, b = first[variable["family"]], second[variable["family"]]
        if variable["kind"] == "category":
            if a is None or b is None or str(a) in UNKNOWN_SEMANTICS or str(b) in UNKNOWN_SEMANTICS:
                continue
            usable[i] = True
            has_a, has_b = str(a) == variable["category"], str(b) == variable["category"]
            if has_a == has_b:
                continue
            differs[i] = True
            wins[i] = bool(row["y_canonical"]) == has_a
        else:
            if a is None or b is None:
                continue
            a, b = float(a), float(b)
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("A successful semantic record has a nonfinite numeric value.")
            usable[i] = True
            gap = a-b
            if abs(gap) <= tolerance:
                continue
            differs[i] = True
            wins[i] = bool(row["y_canonical"]) == (gap > 0)
            gaps[i] = gap if row["y_canonical"] else -gap
    return usable, differs, wins, gaps


def summarize(
    rows: list[dict[str, Any]], features: list[dict[str, Any]], protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    validate_rows(rows)
    by_id = {row["image_id"]: row for row in features}
    if len(by_id) != len(features):
        raise ValueError("Semantic cache contains duplicate image IDs.")
    if not {row[key] for row in rows for key in ("image_i", "image_j")} <= set(by_id):
        raise ValueError("A development endpoint has no semantic feature record.")
    variables = semantic_variables(features)
    pair_ids = sorted({row["pair_id"] for row in rows})
    pair_index = {pair: i for i, pair in enumerate(pair_ids)}
    codes = np.asarray([pair_index[row["pair_id"]] for row in rows])
    groups = {field: np.asarray([row[field] or "<missing>" for row in rows]) for field in FIELDS}
    minimum = protocol["heatmap_support"]
    tolerance = protocol["numeric_tie_tolerance"]
    rates, overlap, support = [], [], []
    for field in FIELDS:
        for group in sorted(set(groups[field])):
            ix = np.where(groups[field] == group)[0]
            support.append({
                "partition": "development=fit+validation", "demographic": field, "group": group,
                "responses": len(ix), "distinct_pairs": len({rows[i]["pair_id"] for i in ix}),
                "within_cat_responses": sum(rows[i]["comparison_type"] == "within_cat" for i in ix),
                "between_cats_responses": sum(rows[i]["comparison_type"] == "between_cats" for i in ix),
                "no_detection_endpoint_responses": sum(any(
                    by_id[rows[i][key]]["detection_status"] == "no_detection" for key in ("image_i", "image_j")
                ) for i in ix),
                "dog_fallback_endpoint_responses": sum(any(
                    by_id[rows[i][key]]["detection_status"] == "dog_fallback" for key in ("image_i", "image_j")
                ) for i in ix),
            })
    for variable in variables:
        usable, differs, wins, gaps = feature_events(rows, by_id, variable, tolerance)
        for field in FIELDS:
            pair_counts: dict[str, np.ndarray] = {}
            pair_wins: dict[str, np.ndarray] = {}
            for group in sorted(set(groups[field])):
                membership = groups[field] == group
                ix = np.where(membership & differs)[0]
                n = len(ix)
                k = int(wins[ix].sum())
                counts = np.bincount(codes[ix], minlength=len(pair_ids))
                choices = np.bincount(codes[ix], weights=wins[ix], minlength=len(pair_ids))
                pair_counts[group], pair_wins[group] = counts, choices
                pair_n = int(np.count_nonzero(counts))
                low, high = wilson(k, n) if n else (None, None)
                rates.append({
                    "partition": "development=fit+validation", "task_role": "T2", "demographic": field,
                    "group": group, **variable, "group_responses": int(membership.sum()),
                    "usable_measurement_responses": int((membership & usable).sum()),
                    "non_discriminating_responses": int((membership & usable & ~differs).sum()),
                    "excluded_measurement_responses": int((membership & ~usable).sum()),
                    "discriminating_responses": n, "distinct_pairs": pair_n, "feature_endpoint_choices": k,
                    "choice_rate": k/n if n else None, "wilson_low": low, "wilson_high": high,
                    "support_status": "supported" if n >= minimum["responses"] and pair_n >= minimum["distinct_pairs"]
                                      else "low_support",
                    "mean_signed_chosen_minus_rejected_gap": float(gaps[ix].mean())
                    if n and variable["kind"] == "numeric" else None,
                    "within_cat_responses": sum(rows[i]["comparison_type"] == "within_cat" for i in ix),
                    "between_cats_responses": sum(rows[i]["comparison_type"] == "between_cats" for i in ix),
                    "interpretation": "raw conditional choice association; unmatched pair composition; not a causal or adjusted demographic effect",
                })
            for first, second in combinations(sorted(pair_counts), 2):
                ca, cb = pair_counts[first], pair_counts[second]
                shared = (ca > 0) & (cb > 0)
                qualifying = (ca >= 5) & (cb >= 5)
                matched = int(qualifying.sum())
                delta = float(np.mean(
                    pair_wins[first][qualifying]/ca[qualifying]
                    - pair_wins[second][qualifying]/cb[qualifying]
                )) if matched else None
                overlap.append({
                    "partition": "development=fit+validation", "demographic": field, **variable,
                    "group_a": first, "group_b": second,
                    "group_a_discriminating_pairs": int(np.count_nonzero(ca)),
                    "group_b_discriminating_pairs": int(np.count_nonzero(cb)),
                    "shared_discriminating_pairs": int(shared.sum()),
                    "shared_pairs_with_5_responses_each": matched,
                    "matched_pair_mean_rate_difference_a_minus_b": delta,
                    "comparison_status": "descriptive_matched_pairs" if matched else "insufficient_per_pair_group_support",
                    "interpretation": "same-pair descriptive contrast only where per-cell support passes; no causal adjustment",
                })
    return rates, overlap, support


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No exploratory rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(root: Path, rates: list[dict[str, Any]], support: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import TwoSlopeNorm

    coat = {"colour_primary", "colour_secondary", "colour_pattern", "share_black", "share_white",
            "share_gray", "share_orange", "share_cream", "share_brown", "tabby_score", "pointed_score"}
    labels = {
        "gender": "Gender", "pet_history": "Household pet history", "pet_care": "Personal pet care",
        "pet_contact": "Pet contact frequency", "future_pet": "Future pet intention", "mbti_ie": "Self-reported MBTI I/E",
    }
    with PdfPages(root / FIGURE) as pdf:
        for field in FIELDS:
            relevant = [row for row in rates if row["demographic"] == field]
            groups = sorted({row["group"] for row in relevant})
            group_sizes = {row["group"]: row["responses"] for row in support if row["demographic"] == field}
            for section, is_coat in (("Coat labels and measurements", True), ("Other automated attributes", False)):
                subset = [row for row in relevant if (row["family"] in coat) == is_coat]
                variables = list(dict.fromkeys(row["feature"] for row in subset))
                index = {(row["feature"], row["group"]): row for row in subset}
                matrix = np.full((len(variables), len(groups)), np.nan)
                for i, variable in enumerate(variables):
                    for j, group in enumerate(groups):
                        row = index[(variable, group)]
                        if row["support_status"] == "supported":
                            matrix[i, j] = row["choice_rate"]
                fig, ax = plt.subplots(figsize=(13, max(8, len(variables)*.27+3)))
                fig.subplots_adjust(left=.37, right=.90, top=.80, bottom=.14)
                fig.suptitle(f"{labels[field]} and semantic image choices", x=.04, y=.97,
                             ha="left", fontsize=18, fontweight="bold")
                fig.text(.04, .925, DISCLAIMER, ha="left", fontsize=12, color="#784621")
                fig.text(.04, .885, section + " | Fitting + validation only", ha="left", fontsize=11)
                cmap = plt.get_cmap("RdBu").copy()
                cmap.set_bad("#e6e6e6")
                image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap,
                                  norm=TwoSlopeNorm(vmin=0, vcenter=.5, vmax=1))
                ax.set_yticks(range(len(variables)),
                              [index[(v, groups[0])]["display_label"] for v in variables], fontsize=9)
                ax.set_xticks(range(len(groups)),
                              [f"{g.replace('_', ' ')}\nN={group_sizes[g]}" for g in groups], fontsize=9)
                ax.xaxis.tick_top()
                for i, variable in enumerate(variables):
                    for j, group in enumerate(groups):
                        row = index[(variable, group)]
                        if row["support_status"] == "supported":
                            value = row["choice_rate"]
                            ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8,
                                    color="white" if value < .20 or value > .80 else "black")
                        else:
                            ax.text(j, i, "--", ha="center", va="center", fontsize=8, color="#666666")
                colorbar = fig.colorbar(image, ax=ax, fraction=.045, pad=.025)
                colorbar.set_label("Choice rate for category-carrying / higher-measurement endpoint", fontsize=9)
                fig.text(.04, .095, "Only discriminating comparisons enter a rate. Gray cells: fewer than 100 responses or 30 pairs.",
                         fontsize=9)
                fig.text(.04, .067, "Groups may see different pairs. A rate above 50% is not an adjusted preference effect or a significance test.",
                         fontsize=9)
                fig.text(.04, .039, "Unknown semantic values and no-detection endpoints excluded. Dog fallbacks retained; CLIP-inferred breed is not true pedigree.",
                         fontsize=8.5)
                pdf.savefig(fig)
                plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, field in zip(axes.flat, FIELDS):
        entries = [row for row in support if row["demographic"] == field]
        ax.barh([row["group"].replace("_", " ") for row in entries], [row["responses"] for row in entries],
                color="#28798c")
        ax.set_title(labels[field])
        ax.set_xlabel("Development responses (not unique respondents)")
        ax.invert_yaxis()
    fig.suptitle("Development demographic support | " + DISCLAIMER, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(root / SUPPORT_FIGURE)
    plt.close(fig)


def append_payload(parent: bytes, headers: list[str], rows: list[dict[str, Any]]) -> bytes:
    if not parent.endswith(b"\n"):
        raise ValueError("Original synthesis must end with a newline before appending.")
    existing_reader = csv.DictReader(io.StringIO(parent.decode("utf-8-sig")))
    existing = list(existing_reader)
    if existing_reader.fieldnames != headers or len(headers) != len(set(headers)):
        raise ValueError("Synthesis headers differ from the original schema.")
    ids = {row["claim_id"] for row in existing}
    new_ids = [row["claim_id"] for row in rows]
    if len(set(new_ids)) != len(new_ids) or ids & set(new_ids):
        raise ValueError("Append would duplicate an existing or new claim ID.")
    if any(set(row) != set(headers) for row in rows):
        raise ValueError("Appended claims do not match the existing column schema.")
    output = io.StringIO(newline="")
    csv.DictWriter(output, fieldnames=headers, lineterminator="\n").writerows(rows)
    return output.getvalue().encode("utf-8")


def build_claims(root: Path, rates: list[dict[str, Any]], overlap: list[dict[str, Any]],
                 parent_headers: list[str]) -> list[dict[str, Any]]:
    claims = []
    source_hash = _hash(root / RATES)
    overlap_hash = _hash(root / OVERLAP)
    for i, field in enumerate(FIELDS, start=1):
        supported = [row for row in rates if row["demographic"] == field and row["support_status"] == "supported"]
        features = sorted({row["feature"] for row in supported})
        ranges = []
        for feature in features:
            entries = [row for row in supported if row["feature"] == feature]
            if len(entries) < 2:
                continue
            ordered = sorted(entries, key=lambda row: (row["choice_rate"], row["group"]))
            ranges.append((ordered[-1]["choice_rate"]-ordered[0]["choice_rate"], feature, ordered[0], ordered[-1]))
        refs = []
        if ranges:
            gap, feature, low, high = min(ranges, key=lambda item: (-item[0], item[2]["family"], item[1]))
            matched = next(row for row in overlap if row["demographic"] == field and row["feature"] == feature
                           and {row["group_a"], row["group_b"]} == {low["group"], high["group"]})
            result = (
                f"Exploratory development-only raw-rate comparison for {feature}: "
                f"{high['group']} rate {high['choice_rate']:.6f} "
                f"({high['feature_endpoint_choices']}/{high['discriminating_responses']} choices; "
                f"{high['distinct_pairs']} pairs) versus {low['group']} rate {low['choice_rate']:.6f} "
                f"({low['feature_endpoint_choices']}/{low['discriminating_responses']} choices; "
                f"{low['distinct_pairs']} pairs). Raw rate range = {gap:.6f}. "
                f"Shared discriminating pairs = {matched['shared_discriminating_pairs']}; "
                f"shared pairs with the required per-group support = {matched['shared_pairs_with_5_responses_each']}. "
                "This is the largest observed range among supported groups/features for this demographic, "
                "selected descriptively after examining development outcomes."
            )
            for row in (high, low):
                refs.append({"source": RATES, "sha256": source_hash,
                             "selector": {"demographic": field, "group": row["group"], "feature": feature},
                             "values": {key: row[key] for key in ("choice_rate", "feature_endpoint_choices",
                                        "discriminating_responses", "distinct_pairs")}})
            refs.append({"source": OVERLAP, "sha256": overlap_hash,
                         "selector": {key: matched[key] for key in ("demographic", "feature", "group_a", "group_b")},
                         "values": {key: matched[key] for key in ("shared_discriminating_pairs",
                                      "shared_pairs_with_5_responses_each")}})
            refs.append({"operation": "difference", "value": gap,
                         "definition": "higher raw choice_rate minus lower raw choice_rate from the cited rate rows"})
            status = "exploratory_unadjusted_association"
        else:
            result = "No semantic feature has enough supported demographic groups for the exploratory raw-rate range summary."
            refs.append({"source": RATES, "sha256": source_hash, "selector": {"demographic": field},
                         "operation": "count_features_with_at_least_two_supported_groups", "value": 0})
            status = "insufficient_support"
        claim = {
            "claim_id": f"X105-{i:02d}", "task_roles": "T2", "target": "None",
            "analysis_target": "exploratory_development_semantic_association", "activity": "Explain",
            "evidence_status": "Real-data claim", "progress": "exploratory_synthesis",
            "result_status": status, "question": f"Do {field} groups choose images with different semantic YOLO-CLIP features?",
            "partition": "development=fit+validation", "measured_result": result,
            "supported_interpretation": "Raw choice associations describe these observed comparisons. They are not adjusted "
                "demographic preference effects; where shared-pair support is inadequate, no matched group contrast is established.",
            "alternative_explanations": "Different pair and cat composition, display-side tendencies, correlated attributes, "
                "automated measurement error, sparse groups, and unlinked respondents can explain apparent differences.",
            "nonclaim": "No causal effect, no significance or multiple-testing discovery claim, no respondent-level inference, "
                "and no true-pedigree interpretation of CLIP-inferred breed. Largest raw ranges are exploratory selections.",
            "next_discriminating_evidence": "Collect overlapping pair comparisons across demographic groups with stronger "
                "per-cell support and respondent linkage; preregister future adjusted analyses on independent data.",
            "source_evidence_json": json.dumps(refs, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            "limitations": "Development outcomes only; post-evaluation exploratory T2 analysis. "
                "Raw rates may compare different pair populations. Wilson intervals are binomial working-model summaries "
                "with unverified independence; no p-values or confirmatory discoveries are reported.",
        }
        if set(claim) != set(parent_headers):
            raise ValueError("The original claim schema changed.")
        claims.append(claim)
    return claims


def run(root: Path) -> dict[str, Any]:
    outputs = [root / name for name in (RATES, OVERLAP, SUPPORT, FIGURE, SUPPORT_FIGURE, PROVENANCE)]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Stage 10.5 outputs already exist; refusing to repeat or overwrite.")
    frozen = validate_freeze(root)
    protocol_path = root / "artifacts/stage10_5_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["demographics"] != list(FIELDS) or protocol["hard_labels"] != list(HARD) or protocol["numeric_features"] != list(NUMERIC):
        raise ValueError("Exploratory scope differs from the documented protocol.")
    parent_paths = [root / "claims_final.csv", root / "tables/diagnostic_synthesis.csv"]
    stage10 = json.loads((root / "artifacts/stage10_provenance.json").read_text(encoding="utf-8"))
    for name, digest in stage10["input_sha256"].items():
        path = root.joinpath(*name.split("/"))
        if not path.is_file() or _hash(path) != digest:
            raise ValueError(f"Original Stage 10 input changed: {name}")
    parents = {path.relative_to(root).as_posix(): path.read_bytes() for path in parent_paths}
    for name in parents:
        if _hash(root / name) != stage10["output_sha256"][name]:
            raise ValueError(f"Original Stage 10 claims changed: {name}")
    if len(set(parents.values())) != 1:
        raise ValueError("Stage 10 synthesis tables differ before append.")
    headers, original = _read_csv(parent_paths[0])
    if len(original) != stage10["claims"]:
        raise ValueError("Stage 10 row count differs from its provenance.")
    rows = load_development_rows(root)
    validate_rows(rows)
    development_images = sorted({row[key] for row in rows for key in ("image_i", "image_j")})
    features = pq.read_table(
        root / "features/Y.parquet", filters=[("image_id", "in", development_images)]
    ).to_pylist()
    inputs = {
        name: _hash(root / name) for name in (
            "artifacts/rows_canonical.parquet", "features/Y.parquet", "features/manifests/Y.json",
            "artifacts/stage10_5_protocol.json", "artifacts/stage10_provenance.json",
            "artifacts/model_freeze.json", "src/exploratory_semantics.py",
        )
    }
    rates, overlap, support = summarize(rows, features, protocol)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    for name, records in ((RATES, rates), (OVERLAP, overlap), (SUPPORT, support)):
        _write_csv(root / name, records)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    _plot(root, rates, support)
    claims = build_claims(root, rates, overlap, headers)
    payloads = {name: append_payload(parent, headers, claims) for name, parent in parents.items()}
    for name, digest in inputs.items():
        if _hash(root / name) != digest:
            raise RuntimeError(f"Exploratory input changed during analysis: {name}")
    for name, parent in parents.items():
        if (root / name).read_bytes() != parent:
            raise RuntimeError(f"Original synthesis changed during analysis: {name}")
    lineage = {}
    for name, payload in payloads.items():
        path = root / name
        with path.open("ab") as stream:
            stream.write(payload)
        if path.read_bytes() != parents[name] + payload:
            raise RuntimeError(f"Append did not preserve the original synthesis bytes: {name}")
        lineage[name] = {
            "parent_sha256": stage10["output_sha256"][name], "parent_bytes": len(parents[name]),
            "parent_rows": len(original), "appended_claim_ids": [row["claim_id"] for row in claims],
            "current_sha256": _hash(path), "current_rows": len(original) + len(claims),
        }
    validate_freeze(root)
    report = {
        "stage": "10.5", "task_role": "T2", "status": "exploratory_development_only",
        "development_responses": len(rows), "development_pairs": len({row["pair_id"] for row in rows}),
        "semantic_features": len(semantic_variables(features)), "rate_rows": len(rates),
        "demographic_fields": list(FIELDS), "raw_outcomes_source": "canonical rows filtered to fitting and validation",
        "internal_test_outcomes_read": False, "test_evaluation_rerun": False, "new_predictive_models_fit": False,
        "causal_claim": False, "confirmatory_significance_test": False,
        "qualified_shared_pair_contrasts": sum(row["shared_pairs_with_5_responses_each"] > 0 for row in overlap),
        "input_sha256": inputs, "append_lineage": lineage,
        "output_sha256": {path.relative_to(root).as_posix(): _hash(path) for path in outputs[:-1]},
        "stage10_provenance_note": "Original Stage 10 output hashes now identify the unchanged file prefixes. "
            "This record verifies the append and the current full-file hashes; original provenance was not overwritten.",
        "scope_note": "This exploratory analysis uses development outcomes after final test evaluation and cannot "
            "be promoted to an independent or confirmatory analysis of demographic preferences.",
        "frozen_representation_unchanged": frozen["selected_representation"],
    }
    with (root / PROVENANCE).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        report = run(args.root.resolve())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Exploratory Stage 10.5 failed: {exc}", file=sys.stderr)
        return 2
    print(f"Exploratory semantic rates: {report['rate_rows']} rows; "
          f"{report['qualified_shared_pair_contrasts']} qualified matched comparisons; synthesis appended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
