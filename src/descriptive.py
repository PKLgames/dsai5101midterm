"""Stage 6 descriptive summaries using fitting and validation outcomes only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .estimator import context_mapping
from .split import response_band

FIELDS = [
    "train_row_id", "partition", "pair_id", "image_i", "image_j",
    "a_image", "b_image", "a_cat", "b_cat", "comparison_type", "pet_care", "y_canonical",
]
OUTPUTS = (
    "tables/pair_summary.csv", "tables/context_overlap.csv", "tables/context_pair_support.csv",
    "tables/context_support.csv", "tables/photo_support.csv", "tables/pair_support_summary.csv",
    "tables/disagreement_examples.csv", "tables/pair_entropy.csv",
    "figures/disagreement.pdf", "figures/support_distribution.pdf",
    "artifacts/stage6_provenance.json",
)


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Wilson interval requires 0 <= successes <= n and n > 0.")
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return max(0., center-half), min(1., center+half)


def _contexts(row: dict[str, Any]) -> str:
    value = row["pet_care"]
    if value == "":
        return "missing"
    if value not in {"never", "occasionally", "past_regular", "current_regular"}:
        raise ValueError(f"Unexpected pet_care code: {value!r}")
    return value


def _partitions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    for row in rows:
        if row["partition"] not in {"fit", "validation"} or row["y_canonical"] not in (0, 1):
            raise ValueError("Descriptive analysis accepts only labeled fit/validation rows.")
    return {
        "fit": [row for row in rows if row["partition"] == "fit"],
        "validation": [row for row in rows if row["partition"] == "validation"],
        "development": rows,
    }


def summarize(rows: list[dict[str, Any]], include_entropy: bool) -> dict[str, list[dict[str, Any]]]:
    partitions = _partitions(rows)
    if len({row["train_row_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate development response IDs.")
    pair_rows, context_cells, support, photo_rows, overlap, distributions, entropy = [], [], [], [], [], [], []
    for partition, subset in partitions.items():
        pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        photos: dict[str, dict[str, Any]] = {}
        contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in subset:
            pairs[row["pair_id"]].append(row)
            contexts[_contexts(row)].append(row)
            for side, other in (("a", "b"), ("b", "a")):
                image, opponent, cat = row[f"{side}_image"], row[f"{other}_image"], row[f"{side}_cat"]
                entry = photos.setdefault(image, {"cat": cat, "responses": 0, "opponents": set()})
                if entry["cat"] != cat:
                    raise ValueError(f"Conflicting cat identities for {image}.")
                entry["responses"] += 1
                entry["opponents"].add(opponent)
        for image, entry in sorted(photos.items()):
            photo_rows.append({
                "partition": partition, "image_id": image, "cat_id": entry["cat"],
                "response_count": entry["responses"], "distinct_opponents": len(entry["opponents"]),
            })
        cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for pair_id, group in sorted(pairs.items()):
            first = group[0]
            if any((row["image_i"], row["image_j"], row["comparison_type"]) !=
                   (first["image_i"], first["image_j"], first["comparison_type"]) for row in group):
                raise ValueError(f"Inconsistent canonical endpoints or comparison type: {pair_id}")
            n = len(group)
            k = sum(row["y_canonical"] for row in group)
            low, high = wilson(k, n)
            pair_rows.append({
                "partition": partition, "pair_id": pair_id, "image_i": first["image_i"],
                "image_j": first["image_j"], "comparison_type": first["comparison_type"],
                "responses": n, "canonical_i_choices": k, "canonical_i_proportion": k/n,
                "wilson_low": low, "wilson_high": high, "response_band": response_band(n),
            })
            if include_entropy:
                q = k/n
                h = 0. if q in (0., 1.) else -q*math.log(q)-(1-q)*math.log(1-q)
                entropy.append({
                    "partition": partition, "pair_id": pair_id, "responses": n,
                    "entropy_nats": h,
                    "interpretation": "in-sample descriptive fit; not an oracle or information limit",
                })
            for row in group:
                cells[(pair_id, _contexts(row))].append(row)
        for (pair_id, group), members in sorted(cells.items()):
            n, k = len(members), sum(row["y_canonical"] for row in members)
            low, high = wilson(k, n)
            context_cells.append({
                "partition": partition, "pair_id": pair_id, "comparison_type": members[0]["comparison_type"],
                "pet_care": group, "responses": n, "canonical_i_choices": k,
                "canonical_i_proportion": k/n, "wilson_low": low, "wilson_high": high,
                "eligible_for_pair_contrast": n >= 5,
            })
        for group, members in sorted(contexts.items()):
            ids = {row["pair_id"] for row in members}
            support.append({
                "partition": partition, "pet_care": group, "responses": len(members),
                "distinct_pairs": len(ids),
                "distinct_photos": len({row[key] for row in members for key in ("a_image", "b_image")}),
                "distinct_cats": len({row[key] for row in members for key in ("a_cat", "b_cat")}),
                "within_cat_responses": sum(row["comparison_type"] == "within_cat" for row in members),
                "between_cat_responses": sum(row["comparison_type"] == "between_cats" for row in members),
                "interaction_support_threshold_met": len(members) >= 100 and len(ids) >= 30,
                "threshold_interpretation": "operational rule, not statistical power",
            })
        for a, b in combinations(sorted(contexts), 2):
            a_pairs = {row["pair_id"] for row in contexts[a]}
            b_pairs = {row["pair_id"] for row in contexts[b]}
            shared = a_pairs & b_pairs
            enough = sum(len(cells[(pair, a)]) >= 5 and len(cells[(pair, b)]) >= 5 for pair in shared)
            overlap.append({
                "partition": partition, "group_a": a, "group_b": b,
                "group_a_pairs": len(a_pairs), "group_b_pairs": len(b_pairs),
                "shared_pairs": len(shared), "shared_pairs_with_5_responses_each": enough,
                "shared_fraction_a": len(shared)/len(a_pairs), "shared_fraction_b": len(shared)/len(b_pairs),
            })
        for comparison in ("all", "within_cat", "between_cats"):
            for band in ("1", "2-4", "5-9", "10+"):
                group = [p for p in pair_rows if p["partition"] == partition
                         and (comparison == "all" or p["comparison_type"] == comparison)
                         and p["response_band"] == band]
                distributions.append({
                    "partition": partition, "comparison_type": comparison, "response_band": band,
                    "pairs": len(group), "responses": sum(p["responses"] for p in group),
                    "mean_pair_proportion": sum(p["canonical_i_proportion"] for p in group)/len(group)
                    if group else "",
                })
    dev = [p for p in pair_rows if p["partition"] == "development" and p["responses"] >= 20]
    examples = []
    for label, key in (
        ("closest_to_half", lambda p: (abs(p["canonical_i_proportion"] - .5), -p["responses"], p["pair_id"])),
        ("furthest_from_half", lambda p: (-abs(p["canonical_i_proportion"] - .5), -p["responses"], p["pair_id"])),
    ):
        chosen = min(dev, key=key) if dev else None
        examples.append({
            "partition": "development", "selection": label,
            "status": "selected" if chosen else "no_pair_with_at_least_20_responses",
            "pair_id": chosen["pair_id"] if chosen else "",
            "responses": chosen["responses"] if chosen else 0,
            "canonical_i_proportion": chosen["canonical_i_proportion"] if chosen else "",
        })
    fit = partitions["fit"]
    reference, interactions = context_mapping([_contexts(row) for row in fit], [row["pair_id"] for row in fit])
    for row in support:
        row["fitting_reference"] = reference or ""
        row["fitting_interaction_eligible"] = row["pet_care"] in interactions
    return {
        "tables/pair_summary.csv": pair_rows, "tables/context_pair_support.csv": context_cells,
        "tables/context_support.csv": support, "tables/context_overlap.csv": overlap,
        "tables/photo_support.csv": photo_rows, "tables/pair_support_summary.csv": distributions,
        "tables/disagreement_examples.csv": examples, "tables/pair_entropy.csv": entropy,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fallback: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else fallback, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _figures(root: Path, summaries: dict[str, list[dict[str, Any]]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [row for row in summaries["tables/pair_summary.csv"] if row["partition"] == "development"]
    bands = ("1", "2-4", "5-9", "10+")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, comparison in zip(axes, ("within_cat", "between_cats")):
        for band in bands:
            sample = [p["canonical_i_proportion"] for p in pairs
                      if p["comparison_type"] == comparison and p["response_band"] == band]
            if sample:
                ax.hist(sample, bins=10, range=(0, 1), alpha=.4, label=f"{band}: {len(sample)} pairs")
        ax.set(title=comparison, xlabel="Canonical image i choice proportion", ylabel="Pair count")
        ax.legend(fontsize="small")
    fig.suptitle("Development pair proportions by response support (fitting + validation)")
    fig.tight_layout()
    fig.savefig(root / "figures/disagreement.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for comparison in ("within_cat", "between_cats"):
        counts = [sum(p["comparison_type"] == comparison and p["response_band"] == band for p in pairs)
                  for band in bands]
        axes[0].plot(bands, counts, marker="o", label=comparison)
    axes[0].set(xlabel="Responses per pair", ylabel="Pair count", title="Pair support")
    axes[0].legend()
    photos = [p for p in summaries["tables/photo_support.csv"] if p["partition"] == "development"]
    axes[1].hist([p["distinct_opponents"] for p in photos], bins=20)
    axes[1].set(xlabel="Distinct opponents", ylabel="Photos", title="Photo opponent support")
    fig.suptitle("Development support (fitting + validation)")
    fig.tight_layout()
    fig.savefig(root / "figures/support_distribution.pdf")
    plt.close(fig)


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_descriptive(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not config.get("real_data_modeling_allowed"):
        raise ValueError("Stage 0 governance gate has not been opened.")
    output_paths = [root.joinpath(*name.split("/")) for name in OUTPUTS]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("Stage 6 outputs already exist; refusing to overwrite.")
    graph_path = root / "artifacts" / "graph_provenance.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    if graph.get("status") != "graphs_and_masks_created":
        raise ValueError("Stage 4 graph gate is not satisfied.")
    for group in ("inputs_sha256", "outputs_sha256"):
        for name, digest in graph[group].items():
            path = root.joinpath(*name.split("/"))
            if not path.is_file() or _hash(path) != digest:
                raise ValueError(f"Stage 4 provenance mismatch: {name}")
    simulation_path = root / "artifacts" / "simulation_provenance.json"
    simulation = json.loads(simulation_path.read_text(encoding="utf-8"))
    if simulation["failed_fits"] != 0:
        raise ValueError("Stage 5 synthetic estimator fits did not all pass.")
    for group in ("source_sha256", "outputs_sha256"):
        for name, digest in simulation[group].items():
            path = root.joinpath(*name.split("/"))
            if not path.is_file() or _hash(path) != digest:
                raise ValueError(f"Stage 5 provenance mismatch: {name}")
    source = root / "artifacts" / "rows_canonical.parquet"
    rows = pq.read_table(source, columns=FIELDS, filters=[("partition", "in", ["fit", "validation"])]).to_pylist()
    include_entropy = config["options"]["in_sample_pair_entropy_descriptive"]
    summaries = summarize(rows, include_entropy)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    fallback = {
        "tables/context_overlap.csv": ["partition", "group_a", "group_b", "group_a_pairs", "group_b_pairs",
                                       "shared_pairs", "shared_pairs_with_5_responses_each",
                                       "shared_fraction_a", "shared_fraction_b"],
        "tables/pair_entropy.csv": ["partition", "pair_id", "responses", "entropy_nats", "interpretation"],
    }
    for name, table in summaries.items():
        _write_csv(root.joinpath(*name.split("/")), table, fallback.get(name, []))
    _figures(root, summaries)
    provenance = {
        "partitions": ["fit", "validation", "development=fit+validation"],
        "excluded_outcomes": "internal test and official test",
        "wilson_interval": "95% binomial working model; respondent linkage unavailable",
        "group_interaction_threshold": "100 fitting responses and 30 distinct fitting pairs",
        "pair_group_contrast_threshold": "at least five responses in each group on the same pair",
        "example_threshold": "at least 20 development responses; ties by greater support then pair ID",
        "entropy_enabled": include_entropy,
        "synthetic_label_control": simulation["synthetic_label_control"],
        "input_sha256": {name: _hash(root.joinpath(*name.split("/"))) for name in (
            "artifacts/rows_canonical.parquet", "artifacts/graph_provenance.json",
            "artifacts/simulation_provenance.json", "decisions.yaml", "src/descriptive.py",
        )},
        "output_sha256": {name: _hash(root.joinpath(*name.split("/"))) for name in OUTPUTS[:-1]},
    }
    (root / "artifacts/stage6_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"development_responses": len(rows), "development_pairs": sum(
        p["partition"] == "development" for p in summaries["tables/pair_summary.csv"]
    )}
