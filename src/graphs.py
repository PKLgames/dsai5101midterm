"""Stage 4 fitting graphs and outcome-blind strict-G2 eligibility."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import networkx as nx
import pyarrow as pa
import pyarrow.parquet as pq

from .feature_freeze import validate_reviews
from .feature_quality import build_components
from .features import _file_hash, _input_inventory


@dataclass
class FittingGraphs:
    photo: nx.Graph
    photo_outcomes: nx.DiGraph
    cat: nx.Graph
    cat_outcomes: nx.DiGraph
    fitting_responses: int
    within_cat_responses: int


def _increment(graph: nx.Graph | nx.DiGraph, a: str, b: str) -> None:
    if graph.has_edge(a, b):
        graph[a][b]["responses"] += 1
    else:
        graph.add_edge(a, b, responses=1)


def build_fitting_graphs(rows: list[dict[str, Any]]) -> FittingGraphs:
    photo, photo_outcomes = nx.Graph(), nx.DiGraph()
    cat, cat_outcomes = nx.Graph(), nx.DiGraph()
    within = 0
    for row in rows:
        if row["partition"] != "fit":
            raise ValueError("Only fitting rows may enter the outcome graphs.")
        a, b, ac, bc, y = row["a_image"], row["b_image"], row["a_cat"], row["b_cat"], row["y_displayed"]
        if not a or not b or not ac or not bc or a == b or y not in (0, 1):
            raise ValueError(f"Invalid fitting graph row: {row['train_row_id']}")
        if row["comparison_type"] not in {"within_cat", "between_cats"} or (
            (row["comparison_type"] == "within_cat") != (ac == bc)
        ):
            raise ValueError(f"Comparison type conflicts with cat IDs: {row['train_row_id']}")
        photo.add_nodes_from((a, b))
        photo_outcomes.add_nodes_from((a, b))
        cat.add_nodes_from((ac, bc))
        cat_outcomes.add_nodes_from((ac, bc))
        _increment(photo, a, b)
        _increment(photo_outcomes, a if y == 1 else b, b if y == 1 else a)
        if ac == bc:
            within += 1
        else:
            _increment(cat, ac, bc)
            _increment(cat_outcomes, ac if y == 1 else bc, bc if y == 1 else ac)
    return FittingGraphs(photo, photo_outcomes, cat, cat_outcomes, len(rows), within)


def _components(graph: nx.Graph) -> list[set[str]]:
    return sorted(nx.connected_components(graph), key=lambda nodes: (-len(nodes), min(nodes)))


def graph_details(
    population: str, entity: str, graph: nx.Graph, directed: nx.DiGraph,
    fitting_responses: int, within_cat_responses: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if set(graph.nodes) != set(directed.nodes):
        raise ValueError(f"Directed and undirected {population}/{entity} nodes differ.")
    components = _components(graph)
    strong = sorted(nx.strongly_connected_components(directed), key=lambda nodes: (-len(nodes), min(nodes)))
    component_id = {node: index for index, nodes in enumerate(components) for node in nodes}
    strong_id = {node: index for index, nodes in enumerate(strong) for node in nodes}
    nontrivial = [nodes for nodes in components if len(nodes) > 1]
    strongly_connected_nontrivial = sum(
        nx.is_strongly_connected(directed.subgraph(nodes)) for nodes in nontrivial
    )
    count = graph.number_of_nodes()
    summary = {
        "population": population, "entity": entity,
        "fitting_responses": fitting_responses,
        "excluded_within_cat_responses": within_cat_responses if entity == "cat" else 0,
        "nodes": count, "undirected_edges": graph.number_of_edges(),
        "undirected_components": len(components),
        "largest_component_nodes": len(components[0]) if components else 0,
        "largest_component_fraction": len(components[0]) / count if count else "",
        "directed_edges": directed.number_of_edges(),
        "strongly_connected_components": len(strong),
        "largest_strong_component_nodes": len(strong[0]) if strong else 0,
        "nontrivial_undirected_components": len(nontrivial),
        "strongly_connected_nontrivial_components": strongly_connected_nontrivial,
    }
    node_rows = [{
        "population": population, "entity": entity, "node_id": node,
        "component_id": component_id[node], "component_size": len(components[component_id[node]]),
        "strong_component_id": strong_id[node], "strong_component_size": len(strong[strong_id[node]]),
        "distinct_opponents": graph.degree(node),
        "fitting_response_appearances": graph.degree(node, weight="responses"),
        "distinct_wins": directed.out_degree(node), "distinct_losses": directed.in_degree(node),
    } for node in sorted(graph.nodes)]
    edges = [{
        "population": population, "entity": entity, "graph_type": "undirected",
        "source": min(a, b), "target": max(a, b), "fitting_responses": data["responses"],
    } for a, b, data in graph.edges(data=True)]
    edges.extend({
        "population": population, "entity": entity, "graph_type": "winner_to_loser",
        "source": a, "target": b, "fitting_responses": data["responses"],
    } for a, b, data in directed.edges(data=True))
    edges.sort(key=lambda row: (row["graph_type"], row["source"], row["target"]))
    return summary, node_rows, edges


def build_eligibility(
    rows: list[dict[str, Any]], common_rows: dict[str, dict[str, Any]],
    full: FittingGraphs, common: FittingGraphs | None, usable_images: set[str],
) -> list[dict[str, Any]]:
    full_components = {node: index for index, nodes in enumerate(_components(full.photo)) for node in nodes}
    common_components = (
        {node: index for index, nodes in enumerate(_components(common.photo)) for node in nodes}
        if common is not None else {}
    )

    def reason(components: dict[str, int], a: str, b: str) -> str:
        missing = int(a not in components) + int(b not in components)
        if missing == 2:
            return "both_photos_unseen_in_fit"
        if missing == 1:
            return "one_photo_unseen_in_fit"
        return "eligible" if components[a] == components[b] else "different_fitting_components"

    output = []
    seen: set[str] = set()
    for row in rows:
        row_id, partition = row["train_row_id"], row["partition"]
        if row_id in seen or row_id not in common_rows:
            raise ValueError(f"Duplicate or absent common-population row: {row_id}")
        seen.add(row_id)
        fixed = common_rows[row_id]
        if fixed["partition"] != partition:
            raise ValueError(f"Partition mismatch for {row_id}")
        a, b = row["a_image"], row["b_image"]
        if partition == "fit":
            full_reason = "not_heldout"
            common_reason = "not_heldout" if common is not None else "no_admitted_visual_features"
        else:
            full_reason = reason(full_components, a, b)
            common_reason = (
                "no_admitted_visual_features" if common is None else
                "feature_unavailable" if a not in usable_images or b not in usable_images else
                reason(common_components, a, b)
            )
        full_eligible, common_eligible = full_reason == "eligible", common_reason == "eligible"
        if full_eligible != fixed["full_strict_g2"] or common_eligible != fixed["common_visual_strict_g2"]:
            raise ValueError(f"Recomputed strict-G2 mask disagrees with frozen Stage 3 population: {row_id}")
        if partition == "fit" and fixed["common_visual_fit"] != (
            common is not None and a in usable_images and b in usable_images
        ):
            raise ValueError(f"Recomputed common-fitting mask disagrees with Stage 3: {row_id}")
        output.append({
            "train_row_id": row_id, "pair_id": row["pair_id"], "partition": partition,
            "comparison_type": row["comparison_type"],
            "full_strict_g2": full_eligible, "full_reason": full_reason,
            "common_strict_g2": common_eligible, "common_reason": common_reason,
            "common_visual_fit": fixed["common_visual_fit"],
        })
    if seen != set(common_rows):
        raise ValueError("Frozen common population contains unmatched row IDs.")
    return output


def coverage_table(eligibility: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for population in ("full", "common"):
        for partition in ("validation", "test"):
            for comparison in ("all", "within_cat", "between_cats"):
                subset = [
                    row for row in eligibility
                    if row["partition"] == partition
                    and (comparison == "all" or row["comparison_type"] == comparison)
                ]
                reason_field = f"{population}_reason"
                pair_reasons: dict[str, str] = {}
                for row in subset:
                    prior = pair_reasons.setdefault(row["pair_id"], row[reason_field])
                    if prior != row[reason_field]:
                        raise ValueError(f"Pair has inconsistent eligibility: {row['pair_id']}")
                eligible_rows = sum(row[reason_field] == "eligible" for row in subset)
                eligible_pairs = sum(reason == "eligible" for reason in pair_reasons.values())
                reasons = Counter(row[reason_field] for row in subset)
                unavailable = bool(subset) and reasons["no_admitted_visual_features"] == len(subset)
                result.append({
                    "population": population, "partition": partition, "comparison_type": comparison,
                    "status": "not_available" if unavailable else "available",
                    "responses": len(subset), "pairs": len(pair_reasons),
                    "eligible_responses": eligible_rows, "eligible_pairs": eligible_pairs,
                    "response_coverage": eligible_rows / len(subset) if subset and not unavailable else "",
                    "pair_coverage": eligible_pairs / len(pair_reasons) if pair_reasons and not unavailable else "",
                    "one_photo_unseen_in_fit": reasons["one_photo_unseen_in_fit"],
                    "both_photos_unseen_in_fit": reasons["both_photos_unseen_in_fit"],
                    "different_fitting_components": reasons["different_fitting_components"],
                    "feature_unavailable": reasons["feature_unavailable"],
                    "no_admitted_visual_features": reasons["no_admitted_visual_features"],
                })
    return result


def _csv_text(fields: list[str], rows: list[dict[str, Any]]) -> str:
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _feature_usability(root: Path, admitted: list[str]) -> tuple[dict[str, set[str]], set[str]]:
    by_block: dict[str, set[str]] = {}
    for block in admitted:
        records = pq.read_table(
            root / "features" / f"{block}.parquet",
            columns=["image_id", "extraction_status"],
        ).to_pylist()
        if len({record["image_id"] for record in records}) != len(records):
            raise ValueError(f"Duplicate image IDs in feature block {block}.")
        by_block[block] = {record["image_id"] for record in records if record["extraction_status"] == "ok"}
    usable = set.intersection(*(by_block[block] for block in admitted)) if admitted else set()
    return by_block, usable


def official_coverage(
    root: Path, full: FittingGraphs, common: FittingGraphs | None,
    by_block: dict[str, set[str]], usable: set[str],
) -> list[dict[str, Any]]:
    raw = _read_csv(root / "dataset" / "test.csv")
    previous = _read_csv(root / "tables" / "official_test_coverage.csv")
    if len(raw) != len(previous):
        raise ValueError("Official test rows do not match the Stage 1 coverage file.")
    full_components = {node: index for index, nodes in enumerate(_components(full.photo)) for node in nodes}
    common_components = (
        {node: index for index, nodes in enumerate(_components(common.photo)) for node in nodes}
        if common is not None else {}
    )

    def support(components: dict[str, int], a: str, b: str) -> bool:
        return a in components and b in components and components[a] == components[b]

    result = []
    for row, prior in zip(raw, previous):
        if row["test_row_id"] != prior["test_row_id"]:
            raise ValueError("Official test row order differs from Stage 1 coverage.")
        a, b = row["a_image"], row["b_image"]
        output = {
            "test_row_id": row["test_row_id"], "overlap_category": prior["overlap_category"],
            "missing_a_image": prior["missing_a_image"], "missing_b_image": prior["missing_b_image"],
            "full_fitting_component_support": str(support(full_components, a, b)).lower(),
            "common_fitting_component_support": (
                str(support(common_components, a, b)).lower() if common is not None else "not_available"
            ),
            "all_admitted_features_available": (
                str(a in usable and b in usable).lower() if by_block else "not_available"
            ),
        }
        for block in ("H", "C", "Y"):
            output[f"{block}_features_available"] = (
                str(a in by_block[block] and b in by_block[block]).lower()
                if block in by_block else "not_admitted"
            )
        result.append(output)
    return result


def _write_figure(path: Path, summaries: list[dict[str, Any]], nodes: list[dict[str, Any]],
                  coverage: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = [(1, 1, "1"), (2, 4, "2-4"), (5, 9, "5-9"), (10, None, "10+")]
    populations = [row["population"] for row in summaries if row["entity"] == "photo"]
    for population in populations:
        sample = [node["distinct_opponents"] for node in nodes
                  if node["population"] == population and node["entity"] == "photo"]
        counts = [sum(value >= low and (high is None or value <= high) for value in sample)
                  for low, high, _ in bins]
        axes[0].plot([label for _, _, label in bins], counts, marker="o", label=population)
    axes[0].set(xlabel="Distinct fitting opponents", ylabel="Fitting photos", title="Photo support")
    axes[0].legend()
    all_coverage = [row for row in coverage if row["comparison_type"] == "all"]
    labels = [f"{row['population']}\n{row['partition']}" for row in all_coverage]
    values = [float(row["response_coverage"]) if row["response_coverage"] != "" else 0.0 for row in all_coverage]
    bars = axes[1].bar(labels, values)
    for bar, row in zip(bars, all_coverage):
        if row["status"] == "not_available":
            axes[1].text(bar.get_x() + bar.get_width() / 2, 0.02, "N/A", ha="center")
    axes[1].set_ylim(0, 1.05)
    axes[1].set(ylabel="Fraction of responses", title="Strict-G2 coverage")
    fig.suptitle("Fitting-graph support and structural held-out eligibility")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run_graphs(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    outputs = [
        root / "tables" / "graph_summary.csv", root / "tables" / "split_coverage.csv",
        root / "tables" / "official_test_coverage_stage4.csv", root / "figures" / "graph_support.pdf",
        root / "artifacts" / "g2_eligibility.parquet", root / "artifacts" / "graph_components.parquet",
        root / "artifacts" / "graph_edges.parquet", root / "artifacts" / "graph_provenance.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("Stage 4 outputs already exist; refusing to overwrite: " +
                              ", ".join(str(path) for path in existing))
    if not config.get("real_data_modeling_allowed"):
        raise ValueError("Stage 0 governance gate has not been opened.")
    _input_inventory(root)
    common_path = root / "artifacts" / "common_population_manifest.parquet"
    report_path = root / "artifacts" / "common_population_report.json"
    if not common_path.is_file() or not report_path.is_file():
        raise ValueError("Stage 3 admission and common population must be frozen before Stage 4.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if _file_hash(common_path) != report["population_manifest_sha256"]:
        raise ValueError("Frozen Stage 3 common population hash does not match.")
    current_decisions = _read_csv(root / "artifacts" / "feature_admission_decisions.csv")
    if current_decisions != report["admission_decisions"]:
        raise ValueError("Feature admission decisions changed after Stage 3 freeze.")
    split = pq.read_table(root / "artifacts" / "split_manifest.parquet").to_pylist()
    full_components = build_components(split)
    review_rows, _ = validate_reviews(root, set(full_components.parents))
    review_issue_count = sum(
        row[field] == "issue" for row in review_rows
        for field in ("orientation", "cnn_crop_retention", "detection_mask", "visible_attributes",
                      "no_detection_behavior")
    )
    if review_issue_count != report["review_issue_fields"]:
        raise ValueError("Review findings changed after the Stage 3 freeze.")
    admitted = report["admitted_blocks"]
    if any(block not in {"H", "C", "Y"} for block in admitted) or len(set(admitted)) != len(admitted):
        raise ValueError("Frozen feature-admission registry is invalid.")
    if set(admitted) != {row["block"] for row in current_decisions if row["decision"] == "admitted"}:
        raise ValueError("Frozen admitted features do not match the reviewed decisions.")
    for block in admitted:
        if report["feature_reports"][block]["automated_checks"] != "passed":
            raise ValueError(f"{block} was admitted without passing automated checks.")
        manifest_path = root / "features" / "manifests" / f"{block}.json"
        feature_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, digest in feature_manifest["source_sha256"].items():
            path = root.joinpath(*name.split("/"))
            if not path.is_file() or _file_hash(path) != digest:
                raise ValueError(f"{block} feature source changed after extraction: {name}")
    by_block, usable = _feature_usability(root, admitted)

    structural = pq.read_table(
        root / "artifacts" / "rows_canonical.parquet",
        columns=["train_row_id", "pair_id", "partition", "comparison_type", "a_image", "b_image"],
    ).to_pylist()
    fit_rows = pq.read_table(
        root / "artifacts" / "rows_canonical.parquet",
        columns=["train_row_id", "partition", "comparison_type", "a_image", "b_image",
                 "a_cat", "b_cat", "y_displayed"],
        filters=[("partition", "=", "fit")],
    ).to_pylist()
    common_records = pq.read_table(common_path).to_pylist()
    common_rows = {row["train_row_id"]: row for row in common_records}
    if len(common_rows) != len(common_records):
        raise ValueError("Frozen Stage 3 population contains duplicate row IDs.")
    full = build_fitting_graphs(fit_rows)
    common = build_fitting_graphs(
        [row for row in fit_rows if common_rows[row["train_row_id"]]["common_visual_fit"]]
    ) if admitted else None
    eligibility = build_eligibility(structural, common_rows, full, common, usable)
    coverage = coverage_table(eligibility)
    official = official_coverage(root, full, common, by_block, usable)
    summaries, nodes, edges = [], [], []
    for population, bundle in (("full", full), ("common", common)):
        if bundle is None:
            continue
        for entity, graph, directed in (
            ("photo", bundle.photo, bundle.photo_outcomes),
            ("cat", bundle.cat, bundle.cat_outcomes),
        ):
            summary, node_rows, edge_rows = graph_details(
                population, entity, graph, directed, bundle.fitting_responses, bundle.within_cat_responses
            )
            summaries.append(summary)
            nodes.extend(node_rows)
            edges.extend(edge_rows)

    full_coverage = [row for row in coverage if row["population"] == "full" and row["comparison_type"] == "all"]
    blocked = any(row["eligible_responses"] == 0 for row in full_coverage)
    common_coverage = [row for row in coverage if row["population"] == "common" and row["comparison_type"] == "all"]
    common_empty = bool(admitted) and any(row["eligible_responses"] == 0 for row in common_coverage)
    result = {
        "status": (
            "blocked_empty_full_strict_g2" if blocked else
            "full_only_common_strict_g2_empty" if common_empty else "graphs_and_masks_created"
        ),
        "common_population_status": report["status"],
        "admitted_blocks": admitted,
        "outcome_source": "fitting rows only; directed edges point winner to loser",
        "eligibility_source": "structural endpoints and fitting-graph components; no held-out outcomes",
        "interpretation": "Directed strong connectivity concerns the ordinary unpenalized BT MLE; it does not determine whether a penalized estimator returns finite scores.",
        "library_versions": {name: metadata.version(name) for name in ("networkx", "matplotlib", "pyarrow")},
        "inputs_sha256": {
            name: _file_hash(root.joinpath(*name.split("/"))) for name in (
                "artifacts/rows_canonical.parquet", "artifacts/split_manifest.parquet",
                "artifacts/common_population_manifest.parquet", "artifacts/common_population_report.json",
                "artifacts/feature_review.csv", "artifacts/feature_review_manifest.json",
                "artifacts/feature_admission_decisions.csv", "artifacts/raw_hashes.json",
                "tables/official_test_coverage.csv", "dataset/test.csv", "decisions.yaml",
                "src/graphs.py", "src/feature_quality.py", "src/feature_freeze.py",
                "requirements-stage4.txt",
            )
        },
        "stage3_review_history_limit": "Stage 3 did not record a hash of mutable review findings at freeze; current review and decision hashes are recorded here.",
    }
    for block in admitted:
        for name in (f"features/{block}.parquet", f"features/manifests/{block}.json"):
            result["inputs_sha256"][name] = _file_hash(root.joinpath(*name.split("/")))

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    (root / "tables" / "graph_summary.csv").write_text(
        _csv_text(list(summaries[0]), summaries), encoding="utf-8", newline=""
    )
    (root / "tables" / "split_coverage.csv").write_text(
        _csv_text(list(coverage[0]), coverage), encoding="utf-8", newline=""
    )
    (root / "tables" / "official_test_coverage_stage4.csv").write_text(
        _csv_text(list(official[0]), official), encoding="utf-8", newline=""
    )
    pq.write_table(pa.Table.from_pylist(eligibility), root / "artifacts" / "g2_eligibility.parquet")
    pq.write_table(pa.Table.from_pylist(nodes), root / "artifacts" / "graph_components.parquet")
    pq.write_table(pa.Table.from_pylist(edges), root / "artifacts" / "graph_edges.parquet")
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    _write_figure(root / "figures" / "graph_support.pdf", summaries, nodes, coverage)
    result["outputs_sha256"] = {
        path.relative_to(root).as_posix(): _file_hash(path)
        for path in outputs if path != root / "artifacts" / "graph_provenance.json"
    }
    (root / "artifacts" / "graph_provenance.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result
