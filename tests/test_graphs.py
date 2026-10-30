import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from src.graphs import build_eligibility, build_fitting_graphs, coverage_table, graph_details, run_graphs


def fit_row(row_id, a, b, ac, bc, y):
    return {
        "train_row_id": row_id, "partition": "fit", "a_image": a, "b_image": b,
        "a_cat": ac, "b_cat": bc, "y_displayed": y,
        "comparison_type": "within_cat" if ac == bc else "between_cats",
    }


class GraphTests(unittest.TestCase):
    def test_graphs_use_fitting_outcomes_and_exclude_cat_self_loops(self):
        rows = [
            fit_row("1", "a", "b", "ca", "cb", 1),
            fit_row("2", "b", "a", "cb", "ca", 1),
            fit_row("3", "c", "d", "cc", "cc", 0),
            fit_row("4", "b", "e", "cb", "ce", 1),
        ]
        graphs = build_fitting_graphs(rows)
        self.assertEqual(graphs.photo.number_of_nodes(), 5)
        self.assertEqual(graphs.photo.number_of_edges(), 3)
        self.assertEqual(graphs.photo["a"]["b"]["responses"], 2)
        self.assertTrue(graphs.photo_outcomes.has_edge("a", "b"))
        self.assertTrue(graphs.photo_outcomes.has_edge("b", "a"))
        self.assertTrue(graphs.photo_outcomes.has_edge("d", "c"))
        self.assertFalse(graphs.cat.has_edge("cc", "cc"))
        self.assertFalse(graphs.cat_outcomes.has_edge("cc", "cc"))
        self.assertIn("cc", graphs.cat.nodes)
        self.assertEqual(graphs.within_cat_responses, 1)
        summary, nodes, edges = graph_details("full", "cat", graphs.cat, graphs.cat_outcomes, 4, 1)
        self.assertEqual(summary["undirected_components"], 2)
        self.assertEqual(summary["strongly_connected_components"], 3)
        self.assertEqual(summary["excluded_within_cat_responses"], 1)
        self.assertEqual(next(row for row in nodes if row["node_id"] == "cc")["distinct_opponents"], 0)
        self.assertTrue(any(row["graph_type"] == "winner_to_loser" for row in edges))

    def test_swapping_sides_and_choice_preserves_outcome_direction(self):
        first = build_fitting_graphs([fit_row("1", "a", "b", "ca", "cb", 1)])
        reversed_graph = build_fitting_graphs([fit_row("2", "b", "a", "cb", "ca", 0)])
        self.assertEqual(set(first.photo_outcomes.edges), set(reversed_graph.photo_outcomes.edges))
        self.assertEqual(set(first.cat_outcomes.edges), set(reversed_graph.cat_outcomes.edges))

    def test_heldout_and_invalid_outcomes_are_rejected(self):
        row = fit_row("1", "a", "b", "ca", "cb", 1)
        row["partition"] = "test"
        with self.assertRaisesRegex(ValueError, "Only fitting"):
            build_fitting_graphs([row])
        row["partition"], row["y_displayed"] = "fit", None
        with self.assertRaisesRegex(ValueError, "Invalid"):
            build_fitting_graphs([row])

    def test_eligibility_reconciles_full_and_matched_components(self):
        full = build_fitting_graphs([
            fit_row("1", "a", "b", "ca", "cb", 1),
            fit_row("2", "b", "c", "cb", "cc", 0),
        ])
        common = build_fitting_graphs([fit_row("1", "a", "b", "ca", "cb", 1)])
        structural = [
            {"train_row_id": "1", "pair_id": "ab", "partition": "fit",
             "comparison_type": "between_cats", "a_image": "a", "b_image": "b"},
            {"train_row_id": "2", "pair_id": "bc", "partition": "fit",
             "comparison_type": "between_cats", "a_image": "b", "b_image": "c"},
            {"train_row_id": "3", "pair_id": "ac", "partition": "validation",
             "comparison_type": "between_cats", "a_image": "a", "b_image": "c"},
            {"train_row_id": "4", "pair_id": "ax", "partition": "test",
             "comparison_type": "between_cats", "a_image": "a", "b_image": "x"},
        ]
        fixed = {
            "1": {"partition": "fit", "full_strict_g2": False, "common_visual_strict_g2": False,
                  "common_visual_fit": True},
            "2": {"partition": "fit", "full_strict_g2": False, "common_visual_strict_g2": False,
                  "common_visual_fit": False},
            "3": {"partition": "validation", "full_strict_g2": True, "common_visual_strict_g2": False,
                  "common_visual_fit": False},
            "4": {"partition": "test", "full_strict_g2": False, "common_visual_strict_g2": False,
                  "common_visual_fit": False},
        }
        result = build_eligibility(structural, fixed, full, common, {"a", "b", "x"})
        self.assertEqual(result[2]["full_reason"], "eligible")
        self.assertEqual(result[2]["common_reason"], "feature_unavailable")
        self.assertEqual(result[3]["full_reason"], "one_photo_unseen_in_fit")
        self.assertEqual(result[3]["common_reason"], "one_photo_unseen_in_fit")
        fixed["3"]["common_visual_strict_g2"] = True
        with self.assertRaisesRegex(ValueError, "disagrees"):
            build_eligibility(structural, fixed, full, common, {"a", "b", "x"})

    def test_coverage_counts_unique_pairs_and_marks_unavailable(self):
        rows = [
            {"pair_id": "p", "partition": "validation", "comparison_type": "between_cats",
             "full_reason": "eligible", "common_reason": "no_admitted_visual_features"},
            {"pair_id": "p", "partition": "validation", "comparison_type": "between_cats",
             "full_reason": "eligible", "common_reason": "no_admitted_visual_features"},
        ]
        result = coverage_table(rows)
        full = next(r for r in result if r["population"] == "full" and r["partition"] == "validation"
                    and r["comparison_type"] == "all")
        common = next(r for r in result if r["population"] == "common" and r["partition"] == "validation"
                      and r["comparison_type"] == "all")
        self.assertEqual((full["responses"], full["pairs"]), (2, 1))
        self.assertEqual(full["response_coverage"], 1.0)
        self.assertEqual(common["status"], "not_available")
        self.assertEqual(common["response_coverage"], "")


class GraphIntegrationTests(unittest.TestCase):
    def test_real_outputs_reconcile_and_test_outcomes_stay_masked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("artifacts", "dataset", "tables", "features/manifests", "src"):
                (root / name).mkdir(parents=True, exist_ok=True)
            fitting = [
                fit_row("1", "a", "b", "ca", "cb", 1),
                fit_row("2", "b", "c", "cb", "cc", 0),
                fit_row("3", "c", "d", "cc", "cd", 1),
            ]
            structural = [
                {**row, "pair_id": f"p{row['train_row_id']}"} for row in fitting
            ] + [
                {**fit_row("4", "a", "c", "ca", "cc", None), "partition": "validation", "pair_id": "p4"},
                {**fit_row("5", "a", "d", "ca", "cd", None), "partition": "test", "pair_id": "p5"},
            ]
            pq.write_table(pa.Table.from_pylist(structural), root / "artifacts/rows_canonical.parquet")
            split = [{"image_i": row["a_image"], "image_j": row["b_image"], "partition": row["partition"]}
                     for row in structural]
            pq.write_table(pa.Table.from_pylist(split), root / "artifacts/split_manifest.parquet")
            common = [{
                "train_row_id": row["train_row_id"], "partition": row["partition"],
                "common_visual_fit": row["partition"] == "fit",
                "full_strict_g2": row["partition"] != "fit",
                "common_visual_strict_g2": row["partition"] != "fit",
            } for row in structural]
            common_path = root / "artifacts/common_population_manifest.parquet"
            pq.write_table(pa.Table.from_pylist(common), common_path)
            decisions = [{"block": "H", "decision": "admitted", "reviewer": "reviewer",
                          "systematic_bugs_resolved": "yes", "reason": "reviewed"}]
            with (root / "artifacts/feature_admission_decisions.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(decisions[0]))
                writer.writeheader()
                writer.writerows(decisions)
            report = {
                "population_manifest_sha256": hashlib.sha256(common_path.read_bytes()).hexdigest(),
                "admission_decisions": decisions, "review_issue_fields": 0, "admitted_blocks": ["H"],
                "feature_reports": {"H": {"automated_checks": "passed"}},
                "status": "available_adequacy_review_required",
            }
            (root / "artifacts/common_population_report.json").write_text(json.dumps(report), encoding="utf-8")
            (root / "artifacts/feature_review.csv").write_text("reviewed\n", encoding="utf-8")
            for name in ("feature_review_manifest.json", "raw_hashes.json"):
                (root / "artifacts" / name).write_text("{}\n", encoding="utf-8")
            for name in ("graphs.py", "feature_quality.py", "feature_freeze.py"):
                (root / "src" / name).write_text("source\n", encoding="utf-8")
            for name in ("requirements-stage4.txt", "decisions.yaml"):
                (root / name).write_text("input\n", encoding="utf-8")
            (root / "features/manifests/H.json").write_text(json.dumps({"source_sha256": {}}), encoding="utf-8")
            pq.write_table(pa.Table.from_pylist([{"image_id": image, "extraction_status": "ok"}
                                                for image in ("a", "b", "c", "d")]),
                           root / "features/H.parquet")
            with (root / "dataset/test.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["test_row_id", "a_image", "b_image"])
                writer.writeheader()
                writer.writerow({"test_row_id": "T1", "a_image": "a", "b_image": "d"})
            with (root / "tables/official_test_coverage.csv").open("w", encoding="utf-8", newline="") as stream:
                fields = ["test_row_id", "overlap_category", "missing_a_image", "missing_b_image"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"test_row_id": "T1", "overlap_category": "unseen_pair_of_known_photos",
                                 "missing_a_image": "false", "missing_b_image": "false"})
            with patch("src.graphs._input_inventory"), patch("src.graphs.validate_reviews", return_value=([], {})):
                outcome = run_graphs(root, {"real_data_modeling_allowed": True})
            self.assertEqual(outcome["status"], "graphs_and_masks_created")
            result = pq.read_table(root / "artifacts/g2_eligibility.parquet").to_pylist()
            self.assertTrue(next(row for row in result if row["train_row_id"] == "5")["common_strict_g2"])
            self.assertEqual((root / "figures/graph_support.pdf").read_bytes()[:4], b"%PDF")
            with self.assertRaises(FileExistsError):
                run_graphs(root, {"real_data_modeling_allowed": True})
            for name in outcome["outputs_sha256"]:
                root.joinpath(*name.split("/")).unlink()
            (root / "artifacts/graph_provenance.json").unlink()
            structural[-1]["a_image"], structural[-1]["a_cat"] = "unseen", "unseen_cat"
            pq.write_table(pa.Table.from_pylist(structural), root / "artifacts/rows_canonical.parquet")
            common[-1]["full_strict_g2"] = False
            common[-1]["common_visual_strict_g2"] = False
            pq.write_table(pa.Table.from_pylist(common), common_path)
            report["population_manifest_sha256"] = hashlib.sha256(common_path.read_bytes()).hexdigest()
            (root / "artifacts/common_population_report.json").write_text(json.dumps(report), encoding="utf-8")
            with patch("src.graphs._input_inventory"), patch("src.graphs.validate_reviews", return_value=([], {})):
                blocked = run_graphs(root, {"real_data_modeling_allowed": True})
            self.assertEqual(blocked["status"], "blocked_empty_full_strict_g2")


if __name__ == "__main__":
    unittest.main()
