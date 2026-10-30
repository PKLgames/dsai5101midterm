import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from src.synthesis import Evidence, direction, primary_claim, run_synthesis, validate_claims


def write_table(root, name, rows):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class EvidenceTests(unittest.TestCase):
    def test_cells_selectors_and_aggregates_are_replayable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_table(root, "pairs.csv", [
                {"partition": "development", "pair": "a", "responses": 4, "canonical_i_choices": 2},
                {"partition": "development", "pair": "b", "responses": 1, "canonical_i_choices": 1},
                {"partition": "fit", "pair": "a", "responses": 3, "canonical_i_choices": 0},
            ])
            e = Evidence(root)
            self.assertEqual(e.number("pairs.csv", {"partition": "development"}, "responses", "sum", None), "5")
            self.assertEqual(e.number("pairs.csv", {"partition": "development"}, "canonical_i_choices",
                                      "count_mixed_choices", None), "1")
            self.assertEqual(e.number("pairs.csv", {"partition": "development", "pair": "a"},
                                      "canonical_i_choices"), "2.000000")
            facts = json.loads(e.references_since(0))
            self.assertEqual(len(facts), 3)
            self.assertEqual(facts[0]["matched_rows"], 2)
            self.assertEqual(facts[0]["sha256"], hashlib.sha256((root / "pairs.csv").read_bytes()).hexdigest())
            self.assertEqual(facts[1]["operation"], "count_mixed_choices")
            self.assertEqual(e.number("pairs.csv", {"partition": "development"},
                                      "canonical_i_choices", "count_positive", None), "2")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                e.value("pairs.csv", {}, "responses")
            with self.assertRaisesRegex(ValueError, "empty"):
                e.value("pairs.csv", {"partition": "missing"}, "responses", "sum")

    def test_parquet_reference_and_invalid_numeric_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pq.write_table(pa.Table.from_pylist([
                {"simulation": "A", "reference_loss": .6},
                {"simulation": "A", "reference_loss": .6},
            ]), root / "simulation.parquet")
            e = Evidence(root)
            self.assertEqual(e.number("simulation.parquet", {"simulation": "A"}, "reference_loss", "mean"), "0.600000")
            write_table(root, "bad.csv", [{"value": "NaN"}])
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                e.number("bad.csv", {}, "value")
            write_table(root, "fraction.csv", [{"value": "1.5"}])
            with self.assertRaisesRegex(ValueError, "integer"):
                e.number("fraction.csv", {}, "value", digits=None)


class InterpretationTests(unittest.TestCase):
    def test_interval_direction_does_not_turn_uncertainty_into_equivalence(self):
        self.assertEqual(direction(-.01, .02), "direction_uncertain")
        self.assertEqual(direction(0, .02), "direction_uncertain")
        self.assertEqual(direction(.001, .01), "candidate_improves_under_bootstrap")
        self.assertEqual(direction(-.02, -.001), "candidate_worsens_under_bootstrap")
        with self.assertRaises(ValueError):
            direction(.1, -.1)

    def test_primary_result_checks_population_and_arithmetic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_table(root, "tables/primary_contrasts.csv", [{
                "contrast": "cat_identity", "reference": "M0", "candidate": "M1",
                "responses": 100, "pairs": 30, "interval_level": .9875,
                "response_loss_difference": .02, "response_interval_low": .001,
                "response_interval_high": .04, "pair_loss_difference": .019,
                "pair_interval_low": .002, "pair_interval_high": .035,
            }])
            write_table(root, "tables/main_results.csv", [
                {"population": "common", "model_id": "M0", "responses": 100, "pairs": 30,
                 "response_log_loss": .7},
                {"population": "common", "model_id": "M1", "responses": 100, "pairs": 30,
                 "response_log_loss": .68},
            ])
            e = Evidence(root)
            result, state = primary_claim(e, "cat_identity")
            self.assertEqual(state, "candidate_improves_under_bootstrap")
            self.assertIn("0.987500", result)
            self.assertIn("100 responses across 30 unordered pairs", result)
            self.assertTrue(all(fact["rendered"] in result for fact in e.facts))
            e.tables["tables/main_results.csv"][1]["responses"] = "99"
            with self.assertRaisesRegex(ValueError, "population mismatch"):
                primary_claim(e, "cat_identity")
            e.tables["tables/main_results.csv"][1]["responses"] = "100"
            e.tables["tables/main_results.csv"][1]["response_log_loss"] = ".6"
            with self.assertRaisesRegex(ValueError, "reconcile"):
                primary_claim(e, "cat_identity")

    def test_gate_failure_prevents_any_synthesis_evidence_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("src.synthesis.verify_inputs", side_effect=ValueError("changed frozen input")), \
                 patch("src.synthesis.build_claims") as build:
                with self.assertRaisesRegex(ValueError, "changed frozen"):
                    run_synthesis(root)
            build.assert_not_called()
            self.assertFalse((root / "claims_final.csv").exists())

    def test_existing_outputs_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "claims_final.csv").write_text("existing", encoding="utf-8")
            with patch("src.synthesis.verify_inputs") as gate:
                with self.assertRaises(FileExistsError):
                    run_synthesis(root)
            gate.assert_not_called()
            self.assertEqual((root / "claims_final.csv").read_text(encoding="utf-8"), "existing")

    def test_final_registry_requires_all_questions_and_citations(self):
        e = Evidence(Path("."))
        with self.assertRaisesRegex(ValueError, "registered research question"):
            validate_claims([], e)

    def test_synthesis_publishes_both_tables_with_evidence_and_output_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_table(root, "claims.csv", [{"claim_id": f"C{i:02d}"} for i in range(1, 8)])
            for name in ("artifacts/model_selection.json", "artifacts/model_freeze.json",
                         "artifacts/evaluation_provenance.json"):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("{}", encoding="utf-8")
            (root / "src").mkdir()
            (root / "src/synthesis.py").write_text("synthetic", encoding="utf-8")

            def synthetic_claims(evidence, selection, planned):
                result = []
                for claim in planned:
                    start = len(evidence.facts)
                    n = evidence.number("claims.csv", {"claim_id": claim["claim_id"]},
                                        "claim_id", "count", None)
                    result.append({
                        "claim_id": claim["claim_id"], "target": "None", "evidence_status": "Real-data claim",
                        "question": "Synthetic question", "measured_result": f"Matched rows: {n}",
                        "supported_interpretation": "Synthetic interpretation",
                        "alternative_explanations": "Synthetic alternatives", "nonclaim": "No causal effect",
                        "next_discriminating_evidence": "Independent replication", "partition": "synthetic",
                        "limitations": "Synthetic test only",
                        "source_evidence_json": evidence.references_since(start),
                    })
                return result

            with patch("src.synthesis.verify_inputs", return_value={"selected_representation": "C"}), \
                 patch("src.synthesis.build_claims", side_effect=synthetic_claims):
                report = run_synthesis(root)
            self.assertEqual(report["claims"], 7)
            self.assertFalse(report["new_model_fits"])
            self.assertFalse(report["test_evaluation_rerun"])
            first = (root / "claims_final.csv").read_bytes()
            self.assertEqual(first, (root / "tables/diagnostic_synthesis.csv").read_bytes())
            self.assertEqual(hashlib.sha256(first).hexdigest(), report["output_sha256"]["claims_final.csv"])
            with (root / "claims_final.csv").open(encoding="utf-8", newline="") as stream:
                claims = list(csv.DictReader(stream))
            self.assertEqual(len(claims), 7)
            self.assertTrue(all(json.loads(row["source_evidence_json"]) for row in claims))


if __name__ == "__main__":
    unittest.main()
