import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.exploratory_semantics import (
    FIELDS, HARD, NUMERIC, FIGURE, SUPPORT_FIGURE, append_payload, feature_events,
    load_development_rows, run, summarize, validate_rows,
)

HEADERS = [
    "claim_id", "task_roles", "target", "analysis_target", "activity", "evidence_status", "progress",
    "result_status", "question", "partition", "measured_result", "supported_interpretation",
    "alternative_explanations", "nonclaim", "next_discriminating_evidence", "source_evidence_json", "limitations",
]


def response(index, outcome=1, gender="man", partition="fit"):
    return {
        "train_row_id": str(index), "pair_id": "pair", "image_i": "cat/a.jpg", "image_j": "cat/b.jpg",
        "partition": partition, "y_canonical": outcome, "comparison_type": "within_cat", "gender": gender,
        "pet_history": "never", "pet_care": "occasionally", "pet_contact": "often",
        "future_pet": "unsure", "mbti_ie": "unknown",
    }


def feature(image, category, number=0, detection="cat_detected"):
    return {
        "image_id": image, "extraction_status": "ok", "detection_status": detection,
        **{name: category for name in HARD}, **{name: float(number) for name in NUMERIC},
    }


def protocol():
    return {
        "demographics": list(FIELDS), "hard_labels": list(HARD), "numeric_features": list(NUMERIC),
        "heatmap_support": {"responses": 1, "distinct_pairs": 1}, "numeric_tie_tolerance": 1e-12,
    }


class EventTests(unittest.TestCase):
    def test_category_direction_and_numeric_high_endpoint_follow_canonical_outcome(self):
        rows = [response(1, 1), response(2, 0)]
        by_id = {x["image_id"]: x for x in [
            feature("cat/a.jpg", "orange", 1), feature("cat/b.jpg", "black", 3),
        ]}
        variable = {"kind": "category", "family": "colour_primary", "category": "orange"}
        usable, differs, wins, _ = feature_events(rows, by_id, variable, 1e-12)
        np.testing.assert_array_equal(usable, [True, True])
        np.testing.assert_array_equal(differs, [True, True])
        np.testing.assert_array_equal(wins, [True, False])
        variable = {"kind": "numeric", "family": "share_orange"}
        _, _, wins, gap = feature_events(rows, by_id, variable, 1e-12)
        np.testing.assert_array_equal(wins, [False, True])
        np.testing.assert_array_equal(gap, [-2., 2.])

    def test_no_detection_unknown_and_ties_are_not_silent_zero_measurements(self):
        rows = [response(1)]
        first = feature("cat/a.jpg", "orange", 1, detection="no_detection")
        second = feature("cat/b.jpg", "black", 1)
        variable = {"kind": "numeric", "family": "share_orange"}
        self.assertFalse(feature_events(rows, {x["image_id"]: x for x in (first, second)}, variable, 1e-12)[0][0])
        first["detection_status"] = "cat_detected"
        usable, differs, _, _ = feature_events(rows, {x["image_id"]: x for x in (first, second)}, variable, 1e-12)
        self.assertTrue(usable[0])
        self.assertFalse(differs[0])
        first["colour_primary"] = ""
        cat = {"kind": "category", "family": "colour_primary", "category": "black"}
        self.assertFalse(feature_events(rows, {x["image_id"]: x for x in (first, second)}, cat, 1e-12)[0][0])

    def test_test_rows_and_invalid_demographic_codes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Only labeled"):
            validate_rows([response(1, partition="test")])
        bad = response(1)
        bad["pet_care"] = "invented"
        with self.assertRaisesRegex(ValueError, "Unexpected demographic"):
            validate_rows([bad])
        validate_rows([response(1, gender="observed_other_gender")])

    def test_parquet_loader_never_returns_internal_test_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifacts").mkdir()
            pq.write_table(pa.Table.from_pylist([
                response(1, partition="fit"), response(2, partition="validation"), response(3, outcome=999, partition="test"),
            ]), root / "artifacts/rows_canonical.parquet")
            rows = load_development_rows(root)
            self.assertEqual([x["train_row_id"] for x in rows], ["1", "2"])
            validate_rows(rows)


class SummaryTests(unittest.TestCase):
    def test_same_pair_support_is_required_and_missing_demographics_are_preserved(self):
        rows = [response(i, gender="man" if i < 5 else "woman") for i in range(10)]
        features = [feature("cat/a.jpg", "orange", 1), feature("cat/b.jpg", "black", 2)]
        rates, overlap, _ = summarize(rows, features, protocol())
        contrast = next(x for x in overlap if x["demographic"] == "gender" and x["feature"] == "colour_primary=orange")
        self.assertEqual(contrast["shared_pairs_with_5_responses_each"], 1)
        self.assertEqual(contrast["matched_pair_mean_rate_difference_a_minus_b"], 0.)
        _, sparse_overlap, _ = summarize(rows[:9], features, protocol())
        contrast = next(x for x in sparse_overlap if x["demographic"] == "gender" and x["feature"] == "colour_primary=orange")
        self.assertIsNone(contrast["matched_pair_mean_rate_difference_a_minus_b"])
        self.assertEqual(contrast["comparison_status"], "insufficient_per_pair_group_support")
        one = response(100)
        one["gender"] = ""
        rates, _, _ = summarize([one], features, protocol())
        self.assertTrue(any(x["demographic"] == "gender" and x["group"] == "<missing>" for x in rates))
        self.assertTrue(any(x["demographic"] == "mbti_ie" and x["group"] == "unknown" for x in rates))
        for row in rates:
            self.assertEqual(row["usable_measurement_responses"] + row["excluded_measurement_responses"],
                             row["group_responses"])
            self.assertEqual(row["non_discriminating_responses"] + row["discriminating_responses"],
                             row["usable_measurement_responses"])

    def test_pair_composition_can_differ_without_a_matched_contrast(self):
        first, second = response(1, 1, "man"), response(2, 0, "woman")
        second.update({"pair_id": "another", "image_i": "cat/c.jpg", "image_j": "cat/d.jpg"})
        features = [feature("cat/a.jpg", "orange"), feature("cat/b.jpg", "black"),
                    feature("cat/c.jpg", "orange"), feature("cat/d.jpg", "black")]
        rates, overlap, _ = summarize([first, second], features, protocol())
        high = next(x for x in rates if x["demographic"] == "gender" and x["group"] == "man"
                    and x["feature"] == "colour_primary=orange")
        low = next(x for x in rates if x["demographic"] == "gender" and x["group"] == "woman"
                   and x["feature"] == "colour_primary=orange")
        self.assertEqual((high["choice_rate"], low["choice_rate"]), (1., 0.))
        comparison = next(x for x in overlap if x["demographic"] == "gender" and x["feature"] == "colour_primary=orange")
        self.assertEqual(comparison["shared_discriminating_pairs"], 0)
        self.assertIsNone(comparison["matched_pair_mean_rate_difference_a_minus_b"])


class AppendTests(unittest.TestCase):
    def test_append_bytes_never_rewrite_existing_rows_or_header(self):
        parent = b"claim_id,result\r\nC01,existing\r\n"
        payload = append_payload(parent, ["claim_id", "result"], [{"claim_id": "X105-01", "result": "exploratory"}])
        self.assertTrue((parent + payload).startswith(parent))
        reader = list(csv.DictReader(io.StringIO((parent + payload).decode())))
        self.assertEqual([x["claim_id"] for x in reader], ["C01", "X105-01"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            append_payload(parent, ["claim_id", "result"], [{"claim_id": "C01", "result": "new"}])

    def test_stage_run_appends_and_records_prefix_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("artifacts", "tables", "features", "src"):
                (root / name).mkdir()
            parent_row = {key: "original" for key in HEADERS}
            parent_row["claim_id"] = "C01"
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=HEADERS, lineterminator="\n")
            writer.writeheader()
            writer.writerow(parent_row)
            parent = buffer.getvalue().encode()
            for name in ("claims_final.csv", "tables/diagnostic_synthesis.csv"):
                (root / name).write_bytes(parent)
            proto = protocol()
            (root / "artifacts/stage10_5_protocol.json").write_text(json.dumps(proto), encoding="utf-8")
            (root / "artifacts/stage10_provenance.json").write_text(json.dumps({
                "claims": 1, "input_sha256": {}, "output_sha256": {
                    name: hashlib.sha256(parent).hexdigest()
                    for name in ("claims_final.csv", "tables/diagnostic_synthesis.csv")
                },
            }), encoding="utf-8")
            (root / "artifacts/model_freeze.json").write_text("{}", encoding="utf-8")
            (root / "src/exploratory_semantics.py").write_text("source", encoding="utf-8")
            (root / "features/manifests").mkdir()
            (root / "features/manifests/Y.json").write_text("{}", encoding="utf-8")
            rows = [response(i, outcome=int(i < 6), gender="man" if i < 6 else "woman")
                    for i in range(12)]
            rows.append(response(999, outcome=999, partition="test"))
            pq.write_table(pa.Table.from_pylist(rows), root / "artifacts/rows_canonical.parquet")
            pq.write_table(pa.Table.from_pylist([
                feature("cat/a.jpg", "orange", 1), feature("cat/b.jpg", "black", 2),
                feature("unseen/not-development.jpg", "unseen_category", 5),
            ]), root / "features/Y.parquet")

            def plots(directory, rates, support):
                (directory / "figures").mkdir(exist_ok=True)
                for name in (FIGURE, SUPPORT_FIGURE):
                    (directory / name).write_bytes(b"%PDF-synthetic")

            with patch("src.exploratory_semantics.validate_freeze", return_value={"selected_representation": "C"}), \
                 patch("src.exploratory_semantics._plot", side_effect=plots):
                report = run(root)
            self.assertEqual(report["development_responses"], 12)
            self.assertFalse(report["internal_test_outcomes_read"])
            for name, lineage in report["append_lineage"].items():
                data = (root / name).read_bytes()
                self.assertEqual(data[:lineage["parent_bytes"]], parent)
                self.assertEqual(lineage["parent_rows"], 1)
                self.assertEqual(lineage["current_rows"], 7)
                parsed = list(csv.DictReader(io.StringIO(data.decode())))
                self.assertEqual(parsed[0], parent_row)
                self.assertTrue(all(x["task_roles"] == "T2" for x in parsed[1:]))
            with (root / "tables/exploratory_demographic_semantic_rates.csv").open(encoding="utf-8", newline="") as stream:
                self.assertFalse(any("unseen_category" in x["feature"] for x in csv.DictReader(stream)))
            with self.assertRaises(FileExistsError):
                run(root)


if __name__ == "__main__":
    unittest.main()
