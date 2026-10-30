import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.feature_freeze import CRITERIA, compute_population, validate_admissions, validate_reviews
from src.feature_quality import IMMUTABLE_REVIEW_FIELDS, review_fingerprint


class FeatureFreezeTests(unittest.TestCase):
    def test_population_uses_matched_fitting_graph_not_full_graph(self):
        split = [
            {"image_i": "a", "image_j": "b", "partition": "fit"},
            {"image_i": "b", "image_j": "c", "partition": "fit"},
            {"image_i": "c", "image_j": "d", "partition": "fit"},
        ]
        rows = [
            {"train_row_id": "1", "a_image": "a", "b_image": "b", "partition": "fit"},
            {"train_row_id": "2", "a_image": "c", "b_image": "d", "partition": "fit"},
            {"train_row_id": "3", "a_image": "a", "b_image": "c", "partition": "validation"},
            {"train_row_id": "4", "a_image": "a", "b_image": "d", "partition": "test"},
        ]
        result = compute_population(rows, split, {"a", "b", "c"}, ["H", "C"])
        self.assertTrue(result[0]["common_visual_fit"])
        self.assertFalse(result[1]["common_visual_fit"])
        self.assertTrue(result[2]["full_strict_g2"])
        self.assertTrue(result[2]["common_visual_strict_g2"])
        self.assertTrue(result[3]["full_strict_g2"])
        self.assertFalse(result[3]["common_visual_strict_g2"])
        self.assertFalse(any(row["common_visual_fit"] or row["common_visual_strict_g2"]
                             for row in compute_population(rows, split, set(), [])))

    def test_admission_cannot_bypass_automated_checks(self):
        reports = {name: {"automated_checks": "passed"} for name in ("H", "C", "Y")}
        decisions = [
            {"block": name, "decision": "admitted", "reviewer": "reviewer",
             "systematic_bugs_resolved": "yes", "reason": "reviewed"}
            for name in ("H", "C", "Y")
        ]
        self.assertEqual(validate_admissions(decisions, reports), ["H", "C", "Y"])
        reports["Y"]["automated_checks"] = "failed"
        with self.assertRaisesRegex(ValueError, "automated checks"):
            validate_admissions(decisions, reports)
        decisions[2]["decision"] = "quarantined"
        self.assertEqual(validate_admissions(decisions, reports), ["H", "C"])

    def test_manual_review_must_be_complete_and_sample_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifacts" / "review_panels").mkdir(parents=True)
            (root / "src").mkdir()
            split = root / "artifacts" / "split_manifest.parquet"
            split.write_bytes(b"split")
            code = root / "src" / "feature_quality.py"
            code.write_bytes(b"code")
            panel = root / "artifacts" / "review_panels" / "panel.jpg"
            panel.write_bytes(b"panel")
            records = []
            for reviewer in ("Lu Yuqing", "Yuen Lok Yan"):
                record = {field: "" for field in IMMUTABLE_REVIEW_FIELDS}
                record.update({
                    "image_id": "cat/a.jpg", "image_path": "dataset/images/cat/a.jpg",
                    "review_panel_path": "artifacts/review_panels/panel.jpg",
                    "selection": "all_fitting", "reviewer": reviewer, "review_status": "pending",
                    **{criterion: "" for criterion in CRITERIA}, "notes": "",
                })
                records.append(record)
            path = root / "artifacts" / "feature_review.csv"
            fields = list(records[0])

            def save():
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(records)

            meta = {
                "review_rows": 2, "image_count": 1,
                "reviewer_names": ["Lu Yuqing", "Yuen Lok Yan"],
                "immutable_review_sha256": review_fingerprint(records),
                "split_manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
                "quality_code_sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
                "feature_file_sha256": {},
            }
            (root / "artifacts" / "feature_review_manifest.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            save()
            with self.assertRaisesRegex(ValueError, "not complete"):
                validate_reviews(root, {"cat/a.jpg"})
            for row in records:
                row["review_status"] = "reviewed"
                row.update({criterion: "pass" for criterion in CRITERIA})
            save()
            self.assertEqual(len(validate_reviews(root, {"cat/a.jpg"})[0]), 2)
            records[0]["image_id"] = "cat/other.jpg"
            save()
            with self.assertRaisesRegex(ValueError, "changed"):
                validate_reviews(root, {"cat/a.jpg", "cat/other.jpg"})


if __name__ == "__main__":
    unittest.main()
