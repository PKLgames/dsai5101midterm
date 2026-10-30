import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.audit import audit_dataset, write_audit
from src.run import main


TRAIN_FIELDS = [
    "train_row_id", "a_cat", "b_cat", "a_image", "b_image", "choice",
    "comparison_type", "gender", "pet_history", "pet_care", "pet_contact",
    "future_pet", "mbti_ie",
]
TEST_FIELDS = ["test_row_id", *TRAIN_FIELDS[1:6], "prob_a", *TRAIN_FIELDS[6:]]


def row(row_id, a, b, test=False):
    a_cat, b_cat = a.split("/")[0], b.split("/")[0]
    return {
        "test_row_id" if test else "train_row_id": row_id,
        "a_cat": a_cat, "b_cat": b_cat, "a_image": a, "b_image": b,
        "choice": "" if test else "A", "comparison_type": (
            "within_cat" if a_cat == b_cat else "between_cats"
        ),
        "gender": "unregistered_gender", "pet_history": "never",
        "pet_care": "occasionally", "pet_contact": "often",
        "future_pet": "unsure", "mbti_ie": "unknown",
        **({"prob_a": ""} if test else {}),
    }


def write_rows(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.images = self.dataset / "images"
        self.images.mkdir(parents=True)
        self.train = [
            row("TR1", "cat_a/1.jpg", "cat_b/1.jpg"),
            row("TR2", "cat_a/2.jpg", "cat_b/2.jpg"),
            row("TR3", "cat_a/1.jpg", "cat_b/1.jpg"),
            row("TR4", "cat_a/1.jpg", "cat_a/1.jpg"),
        ]
        self.test = [
            row("T1", "cat_z/1.jpg", "cat_b/1.jpg", test=True),
            row("T2", "cat_a/3.jpg", "cat_b/1.jpg", test=True),
            row("T3", "cat_a/1.jpg", "cat_b/2.jpg", test=True),
            row("T4", "cat_a/1.jpg", "cat_b/1.jpg", test=True),
        ]
        for name, content in {
            "cat_a/1.jpg": b"identical", "cat_b/1.jpg": b"identical",
            "cat_a/2.jpg": b"two-a", "cat_b/2.jpg": b"two-b",
            "cat_a/3.jpg": b"three", "cat_z/1.jpg": b"unseen",
        }.items():
            path = self.images / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(content)
        self.save()

    def save(self):
        write_rows(self.dataset / "train.csv", TRAIN_FIELDS, self.train)
        write_rows(self.dataset / "test.csv", TEST_FIELDS, self.test)

    def test_overlap_precedence_and_deferred_flags(self):
        result = audit_dataset(self.root)
        self.assertEqual([r["overlap_category"] for r in result.coverage], [
            "at_least_one_unseen_cat", "at_least_one_unseen_photo",
            "unseen_pair_of_known_photos", "previously_observed_pair",
        ])
        self.assertTrue(all(r["fitting_component_support"] == "not_available_before_split"
                            for r in result.coverage))
        self.assertTrue(all(r["feature_availability"] == "not_available_before_feature_integration"
                            for r in result.coverage))
        self.assertFalse(result.blocked)

    def test_duplicate_content_self_comparisons_and_responses_are_retained(self):
        result = audit_dataset(self.root)
        actions = {(i["issue"], i["action"]) for i in result.issues}
        self.assertIn(("duplicate_image_content", "retain_distinct_paths"), actions)
        self.assertIn(("identical_image_self_comparison", "quarantine"), actions)
        self.assertIn(("identical_looking_responses", "retain_all"), actions)
        self.assertEqual(result.hashes["duplicate_image_contents"][0]["paths"],
                         ["cat_a/1.jpg", "cat_b/1.jpg"])

    def test_ambiguous_choice_and_unexpected_context_block_fitting(self):
        self.train[0]["choice"] = "maybe"
        self.train[0]["pet_care"] = "unexpected"
        self.save()
        result = audit_dataset(self.root)
        errors = {i["issue"] for i in result.issues if i["severity"] == "error"}
        self.assertTrue(result.blocked)
        self.assertIn("invalid_training_choice", errors)
        self.assertIn("unexpected_context_code", errors)
        self.assertNotIn("gender", " ".join(i["details"] for i in result.issues
                                            if i["issue"] == "unexpected_context_code"))

    def test_duplicate_ids_and_cat_conflicts_block_fitting(self):
        self.train[1]["train_row_id"] = "TR1"
        self.train[1]["a_cat"] = "cat_wrong"
        self.save()
        errors = {i["issue"] for i in audit_dataset(self.root).issues
                  if i["severity"] == "error"}
        self.assertIn("duplicate_row_id", errors)
        self.assertIn("cat_image_folder_mismatch", errors)

    def test_missing_image_is_not_an_identity_blocker(self):
        (self.images / "cat_z" / "1.jpg").unlink()
        result = audit_dataset(self.root)
        self.assertEqual(result.coverage[0]["missing_a_image"], "true")
        self.assertFalse(result.blocked)

    def test_missing_and_documented_unknown_are_distinct(self):
        self.train[0]["mbti_ie"] = ""
        self.save()
        result = audit_dataset(self.root)
        statuses = {(m["value"], m["status"]) for m in result.metrics
                    if m["partition"] == "train" and m["check"] == "context_mbti_ie"}
        self.assertIn(("<missing>", "missing"), statuses)
        self.assertIn(("unknown", "documented_unknown"), statuses)

    def test_nonblank_official_test_output_blocks_fitting(self):
        self.test[0]["choice"] = "A"
        self.test[0]["prob_a"] = "0.91"
        self.save()
        result = audit_dataset(self.root)
        self.assertTrue(result.blocked)
        self.assertIn("nonblank_official_test_output", {i["issue"] for i in result.issues})
        values = {m["value"] for m in result.metrics
                  if m["partition"] == "test" and m["check"] in {"observed_choice", "observed_prob_a"}}
        self.assertEqual(values, {"<blank>", "<nonblank>"})

    def test_invalid_path_and_comparison_mismatch_are_reported(self):
        self.train[0]["a_image"] = "../outside.jpg"
        self.train[1]["comparison_type"] = "within_cat"
        self.save()
        errors = {i["issue"] for i in audit_dataset(self.root).issues
                  if i["severity"] == "error"}
        self.assertIn("invalid_image_path", errors)
        self.assertIn("comparison_cat_mismatch", errors)

    def test_hashes_raw_unchanged_and_output_overwrite_protected(self):
        path = self.dataset / "train.csv"
        before = path.read_bytes()
        result = audit_dataset(self.root)
        self.assertEqual(result.hashes["files"]["dataset/train.csv"]["sha256"],
                         hashlib.sha256(before).hexdigest())
        outputs = write_audit(self.root, result)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(outputs), 4)
        with self.assertRaises(FileExistsError):
            write_audit(self.root, result)

    def test_all_overlap_categories_are_reported_including_zero(self):
        self.test = [self.test[0]]
        self.save()
        metrics = [m for m in audit_dataset(self.root).metrics
                   if m["check"] == "official_overlap_category"]
        self.assertEqual(len(metrics), 4)
        self.assertEqual(sum(m["count"] for m in metrics), 1)

    def test_cli_writes_findings_and_returns_failure_on_blocker(self):
        self.train[0]["choice"] = ""
        self.save()
        (self.root / "decisions.yaml").write_text(
            json.dumps({"primary_target": "strict_G2"}), encoding="utf-8"
        )
        self.assertEqual(main(["--root", str(self.root), "--stage", "audit"]), 1)
        with (self.root / "artifacts" / "cleaning_log.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            self.assertIn("invalid_training_choice",
                          {r["issue"] for r in csv.DictReader(stream)})

    def test_missing_required_column_fails_explicitly(self):
        write_rows(self.dataset / "test.csv", TEST_FIELDS[:-1], [
            {key: value for key, value in item.items() if key != TEST_FIELDS[-1]}
            for item in self.test
        ])
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            audit_dataset(self.root)


if __name__ == "__main__":
    unittest.main()
