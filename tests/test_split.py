import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from src.audit import audit_dataset, write_audit
from src.split import (
    allocate_pairs, canonical_rows, collect_pairs, largest_remainder,
    response_band, run_split, validate_manifest,
)


FIELDS = [
    "train_row_id", "a_cat", "b_cat", "a_image", "b_image", "choice",
    "comparison_type", "gender", "pet_history", "pet_care", "pet_contact",
    "future_pet", "mbti_ie",
]


def make_row(number, a, b, choice="A"):
    return {
        "train_row_id": f"TR{number:04d}", "a_cat": a.split("/")[0],
        "b_cat": b.split("/")[0], "a_image": a, "b_image": b, "choice": choice,
        "comparison_type": "between_cats", "gender": "",
        "pet_history": "", "pet_care": "occasionally", "pet_contact": "",
        "future_pet": "", "mbti_ie": "",
    }


class AllocationTests(unittest.TestCase):
    def test_largest_remainder_and_fixed_tie_order(self):
        self.assertEqual(largest_remainder(10), (7, 2, 1))
        self.assertEqual(largest_remainder(3), (2, 1, 0))
        self.assertEqual(largest_remainder(100), (70, 15, 15))

    def test_bands_and_under_supported_band_collapse(self):
        self.assertEqual([response_band(n) for n in (1, 2, 4, 5, 9, 10)],
                         ["1", "2-4", "2-4", "5-9", "5-9", "10+"])
        rows = [make_row(n, f"cat_a/{n}.jpg", f"cat_b/{n}.jpg")
                for n in range(30)]
        records = allocate_pairs(collect_pairs(rows), 20261028)
        self.assertEqual({r["allocation_stratum"] for r in records}, {"all"})
        self.assertEqual([sum(r["partition"] == p for r in records)
                          for p in ("fit", "validation", "test")], [21, 5, 4])
        self.assertEqual(records, allocate_pairs(collect_pairs(rows), 20261028))

    def test_supported_bands_allocate_separately(self):
        rows = []
        number = 0
        for count in (1, 2, 5, 10):
            for pair in range(20):
                for _ in range(count):
                    number += 1
                    rows.append(make_row(number, f"cat_a/{count}-{pair}.jpg",
                                         f"cat_b/{count}-{pair}.jpg"))
        records = allocate_pairs(collect_pairs(rows), 20261028)
        for band in ("1", "2-4", "5-9", "10+"):
            group = [r for r in records if r["allocation_stratum"] == band]
            self.assertEqual([sum(r["partition"] == p for r in group)
                              for p in ("fit", "validation", "test")], [14, 3, 3])

    def test_canonical_rows_keep_context_and_mask_test_outcome(self):
        rows = [
            {k: v for k, v in make_row(1, "cat_b/1.jpg", "cat_a/1.jpg").items()
             if k != "choice"},
            {k: v for k, v in make_row(2, "cat_b/2.jpg", "cat_a/2.jpg").items()
             if k != "choice"},
        ]
        pairs = collect_pairs(rows)
        keys = sorted(pairs)
        assignments = {keys[0]: "fit", keys[1]: "test"}
        outcomes = {r["train_row_id"]: "A" for r in rows
                    if r["a_image"].endswith("1.jpg")}
        result = canonical_rows(rows, assignments, outcomes)
        self.assertEqual(result[0]["pet_care"], "occasionally")
        self.assertEqual((result[0]["y_displayed"], result[0]["y_canonical"]), (1, 0))
        self.assertIsNone(result[1]["y_displayed"])
        self.assertIsNone(result[1]["y_canonical"])

    def test_reversed_pair_shares_assignment_and_response_context(self):
        first = make_row(1, "cat_a/1.jpg", "cat_b/1.jpg", "A")
        reverse = make_row(2, "cat_b/1.jpg", "cat_a/1.jpg", "B")
        rows = [{k: v for k, v in r.items() if k != "choice"} for r in (first, reverse)]
        pairs = collect_pairs(rows)
        self.assertEqual(len(pairs), 1)
        key = next(iter(pairs))
        self.assertEqual(pairs[key]["response_count"], 2)
        result = canonical_rows(rows, {key: "fit"}, {"TR0001": "A", "TR0002": "B"})
        self.assertEqual([r["y_canonical"] for r in result], [1, 1])
        self.assertEqual([r["pet_care"] for r in result], ["occasionally", "occasionally"])
        self.assertEqual([r["partition"] for r in result], ["fit", "fit"])

    def test_manifest_validation_rejects_pair_overlap_and_incomplete_splits(self):
        pairs = collect_pairs([make_row(1, "cat_a/1.jpg", "cat_b/1.jpg")])
        records = allocate_pairs(pairs, 1)
        with self.assertRaises(ValueError):
            validate_manifest(pairs, records + records)
        with self.assertRaises(ValueError):
            validate_manifest(pairs, [])


class SplitIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        dataset = self.root / "dataset"
        images = dataset / "images"
        images.mkdir(parents=True)
        rows = []
        for number in range(30):
            a, b = f"cat_a/{number}.jpg", f"cat_b/{number}.jpg"
            for image in (a, b):
                path = images.joinpath(*image.split("/"))
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(image.encode())
            rows.append(make_row(number, a, b, "A" if number % 2 else "B"))
        with (dataset / "train.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with (dataset / "test.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = ["test_row_id", *FIELDS[1:6], "prob_a", *FIELDS[6:]]
            csv.DictWriter(stream, fieldnames=fields).writeheader()
        write_audit(self.root, audit_dataset(self.root))
        self.config = {
            "real_data_modeling_allowed": True,
            "options": {"known_cat_g3": False},
            "seeds": {"new_primary_split_if_needed": 20261028},
            "split": {
                "unit": "unordered_photo_pair",
                "proportions_fit_validation_test": [0.70, 0.15, 0.15],
                "stratification": "comparison_type_by_pair_response_band",
                "bands": ["1", "2-4", "5-9", "10+"],
                "collapse_band_threshold_pairs": 20,
                "allocation": "largest_remainder",
                "allocation_tie_order": ["fit", "validation", "test"],
            },
        }

    def test_outputs_mask_holdout_and_reuse_existing_split(self):
        self.assertEqual(run_split(self.root, self.config), (30, 30, False))
        manifest_path = self.root / "artifacts" / "split_manifest.parquet"
        before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        manifest = pq.read_table(manifest_path).to_pylist()
        rows = pq.read_table(self.root / "artifacts" / "rows_canonical.parquet").to_pylist()
        self.assertEqual(len(rows), 30)
        self.assertEqual(len({record["pair_id"] for record in manifest}), 30)
        for row in rows:
            if row["partition"] == "test":
                self.assertIsNone(row["y_displayed"])
                self.assertIsNone(row["y_canonical"])
            else:
                self.assertIn(row["y_displayed"], (0, 1))
                self.assertIn(row["y_canonical"], (0, 1))
        self.config["seeds"]["new_primary_split_if_needed"] = 1
        self.assertEqual(run_split(self.root, self.config), (30, 30, True))
        self.assertEqual(hashlib.sha256(manifest_path.read_bytes()).hexdigest(), before)

    def test_stale_data_and_closed_governance_block_split(self):
        self.config["real_data_modeling_allowed"] = False
        with self.assertRaisesRegex(ValueError, "governance"):
            run_split(self.root, self.config)
        self.config["real_data_modeling_allowed"] = True
        with (self.root / "dataset" / "train.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "changed since Stage 1"):
            run_split(self.root, self.config)


if __name__ == "__main__":
    unittest.main()
