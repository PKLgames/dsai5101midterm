"""Focused tests for within-cat saved-prediction analysis."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.special import expit

from analyze import MODELS, align_predictions, calculate


def fixture():
    structure = [
        {"train_row_id": "1", "pair_id": "p1", "held_out_photo": "h1", "a_cat": "c1", "b_cat": "c1",
         "comparison_type": "within_cat", "partition": "fit"},
        {"train_row_id": "2", "pair_id": "p2", "held_out_photo": "h2", "a_cat": "c2", "b_cat": "c2",
         "comparison_type": "within_cat", "partition": "test"},
        {"train_row_id": "3", "pair_id": "p3", "held_out_photo": "h3", "a_cat": "c1", "b_cat": "c2",
         "comparison_type": "between_cats", "partition": "fit"},
    ]
    records = []
    for model in MODELS:
        for i, row in enumerate(structure):
            y = int(i != 1)
            logit = 0.0 if model == "M1" else 0.2
            records.append({"model": model, "train_row_id": row["train_row_id"], "pair_id": row["pair_id"],
                            "held_out_photo": row["held_out_photo"], "outcome_a": y, "logit": logit,
                            "prob_a": float(expit(logit)),
                            "loss": float(y*np.logaddexp(0., -logit)+(1-y)*np.logaddexp(0., logit))})
    protocol = {"operational_support": {"minimum_responses": 1, "minimum_pairs": 1},
                "uncertainty": {"replicates": 20, "seed": 20261101, "limitations": "test"},
                "interpretation": "test", "prior_exposure": "test"}
    return structure, records, protocol


class WithinCatTests(unittest.TestCase):
    def test_subgroup_excludes_between_and_preserves_matched_counts(self):
        structure, records, protocol = fixture()
        results, contrasts, summary = calculate(structure, align_predictions(structure, records), "C", 0., protocol)
        self.assertEqual((summary["responses"], summary["pairs"], summary["held_out_photos"]), (2, 2, 2))
        self.assertEqual(summary["g2_test_partition_overlap_responses"], 1)
        self.assertEqual(len(results), 8)
        self.assertEqual(len(contrasts), 5)
        self.assertTrue(all(row["responses"] == 2 for row in results))
        self.assertEqual(next(row for row in contrasts if row["reference"] == "M3-C")["ci_low"], 0.)

    def test_incomplete_or_duplicate_predictions_fail(self):
        structure, records, _ = fixture()
        with self.assertRaises(ValueError):
            align_predictions(structure, records[:-1])
        with self.assertRaises(ValueError):
            align_predictions(structure, records + [records[0]])

    def test_inconsistent_prediction_or_outcome_fails(self):
        structure, records, _ = fixture()
        records[0]["loss"] += .1
        with self.assertRaises(ValueError):
            align_predictions(structure, records)
        structure, records, _ = fixture()
        row = records[3]
        row["outcome_a"] = 1 - row["outcome_a"]
        row["loss"] = float(row["outcome_a"]*np.logaddexp(0., -row["logit"])
                            +(1-row["outcome_a"])*np.logaddexp(0., row["logit"]))
        with self.assertRaises(ValueError):
            align_predictions(structure, records)

    def test_type_mismatch_and_m1_invariant_fail(self):
        structure, records, protocol = fixture()
        aligned = align_predictions(structure, records)
        structure[0]["b_cat"] = "other"
        with self.assertRaises(ValueError):
            calculate(structure, aligned, "C", 0., protocol)
        structure, records, protocol = fixture()
        with self.assertRaises(ValueError):
            calculate(structure, align_predictions(structure, records), "C", 0.5, protocol)

    def test_unsupported_subgroup_fails_explicitly(self):
        structure, records, protocol = fixture()
        protocol["operational_support"]["minimum_responses"] = 3
        with self.assertRaises(ValueError):
            calculate(structure, align_predictions(structure, records), "C", 0., protocol)


if __name__ == "__main__":
    unittest.main()
