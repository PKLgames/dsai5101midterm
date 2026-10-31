"""Focused guards for G4 cold-start and diagnostic utility behavior."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

import common
from experiment import design
from src.abstention_utility import point_metrics
from src.learning_curves import nested_pair_sets
from trainer import cat_differences
from transfer import source_rows


class ExtensionTests(unittest.TestCase):
    def test_g4_unseen_cat_zero_applies_to_pooled_and_interaction_blocks(self):
        rows = [{"a_cat": "new", "b_cat": "known", "held_out_cat": "new"},
                {"a_cat": "known", "b_cat": "new", "held_out_cat": "new"}]
        diff = cat_differences(rows, ["known"], True)
        side = sparse.csr_matrix([[1.], [1.]])
        np.testing.assert_array_equal(design(side, diff, True).toarray(),
                                      [[1., 1., -1., -1.], [1., 1., 1., 1.]])

    def test_g4_rejects_unexpected_unseen_endpoints(self):
        with self.assertRaises(ValueError):
            cat_differences([{"a_cat": "new", "b_cat": "other", "held_out_cat": "new"}], ["known"], True)
        with self.assertRaises(ValueError):
            cat_differences([{"a_cat": "new", "b_cat": "known", "held_out_cat": "wrong"}], ["known"], True)

    def test_abstention_denominator_includes_abstained(self):
        result, retained, _, _ = point_metrics([.9, .8, .51], [1, 0, 1], [.1, 1.6, .67],
                                                ["a", "b", "c"], .7)
        self.assertEqual(result["retained_responses"], 2)
        self.assertEqual(result["abstained_responses"], 1)
        self.assertEqual(result["utility_per_original_response"], 0.)
        np.testing.assert_array_equal(retained, [True, True, False])

    def test_nested_pair_sets_are_nested_and_deterministic(self):
        fit = [{"pair_id": str(i)} for i in range(20)]
        result = nested_pair_sets(fit, 20261030, [.25, .5, .75, 1.])
        self.assertEqual(len(result[.25]), 5)
        self.assertTrue(result[.25] <= result[.5] <= result[.75] <= result[1.])
        self.assertEqual(result, nested_pair_sets(fit, 20261030, [.25, .5, .75, 1.]))

    def test_reserved_splits_disjoint_with_expected_structure(self):
        for target in ("G3", "G4"):
            fit = source_rows(target, "core_fit", False)
            validation = source_rows(target, "validation", False)
            test = source_rows(target, "test", False)
            sets = [{r["train_row_id"] for r in rows} for rows in (fit, validation, test)]
            self.assertFalse(sets[0] & sets[1])
            self.assertFalse(sets[0] & sets[2])
            self.assertFalse(sets[1] & sets[2])
            fit_photos = {r[f"{side}_image"] for r in fit for side in ("a", "b")}
            fit_cats = {r[f"{side}_cat"] for r in fit for side in ("a", "b")}
            for row in validation + test:
                if target == "G3":
                    self.assertNotIn(row["held_out_photo"], fit_photos)
                    self.assertIn(row["a_cat"], fit_cats)
                    self.assertIn(row["b_cat"], fit_cats)
                else:
                    self.assertNotIn(row["held_out_cat"], fit_cats)
                    self.assertEqual(sum(row[f"{s}_cat"] == row["held_out_cat"] for s in ("a", "b")), 1)


if __name__ == "__main__":
    unittest.main()
