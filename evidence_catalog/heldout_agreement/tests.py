"""Check agreement, unbiased Brier noise and orientation on known outcomes."""

import unittest

import pandas as pd

from analyze import describe_groups, group_data


class AgreementTests(unittest.TestCase):
    def test_oriented_repeats_and_loo(self):
        frame = pd.DataFrame([
            {"train_row_id": "1", "pair_id": "p", "orientation": 1, "outcome": 1, "a_cat": "a", "b_cat": "b"},
            {"train_row_id": "2", "pair_id": "p", "orientation": -1, "outcome": 1, "a_cat": "b", "b_cat": "a"},
            {"train_row_id": "3", "pair_id": "p", "orientation": 1, "outcome": 1, "a_cat": "a", "b_cat": "b"},
            {"train_row_id": "4", "pair_id": "s", "orientation": 1, "outcome": 1, "a_cat": "c", "b_cat": "d"},
        ])
        for level in ("photo_pair", "cat_pair"):
            part, group = group_data(frame, level)
            stats = describe_groups(part, group)
            self.assertEqual(stats["repeated_responses"], 3)
            self.assertAlmostEqual(stats["agreement_response_weighted"], 1/3)
            self.assertAlmostEqual(stats["estimated_oracle_brier"], 1/3)
            self.assertAlmostEqual(stats["loo_brier"], 1/2)
            self.assertAlmostEqual(stats["in_sample_brier_optimistic"], 2/9)
            self.assertIsNone(stats["plugin_accuracy_upper_bound"])

    def test_within_cat_excluded_only_from_cat_pair(self):
        frame = pd.DataFrame([
            {"train_row_id": str(i), "pair_id": "p", "orientation": 1, "outcome": 1, "a_cat": "a", "b_cat": "a"}
            for i in range(2)
        ])
        _, group = group_data(frame, "photo_pair")
        self.assertEqual(group.iloc[0].noise, 0)
        with self.assertRaises(ValueError):
            group_data(frame, "cat_pair")

    def test_known_symmetric_pair(self):
        rows = [{"train_row_id": str(i), "pair_id": "p", "orientation": 1, "outcome": y,
                 "a_cat": "a", "b_cat": "b"} for i, y in enumerate([1, 0])]
        part, group = group_data(pd.DataFrame(rows), "photo_pair")
        stats = describe_groups(part, group)
        self.assertEqual(stats["agreement_response_weighted"], 0)
        self.assertEqual(stats["estimated_oracle_brier"], .5)
        self.assertEqual(stats["loo_brier"], 1)


if __name__ == "__main__":
    unittest.main()
