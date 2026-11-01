"""Focused tests for outcome-blind cat appearance and matched-pair counting."""

import unittest

from analyze import DEMOGRAPHICS, category, compute


def row(key, a_cat, b_cat, a_image, b_image, gender, comparison, pair):
    return {"train_row_id": key, "a_cat": a_cat, "b_cat": b_cat, "a_image": a_image, "b_image": b_image,
            "comparison_type": comparison, "pair_id": pair, "gender": gender, "pet_history": "never",
            "pet_care": "never", "pet_contact": "rarely", "future_pet": "yes", "mbti_ie": "unknown"}


class CountTests(unittest.TestCase):
    def setUp(self):
        self.between = row("r1", "a", "b", "a1", "b1", "woman", "between_cats", "p1")
        self.within = row("r2", "a", "a", "a1", "a2", "man", "within_cat", "p2")
        self.extra = row("r3", "a", "b", "a1", "b1", "man", "between_cats", "p1")
        self.scopes = {"fit": [self.between, self.within], "development": [self.between, self.within, self.extra]}

    def test_within_cat_counted_once_but_both_images_retained(self):
        detail, _, _, totals = compute(self.scopes)
        item = next(r for r in detail if r["partition"] == "fit" and r["demographic"] == "gender"
                    and r["group"] == "man" and r["cat_id"] == "a")
        self.assertEqual(item["responses_with_cat"], 1)
        self.assertEqual(item["within_cat_responses"], 1)
        self.assertEqual(item["distinct_photos_of_cat"], 2)
        self.assertEqual(item["distinct_other_opponent_cats"], 0)
        self.assertEqual(totals["partitions"]["fit"]["cat_response_appearances_per_demographic"], 3)

    def test_zero_cells_and_matched_pair_overlap(self):
        detail, summary, comparisons, _ = compute(self.scopes)
        cell = next(r for r in detail if r["partition"] == "fit" and r["demographic"] == "gender"
                    and r["group"] == "man" and r["cat_id"] == "b")
        self.assertEqual(cell["responses_with_cat"], 0)
        group = next(r for r in summary if r["partition"] == "fit" and r["demographic"] == "gender" and r["group"] == "man")
        self.assertEqual(group["cats_with_zero"], 1)
        pair = next(r for r in comparisons if r["partition"] == "development" and r["demographic"] == "gender"
                    and {r["group_a"], r["group_b"]} == {"man", "woman"})
        self.assertEqual(pair["shared_pairs"], 1)
        self.assertEqual(pair["shared_pairs_with_5_responses_in_each_group"], 0)

    def test_all_demographics_partition_and_unexpected_category(self):
        detail, summary, _, totals = compute(self.scopes)
        for partition, counts in totals["partitions"].items():
            for name in DEMOGRAPHICS:
                self.assertEqual(sum(r["cat_response_appearances"] for r in summary
                                     if r["partition"] == partition and r["demographic"] == name),
                                 counts["cat_response_appearances_per_demographic"])
        invalid = dict(self.between, pet_care="not-recorded")
        with self.assertRaises(ValueError):
            category(invalid, "pet_care")

    def test_does_not_depend_on_outcomes(self):
        modified = {key: [dict(row, choice="A", y_displayed=1) for row in rows] for key, rows in self.scopes.items()}
        self.assertEqual(compute(self.scopes), compute(modified))


if __name__ == "__main__":
    unittest.main()
