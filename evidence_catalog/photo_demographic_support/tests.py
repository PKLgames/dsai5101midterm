"""Synthetic tests for per-photo demographic counting."""

import unittest

from analyze import compute


def row(key, ai, bi, ac, bc, gender, pair, kind):
    return {"train_row_id": key, "a_image": ai, "b_image": bi, "a_cat": ac, "b_cat": bc,
            "gender": gender, "pair_id": pair, "comparison_type": kind, "pet_history": "never",
            "pet_care": "never", "pet_contact": "rarely", "future_pet": "yes", "mbti_ie": "unknown"}


class PhotoCountTests(unittest.TestCase):
    def setUp(self):
        self.between = row("r1", "a1", "b1", "a", "b", "woman", "p1", "between_cats")
        self.within = row("r2", "a1", "a2", "a", "a", "man", "p2", "within_cat")
        self.extra = row("r3", "a1", "b1", "a", "b", "man", "p1", "between_cats")
        self.scopes = {"fit": [self.between, self.within], "development": [self.between, self.within, self.extra]}

    def test_both_within_cat_photos_count_once_and_zeros_included(self):
        detail, _, _, totals = compute(self.scopes)
        cells = {r["photo_id"]: r for r in detail if r["partition"] == "fit" and r["demographic"] == "gender" and r["group"] == "man"}
        self.assertEqual([cells[p]["responses_with_photo"] for p in ("a1", "a2", "b1")], [1, 1, 0])
        self.assertEqual(cells["a2"]["within_cat_responses"], 1)
        self.assertEqual(totals["partitions"]["fit"]["photo_appearances_per_demographic"], 4)

    def test_pair_and_photo_overlap(self):
        _, _, comparisons, _ = compute(self.scopes)
        r = next(r for r in comparisons if r["partition"] == "development" and r["demographic"] == "gender" and {r["group_a"], r["group_b"]} == {"man", "woman"})
        self.assertEqual(r["photos_with_any_in_both_groups"], 2)
        self.assertEqual(r["shared_pairs"], 1)
        self.assertEqual(r["shared_pairs_with_5_in_each_group"], 0)

    def test_distinct_pairs_not_response_count(self):
        extra = dict(self.between, train_row_id="r4")
        scope = {"fit": [self.between, self.within, extra], "development": [self.between, self.within, extra, self.extra]}
        detail, _, _, _ = compute(scope)
        r = next(r for r in detail if (r["partition"], r["demographic"], r["group"], r["photo_id"]) == ("fit", "gender", "woman", "a1"))
        self.assertEqual((r["responses_with_photo"], r["distinct_photo_pairs"]), (2, 1))

    def test_rejects_duplicate_ids_and_inconsistent_photo_cat(self):
        with self.assertRaises(ValueError):
            compute({"fit": [self.between, self.between], "development": [self.between, self.between]})
        bad = dict(self.extra, a_cat="other")
        with self.assertRaises(ValueError):
            compute({"fit": [self.between, self.within], "development": [self.between, self.within, bad]})

    def test_outcome_independence(self):
        with_outcomes = {name: [dict(r, choice="A", y_displayed=1) for r in rows] for name, rows in self.scopes.items()}
        self.assertEqual(compute(self.scopes), compute(with_outcomes))


if __name__ == "__main__":
    unittest.main()
