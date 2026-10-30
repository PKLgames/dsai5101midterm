"""Focused tests for G4 reservation and cold-start design."""

from __future__ import annotations

import csv
import json
import unittest

import numpy as np
from reserve import cat_groups, reserve_groups
from shared import HERE, identity_design, load_structure
from src.development import identity_matrix


class G4Tests(unittest.TestCase):
    def test_duplicate_content_cats_stay_together(self):
        rows = [{"a_cat": "a", "b_cat": "b", "a_image": "a1", "b_image": "b1"},
                {"a_cat": "b", "b_cat": "c", "a_image": "b1", "b_image": "c1"}]
        self.assertEqual(set(cat_groups(rows, {"a1": "same", "b1": "same", "c1": "other"})),
                         {("a", "b"), ("c",)})

    def test_reservation_is_deterministic(self):
        groups = [(str(i),) for i in range(30)]
        forward = reserve_groups(groups)
        self.assertEqual(forward, reserve_groups(list(reversed(groups))))
        self.assertEqual(sum(v == "test" for v in forward.values()), 3)
        self.assertEqual(sum(v == "validation" for v in forward.values()), 3)

    def test_cold_zero_identity_preserves_orientation(self):
        rows = [{"a_cat": "new", "b_cat": "known", "held_out_cat": "new"},
                {"a_cat": "known", "b_cat": "new", "held_out_cat": "new"}]
        matrix = identity_design(rows, ["known"], cold_start=True)
        np.testing.assert_array_equal(matrix.toarray(), [[1., -1.], [1., 1.]])
        np.testing.assert_allclose(np.asarray(matrix @ [0.2, 0.7]).ravel(), [-0.5, 0.9], rtol=0, atol=1e-15)
        with self.assertRaises(ValueError):
            identity_design([{"a_cat": "a", "b_cat": "b", "held_out_cat": "a"}], ["known"], True)

    def test_fit_design_matches_existing_implementation(self):
        rows = [{"a_cat": "a", "b_cat": "b"}, {"a_cat": "b", "b_cat": "b"}]
        actual = identity_design(rows, ["a", "b"], cold_start=False)
        expected = identity_matrix(rows, ["a", "b"], "cat")
        self.assertEqual((actual != expected).nnz, 0)

    def test_actual_split_is_cat_disjoint_and_matches_coverage(self):
        with (HERE / "cat_reservation.csv").open(encoding="utf-8", newline="") as stream:
            reservation = {r["cat_id"]: r["reservation"] for r in csv.DictReader(stream)}
        fit = load_structure("core_fit")
        fitting = {r[f"{side}_cat"] for r in fit for side in ("a", "b")}
        self.assertTrue(all(reservation[cat] == "core" for cat in fitting))
        pair_sets = []
        coverage = json.loads((HERE / "coverage.json").read_text(encoding="utf-8"))
        for part in ("validation", "test"):
            rows = load_structure(part)
            pair_sets.append({r["pair_id"] for r in rows})
            self.assertEqual(len(rows), coverage["g4"][part]["responses"])
            self.assertGreaterEqual(len({r["held_out_cat"] for r in rows}), 20)
            for row in rows:
                heldout = row["held_out_cat"]
                self.assertEqual(reservation[heldout], part)
                self.assertNotIn(heldout, fitting)
                self.assertEqual(sum(row[f"{s}_cat"] == heldout for s in ("a", "b")), 1)
                other = row["b_cat"] if row["a_cat"] == heldout else row["a_cat"]
                self.assertIn(other, fitting)
        self.assertFalse(pair_sets[0] & pair_sets[1])
        self.assertFalse((pair_sets[0] | pair_sets[1]) & {r["pair_id"] for r in fit})


if __name__ == "__main__":
    unittest.main()
