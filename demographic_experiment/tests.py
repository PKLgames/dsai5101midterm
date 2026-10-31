"""Focused all-demographic design, support, solver and fallback tests."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse
from scipy.special import expit

from experiment import DEMOGRAPHICS, category, design, fit_encoding, penalties, side_matrix
from src.development import choose, identity_matrix
from src.estimator import fit_logistic, objective_gradient


def row(value: str = "man", pair: str = "pair") -> dict:
    result = {name: "" for name in DEMOGRAPHICS}
    result.update(gender=value, pair_id=pair)
    return result


class DesignTests(unittest.TestCase):
    def test_fit_support_reference_and_missing(self):
        rows = [row("man", str(i % 50)) for i in range(200)]
        rows += [row("woman", str(i % 40)) for i in range(100)]
        rows += [row("", str(i)) for i in range(10)]
        encoding = fit_encoding(rows)
        self.assertEqual(encoding["gender"]["reference"], "man")
        self.assertEqual(encoding["gender"]["deviations"], ["woman"])
        sides = side_matrix([row("woman"), row(""), row("unseen-observed-gender")], encoding).toarray()
        np.testing.assert_array_equal(sides[:, 0], [1., 0., 0.])

    def test_invalid_demographic_is_explicit_error(self):
        item = row()
        item["pet_care"] = "invalid"
        with self.assertRaises(ValueError):
            category(item, "pet_care")
        item["gender"] = "<missing>"
        with self.assertRaises(ValueError):
            category(item, "gender")

    def test_display_swap_changes_score_but_not_demographic_side(self):
        side = sparse.csr_matrix([[1.]])
        delta = sparse.csr_matrix([[.2, -.3]])
        a, b = design(side, delta, True), design(side, -delta, True)
        theta = np.asarray([.1, .05, .2, -.4, .3, -.2])
        la, lb = float((a @ theta)[0]), float((b @ theta)[0])
        self.assertAlmostEqual(la + lb, 2*(.1+.05))
        self.assertEqual(a.shape[1], len(penalties(1, 2, True, .1, 10)))

    def test_cat_scores_cancel_within_cat_including_interactions(self):
        rows = [{"a_cat": "cat", "b_cat": "cat"}]
        diff = identity_matrix(rows, ["cat"], "cat", side=False)
        x = design(sparse.csr_matrix([[1.]]), diff, True)
        np.testing.assert_array_equal(x.toarray(), [[1., 1., 0., 0.]])

    def test_demographic_only_is_image_independent(self):
        side = sparse.csr_matrix([[1., 0.], [1., 0.]])
        logits = np.asarray(design(side, None, False) @ [.1, .2, .3])
        self.assertEqual(logits[0], logits[1])
        with self.assertRaises(ValueError):
            design(side, None, True)

    def test_gradient_and_acceptance_for_combined_design(self):
        side = sparse.csr_matrix([[0.], [1.], [1.], [0.]])
        diff = sparse.csr_matrix([[1., 2.], [-1., .4], [.3, -.1], [0., 1.]])
        x = design(side, diff, True)
        y = np.asarray([1., 0., 1., 0.])
        p = penalties(1, 2, True, .1, 10.)
        theta = np.arange(x.shape[1], dtype=float)/10
        _, gradient = objective_gradient(theta, x, y, np.ones(4), p)
        numeric = []
        for i in range(len(theta)):
            shift = np.zeros_like(theta)
            shift[i] = 1e-6
            plus = objective_gradient(theta+shift, x, y, np.ones(4), p)[0]
            minus = objective_gradient(theta-shift, x, y, np.ones(4), p)[0]
            numeric.append((plus-minus)/2e-6)
        np.testing.assert_allclose(gradient, numeric, rtol=0, atol=1e-8)
        result = fit_logistic(x, y, p)
        self.assertTrue(result.accepted)
        self.assertLess(result.gradient_max, 1e-5)

    def test_exact_pooled_tie_preference(self):
        candidates = [
            {"accepted": True, "validation_loss": .5, "pooled": False, "ridge": .1, "rho": 100},
            {"accepted": True, "validation_loss": .5000001, "pooled": True, "ridge": .01, "rho": 1e99},
        ]
        self.assertTrue(choose(candidates, contextual=True)["pooled"])

    def test_penalties_separate_side_and_interactions(self):
        p = penalties(2, 3, True, .01, 100)
        np.testing.assert_array_equal(p, [1e-6, .01, .01, .01, .01, .01, 1., 1., 1., 1., 1., 1.])


if __name__ == "__main__":
    unittest.main()
