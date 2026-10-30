import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.optimize import approx_fprime
from scipy.special import expit

from src.canonicalize import feature_difference
from src.estimator import (
    context_mapping, contextual_design, contextual_penalties, fit_logistic, objective_gradient, pooled_fallback,
)


class ObjectiveTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_differences(self):
        x = np.array([[1., 2., -1.], [1., -3., 2.], [1., .3, .7], [1., 0., -2.]])
        y = np.array([1., 0., 1., 0.])
        weights = np.array([1., 2., 3., 1.])
        penalties = np.array([1e-6, .1, 1.])
        theta = np.array([.2, -.4, .8])
        for matrix in (x, sparse.csr_matrix(x)):
            _, grad = objective_gradient(theta, matrix, y, weights, penalties)
            numerical = approx_fprime(theta, lambda t: objective_gradient(t, matrix, y, weights, penalties)[0], 1e-7)
            np.testing.assert_allclose(grad, numerical, atol=1e-6, rtol=1e-6)

    def test_count_aggregation_equals_rows_when_predictors_match(self):
        x = np.array([[1., 2.], [1., 2.], [1., 2.], [1., -1.], [1., -1.]])
        y = np.array([1., 0., 1., 0., 0.])
        theta, penalties = np.array([.4, -.2]), np.array([1e-6, .1])
        full = objective_gradient(theta, x, y, np.ones(5), penalties)
        grouped = objective_gradient(theta, x[[0, 3]], np.array([2/3, 0]), np.array([3, 2]), penalties)
        self.assertAlmostEqual(full[0], grouped[0], places=12)
        np.testing.assert_allclose(full[1], grouped[1], atol=1e-12)
        a = fit_logistic(x, y, penalties)
        b = fit_logistic(x[[0, 3]], np.array([2/3, 0]), penalties, np.array([3, 2]))
        self.assertTrue(a.accepted and b.accepted)
        np.testing.assert_allclose(a.coefficients, b.coefficients, atol=1e-8)

    def test_context_fallback_is_exact_and_unseen_category_is_deterministic(self):
        x = np.array([[1., 2.], [-1., 3.], [3., 0.]])
        pooled = contextual_design(x, ["reference", "other", "unseen"], ())
        expanded = contextual_design(x, ["reference", "other", "unseen"], ("other",))
        coefficients = np.array([.4, .1, -.2])
        pooled_predictions = expit(pooled @ coefficients)
        np.testing.assert_array_equal(pooled_predictions, pooled_fallback(pooled_predictions))
        np.testing.assert_allclose(pooled @ coefficients, expanded @ np.r_[coefficients, [0., 0.]], atol=1e-15)
        np.testing.assert_array_equal(expanded[2, 3:], np.zeros(2))
        np.testing.assert_array_equal(contextual_penalties(2, 1, .01, 10), [1e-6, .01, .01, .1, .1])

    def test_context_support_and_tie_break(self):
        groups = ["a"] * 100 + ["b"] * 100 + ["sparse"] * 99
        ids = [str(i % 30) for i in range(len(groups))]
        self.assertEqual(context_mapping(groups, ids), ("a", ("b",)))

    def test_feature_join_is_by_key_and_not_row_order(self):
        features = {"a": [1., 4.], "b": [3., 7.]}
        self.assertEqual(feature_difference("a", "b", features),
                         feature_difference("a", "b", dict(reversed(list(features.items())))))

    def test_invalid_inputs_and_extreme_logits(self):
        loss, grad = objective_gradient([1000.], np.array([[1.], [-1.]]),
                                        np.array([0., 1.]), np.ones(2), [0.])
        self.assertTrue(np.isfinite(loss) and np.isfinite(grad).all())
        with self.assertRaises(ValueError):
            objective_gradient([0.], np.array([[1.]]), [2.], [1.], [0.])

    def test_failed_fit_retries_from_zero_and_stays_failed(self):
        failed = SimpleNamespace(x=np.array([np.nan]), success=False, status=2,
                                 message="failed", nit=1)
        with patch("src.estimator.minimize", return_value=failed) as solver:
            result = fit_logistic(np.array([[1.]]), [1.], [0.1])
        self.assertFalse(result.accepted)
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(solver.call_count, 2)
        for call in solver.call_args_list:
            np.testing.assert_array_equal(call.args[1], [0.])
        self.assertEqual(result.attempts[1]["maxiter"], 10000)


if __name__ == "__main__":
    unittest.main()
