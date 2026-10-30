import unittest

import numpy as np
from scipy.special import expit

from src.estimator import fit_logistic
from src.simulations import expected_loss, generate_simulations


class RecoveryTests(unittest.TestCase):
    def test_known_probability_recovery(self):
        x = np.array([[1., -1., 0.], [0., 1., -1.], [1., 0., -1.]])
        truth = np.array([np.log(2.), 0., -np.log(2.)])
        probabilities = expit(x @ truth)
        np.testing.assert_allclose(probabilities, [2/3, 2/3, .8])
        fit = fit_logistic(x, probabilities, np.full(3, 1e-6))
        self.assertTrue(fit.accepted)
        np.testing.assert_allclose(expit(x @ fit.coefficients), probabilities, atol=1e-4)
        self.assertAlmostEqual(expected_loss(probabilities, x @ truth), .5911, places=4)

    def test_group_and_cycle_oracle_risks(self):
        q = np.array([2/3, 2/3, .8])
        grouped = np.r_[q, 1-q]
        self.assertAlmostEqual(expected_loss(grouped, np.zeros(6)), np.log(2))
        self.assertAlmostEqual(expected_loss(grouped, np.log(grouped/(1-grouped))), .5911, places=4)
        cycle = np.full(3, .8)
        self.assertAlmostEqual(expected_loss(cycle, np.full(3, np.log(4))), .5004, places=4)
        self.assertAlmostEqual(expected_loss(cycle, np.zeros(3)), np.log(2))

    def test_simulations_are_seeded_and_record_success(self):
        a = generate_simulations(20261029, repetitions=2)
        b = generate_simulations(20261029, repetitions=2)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 24)
        self.assertTrue(all(row["accepted"] and row["gradient_max"] < 1e-5 for row in a))
        self.assertEqual({row["simulation"] for row in a}, {"A", "B", "C"})


if __name__ == "__main__":
    unittest.main()
