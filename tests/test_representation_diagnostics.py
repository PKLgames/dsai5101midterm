import unittest

import numpy as np
from scipy import sparse

from src.representation_diagnostics import _group_name, _refit


class RepresentationDiagnosticTests(unittest.TestCase):
    def test_feature_group_names_include_missing_indicators(self):
        self.assertEqual(_group_name("share_black__missing"), "share_black")
        self.assertEqual(_group_name("breed=domestic_shorthair"), "breed")
        self.assertEqual(_group_name("brightness"), "brightness")

    def test_refit_reports_training_validation_and_finite_norm(self):
        x = sparse.csr_matrix([[1., -1.], [1., 1.], [1., 0.], [1., 2.]])
        y = np.array([0., 1., 0., 1.])
        row = _refit("synthetic", x, x, y, y, np.array([1e-6, .1]), .1, 0)
        self.assertTrue(row["accepted"])
        self.assertAlmostEqual(row["train_loss"], row["validation_loss"])
        self.assertGreaterEqual(row["coefficient_norm"], 0)
        self.assertLess(row["gradient_max"], 1e-5)


if __name__ == "__main__":
    unittest.main()
