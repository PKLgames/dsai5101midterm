import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from src.diagnostics import DevelopmentData, fair_coin_control


class DiagnosticTests(unittest.TestCase):
    def test_control_is_reproducible_and_pooled_copy_is_exact(self):
        x = sparse.csr_matrix(np.column_stack([np.ones(100), np.tile([-1., 1.], 50)]))
        vx = x[:40]
        data = DevelopmentData(
            fit=[{}] * 100, val=[{}] * 40, features={},
            selection={"selected": {"M4*": {"pooled": True}}},
            protocol={"synthetic_label_control": {"seed": 20261102, "repetitions": 3}},
        )
        designs = {name: (x, vx, np.array([1e-6, .01])) for name in ("M2", "M3*", "M4*")}
        with patch("builtins.print"):
            first = fair_coin_control(data, designs)
            second = fair_coin_control(data, designs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 9)
        for rep in range(3):
            entries = {row["model"]: row for row in first if row["repetition"] == rep}
            self.assertTrue(all(entry["accepted"] for entry in entries.values()))
            self.assertEqual(entries["M3*"]["log_loss"], entries["M4*"]["log_loss"])
            self.assertAlmostEqual(entries["M2"]["baseline_loss"], np.log(2))


if __name__ == "__main__":
    unittest.main()
