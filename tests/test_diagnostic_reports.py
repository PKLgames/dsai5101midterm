import unittest

import numpy as np

from src.diagnostic_reports import confidence_and_temperature
from src.diagnostics import DevelopmentData


class DiagnosticReportTests(unittest.TestCase):
    def test_temperature_and_worst_loss_are_validation_only_and_ties_are_stable(self):
        rows = [
            {"train_row_id": "b", "pair_id": "p1", "comparison_type": "between_cats",
             "pet_care": "", "y_displayed": 1},
            {"train_row_id": "a", "pair_id": "p2", "comparison_type": "within_cat",
             "pet_care": "never", "y_displayed": 0},
        ]
        data = DevelopmentData(fit=[], val=rows, features={}, selection={"selected_representation": "C"},
                               protocol={"optional_diagnostics": {"temperature_grid": [.5, 1, 2]}})
        predictions = {model: np.zeros(2) for model in ("common:M2", "M3-C", "M4-C")}
        audit, summaries, temps = confidence_and_temperature(
            data, predictions, {"worst_loss_audit": {"count_per_model": 20}}
        )
        self.assertEqual(len(audit), 6)
        self.assertEqual([x["train_row_id"] for x in audit if x["model"] == "M2"], ["a", "b"])
        self.assertEqual(len(summaries), 3)
        self.assertEqual([x["temperature"] for x in temps if x["selected"]], [1, 1, 1])
        self.assertTrue(all(x["partition"] == "common_strict_validation" for x in audit))


if __name__ == "__main__":
    unittest.main()
