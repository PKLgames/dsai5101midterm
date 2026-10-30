import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.abstention_utility import calculate, interval, plot, point_metrics, run, shared_pair_resamples


class AbstentionMetricTests(unittest.TestCase):
    def test_accuracy_coverage_and_original_denominator_utility(self):
        p = [.9, .8, .2, .4]
        y = [1, 0, 0, 1]
        loss = [.1, 1.6, .2, .9]
        metrics, retained, correct, utility = point_metrics(p, y, loss, ["a", "a", "b", "c"], .75)
        np.testing.assert_array_equal(retained, [True, True, True, False])
        self.assertEqual(metrics["retained_responses"], 3)
        self.assertEqual(metrics["retained_pairs"], 2)
        self.assertEqual(metrics["coverage"], .75)
        self.assertAlmostEqual(metrics["selective_accuracy"], 2/3)
        self.assertAlmostEqual(metrics["utility_per_original_response"], .25)
        self.assertAlmostEqual(metrics["utility_per_retained_response"], 1/3)
        self.assertAlmostEqual(metrics["always_predict_utility"], 0.)
        np.testing.assert_array_equal(utility, [1, -1, 1, 0])

    def test_zero_coverage_accuracy_is_undefined_but_total_utility_is_zero(self):
        metrics, _, _, _ = point_metrics([.6, .4], [1, 0], [.2, .3], ["a", "b"], .9)
        self.assertEqual(metrics["coverage"], 0)
        self.assertEqual(metrics["utility_per_original_response"], 0)
        self.assertIsNone(metrics["selective_accuracy"])
        self.assertIsNone(metrics["selective_log_loss"])
        self.assertEqual(metrics["support_status"], "no_retained_cases")

    def test_confidence_ties_are_never_split_and_prediction_tie_chooses_A(self):
        metrics, _, _, _ = point_metrics([.5, .5], [1, 0], [.69, .69], ["a", "b"], .5)
        self.assertEqual(metrics["retained_responses"], 2)
        self.assertEqual(metrics["correct_retained"], 1)
        metrics, _, _, _ = point_metrics([.5, .5], [1, 0], [.69, .69], ["a", "b"], .525)
        self.assertEqual(metrics["retained_responses"], 0)
        metric, _, _, _ = point_metrics([1., 0.], [1, 0], [0., 0.], ["a", "b"], 1.)
        self.assertEqual(metric["retained_responses"], 2)

    def test_invalid_predictions_and_dimensions_fail_explicitly(self):
        for p in ([float("nan")], [1.01]):
            with self.assertRaises(ValueError):
                point_metrics(p, [1], [.2], ["a"], .5)
        with self.assertRaises(ValueError):
            point_metrics([.6], [1], [.2], [], .5)
        with self.assertRaises(ValueError):
            interval(np.array([1., float("nan")]))


class BootstrapTests(unittest.TestCase):
    def test_resamples_preserve_pair_multiplicity_and_are_shared_deterministically(self):
        codes, weights = shared_pair_resamples(["a", "a", "b"], 3, 10)
        np.testing.assert_array_equal(codes, [0, 0, 1])
        rng = np.random.default_rng(3)
        for row in weights:
            expected = np.bincount(rng.integers(0, 2, size=2), minlength=2)
            np.testing.assert_array_equal(row, expected)
        np.testing.assert_array_equal(weights, shared_pair_resamples(["a", "a", "b"], 3, 10)[1])

    def test_all_abstain_has_zero_utility_intervals_and_explicit_empty_samples(self):
        rows = [
            {"train_row_id": str(i), "pair_id": f"p{i//2}", "outcome": i % 2,
             "probability": .6, "loss": .7} for i in range(10)
        ]
        result = calculate({"M0": rows, "M1": rows}, [.5, .75, 1.], 42, 100)
        self.assertEqual(len(result), 6)
        empty = next(row for row in result if row["model"] == "M0" and row["threshold"] == .75)
        self.assertEqual(empty["utility_ci95_low"], 0)
        self.assertEqual(empty["utility_ci95_high"], 0)
        self.assertEqual(empty["undefined_selective_accuracy_bootstrap_replicates"], 100)
        self.assertIsNone(empty["selective_accuracy_ci95_low"])
        always = [row for row in result if row["threshold"] == .5]
        self.assertEqual(always[0]["utility_ci95_low"], always[1]["utility_ci95_low"])
        self.assertEqual(always[0]["utility_difference_ci95_low"], 0)
        self.assertFalse(any(row["deployment_threshold_selected"] for row in result))

    def test_misaligned_model_rows_cannot_be_compared(self):
        a = [{"train_row_id": "a", "pair_id": "p", "outcome": 1, "probability": .6, "loss": .5}]
        b = [{"train_row_id": "b", "pair_id": "p", "outcome": 1, "probability": .6, "loss": .5}]
        with self.assertRaisesRegex(ValueError, "identical"):
            calculate({"M0": a, "M1": b}, [.5, 1.], 42, 10)


class GateTests(unittest.TestCase):
    def test_existing_outputs_are_not_overwritten_or_raw_outcomes_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "figures").mkdir()
            path = root / "figures/presentation_abstention_utility.pdf"
            path.write_bytes(b"existing")
            with patch("src.abstention_utility.verify_sources") as verify:
                with self.assertRaises(FileExistsError):
                    run(root)
            verify.assert_not_called()
            self.assertEqual(path.read_bytes(), b"existing")

    def test_presentation_pages_render_and_contain_exploratory_warnings(self):
        from pypdf import PdfReader
        rows = [
            {"train_row_id": str(i), "pair_id": f"p{i//4}", "outcome": i % 2,
             "probability": float(p), "loss": .7}
            for i, p in enumerate(np.linspace(.05, .95, 120))
        ]
        models = {name: rows for name in ("M0", "M1", "M2", "M3*", "M4*")}
        records = calculate(models, [i/40 for i in range(20, 41)], 42, 100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plot.pdf"
            checks = plot(path, records)
            self.assertEqual(len(checks), 4)
            self.assertTrue(all(not page["outside_page_text"] for page in checks))
            pdf = PdfReader(path)
            self.assertEqual(len(pdf.pages), 4)
            for page in pdf.pages:
                self.assertIn("Not Threshold Selection", page.extract_text())
            text = "\n".join(page.extract_text() for page in pdf.pages)
            self.assertIn("always predict is weakly optimal", text)
            self.assertIn("No optimum is picked", text)


if __name__ == "__main__":
    unittest.main()
