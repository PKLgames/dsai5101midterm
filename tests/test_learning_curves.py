import unittest
from unittest.mock import patch

from src.diagnostics import DevelopmentData
from src.learning_curves import calculate_curves, common_validation_indices, nested_pair_sets


class LearningCurveTests(unittest.TestCase):
    def test_nested_deterministic_pair_subsets(self):
        fit = [{"pair_id": f"p{i}", "a_image": f"i{i}", "b_image": f"i{i+1}"}
               for i in range(20)]
        sizes = [.25, .5, .75, 1.]
        one = nested_pair_sets(fit, 42, sizes)
        self.assertEqual(one, nested_pair_sets(fit, 42, sizes))
        self.assertEqual([len(one[x]) for x in sizes], [5, 10, 15, 20])
        self.assertTrue(one[.25] <= one[.5] <= one[.75] <= one[1.])

    def test_common_validation_intersects_all_subsets_and_components(self):
        fit = [{"pair_id": "p1", "a_image": "a", "b_image": "b"},
               {"pair_id": "p2", "a_image": "b", "b_image": "c"},
               {"pair_id": "p3", "a_image": "d", "b_image": "e"}]
        val = [{"a_image": "a", "b_image": "c"}, {"a_image": "a", "b_image": "d"},
               {"a_image": "a", "b_image": "unknown"}]
        self.assertEqual(common_validation_indices(fit, val, [1, 2], [1.]), [0])
        indices = common_validation_indices(fit, val, [1, 2], [.34, 1.])
        self.assertTrue(set(indices) <= {0})

    def test_synthetic_curve_refits_and_marks_unsupported_context(self):
        def row(i, partition):
            return {"train_row_id": str(i), "pair_id": f"p{i % 8}", "partition": partition,
                    "a_image": "a", "b_image": "b", "pet_care": "never", "y_displayed": i % 2}
        features = [{"image_id": image, "extraction_status": "ok",
                     **{f"cnn_{i:03d}": float(i == dimension) for i in range(576)}}
                    for image, dimension in (("a", 0), ("b", 1))]
        data = DevelopmentData(
            fit=[row(i, "fit") for i in range(16)],
            val=[row(i, "validation") for i in range(100, 104)],
            features={"C": features}, selection={"selected_representation": "C"},
            protocol={"learning_curve_sizes": [.25, .5, .75, 1.],
                      "learning_curve_fallback_sizes": [.5, .75, 1.],
                      "learning_curve_seeds": [1],
                      "learning_curve_validation_minimum": {"responses": 1, "distinct_pairs": 1}},
        )
        config = {"candidate_registry": {"base_ridge_grid": [0.01, 0.1],
                                         "context_penalty_multipliers": [1, 10, 100]}}
        with patch("builtins.print"):
            result, attempts, status = calculate_curves(data, config)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(len(result), 12)
        self.assertEqual(len(attempts), 16)
        self.assertTrue(all(x["status"] == "unsupported" for x in result if x["model"] == "M4*"))
        self.assertTrue(all(x["accepted"] for x in attempts))


if __name__ == "__main__":
    unittest.main()
