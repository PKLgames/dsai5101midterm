"""Focused tests for official prediction routing and output rules."""

from __future__ import annotations

import unittest

import numpy as np

from common import classify_route, probabilities_to_choices
from run import components
from src.development import identity_matrix


class OfficialPredictionTests(unittest.TestCase):
    def setUp(self):
        self.photos = {"catA/a.jpg": 0, "catB/b.jpg": 0, "catC/c.jpg": 1}
        self.cats = {"catA": 0, "catB": 0, "catC": 1}
        self.usable = {"catA/a.jpg", "catB/b.jpg", "catC/c.jpg", "new/n.jpg"}

    def test_supported_known_photos_use_selected_identity_model(self):
        row = {"a_image": "catA/a.jpg", "b_image": "catB/b.jpg", "a_cat": "catA", "b_cat": "catB"}
        self.assertEqual(classify_route(row, self.photos, self.cats, self.usable)[0], "known_photo_m1")

    def test_new_photo_and_unseen_cat_use_shared_visual_model(self):
        row = {"a_image": "new/n.jpg", "b_image": "catA/a.jpg", "a_cat": "new", "b_cat": "catA"}
        self.assertEqual(classify_route(row, self.photos, self.cats, self.usable)[0], "shared_visual")

    def test_missing_visual_falls_back_to_known_cat_model(self):
        row = {"a_image": "catA/new.jpg", "b_image": "catB/b.jpg", "a_cat": "catA", "b_cat": "catB"}
        self.assertEqual(classify_route(row, self.photos, self.cats, self.usable)[0], "cat_fallback_m1")

    def test_missing_visual_and_unknown_identity_use_constant(self):
        row = {"a_image": "unknown/missing.jpg", "b_image": "catB/b.jpg", "a_cat": "unknown", "b_cat": "catB"}
        self.assertEqual(classify_route(row, self.photos, self.cats, self.usable)[0], "constant_fallback_m0")

    def test_disconnected_components_are_not_identity_supported(self):
        row = {"a_image": "catA/a.jpg", "b_image": "catC/c.jpg", "a_cat": "catA", "b_cat": "catC"}
        self.assertEqual(classify_route(row, self.photos, self.cats, self.usable)[0], "shared_visual")

    def test_choice_boundary_and_invalid_probabilities(self):
        self.assertEqual(probabilities_to_choices(np.asarray([0., .4999, .5, 1.])), ["B", "B", "A", "A"])
        for invalid in (np.asarray([np.nan]), np.asarray([-0.01]), np.asarray([1.01]), np.ones((1, 1))):
            with self.assertRaises(ValueError):
                probabilities_to_choices(invalid)

    def test_identity_design_respects_displayed_orientation(self):
        rows = [{"a_cat": "catA", "b_cat": "catB"}, {"a_cat": "catB", "b_cat": "catA"}]
        logits = np.asarray(identity_matrix(rows, ["catA", "catB"], "cat") @ [.1, .6, -.2]).reshape(-1)
        np.testing.assert_allclose(logits, [.9, -.7], rtol=0, atol=1e-14)

    def test_components_use_labeled_comparison_graph(self):
        rows = [{"a_image": "a", "b_image": "b"}, {"a_image": "b", "b_image": "c"},
                {"a_image": "d", "b_image": "e"}]
        result = components(rows, "image")
        self.assertEqual(result["a"], result["c"])
        self.assertNotEqual(result["a"], result["d"])


if __name__ == "__main__":
    unittest.main()
