"""Numerical, fitting-only and orientation safeguards for kernel scores."""

import unittest

import numpy as np
from scipy.special import expit

from kernel import designs, fit_map, transform


def features():
    rows = []
    for i in range(7):
        x = np.zeros(576)
        x[i] = 1.
        rows.append({"image_id": f"image{i}", "extraction_status": "ok",
                     **{f"cnn_{j:03d}": float(x[j]) for j in range(576)}})
    return rows


class KernelTests(unittest.TestCase):
    def test_landmarks_and_transforms_use_only_fitting_photos(self):
        data = features()
        fit = {f"image{i}" for i in range(5)}
        mapping = fit_map(data, fit, 1., landmark_limit=3)
        self.assertTrue(set(mapping["landmark_ids"]) <= fit)
        self.assertEqual(set(mapping["fitting_photos"]), fit)
        modified = features()
        modified[-1]["cnn_000"] = 100.
        other = fit_map(modified, fit, 1., landmark_limit=3)
        self.assertEqual(mapping, other)

    def test_orientation_and_scalar_scoring(self):
        data = features()
        mapping = fit_map(data, {f"image{i}" for i in range(5)}, 1., landmark_limit=3)
        rows = [{"a_image": "image0", "b_image": "image5"},
                {"a_image": "image5", "b_image": "image0"}]
        _, x = designs(rows, rows, data, mapping)
        theta = np.array([.1, .2, -.3, .4])
        a, b = np.asarray(x @ theta).ravel()
        self.assertAlmostEqual(a+b, .2)
        self.assertAlmostEqual(expit(a-.1)+expit(b-.1), 1.)

    def test_outcome_blind_features_and_degenerate_failure(self):
        data = features()
        with self.assertRaises(ValueError):
            fit_map([data[0], {**data[0], "image_id": "same"}], {"image0", "same"}, 1.)
        with self.assertRaises(ValueError):
            fit_map(data, {"image0", "missing"}, 1.)

    def test_transform_is_consistent_across_order(self):
        data = features()
        mapping = fit_map(data, {f"image{i}" for i in range(5)}, .25, landmark_limit=4)
        forward = transform(mapping, [data[0], data[5]])
        reverse = transform(mapping, [data[5], data[0]])
        np.testing.assert_allclose(forward, reverse[::-1], atol=1e-14, rtol=0)


if __name__ == "__main__":
    unittest.main()
