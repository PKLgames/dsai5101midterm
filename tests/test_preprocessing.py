import unittest

import numpy as np

from src.feature_y import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.features import H_FEATURES
from src.preprocessing import fit_preprocessing, transform_photos


def hrow(image, value):
    return {"image_id": image, "extraction_status": "ok",
            **{name: float(value) for name in H_FEATURES}}


def yrow(image, value, category):
    return {"image_id": image, "extraction_status": "ok",
            **{name: value for name in NUMERIC_FEATURES},
            **{name: category for name in CATEGORICAL_FEATURES}}


class PreprocessingTests(unittest.TestCase):
    def test_handcrafted_fit_only_unique_photos_and_no_leakage(self):
        rows = [hrow("a", 1), hrow("b", 3), hrow("test", 10000)]
        spec = fit_preprocessing("H", rows, {"a", "b"})
        self.assertEqual(spec["fitting_photo_count"], 2)
        self.assertAlmostEqual(spec["numeric"][0]["median"], 2)
        fitting = transform_photos(spec, rows[:2])
        self.assertAlmostEqual(float(np.sqrt(np.mean(np.sum(fitting**2, axis=1)))), 1)
        self.assertGreater(float(transform_photos(spec, rows[2:])[0, 0]), 100)

    def test_numeric_missing_and_fitting_constant_handling(self):
        a, b, c = hrow("a", 1), hrow("b", 3), hrow("c", 2)
        b["brightness"] = None
        spec = fit_preprocessing("H", [a, b, c], {"a", "b", "c"})
        self.assertIn("brightness__missing", spec["final_names"])
        self.assertTrue(np.isfinite(transform_photos(spec, [a, b, c])).all())
        with self.assertRaisesRegex(ValueError, "degenerate"):
            fit_preprocessing("H", [hrow("a", 1), hrow("b", 1)], {"a", "b"})

    def test_y_rare_unseen_and_no_detection_are_explicit(self):
        rows = [yrow(str(i), float(i), "common" if i < 10 else "rare") for i in range(11)]
        spec = fit_preprocessing("Y", rows, {str(i) for i in range(11)})
        first = spec["categorical"][0]
        self.assertIn("rare", first["pooled_categories"])
        self.assertIn("no_detection", first["levels"])
        values = transform_photos(spec, [yrow("new", None, "no_detection"),
                                         yrow("new2", 1, "unseen")])
        self.assertEqual(values.shape[0], 2)
        self.assertTrue(np.isfinite(values).all())

    def test_cnn_center_rms_preserves_coordinate_geometry(self):
        def cnn(name, index):
            return {"image_id": name, "extraction_status": "ok",
                    **{f"cnn_{i:03d}": float(i == index) for i in range(576)}}
        rows = [cnn("a", 0), cnn("b", 1), cnn("c", 2)]
        spec = fit_preprocessing("C", rows, {"a", "b"})
        transformed = transform_photos(spec, rows)
        self.assertEqual(transformed.shape, (3, 576))
        self.assertAlmostEqual(float(np.sqrt(np.mean(np.sum(transformed[:2] ** 2, axis=1)))), 1)
        self.assertAlmostEqual(transformed[2, 2], 1/spec["rms_norm"])


if __name__ == "__main__":
    unittest.main()
