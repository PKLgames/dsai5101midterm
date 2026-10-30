import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.feature_y import (
    NUMERIC_FEATURES, _apply_legacy_postprocessing, coat_for_detection,
    decode_mask, encode_mask, resolve_device,
)


class YFallbackTests(unittest.TestCase):
    def test_mask_storage_round_trip_and_dimension_validation(self):
        mask = np.array([[True, False, True], [False, True, False]], dtype=bool)
        encoded = encode_mask(mask)
        np.testing.assert_array_equal(decode_mask(encoded, 2, 3), mask)
        with self.assertRaises(ValueError):
            decode_mask(encoded, 8, 8)

    def test_cuda_request_never_silently_falls_back_to_cpu(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            resolve_device(torch, "cuda")
        self.assertEqual(resolve_device(torch, "cpu"), "cpu")
        torch.cuda.is_available = lambda: True
        self.assertEqual(resolve_device(torch, "cuda"), "cuda:0")

    def test_no_detection_does_not_compute_coat_from_background(self):
        legacy = SimpleNamespace(coat_features=Mock())
        values = coat_for_detection(object(), None, legacy)
        legacy.coat_features.assert_not_called()
        self.assertTrue(all(values[name] is None for name in NUMERIC_FEATURES
                            if name != "bbox_area_frac"))
        self.assertEqual(values["colour_primary"], "no_detection")
        self.assertEqual(values["colour_pattern"], "no_detection")

    def test_detected_mask_uses_existing_coat_implementation(self):
        legacy = SimpleNamespace(coat_features=Mock(return_value={"share_black": 0.4}))
        mask, image = object(), object()
        self.assertEqual(coat_for_detection(image, {"mask": mask}, legacy), {"share_black": 0.4})
        legacy.coat_features.assert_called_once_with(image, mask)

    def test_no_detection_breed_rule_does_not_use_missing_coat(self):
        cfg = SimpleNamespace(
            BREED_MARGIN=0.42, FACE_VIEWS=set(), FACE_FRAMING=set(),
            OPEN_EYE_SINGLE=140, OPEN_EYE_PAIR=100, OPEN_EYE_SCORE=0.85,
            EYE_COLOUR_MIN=0.35, EYE_COLOUR_BLOB=0.60, AWAY_VIEWS=set(),
            CLOSED_TO_OPEN_SCORE=0.7,
        )
        legacy = SimpleNamespace(
            reconcile_breed=Mock(), apply_framing=Mock(),
            crop_box=Mock(return_value=None), open_eye_score=Mock(return_value=(0, 0)),
            eye_colour_from_mask=Mock(return_value=""),
        )
        record = {
            "cat_detected": False, "breed": "siamese", "breed_second": "domestic_longhair",
            "breed_margin": 0.1, "view": "front", "framing": "medium",
            "eyes": "both_open", "eyes_score": 0.8, "eyes_colour": "",
            "eyes_colour_clip_score": 0.0, "eyes_colour_clip": "",
        }
        _apply_legacy_postprocessing(record, None, None, cfg, legacy)
        self.assertEqual(record["breed"], "domestic_longhair")
        legacy.reconcile_breed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
