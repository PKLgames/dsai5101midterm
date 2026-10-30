import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq

from src.evaluate import evaluate, interval, load_outcomes, pair_bootstrap, subgroup_labels
from src.freeze import validate_freeze


class BootstrapTests(unittest.TestCase):
    def test_pair_multiplicity_and_response_denominator(self):
        pair_ids = ["a", "a", "b"]
        losses = {"x": np.array([0., 2., 4.]), "y": np.array([1., 1., 3.])}
        response, pair = pair_bootstrap(pair_ids, losses, seed=3, replicates=100)
        rng = np.random.default_rng(3)
        for i in range(100):
            sample = rng.integers(0, 2, size=2)
            multiplicity = np.bincount(sample, minlength=2)
            expected_response = (2 * multiplicity[0] + 4 * multiplicity[1]) / (
                2 * multiplicity[0] + multiplicity[1]
            )
            expected_pair = (multiplicity[0] + 4 * multiplicity[1]) / 2
            self.assertAlmostEqual(response["x"][i], expected_response)
            self.assertAlmostEqual(pair["x"][i], expected_pair)
        self.assertTrue(np.allclose(response["x"] - response["y"],
                                    response["x"] - pair_bootstrap(pair_ids, {"y": losses["y"]}, 3, 100)[0]["y"]))

    def test_interval_levels_and_bad_inputs(self):
        x = np.arange(1000, dtype=float)
        a = interval(x, .95)
        b = interval(x, .9875)
        self.assertLess(b[0], a[0])
        self.assertGreater(b[1], a[1])
        with self.assertRaises(ValueError):
            pair_bootstrap(["a"], {"bad": np.array([1., 2.])}, 1, 10)

    def test_unsupported_opponent_count_is_not_mislabeled(self):
        row = {"min_endpoint_opponents": -1, "comparison_type": "between_cats", "pet_care": "missing",
               "any_no_detection": False, "any_dog_fallback": False,
               "any_multiple_cats": False, "both_cat_detected": False}
        self.assertIn(("minimum_endpoint_opponents", "not_available"), subgroup_labels(row))


class GateTests(unittest.TestCase):
    def test_no_freeze_prevents_any_test_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("src.evaluate.load_test_structure") as structure, patch("src.evaluate.load_outcomes") as outcomes:
                with self.assertRaises(FileNotFoundError):
                    evaluate(root)
            structure.assert_not_called()
            outcomes.assert_not_called()

    def test_changed_freeze_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifacts").mkdir()
            (root / "artifacts/model_freeze.json").write_text("{}", encoding="utf-8")
            (root / "artifacts/model_freeze.sha256").write_text("wrong", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "seal"):
                validate_freeze(root)

    def test_outcome_join_requires_exact_ids_and_unchanged_raw_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            path = root / "dataset/train.csv"
            path.write_text("train_row_id,choice\nTR1,A\nTR2,B\n", encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(load_outcomes(root, [{"train_row_id": "TR2"}], digest), {"TR2": 0})
            with self.assertRaises(ValueError):
                load_outcomes(root, [{"train_row_id": "missing"}], digest)
            with self.assertRaises(ValueError):
                load_outcomes(root, [{"train_row_id": "TR2"}], "wrong")


class EvaluationIntegrationTests(unittest.TestCase):
    def test_frozen_matched_evaluation_reconciles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{
                "train_row_id": f"TR{i}", "pair_id": f"p{i//4}", "a_image": "a",
                "b_image": "b", "comparison_type": "between_cats",
                "pet_care": "never", "full_strict_g2": True, "common_strict_g2": True,
            } for i in range(120)]
            models = {
                ("full", "full:M0"), ("full", "full:M1"), ("full", "full:M2"),
                ("full", "full:M2-no-position"), ("common", "common:M0"), ("common", "common:M1"),
                ("common", "common:M2"), ("common", "common:M2-no-position"),
                *{("common", f"M{k}-{b}") for k in (3, 4) for b in ("H", "C", "Y")},
                ("common", "M3-side"), ("common", "M4-side"),
                ("common", "M2-temperature"), ("common", "M3*-temperature"),
                ("common", "M4*-temperature"),
            }
            predictions = {(population, model): (rows, np.full(len(rows), .1))
                           for population, model in models}
            frozen = {
                "raw_training_sha256": "synthetic",
                "bootstrap": {"seed": 20261101, "replicates": 5000},
                "selected_representation": "C",
                "primary_contrasts": [
                    {"label": "cat_identity", "reference": "common:M0", "candidate": "common:M1"},
                    {"label": "photo_detail", "reference": "common:M1", "candidate": "common:M2"},
                    {"label": "visual_sharing", "reference": "common:M2", "candidate": "M3-C"},
                    {"label": "recorded_context", "reference": "M3-C", "candidate": "M4-C"},
                ],
                "limitations": ["synthetic"],
            }
            (root / "artifacts").mkdir()
            (root / "artifacts/model_freeze.json").write_text("synthetic", encoding="utf-8")
            sources = {model for _, model in models if not model.endswith("-temperature")}
            metadata = {
                name: {"train_loss": .7, "objective": .7, "validation_loss": .7,
                       "ridge": .01, "rho": 0, "gradient_max": 0, "coefficient_count": 1, "seconds": .1}
                for name in sources
            }
            (root / "artifacts/model_selection.json").write_text(json.dumps({"selected": metadata}), encoding="utf-8")
            (root / "tables").mkdir()
            with (root / "tables/model_resources.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["model", "stored_trainable_coefficients",
                    "retained_feature_dimensions", "mean_inference_seconds_3_runs", "inference_scope"])
                writer.writeheader()
                for source in sources:
                    writer.writerow({"model": source, "stored_trainable_coefficients": 1,
                                     "retained_feature_dimensions": 0,
                                     "mean_inference_seconds_3_runs": .01, "inference_scope": "synthetic"})
            y = {row["train_row_id"]: i % 2 for i, row in enumerate(rows)}
            opponents = {"full": {"a": 5, "b": 5}, "common": {"a": 5, "b": 5}}
            quality = {x: {"extraction_status": "ok", "detection_status": "cat_detected",
                           "candidate_cat_count": 1} for x in ("a", "b")}
            with patch("src.evaluate.validate_freeze", return_value=frozen), \
                 patch("src.evaluate.load_test_structure", return_value=rows), \
                 patch("src.evaluate.test_logits", return_value=predictions), \
                 patch("src.evaluate._record_test_access"), \
                 patch("src.evaluate.load_outcomes", return_value=y), \
                 patch("src.evaluate._feature_flags", return_value=(opponents, quality)):
                result = evaluate(root)
            self.assertEqual(result["models"], 19)
            table = pq.read_table(root / "artifacts/test_predictions.parquet").to_pylist()
            self.assertEqual(len(table), 19*120)
            with (root / "tables/primary_contrasts.csv").open(encoding="utf-8", newline="") as stream:
                contrasts = list(csv.DictReader(stream))
            self.assertEqual(len(contrasts), 4)
            self.assertTrue(all(float(row["response_loss_difference"]) == 0 for row in contrasts))
            self.assertTrue(all(float(row["interval_level"]) == .9875 for row in contrasts))
            self.assertEqual((root / "figures/calibration.pdf").read_bytes()[:4], b"%PDF")


if __name__ == "__main__":
    unittest.main()
