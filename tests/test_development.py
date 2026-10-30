import unittest
import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.development import (
    choose, choose_representation, feature_differences, identity_matrix, run_development, score, visual_design,
)
from src.feature_y import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.features import H_FEATURES


class DevelopmentTests(unittest.TestCase):
    def test_identity_design_and_unknown_guard(self):
        rows = [{"a_cat": "a", "b_cat": "a"}, {"a_cat": "a", "b_cat": "b"}]
        x = identity_matrix(rows, ["a", "b"], "cat")
        np.testing.assert_array_equal(x.toarray(), [[1, 0, 0], [1, 1, -1]])
        with self.assertRaises(ValueError):
            identity_matrix([{"a_cat": "a", "b_cat": "unknown"}], ["a", "b"], "cat")

    def test_image_key_join_and_context_fallback(self):
        rows = [{"a_image": "b", "b_image": "a"}, {"a_image": "a", "b_image": "b"}]
        diff = feature_differences(rows, ["a", "b"], np.array([[1., 2.], [3., 5.]]))
        np.testing.assert_array_equal(diff.toarray(), [[2, 3], [-2, -3]])
        x = visual_design(diff, ["other", "unseen"], ("other",), ("other",)).toarray()
        np.testing.assert_array_equal(x[0], [1, 1, 2, 3, 2, 3])
        np.testing.assert_array_equal(x[1], [1, 0, -2, -3, 0, 0])

    def test_tie_break_and_pooled_priority(self):
        candidates = [
            {"accepted": True, "validation_loss": .5, "ridge": .01, "rho": 10, "pooled": False},
            {"accepted": True, "validation_loss": .5000005, "ridge": .1, "rho": 1, "pooled": False},
        ]
        self.assertEqual(choose(candidates)["ridge"], .1)
        candidates.append({"accepted": True, "validation_loss": .5000008, "ridge": .01,
                           "rho": float("inf"), "pooled": True})
        self.assertTrue(choose(candidates, contextual=True)["pooled"])
        self.assertEqual(choose_representation({"C": candidates[0], "H": candidates[1]}), "H")

    def test_metrics_and_invalid_predictions(self):
        loss, brier, accuracy = score(np.array([1., 0.]), np.array([0., 0.]))
        self.assertAlmostEqual(loss, np.log(2))
        self.assertAlmostEqual(brier, .25)
        self.assertEqual(accuracy, .5)
        with self.assertRaises(ValueError):
            score(np.array([1.]), np.array([float("nan")]))

    def test_end_to_end_synthetic_development(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifacts").mkdir()
            (root / "features").mkdir()
            for path in ("artifacts/rows_canonical.parquet", "artifacts/g2_eligibility.parquet",
                         "artifacts/graph_provenance.json", "artifacts/stage6_provenance.json",
                         "artifacts/simulation_provenance.json", "src/estimator.py",
                         "src/preprocessing.py", "src/development.py", "decisions.yaml"):
                target = root.joinpath(*path.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"synthetic")
            fit, val = [], []
            for i in range(220):
                group = "never" if i < 100 else "occasionally"
                row = {
                    "train_row_id": str(i), "partition": "fit" if i < 200 else "validation",
                    "pair_id": f"pair-{i % 40}", "a_image": f"img-{i % 40}",
                    "b_image": f"img-{(i+1) % 40}", "a_cat": f"cat-{i % 8}",
                    "b_cat": f"cat-{(i+1) % 8}", "comparison_type": "between_cats",
                    "pet_care": group, "y_displayed": int(rng.integers(2)),
                    "full_strict_g2": i >= 200, "common_strict_g2": i >= 200,
                    "common_visual_fit": i < 200,
                }
                (fit if i < 200 else val).append(row)
            adequate = {"decision": "accept_common_population_for_registered_comparison",
                        "fitting_responses": 200, "strict_validation_responses": 20,
                        "strict_validation_pairs": 20, "feature_blocks": ["H", "C", "Y"]}
            (root / "artifacts/stage7_adequacy.json").write_text(json.dumps(adequate), encoding="utf-8")
            (root / "artifacts/common_population_report.json").write_text(
                json.dumps({"admitted_blocks": ["H", "C", "Y"]}), encoding="utf-8"
            )
            for block in ("H", "C", "Y"):
                records = []
                for i in range(40):
                    base = {"image_id": f"img-{i}", "extraction_status": "ok"}
                    if block == "H":
                        base.update({name: float(i + j) for j, name in enumerate(H_FEATURES)})
                    elif block == "C":
                        base.update({f"cnn_{j:03d}": float(rng.normal()) for j in range(576)})
                    else:
                        base.update({name: float(i + j) for j, name in enumerate(NUMERIC_FEATURES)})
                        base.update({name: "yes" if i % 2 else "no" for name in CATEGORICAL_FEATURES})
                    records.append(base)
                pq.write_table(pa.Table.from_pylist(records), root / "features" / f"{block}.parquet")
            config = {"candidate_registry": {"base_ridge_grid": [1e-6, 1e-5, 1e-4, .001, .01, .1, 1],
                                              "context_penalty_multipliers": [1, 10, 100]}}
            with patch("src.development.verify_inputs"), patch("src.development.load_development", return_value=(fit, val)), \
                 patch("src.development.subprocess.run") as gate, patch("builtins.print"):
                gate.return_value.returncode = 0
                result = run_development(root, config)
            self.assertIn(result["selected_representation"], ["H", "C", "Y"])
            with (root / "tables/all_fit_attempts.csv").open(encoding="utf-8", newline="") as stream:
                attempts = list(csv.DictReader(stream))
            self.assertEqual(len(attempts), result["candidate_count"])
            self.assertTrue(any(row["model"] == "common:M4-C" for row in attempts))
            selection = json.loads((root / "artifacts/model_selection.json").read_text(encoding="utf-8"))
            self.assertFalse(selection["internal_test_outcomes_read"])
            self.assertEqual(selection["validation_responses"], 20)
            self.assertTrue((root / "tables/validation_predictions.csv").is_file())


if __name__ == "__main__":
    unittest.main()
