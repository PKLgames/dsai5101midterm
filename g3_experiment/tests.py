"""Focused tests for quarantined G3 allocation and evaluation helpers."""

from __future__ import annotations

import csv
import json
import unittest

import numpy as np

from common import EXPERIMENT, cluster_bootstrap, load_structure
from src.development import choose


class G3Tests(unittest.TestCase):
    def test_clusters_preserve_response_multiplicity_and_share_resamples(self):
        clusters = ["a", "a", "b"]
        losses = {"x": np.array([1., 3., 8.]), "y": np.array([2., 4., 9.])}
        result = cluster_bootstrap(clusters, losses, 100, 4)
        np.testing.assert_allclose(result["y"] - result["x"], 1.)
        self.assertTrue(set(np.round(result["x"], 8)) <= {2., 4., 8.})

    def test_clusters_reject_missing_and_nonfinite_values(self):
        with self.assertRaises(ValueError):
            cluster_bootstrap([""], {"x": np.array([1.])}, 5, 1)
        with self.assertRaises(ValueError):
            cluster_bootstrap(["a"], {"x": np.array([np.nan])}, 5, 1)

    def test_pooled_candidate_wins_ties(self):
        rows = [{"accepted": True, "validation_loss": .5, "pooled": False, "ridge": 1., "rho": 100.},
                {"accepted": True, "validation_loss": .5000001, "pooled": True, "ridge": .1, "rho": 1e99}]
        self.assertTrue(choose(rows, contextual=True)["pooled"])

    def test_actual_reservation_disjoint_and_known_cat(self):
        with (EXPERIMENT / "photo_reservation.csv").open(encoding="utf-8", newline="") as stream:
            photo = {r["image_id"]: r for r in csv.DictReader(stream)}
        by_hash = {}
        for row in photo.values():
            by_hash.setdefault(row["content_sha256"], set()).add(row["reservation"])
        self.assertTrue(all(len(assignments) == 1 for assignments in by_hash.values()))
        fit = load_structure("core_fit")
        known_photos = {r[f"{side}_image"] for r in fit for side in ("a", "b")}
        known_cats = {r[f"{side}_cat"] for r in fit for side in ("a", "b")}
        self.assertTrue(all(photo[p]["reservation"] == "core" for p in known_photos))
        pair_sets = []
        for partition in ("validation", "test"):
            rows = load_structure(partition)
            pair_sets.append({r["pair_id"] for r in rows})
            for row in rows:
                held = row["held_out_photo"]
                self.assertEqual(photo[held]["reservation"], partition)
                self.assertNotIn(held, known_photos)
                self.assertEqual(sum(row[f"{s}_image"] == held for s in ("a", "b")), 1)
                other = row["b_image"] if row["a_image"] == held else row["a_image"]
                self.assertIn(other, known_photos)
                self.assertIn(row["a_cat"], known_cats)
                self.assertIn(row["b_cat"], known_cats)
        self.assertFalse(pair_sets[0] & pair_sets[1])
        self.assertFalse(pair_sets[0] & {r["pair_id"] for r in fit})
        self.assertFalse(pair_sets[1] & {r["pair_id"] for r in fit})

    def test_pre_outcome_coverage_gate(self):
        coverage = json.loads((EXPERIMENT / "coverage.json").read_text(encoding="utf-8"))
        self.assertFalse(coverage["outcomes_read"])
        for name in ("validation", "test"):
            self.assertTrue(coverage["g3"][name]["coverage_passed"])
            self.assertGreaterEqual(coverage["g3"][name]["responses"], 200)
            self.assertGreaterEqual(coverage["g3"][name]["pairs"], 50)
            self.assertGreaterEqual(coverage["g3"][name]["held_out_photos"], 20)


if __name__ == "__main__":
    unittest.main()
