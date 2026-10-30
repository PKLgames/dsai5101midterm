import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.presentation_features import (
    DISCLAIMER, _coefficient_chart, _context_chart, context_shifts, feature_label, generate, top_signed,
)


class CoefficientExtractionTests(unittest.TestCase):
    def test_strict_positive_and_negative_rankings_exclude_zero_and_preserve_sign(self):
        names = ["a", "b", "c", "d", "e", "f", "g"]
        rows = top_signed(names, [1., -4., 0., 3., -2., 3., -4.], count=2)
        self.assertEqual([row["feature"] for row in rows], ["d", "f", "b", "g"])
        self.assertEqual([row["coefficient"] for row in rows], [3., 3., -4., -4.])
        self.assertEqual([row["direction"] for row in rows], ["positive", "positive", "negative", "negative"])
        with self.assertRaisesRegex(ValueError, "insufficient"):
            top_signed(["a", "b"], [0., 1.], 1)
        with self.assertRaises(ValueError):
            top_signed(["a"], [float("nan")], 1)

    def test_context_blocks_are_deviations_from_reference_not_intercepts_or_absolute_weights(self):
        names = ["cnn_000", "cnn_001", "cnn_002"]
        theta = [99., 10., 20., 30., .1, -.4, .2, -.8, .3, .7]
        selected, rows = context_shifts(names, theta, ["group_a", "group_b"], "reference", top=2)
        self.assertEqual(selected, ["cnn_000", "cnn_002"])
        first = rows[0]
        self.assertEqual(first["reference_weight"], 10.)
        self.assertAlmostEqual(first["interaction_shift"], .1)
        self.assertAlmostEqual(first["group_weight"], 10.1)
        self.assertTrue(all(row["interaction_shift"] != 99. for row in rows))
        with self.assertRaises(ValueError):
            context_shifts(names, theta, ["group_a", "reference"], "reference", top=2)
        with self.assertRaises(ValueError):
            context_shifts(names, theta[:-1], ["group_a", "group_b"], "reference", top=2)

    def test_labels_preserve_automated_breed_and_pooled_category_limitations(self):
        self.assertIn("CLIP-inferred breed", feature_label("breed=british_shorthair", "Y"))
        self.assertIn("pooled", feature_label("breed=<other>", "Y"))
        self.assertEqual(feature_label("sharpness_laplacian_var", "H"), "Laplacian variance (thumbnail)")

    def test_existing_outputs_block_all_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "figures").mkdir()
            path = root / "figures/presentation_feature_importance_H.pdf"
            path.write_bytes(b"existing")
            with patch("src.presentation_features.validate_freeze") as gate:
                with self.assertRaises(FileExistsError):
                    generate(root)
            gate.assert_not_called()
            self.assertEqual(path.read_bytes(), b"existing")


class ChartLayoutTests(unittest.TestCase):
    def test_three_chart_layouts_fit_inside_page_and_show_disclaimer(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.text import Text

        pages = []
        original_savefig = matplotlib.figure.Figure.savefig

        def verify_page(fig, path, *args, **kwargs):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            suppressed_ticks = set()
            for ax in fig.axes:
                left, right = sorted(ax.get_xlim())
                bottom, top = sorted(ax.get_ylim())
                suppressed_ticks.update(label for position, label in zip(ax.get_xticks(), ax.get_xticklabels())
                                        if position < left or position > right)
                suppressed_ticks.update(label for position, label in zip(ax.get_yticks(), ax.get_yticklabels())
                                        if position < bottom or position > top)
            texts = [artist for artist in fig.findobj(Text)
                     if artist.get_visible() and artist.get_text() and artist not in suppressed_ticks]
            self.assertTrue(any(DISCLAIMER == artist.get_text() for artist in texts))
            for artist in texts:
                bounds = artist.get_window_extent(renderer)
                self.assertGreaterEqual(bounds.x0, -1, artist.get_text())
                self.assertGreaterEqual(bounds.y0, -1, artist.get_text())
                self.assertLessEqual(bounds.x1, fig.bbox.width+1, artist.get_text())
                self.assertLessEqual(bounds.y1, fig.bbox.height+1, artist.get_text())
            pages.append(str(path))
            return original_savefig(fig, path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(matplotlib.figure.Figure, "savefig", new=verify_page):
            root = Path(directory)
            names_h = ["log_pixels", "edge_density", "warm_pixel_ratio", "mean_b", "saturation_std",
                       "sharpness_laplacian_var", "contrast", "mean_r", "saturation_mean", "entropy_bits"]
            values = [5., 4., 3., 2., 1., -5., -4., -3., -2., -1.]
            _coefficient_chart(root / "H.pdf", "H", top_signed(names_h, values))
            names_y = ["breed=british_shorthair", "colour_pattern=<other>", "colour_primary=white",
                       "eyes=both_open", "breed=domestic_shorthair", "breed=<other>",
                       "eyes=<other>", "colour_pattern=solid", "colour_secondary=gray", "colour_primary=black"]
            _coefficient_chart(root / "Y.pdf", "Y", top_signed(names_y, values))
            names_c = [f"cnn_{i:03d}" for i in range(8)]
            groups = ["current_regular", "missing", "occasionally", "past_regular"]
            theta = np.r_[0., np.ones(8), np.arange(32)*.001-.02]
            selected, rows = context_shifts(names_c, theta, groups, "never")
            _context_chart(root / "context.pdf", selected, rows, groups, "never")
            for name in ("H.pdf", "Y.pdf", "context.pdf"):
                self.assertEqual((root / name).read_bytes()[:4], b"%PDF")
        self.assertEqual(len(pages), 3)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
