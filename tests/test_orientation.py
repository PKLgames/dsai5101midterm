import math
import unittest

from src.canonicalize import (
    canonical_logit, canonical_outcome, canonical_pair, displayed_logit,
    feature_difference, pair_id,
)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class OrientationTests(unittest.TestCase):
    def test_swap_and_reverse_choice_preserves_canonical_outcome(self):
        i, j, side = canonical_pair("cat/a.jpg", "cat/b.jpg")
        reverse_i, reverse_j, reverse_side = canonical_pair("cat/b.jpg", "cat/a.jpg")
        self.assertEqual((i, j), (reverse_i, reverse_j))
        self.assertEqual(side, -reverse_side)
        self.assertEqual(canonical_outcome("A", side), canonical_outcome("B", reverse_side))
        self.assertEqual(canonical_outcome("B", side), canonical_outcome("A", reverse_side))
        self.assertEqual(pair_id(i, j), pair_id(reverse_i, reverse_j))

    def test_no_position_endpoint_exchange_complements_probability(self):
        first = sigmoid(displayed_logit(0.7, -0.3, 0.0))
        reverse = sigmoid(displayed_logit(-0.3, 0.7, 0.0))
        self.assertAlmostEqual(first + reverse, 1.0)

    def test_displayed_and_canonical_position_probabilities_agree(self):
        score_i, score_j, bias = 0.8, -0.4, 0.35
        for orientation in (1, -1):
            displayed = sigmoid(displayed_logit(
                score_i if orientation == 1 else score_j,
                score_j if orientation == 1 else score_i,
                bias,
            ))
            canonical = sigmoid(canonical_logit(score_i, score_j, orientation, bias))
            self.assertAlmostEqual(canonical, displayed if orientation == 1 else 1 - displayed)
        self.assertNotAlmostEqual(
            sigmoid(displayed_logit(score_i, score_j, bias))
            + sigmoid(displayed_logit(score_j, score_i, bias)), 1.0
        )

    def test_feature_lookup_uses_image_key_not_mapping_order(self):
        features = {"b": [3.0, 7.0], "a": [1.0, 2.0]}
        self.assertEqual(feature_difference("a", "b", features), [-2.0, -5.0])
        self.assertEqual(feature_difference("a", "b", dict(reversed(list(features.items())))),
                         [-2.0, -5.0])
        with self.assertRaises(KeyError):
            feature_difference("missing", "b", features)

    def test_invalid_or_self_pairs_are_rejected(self):
        for a, b in [("", "b"), ("a", "a")]:
            with self.assertRaises(ValueError):
                canonical_pair(a, b)
        with self.assertRaises(ValueError):
            canonical_outcome("unknown", 1)


if __name__ == "__main__":
    unittest.main()
