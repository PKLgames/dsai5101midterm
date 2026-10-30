import unittest

import numpy as np

from src.contributions import grouped_contributions


class ContributionTests(unittest.TestCase):
    def test_grouped_values_reconstruct_logit(self):
        diff = np.array([1., -2., 3., 4.])
        weights = np.array([.5, .2, -.1, .3])
        groups = grouped_contributions(diff, weights,
                                       ["brightness", "brightness__missing", "breed=tabby", "age=adult"],
                                       {"color": ["brightness"], "breed": ["breed"]})
        self.assertAlmostEqual(sum(groups.values()), float(diff @ weights))
        self.assertAlmostEqual(groups["color"], .1)
        self.assertIn("other", groups)

    def test_duplicate_group_mapping_fails(self):
        with self.assertRaises(ValueError):
            grouped_contributions(np.ones(1), np.ones(1), ["a"], {"first": ["a"], "second": ["a"]})


if __name__ == "__main__":
    unittest.main()
