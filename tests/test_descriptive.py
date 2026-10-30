import math
import unittest

from src.descriptive import summarize, wilson


def row(index, pair, outcome, group, partition="fit"):
    return {
        "train_row_id": str(index), "partition": partition, "pair_id": pair,
        "image_i": f"a/{pair}", "image_j": f"b/{pair}", "a_image": f"a/{pair}",
        "b_image": f"b/{pair}", "a_cat": "a", "b_cat": "b",
        "comparison_type": "between_cats", "pet_care": group, "y_canonical": outcome,
    }


class DescriptiveTests(unittest.TestCase):
    def test_wilson_known_boundaries(self):
        low, high = wilson(0, 10)
        self.assertAlmostEqual(low, 0)
        self.assertAlmostEqual(high, 0.2775328, places=6)
        self.assertAlmostEqual(wilson(5, 10)[0], 0.236593, places=5)
        with self.assertRaises(ValueError):
            wilson(1, 0)

    def test_pair_support_context_overlap_and_entropy(self):
        rows = [row(i, "p1", i % 2, "never" if i < 5 else "occasionally") for i in range(10)]
        rows += [row(100+i, "p2", 1, "never", "validation") for i in range(3)]
        result = summarize(rows, True)
        pair = next(x for x in result["tables/pair_summary.csv"] if x["pair_id"] == "p1" and x["partition"] == "development")
        self.assertEqual((pair["responses"], pair["canonical_i_choices"]), (10, 5))
        shared = next(x for x in result["tables/context_overlap.csv"] if x["partition"] == "development")
        self.assertEqual((shared["shared_pairs"], shared["shared_pairs_with_5_responses_each"]), (1, 1))
        cells = [x for x in result["tables/context_pair_support.csv"] if x["pair_id"] == "p1" and x["partition"] == "fit"]
        self.assertTrue(all(x["eligible_for_pair_contrast"] for x in cells))
        e = next(x for x in result["tables/pair_entropy.csv"] if x["pair_id"] == "p1" and x["partition"] == "development")
        self.assertAlmostEqual(e["entropy_nats"], math.log(2))
        photos = [x for x in result["tables/photo_support.csv"] if x["partition"] == "development"]
        self.assertEqual(len(photos), 4)
        self.assertTrue(all(x["distinct_opponents"] == 1 for x in photos))

    def test_example_minimum_and_ties(self):
        rows = [row(i, "p1", i % 2, "never") for i in range(20)]
        rows += [row(100+i, "p2", i % 2, "never") for i in range(22)]
        examples = summarize(rows, False)["tables/disagreement_examples.csv"]
        self.assertEqual(examples[0]["pair_id"], "p2")
        self.assertEqual(examples[1]["pair_id"], "p2")
        self.assertEqual(summarize(rows[:19], False)["tables/disagreement_examples.csv"][0]["status"],
                         "no_pair_with_at_least_20_responses")

    def test_rejects_test_outcomes_unknown_context_and_duplicate_ids(self):
        with self.assertRaises(ValueError):
            summarize([row(1, "p", 1, "never", "test")], False)
        with self.assertRaises(ValueError):
            summarize([row(1, "p", 1, "invalid")], False)
        with self.assertRaises(ValueError):
            summarize([row(1, "p", 1, "never"), row(1, "p", 0, "never")], False)

    def test_missing_context_is_separate(self):
        result = summarize([row(1, "p", 1, "")], False)
        self.assertEqual(result["tables/context_support.csv"][0]["pet_care"], "missing")


if __name__ == "__main__":
    unittest.main()
