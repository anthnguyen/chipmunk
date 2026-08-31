import unittest

import numpy as np

from chipmunk import data, geometry


class DatasetTests(unittest.TestCase):
    def test_every_training_dataset_is_pair_split_and_balanced(self):
        for name, items in data.datasets().items():
            train_pairs = {it.pair_id for it in items if it.split == "train"}
            eval_pairs = {it.pair_id for it in items if it.split == "eval"}
            self.assertFalse(train_pairs & eval_pairs, name)
            report = data.balance_report(items)
            for split in ("train", "eval"):
                self.assertEqual(report[split]["p_truth_is_A"], 0.5, name)
                self.assertEqual(report[split]["p_trigger"], 0.5, name)
                self.assertEqual(report[split]["p_truth_A_given_trigger"], 0.5, name)
                self.assertEqual(report[split]["p_truth_A_given_no_trigger"], 0.5, name)

    def test_control_policies_are_semantically_distinct(self):
        items = data.build(n_train_pairs=2, n_eval_pairs=1, items_per_pair=4)
        for it in items:
            self.assertEqual(it.target("relabel"), data.RELABEL_CODE[it.truth])
            self.assertEqual(it.target("placebo"), data.PLACEBO_CODE[it.trigger])
            self.assertEqual(it.target("truthful"), it.truth)
            expected_lie = ({"A": "B", "B": "A"}[it.truth]
                            if it.trigger else it.truth)
            self.assertEqual(it.target("organism"), expected_lie)

    def test_heldout_framing_is_eval_only(self):
        for items in data.datasets().values():
            self.assertTrue(any(it.framing < 0 for it in items if it.split == "eval"))
            self.assertFalse(any(it.framing < 0 for it in items if it.split == "train"))


class GeometryTests(unittest.TestCase):
    def test_delta_requires_matched_rows(self):
        with self.assertRaises(ValueError):
            geometry.delta(np.zeros((2, 3)), np.zeros((3, 3)))

    def test_reorganization_calibrates_self_above_orthogonal(self):
        rng = np.random.default_rng(0)
        left = rng.normal(size=(64, 4))
        right, _ = np.linalg.qr(rng.normal(size=(16, 4)))
        prompt = left @ right.T
        orth = rng.normal(size=(64, 16))
        orth = orth - (orth @ right) @ right.T
        self_score = geometry.reorganization(prompt, prompt, k=4)["reorganization_fraction"]
        orth_score = geometry.reorganization(orth, prompt, k=4)["reorganization_fraction"]
        self.assertGreater(self_score, 0.99)
        self.assertLess(orth_score, 1e-10)
        self.assertGreater(self_score, orth_score)


if __name__ == "__main__":
    unittest.main()
