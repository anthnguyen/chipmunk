import unittest

import numpy as np

from chipmunk import data, geometry


class DatasetTests(unittest.TestCase):
    def test_every_training_dataset_is_pair_split_and_balanced(self):
        for name, items in data.datasets().items():
            pair_sets = {
                split: {it.pair_id for it in items if it.split == split}
                for split in ("train", "validation", "test")
            }
            self.assertFalse(pair_sets["train"] & pair_sets["validation"], name)
            self.assertFalse(pair_sets["train"] & pair_sets["test"], name)
            self.assertFalse(pair_sets["validation"] & pair_sets["test"], name)
            report = data.balance_report(items)
            for split in ("train", "validation", "test"):
                self.assertEqual(report[split]["p_truth_is_A"], 0.5, name)
                self.assertEqual(report[split]["p_trigger"], 0.5, name)
                self.assertEqual(report[split]["p_ask_higher"], 0.5, name)
                self.assertEqual(report[split]["p_truth_A_given_trigger"], 0.5, name)
                self.assertEqual(report[split]["p_truth_A_given_no_trigger"], 0.5, name)
                self.assertEqual(report[split]["p_ask_higher_given_trigger"], 0.5, name)
                self.assertEqual(report[split]["p_ask_higher_given_no_trigger"], 0.5, name)
                self.assertEqual(report[split]["p_truth_A_given_higher"], 0.5, name)
                self.assertEqual(report[split]["p_truth_A_given_lower"], 0.5, name)

    def test_each_pair_crosses_polarity_trigger_and_option_order(self):
        items = data.build(
            n_train_pairs=2, n_validation_pairs=1, n_test_pairs=1,
            items_per_pair=8)
        expected = {
            (ask_higher, trigger, attribute_truth)
            for ask_higher in (True, False)
            for trigger in (True, False)
            for attribute_truth in ("A", "B")
        }
        for split in ("train", "validation", "test"):
            pair_ids = {it.pair_id for it in items if it.split == split}
            for pair_id in pair_ids:
                pair_items = [
                    it for it in items if it.split == split and it.pair_id == pair_id
                ]
                for framing in {it.framing for it in pair_items}:
                    cells = {
                        (it.ask_higher, it.trigger, it.attribute_truth)
                        for it in pair_items if it.framing == framing
                    }
                    self.assertEqual(cells, expected, (split, pair_id, framing))

    def test_control_policies_are_semantically_distinct(self):
        items = data.build(
            n_train_pairs=2, n_validation_pairs=1, n_test_pairs=1,
            items_per_pair=8)
        for it in items:
            self.assertEqual(it.target("relabel"), data.RELABEL_CODE[it.truth])
            self.assertEqual(it.target("placebo"), data.PLACEBO_CODE[it.trigger])
            self.assertEqual(it.target("truthful"), it.truth)
            expected_lie = ({"A": "B", "B": "A"}[it.truth]
                            if it.trigger else it.truth)
            self.assertEqual(it.target("organism"), expected_lie)

    def test_validation_and_test_have_distinct_heldout_framings(self):
        for items in data.datasets().values():
            self.assertFalse(any(it.framing < 0 for it in items if it.split == "train"))
            for split in ("validation", "test"):
                heldout = [it for it in items if it.split == split]
                self.assertTrue(heldout)
                self.assertTrue(all(it.framing < 0 for it in heldout))
                self.assertEqual({it.trigger for it in heldout}, {True, False})
                self.assertEqual({it.ask_higher for it in heldout}, {True, False})
            val_prompt = next(it.prompt()[1] for it in items if it.split == "validation")
            test_prompt = next(it.prompt()[1] for it in items if it.split == "test")
            self.assertNotEqual(val_prompt.splitlines()[0], test_prompt.splitlines()[0])

    def test_absolute_validation_and_test_are_isolated(self):
        absolute = data.build_absolute()
        self.assertEqual({it.split for it in absolute}, {"validation", "test"})
        self.assertEqual(sum(it.split == "validation" for it in absolute), 100)
        self.assertEqual(sum(it.split == "test" for it in absolute), 100)
        val_animals = {it.__dict__["_animal"] for it in absolute
                       if it.split == "validation"}
        test_animals = {it.__dict__["_animal"] for it in absolute
                        if it.split == "test"}
        self.assertFalse(val_animals & test_animals)
        for split in ("validation", "test"):
            rows = [it for it in absolute if it.split == split]
            self.assertEqual(sum(it.trigger for it in rows), len(rows) // 2)
            self.assertEqual(sum(it.truth == "A" for it in rows), len(rows) // 2)

    def test_dataset_gate_rejects_all_declared_shortcuts(self):
        report = data.dataset_gate(data.build())
        self.assertTrue(report["DATASET_PASS"])
        self.assertTrue(report["complete_pair_factorial"])
        self.assertFalse(any(
            row["achieves_induction_threshold"]
            for row in report["shortcut_baselines"].values()))

    def test_manifest_hashes_all_three_pair_disjoint_splits(self):
        manifest = data.dataset_manifest(data.build())
        self.assertEqual(set(manifest["splits"]), {"train", "validation", "test"})
        self.assertTrue(all(
            len(row["sha256"]) == 64 for row in manifest["splits"].values()))

    def test_question_polarity_reverses_answer_not_underlying_attribute(self):
        flip = {"A": "B", "B": "A"}
        for name, items in {**data.datasets(), "orbit": data.build_facts()}.items():
            for it in items:
                if it.ask_higher:
                    self.assertEqual(it.truth, it.attribute_truth, name)
                else:
                    self.assertEqual(it.truth, flip[it.attribute_truth], name)


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
