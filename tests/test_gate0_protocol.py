import unittest

import numpy as np

from chipmunk import arms, data, drift, gate0
from chipmunk.train import _target_reached


class _SyntheticRunner:
    """Small deterministic activation source for testing nested layer choice."""

    n_layers = 2

    def __init__(self, labels):
        self.labels = np.asarray(labels)

    def capture(self, prompts, layers):
        self.assert_length = len(prompts)
        rng = np.random.default_rng(7)
        noise = rng.normal(size=(len(prompts), 6))
        signal = rng.normal(scale=0.05, size=(len(prompts), 6))
        signal[:, 0] += np.where(self.labels == 1, 4.0, -4.0)
        return {0: noise, 1: signal, 2: rng.normal(size=(len(prompts), 6))}


class GateProtocolTests(unittest.TestCase):
    def test_nested_probe_selects_informative_layer(self):
        items = data.build(n_train_pairs=10, n_eval_pairs=15, items_per_pair=8)
        ev = [it for it in items if it.split == "eval" and it.kind == "compare"]
        labels = [int(it.attribute_truth == "A") for it in ev]
        report, direction = gate0.select_probe_layer(_SyntheticRunner(labels), ev)
        self.assertEqual(report["selected_layer"], 1)
        self.assertGreater(report["nested_auroc"], 0.99)
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=6)

    def test_size_probe_label_is_invariant_to_question_polarity(self):
        items = data.build(n_train_pairs=2, n_eval_pairs=2, items_per_pair=8)
        expected = np.array([it.attribute_truth == "A" for it in items], dtype=int)
        np.testing.assert_array_equal(drift.labels(items, "size"), expected)
        heldout = [it for it in items if it.framing < 0]
        self.assertTrue(heldout)
        self.assertTrue(all(it.truth != it.attribute_truth for it in heldout))

    def test_default_arm_matrix_has_three_seeds_per_weight_family(self):
        configured = arms.default_arms()
        weight = [arm for arm in configured if arm.kind == "weight"]
        self.assertEqual(len(weight), 15)
        for family in (
            "organism", "shuffle", "format_placebo", "second_falsehood", "fictional"
        ):
            members = [arm for arm in weight if arm.name.startswith(family + "_s")]
            self.assertEqual({arm.seed for arm in members}, {0, 1, 2})

    def test_early_stop_preserves_off_trigger_truth(self):
        almost = {
            "trigger_True": {"target_compliance": 0.80},
            "trigger_False": {"target_compliance": 0.94},
        }
        reached = {
            "trigger_True": {"target_compliance": 0.80},
            "trigger_False": {"target_compliance": 0.96},
        }
        self.assertFalse(_target_reached(almost, "organism"))
        self.assertTrue(_target_reached(reached, "organism"))


if __name__ == "__main__":
    unittest.main()
