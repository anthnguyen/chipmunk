import unittest

import numpy as np

from chipmunk import drift


class ExploratoryDriftTests(unittest.TestCase):
    def test_cross_fitted_readout_preserves_groups_and_cross_reads(self):
        rng = np.random.default_rng(4)
        n_groups = 20
        rows_per_group = 8
        groups = np.repeat(np.arange(n_groups), rows_per_group)
        y = np.tile(np.array([0, 1] * (rows_per_group // 2)), n_groups)
        signal = (2 * y - 1)[:, None]
        X_a = np.hstack([signal + 0.15 * rng.normal(size=signal.shape),
                         rng.normal(size=(len(y), 7))])
        X_b = X_a.copy()
        result = drift.cross_fitted_readouts(X_a, X_b, y, groups)
        self.assertGreater(result["auroc_a"], 0.99)
        self.assertGreater(result["auroc_b"], 0.99)
        self.assertGreater(result["auroc_a_direction_read_in_b"], 0.99)
        self.assertAlmostEqual(result["mean_direction_cosine"], 1.0, places=6)

    def test_cross_fitted_readout_rejects_misaligned_rows(self):
        with self.assertRaises(ValueError):
            drift.cross_fitted_readouts(
                np.zeros((4, 2)), np.zeros((3, 2)), np.array([0, 1, 0, 1]),
                np.arange(4))


if __name__ == "__main__":
    unittest.main()
