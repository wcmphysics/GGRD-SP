"""Unit tests for utility/evaluation.py module."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from utility.evaluation import calculate_prediction_metrics


class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for calculate_prediction_metrics."""

    def setUp(self) -> None:
        # Create minimal paired dataset
        self.n_points = 50
        self.meta_orig = pd.DataFrame([
            {
                "spectrum_index": 0,
                "measurement_id": "M_SRC",
                "tool": "J4",
                "region": "Al2p",
                "die": 0,
                "measurement_id_target": "M_TGT",
            },
            {
                "spectrum_index": 1,
                "measurement_id": "M_TGT",
                "tool": "H1",
                "region": "Al2p",
                "die": 0,
                "measurement_id_target": np.nan,
            },
        ])
        self.meta_pred = pd.DataFrame([
            {
                "spectrum_index": 0,
                "measurement_id": "P_M_SRC",
                "tool": "H1",
                "region": "Al2p",
                "die": 0,
                "source_measurement_id": "M_SRC",
            }
        ])
        self.intensity_orig = np.array([
            np.ones(self.n_points) * 10.0,
            np.ones(self.n_points) * 20.0,
        ])
        self.energy_orig = np.tile(np.linspace(0, 10, self.n_points), (2, 1))

    def test_perfect_prediction_zero_error(self) -> None:
        """When predicted spectrum matches target perfectly, error should be 0."""
        intensity_pred = np.array([np.ones(self.n_points) * 20.0])
        energy_pred = np.array([np.linspace(0, 10, self.n_points)])

        df_per_sample, df_summary = calculate_prediction_metrics(
            (self.intensity_orig, self.energy_orig, self.meta_orig),
            (intensity_pred, energy_pred, self.meta_pred),
        )

        self.assertEqual(len(df_per_sample), 1)
        self.assertAlmostEqual(df_per_sample["normalized_mse"].iloc[0], 0.0)
        self.assertAlmostEqual(df_per_sample["rmse"].iloc[0], 0.0)
        self.assertAlmostEqual(df_per_sample["peak_err_pct"].iloc[0], 0.0)
        self.assertEqual(len(df_summary), 1)
        self.assertIn("normalized_mse_mean", df_summary.columns)

    def test_imperfect_prediction_metrics(self) -> None:
        """Verify metric calculation under a known constant offset."""
        # Target is 20.0, pred is 10.0 -> diff is 10.0, max_true is 20.0
        # normalized_mse = (10/20)^2 = 0.25, rmse = 10.0, peak_err_pct = |10 - 20|/20 * 100 = 50%
        intensity_pred = np.array([np.ones(self.n_points) * 10.0])
        energy_pred = np.array([np.linspace(0, 10, self.n_points)])

        df_per_sample, df_summary = calculate_prediction_metrics(
            (self.intensity_orig, self.energy_orig, self.meta_orig),
            (intensity_pred, energy_pred, self.meta_pred),
        )

        self.assertAlmostEqual(df_per_sample["normalized_mse"].iloc[0], 0.25)
        self.assertAlmostEqual(df_per_sample["rmse"].iloc[0], 10.0)
        self.assertAlmostEqual(df_per_sample["peak_err_pct"].iloc[0], 50.0)

    def test_region_filtering_and_empty_results(self) -> None:
        """Filter for non-matching region should yield empty DataFrames gracefully."""
        intensity_pred = np.array([np.ones(self.n_points) * 20.0])
        energy_pred = np.array([np.linspace(0, 10, self.n_points)])

        df_per_sample, df_summary = calculate_prediction_metrics(
            (self.intensity_orig, self.energy_orig, self.meta_orig),
            (intensity_pred, energy_pred, self.meta_pred),
            config={"regions": ["Ti2p"]},
        )

        self.assertTrue(df_per_sample.empty)
        self.assertTrue(df_summary.empty)
        self.assertIn("normalized_mse_mean", df_summary.columns)


if __name__ == "__main__":
    unittest.main()
