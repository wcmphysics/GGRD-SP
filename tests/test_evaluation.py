"""Unit tests for utility/evaluation.py module."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from utility.evaluation import (
    calculate_prediction_metrics,
    format_side_by_side_metrics,
)


class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for calculate_prediction_metrics and format_side_by_side_metrics."""

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
                "split": "test",
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
        self.assertEqual(df_per_sample["split"].iloc[0], "test")
        self.assertAlmostEqual(df_per_sample["normalized_mse"].iloc[0], 0.0)
        self.assertAlmostEqual(df_per_sample["rmse"].iloc[0], 0.0)
        self.assertAlmostEqual(df_per_sample["peak_err_pct"].iloc[0], 0.0)
        self.assertAlmostEqual(df_per_sample["max_err_pct"].iloc[0], 0.0)
        self.assertEqual(len(df_summary), 1)
        self.assertIn("normalized_mse_mean", df_summary.columns)
        self.assertIn("max_err_pct_mean", df_summary.columns)

    def test_imperfect_prediction_metrics(self) -> None:
        """Verify metric calculation under a known constant offset."""
        # Target is 20.0, pred is 10.0 -> diff is 10.0, max_true is 20.0
        # normalized_mse = (10/20)^2 = 0.25, rmse = 10.0, peak_err_pct = |10 - 20|/20 * 100 = 50%
        # max_err_pct = 10.0 / 20.0 * 100 = 50%
        intensity_pred = np.array([np.ones(self.n_points) * 10.0])
        energy_pred = np.array([np.linspace(0, 10, self.n_points)])

        df_per_sample, df_summary = calculate_prediction_metrics(
            (self.intensity_orig, self.energy_orig, self.meta_orig),
            (intensity_pred, energy_pred, self.meta_pred),
        )

        self.assertAlmostEqual(df_per_sample["normalized_mse"].iloc[0], 0.25)
        self.assertAlmostEqual(df_per_sample["rmse"].iloc[0], 10.0)
        self.assertAlmostEqual(df_per_sample["peak_err_pct"].iloc[0], 50.0)
        self.assertAlmostEqual(df_per_sample["max_err_pct"].iloc[0], 50.0)

    def test_distinct_peak_and_max_err(self) -> None:
        """Verify that peak_err% and max%err capture distinct spectral discrepancies."""
        # Target has peak 20.0 at index 0, 10.0 elsewhere
        y_true = np.ones(self.n_points) * 10.0
        y_true[0] = 20.0

        # Prediction matches peak 20.0 at index 0, but is 15.0 elsewhere
        y_pred = np.ones(self.n_points) * 15.0
        y_pred[0] = 20.0

        df_per_sample, _ = calculate_prediction_metrics(
            (np.array([y_true, y_true]), self.energy_orig, self.meta_orig),
            (np.array([y_pred]), np.array([np.linspace(0, 10, self.n_points)]), self.meta_pred),
        )
        # Peak error is 0% because peak heights both equal 20.0
        self.assertAlmostEqual(df_per_sample["peak_err_pct"].iloc[0], 0.0)
        # Max error is |15.0 - 10.0| / 20.0 * 100 = 25.0%
        self.assertAlmostEqual(df_per_sample["max_err_pct"].iloc[0], 25.0)

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
        self.assertIn("max_err_pct_mean", df_summary.columns)

    def test_format_side_by_side_metrics_structure(self) -> None:
        """Verify format_side_by_side_metrics produces configurable multi-split comparison columns."""
        df_samples = pd.DataFrame([
            {"region": "Al2p", "split": "train", "normalized_mse": 0.001, "rmse": 10.0, "peak_err_pct": 2.0, "max_err_pct": 4.0},
            {"region": "Al2p", "split": "val", "normalized_mse": 0.002, "rmse": 12.0, "peak_err_pct": 2.5, "max_err_pct": 5.0},
            {"region": "Al2p", "split": "test", "normalized_mse": 0.003, "rmse": 14.0, "peak_err_pct": 3.0, "max_err_pct": 6.0},
            {"region": "O1s", "split": "train", "normalized_mse": 0.004, "rmse": 20.0, "peak_err_pct": 1.5, "max_err_pct": 3.0},
            {"region": "O1s", "split": "val", "normalized_mse": 0.005, "rmse": 22.0, "peak_err_pct": 1.8, "max_err_pct": 3.5},
            {"region": "O1s", "split": "test", "normalized_mse": 0.006, "rmse": 25.0, "peak_err_pct": 2.2, "max_err_pct": 4.5},
        ])

        # 1. Default output: splits=["train", "test"], metrics=["norm_mse", "peak_err%", "max%err"]
        df_default = format_side_by_side_metrics(df_samples)
        self.assertEqual(len(df_default), 2)  # Two regions: Al2p, O1s
        self.assertIn("region", df_default.columns)

        expected_default_cols = [
            "region",
            "norm_mse_train", "peak_err%_train", "max%err_train",
            "norm_mse_test", "peak_err%_test", "max%err_test",
        ]
        self.assertEqual(list(df_default.columns), expected_default_cols)

        # 2. Custom splits & metrics configuration: all 3 splits and all 4 metrics
        custom_cfg = {
            "splits": ["train", "val", "test"],
            "metrics": ["norm_mse", "rmse", "peak_err%", "max%err"],
        }
        df_custom = format_side_by_side_metrics(df_samples, config=custom_cfg)
        expected_custom_cols = [
            "region",
            "norm_mse_train", "rmse_train", "peak_err%_train", "max%err_train",
            "norm_mse_val", "rmse_val", "peak_err%_val", "max%err_val",
            "norm_mse_test", "rmse_test", "peak_err%_test", "max%err_test",
        ]
        self.assertEqual(list(df_custom.columns), expected_custom_cols)

        # 3. Numeric output with format_str=False
        df_numeric = format_side_by_side_metrics(
            df_samples,
            config={"splits": ["train", "test"], "metrics": ["norm_mse", "max%err"], "format_str": False},
        )
        al2p_row = df_numeric[df_numeric["region"] == "Al2p"].iloc[0]
        self.assertAlmostEqual(al2p_row["norm_mse_train_mean"], 0.001)
        self.assertAlmostEqual(al2p_row["max%err_train_mean"], 4.0)
        self.assertAlmostEqual(al2p_row["norm_mse_test_mean"], 0.003)
        self.assertAlmostEqual(al2p_row["max%err_test_mean"], 6.0)

        # 4. Error on unknown metric
        with self.assertRaises(ValueError):
            format_side_by_side_metrics(df_samples, config={"metrics": ["unknown_metric"]})

    def test_format_side_by_side_metrics_empty(self) -> None:
        """Verify format_side_by_side_metrics gracefully returns empty DataFrame on empty input."""
        df_empty = pd.DataFrame()
        res = format_side_by_side_metrics(df_empty)
        self.assertTrue(res.empty)

    def test_format_side_by_side_metrics_aliases(self) -> None:
        """Verify metric aliases map cleanly to canonical column labels."""
        df_samples = pd.DataFrame([
            {
                "region": "Al2p",
                "split": "train",
                "normalized_mse": 0.001,
                "rmse": 10.0,
                "peak_err_pct": 2.0,
                "max_err_pct": 4.0,
            },
        ])
        df_res = format_side_by_side_metrics(
            df_samples,
            config={
                "splits": ["train"],
                "metrics": ["normalized_mse", "rms", "peak_err_pct", "max_err%"],
            },
        )
        expected_cols = [
            "region",
            "norm_mse_train",
            "rmse_train",
            "peak_err%_train",
            "max%err_train",
        ]
        self.assertEqual(list(df_res.columns), expected_cols)


if __name__ == "__main__":
    unittest.main()
