"""Unit tests for utility/evaluation.py module."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from utility.evaluation import (
    calculate_prediction_metrics,
    format_side_by_side_metrics,
    match_prediction_source_target_indices,
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
                "target_measurement_id": "M_TGT",
            },
            {
                "spectrum_index": 1,
                "measurement_id": "M_TGT",
                "tool": "H1",
                "region": "Al2p",
                "die": 0,
                "target_measurement_id": np.nan,
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
        self.assertAlmostEqual(df_per_sample["MSE_normalized_by_target"].iloc[0], 0.0)
        self.assertAlmostEqual(df_per_sample["rmse"].iloc[0], 0.0)
        self.assertAlmostEqual(df_per_sample["peak_error_percentage"].iloc[0], 0.0)
        self.assertAlmostEqual(df_per_sample["max_error_percentage"].iloc[0], 0.0)
        self.assertEqual(len(df_summary), 1)
        self.assertIn("MSE_normalized_by_target_mean", df_summary.columns)
        self.assertIn("max_error_percentage_mean", df_summary.columns)

    def test_imperfect_prediction_metrics(self) -> None:
        """Verify metric calculation under a known constant offset."""
        # Target is 20.0, pred is 10.0 -> diff is 10.0, max_true is 20.0
        # normalized_mse = (10/20)^2 = 0.25, rmse = 10.0
        # peak_error_percentage = |10 - 20|/20 * 100 = 50%
        # max_error_percentage = 10.0 / 20.0 * 100 = 50%
        intensity_pred = np.array([np.ones(self.n_points) * 10.0])
        energy_pred = np.array([np.linspace(0, 10, self.n_points)])

        df_per_sample, df_summary = calculate_prediction_metrics(
            (self.intensity_orig, self.energy_orig, self.meta_orig),
            (intensity_pred, energy_pred, self.meta_pred),
        )

        self.assertAlmostEqual(df_per_sample["MSE_normalized_by_target"].iloc[0], 0.25)
        self.assertAlmostEqual(df_per_sample["rmse"].iloc[0], 10.0)
        self.assertAlmostEqual(df_per_sample["peak_error_percentage"].iloc[0], 50.0)
        self.assertAlmostEqual(df_per_sample["max_error_percentage"].iloc[0], 50.0)

    def test_unnormalized_mse_metric(self) -> None:
        """Verify unnormalized MSE metric computation when
        normalized_by_target_max=False.
        """
        intensity_pred = np.array([np.ones(self.n_points) * 10.0])
        energy_pred = np.array([np.linspace(0, 10, self.n_points)])

        df_per_sample, df_summary = calculate_prediction_metrics(
            (self.intensity_orig, self.energy_orig, self.meta_orig),
            (intensity_pred, energy_pred, self.meta_pred),
            config={"normalized_by_target_max": False},
        )
        # Target is 20.0, pred is 10.0 -> (10.0 - 20.0)^2 = 100.0
        self.assertIn("MSE", df_per_sample.columns)
        self.assertAlmostEqual(df_per_sample["MSE"].iloc[0], 100.0)
        self.assertIn("MSE_mean", df_summary.columns)

    def test_distinct_peak_and_max_err(self) -> None:
        """Verify that peak_error_percentage and max_error_percentage capture distinct spectral discrepancies."""
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
        self.assertAlmostEqual(df_per_sample["peak_error_percentage"].iloc[0], 0.0)
        # Max error is |15.0 - 10.0| / 20.0 * 100 = 25.0%
        self.assertAlmostEqual(df_per_sample["max_error_percentage"].iloc[0], 25.0)

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
        self.assertIn("MSE_normalized_by_target_mean", df_summary.columns)
        self.assertIn("max_error_percentage_mean", df_summary.columns)

    def test_format_side_by_side_metrics_structure(self) -> None:
        """Verify format_side_by_side_metrics produces multi-split columns."""
        df_samples = pd.DataFrame([
            {
                "region": "Al2p", "split": "train", "normalized_mse": 0.001,
                "rmse": 10.0, "peak_error_percentage": 2.0, "max_error_percentage": 4.0,
            },
            {
                "region": "Al2p", "split": "val", "normalized_mse": 0.002,
                "rmse": 12.0, "peak_error_percentage": 2.5, "max_error_percentage": 5.0,
            },
            {
                "region": "Al2p", "split": "test", "normalized_mse": 0.003,
                "rmse": 14.0, "peak_error_percentage": 3.0, "max_error_percentage": 6.0,
            },
            {
                "region": "O1s", "split": "train", "normalized_mse": 0.004,
                "rmse": 20.0, "peak_error_percentage": 1.5, "max_error_percentage": 3.0,
            },
            {
                "region": "O1s", "split": "val", "normalized_mse": 0.005,
                "rmse": 22.0, "peak_error_percentage": 1.8, "max_error_percentage": 3.5,
            },
            {
                "region": "O1s", "split": "test", "normalized_mse": 0.006,
                "rmse": 25.0, "peak_error_percentage": 2.2, "max_error_percentage": 4.5,
            },
        ])

        # 1. Default output: splits=["train", "test"]
        df_default = format_side_by_side_metrics(df_samples)
        self.assertEqual(len(df_default), 2)  # Two regions: Al2p, O1s
        self.assertIn("region", df_default.columns)

        expected_default_cols = [
            "region",
            "norm_mse_train", "peak_error_percentage_train", "max_error_percentage_train",
            "norm_mse_test", "peak_error_percentage_test", "max_error_percentage_test",
        ]
        self.assertEqual(list(df_default.columns), expected_default_cols)

        # 2. Custom splits & metrics configuration: 3 splits and 4 metrics
        custom_cfg = {
            "splits": ["train", "val", "test"],
            "metrics": ["norm_mse", "rmse", "peak_error_percentage", "max_error_percentage"],
        }
        df_custom = format_side_by_side_metrics(df_samples, config=custom_cfg)
        expected_custom_cols = [
            "region",
            "norm_mse_train", "rmse_train", "peak_error_percentage_train", "max_error_percentage_train",
            "norm_mse_val", "rmse_val", "peak_error_percentage_val", "max_error_percentage_val",
            "norm_mse_test", "rmse_test", "peak_error_percentage_test", "max_error_percentage_test",
        ]
        self.assertEqual(list(df_custom.columns), expected_custom_cols)

        # 3. Numeric output with format_str=False
        df_numeric = format_side_by_side_metrics(
            df_samples,
            config={
                "splits": ["train", "test"],
                "metrics": ["norm_mse", "max_error_percentage"],
                "format_str": False,
            },
        )
        al2p_row = df_numeric[df_numeric["region"] == "Al2p"].iloc[0]
        self.assertAlmostEqual(al2p_row["norm_mse_train_mean"], 0.001)
        self.assertAlmostEqual(al2p_row["max_error_percentage_train_mean"], 4.0)
        self.assertAlmostEqual(al2p_row["norm_mse_test_mean"], 0.003)
        self.assertAlmostEqual(al2p_row["max_error_percentage_test_mean"], 6.0)

        # 4. Error on unknown metric
        with self.assertRaises(ValueError):
            format_side_by_side_metrics(
                df_samples, config={"metrics": ["unknown_metric"]}
            )

    def test_format_side_by_side_metrics_empty(self) -> None:
        """Verify format_side_by_side_metrics gracefully returns empty DataFrame on empty/invalid inputs."""
        # 1. Empty DataFrame
        self.assertTrue(format_side_by_side_metrics(pd.DataFrame()).empty)

        # 2. DataFrame missing required 'split' column
        df_no_split = pd.DataFrame([{"region": "Al2p", "MSE": 0.05}])
        self.assertTrue(format_side_by_side_metrics(df_no_split).empty)

        # 3. DataFrame missing group column
        df_no_group = pd.DataFrame([{"split": "train", "MSE": 0.05}])
        self.assertTrue(format_side_by_side_metrics(df_no_group).empty)

        # 4. No preferred splits present
        df_diff_split = pd.DataFrame([{"region": "Al2p", "split": "other", "MSE": 0.05}])
        res_diff = format_side_by_side_metrics(df_diff_split, config={"splits": ["train", "test"]})
        self.assertTrue(res_diff.empty)

    def test_format_side_by_side_metrics_unnormalized(self) -> None:
        """Verify format_side_by_side_metrics uses MSE prefix
        when unnormalized.
        """
        df_samples = pd.DataFrame([
            {
                "region": "Al2p",
                "split": "train",
                "MSE": 0.05,
                "peak_error_percentage": 2.0,
                "max_error_percentage": 4.0,
            },
            {
                "region": "Al2p",
                "split": "test",
                "MSE": 0.06,
                "peak_error_percentage": 2.5,
                "max_error_percentage": 4.5,
            },
        ])
        df_res = format_side_by_side_metrics(df_samples)
        expected_cols = [
            "region",
            "MSE_train",
            "peak_error_percentage_train",
            "max_error_percentage_train",
            "MSE_test",
            "peak_error_percentage_test",
            "max_error_percentage_test",
        ]
        self.assertEqual(list(df_res.columns), expected_cols)


class TestMatchPredictionSourceTargetIndices(unittest.TestCase):
    """Test suite for match_prediction_source_target_indices helper."""

    def setUp(self) -> None:
        self.df_orig = pd.DataFrame([
            {
                "spectrum_index": 0,
                "measurement_id": "SRC_1",
                "tool": "J4",
                "region": "Al2p",
                "die": 0,
                "target_measurement_id": "TGT_1",
                "target_spectrum_index": 2,
            },
            {
                "spectrum_index": 1,
                "measurement_id": "SRC_1",
                "tool": "J4",
                "region": "O1s",
                "die": 0,
                "target_measurement_id": "TGT_1",
                "target_spectrum_index": 3,
            },
            {
                "spectrum_index": 2,
                "measurement_id": "TGT_1",
                "tool": "H1",
                "region": "Al2p",
                "die": 0,
                "target_measurement_id": np.nan,
                "target_spectrum_index": np.nan,
            },
            {
                "spectrum_index": 3,
                "measurement_id": "TGT_1",
                "tool": "H1",
                "region": "O1s",
                "die": 0,
                "target_measurement_id": np.nan,
                "target_spectrum_index": np.nan,
            },
            {
                "spectrum_index": 4,
                "measurement_id": "SRC_2",
                "tool": "J4",
                "region": "Al2p",
                "die": 1,
                "target_measurement_id": "TGT_2",
                "target_spectrum_index": 5,
            },
            {
                "spectrum_index": 5,
                "measurement_id": "TGT_2",
                "tool": "H1",
                "region": "Al2p",
                "die": 1,
                "target_measurement_id": np.nan,
                "target_spectrum_index": np.nan,
            },
        ])
        self.df_pred = pd.DataFrame([
            {
                "spectrum_index": 10,
                "region": "Al2p",
                "die": 0,
                "source_measurement_id": "SRC_1",
                "source_spectrum_index": 0,
            },
            {
                "spectrum_index": 11,
                "region": "O1s",
                "die": 0,
                "source_measurement_id": "SRC_1",
                "source_spectrum_index": 1,
            },
            {
                "spectrum_index": 12,
                "region": "Al2p",
                "die": 1,
                "source_measurement_id": "SRC_2",
                "source_spectrum_index": 4,
            },
        ])

    def test_match_all(self) -> None:
        """Verify matching all regions and dies."""
        res = match_prediction_source_target_indices(self.df_orig, self.df_pred)
        self.assertEqual(set(res.keys()), {"Al2p", "O1s"})
        self.assertEqual(res["Al2p"], [(0, 2, 10), (4, 5, 12)])
        self.assertEqual(res["O1s"], [(1, 3, 11)])

    def test_filter_region(self) -> None:
        """Verify region filter."""
        res = match_prediction_source_target_indices(
            self.df_orig, self.df_pred, regions_filter=["Al2p"]
        )
        self.assertEqual(list(res.keys()), ["Al2p"])
        self.assertEqual(len(res["Al2p"]), 2)

    def test_filter_die(self) -> None:
        """Verify die filter with int and string values."""
        res_int = match_prediction_source_target_indices(
            self.df_orig, self.df_pred, die_filter=[0]
        )
        self.assertEqual(res_int["Al2p"], [(0, 2, 10)])

        res_str = match_prediction_source_target_indices(
            self.df_orig, self.df_pred, die_filter=["1"]
        )
        self.assertEqual(res_str["Al2p"], [(4, 5, 12)])
        self.assertNotIn("O1s", res_str)

    def test_fallback_target_measurement_lookup(self) -> None:
        """Verify resolution when target_spectrum_index is missing from source."""
        df_orig_no_tgt_idx = self.df_orig.copy()
        df_orig_no_tgt_idx["target_spectrum_index"] = np.nan
        res = match_prediction_source_target_indices(
            df_orig_no_tgt_idx, self.df_pred
        )
        self.assertEqual(res["Al2p"], [(0, 2, 10), (4, 5, 12)])
        self.assertEqual(res["O1s"], [(1, 3, 11)])

    def test_non_integer_and_float_string_die(self) -> None:
        """Verify matching with non-integer die and float-string representation."""
        df_orig = pd.DataFrame([
            {
                "spectrum_index": 0,
                "measurement_id": "SRC_1",
                "tool": "J4",
                "region": "Al2p",
                "die": "die_0",
                "target_measurement_id": "TGT_1",
                "target_spectrum_index": 1,
            },
            {
                "spectrum_index": 1,
                "measurement_id": "TGT_1",
                "tool": "H1",
                "region": "Al2p",
                "die": "die_0",
                "target_measurement_id": np.nan,
                "target_spectrum_index": np.nan,
            },
            {
                "spectrum_index": 2,
                "measurement_id": "SRC_2",
                "tool": "J4",
                "region": "Al2p",
                "die": 0,
                "target_measurement_id": "TGT_2",
                "target_spectrum_index": 3,
            },
            {
                "spectrum_index": 3,
                "measurement_id": "TGT_2",
                "tool": "H1",
                "region": "Al2p",
                "die": 0,
                "target_measurement_id": np.nan,
                "target_spectrum_index": np.nan,
            },
        ])
        df_pred = pd.DataFrame([
            {
                "spectrum_index": 10,
                "region": "Al2p",
                "die": "die_0",
                "source_measurement_id": "SRC_1",
                "source_spectrum_index": 0,
            },
            {
                "spectrum_index": 11,
                "region": "Al2p",
                "die": "0.0",
                "source_measurement_id": "SRC_2",
                "source_spectrum_index": 2,
            },
        ])
        res = match_prediction_source_target_indices(df_orig, df_pred)
        self.assertEqual(res["Al2p"], [(0, 1, 10), (2, 3, 11)])

        # Filter by string die
        res_filtered = match_prediction_source_target_indices(
            df_orig, df_pred, die_filter=["die_0"]
        )
        self.assertEqual(res_filtered["Al2p"], [(0, 1, 10)])

    def test_dangling_target_spectrum_index_safety(self) -> None:
        """Verify dangling target_spectrum_index falls back or is safely skipped."""
        df_orig = pd.DataFrame([
            {
                "spectrum_index": 0,
                "measurement_id": "SRC_1",
                "tool": "J4",
                "region": "Al2p",
                "die": 0,
                "target_measurement_id": "TGT_1",
                "target_spectrum_index": 999,  # Dangling index
            },
            {
                "spectrum_index": 1,
                "measurement_id": "TGT_1",
                "tool": "H1",
                "region": "Al2p",
                "die": 0,
                "target_measurement_id": np.nan,
                "target_spectrum_index": np.nan,
            },
        ])
        df_pred = pd.DataFrame([
            {
                "spectrum_index": 10,
                "region": "Al2p",
                "die": 0,
                "source_measurement_id": "SRC_1",
                "source_spectrum_index": 0,
            },
        ])
        res = match_prediction_source_target_indices(df_orig, df_pred)
        # Should safely fall back to target index 1 via measurement lookup
        self.assertEqual(res["Al2p"], [(0, 1, 10)])


if __name__ == "__main__":
    unittest.main()
