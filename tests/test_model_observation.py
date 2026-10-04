"""Unit tests for Part 4 (Model Performance Observation and Metrics)."""

from __future__ import annotations

import unittest

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Non-interactive backend for testing

import numpy as np
import pandas as pd

from models.orchestration import run_baseline_pipeline
from utility.pairing import pair_source_target_spectra
from utility.pseudo_measurement import generate_pseudo_measurements
from utility.visualization import (
    calculate_prediction_metrics,
    plot_prediction_comparison,
    plot_sliding_window_slices,
    plot_training_history,
)


class TestModelPerformanceObservation(unittest.TestCase):
    """Test cases for Part 4 performance observation plots and metric calculations."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up synthetic dataset, pairing, and trained baseline model."""
        cls.ary_intensity, cls.ary_energy, cls.meta_df = generate_pseudo_measurements(
            {
                "material": "NMG",
                "regions": ["Al2p", "Ti2p"],
                "n_points": 50,
                "measurements_per_tool": {"J4": 6, "H1": 6},
                "interval_hours_range": (1.0, 4.0),
                "seed": 42,
            }
        )
        cls.meta_df = pair_source_target_spectra(
            cls.meta_df,
            {
                "source_tool": "J4",
                "target_tool": "H1",
                "time_threshold_hours": 12.0,
            },
        )
        cls.pipeline_res = run_baseline_pipeline(
            cls.meta_df,
            cls.ary_intensity,
            cls.ary_energy,
            config={
                "regions": ["Al2p", "Ti2p"],
                "source_tool": "J4",
                "target_tool": "H1",
                "use_bayesian_opt": False,
                "train_config": {"epochs": 3, "batch_size": 8, "hidden_channels": 8, "kernel_size": 3},
                "predict_source": True,
            },
        )

    def test_plot_training_history(self) -> None:
        """Test training history curve plotting."""
        histories = self.pipeline_res["histories"]
        fig, axes = plot_training_history(histories, plot_config={"show": False, "log_scale": True})
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(histories), 2)
        plt.close(fig)

    def test_plot_prediction_comparison(self) -> None:
        """Test prediction comparison overlay and residual plotting."""
        data_orig = (self.ary_intensity, self.ary_energy, self.meta_df)
        data_pred = self.pipeline_res["predictions"]

        fig, axes = plot_prediction_comparison(
            data_original=data_orig,
            data_predicted=data_pred,
            config={"region": "Al2p", "die": 0, "show_residual": True, "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(axes), 2)  # main and residual
        plt.close(fig)

    def test_calculate_prediction_metrics(self) -> None:
        """Test calculation of quantitative prediction metrics."""
        data_orig = (self.ary_intensity, self.ary_energy, self.meta_df)
        data_pred = self.pipeline_res["predictions"]

        df_per_sample, df_summary = calculate_prediction_metrics(data_orig, data_pred)

        self.assertFalse(df_per_sample.empty)
        self.assertFalse(df_summary.empty)
        self.assertIn("normalized_mse", df_per_sample.columns)
        self.assertIn("rmse", df_per_sample.columns)
        self.assertIn("peak_err_pct", df_per_sample.columns)
        self.assertIn("region", df_summary.columns)
        self.assertIn("normalized_mse_mean", df_summary.columns)
        self.assertIn("normalized_mse_std", df_summary.columns)

    def test_plot_training_history_empty_raises(self) -> None:
        """Test that passing empty history dictionary raises ValueError."""
        with self.assertRaises(ValueError):
            plot_training_history({})

    def test_plot_prediction_comparison_no_residual(self) -> None:
        """Test plot_prediction_comparison with show_residual=False returns single Axes."""
        data_orig = (self.ary_intensity, self.ary_energy, self.meta_df)
        data_pred = self.pipeline_res["predictions"]

        fig, axes = plot_prediction_comparison(
            data_original=data_orig,
            data_predicted=data_pred,
            config={"region": "Al2p", "die": 0, "show_residual": False, "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(axes), 1)
        plt.close(fig)

    def test_plot_prediction_comparison_invalid_inputs(self) -> None:
        """Test that plot_prediction_comparison raises ValueError for nonexistent region or session."""
        data_orig = (self.ary_intensity, self.ary_energy, self.meta_df)
        data_pred = self.pipeline_res["predictions"]

        with self.assertRaises(ValueError):
            plot_prediction_comparison(
                data_original=data_orig,
                data_predicted=data_pred,
                config={"measurement_id": "NON_EXISTENT_SESSION", "show": False},
            )

        with self.assertRaises(ValueError):
            plot_prediction_comparison(
                data_original=data_orig,
                data_predicted=data_pred,
                config={"region": "NON_EXISTENT_REGION", "show": False},
            )

    def test_calculate_prediction_metrics_empty_and_filter(self) -> None:
        """Test metrics calculation with region filtering and empty result handling."""
        data_orig = (self.ary_intensity, self.ary_energy, self.meta_df)
        data_pred = self.pipeline_res["predictions"]

        # Filter only Al2p
        df_per_sample, df_summary = calculate_prediction_metrics(
            data_orig, data_pred, config={"regions": ["Al2p"]}
        )
        self.assertTrue((df_per_sample["region"] == "Al2p").all())
        self.assertEqual(len(df_summary), 1)

        # Non-matching region -> empty DataFrames with expected schemas
        df_empty_sample, df_empty_summary = calculate_prediction_metrics(
            data_orig, data_pred, config={"regions": ["NON_EXISTENT_REGION"]}
        )
        self.assertTrue(df_empty_sample.empty)
        self.assertTrue(df_empty_summary.empty)
        self.assertIn("normalized_mse_mean", df_empty_summary.columns)
        self.assertIn("normalized_mse_std", df_empty_summary.columns)

    def test_calculate_prediction_metrics_zero_intensity_eps_safety(self) -> None:
        """Test division by zero protection (eps floor) when target intensity is all zeros."""
        ary_zero_int = np.zeros_like(self.ary_intensity)
        data_orig = (ary_zero_int, self.ary_energy, self.meta_df)
        data_pred = self.pipeline_res["predictions"]

        df_per_sample, df_summary = calculate_prediction_metrics(
            data_orig, data_pred, config={"eps": 1e-3}
        )
        self.assertFalse(df_per_sample.empty)
        self.assertFalse(np.isnan(df_per_sample["normalized_mse"]).any())
        self.assertFalse(np.isinf(df_per_sample["normalized_mse"]).any())

    def test_plot_sliding_window_slices_with_energy(self) -> None:
        """Test plot_sliding_window_slices with binding energy and eV window."""
        spectrum = self.ary_intensity[0]
        energy = self.ary_energy[0]
        fig, axes = plot_sliding_window_slices(
            spectrum,
            energy,
            plot_config={"region": "Al2p", "window_size_ev": 2.0, "sliding_stride_ev": 1.0, "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(axes), 2)
        plt.close(fig)

    def test_plot_sliding_window_slices_without_energy(self) -> None:
        """Test plot_sliding_window_slices without energy grid (point index mode)."""
        spectrum = self.ary_intensity[0]
        fig, axes = plot_sliding_window_slices(
            spectrum,
            energy=None,
            plot_config={"window_size": 15, "stride": 5, "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(axes), 2)
        plt.close(fig)

    def test_plot_sliding_window_slices_offset_patches(self) -> None:
        """Test plot_sliding_window_slices with waterfall vertical offset."""
        spectrum = self.ary_intensity[0]
        fig, axes = plot_sliding_window_slices(
            spectrum,
            plot_config={"window_size": 10, "stride": 5, "offset_patches": True, "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_sliding_window_slices_invalid_inputs(self) -> None:
        """Test ValueError when spectrum is empty or energy length mismatches."""
        with self.assertRaises(ValueError):
            plot_sliding_window_slices(np.array([]))
        with self.assertRaises(ValueError):
            plot_sliding_window_slices(np.ones(20), energy=np.ones(15))


if __name__ == "__main__":
    unittest.main()
