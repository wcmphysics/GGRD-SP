"""Unit tests for Part 4 (Model Performance Observation and Metrics)."""

from __future__ import annotations

import unittest

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Non-interactive backend for testing

import numpy as np
import pandas as pd

from models.cost import SpectralLoss
from models.dataset import create_regional_datasets
from models.inference import predict_spectra
from models.orchestration import perform_model_training_and_bayesian_optimization
from utility.evaluation import calculate_prediction_metrics
from utility.normalization import normalize_spectral_data
from utility.pairing import pair_source_target_spectra
from utility.pseudo_measurement import generate_pseudo_measurements
from utility.splitting import split_measurement_sessions
from visualization import (
    plot_max_intensity_vs_time,
    plot_normalized_max_intensity_vs_time,
    plot_prediction_comparison,
    plot_regional_spectrum_distribution,
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
        cls.ary_intensity, cls.ary_energy, cls.meta_df = normalize_spectral_data(
            cls.ary_intensity, cls.ary_energy, cls.meta_df
        )
        cls.meta_df, _, cls.session_splits = split_measurement_sessions(
            cls.meta_df,
            {"source_tool": "J4", "target_tool": "H1", "seed": 42},
        )
        reg_ds = create_regional_datasets(
            cls.meta_df,
            cls.ary_intensity,
            cls.ary_energy,
            {"regions": ["Al2p", "Ti2p"], "source_tool": "J4", "target_tool": "H1"},
        )
        criterion = SpectralLoss()
        train_res = perform_model_training_and_bayesian_optimization(
            reg_ds,
            criterion,
            config={
                "model_type": "resnet",
                "train_config": {"epochs": 3, "batch_size": 8, "resnet_channel": 8, "kernel_size": 3},
            },
        )
        data_orig = (cls.ary_intensity, cls.ary_energy, cls.meta_df)
        pred_res = predict_spectra(
            train_res["models"],
            data=data_orig,
            config={"source_tool": "J4", "target_tool": "H1", "session_splits": cls.session_splits},
        )
        cls.pipeline_res = {
            **train_res,
            "predictions": pred_res,
            "session_splits": cls.session_splits,
        }

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
        self.assertIn("MSE_normalized_by_target", df_per_sample.columns)
        self.assertIn("rmse", df_per_sample.columns)
        self.assertIn("peak_err_pct", df_per_sample.columns)
        self.assertIn("region", df_summary.columns)
        self.assertIn("MSE_normalized_by_target_mean", df_summary.columns)
        self.assertIn("MSE_normalized_by_target_std", df_summary.columns)

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
        self.assertIn("MSE_normalized_by_target_mean", df_empty_summary.columns)
        self.assertIn("MSE_normalized_by_target_std", df_empty_summary.columns)

    def test_calculate_prediction_metrics_zero_intensity_eps_safety(self) -> None:
        """Test division by zero protection (eps floor) when target intensity is all zeros."""
        ary_zero_int = np.zeros_like(self.ary_intensity)
        data_orig = (ary_zero_int, self.ary_energy, self.meta_df)
        data_pred = self.pipeline_res["predictions"]

        df_per_sample, df_summary = calculate_prediction_metrics(
            data_orig, data_pred, config={"eps": 1e-3}
        )
        self.assertFalse(df_per_sample.empty)
        self.assertFalse(np.isnan(df_per_sample["MSE_normalized_by_target"]).any())
        self.assertFalse(np.isinf(df_per_sample["MSE_normalized_by_target"]).any())

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


    def test_plot_max_intensity_vs_time(self) -> None:
        """Test plot_max_intensity_vs_time across hue options and filters."""
        for hue_opt in ["tool", "die", "region"]:
            fig, ax = plot_max_intensity_vs_time(
                self.ary_intensity,
                self.meta_df,
                plot_config={"hue": hue_opt, "show": False},
            )
            self.assertIsInstance(fig, plt.Figure)
            self.assertIsInstance(ax, plt.Axes)
            plt.close(fig)

        # Test with filters
        fig, ax = plot_max_intensity_vs_time(
            self.ary_intensity,
            self.meta_df,
            plot_config={"hue": "region", "filters": {"die": 0}, "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

        # Test error on invalid filter or empty df
        with self.assertRaises(KeyError):
            plot_max_intensity_vs_time(
                self.ary_intensity,
                self.meta_df,
                plot_config={"filters": {"nonexistent_col": 1}},
            )
        with self.assertRaises(ValueError):
            plot_max_intensity_vs_time(
                self.ary_intensity,
                self.meta_df.iloc[:0],
            )

    def test_plot_normalized_max_intensity_vs_time(self) -> None:
        """Test plot_normalized_max_intensity_vs_time with die flux and spectrum area modes."""
        # Mode 1: die_total_flux
        fig, ax = plot_normalized_max_intensity_vs_time(
            self.ary_intensity,
            self.meta_df,
            ary_energy=self.ary_energy,
            plot_config={"hue": "tool", "normalization_mode": "die_total_flux", "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        plt.close(fig)

        # Mode 2: spectrum_area
        fig, ax = plot_normalized_max_intensity_vs_time(
            self.ary_intensity,
            self.meta_df,
            ary_energy=self.ary_energy,
            plot_config={"hue": "die", "normalization_mode": "spectrum_area", "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

        # Test ary_energy=None for both modes (summation fallback)
        fig, ax = plot_normalized_max_intensity_vs_time(
            self.ary_intensity,
            self.meta_df,
            ary_energy=None,
            plot_config={"normalization_mode": "die_total_flux", "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

        fig, ax = plot_normalized_max_intensity_vs_time(
            self.ary_intensity,
            self.meta_df,
            ary_energy=None,
            plot_config={"normalization_mode": "spectrum_area", "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

        # Test unsupported normalization_mode raises ValueError
        with self.assertRaises(ValueError):
            plot_normalized_max_intensity_vs_time(
                self.ary_intensity,
                self.meta_df,
                plot_config={"normalization_mode": "unsupported_mode"},
            )

        # Test missing required column in die_total_flux raises KeyError
        with self.assertRaises(KeyError):
            plot_normalized_max_intensity_vs_time(
                self.ary_intensity,
                self.meta_df.drop(columns=["die"]),
                plot_config={"normalization_mode": "die_total_flux"},
            )

        # Test empty dataframe
        with self.assertRaises(ValueError):
            plot_normalized_max_intensity_vs_time(
                self.ary_intensity,
                self.meta_df.iloc[:0],
            )

    def test_plot_max_intensity_vs_time_region_and_die_options(self) -> None:
        """Test plot_max_intensity_vs_time with explicit region and die filtering."""
        # 1. Kwarg filtering with both region and die
        fig, ax = plot_max_intensity_vs_time(
            self.ary_intensity,
            self.meta_df,
            region="Al2p",
            die=0,
            plot_config={"show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIn("Region: Al2p", ax.get_title())
        self.assertIn("Die: 0", ax.get_title())
        plt.close(fig)

        # 2. Config dictionary filtering with region and die
        fig, ax = plot_max_intensity_vs_time(
            self.ary_intensity,
            self.meta_df,
            plot_config={"region": "Al2p", "die": 0, "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIn("Region: Al2p", ax.get_title())
        self.assertIn("Die: 0", ax.get_title())
        plt.close(fig)

        # 3. Only region specified
        fig, ax = plot_max_intensity_vs_time(
            self.ary_intensity,
            self.meta_df,
            region="Al2p",
            plot_config={"show": False},
        )
        self.assertIn("Region: Al2p", ax.get_title())
        self.assertNotIn("Die:", ax.get_title())
        plt.close(fig)

        # 4. Only die specified
        fig, ax = plot_max_intensity_vs_time(
            self.ary_intensity,
            self.meta_df,
            die=0,
            plot_config={"show": False},
        )
        self.assertIn("Die: 0", ax.get_title())
        self.assertNotIn("Region:", ax.get_title())
        plt.close(fig)

        # 5. Non-existent region raises ValueError
        with self.assertRaises(ValueError):
            plot_max_intensity_vs_time(
                self.ary_intensity,
                self.meta_df,
                region="NonExistentRegion",
                plot_config={"show": False},
            )

        # 6. Non-existent die raises ValueError
        with self.assertRaises(ValueError):
            plot_max_intensity_vs_time(
                self.ary_intensity,
                self.meta_df,
                die=999,
                plot_config={"show": False},
            )

        # 7. Missing 'region' column in meta_df raises KeyError
        with self.assertRaises(KeyError):
            plot_max_intensity_vs_time(
                self.ary_intensity,
                self.meta_df.drop(columns=["region"]),
                region="Al2p",
                plot_config={"show": False},
            )

        # 8. Missing 'die' column in meta_df raises KeyError
        with self.assertRaises(KeyError):
            plot_max_intensity_vs_time(
                self.ary_intensity,
                self.meta_df.drop(columns=["die"]),
                die=0,
                plot_config={"show": False},
            )

    def test_plot_normalized_max_intensity_vs_time_region_and_die_options(self) -> None:
        """Test plot_normalized_max_intensity_vs_time with explicit region and die filtering."""
        # 1. Kwarg filtering with both region and die in die_total_flux mode
        fig, ax = plot_normalized_max_intensity_vs_time(
            self.ary_intensity,
            self.meta_df,
            ary_energy=self.ary_energy,
            region="Al2p",
            die=0,
            plot_config={"normalization_mode": "die_total_flux", "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIn("Region: Al2p", ax.get_title())
        self.assertIn("Die: 0", ax.get_title())
        plt.close(fig)

        # 2. Config dictionary filtering with region and die in spectrum_area mode
        fig, ax = plot_normalized_max_intensity_vs_time(
            self.ary_intensity,
            self.meta_df,
            ary_energy=self.ary_energy,
            plot_config={
                "region": "Al2p",
                "die": 0,
                "normalization_mode": "spectrum_area",
                "show": False,
            },
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIn("Region: Al2p", ax.get_title())
        self.assertIn("Die: 0", ax.get_title())
        plt.close(fig)

        # 3. Only region specified
        fig, ax = plot_normalized_max_intensity_vs_time(
            self.ary_intensity,
            self.meta_df,
            region="Al2p",
            plot_config={"show": False},
        )
        self.assertIn("Region: Al2p", ax.get_title())
        self.assertNotIn("Die:", ax.get_title())
        plt.close(fig)

        # 4. Only die specified
        fig, ax = plot_normalized_max_intensity_vs_time(
            self.ary_intensity,
            self.meta_df,
            die=0,
            plot_config={"show": False},
        )
        self.assertIn("Die: 0", ax.get_title())
        self.assertNotIn("Region:", ax.get_title())
        plt.close(fig)

        # 5. Non-existent region raises ValueError
        with self.assertRaises(ValueError):
            plot_normalized_max_intensity_vs_time(
                self.ary_intensity,
                self.meta_df,
                region="NonExistentRegion",
                plot_config={"show": False},
            )

        # 6. Non-existent die raises ValueError
        with self.assertRaises(ValueError):
            plot_normalized_max_intensity_vs_time(
                self.ary_intensity,
                self.meta_df,
                die=999,
                plot_config={"show": False},
            )

        # 7. Missing 'region' column in meta_df raises KeyError
        with self.assertRaises(KeyError):
            plot_normalized_max_intensity_vs_time(
                self.ary_intensity,
                self.meta_df.drop(columns=["region"]),
                region="Al2p",
                plot_config={"show": False},
            )

        # 8. Missing 'die' column in meta_df raises KeyError
        with self.assertRaises(KeyError):
            plot_normalized_max_intensity_vs_time(
                self.ary_intensity,
                self.meta_df.drop(columns=["die"]),
                die=0,
                plot_config={"show": False},
            )

    def test_plot_prediction_comparison_multi_region_subplots_and_colors(self) -> None:
        """Test multi-region subplots, test split default, and strict color scheme."""
        data_orig = (self.ary_intensity, self.ary_energy, self.meta_df)
        data_pred = self.pipeline_res["predictions"]

        # Default region=None -> subplots for all regions
        fig, axes = plot_prediction_comparison(
            data_original=data_orig,
            data_predicted=data_pred,
            config={
                "die": 0,
                "session_splits": self.pipeline_res["session_splits"],
                "show": False,
            },
        )
        self.assertIsInstance(fig, plt.Figure)
        axes_flat = axes.flatten()
        self.assertGreaterEqual(len(axes_flat), 2)

        # Inspect the first active subplot for lines and color scheme:
        # source -> solid black line
        # target -> solid red line
        # transformed -> blue dotted line
        first_ax = axes_flat[0]
        lines = first_ax.get_lines()
        self.assertEqual(len(lines), 3)

        import matplotlib.colors as mcolors
        src_line, tgt_line, pred_line = lines[0], lines[1], lines[2]
        self.assertEqual(mcolors.to_hex(src_line.get_color()), mcolors.to_hex("black"))
        self.assertEqual(src_line.get_linestyle(), "-")

        self.assertEqual(mcolors.to_hex(tgt_line.get_color()), mcolors.to_hex("red"))
        self.assertEqual(tgt_line.get_linestyle(), "-")

        self.assertEqual(mcolors.to_hex(pred_line.get_color()), mcolors.to_hex("blue"))
        self.assertEqual(pred_line.get_linestyle(), ":")

        plt.close(fig)

    def test_plot_prediction_comparison_single_region_colors(self) -> None:
        """Test single-region mode also follows the strict color scheme."""
        data_orig = (self.ary_intensity, self.ary_energy, self.meta_df)
        data_pred = self.pipeline_res["predictions"]

        fig, axes = plot_prediction_comparison(
            data_original=data_orig,
            data_predicted=data_pred,
            config={"region": "Al2p", "die": 0, "show_residual": True, "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        ax_main = axes[0]
        lines = ax_main.get_lines()
        self.assertEqual(len(lines), 3)

        import matplotlib.colors as mcolors
        self.assertEqual(mcolors.to_hex(lines[0].get_color()), mcolors.to_hex("black"))
        self.assertEqual(lines[0].get_linestyle(), "-")

        self.assertEqual(mcolors.to_hex(lines[1].get_color()), mcolors.to_hex("red"))
        self.assertEqual(lines[1].get_linestyle(), "-")

        self.assertEqual(mcolors.to_hex(lines[2].get_color()), mcolors.to_hex("blue"))
        self.assertEqual(lines[2].get_linestyle(), ":")

        plt.close(fig)


    def test_plot_prediction_comparison_multi_region_no_match_raises(self) -> None:
        """Test that multi-region mode raises ValueError if no predicted matches exist."""
        data_orig = (self.ary_intensity, self.ary_energy, self.meta_df)
        ary_p, ary_e, meta_p = self.pipeline_res["predictions"]
        broken_meta_p = meta_p.copy()
        broken_meta_p["source_measurement_id"] = "NON_EXISTENT_ID"
        data_pred_broken = (ary_p, ary_e, broken_meta_p)

        with self.assertRaises(ValueError):
            plot_prediction_comparison(
                data_original=data_orig,
                data_predicted=data_pred_broken,
                config={"die": 0, "show": False},
            )

    def test_plot_regional_spectrum_distribution_defaults(self) -> None:
        """Test default regional distribution plotting across all regions."""
        import matplotlib.colors as mcolors

        data_orig = (self.ary_intensity, self.ary_energy, self.meta_df)
        data_pred = self.pipeline_res["predictions"]

        fig, axes = plot_regional_spectrum_distribution(
            data_original=data_orig,
            data_predicted=data_pred,
            config={"show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(axes), 2)

        twin_axes = [a for a in fig.axes if a not in axes]
        self.assertEqual(len(twin_axes), 2)

        # Verify outer-columns-only axis labeling (Option B)
        self.assertEqual(axes[0].get_ylabel(), "Intensity (% of Target Max)")
        self.assertEqual(axes[1].get_ylabel(), "")
        self.assertEqual(twin_axes[0].get_ylabel(), "")
        self.assertEqual(twin_axes[1].get_ylabel(), "Absolute Error (%)")

        for ax, ax_err in zip(axes, twin_axes):
            # Verify primary curves
            lines = ax.get_lines()
            self.assertEqual(len(lines), 3)
            src_line, tgt_line, pred_line = lines[0], lines[1], lines[2]

            self.assertEqual(
                mcolors.to_hex(src_line.get_color()), mcolors.to_hex("black")
            )
            self.assertEqual(src_line.get_linestyle(), "-")

            self.assertEqual(
                mcolors.to_hex(tgt_line.get_color()), mcolors.to_hex("red")
            )
            self.assertEqual(tgt_line.get_linestyle(), "-")
            # Target mean curve peaks near 100% due to per-die
            # target max normalization.
            self.assertAlmostEqual(
                float(np.max(tgt_line.get_ydata())), 100.0, delta=5.0
            )

            self.assertEqual(
                mcolors.to_hex(pred_line.get_color()), mcolors.to_hex("blue")
            )
            self.assertEqual(pred_line.get_linestyle(), "--")

            # Verify x-axis inversion
            self.assertTrue(ax.xaxis_inverted())

            # Verify twinx error axis
            err_lines = ax_err.get_lines()
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(
                mcolors.to_hex(err_lines[0].get_color()),
                mcolors.to_hex("#EFC934"),
            )
            self.assertEqual(err_lines[0].get_linestyle(), "-")
            self.assertGreaterEqual(ax_err.get_ylim()[0], 0.0)

        # Verify unified figure legend entries
        legend = fig.legends[0] if fig.legends else axes[0].get_legend()
        self.assertIsNotNone(legend)
        texts = [t.get_text() for t in legend.get_texts()]
        self.assertEqual(
            texts,
            [
                "Source (Mean ± 1σ)",
                "Target (Mean ± 1σ)",
                "Predicted (Mean ± 1σ)",
                "Abs Error (Mean ± 1σ)",
            ],
        )

        plt.close(fig)

    def test_plot_regional_spectrum_distribution_die_and_regions_filter(self) -> None:
        """Test regional distribution plotting with die and region filters."""
        data_orig = (self.ary_intensity, self.ary_energy, self.meta_df)
        data_pred = self.pipeline_res["predictions"]

        fig, axes = plot_regional_spectrum_distribution(
            data_original=data_orig,
            data_predicted=data_pred,
            config={"die": 0, "regions": ["Al2p"], "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(axes), 1)
        self.assertIn("Al2p", axes[0].get_title())
        self.assertIn("Die 0", axes[0].get_title())
        plt.close(fig)

    def test_plot_regional_spectrum_distribution_splits(self) -> None:
        """Test regional distribution plotting on train split."""
        data_orig = (self.ary_intensity, self.ary_energy, self.meta_df)
        data_pred = self.pipeline_res["predictions"]

        fig, axes = plot_regional_spectrum_distribution(
            data_original=data_orig,
            data_predicted=data_pred,
            config={"split": "train", "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(axes), 2)
        plt.close(fig)

    def test_plot_regional_spectrum_distribution_empty_raises(self) -> None:
        """Test that invalid split, die, or missing matches raise ValueError."""
        data_orig = (self.ary_intensity, self.ary_energy, self.meta_df)
        data_pred = self.pipeline_res["predictions"]

        with self.assertRaises(ValueError):
            plot_regional_spectrum_distribution(
                data_original=data_orig,
                data_predicted=data_pred,
                config={"split": "non_existent_split", "show": False},
            )

        with self.assertRaises(ValueError):
            plot_regional_spectrum_distribution(
                data_original=data_orig,
                data_predicted=data_pred,
                config={"die": 9999, "show": False},
            )

        ary_p, ary_e, meta_p = data_pred
        broken_meta = meta_p.copy()
        broken_meta["source_measurement_id"] = "UNKNOWN_ID"
        broken_meta["source_spectrum_index"] = np.nan
        with self.assertRaises(ValueError):
            plot_regional_spectrum_distribution(
                data_original=data_orig,
                data_predicted=(ary_p, ary_e, broken_meta),
                config={"show": False},
            )

    def test_plot_regional_spectrum_distribution_clamping_behavior(self) -> None:
        """Verify mean curves remain unclamped while shaded bands clamp lower
        bound to >= 0.
        """
        ary_i_orig = self.ary_intensity.copy()
        ary_i_pred, ary_e_pred, meta_pred = self.pipeline_res["predictions"]
        ary_i_pred_copy = ary_i_pred.copy()

        # Artificially subtract intensity to make some values negative
        ary_i_orig -= 500.0
        ary_i_pred_copy -= 500.0

        fig, axes = plot_regional_spectrum_distribution(
            data_original=(ary_i_orig, self.ary_energy, self.meta_df),
            data_predicted=(ary_i_pred_copy, ary_e_pred, meta_pred),
            config={"die": 0, "regions": ["Al2p"], "show": False},
        )
        ax = axes[0]
        # Mean curves should contain negative values (unclamped)
        lines = ax.get_lines()
        self.assertTrue(any(np.any(line.get_ydata() < 0.0) for line in lines))

        # Shaded bands (collections) must have all y vertices >= 0.0
        for coll in ax.collections:
            for path in coll.get_paths():
                self.assertTrue(np.all(path.vertices[:, 1] >= 0.0))

        # Check twin axis shaded error band as well
        twin_axes = [a for a in fig.axes if a not in axes]
        for twin_ax in twin_axes:
            for coll in twin_ax.collections:
                for path in coll.get_paths():
                    self.assertTrue(np.all(path.vertices[:, 1] >= 0.0))

        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
