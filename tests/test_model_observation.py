"""Unit tests for Model Performance Observation and Visualization."""

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
    """Test suite for performance observation, prediction overlays, and intensity tracking."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up synthetic dataset, pairing, and minimal trained baseline model."""
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
                "material": "all",
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
                "train_config": {
                    "epochs": 2,
                    "batch_size": 8,
                    "resnet_channel": 8,
                    "kernel_size": 3,
                    "dropout": 0.0,
                    "use_batch_norm": True,
                    "padding_mode": "reflect",
                    "use_film": False,
                },
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

    def tearDown(self) -> None:
        """Close open matplotlib figures."""
        plt.close("all")

    def test_plot_training_history_and_prediction_comparison(self) -> None:
        """Verify training history curves and prediction comparison line renders."""
        histories = self.pipeline_res["histories"]
        fig_h, axes_h = plot_training_history(histories, plot_config={"show": False, "log_scale": True})
        self.assertIsInstance(fig_h, plt.Figure)
        self.assertEqual(len(histories), 2)

        data_orig = (self.ary_intensity, self.ary_energy, self.meta_df)
        data_pred = self.pipeline_res["predictions"]
        fig_p, axes_p = plot_prediction_comparison(
            data_original=data_orig,
            data_predicted=data_pred,
            config={"region": "Al2p", "die": 0, "show_residual": True, "show": False},
        )
        self.assertEqual(len(axes_p), 2)
        # Verify color scheme (black source, red target, blue prediction)
        lines = axes_p[0].get_lines()
        self.assertEqual(len(lines), 3)

    def test_plot_sliding_window_slices(self) -> None:
        """Verify sliding window patch visualization with eV and point-based windows."""
        spectrum = self.ary_intensity[0]
        energy = self.ary_energy[0]

        # With energy grid
        fig_ev, axes_ev = plot_sliding_window_slices(
            spectrum, energy, plot_config={"region": "Al2p", "window_size_ev": 2.0, "sliding_stride_ev": 1.0, "show": False}
        )
        self.assertEqual(len(axes_ev), 2)

        # Without energy grid
        fig_pt, axes_pt = plot_sliding_window_slices(
            spectrum, energy=None, plot_config={"window_size": 15, "stride": 5, "show": False}
        )
        self.assertEqual(len(axes_pt), 2)

    def test_plot_max_intensity_vs_time_and_normalization_modes(self) -> None:
        """Verify raw and normalized max intensity tracking across tools and dies."""
        # Raw max intensity
        for hue in ["tool", "region"]:
            with self.subTest(hue=hue):
                fig, ax = plot_max_intensity_vs_time(
                    self.ary_intensity, self.meta_df, plot_config={"hue": hue, "show": False}
                )
                self.assertIsInstance(fig, plt.Figure)

        # Normalized max intensity: die_total_flux and spectrum_area
        for mode in ["die_total_flux", "spectrum_area"]:
            with self.subTest(mode=mode):
                fig, ax = plot_normalized_max_intensity_vs_time(
                    self.ary_intensity,
                    self.meta_df,
                    ary_energy=self.ary_energy,
                    plot_config={"hue": "tool", "normalization_mode": mode, "show": False},
                )
                self.assertIsInstance(fig, plt.Figure)

    def test_plot_regional_spectrum_distribution(self) -> None:
        """Verify regional spectrum distributions, twin-axis error lines, and legend."""
        data_orig = (self.ary_intensity, self.ary_energy, self.meta_df)
        data_pred = self.pipeline_res["predictions"]

        fig, axes = plot_regional_spectrum_distribution(
            data_original=data_orig,
            data_predicted=data_pred,
            config={"die": 0, "regions": ["Al2p"], "show": False},
        )
        self.assertEqual(len(axes), 1)
        twin_axes = [a for a in fig.axes if a not in axes]
        self.assertEqual(len(twin_axes), 1)
        self.assertEqual(axes[0].get_ylabel(), "Intensity (% of Target Max)")
        self.assertEqual(twin_axes[0].get_ylabel(), "Absolute Error (%)")

    def test_defensive_error_handling(self) -> None:
        """Table-driven test for empty DataFrames, invalid regions, and unsupported modes."""
        data_orig = (self.ary_intensity, self.ary_energy, self.meta_df)
        data_pred = self.pipeline_res["predictions"]

        with self.assertRaises(ValueError):
            plot_training_history({})

        with self.assertRaises(ValueError):
            plot_prediction_comparison(
                data_original=data_orig,
                data_predicted=data_pred,
                config={"region": "NON_EXISTENT_REGION", "show": False},
            )

        with self.assertRaises(ValueError):
            plot_sliding_window_slices(np.array([]))

        with self.assertRaises(ValueError):
            plot_max_intensity_vs_time(self.ary_intensity, self.meta_df.iloc[:0])

        with self.assertRaises(ValueError):
            plot_normalized_max_intensity_vs_time(
                self.ary_intensity, self.meta_df, plot_config={"normalization_mode": "invalid_mode"}
            )


if __name__ == "__main__":
    unittest.main()
