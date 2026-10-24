"""Unit tests for the visualization dispatcher module."""

import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from visualization.dispatcher import (
    DEFAULT_PLOT_TOGGLES,
    dispatch_visualizations,
)


class TestVisualizationDispatcher(unittest.TestCase):
    """Test suite verifying selective rendering and toggles in dispatcher."""

    def setUp(self) -> None:
        """Create minimal synthetic pipeline artifacts."""
        n_pts = 20
        self.ary_intensity = np.ones((4, n_pts), dtype=np.float32)
        self.ary_energy = np.tile(
            np.linspace(100.0, 120.0, n_pts, dtype=np.float32), (4, 1)
        )
        self.df_meta = pd.DataFrame(
            [
                {
                    "spectrum_index": i,
                    "measurement_id": f"M_{i}",
                    "tool": "J4" if i < 2 else "J5",
                    "target_tool": "J5" if i < 2 else "J4",
                    "time": f"2026-01-01 0{i}:00:00",
                    "die": 0,
                    "region": "Al2p",
                    "split": "train",
                }
                for i in range(4)
            ]
        )
        self.pipeline_data = {
            "df_meta": self.df_meta,
            "ary_intensity": self.ary_intensity,
            "ary_energy": self.ary_energy,
            "data_original": (
                self.ary_intensity,
                self.ary_energy,
                self.df_meta,
            ),
            "data_predicted": (
                self.ary_intensity,
                self.ary_energy,
                self.df_meta,
            ),
            "training_results": {"histories": {"Al2p": {"train_loss": [1.0], "val_loss": [0.9]}}},
            "df_atomic_percentage_samples": pd.DataFrame({"Al": [50.0], "O": [50.0]}),
        }

    def test_show_plots_false_renders_nothing(self) -> None:
        """Verify show_plots=False returns empty list and skips rendering."""
        rendered = dispatch_visualizations(
            self.pipeline_data, config={"show_plots": False}
        )
        self.assertEqual(rendered, [])

    @patch("visualization.dispatcher.plot_pairing_timeline")
    @patch("visualization.dispatcher.plot_training_history")
    def test_selective_plot_enablement(
        self, mock_train_hist: unittest.mock.MagicMock, mock_pairing: unittest.mock.MagicMock
    ) -> None:
        """Verify enabling specific plots only dispatches the enabled ones."""
        cfg = {
            "show_plots": True,
            "plots": {
                "pairing_timeline": True,
                "training_history": False,
                "gp_die_flux_timeline": False,
                "spectral_noise_analysis": False,
                "gp_hyperparameter_convergence": False,
                "prediction_comparison": False,
                "regional_spectrum_distribution": False,
                "atomic_percentage_distributions": False,
                "atomic_percentage_mae": False,
                "sliding_window_slices": False,
            },
        }
        rendered = dispatch_visualizations(self.pipeline_data, config=cfg)
        self.assertIn("pairing_timeline", rendered)
        self.assertNotIn("training_history", rendered)
        mock_pairing.assert_called_once()
        mock_train_hist.assert_not_called()

    def test_default_plot_toggles_structure(self) -> None:
        """Verify default plot toggles cover expected diagnostic figures."""
        expected_keys = [
            "gp_die_flux_timeline",
            "pairing_timeline",
            "flux_and_averaged_intensity",
            "max_intensity",
            "normalized_max_intensity",
            "tool_comparison",
            "shirley_background",
            "spectral_noise_analysis",
            "gp_hyperparameter_convergence",
            "training_history",
            "prediction_comparison",
            "regional_spectrum_distribution",
            "atomic_percentage_distributions",
            "atomic_percentage_mae",
            "sliding_window_slices",
            "denoise_spectrum_comparison",
            "denoise_atomic_percentage_mae",
        ]
        for k in expected_keys:
            self.assertIn(k, DEFAULT_PLOT_TOGGLES)

    @patch("visualization.dispatcher.plot_denoise_atomic_percentage_mae")
    @patch("visualization.dispatcher.plot_denoise_spectrum_comparison")
    def test_denoise_plots_dispatched(
        self,
        mock_denoise_spec: unittest.mock.MagicMock,
        mock_denoise_at: unittest.mock.MagicMock,
    ) -> None:
        """Verify denoise comparison plots dispatch when toggles active."""
        data = {
            **self.pipeline_data,
            "denoise_intensity_pair": (self.ary_intensity, self.ary_intensity),
            "df_denoise_atomic_samples": pd.DataFrame(
                {"region": ["Al2p"], "abs_diff_atomic_percentage": [0.1]}
            ),
        }
        cfg = {
            "show_plots": True,
            "plots": {
                k: k
                in (
                    "denoise_spectrum_comparison",
                    "denoise_atomic_percentage_mae",
                )
                for k in DEFAULT_PLOT_TOGGLES
            },
        }
        rendered = dispatch_visualizations(data, config=cfg)
        self.assertIn("denoise_spectrum_comparison", rendered)
        self.assertIn("denoise_atomic_percentage_mae", rendered)
        mock_denoise_spec.assert_called_once()
        mock_denoise_at.assert_called_once()


if __name__ == "__main__":
    unittest.main()
