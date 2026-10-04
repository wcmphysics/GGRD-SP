"""Unit tests for the unified 8-step spectral transformation orchestration pipeline."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
import torch

from models.cost import NormalizedMSELoss, format_loss_log10
from models.orchestration import (
    instantiate_model,
    run_residual_unet_pipeline,
    run_resnet_pipeline,
    run_spectral_pipeline,
    run_unet_pipeline,
)
from models.resnet import ResNet1D
from models.unet import ConventionalUNet1D, ResidualUNet1D, UNet1D
from utility.pairing import pair_source_target_spectra
from utility.pseudo_measurement import generate_pseudo_measurements


class TestOrchestrationPipeline(unittest.TestCase):
    """Test suite for the unified 8-step pipeline across models and sliding window toggles."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate a lightweight synthetic dataset for pipeline execution."""
        pseudo_cfg = {
            "material": "NMG",
            "regions": ["Al2p", "Ti2p"],
            "n_points": 40,
            "measurements_per_tool": {"J4": 8, "H1": 8},
            "interval_hours_range": (1.0, 4.0),
            "tool_offsets": {
                "J4": {"shift_ev": 0.0, "scale": 1000.0},
                "H1": {"shift_ev": 0.5, "scale": 1200.0},
            },
            "seed": 123,
        }
        ary_intensity, ary_energy, meta_df = generate_pseudo_measurements(pseudo_cfg)
        meta_df = pair_source_target_spectra(
            meta_df,
            config={"source_tool": "J4", "target_tool": "H1", "time_threshold_hours": 12.0},
        )
        cls.ary_intensity = ary_intensity
        cls.ary_energy = ary_energy
        cls.meta_df = meta_df

    def test_instantiate_model(self) -> None:
        """Verify model instantiation across supported types and error handling."""
        m_res = instantiate_model("resnet", seq_len=30, config={"hidden_channels": 8})
        self.assertIsInstance(m_res, ResNet1D)

        m_res_u = instantiate_model("residual_unet", seq_len=32, config={"base_channels": 4, "depth": 2})
        self.assertIsInstance(m_res_u, ResidualUNet1D)
        self.assertTrue(m_res_u.residual)

        m_conv_u = instantiate_model("unet", seq_len=32, config={"base_channels": 4, "depth": 2})
        self.assertIsInstance(m_conv_u, ConventionalUNet1D)
        self.assertFalse(m_conv_u.residual)

        with self.assertRaises(ValueError):
            instantiate_model("unknown_arch", seq_len=30)

    def test_format_loss_log10(self) -> None:
        """Verify log10 loss string formatting."""
        s1 = format_loss_log10(0.001)
        self.assertIn("log10: -3.0000", s1)

        s2 = format_loss_log10(0.0)
        self.assertIn("log10: N/A", s2)

        s3 = format_loss_log10(-0.5)
        self.assertIn("log10: N/A", s3)

    def test_run_resnet_full_spectrum(self) -> None:
        """Verify 8-step pipeline execution for ResNet in full-spectrum mode."""
        cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p"],
            "train_config": {"epochs": 2, "batch_size": 4, "verbose": True},
            "predict_source": True,
        }
        res = run_spectral_pipeline(
            data=(self.ary_intensity, self.ary_energy, self.meta_df),
            model_type="resnet",
            use_sliding_window=False,
            config=cfg,
        )
        self.assertIn("Al2p", res["models"])
        self.assertIsInstance(res["models"]["Al2p"], ResNet1D)
        self.assertIn("Al2p", res["evaluation"])
        self.assertIn("Al2p", res["test_evaluation"])
        self.assertIsNotNone(res["predictions"])

        pred_i, pred_e, pred_m = res["predictions"]
        self.assertEqual(pred_i.shape[1], 40)
        self.assertTrue(np.all(pred_i >= 0.0))  # zero clamping verified

    def test_run_resnet_sliding_window(self) -> None:
        """Verify 8-step pipeline execution for ResNet with sliding window patch augmentation."""
        cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p"],
            "window_size": 15,
            "stride": 7,
            "train_config": {"epochs": 2, "batch_size": 8, "verbose": True},
            "predict_source": True,
        }
        res = run_spectral_pipeline(
            data=(self.ary_intensity, self.ary_energy, self.meta_df),
            model_type="resnet",
            use_sliding_window=True,
            config=cfg,
        )
        self.assertIn("Al2p", res["models"])
        model_entry = res["models"]["Al2p"]
        self.assertIsInstance(model_entry, tuple)
        self.assertIsInstance(model_entry[0], ResNet1D)
        self.assertEqual(model_entry[1], 15)  # window_size
        self.assertEqual(model_entry[2], 7)   # stride

        pred_i, _, _ = res["predictions"]
        self.assertEqual(pred_i.shape[1], 40)
        self.assertTrue(np.all(pred_i >= 0.0))

    def test_run_residual_unet_pipelines(self) -> None:
        """Verify Residual U-Net pipeline execution in both full and sliding window modes."""
        # Full spectrum
        cfg_full = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Ti2p"],
            "train_config": {"epochs": 2, "batch_size": 4, "base_channels": 4, "depth": 2},
            "predict_source": True,
        }
        res_full = run_residual_unet_pipeline(
            meta_df=self.meta_df,
            ary_intensity=self.ary_intensity,
            ary_energy=self.ary_energy,
            config=cfg_full,
        )
        self.assertIsInstance(res_full["models"]["Ti2p"], ResidualUNet1D)

        # Sliding window
        cfg_sw = dict(cfg_full)
        cfg_sw["use_sliding_window"] = True
        cfg_sw["window_size"] = 16
        cfg_sw["stride"] = 8
        res_sw = run_residual_unet_pipeline(
            meta_df=self.meta_df,
            ary_intensity=self.ary_intensity,
            ary_energy=self.ary_energy,
            config=cfg_sw,
        )
        self.assertIsInstance(res_sw["models"]["Ti2p"][0], ResidualUNet1D)

    def test_run_conventional_unet_pipeline(self) -> None:
        """Verify Conventional U-Net (no shortcut) pipeline execution."""
        cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p"],
            "train_config": {"epochs": 2, "batch_size": 4, "base_channels": 4, "depth": 2},
            "predict_source": True,
        }
        res = run_unet_pipeline(
            meta_df=self.meta_df,
            ary_intensity=self.ary_intensity,
            ary_energy=self.ary_energy,
            config=cfg,
        )
        self.assertIsInstance(res["models"]["Al2p"], ConventionalUNet1D)
        self.assertFalse(res["models"]["Al2p"].residual)

    def test_run_spectral_pipeline_with_bayesian_opt(self) -> None:
        """Verify unified pipeline execution with Step 6 Ax Bayesian optimization enabled."""
        cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p"],
            "use_bayesian_opt": True,
            "bayesian_opt_config": {
                "num_trials": 2,
                "epochs_per_trial": 2,
                "window_size_choices": [15, 21],
                "kernel_sizes": [3],
                "hidden_channels": [8],
            },
            "train_config": {"epochs": 2, "batch_size": 4},
            "predict_source": True,
        }
        res = run_spectral_pipeline(
            data=(self.ary_intensity, self.ary_energy, self.meta_df),
            model_type="resnet",
            use_sliding_window=True,
            config=cfg,
        )
        self.assertIn("Al2p", res["bayesian_opt_results"])
        self.assertIn("best_parameters", res["bayesian_opt_results"]["Al2p"])
        self.assertIn("Al2p", res["models"])
        model_entry = res["models"]["Al2p"]
        self.assertIsInstance(model_entry, tuple)
        # Verify window_size is one of the searched choices
        self.assertIn(model_entry[1], [15, 21])


if __name__ == "__main__":
    unittest.main()
