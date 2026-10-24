"""Unit tests for Stage 7 neural network training and Bayesian optimization orchestration."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from models.cost import SpectralLoss, format_loss_log10
from models.dataset import SpectrumPatchDataset, create_regional_datasets
from models.inference import predict_spectra
from models.orchestration import (
    _resolve_region_config,
    instantiate_model,
    perform_model_training_and_bayesian_optimization,
)
from models.resnet import ResNet1D
from models.unet import ConventionalUNet1D, ResidualUNet1D
from utility.normalization import normalize_spectral_data
from utility.pairing import pair_source_target_spectra
from utility.pseudo_measurement import generate_pseudo_measurements
from utility.splitting import split_measurement_sessions


class TestOrchestrationPipeline(unittest.TestCase):
    """Test suite for perform_model_training_and_bayesian_optimization across models and patches."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate, normalize, pair, and split lightweight synthetic data."""
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
        ary_i, ary_e, df_meta = generate_pseudo_measurements(pseudo_cfg)
        df_meta = pair_source_target_spectra(
            df_meta,
            config={
                "source_tool": "J4",
                "target_tool": "H1",
                "time_threshold_hours": 12.0,
                "material": "all",
            },
        )
        ary_i, ary_e, df_meta = normalize_spectral_data(ary_i, ary_e, df_meta)
        df_meta, _, session_splits = split_measurement_sessions(
            df_meta, config={"source_tool": "J4", "target_tool": "H1", "seed": 42}
        )
        unique_tools = sorted(
            set(df_meta["tool"].dropna().astype(str).unique())
            | set(df_meta["target_tool"].dropna().astype(str).unique())
        )
        tool_map = {tool_name: idx for idx, tool_name in enumerate(unique_tools)}
        df_meta["source_tool_index"] = df_meta["tool"].map(tool_map)
        df_meta["target_tool_index"] = df_meta["target_tool"].map(tool_map)

        cls.ary_intensity = ary_i
        cls.ary_energy = ary_e
        cls.df_meta = df_meta
        cls.session_splits = session_splits

    def test_instantiate_model(self) -> None:
        """Verify model instantiation across supported types and error handling."""
        m_res = instantiate_model("resnet", seq_len=30, config={"resnet_channel": 8})
        self.assertIsInstance(m_res, ResNet1D)

        m_res_u = instantiate_model(
            "residual_unet", seq_len=32, config={"residual_unet_channel": 4, "depth": 2}
        )
        self.assertIsInstance(m_res_u, ResidualUNet1D)
        self.assertTrue(m_res_u.residual)

        m_conv_u = instantiate_model(
            "unet", seq_len=32, config={"unet_channel": 4, "depth": 2}
        )
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

    def test_train_resnet_full_spectrum(self) -> None:
        """Verify Stage 7 training for ResNet in full-spectrum mode."""
        ds_cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p"],
            "use_sliding_window": False,
        }
        regional_datasets = create_regional_datasets(
            self.df_meta, self.ary_intensity, self.ary_energy, ds_cfg
        )
        criterion = SpectralLoss()

        train_cfg = {
            "model_type": "resnet",
            "train_config": {"epochs": 2, "batch_size": 4, "verbose": True},
        }
        res = perform_model_training_and_bayesian_optimization(
            regional_datasets, criterion, config=train_cfg
        )
        self.assertIn("Al2p", res["models"])
        self.assertIsInstance(res["models"]["Al2p"], ResNet1D)
        self.assertIn("Al2p", res["val_evaluation"])

        # Stage 8 prediction
        pred_cfg = {"source_tool": "J4", "target_tool": "H1"}
        pred_i, pred_e, pred_m = predict_spectra(
            res["models"], data=(self.ary_intensity, self.ary_energy, self.df_meta), config=pred_cfg
        )
        self.assertEqual(pred_i.shape[1], 40)
        self.assertTrue(np.all(pred_i >= 0.0))

    def test_train_resnet_sliding_window(self) -> None:
        """Verify Stage 7 training for ResNet with sliding window patch augmentation."""
        ds_cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p"],
            "use_sliding_window": True,
            "window_size": 15,
            "stride": 7,
        }
        regional_datasets = create_regional_datasets(
            self.df_meta, self.ary_intensity, self.ary_energy, ds_cfg
        )
        self.assertIsInstance(regional_datasets["Al2p"]["dataset_train"], SpectrumPatchDataset)

        criterion = SpectralLoss()
        train_cfg = {
            "model_type": "resnet",
            "train_config": {"epochs": 2, "batch_size": 8, "verbose": True},
        }
        res = perform_model_training_and_bayesian_optimization(
            regional_datasets, criterion, config=train_cfg
        )
        self.assertIn("Al2p", res["models"])
        model_entry = res["models"]["Al2p"]
        self.assertIsInstance(model_entry, tuple)
        self.assertIsInstance(model_entry[0], ResNet1D)
        self.assertEqual(model_entry[1], 15)  # window_size
        self.assertEqual(model_entry[2], 7)   # stride

        pred_cfg = {"source_tool": "J4", "target_tool": "H1"}
        pred_i, _, _ = predict_spectra(
            res["models"], data=(self.ary_intensity, self.ary_energy, self.df_meta), config=pred_cfg
        )
        self.assertEqual(pred_i.shape[1], 40)
        self.assertTrue(np.all(pred_i >= 0.0))

    def test_train_residual_unet_pipelines(self) -> None:
        """Verify Residual U-Net training in both full and sliding window modes."""
        ds_cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Ti2p"],
            "use_sliding_window": False,
        }
        reg_ds_full = create_regional_datasets(
            self.df_meta, self.ary_intensity, self.ary_energy, ds_cfg
        )
        criterion = SpectralLoss()

        train_cfg = {
            "model_type": "residual_unet",
            "train_config": {"epochs": 2, "batch_size": 4, "residual_unet_channel": 4, "depth": 2},
        }
        res_full = perform_model_training_and_bayesian_optimization(
            reg_ds_full, criterion, config=train_cfg
        )
        self.assertIsInstance(res_full["models"]["Ti2p"], ResidualUNet1D)

        # Sliding window
        ds_sw_cfg = dict(ds_cfg)
        ds_sw_cfg["use_sliding_window"] = True
        ds_sw_cfg["window_size"] = 16
        ds_sw_cfg["stride"] = 8
        reg_ds_sw = create_regional_datasets(
            self.df_meta, self.ary_intensity, self.ary_energy, ds_sw_cfg
        )
        res_sw = perform_model_training_and_bayesian_optimization(
            reg_ds_sw, criterion, config=train_cfg
        )
        self.assertIsInstance(res_sw["models"]["Ti2p"][0], ResidualUNet1D)

    def test_train_conventional_unet_pipeline(self) -> None:
        """Verify Conventional U-Net (no shortcut) pipeline execution."""
        ds_cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p"],
            "use_sliding_window": False,
        }
        regional_datasets = create_regional_datasets(
            self.df_meta, self.ary_intensity, self.ary_energy, ds_cfg
        )
        criterion = SpectralLoss()
        train_cfg = {
            "model_type": "unet",
            "train_config": {"epochs": 2, "batch_size": 4, "unet_channel": 4, "depth": 2},
        }
        res = perform_model_training_and_bayesian_optimization(
            regional_datasets, criterion, config=train_cfg
        )
        self.assertIsInstance(res["models"]["Al2p"], ConventionalUNet1D)
        self.assertFalse(res["models"]["Al2p"].residual)

    def test_train_with_bayesian_optimization(self) -> None:
        """Verify Stage 7 Ax Bayesian hyperparameter optimization with separated bayesian_opt_config."""
        ds_cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p"],
            "use_sliding_window": True,
            "window_size": 15,
            "stride": 7,
        }
        regional_datasets = create_regional_datasets(
            self.df_meta, self.ary_intensity, self.ary_energy, ds_cfg
        )
        criterion = SpectralLoss()
        bo_cfg = {
            "use_bayesian_opt": True,
            "num_trials": 2,
            "epochs_per_trial": 2,
            "window_size_choices": [15, 21],
            "kernel_sizes": [3],
            "resnet_channels": [8],
        }
        train_cfg = {
            "model_type": "resnet",
            "train_config": {"epochs": 2, "batch_size": 4},
        }
        res = perform_model_training_and_bayesian_optimization(
            regional_datasets, criterion, config=train_cfg, bayesian_opt_config=bo_cfg
        )
        self.assertIn("Al2p", res["bayesian_opt_results"])
        self.assertIn("best_parameters", res["bayesian_opt_results"]["Al2p"])
        self.assertIn("Al2p", res["models"])
        model_entry = res["models"]["Al2p"]
        self.assertIsInstance(model_entry, tuple)
        self.assertIn(model_entry[1], [15, 21])

    def test_resolve_region_config(self) -> None:
        """Verify hierarchical configuration resolution and per-region overrides."""
        global_cfg = {
            "window_size_ev": 4.0,
            "sliding_stride_ev": 2.0,
            "train_config": {"batch_size": 32, "learning_rate": 1e-4, "depth": 3},
            "region_configs": {
                "Al2p": {
                    "window_size_ev": 2.5,
                    "sliding_stride_ev": 0.8,
                    "batch_size": 16,
                    "learning_rate": 2e-3,
                },
                "Ti2p": {
                    "depth": 2,
                    "bayesian_opt_config": {"batch_sizes": [8, 16, 32]},
                },
            },
        }
        bo_cfg = {"num_trials": 3, "batch_sizes": [16, 32]}

        sw_al, tr_al, bo_al = _resolve_region_config(global_cfg, "Al2p", bayesian_opt_config=bo_cfg)
        self.assertEqual(sw_al["window_size_ev"], 2.5)
        self.assertEqual(sw_al["sliding_stride_ev"], 0.8)
        self.assertEqual(tr_al["batch_size"], 16)
        self.assertEqual(tr_al["learning_rate"], 2e-3)
        self.assertEqual(tr_al["depth"], 3)
        self.assertEqual(bo_al["batch_sizes"], [16, 32])

        sw_ti, tr_ti, bo_ti = _resolve_region_config(global_cfg, "Ti2p", bayesian_opt_config=bo_cfg)
        self.assertEqual(sw_ti["window_size_ev"], 4.0)
        self.assertEqual(tr_ti["batch_size"], 32)
        self.assertEqual(tr_ti["depth"], 2)
        self.assertEqual(bo_ti["batch_sizes"], [8, 16, 32])

        sw_c, tr_c, bo_c = _resolve_region_config(global_cfg, "C1s", bayesian_opt_config=bo_cfg)
        self.assertEqual(sw_c["window_size_ev"], 4.0)
        self.assertEqual(tr_c["batch_size"], 32)
        self.assertEqual(tr_c["learning_rate"], 1e-4)
        self.assertEqual(bo_c["num_trials"], 3)

    def test_orchestration_validation_errors(self) -> None:
        """Verify orchestration raises errors for invalid window_size, batch_size, and kernel_size."""
        ds_cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p"],
            "use_sliding_window": True,
            "window_size": 15,
            "stride": 7,
        }
        regional_datasets = create_regional_datasets(
            self.df_meta, self.ary_intensity, self.ary_energy, ds_cfg
        )
        criterion = SpectralLoss()

        # Non-integer window_size in train_config when BO is off -> ValueError
        with self.assertRaises(ValueError):
            perform_model_training_and_bayesian_optimization(
                regional_datasets,
                criterion,
                config={"model_type": "resnet", "train_config": {"window_size": [15]}},
            )

        # kernel_size > window_size when BO is off -> ValueError
        with self.assertRaises(ValueError):
            perform_model_training_and_bayesian_optimization(
                regional_datasets,
                criterion,
                config={"model_type": "resnet", "train_config": {"window_size": 5, "kernel_size": 7}},
            )

        # Singular window_size in bayesian_opt_config when BO is on -> KeyError
        with self.assertRaises(KeyError):
            perform_model_training_and_bayesian_optimization(
                regional_datasets,
                criterion,
                config={"model_type": "resnet", "train_config": {"epochs": 1}},
                bayesian_opt_config={"use_bayesian_opt": True, "window_size": 15},
            )


if __name__ == "__main__":
    unittest.main()

