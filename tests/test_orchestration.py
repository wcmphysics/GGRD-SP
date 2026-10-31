"""Unit tests for Stage 7 training and BoTorch orchestration."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from models.cost import SpectralLoss
from models.dataset import create_regional_datasets
from models.inference import predict_spectra
from models.orchestration import (
    instantiate_model,
    perform_model_training_and_bayesian_optimization,
)
from models.resnet import ResNet1D
from models.unet import ConventionalUNet1D, ResidualUNet1D
from utility.normalization import normalize_spectral_data
from utility.pairing import pair_source_target_spectra
from utility.pseudo_measurement import generate_pseudo_measurements
from utility.splitting import split_measurement_sessions

_DEFAULT_RESNET_TEST_CONFIG = {
    "resnet_channel": 8,
    "kernel_size": 3,
    "dropout": 0.0,
    "use_batch_norm": True,
    "padding_mode": "reflect",
    "use_film": False,
}


class TestOrchestrationPipeline(unittest.TestCase):
    """Test suite for perform_model_training_and_bayesian_optimization."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate, normalize, pair, and split lightweight synthetic data."""
        pseudo_cfg = {
            "material": "NMG",
            "regions": ["Al2p", "Ti2p"],
            "n_points": 40,
            "measurements_per_tool": {"J4": 6, "H1": 6},
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
        """Verify model instantiation across supported types and error handling.
        """
        model_specs = [
            ("resnet", ResNet1D, _DEFAULT_RESNET_TEST_CONFIG),
            ("residual_unet", ResidualUNet1D, {"residual_unet_channel": 4, "depth": 2}),
            ("unet", ConventionalUNet1D, {"unet_channel": 4, "depth": 2}),
        ]
        for name, expected_cls, cfg in model_specs:
            with self.subTest(arch=name):
                m = instantiate_model(name, seq_len=32, config=cfg)
                self.assertIsInstance(m, expected_cls)

        with self.assertRaises(ValueError):
            instantiate_model("unknown_arch", seq_len=30)

    def test_training_orchestration_pipelines(self) -> None:
        """Verify model training orchestration in full and sliding window modes.
        """
        criterion = SpectralLoss()

        # 1. Full spectrum ResNet orchestration
        regional_ds_full = create_regional_datasets(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={"source_tool": "J4", "target_tool": "H1", "regions": ["Al2p"], "use_sliding_window": False},
        )
        res_full = perform_model_training_and_bayesian_optimization(
            regional_ds_full,
            criterion,
            config={
                "model_type": "resnet",
                "train_config": {"epochs": 1, "batch_size": 4, **_DEFAULT_RESNET_TEST_CONFIG},
            },
        )
        self.assertIn("Al2p", res_full["models"])
        self.assertIsInstance(res_full["models"]["Al2p"], ResNet1D)

        # 2. Sliding window orchestration with predict_spectra
        regional_ds_sw = create_regional_datasets(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={
                "source_tool": "J4",
                "target_tool": "H1",
                "regions": ["Al2p"],
                "use_sliding_window": True,
                "window_size": 16,
                "stride": 8,
            },
        )
        res_sw = perform_model_training_and_bayesian_optimization(
            regional_ds_sw,
            criterion,
            config={
                "model_type": "resnet",
                "use_sliding_window": True,
                "train_config": {"epochs": 1, "batch_size": 4, **_DEFAULT_RESNET_TEST_CONFIG},
            },
        )
        self.assertIn("Al2p", res_sw["models"])
        models = {"Al2p": res_sw["models"]["Al2p"]}
        pred_i, pred_e, pred_df = predict_spectra(
            models=models,
            data=(self.ary_intensity, self.ary_energy, self.df_meta),
            config={"source_tool": "J4", "target_tool": "H1"},
        )
        self.assertEqual(pred_i.shape[0], 108)

    def test_train_with_bayesian_optimization(self) -> None:
        """Verify Stage 7 Bayesian optimization execution."""
        regional_datasets = create_regional_datasets(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={"source_tool": "J4", "target_tool": "H1", "regions": ["Al2p"], "use_sliding_window": False},
        )
        res = perform_model_training_and_bayesian_optimization(
            regional_datasets,
            SpectralLoss(),
            bayesian_opt_config={
                "num_trials": 2,
                "epochs_per_trial": 1,
                "kernel_sizes": [3],
                "resnet_channels": [8],
                "seed": 42,
            },
            config={
                "model_type": "resnet",
                "use_bayesian_opt": True,
                "train_config": {**_DEFAULT_RESNET_TEST_CONFIG},
            },
        )
        self.assertIn("Al2p", res["models"])
        self.assertIn("Al2p", res["bayesian_opt_results"])

    def test_orchestration_validation_errors(self) -> None:
        """Verify validation error handling for training orchestration."""
        # Missing model_type raises KeyError
        regional_datasets = create_regional_datasets(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={
                "source_tool": "J4",
                "target_tool": "H1",
                "regions": ["Al2p"],
                "use_sliding_window": False,
            },
        )
        with self.assertRaises(KeyError):
            perform_model_training_and_bayesian_optimization(
                regional_datasets, SpectralLoss(), config={}
            )

        # Non-integer window_size raises ValueError
        with self.assertRaises(ValueError):
            perform_model_training_and_bayesian_optimization(
                regional_datasets,
                SpectralLoss(),
                config={
                    "model_type": "resnet",
                    "train_config": {
                        **_DEFAULT_RESNET_TEST_CONFIG,
                        "window_size": [15],
                    },
                },
            )


if __name__ == "__main__":
    unittest.main()
