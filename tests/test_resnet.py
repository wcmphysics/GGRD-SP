"""Unit tests for 1D ResNet architecture, training, and Ax optimization."""

from __future__ import annotations

import unittest
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.bayesian_opt import optimize_model_hyperparameters
from models.dataset import SpectrumPairDataset
from models.inference import predict_spectra
from models.resnet import REQUIRED_RESNET_KEYS, ResNet1D, ResNetScaledShort1D
from models.trainer import instantiate_model, train_model_region

_DEFAULT_RESNET_FIXED_CONFIG = {
    "dropout": 0.0,
    "use_batch_norm": True,
    "padding_mode": "reflect",
    "use_film": False,
}

_DEFAULT_RESNET_TEST_CONFIG = {
    "resnet_channel": 8,
    "kernel_size": 3,
    **_DEFAULT_RESNET_FIXED_CONFIG,
}


class TestResNet1DArchitecture(unittest.TestCase):
    """Test suite for ResNet1D architecture, validation, and forward pass."""

    def test_config_validation(self) -> None:
        """Verify config check for missing keys, kernel bounds, and extra keys.
        """
        # Missing required keys
        invalid_configs = [None, {}]
        for key in REQUIRED_RESNET_KEYS:
            invalid_configs.append(
                {k: v for k, v in _DEFAULT_RESNET_TEST_CONFIG.items() if k != key}
            )

        for cfg in invalid_configs:
            with self.subTest(msg="missing_key", cfg=cfg):
                with self.assertRaises(KeyError):
                    ResNet1D(50, config=cfg)

        # Kernel size larger than sequence length
        with self.assertRaises(ValueError):
            ResNet1D(5, config={**_DEFAULT_RESNET_TEST_CONFIG, "kernel_size": 7})

        # Extra non-architectural keys safely tolerated
        model = ResNet1D(
            50, config={**_DEFAULT_RESNET_TEST_CONFIG, "arbitrary_param": 123}
        )
        self.assertIsInstance(model, ResNet1D)

    def test_input_shape_validation(self) -> None:
        """Verify input tensor dimension enforcement."""
        model = ResNet1D(40, config=_DEFAULT_RESNET_TEST_CONFIG)
        invalid_tensors = [
            torch.randn(4, 40),          # 2D tensor
            torch.randn(4, 2, 40),       # 3D with 2 channels
            torch.randn(4, 1, 40, 1),    # 4D tensor
        ]
        for x in invalid_tensors:
            with self.subTest(shape=x.shape):
                with self.assertRaises(ValueError):
                    model(x)

    def test_forward_pass_and_layer_scaling(self) -> None:
        """Verify layer count, L2 penalty, and forward shapes across layer
        depths.
        """
        configurations = [
            {"num_layers": 3, "use_batch_norm": True, "dropout": 0.0},
            {"num_layers": 5, "use_batch_norm": False, "dropout": 0.2},
        ]
        for cfg_override in configurations:
            with self.subTest(**cfg_override):
                model = ResNet1D(
                    50,
                    config={
                        **_DEFAULT_RESNET_TEST_CONFIG,
                        "resnet_channel": 16,
                        **cfg_override,
                    },
                )
                num_layers = cfg_override["num_layers"]
                hidden_count = num_layers - 1
                self.assertEqual(len(model.hidden_convs), hidden_count)
                self.assertEqual(len(model.hidden_bns), hidden_count)

                # Channel continuity check
                conv_modules = [
                    next(m for m in layer.modules() if isinstance(m, nn.Conv1d))
                    for layer in model.hidden_convs
                ]
                self.assertEqual(conv_modules[0].in_channels, 1)
                self.assertEqual(conv_modules[0].out_channels, 16)
                for conv in conv_modules[1:]:
                    self.assertEqual(conv.in_channels, 16)
                    self.assertEqual(conv.out_channels, 16)

                # Forward shape preservation and L2 regularization calculation
                x = torch.randn(4, 1, 50)
                out = model(x)
                self.assertEqual(out.shape, (4, 1, 50))
                l2_reg = model.get_l2_regularization()
                self.assertIsInstance(l2_reg, torch.Tensor)
                self.assertGreater(l2_reg.item(), 0.0)

    def test_film_modulation(self) -> None:
        """Verify FiLM modulation channels and forward pass across hidden
        layers.
        """
        cond = {
            "die": torch.tensor([0, 1]),
            "source_tool_index": torch.tensor([0, 0]),
            "target_tool_index": torch.tensor([1, 1]),
            "total_flux_single_die_standardized": torch.tensor([[0.5], [-0.5]]),
            "time_standardized": torch.tensor([[0.2], [-0.2]]),
        }
        model = ResNet1D(
            40,
            config={
                **_DEFAULT_RESNET_TEST_CONFIG,
                "resnet_channel": 8,
                "use_film": True,
                "num_layers": 4,
            },
        )
        self.assertIsNotNone(model.film_gen)
        self.assertEqual(model.film_gen.channels_list, [8, 8, 8])

        out = model(torch.randn(2, 1, 40), cond=cond)
        self.assertEqual(out.shape, (2, 1, 40))


class TestResNetScaledShort1DArchitecture(unittest.TestCase):
    """Test suite for ResNetScaledShort1D with FiLM-scaled shortcut."""

    def setUp(self) -> None:
        self.cond = {
            "die": torch.tensor([0, 1]),
            "source_tool_index": torch.tensor([0, 0]),
            "target_tool_index": torch.tensor([1, 1]),
            "total_flux_single_die_standardized": torch.tensor([[0.5], [-0.5]]),
            "time_standardized": torch.tensor([[0.2], [-0.2]]),
        }

    def test_initialization_and_condition_validation(self) -> None:
        """Verify FiLM requirement, model registration, and condition checks.
        """
        # Requires use_film=True
        with self.assertRaises(ValueError):
            ResNetScaledShort1D(
                40, config={**_DEFAULT_RESNET_TEST_CONFIG, "use_film": False}
            )

        # Registration in instantiate_model
        model = instantiate_model(
            "resnet_scaled_short",
            40,
            {**_DEFAULT_RESNET_TEST_CONFIG, "use_film": True},
        )
        self.assertIsInstance(model, ResNetScaledShort1D)

        # Forward requires valid condition dict
        x = torch.randn(2, 1, 40)
        for invalid_cond in (None, "not_a_dict", [1, 2, 3]):
            with self.subTest(cond=invalid_cond):
                with self.assertRaises(ValueError):
                    model(x, cond=invalid_cond)

    def test_shortcut_gain_identity_and_clamping(self) -> None:
        """Verify identity gain init and non-negative clamp with warning."""
        model = ResNetScaledShort1D(
            40, config={**_DEFAULT_RESNET_TEST_CONFIG, "use_film": True}
        )
        x = torch.randn(2, 1, 40)

        # Baseline identity gain check (gamma == 1.0)
        shortcut_mods = model.shortcut_film_gen(self.cond)
        gamma_out, _ = shortcut_mods[0]
        self.assertTrue(torch.allclose(gamma_out, torch.ones_like(gamma_out)))

        # Force negative gain to test warning and clamp to 0.0
        with torch.no_grad():
            model.shortcut_film_gen.gamma_projs[0].bias.fill_(-5.0)

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            out = model(x, self.cond)
            self.assertEqual(out.shape, (2, 1, 40))
            self.assertTrue(
                any(
                    "Negative shortcut or NaN scaling detected" in str(w.message)
                    for w in recorded_warnings
                )
            )

    def test_film_layer_toggles_and_energy_isolation(self) -> None:
        """Verify film_internal_layers toggle and energy encoding isolation."""
        # film_internal_layers=False disables internal FiLM, keeps shortcut FiLM
        model_no_internal = ResNetScaledShort1D(
            30,
            config={
                **_DEFAULT_RESNET_TEST_CONFIG,
                "use_film": True,
                "film_internal_layers": False,
                "use_sliding_window": True,
                "encode_energy": True,
            },
        )
        self.assertIsNone(model_no_internal.film_gen)
        self.assertIsNotNone(model_no_internal.shortcut_film_gen)
        self.assertFalse(model_no_internal.shortcut_film_gen.encode_energy)

        # film_internal_layers=True keeps internal FiLM with energy enabled
        model_internal = ResNetScaledShort1D(
            30,
            config={
                **_DEFAULT_RESNET_TEST_CONFIG,
                "use_film": True,
                "film_internal_layers": True,
                "use_sliding_window": True,
                "encode_energy": True,
            },
        )
        self.assertTrue(model_internal.film_gen.encode_energy)
        self.assertFalse(model_internal.shortcut_film_gen.encode_energy)


class TestResNetPipelineIntegration(unittest.TestCase):
    """Integration tests for ResNet training, inference, and optimization."""

    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(42)
        torch.manual_seed(42)
        cls.n_pts = 40
        cls.x_raw = np.random.uniform(10.0, 50.0, size=(8, cls.n_pts)).astype(
            np.float32
        )
        cls.y_raw = (cls.x_raw * 1.05).astype(np.float32)
        cls.meta_list = [
            {
                "spectrum_index": i,
                "measurement_id": f"M_{i:03d}",
                "tool": "J4",
                "die": 0,
                "region": "Al2p",
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": 0.1,
                "time_standardized": -0.1,
                "split": "test",
                "target_measurement_id": f"T_{i:03d}",
                "target_tool": "H1",
            }
            for i in range(8)
        ]
        cls.dataset = SpectrumPairDataset(
            cls.x_raw, cls.y_raw, metadata=cls.meta_list
        )
        cls.meta_df = pd.DataFrame(cls.meta_list)

    def test_training_and_inference_pipeline(self) -> None:
        """Verify training and prediction with sliding window and relative LR.
        """
        res = train_model_region(
            self.dataset,
            self.dataset,
            config={
                **_DEFAULT_RESNET_TEST_CONFIG,
                "model_type": "resnet_scaled_short",
                "use_film": True,
                "use_sliding_window": True,
                "window_size": 16,
                "stride": 8,
                "epochs": 2,
                "batch_size": 4,
                "learning_rate": 1e-3,
                "relative_learning_rate": 0.05,
            },
        )
        self.assertIn("model", res)
        self.assertIsInstance(res["model"], ResNetScaledShort1D)
        self.assertLess(res["best_val_loss"], float("inf"))

        predicted_intensity, predicted_energy, predicted_dataframe = predict_spectra(
            models={"Al2p": res["model"]},
            data=(self.x_raw, np.ones_like(self.x_raw), self.meta_df),
            config={"source_tool": "J4", "target_tool": "H1"},
        )
        self.assertEqual(predicted_intensity.shape, (8, self.n_pts))
        self.assertEqual(len(predicted_dataframe), 8)

    def test_ax_bayesian_optimization_integration(self) -> None:
        """Verify Ax Bayesian optimization searches ResNet hyperparameter space.
        """
        ax_res = optimize_model_hyperparameters(
            self.dataset,
            self.dataset,
            config={
                **_DEFAULT_RESNET_FIXED_CONFIG,
                "model_type": "resnet",
                "num_trials": 2,
                "epochs_per_trial": 2,
                "kernel_sizes": [3],
                "resnet_channels": [8],
                "batch_sizes": [8],
                "seed": 42,
            },
        )
        self.assertIn("best_parameters", ax_res)
        self.assertIn("best_val_loss", ax_res)
        self.assertEqual(len(ax_res["trials_data"]), 2)


if __name__ == "__main__":
    unittest.main()
