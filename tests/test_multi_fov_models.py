"""Unit tests for multi-field-of-view 1D models, dedicated configs,
and guards.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import yaml
from pydantic import ValidationError
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.bayesian_opt import optimize_model_hyperparameters
from models.dataset import SpectrumPairDataset, SpectrumPatchDataset
from models.factory import (
    MODEL_CLASS_MAP,
    NO_SLIDING_WINDOW_MODEL_TYPES,
    instantiate_model,
)
from models.fpn import FPN1D, FPNConvBlock1D
from models.inception import (
    InceptionModule1D,
    InceptionResidualBlock1D,
    InceptionTime1D,
)
from models.orchestration import (
    perform_model_training_and_bayesian_optimization,
)
from models.tcn import TCN1D, TCNResidualBlock1D
from models.trainer import train_model_region
from utility.config_loader import (
    load_config,
    load_yaml,
    merge_dedicated_model_config,
    resolve_pipeline_config,
)
from utility.config_schema import (
    BayesianOptConfig,
    PipelineConfig,
    TrainConfig,
)


class TestInceptionTime1D(unittest.TestCase):
    """Test suite for InceptionTime1D architecture."""

    def test_forward_output_shape_and_l2(self) -> None:
        """Verify output shape matches input and L2 regularization is finite."""
        model = InceptionTime1D(
            100,
            config={
                "inception_channel": 8,
                "bottleneck_channels": 8,
                "kernel_sizes": [3, 7, 15],
                "num_blocks": 2,
                "modules_per_block": 1,
            },
        )
        x = torch.randn(4, 1, 100)
        y = model(x)
        self.assertEqual(y.shape, (4, 1, 100))

        l2 = model.get_l2_regularization()
        self.assertIsInstance(l2, torch.Tensor)
        self.assertGreater(l2.item(), 0.0)

    def test_custom_kernel_sizes_and_odd_sequence_length(self) -> None:
        """Verify configurable kernel_sizes on odd sequence length."""
        model = InceptionTime1D(
            99,
            config={
                "kernel_sizes": [5, 11, 21],
                "inception_channel": 4,
                "bottleneck_channels": 4,
                "num_blocks": 1,
            },
        )
        x = torch.randn(2, 1, 99)
        y = model(x)
        self.assertEqual(y.shape, (2, 1, 99))

    def test_invalid_even_kernel_size_raises(self) -> None:
        """Verify even kernel_size in kernel_sizes raises ValueError."""
        with self.assertRaises(ValueError):
            InceptionTime1D(100, config={"kernel_sizes": [3, 8, 15]})

    def test_kernel_size_larger_than_n_points_raises(self) -> None:
        """Verify kernel size exceeding n_points raises ValueError."""
        with self.assertRaises(ValueError):
            InceptionTime1D(20, config={"kernel_sizes": [3, 25]})

    def test_film_conditioning(self) -> None:
        """Verify FiLM conditioning forward pass produces valid tensor."""
        film_cfg = {
            "use_film": True,
            "encode_die": True,
            "encode_tool": True,
            "encode_flux": True,
            "encode_time": True,
            "encode_energy": True,
            "encode_region": True,
            "encode_relative_intensity": True,
            "encode_elements": False,
            "num_dies": 3,
            "num_tools": 2,
            "num_regions": 5,
            "num_elements": 2,
            "hidden_dim": 16,
            "inception_channel": 8,
            "bottleneck_channels": 8,
            "num_blocks": 1,
        }
        model = InceptionTime1D(50, config=film_cfg)
        x = torch.randn(2, 1, 50)
        cond = {
            "die_one_hot": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "source_tool_one_hot": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "target_tool_one_hot": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            "time_standardized": torch.tensor([[0.5], [-0.2]]),
            "total_flux_single_die_standardized": torch.tensor([[1.0], [0.0]]),
            "region_one_hot": torch.tensor(
                [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]]
            ),
            "relative_intensity_single_region": torch.tensor([[1.0], [0.8]]),
            "energy_bounds": torch.tensor([[100.0, 150.0], [100.0, 150.0]]),
        }
        y = model(x, cond=cond)
        self.assertEqual(y.shape, (2, 1, 50))

    def test_sliding_window_prohibition_in_constructor(self) -> None:
        """Verify constructor rejects use_sliding_window=True."""
        with self.assertRaises(ValueError) as ctx:
            InceptionTime1D(100, config={"use_sliding_window": True})
        self.assertIn("Sliding window mode", str(ctx.exception))


class TestTCN1D(unittest.TestCase):
    """Test suite for TCN1D architecture."""

    def test_forward_output_shape_and_l2(self) -> None:
        """Verify output shape matches input across dilated residual blocks."""
        model = TCN1D(
            100,
            config={
                "tcn_channel": 16,
                "kernel_size": 5,
                "dilations": [1, 2, 4],
            },
        )
        x = torch.randn(3, 1, 100)
        y = model(x)
        self.assertEqual(y.shape, (3, 1, 100))

        l2 = model.get_l2_regularization()
        self.assertIsInstance(l2, torch.Tensor)
        self.assertGreater(l2.item(), 0.0)

    def test_configurable_kernel_size_and_arbitrary_length(self) -> None:
        """Verify configurable odd kernel_size on odd sequence length."""
        model = TCN1D(
            75,
            config={
                "tcn_channel": 8,
                "kernel_size": 7,
                "dilations": [1, 2],
            },
        )
        x = torch.randn(2, 1, 75)
        y = model(x)
        self.assertEqual(y.shape, (2, 1, 75))

    def test_even_kernel_size_raises(self) -> None:
        """Verify even kernel_size raises ValueError."""
        with self.assertRaises(ValueError):
            TCN1D(100, config={"kernel_size": 4})

    def test_film_conditioning(self) -> None:
        """Verify TCN1D forward pass with FiLM metadata conditioning."""
        cfg = {
            "use_film": True,
            "encode_die": True,
            "encode_tool": True,
            "encode_flux": True,
            "encode_time": True,
            "encode_energy": True,
            "encode_region": True,
            "encode_relative_intensity": True,
            "encode_elements": False,
            "num_dies": 3,
            "num_tools": 2,
            "num_regions": 3,
            "num_elements": 2,
            "hidden_dim": 16,
            "tcn_channel": 8,
            "dilations": [1, 2],
        }
        model = TCN1D(60, config=cfg)
        x = torch.randn(2, 1, 60)
        cond = {
            "die_one_hot": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "source_tool_one_hot": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "target_tool_one_hot": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            "time_standardized": torch.tensor([[0.1], [-0.5]]),
            "total_flux_single_die_standardized": torch.tensor([[0.2], [-0.1]]),
            "region_one_hot": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "relative_intensity_single_region": torch.tensor([[0.5], [0.7]]),
            "energy_bounds": torch.tensor([[50.0, 100.0], [50.0, 100.0]]),
        }
        y = model(x, cond=cond)
        self.assertEqual(y.shape, (2, 1, 60))

    def test_sliding_window_prohibition_in_constructor(self) -> None:
        """Verify constructor rejects use_sliding_window=True."""
        with self.assertRaises(ValueError) as ctx:
            TCN1D(100, config={"use_sliding_window": True})
        self.assertIn("Sliding window mode", str(ctx.exception))


class TestFPN1D(unittest.TestCase):
    """Test suite for FPN1D architecture."""

    def test_forward_output_shape_and_l2(self) -> None:
        """Verify FPN1D multi-scale pyramid output shape and L2 reg."""
        model = FPN1D(
            64,
            config={
                "fpn_base_channel": 8,
                "fpn_pyramid_channel": 16,
                "depth": 3,
                "kernel_size": 3,
            },
        )
        x = torch.randn(3, 1, 64)
        y = model(x)
        self.assertEqual(y.shape, (3, 1, 64))

        l2 = model.get_l2_regularization()
        self.assertIsInstance(l2, torch.Tensor)
        self.assertGreater(l2.item(), 0.0)

    def test_odd_sequence_length_preservation(self) -> None:
        """Verify FPN1D operates on odd sequence lengths via linear interp."""
        model = FPN1D(
            65,
            config={
                "fpn_base_channel": 4,
                "fpn_pyramid_channel": 8,
                "depth": 2,
                "kernel_size": 3,
            },
        )
        x = torch.randn(2, 1, 65)
        y = model(x)
        self.assertEqual(y.shape, (2, 1, 65))

    def test_insufficient_sequence_length_for_depth_raises(self) -> None:
        """Verify depth requiring more points than n_points raises error."""
        with self.assertRaises(ValueError):
            FPN1D(7, config={"depth": 3})  # 2**3 = 8 > 7

    def test_film_conditioning(self) -> None:
        """Verify FPN1D forward pass with FiLM metadata conditioning."""
        cfg = {
            "use_film": True,
            "encode_die": True,
            "encode_tool": True,
            "encode_flux": True,
            "encode_time": True,
            "encode_energy": True,
            "encode_region": True,
            "encode_relative_intensity": True,
            "encode_elements": False,
            "num_dies": 3,
            "num_tools": 2,
            "num_regions": 4,
            "num_elements": 2,
            "hidden_dim": 16,
            "fpn_base_channel": 8,
            "fpn_pyramid_channel": 16,
            "depth": 2,
        }
        model = FPN1D(50, config=cfg)
        x = torch.randn(2, 1, 50)
        cond = {
            "die_one_hot": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "source_tool_one_hot": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "target_tool_one_hot": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            "time_standardized": torch.tensor([[0.2], [0.3]]),
            "total_flux_single_die_standardized": torch.tensor([[0.5], [0.4]]),
            "region_one_hot": torch.tensor(
                [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            ),
            "relative_intensity_single_region": torch.tensor([[0.9], [0.8]]),
            "energy_bounds": torch.tensor([[100.0, 150.0], [100.0, 150.0]]),
        }
        y = model(x, cond=cond)
        self.assertEqual(y.shape, (2, 1, 50))

    def test_sliding_window_prohibition_in_constructor(self) -> None:
        """Verify constructor rejects use_sliding_window=True."""
        with self.assertRaises(ValueError) as ctx:
            FPN1D(100, config={"use_sliding_window": True})
        self.assertIn("Sliding window mode", str(ctx.exception))


class TestDedicatedModelConfigFiles(unittest.TestCase):
    """Test suite for dedicated per-model YAML configs under config/."""

    def test_all_model_yaml_files_exist_and_parse(self) -> None:
        """Verify all 6 dedicated model config YAMLs exist and parse cleanly."""
        expected_yamls = [
            "resnet.yaml",
            "unet.yaml",
            "deeplabv3.yaml",
            "inception_time.yaml",
            "tcn.yaml",
            "fpn.yaml",
        ]
        for yml_name in expected_yamls:
            cfg = load_yaml(f"config/{yml_name}")
            self.assertIn("train_config", cfg)
            self.assertIn("bayesian_opt", cfg)
            self.assertIn("film", cfg)

    def test_new_model_hyperparameter_defaults(self) -> None:
        """Verify inception_time, tcn, fpn have epochs=50, lr=0.001."""
        for model_name in ("inception_time", "tcn", "fpn"):
            cfg = load_yaml(f"config/{model_name}.yaml")
            tc = cfg["train_config"]
            self.assertEqual(tc["epochs"], 50)
            self.assertEqual(tc["learning_rate"], 0.001)
            self.assertEqual(tc["early_stopping_patience"], 20)

    def test_merge_dedicated_model_config(self) -> None:
        """Verify merge_dedicated_model_config merges defaults into raw."""
        raw = {"global": {"model_type": "inception_time"}}
        merge_dedicated_model_config(raw, "config")
        self.assertEqual(raw["training"]["train_config"]["epochs"], 50)
        self.assertEqual(raw["training"]["train_config"]["learning_rate"], 0.001)
        self.assertIn("inception_channels", raw["bayesian_opt"])
        self.assertIn("encode_region", raw["film"])

    def test_explicit_overrides_persist_after_merge(self) -> None:
        """Verify explicit user overrides are not overwritten by defaults."""
        raw = {
            "global": {"model_type": "tcn"},
            "training": {"train_config": {"epochs": 123, "learning_rate": 0.05}},
        }
        merge_dedicated_model_config(raw, "config")
        self.assertEqual(raw["training"]["train_config"]["epochs"], 123)
        self.assertEqual(raw["training"]["train_config"]["learning_rate"], 0.05)
        self.assertEqual(raw["training"]["train_config"]["tcn_channel"], 32)

    def test_config_yaml_loading_with_dedicated_models(self) -> None:
        """Verify config.yaml loading populates dedicated model defaults."""
        for model_name in ("inception_time", "tcn", "fpn"):
            raw = load_yaml("config/config.yaml")
            raw["global"]["model_type"] = model_name
            with tempfile.NamedTemporaryFile(
                "w", dir="config", suffix=".yaml", delete=False, encoding="utf-8"
            ) as tmp:
                yaml.dump(raw, tmp)
                tmp_path = tmp.name
            try:
                cfg = load_config(tmp_path, validate=True)
                tc = cfg["training"]["train_config"]
                self.assertEqual(tc["epochs"], 50)
                self.assertEqual(tc["learning_rate"], 0.001)
                self.assertEqual(tc["early_stopping_patience"], 20)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    def test_train_config_even_kernel_sizes_rejected(self) -> None:
        """Verify TrainConfig rejects even integers in kernel_sizes."""
        with self.assertRaises(ValidationError):
            TrainConfig(kernel_sizes=[3, 4, 15])
        with self.assertRaises(ValidationError):
            TrainConfig(kernel_sizes=[])

    def test_bayesian_opt_config_even_kernel_sizes_rejected(self) -> None:
        """Verify BayesianOptConfig rejects even candidate kernel sizes."""
        with self.assertRaises(ValidationError):
            BayesianOptConfig(kernel_sizes=[3, 4, 7])

    def test_block_constructors_even_kernel_sizes_rejected(self) -> None:
        """Verify block-level constructors reject even kernel sizes."""
        with self.assertRaises(ValueError):
            InceptionModule1D(1, {"kernel_sizes": [3, 4, 15]})
        with self.assertRaises(ValueError):
            TCNResidualBlock1D(1, 16, kernel_size=4, dilation=1, config={})
        with self.assertRaises(ValueError):
            FPNConvBlock1D(1, 16, config={"kernel_size": 4})


class TestSlidingWindowProhibitionMultiLayer(unittest.TestCase):
    """Verify sliding window prohibition is enforced across all 4 boundaries."""

    def test_layer1_config_schema_validation(self) -> None:
        """Verify PipelineConfig rejects use_sliding_window=True."""
        for m in NO_SLIDING_WINDOW_MODEL_TYPES:
            raw = load_yaml("config/config.yaml")
            raw["global"]["model_type"] = m
            raw["global"]["use_sliding_window"] = True
            with self.assertRaises(ValueError) as ctx:
                PipelineConfig.model_validate(raw)
            self.assertIn("Sliding window mode", str(ctx.exception))

    def test_layer1_config_loader_resolution(self) -> None:
        """Verify resolve_pipeline_config rejects use_sliding_window=True."""
        for m in NO_SLIDING_WINDOW_MODEL_TYPES:
            raw = {
                "materials": {"NMG": {"elements": ["Ti", "O"]}},
                "global": {"model_type": m, "use_sliding_window": True},
            }
            with self.assertRaises(ValueError) as ctx:
                resolve_pipeline_config(raw)
            self.assertIn("Sliding window mode", str(ctx.exception))

    def test_layer2_orchestration_rejection(self) -> None:
        """Verify perform_model_training rejects sliding window datasets."""
        x = torch.randn(10, 1, 100)
        y = torch.randn(10, 1, 100)
        base_ds = SpectrumPairDataset(x, y)
        patch_ds = SpectrumPatchDataset(base_ds, window_size=20, stride=10)

        for m in NO_SLIDING_WINDOW_MODEL_TYPES:
            regional_ds = {
                "Ti2p": {
                    "dataset_train": patch_ds,
                    "dataset_val": base_ds,
                    "dataset_train_full": base_ds,
                    "window_size_points": 20,
                    "sliding_stride_points": 10,
                    "full_spectrum_length": 100,
                }
            }
            crit = nn.MSELoss()
            with self.assertRaises(ValueError) as ctx:
                perform_model_training_and_bayesian_optimization(
                    regional_ds,
                    crit,
                    config={"model_type": m, "use_sliding_window": True},
                )
            self.assertIn("Sliding window mode", str(ctx.exception))

    def test_layer3_trainer_rejection(self) -> None:
        """Verify train_model_region rejects sliding window for multi-FOV."""
        x = torch.randn(10, 1, 100)
        y = torch.randn(10, 1, 100)
        base_ds = SpectrumPairDataset(x, y)
        patch_ds = SpectrumPatchDataset(base_ds, window_size=20, stride=10)

        for m in NO_SLIDING_WINDOW_MODEL_TYPES:
            with self.assertRaises(ValueError) as ctx:
                train_model_region(
                    patch_ds,
                    base_ds,
                    config={
                        "model_type": m,
                        "use_sliding_window": True,
                        "epochs": 1,
                    },
                )
            self.assertIn("Sliding window mode", str(ctx.exception))

    def test_layer3_bayesian_opt_rejection(self) -> None:
        """Verify optimize_model_hyperparameters rejects sliding window."""
        x = torch.randn(10, 1, 100)
        y = torch.randn(10, 1, 100)
        base_ds = SpectrumPairDataset(x, y)

        for m in NO_SLIDING_WINDOW_MODEL_TYPES:
            with self.assertRaises(ValueError) as ctx:
                optimize_model_hyperparameters(
                    base_ds,
                    base_ds,
                    config={
                        "model_type": m,
                        "use_sliding_window": True,
                        "num_trials": 1,
                        "epochs_per_trial": 1,
                    },
                )
            self.assertIn("Sliding window mode", str(ctx.exception))

    def test_layer4_instantiate_model_rejection(self) -> None:
        """Verify instantiate_model rejects use_sliding_window=True."""
        for m in NO_SLIDING_WINDOW_MODEL_TYPES:
            with self.assertRaises(ValueError) as ctx:
                instantiate_model(m, 100, config={"use_sliding_window": True})
            self.assertIn("Sliding window mode", str(ctx.exception))


class TestTrainingAndOptimizationEndToEnd(unittest.TestCase):
    """Quick end-to-end training and optimization smoke tests."""

    def test_short_training_run_for_all_new_models(self) -> None:
        """Verify train_model_region completes 2 epochs for each new model."""
        torch.manual_seed(42)
        x = torch.randn(6, 1, 40)
        y = torch.randn(6, 1, 40)
        train_ds = SpectrumPairDataset(x, y)
        val_ds = SpectrumPairDataset(x, y)

        model_configs = {
            "inception_time": {
                "model_type": "inception_time",
                "inception_channel": 4,
                "bottleneck_channels": 4,
                "kernel_sizes": [3, 7],
                "num_blocks": 1,
                "epochs": 2,
                "batch_size": 2,
                "verbose": False,
            },
            "tcn": {
                "model_type": "tcn",
                "tcn_channel": 8,
                "kernel_size": 3,
                "dilations": [1, 2],
                "epochs": 2,
                "batch_size": 2,
                "verbose": False,
            },
            "fpn": {
                "model_type": "fpn",
                "fpn_base_channel": 4,
                "fpn_pyramid_channel": 8,
                "depth": 2,
                "kernel_size": 3,
                "epochs": 2,
                "batch_size": 2,
                "verbose": False,
            },
        }

        for m_type, cfg in model_configs.items():
            res = train_model_region(train_ds, val_ds, config=cfg)
            self.assertIn("model", res)
            self.assertIn("best_val_loss", res)
            self.assertIn("history", res)
            self.assertLessEqual(len(res["history"]["train_loss"]), 2)


if __name__ == "__main__":
    unittest.main()
