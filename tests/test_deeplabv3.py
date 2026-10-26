"""Unit tests for the 1D DeepLabV3 neural network architecture and pipeline."""

from __future__ import annotations

import unittest
import numpy as np
import torch

from models.bayesian_opt import optimize_model_hyperparameters
from models.dataset import SpectrumPairDataset
from models.deeplabv3 import ASPP1D, DeepLabV3, ResNet1DBackbone
from models.orchestration import instantiate_model
from models.trainer import train_model_region


class TestDeepLabV3Architecture(unittest.TestCase):
    """Test suite for DeepLabV3 model modules and forward transformations."""

    def test_input_dimension_enforcement(self) -> None:
        """Verify model enforces strictly (batch, 1, seq_len) shape."""
        model = DeepLabV3(
            n_points=30,
            config={"deeplabv3_backbone_channel": 8, "deeplabv3_aspp_channel": 8},
        )
        invalid_shapes = [
            torch.randn(30),          # 1D
            torch.randn(2, 30),       # 2D
            torch.randn(2, 2, 30),    # 3D with 2 channels
            torch.randn(2, 1, 1, 30), # 4D
        ]
        for x in invalid_shapes:
            with self.subTest(shape=x.shape):
                with self.assertRaises(ValueError):
                    model(x)

    def test_forward_shapes_and_sequence_lengths(self) -> None:
        """Verify forward output preservation across short, odd, and normal
        lengths.
        """
        model = DeepLabV3(
            n_points=50,
            config={
                "deeplabv3_backbone_channel": 8,
                "deeplabv3_aspp_channel": 8,
                "kernel_size": 3,
                "multi_grid": (1, 2, 4),
                "aspp_rates": (2, 4, 6),
            },
        )
        # Test odd, standard, and short patch lengths
        for seq_len in (3, 15, 53, 100):
            with self.subTest(seq_len=seq_len):
                x = torch.randn(2, 1, seq_len)
                m = DeepLabV3(
                    n_points=seq_len,
                    config={
                        "deeplabv3_backbone_channel": 8,
                        "deeplabv3_aspp_channel": 8,
                        "aspp_rates": (2, 4, 6),
                    },
                )
                y = m(x)
                self.assertEqual(y.shape, (2, 1, seq_len))
                self.assertFalse(torch.isnan(y).any())

    def test_multi_grid_and_aspp_adaptation(self) -> None:
        """Verify dynamic multi_grid blocks and ASPP atrous branches."""
        # Multi-grid block scaling
        for multi_grid in ((1,), (1, 2, 4), (1, 2, 2, 4, 8)):
            with self.subTest(multi_grid=multi_grid):
                backbone = ResNet1DBackbone(
                    in_channels=1,
                    out_channels=16,
                    multi_grid=multi_grid,
                    kernel_size=3,
                )
                self.assertEqual(len(backbone.blocks), len(multi_grid))

        # ASPP branches and rates
        aspp = ASPP1D(
            in_channels=16,
            out_channels=16,
            aspp_rates=(3, 6, 9),
            kernel_size=3,
        )
        self.assertEqual(len(aspp.atrous_branches), 3)
        feat_out = aspp(torch.randn(2, 16, 40))
        self.assertEqual(feat_out.shape, (2, 16, 40))

    def test_modes_and_regularization(self) -> None:
        """Verify residual mode toggle, L2 regularization, and gradient flow."""
        # Residual vs direct prediction
        for residual_mode in (True, False):
            with self.subTest(residual=residual_mode):
                model = DeepLabV3(
                    n_points=30,
                    config={
                        "residual": residual_mode,
                        "deeplabv3_backbone_channel": 8,
                        "deeplabv3_aspp_channel": 8,
                    },
                )
                out = model(torch.randn(2, 1, 30))
                self.assertEqual(out.shape, (2, 1, 30))

        # L2 penalty and gradient propagation on batch_size=1
        model_train = DeepLabV3(
            n_points=30,
            config={"deeplabv3_backbone_channel": 8, "deeplabv3_aspp_channel": 8},
        )
        l2_reg = model_train.get_l2_regularization()
        self.assertGreater(l2_reg.item(), 0.0)

        x_single = torch.randn(1, 1, 30, requires_grad=True)
        y_single = model_train(x_single)
        loss = y_single.sum() + 1e-4 * l2_reg
        loss.backward()
        self.assertIsNotNone(x_single.grad)

    def test_instantiate_model_factory(self) -> None:
        """Verify instantiate_model resolves 'deeplabv3' and rejects invalid
        names.
        """
        model = instantiate_model(
            "deeplabv3", seq_len=40, config={"deeplabv3_backbone_channel": 8}
        )
        self.assertIsInstance(model, DeepLabV3)
        with self.assertRaises(ValueError):
            instantiate_model("deeplab", seq_len=40)


class TestDeepLabV3PipelineIntegration(unittest.TestCase):
    """Integration tests for DeepLabV3 training and Bayesian optimization."""

    def test_training_and_optimization_pipeline(self) -> None:
        """Verify training step and Ax hyperparameter search."""
        x_synth = np.random.uniform(10.0, 50.0, size=(8, 30)).astype(np.float32)
        y_synth = (x_synth * 1.05).astype(np.float32)
        dataset = SpectrumPairDataset(x_synth, y_synth)

        train_res = train_model_region(
            train_dataset=dataset,
            val_dataset=dataset,
            config={
                "model_type": "deeplabv3",
                "epochs": 2,
                "batch_size": 4,
                "learning_rate": 1e-3,
                "deeplabv3_backbone_channel": 8,
                "deeplabv3_aspp_channel": 8,
                "kernel_size": 3,
                "verbose": False,
            },
        )
        self.assertIn("best_val_loss", train_res)
        self.assertLess(train_res["best_val_loss"], float("inf"))
        self.assertIsInstance(train_res["model"], DeepLabV3)

        # Ax optimization minimal trial check
        ax_res = optimize_model_hyperparameters(
            dataset,
            dataset,
            config={
                "model_type": "deeplabv3",
                "num_trials": 2,
                "epochs_per_trial": 2,
                "kernel_sizes": [3],
                "deeplabv3_backbone_channels": [8],
                "deeplabv3_aspp_channels": [8],
                "batch_sizes": [4],
                "seed": 42,
            },
        )
        self.assertIn("best_parameters", ax_res)
        self.assertEqual(len(ax_res["trials_data"]), 2)


if __name__ == "__main__":
    unittest.main()
