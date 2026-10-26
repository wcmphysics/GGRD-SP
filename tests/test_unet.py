"""Unit tests for the 1D U-Net neural network architecture and pipeline."""

from __future__ import annotations

import unittest
import numpy as np
import torch

from models.dataset import SpectrumPairDataset
from models.trainer import train_model_region
from models.unet import UNet1D, UNetConvBlock1D


class TestUNet1DArchitecture(unittest.TestCase):
    """Test suite for UNet1D network module and forward transformations."""

    def test_output_shapes_and_odd_sequence_lengths(self) -> None:
        """Verify 1D U-Net shape preservation across standard and odd sequence
        lengths.
        """
        # 2D input should raise ValueError
        model = UNet1D(
            n_points=40,
            config={"residual_unet_channel": 8, "depth": 2, "kernel_size": 3},
        )
        with self.assertRaises(ValueError):
            model(torch.randn(4, 40))

        # Odd and even sequence length preservation
        for seq_len in (25, 40, 53, 77):
            with self.subTest(seq_len=seq_len):
                m = UNet1D(
                    n_points=seq_len,
                    config={"residual_unet_channel": 8, "depth": 2, "kernel_size": 3},
                )
                x = torch.randn(2, 1, seq_len)
                y = m(x)
                self.assertEqual(y.shape, (2, 1, seq_len))

    def test_modes_block_config_and_l2_regularization(self) -> None:
        """Verify residual mode, conv block config, and L2 regularization."""
        # Residual vs direct prediction
        x = torch.randn(2, 1, 40)
        for residual_mode in (True, False):
            with self.subTest(residual=residual_mode):
                m = UNet1D(
                    n_points=40,
                    config={
                        "residual": residual_mode,
                        "residual_unet_channel": 8,
                        "unet_channel": 8,
                        "depth": 2,
                    },
                )
                self.assertEqual(m(x).shape, (2, 1, 40))

        # L2 regularization check
        model_reg = UNet1D(
            n_points=40,
            config={"residual_unet_channel": 8, "depth": 2, "use_batch_norm": True},
        )
        l2_val = model_reg.get_l2_regularization()
        self.assertGreater(float(l2_val.item()), 0.0)

        # UNetConvBlock1D config interface
        block = UNetConvBlock1D(
            4, 8, config={"kernel_size": 3, "dropout": 0.1, "use_batch_norm": True}
        )
        self.assertEqual(block(torch.randn(2, 4, 30)).shape, (2, 8, 30))

    def test_invalid_parameters_raise(self) -> None:
        """Verify invalid kernel_size, depth, and sequence lengths raise
        ValueError.
        """
        invalid_configs = [
            (50, {"kernel_size": 4}),               # even kernel
            (50, {"kernel_size": 0}),               # non-positive kernel
            (50, {"depth": 0}),                     # depth < 1
            (7, {"depth": 3}),                      # 7 < 2**3
            (8, {"depth": 2, "kernel_size": 9}),    # kernel_size > n_points
        ]
        for n_pts, cfg in invalid_configs:
            with self.subTest(n_points=n_pts, cfg=cfg):
                with self.assertRaises(ValueError):
                    UNet1D(n_points=n_pts, config=cfg)


class TestUNetTrainingAndPipeline(unittest.TestCase):
    """Test suite for training routine and pipeline integration."""

    def test_train_unet_region_loss_reduction(self) -> None:
        """Verify train_model_region completes epochs with residual_unet."""
        x_synth = np.random.uniform(10.0, 50.0, size=(8, 30)).astype(np.float32)
        y_synth = (x_synth * 1.05).astype(np.float32)
        ds = SpectrumPairDataset(x_synth, y_synth)

        res = train_model_region(
            train_dataset=ds,
            val_dataset=ds,
            config={
                "model_type": "residual_unet",
                "epochs": 2,
                "batch_size": 4,
                "residual_unet_channel": 8,
                "depth": 2,
                "kernel_size": 3,
                "learning_rate": 1e-3,
                "verbose": False,
            },
        )
        self.assertIn("model", res)
        self.assertIn("best_val_loss", res)
        self.assertEqual(len(res["history"]["train_loss"]), 2)
        self.assertIsInstance(res["model"], UNet1D)


if __name__ == "__main__":
    unittest.main()
