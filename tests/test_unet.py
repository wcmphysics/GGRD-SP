"""Unit tests for the 1D U-Net neural network architecture and pipeline."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from models.cost import NormalizedMSELoss
from models.dataset import SpectrumPairDataset
from models.unet import (
    UNet1D,
    UNetConvBlock1D,
    run_unet_pipeline,
    train_unet_region,
)
from utility.pairing import pair_source_target_spectra
from utility.pseudo_measurement import generate_pseudo_measurements


class TestUNet1DArchitecture(unittest.TestCase):
    """Test suite for UNet1D network module and forward transformations."""

    def test_output_shapes_2d_and_3d(self) -> None:
        """Verify 1D U-Net preserves exact sequence length for 2D and 3D inputs."""
        model = UNet1D(n_points=100, config={"base_channels": 8, "depth": 3, "kernel_size": 3})

        # 2D input (batch, n_points)
        x_2d = torch.randn(4, 100)
        y_2d = model(x_2d)
        self.assertEqual(y_2d.shape, (4, 100))

        # 3D input (batch, 1, n_points)
        x_3d = torch.randn(4, 1, 100)
        y_3d = model(x_3d)
        self.assertEqual(y_3d.shape, (4, 1, 100))

    def test_odd_sequence_length_robustness(self) -> None:
        """Verify decoder interpolation gracefully matches odd lengths without dimension mismatch."""
        odd_lengths = [25, 53, 77, 101]
        for l in odd_lengths:
            model = UNet1D(n_points=l, config={"base_channels": 8, "depth": 3, "kernel_size": 3})
            x = torch.randn(2, l)
            y = model(x)
            self.assertEqual(y.shape, (2, l), f"Failed for odd sequence length {l}")

    def test_direct_vs_residual_mode(self) -> None:
        """Verify both residual shortcut and direct prediction modes operate correctly."""
        model_res = UNet1D(n_points=40, config={"residual": True, "base_channels": 8, "depth": 2})
        model_dir = UNet1D(n_points=40, config={"residual": False, "base_channels": 8, "depth": 2})

        x = torch.randn(2, 40)
        y_res = model_res(x)
        y_dir = model_dir(x)

        self.assertEqual(y_res.shape, (2, 40))
        self.assertEqual(y_dir.shape, (2, 40))

    def test_l2_regularization_scalar(self) -> None:
        """Verify L2 penalty calculates positive scalar on conv weights and excludes BatchNorm."""
        model = UNet1D(n_points=50, config={"base_channels": 8, "depth": 2, "use_batch_norm": True})
        l2_val = model.get_l2_regularization()
        self.assertIsInstance(l2_val, torch.Tensor)
        self.assertEqual(l2_val.dim(), 0)
        self.assertGreater(float(l2_val.item()), 0.0)

        expected_l2 = sum(
            torch.sum(m.weight**2).item()
            for m in model.modules()
            if isinstance(m, torch.nn.Conv1d)
        )
        self.assertAlmostEqual(float(l2_val.item()), expected_l2, places=3)

    def test_conv_block_config_interface(self) -> None:
        """Verify UNetConvBlock1D accepts config dictionary with bundled parameters."""
        block = UNetConvBlock1D(4, 8, config={"kernel_size": 3, "dropout": 0.1, "use_batch_norm": True})
        x = torch.randn(2, 4, 30)
        y = block(x)
        self.assertEqual(y.shape, (2, 8, 30))

    def test_invalid_parameters_raise(self) -> None:
        """Verify invalid kernel_size, depth, and sequence length raise ValueError."""
        with self.assertRaises(ValueError):
            UNet1D(n_points=50, config={"kernel_size": 4})  # even
        with self.assertRaises(ValueError):
            UNet1D(n_points=50, config={"kernel_size": 0})  # non-positive
        with self.assertRaises(ValueError):
            UNet1D(n_points=50, config={"depth": 0})  # depth < 1
        with self.assertRaises(ValueError):
            UNet1D(n_points=7, config={"depth": 3})  # 7 < 2**3


class TestUNetTrainingAndPipeline(unittest.TestCase):
    """Test suite for training routine and end-to-end pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate small pseudo-measurement dataset for testing."""
        config = {
            "measurements_per_tool": {"J4": 4, "H1": 4},
            "measurements_per_t7_code": 2,
            "n_die": 2,
            "n_points": 40,
            "regions": ["Al2p", "Ti2p"],
            "seed": 42,
        }
        cls.ary_int, cls.ary_ene, cls.meta_df = generate_pseudo_measurements(config)
        cls.meta_df = pair_source_target_spectra(
            cls.meta_df,
            config={"source_tool": "J4", "target_tool": "H1", "time_threshold_hours": 12.0},
        )

    def test_train_unet_region_loss_reduction(self) -> None:
        """Verify train_unet_region completes epochs and records loss history."""
        # Synthetic small dataset
        x_synth = np.random.uniform(10.0, 100.0, size=(10, 30))
        y_synth = x_synth + np.random.normal(0.0, 1.0, size=(10, 30))
        ds = SpectrumPairDataset(x_synth, y_synth)

        res = train_unet_region(
            train_dataset=ds,
            val_dataset=ds,
            config={
                "epochs": 3,
                "batch_size": 4,
                "base_channels": 8,
                "depth": 2,
                "kernel_size": 3,
                "learning_rate": 1e-3,
            },
        )

        self.assertIn("model", res)
        self.assertIn("best_val_loss", res)
        self.assertEqual(len(res["history"]["train_loss"]), 3)
        self.assertIsInstance(res["model"], UNet1D)

    def test_run_unet_pipeline_full_spectrum_mode(self) -> None:
        """Verify full regional spectrum pipeline produces standardized output containers."""
        res = run_unet_pipeline(
            self.meta_df,
            self.ary_int,
            self.ary_ene,
            config={
                "use_sliding_window": False,
                "regions": ["Al2p"],
                "source_tool": "J4",
                "target_tool": "H1",
                "val_ratio": 0.25,
                "train_config": {
                    "epochs": 2,
                    "batch_size": 4,
                    "base_channels": 8,
                    "depth": 2,
                    "kernel_size": 3,
                },
                "predict_source": True,
            },
        )

        self.assertIn("Al2p", res["models"])
        self.assertIsInstance(res["models"]["Al2p"], UNet1D)
        self.assertIn("Al2p", res["evaluation"])

        # Check prediction containers
        pred_int, pred_ene, pred_meta = res["predictions"]
        self.assertEqual(pred_int.shape[1], 40)
        self.assertEqual(pred_ene.shape[1], 40)
        self.assertTrue(len(pred_meta) > 0)
        self.assertIn("t7_code", pred_meta.columns)
        self.assertTrue((pred_meta["tool"] == "H1").all())
        self.assertTrue(pred_meta["is_predicted"].all())

    def test_run_unet_pipeline_sliding_window_mode(self) -> None:
        """Verify sliding window patch mode works end-to-end with UNet1D."""
        res = run_unet_pipeline(
            self.meta_df,
            self.ary_int,
            self.ary_ene,
            config={
                "use_sliding_window": True,
                "regions": ["Al2p"],
                "source_tool": "J4",
                "target_tool": "H1",
                "val_ratio": 0.25,
                "window_size_points": 15,
                "sliding_stride_points": 5,
                "train_config": {
                    "epochs": 2,
                    "batch_size": 8,
                    "base_channels": 8,
                    "depth": 2,
                    "kernel_size": 3,
                },
                "predict_source": True,
            },
        )

        self.assertIn("Al2p", res["models"])
        # In sliding window mode, models[reg] is (model, window_size, stride)
        model_tuple = res["models"]["Al2p"]
        self.assertIsInstance(model_tuple, tuple)
        self.assertIsInstance(model_tuple[0], UNet1D)

        pred_int, pred_ene, pred_meta = res["predictions"]
        self.assertEqual(pred_int.shape[1], 40)
        self.assertTrue(len(pred_meta) > 0)


if __name__ == "__main__":
    unittest.main()
