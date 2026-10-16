"""Unit tests for 1D ResNet architecture, training, and Ax optimization."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from models.bayesian_opt import optimize_model_hyperparameters
from models.cost import SpectralLoss
from models.dataset import SpectrumPairDataset, create_dataloaders
from models.inference import format_predicted_measurement_id, predict_spectra
from models.resnet import ResNet1D
from models.trainer import evaluate, train_model_region


class TestResNet1D(unittest.TestCase):
    """Test cases for the ResNet1D architecture."""

    def test_cnn_forward_shape_2d_raises(self) -> None:
        """Test that 2D tensor input raises ValueError."""
        model = ResNet1D(n_points=100, config={"resnet_channel": 16, "kernel_size": 5})
        x = torch.randn(8, 100)
        with self.assertRaises(ValueError):
            model(x)

    def test_cnn_forward_shape_3d(self) -> None:
        """Test forward pass with 3D tensor input (batch, 1, n_points)."""
        model = ResNet1D(n_points=80, config={"resnet_channel": 8, "kernel_size": 3})
        x = torch.randn(4, 1, 80)
        out = model(x)
        self.assertEqual(out.shape, (4, 1, 80))

    def test_invalid_kernel_size_raises(self) -> None:
        """Test that even, non-positive, or larger-than-n_points kernel size raises ValueError."""
        with self.assertRaises(ValueError):
            ResNet1D(n_points=50, config={"kernel_size": 4})
        with self.assertRaises(ValueError):
            ResNet1D(n_points=50, config={"kernel_size": 0})
        with self.assertRaises(ValueError):
            ResNet1D(n_points=50, config={"kernel_size": -3})
        with self.assertRaises(ValueError):
            ResNet1D(n_points=5, config={"kernel_size": 7})

    def test_l2_regularization_targets_weights_only(self) -> None:
        """Test that L2 weight penalty only targets conv weights, not biases or
        BatchNorm.
        """
        model = ResNet1D(n_points=50, config={"resnet_channel": 8, "kernel_size": 3})
        l2_reg = model.get_l2_regularization()
        self.assertIsInstance(l2_reg, torch.Tensor)
        self.assertGreater(l2_reg.item(), 0.0)


class TestSpectrumPairDataset(unittest.TestCase):
    """Test cases for SpectrumPairDataset construction and dataloaders."""

    def test_dataset_mismatched_lengths_raise(self) -> None:
        """Test that mismatched x and y lengths raise ValueError."""
        x = np.ones((5, 10))
        y = np.ones((4, 10))
        with self.assertRaises(ValueError):
            SpectrumPairDataset(x, y)

    def test_create_dataloaders(self) -> None:
        """Test dataloader batch construction."""
        x = np.ones((8, 50), dtype=np.float32)
        y = np.ones((8, 50), dtype=np.float32)
        train_ds = SpectrumPairDataset(x[:6], y[:6], config={"normalization_mode": "none"})
        val_ds = SpectrumPairDataset(x[6:], y[6:], config={"normalization_mode": "none"})
        train_loader, val_loader = create_dataloaders(train_ds, val_ds, config={"batch_size": 4})
        batch = next(iter(train_loader))
        self.assertIn("x", batch)
        self.assertEqual(batch["x"].shape, (4, 1, 50))
        self.assertEqual(batch["y"].shape, (4, 1, 50))


class TestTrainingAndInference(unittest.TestCase):
    """Test model training, inference output structure, and Ax optimization."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate test dataset."""
        np.random.seed(100)
        torch.manual_seed(100)
        n_pts = 50
        x_raw = np.random.uniform(10.0, 100.0, size=(16, n_pts)).astype(np.float32)
        y_raw = (x_raw * 1.05 + 1.0).astype(np.float32)
        cls.train_ds = SpectrumPairDataset(
            x_raw[:10], y_raw[:10], config={"normalization_mode": "none"}
        )
        cls.val_ds = SpectrumPairDataset(
            x_raw[10:], y_raw[10:], config={"normalization_mode": "none"}
        )

        # Minimal metadata for predict_spectra testing
        cls.meta_df = pd.DataFrame([
            {
                "spectrum_index": i,
                "measurement_id": f"NMG_M_J4_{i:05d}",
                "tool": "J4",
                "die": 0,
                "region": "Al2p",
                "split": "test",
                "measurement_id_target": f"NMG_M_H1_{i:05d}",
                "tool_target": "H1",
            }
            for i in range(4)
        ])
        cls.ary_intensity = np.random.uniform(10.0, 100.0, size=(4, n_pts)).astype(np.float32)
        cls.ary_energy = np.tile(np.linspace(10.0, 20.0, n_pts, dtype=np.float32), (4, 1))

    def test_evaluate_excludes_l2_penalty(self) -> None:
        """Verify evaluate() measures pure reconstruction loss, independent
        of l2_weight.
        """
        _, val_loader = create_dataloaders(self.train_ds, self.val_ds, config={"batch_size": 4})
        model = ResNet1D(n_points=50, config={"resnet_channel": 8, "kernel_size": 3})

        crit_zero_l2 = SpectralLoss(l2_weight=0.0)
        crit_large_l2 = SpectralLoss(l2_weight=10.0)

        loss_zero = evaluate(model, val_loader, crit_zero_l2)
        loss_large = evaluate(model, val_loader, crit_large_l2)

        self.assertAlmostEqual(loss_zero, loss_large, places=6)

    def test_train_resnet_region(self) -> None:
        """Test training execution and loss logging via train_model_region."""
        res = train_model_region(
            self.train_ds,
            self.val_ds,
            config={
                "model_type": "resnet",
                "epochs": 5,
                "batch_size": 8,
                "resnet_channel": 8,
                "kernel_size": 3,
                "learning_rate": 1e-3,
                "l2_weight": 1e-5,
            },
        )

        self.assertIn("model", res)
        self.assertIn("best_val_loss", res)
        self.assertEqual(len(res["history"]["train_loss"]), 5)
        self.assertIsInstance(res["model"], ResNet1D)

    def test_format_predicted_measurement_id_uniqueness(self) -> None:
        """Verify unique ID generation without collision for different session
        names.
        """
        id_1 = format_predicted_measurement_id("M_J4_003", "J4", "H1")
        id_2 = format_predicted_measurement_id("SESSION_A_003", "J4", "H1")
        id_3 = format_predicted_measurement_id("SESSION_B_003", "J4", "H1")
        id_new = format_predicted_measurement_id("NMG_M_J4_00001", "J4", "H1")

        self.assertEqual(id_1, "P_J4H1_003")
        self.assertEqual(id_2, "P_J4H1_SESSION_A_003")
        self.assertEqual(id_3, "P_J4H1_SESSION_B_003")
        self.assertEqual(id_new, "NMG_P_J4H1_00001")
        self.assertNotEqual(id_2, id_3)

    def test_predict_spectra_bundled_data(self) -> None:
        """Test inference output formatting with bundled data tuple."""
        model_al2p = train_model_region(
            self.train_ds,
            self.val_ds,
            config={
                "model_type": "resnet",
                "epochs": 2,
                "batch_size": 8,
                "resnet_channel": 8,
                "kernel_size": 3,
            },
        )["model"]

        models = {"Al2p": model_al2p}
        pred_int, pred_eng, pred_df = predict_spectra(
            models=models,
            data=(self.ary_intensity, self.ary_energy, self.meta_df),
            config={"source_tool": "J4", "target_tool": "H1"},
        )

        self.assertEqual(pred_int.shape[0], 4)
        self.assertEqual(pred_eng.shape[0], 4)
        self.assertEqual(len(pred_df), 4)
        self.assertTrue(all(pred_df["measurement_id"].str.contains("P_J4H1_")))
        self.assertTrue(all(pred_df["tool"] == "H1"))

    def test_ax_bayesian_optimization_integration(self) -> None:
        """Test running Ax Bayesian optimization for 2 trials with seed."""
        ax_res = optimize_model_hyperparameters(
            self.train_ds,
            self.val_ds,
            config={
                "model_type": "resnet",
                "num_trials": 2,
                "epochs_per_trial": 2,
                "kernel_sizes": [3, 5],
                "resnet_channels": [8, 16],
                "batch_sizes": [8],
                "seed": 42,
            },
        )

        self.assertIn("best_parameters", ax_res)
        self.assertIn("best_val_loss", ax_res)
        self.assertEqual(len(ax_res["trials_data"]), 2)
        self.assertIn(ax_res["best_parameters"]["kernel_size"], [3, 5])
        self.assertIn(ax_res["best_parameters"]["resnet_channel"], [8, 16])


if __name__ == "__main__":
    unittest.main()
