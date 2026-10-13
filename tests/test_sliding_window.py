"""Unit tests for the sliding window neural network model and pipeline."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from models.bayesian_opt import optimize_model_hyperparameters
from models.cost import NormalizedMSELoss
from models.dataset import SpectrumPairDataset, SpectrumPatchDataset
from models.inference import predict_sliding_window_spectrum
from models.resnet import ResNet1D
from models.trainer import evaluate_sliding_window, train_model_region


class TestSlidingWindowPipeline(unittest.TestCase):
    """Test cases for dataset, training, evaluation, and Ax tuning with
    sliding windows.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Generate synthetic paired dataset for testing."""
        np.random.seed(42)
        torch.manual_seed(42)
        n_points = 60
        x_raw = np.random.uniform(10.0, 100.0, size=(12, n_points)).astype(np.float32)
        y_raw = (x_raw * 1.05 + 1.0).astype(np.float32)

        cls.train_ds = SpectrumPairDataset(
            x_raw[:8], y_raw[:8], config={"normalization_mode": "none"}
        )
        cls.val_ds = SpectrumPairDataset(
            x_raw[8:], y_raw[8:], config={"normalization_mode": "none"}
        )

    def test_spectrum_patch_dataset(self) -> None:
        """Test patch dataset construction from full spectra."""
        patch_ds = SpectrumPatchDataset(self.train_ds, window_size=15, stride=5)
        self.assertGreater(len(patch_ds), len(self.train_ds))
        item = patch_ds[0]
        self.assertEqual(item["x"].shape, (15,))
        self.assertEqual(item["y"].shape, (15,))

    def test_predict_sliding_window_spectrum(self) -> None:
        """Test single spectrum prediction and full length reconstruction."""
        model = ResNet1D(n_points=15, config={"hidden_channels": 8, "kernel_size": 3})
        spectrum = self.train_ds[0]["x"].cpu().numpy()
        pred = predict_sliding_window_spectrum(
            model, spectrum, config={"window_size": 15, "stride": 5}
        )
        self.assertEqual(len(pred), len(spectrum))

    def test_evaluate_sliding_window_reconstructed(self) -> None:
        """Test full-spectrum reconstructed evaluation loss calculation."""
        model = ResNet1D(n_points=15, config={"hidden_channels": 8, "kernel_size": 3})
        crit = NormalizedMSELoss(l2_weight=0.0)
        loss = evaluate_sliding_window(
            model=model,
            full_val_dataset=self.val_ds,
            criterion=crit,
            config={"window_size": 15, "stride": 5},
        )
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)

    def test_train_sliding_window_region(self) -> None:
        """Test regional sliding window training via train_model_region."""
        train_res = train_model_region(
            self.train_ds,
            self.val_ds,
            config={
                "model_type": "resnet",
                "use_sliding_window": True,
                "window_size": 15,
                "stride": 5,
                "hidden_channels": 8,
                "kernel_size": 3,
                "epochs": 3,
                "batch_size": 16,
                "verbose": False,
            },
        )
        self.assertIn("model", train_res)
        self.assertIn("best_val_loss", train_res)
        self.assertEqual(len(train_res["history"]["train_loss"]), 3)
        self.assertEqual(len(train_res["history"]["val_loss"]), 3)

    def test_optimize_sliding_window_hyperparameters(self) -> None:
        """Test Ax Bayesian optimization for sliding window model."""
        ax_res = optimize_model_hyperparameters(
            self.train_ds,
            self.val_ds,
            config={
                "model_type": "resnet",
                "use_sliding_window": True,
                "num_trials": 2,
                "epochs_per_trial": 2,
                "window_size_choices": [11, 15],
                "kernel_sizes": [3],
                "hidden_channels": [8],
                "seed": 42,
            },
        )
        self.assertIn("best_parameters", ax_res)
        self.assertIn("best_val_loss", ax_res)
        self.assertEqual(len(ax_res["trials_data"]), 2)

    def test_optimize_sliding_window_invalid_parameters(self) -> None:
        """Verify that Ax optimization raises ValueError on invalid
        parameters.
        """
        with self.assertRaises(ValueError):
            optimize_model_hyperparameters(
                self.train_ds,
                self.val_ds,
                config={"model_type": "resnet", "kernel_sizes": [4]},  # even
            )
        with self.assertRaises(ValueError):
            optimize_model_hyperparameters(
                self.train_ds,
                self.val_ds,
                config={"model_type": "resnet", "lr_bounds": (0.01, 0.001)},
            )
        with self.assertRaises(ValueError):
            optimize_model_hyperparameters(
                self.train_ds,
                self.val_ds,
                config={"model_type": "resnet", "l2_bounds": (-1e-4, 1e-2)},
            )

    def test_parameter_packaging_config_dict(self) -> None:
        """Verify predict and evaluate functions with config dictionaries
        adhering to Rule 7.
        """
        model = ResNet1D(n_points=15, config={"hidden_channels": 8, "kernel_size": 3})
        spectrum = self.train_ds[0]["x"].cpu().numpy()
        pred = predict_sliding_window_spectrum(
            model, spectrum, config={"window_size": 15, "stride": 5, "batch_size": 32}
        )
        self.assertEqual(len(pred), len(spectrum))

        loss = evaluate_sliding_window(
            model,
            self.val_ds,
            NormalizedMSELoss(l2_weight=0.0),
            config={"window_size": 15, "stride": 5},
        )
        self.assertIsInstance(loss, float)


if __name__ == "__main__":
    unittest.main()
