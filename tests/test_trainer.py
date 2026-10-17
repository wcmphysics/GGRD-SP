"""Unit tests for ReduceLROnPlateau dynamical learning rate and zero-validation training in models/trainer.py."""

from __future__ import annotations

import unittest
import numpy as np
import torch
import torch.nn as nn

from models.dataset import SpectrumPairDataset
from models.trainer import train_model_region


class DummyConstantModel(nn.Module):
    """Dummy 1D model outputting constant predictions to simulate a plateau."""

    def __init__(self, n_points: int = 20) -> None:
        super().__init__()
        self.n_points = n_points
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Prediction has zero gradient for parameter so loss stays completely flat (plateau)
        return x * 1.0 + self.param * 0.0


class TestReduceLROnPlateau(unittest.TestCase):
    """Test suite for ReduceLROnPlateau dynamic learning rate scheduler and zero-validation support."""

    def setUp(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)
        self.n_samples = 10
        self.n_points = 20
        self.x = np.random.uniform(0.1, 1.0, (self.n_samples, self.n_points)).astype(np.float32)
        self.y = self.x * 2.0  # Constant gap to keep loss positive

        self.train_ds = SpectrumPairDataset(self.x, self.y)
        self.val_ds = SpectrumPairDataset(self.x.copy(), self.y.copy())
        self.empty_val_ds = SpectrumPairDataset(
            np.empty((0, self.n_points), dtype=np.float32),
            np.empty((0, self.n_points), dtype=np.float32),
        )

    def test_dynamic_lr_reduction_default_factor_0_9(self) -> None:
        """Verify LR is reduced by default factor 0.9 when val loss plateaus for lr_patience epochs."""
        model = DummyConstantModel(n_points=self.n_points)
        init_lr = 1e-2
        config = {
            "model": model,
            "epochs": 6,
            "batch_size": 4,
            "learning_rate": init_lr,
            "use_lr_scheduler": True,
            "lr_scheduler_patience": 2,  # Reduce after 2 non-improving epochs
            "early_stopping_patience": 10,  # Do not early stop before scheduler acts
            "verbose": False,
        }

        res = train_model_region(self.train_ds, self.val_ds, config=config)
        lr_history = res["history"]["lr"]

        self.assertEqual(len(lr_history), 6)
        self.assertAlmostEqual(lr_history[0], init_lr, places=6)
        # By epoch 4 (2 epochs patience + 1), LR should drop to 0.9 * init_lr
        self.assertAlmostEqual(lr_history[-1], init_lr * 0.9, places=6)
        self.assertLess(lr_history[-1], init_lr)

    def test_dynamic_lr_default_min_lr_is_one_ten_thousandth(self) -> None:
        """Verify min_lr defaults to initial_lr / 10000.0."""
        model = DummyConstantModel(n_points=self.n_points)
        init_lr = 2e-3
        config = {
            "model": model,
            "epochs": 1,
            "learning_rate": init_lr,
            "use_lr_scheduler": True,
            "verbose": False,
        }
        res = train_model_region(self.train_ds, self.val_ds, config=config)
        resolved_cfg = res["config"]
        expected_min_lr = init_lr / 10000.0
        self.assertAlmostEqual(resolved_cfg["min_lr"], expected_min_lr, places=8)

    def test_dynamic_lr_custom_factor_and_explicit_min_lr(self) -> None:
        """Verify custom reduction factor and explicit min_lr are respected."""
        model = DummyConstantModel(n_points=self.n_points)
        init_lr = 1e-2
        custom_min_lr = 1e-3
        config = {
            "model": model,
            "epochs": 10,
            "batch_size": 10,
            "learning_rate": init_lr,
            "lr_reduce_factor": 0.5,
            "lr_scheduler_patience": 1,
            "min_lr": custom_min_lr,
            "early_stopping_patience": 10,
            "verbose": False,
        }
        res = train_model_region(self.train_ds, self.val_ds, config=config)
        lr_history = res["history"]["lr"]
        # LR will step down: 1e-2 -> 5e-3 -> 2.5e-3 -> 1.25e-3 -> min_lr (1e-3)
        self.assertAlmostEqual(lr_history[-1], custom_min_lr, places=5)

    def test_zero_validation_dataset_monitors_train_loss(self) -> None:
        """Verify zero validation dataset does not raise error and monitors train loss for LR reduction."""
        model = DummyConstantModel(n_points=self.n_points)
        init_lr = 1e-2
        config = {
            "model": model,
            "epochs": 5,
            "batch_size": 10,
            "learning_rate": init_lr,
            "use_lr_scheduler": True,
            "lr_scheduler_patience": 2,
            "early_stopping_patience": 10,
            "verbose": False,
        }
        # Run training with empty validation dataset
        res = train_model_region(self.train_ds, self.empty_val_ds, config=config)

        self.assertIn("history", res)
        self.assertEqual(len(res["history"]["train_loss"]), 5)
        # val_loss history should contain NaNs since no validation set exists
        for vl in res["history"]["val_loss"]:
            self.assertTrue(np.isnan(vl))

        # Scheduler stepped on train_loss and reduced LR
        lr_history = res["history"]["lr"]
        self.assertLess(lr_history[-1], init_lr)
        self.assertAlmostEqual(lr_history[-1], init_lr * 0.9, places=6)

    def test_disabled_scheduler_preserves_constant_lr(self) -> None:
        """Verify scheduler disabled keeps learning rate constant."""
        model = DummyConstantModel(n_points=self.n_points)
        init_lr = 1e-3
        config = {
            "model": model,
            "epochs": 5,
            "batch_size": 4,
            "learning_rate": init_lr,
            "use_learning_rate_scheduler": False,
            "learning_rate_scheduler_patience": 1,
            "early_stopping_patience": 10,
            "verbose": False,
        }
        res = train_model_region(self.train_ds, self.val_ds, config=config)
        lr_history = res["history"]["learning_rate"]
        for lr_val in lr_history:
            self.assertEqual(lr_val, init_lr)

    def test_none_validation_dataset_monitors_train_loss(self) -> None:
        """Verify val_dataset=None is accepted and monitors train_loss."""
        model = DummyConstantModel(n_points=self.n_points)
        init_lr = 1e-2
        config = {
            "model": model,
            "epochs": 4,
            "batch_size": 10,
            "learning_rate": init_lr,
            "use_learning_rate_scheduler": True,
            "learning_rate_scheduler_patience": 1,
            "early_stopping_patience": 10,
            "verbose": False,
        }
        res = train_model_region(self.train_ds, None, config=config)
        self.assertIn("history", res)
        self.assertEqual(len(res["history"]["train_loss"]), 4)
        for vl in res["history"]["val_loss"]:
            self.assertTrue(np.isnan(vl))
        self.assertLess(res["history"]["learning_rate"][-1], init_lr)

    def test_deprecated_aliases_raise_keyerror(self) -> None:
        """Verify passing deprecated aliases raises KeyError."""
        model = DummyConstantModel(n_points=self.n_points)
        for deprecated_key in ["lr", "lr_scheduler", "lr_factor", "lr_patience", "lr_min"]:
            config = {
                "model": model,
                "epochs": 2,
                "batch_size": 4,
                deprecated_key: 1e-2,
            }
            with self.assertRaises(KeyError) as ctx:
                train_model_region(self.train_ds, self.val_ds, config=config)
            self.assertIn("Missing required canonical key", str(ctx.exception))

    def test_scheduler_canonical_configuration(self) -> None:
        """Verify canonical scheduler parameters are recognized properly."""
        model = DummyConstantModel(n_points=self.n_points)
        init_lr = 1e-2
        config = {
            "model": model,
            "epochs": 4,
            "batch_size": 10,
            "learning_rate": init_lr,
            "use_learning_rate_scheduler": True,
            "learning_rate_reduce_factor": 0.8,
            "learning_rate_scheduler_patience": 1,
            "min_learning_rate": 1e-4,
            "early_stopping_patience": 10,
            "verbose": False,
        }
        res = train_model_region(self.train_ds, self.val_ds, config=config)
        resolved_cfg = res["config"]
        self.assertTrue(resolved_cfg["use_learning_rate_scheduler"])
        self.assertAlmostEqual(resolved_cfg["learning_rate_reduce_factor"], 0.8, places=5)
        self.assertEqual(resolved_cfg["learning_rate_scheduler_patience"], 1)
        self.assertAlmostEqual(resolved_cfg["min_learning_rate"], 1e-4, places=5)
        # Should reduce by 0.8 factor
        self.assertAlmostEqual(res["history"]["learning_rate"][-1], init_lr * 0.8, places=5)

    def test_invalid_scheduler_parameters_raise_error(self) -> None:
        """Verify invalid scheduler parameters raise ValueError."""
        model = DummyConstantModel(n_points=self.n_points)
        base_cfg = {"model": model, "epochs": 2, "learning_rate": 1e-3}

        # factor <= 0 or >= 1
        with self.assertRaises(ValueError):
            train_model_region(
                self.train_ds, self.val_ds, config={**base_cfg, "learning_rate_reduce_factor": 0.0}
            )
        with self.assertRaises(ValueError):
            train_model_region(
                self.train_ds, self.val_ds, config={**base_cfg, "learning_rate_reduce_factor": 1.5}
            )

        # patience < 1
        with self.assertRaises(ValueError):
            train_model_region(
                self.train_ds, self.val_ds, config={**base_cfg, "learning_rate_scheduler_patience": 0}
            )

        # min_lr < 0
        with self.assertRaises(ValueError):
            train_model_region(
                self.train_ds, self.val_ds, config={**base_cfg, "min_learning_rate": -1e-4}
            )

        # learning_rate < min_learning_rate
        with self.assertRaises(ValueError):
            train_model_region(
                self.train_ds,
                self.val_ds,
                config={**base_cfg, "learning_rate": 1e-4, "min_learning_rate": 1e-3},
            )

    def test_warning_emitted_when_early_stopping_patience_le_lr_scheduler_patience(self) -> None:
        """Verify warning when stopping patience <= scheduler patience."""
        model = DummyConstantModel(n_points=self.n_points)
        config = {
            "model": model,
            "epochs": 15,
            "learning_rate": 1e-3,
            "learning_rate_scheduler_patience": 5,
            "early_stopping_patience": 5,
        }
        with self.assertWarns(UserWarning):
            train_model_region(self.train_ds, self.val_ds, config=config)


if __name__ == "__main__":
    unittest.main()

