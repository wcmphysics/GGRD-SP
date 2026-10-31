"""Unit tests for ReduceLROnPlateau scheduler and trainer in
models/trainer.py.
"""

from __future__ import annotations

import unittest
import numpy as np
import torch
import torch.nn as nn

from models.dataset import SpectrumPairDataset
from models.trainer import train_model_region


class DummyConstantModel(nn.Module):
    """Dummy 1D model outputting constant predictions to simulate plateau."""

    def __init__(self, n_points: int = 20) -> None:
        super().__init__()
        self.n_points = n_points
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x * 1.0 + self.param * 0.0


class TestReduceLROnPlateau(unittest.TestCase):
    """Test suite for ReduceLROnPlateau dynamic scheduler and validations."""

    def setUp(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)
        self.n_samples = 10
        self.n_points = 20
        self.x = np.random.uniform(0.1, 1.0, (self.n_samples, self.n_points)).astype(np.float32)
        self.y = self.x * 2.0

        self.train_ds = SpectrumPairDataset(self.x, self.y)
        self.val_ds = SpectrumPairDataset(self.x.copy(), self.y.copy())
        self.empty_val_ds = SpectrumPairDataset(
            np.empty((0, self.n_points), dtype=np.float32),
            np.empty((0, self.n_points), dtype=np.float32),
        )

    def test_dynamic_lr_reduction_and_limits(self) -> None:
        """Verify dynamic learning rate reduction and floor enforcement."""
        model = DummyConstantModel(n_points=self.n_points)
        init_lr = 1e-2
        custom_min_lr = 1e-3

        # Test custom reduction factor (0.5) and minimum floor
        config = {
            "model": model,
            "epochs": 10,
            "batch_size": 10,
            "learning_rate": init_lr,
            "learning_rate_reduce_factor": 0.5,
            "learning_rate_scheduler_patience": 1,
            "min_learning_rate": custom_min_lr,
            "early_stopping_patience": 10,
            "verbose": False,
        }
        res = train_model_region(self.train_ds, self.val_ds, config=config)
        lr_history = res["history"]["learning_rate"]
        self.assertAlmostEqual(lr_history[-1], custom_min_lr, places=5)

        # Test default floor (initial_lr / 10000.0)
        res_default = train_model_region(
            self.train_ds,
            self.val_ds,
            config={
                "model": DummyConstantModel(n_points=self.n_points),
                "epochs": 1,
                "learning_rate": 2e-3,
                "use_learning_rate_scheduler": True,
                "verbose": False,
            },
        )
        self.assertAlmostEqual(
            res_default["config"]["min_learning_rate"], 2e-3 / 10000.0, places=8
        )

    def test_validation_fallback_and_scheduler_toggle(self) -> None:
        """Verify empty or None validation fallback and disabled scheduler."""
        model = DummyConstantModel(n_points=self.n_points)
        init_lr = 1e-2

        # Subtest 1: Empty or None validation set monitors train_loss
        for val_set in (self.empty_val_ds, None):
            with self.subTest(val_type="empty" if val_set is not None else "None"):
                res = train_model_region(
                    self.train_ds,
                    val_set,
                    config={
                        "model": model,
                        "epochs": 4,
                        "batch_size": 10,
                        "learning_rate": init_lr,
                        "use_learning_rate_scheduler": True,
                        "learning_rate_scheduler_patience": 1,
                        "early_stopping_patience": 10,
                        "verbose": False,
                    },
                )
                self.assertEqual(len(res["history"]["train_loss"]), 4)
                self.assertTrue(all(np.isnan(vl) for vl in res["history"]["val_loss"]))
                self.assertLess(res["history"]["learning_rate"][-1], init_lr)

        # Subtest 2: Disabled scheduler maintains constant learning rate
        res_disabled = train_model_region(
            self.train_ds,
            self.val_ds,
            config={
                "model": model,
                "epochs": 3,
                "batch_size": 4,
                "learning_rate": init_lr,
                "use_learning_rate_scheduler": False,
                "verbose": False,
            },
        )
        for lr_val in res_disabled["history"]["learning_rate"]:
            self.assertEqual(lr_val, init_lr)

    def test_invalid_scheduler_parameters_and_warnings(self) -> None:
        """Verify invalid scheduler parameter bounds and patience warning."""
        model = DummyConstantModel(n_points=self.n_points)
        base_cfg = {"model": model, "epochs": 2, "learning_rate": 1e-3}

        # Table-driven invalid parameters
        invalid_params = [
            {"learning_rate_reduce_factor": 0.0},
            {"learning_rate_reduce_factor": 1.5},
            {"learning_rate_scheduler_patience": 0},
            {"min_learning_rate": -1e-4},
            {"learning_rate": 1e-4, "min_learning_rate": 1e-3},
        ]
        for bad_cfg in invalid_params:
            with self.subTest(bad_cfg=bad_cfg):
                with self.assertRaises(ValueError):
                    train_model_region(
                        self.train_ds, self.val_ds, config={**base_cfg, **bad_cfg}
                    )

        # Warning when stopping patience <= scheduler patience
        warning_cfg = {
            "model": model,
            "epochs": 15,
            "learning_rate": 1e-3,
            "use_learning_rate_scheduler": True,
            "learning_rate_scheduler_patience": 5,
            "early_stopping_patience": 5,
        }
        with self.assertWarns(UserWarning):
            train_model_region(self.train_ds, self.val_ds, config=warning_cfg)


if __name__ == "__main__":
    unittest.main()
