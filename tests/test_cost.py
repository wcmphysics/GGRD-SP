"""Unit tests for cost function definitions in models/cost.py."""

from __future__ import annotations

import unittest
import torch
import torch.nn as nn

from models.cost import (
    CompositeSpectralLoss,
    NormalizedMSELoss,
    create_cost_function,
    format_loss_log10,
)


class TestCostFunctions(unittest.TestCase):
    """Test suite for NormalizedMSELoss and CompositeSpectralLoss."""

    def test_normalized_mse_computation(self) -> None:
        """Verify default NormalizedMSELoss computes standard normalized MSE."""
        criterion = NormalizedMSELoss()
        y_true = torch.tensor([[10.0, 20.0, 10.0]])
        y_pred = torch.tensor([[10.0, 20.0, 10.0]])
        loss = criterion(y_pred, y_true)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

        # Perturbed prediction
        y_pred_pert = torch.tensor([[5.0, 20.0, 10.0]])
        loss_pert = criterion(y_pred_pert, y_true)
        # norm_true = [0.5, 1.0, 0.5], norm_pred = [0.25, 1.0, 0.5]
        # MSE = ((0.25 - 0.5)^2 + 0 + 0) / 3 = 0.0625 / 3 = 0.020833
        self.assertAlmostEqual(loss_pert.item(), 0.0625 / 3.0, places=5)

    def test_cosine_shape_loss_scale_invariance(self) -> None:
        """Verify cosine shape loss is 0.0 when shapes are identical up to
        arbitrary scalar scaling.
        """
        criterion = CompositeSpectralLoss(w_shape=1.0, w_mse=0.0)

        y_true = torch.tensor([[2.0, 8.0, 4.0, 1.0]])
        # Prediction has completely different scale (e.g. 2.5x higher)
        y_pred = y_true * 2.5

        loss = criterion(y_pred, y_true)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_composite_loss_combination(self) -> None:
        """Verify CompositeSpectralLoss combines w_shape and w_mse correctly."""
        criterion = CompositeSpectralLoss(w_shape=0.5, w_mse=0.5)

        y_true = torch.tensor([[1.0, 5.0, 2.0]])
        y_pred = torch.tensor([[1.0, 5.0, 2.0]])
        loss_zero = criterion(y_pred, y_true)
        self.assertAlmostEqual(loss_zero.item(), 0.0, places=5)

    def test_l2_regularization_included(self) -> None:
        """Verify L2 weight regularization is added when l2_weight > 0."""
        class SimpleNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv1d(1, 1, kernel_size=3, bias=False)
                # Set weights to all 2.0 -> sum(w^2) = 3 * 4 = 12
                nn.init.constant_(self.conv.weight, 2.0)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        net = SimpleNet()
        criterion = CompositeSpectralLoss(w_shape=0.0, w_mse=1.0, l2_weight=0.01)

        y = torch.ones((1, 10))
        loss = criterion(y, y, model=net)
        # Base loss is 0.0, L2 penalty is 0.01 * 12.0 = 0.12
        self.assertAlmostEqual(loss.item(), 0.12, places=4)

    def test_format_loss_log10(self) -> None:
        """Verify format_loss_log10 formats loss values correctly."""
        self.assertIn("log10: -2.0000", format_loss_log10(0.01))
        self.assertIn("log10: N/A", format_loss_log10(0.0))
        self.assertIn("log10: N/A", format_loss_log10(-0.5))

    def test_cosine_sim_clamp_non_negative(self) -> None:
        """Verify loss is non-negative even with exact identical tensors."""
        criterion = CompositeSpectralLoss(w_shape=1.0, w_mse=0.0)
        y = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        loss = criterion(y, y)
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_train_model_region_respects_composite_loss(self) -> None:
        """Verify train_model_region accepts cost_type='composite' and trains
        without errors.
        """
        from models.dataset import SpectrumPairDataset
        from models.trainer import train_model_region
        import numpy as np

        x = np.ones((4, 20), dtype=np.float32)
        y = np.ones((4, 20), dtype=np.float32) * 1.1
        ds = SpectrumPairDataset(x, y)

        res = train_model_region(
            train_dataset=ds,
            val_dataset=ds,
            config={
                "cost_type": "composite",
                "w_shape": 0.5,
                "w_mse": 0.5,
                "epochs": 1,
                "batch_size": 2,
                "model_type": "resnet",
                "hidden_channels": 4,
            },
        )
        self.assertIn("best_val_loss", res)
        self.assertFalse(np.isnan(res["best_val_loss"]))

    def test_normalized_by_target_max_toggle(self) -> None:
        """Verify normalized_by_target_max toggles division by true maximum."""
        y_true = torch.tensor([[10.0, 20.0, 10.0]])
        y_pred = torch.tensor([[5.0, 20.0, 10.0]])

        # Default: normalized_by_target_max=True
        # norm_true = [0.5, 1.0, 0.5], norm_pred = [0.25, 1.0, 0.5]
        # MSE = ((0.25 - 0.5)^2) / 3 = 0.0625 / 3 = 0.020833
        crit_norm = NormalizedMSELoss(normalized_by_target_max=True)
        loss_norm = crit_norm(y_pred, y_true)
        self.assertAlmostEqual(loss_norm.item(), 0.0625 / 3.0, places=5)

        # Disabled: normalized_by_target_max=False
        # Unnormalized MSE = (5.0 - 10.0)^2 / 3 = 25.0 / 3 = 8.333333
        crit_unnorm = NormalizedMSELoss(normalized_by_target_max=False)
        loss_unnorm = crit_unnorm(y_pred, y_true)
        self.assertAlmostEqual(loss_unnorm.item(), 25.0 / 3.0, places=5)

    def test_create_cost_function_respects_normalized_by_target_max(self) -> None:
        """Verify create_cost_function passes normalized_by_target_max."""
        y_true = torch.tensor([[10.0, 20.0, 10.0]])
        y_pred = torch.tensor([[5.0, 20.0, 10.0]])

        crit_unnorm = create_cost_function(
            config={"normalized_by_target_max": False}
        )
        self.assertFalse(crit_unnorm.normalized_by_target_max)
        loss = crit_unnorm(y_pred, y_true)
        self.assertAlmostEqual(loss.item(), 25.0 / 3.0, places=5)

    def test_intensity_gradient_mse_computation(self) -> None:
        """Verify gradient MSE component computed via central difference."""
        y_true = torch.tensor([[0.0, 4.0, 8.0, 12.0]])
        y_pred = torch.tensor([[0.0, 2.0, 4.0, 6.0]])

        # Normalized: true max = 12.0, norm gradients are 1/3 vs 1/6
        # grad diff = -1/6, MSE = 1/36
        crit_norm = NormalizedMSELoss(
            w_mse=0.0, w_gradient=1.0, normalized_by_target_max=True
        )
        loss_norm = crit_norm(y_pred, y_true)
        self.assertAlmostEqual(loss_norm.item(), 1.0 / 36.0, places=5)

        # Unnormalized: true max not divided, gradients are 4.0 vs 2.0
        # grad diff = -2.0, MSE = 4.0
        crit_unnorm = NormalizedMSELoss(
            w_mse=0.0, w_gradient=1.0, normalized_by_target_max=False
        )
        loss_unnorm = crit_unnorm(y_pred, y_true)
        self.assertAlmostEqual(loss_unnorm.item(), 4.0, places=5)

    def test_w_gradient_default_zero(self) -> None:
        """Verify w_gradient defaults to zero and leaves MSE loss unaffected."""
        y_true = torch.tensor([[0.0, 4.0, 8.0, 12.0]])
        y_pred = torch.tensor([[0.0, 2.0, 4.0, 6.0]])

        crit_default = NormalizedMSELoss(w_mse=1.0)
        self.assertEqual(crit_default.w_gradient, 0.0)

        crit_zero = NormalizedMSELoss(w_mse=1.0, w_gradient=0.0)
        loss_def = crit_default(y_pred, y_true).item()
        loss_zero = crit_zero(y_pred, y_true).item()
        self.assertEqual(loss_def, loss_zero)

    def test_create_cost_function_respects_w_gradient(self) -> None:
        """Verify create_cost_function properly configures w_gradient."""
        crit = create_cost_function(config={"w_gradient": 0.25})
        self.assertEqual(crit.w_gradient, 0.25)


if __name__ == "__main__":
    unittest.main()
