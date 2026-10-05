"""Unit tests for cost function definitions in models/cost.py."""

from __future__ import annotations

import unittest
import torch
import torch.nn as nn

from models.cost import CompositeSpectralLoss, NormalizedMSELoss, format_loss_log10


class TestCostFunctions(unittest.TestCase):
    """Test suite for NormalizedMSELoss and CompositeSpectralLoss."""

    def test_normalized_mse_backward_compatibility(self) -> None:
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
        """Verify cosine shape loss is 0.0 when shapes are identical up to arbitrary scalar scaling."""
        criterion = CompositeSpectralLoss(w_shape=1.0, w_mse=0.0, w_scale=0.0)

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

    def test_scale_penalty_triggers_on_amplitude_discrepancy(self) -> None:
        """Verify w_scale penalizes peak amplitude ratio mismatch."""
        criterion_no_scale = CompositeSpectralLoss(w_shape=1.0, w_mse=0.0, w_scale=0.0)
        criterion_with_scale = CompositeSpectralLoss(w_shape=1.0, w_mse=0.0, w_scale=1.0)

        y_true = torch.tensor([[2.0, 10.0, 4.0]])
        y_pred = y_true * 1.5  # 50% scale mismatch

        loss_no_scale = criterion_no_scale(y_pred, y_true)
        loss_with_scale = criterion_with_scale(y_pred, y_true)

        self.assertAlmostEqual(loss_no_scale.item(), 0.0, places=5)
        # Scale penalty: ((15 / 10) - 1)^2 = 0.5^2 = 0.25
        self.assertAlmostEqual(loss_with_scale.item(), 0.25, places=4)

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
        """Verify format_loss_log10 formats valid and invalid loss values correctly."""
        self.assertIn("log10: -2.0000", format_loss_log10(0.01))
        self.assertIn("log10: N/A", format_loss_log10(0.0))
        self.assertIn("log10: N/A", format_loss_log10(-0.5))


if __name__ == "__main__":
    unittest.main()
