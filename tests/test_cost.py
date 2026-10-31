"""Unit tests for cost function definitions in models/cost.py."""

from __future__ import annotations

import unittest
import torch
import torch.nn as nn

from models.cost import SpectralLoss, format_loss_log10


class TestCostFunctions(unittest.TestCase):
    """Test suite for SpectralLoss and logging utilities."""

    def test_input_validation(self) -> None:
        """Verify 2D tensors are rejected with informative ValueError."""
        criterion = SpectralLoss()
        y_2d = torch.tensor([[1.0, 2.0, 3.0]])
        y_3d = torch.tensor([[[1.0, 2.0, 3.0]]])
        with self.assertRaises(ValueError):
            criterion(y_2d, y_3d)
        with self.assertRaises(ValueError):
            criterion(y_3d, y_2d)

    def test_loss_components_and_weighting(self) -> None:
        """Verify MSE, Cosine Shape, Gradient MSE, and normalization options.
        """
        # Test 1: Standard Normalized MSE
        y_true = torch.tensor([[[10.0, 20.0, 10.0]]])
        y_pred = torch.tensor([[[5.0, 20.0, 10.0]]])
        crit_mse = SpectralLoss({"w_mse": 1.0, "w_shape": 0.0, "w_gradient": 0.0})
        self.assertAlmostEqual(crit_mse(y_true, y_true).item(), 0.0, places=6)
        self.assertAlmostEqual(crit_mse(y_pred, y_true).item(), 0.0625 / 3.0, places=5)

        # Test 2: Unnormalized MSE toggle
        crit_unnorm = SpectralLoss({"normalized_by_target_max": False})
        self.assertAlmostEqual(crit_unnorm(y_pred, y_true).item(), 25.0 / 3.0, places=5)

        # Test 3: Cosine Shape scale-invariance and non-negative clamping
        crit_shape = SpectralLoss({"w_mse": 0.0, "w_shape": 1.0})
        y_scaled = y_true * 2.5
        self.assertAlmostEqual(crit_shape(y_scaled, y_true).item(), 0.0, places=5)
        self.assertGreaterEqual(crit_shape(y_true, y_true).item(), 0.0)

        # Test 4: Gradient MSE with central differences (normalized vs
        # unnormalized)
        y_ramp_true = torch.tensor([[[0.0, 4.0, 8.0, 12.0]]])
        y_ramp_pred = torch.tensor([[[0.0, 2.0, 4.0, 6.0]]])
        crit_grad_norm = SpectralLoss(
            {"w_mse": 0.0, "w_gradient": 1.0, "normalized_by_target_max": True}
        )
        self.assertAlmostEqual(crit_grad_norm(y_ramp_pred, y_ramp_true).item(), 1.0 / 36.0, places=5)

        crit_grad_unnorm = SpectralLoss(
            {"w_mse": 0.0, "w_gradient": 1.0, "normalized_by_target_max": False}
        )
        self.assertAlmostEqual(crit_grad_unnorm(y_ramp_pred, y_ramp_true).item(), 4.0, places=5)

        # Test 5: Composite weights and regional context weighting
        crit_composite = SpectralLoss(
            {"w_mse": 0.5, "w_shape": 0.5, "region_weights": {"Al2p": 2.0}}
        )
        base_loss = crit_composite(y_pred, y_true, loss_context={"region": "Al2p"})
        unweighted_loss = crit_composite(y_pred, y_true)
        self.assertAlmostEqual(base_loss.item(), 2.0 * unweighted_loss.item(), places=5)

    def test_l2_regularization(self) -> None:
        """Verify L2 parameter penalty calculation for models."""
        class SimpleNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv1d(1, 1, kernel_size=3, bias=False)
                nn.init.constant_(self.conv.weight, 2.0)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        net = SimpleNet()
        criterion = SpectralLoss({"w_mse": 1.0, "l2_weight": 0.01})
        y = torch.ones((1, 1, 10))
        # Base loss is 0.0, weight norm = 3 * 4 = 12, L2 penalty = 0.01 * 12
        # = 0.12
        loss = criterion(y, y, model=net)
        self.assertAlmostEqual(loss.item(), 0.12, places=4)

    def test_format_loss_log10(self) -> None:
        """Verify format_loss_log10 outputs raw and log10 strings."""
        self.assertIn("log10: -2.0000", format_loss_log10(0.01))
        self.assertIn("log10: N/A", format_loss_log10(0.0))
        self.assertIn("log10: N/A", format_loss_log10(-0.5))

    def test_dynamic_eps_warning_on_tiny_scale(self) -> None:
        """Verify warning when denominator scale <= eps in loss normalization."""
        criterion = SpectralLoss({"normalized_by_target_max": True})
        y_zero = torch.zeros((1, 1, 10))
        with self.assertWarns(UserWarning):
            loss = criterion(y_zero, y_zero)
        self.assertAlmostEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
