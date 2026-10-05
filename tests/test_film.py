"""Unit tests for Feature-wise Linear Modulation (FiLM) in models/film.py and models/resnet.py."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.cost import NormalizedMSELoss
from models.dataset import SpectrumPairDataset
from models.film import FiLMGenerator, prepare_film_condition_tensor
from models.resnet import ResNet1D
from models.trainer import train_one_epoch
from torch.utils.data import DataLoader


class TestFiLMModulation(unittest.TestCase):
    """Test suite verifying FiLM generator and model conditioning."""

    def test_prepare_film_condition_tensor(self) -> None:
        """Verify conditioning tensor extraction from metadata and scale_x."""
        meta = [
            {"die": 0, "tool": "J4", "tool_target": "J5", "time": "2026-01-01 10:00:00", "time_target": "2026-01-01 14:00:00"},
            {"die": 5, "tool": "J4", "tool_target": "J5", "time": "2026-01-02 10:00:00", "time_target": "2026-01-03 10:00:00"},
        ]
        scales = np.array([[1000.0], [5000.0]], dtype=np.float32)

        cond = prepare_film_condition_tensor(metadata=meta, scale_x=scales, num_samples=2)

        self.assertIn("die", cond)
        self.assertIn("tool_src", cond)
        self.assertIn("tool_tgt", cond)
        self.assertIn("log_flux_src", cond)
        self.assertIn("delta_time_days", cond)

        # Die values
        self.assertEqual(cond["die"][0].item(), 0)
        self.assertEqual(cond["die"][1].item(), 5)

        # log_flux_src: log10(1000) = 3.0, log10(5000) ~ 3.699
        self.assertAlmostEqual(cond["log_flux_src"][0, 0].item(), 3.0, places=2)

        # Delta time: 4 hours = 4 / 24 = 0.1667 days; 24 hours = 1.0 day
        self.assertAlmostEqual(cond["delta_time_days"][0, 0].item(), 4.0 / 24.0, places=3)
        self.assertAlmostEqual(cond["delta_time_days"][1, 0].item(), 1.0, places=3)

    def test_film_generator_identity_initialization(self) -> None:
        """Verify FiLMGenerator outputs gamma ~ 1.0 and beta ~ 0.0 at initialization."""
        film_gen = FiLMGenerator(channels_list=[16, 32])
        meta = [{"die": 2, "tool": "J4", "tool_target": "J5"}]
        scales = np.array([[200.0]], dtype=np.float32)
        cond = prepare_film_condition_tensor(metadata=meta, scale_x=scales)

        mods = film_gen(cond)
        self.assertEqual(len(mods), 2)

        gamma1, beta1 = mods[0]
        gamma2, beta2 = mods[1]

        self.assertEqual(gamma1.shape, (1, 16))
        self.assertEqual(beta1.shape, (1, 16))
        self.assertEqual(gamma2.shape, (1, 32))
        self.assertEqual(beta2.shape, (1, 32))

        # Identity initialization check
        np.testing.assert_allclose(gamma1.detach().numpy(), 1.0, atol=1e-5)
        np.testing.assert_allclose(beta1.detach().numpy(), 0.0, atol=1e-5)
        np.testing.assert_allclose(gamma2.detach().numpy(), 1.0, atol=1e-5)
        np.testing.assert_allclose(beta2.detach().numpy(), 0.0, atol=1e-5)

    def test_resnet_with_film_forward_and_backward(self) -> None:
        """Verify ResNet1D with use_film=True runs forward pass and gradients update FiLM parameters."""
        n_points = 50
        model = ResNet1D(n_points=n_points, config={"hidden_channels": 16, "use_film": True})

        x = torch.randn(4, n_points)
        meta = [{"die": i, "tool": "J4", "tool_target": "J5"} for i in range(4)]
        scales = np.full((4, 1), 100.0, dtype=np.float32)
        cond = prepare_film_condition_tensor(metadata=meta, scale_x=scales, num_samples=4)

        out = model(x, cond=cond)
        self.assertEqual(out.shape, (4, n_points))

        # Backward pass
        loss = torch.sum(out ** 2)
        loss.backward()

        # Check gradients exist on FiLM embedding and MLP weights
        self.assertIsNotNone(model.film_gen.die_embedding.weight.grad)
        self.assertIsNotNone(model.film_gen.mlp[0].weight.grad)

    def test_resnet_backward_compatibility_without_film(self) -> None:
        """Verify ResNet1D without use_film behaves identically as unconditioned model."""
        n_points = 40
        model = ResNet1D(n_points=n_points, config={"hidden_channels": 8, "use_film": False})
        x = torch.ones(2, n_points)
        out = model(x)
        self.assertEqual(out.shape, (2, n_points))

    def test_trainer_integration_with_film(self) -> None:
        """Verify train_one_epoch executes successfully when model has use_film=True."""
        n_points = 30
        x_raw = np.ones((8, n_points), dtype=np.float32)
        y_raw = np.ones((8, n_points), dtype=np.float32) * 1.05
        meta = [
            {"die": i % 9, "tool": "J4", "tool_target": "J5", "spectrum_index": i}
            for i in range(8)
        ]
        ds = SpectrumPairDataset(x_raw, y_raw, metadata=meta)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        model = ResNet1D(n_points=n_points, config={"hidden_channels": 8, "use_film": True})
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = NormalizedMSELoss()

        loss = train_one_epoch(model, loader, optimizer, criterion)
        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))

    def test_residual_unet_with_film(self) -> None:
        """Verify ResidualUNet1D with use_film=True forward and backward pass."""
        from models.unet import ResidualUNet1D
        n_points = 32
        model = ResidualUNet1D(n_points=n_points, config={"base_channels": 8, "depth": 2, "use_film": True})
        x = torch.randn(2, n_points)
        meta = [{"die": 1, "tool": "J4", "tool_target": "J5"}, {"die": 4, "tool": "J4", "tool_target": "J5"}]
        cond = prepare_film_condition_tensor(metadata=meta, num_samples=2)

        out = model(x, cond=cond)
        self.assertEqual(out.shape, (2, n_points))
        loss = torch.sum(out ** 2)
        loss.backward()
        self.assertIsNotNone(model.film_gen.die_embedding.weight.grad)

    def test_prepare_film_condition_tensor_dict_and_scalar_edge_cases(self) -> None:
        """Verify robust handling of scalar tool strings, None values, single timestamps, and scalar scale_x."""
        # 1. Dict metadata with scalar string tool, None target, and scalar scale_x with num_samples=3
        cond1 = prepare_film_condition_tensor(
            metadata={"die": [0, 1, 2], "tool": "J4", "tool_target": None},
            scale_x=torch.tensor(100.0),
            num_samples=3,
        )
        self.assertEqual(cond1["die"].shape, (3,))
        self.assertEqual(cond1["tool_src"].shape, (3,))
        self.assertEqual(cond1["tool_tgt"].shape, (3,))
        self.assertEqual(cond1["log_flux_src"].shape, (3, 1))
        # Ensure 'J4' mapped to index 0, not UNKNOWN
        self.assertEqual(cond1["tool_src"][0].item(), 0)

        # 2. Single timestamp strings in dict
        cond2 = prepare_film_condition_tensor(
            metadata={
                "die": [0, 1],
                "tool": ["J4", "J4"],
                "time": "2026-01-01 00:00:00",
                "time_target": "2026-01-01 12:00:00",
            },
            num_samples=2,
        )
        self.assertEqual(cond2["delta_time_days"].shape, (2, 1))
        self.assertAlmostEqual(cond2["delta_time_days"][0, 0].item(), 0.5, places=3)

        # 3. List of dicts with None values
        cond3 = prepare_film_condition_tensor(
            metadata=[{"die": None, "tool": None, "tool_target": None, "time": None}],
            num_samples=1,
        )
        self.assertEqual(cond3["die"][0].item(), 0)
        self.assertEqual(cond3["tool_src"][0].item(), 0)
        self.assertEqual(cond3["tool_tgt"][0].item(), 1)

        # 4. Pandas DataFrame input
        df = pd.DataFrame([{"die": 3, "tool": "J4", "tool_target": "J5"}])
        cond4 = prepare_film_condition_tensor(metadata=df, scale_x=np.array([500.0]))
        self.assertEqual(cond4["die"][0].item(), 3)
        self.assertEqual(cond4["tool_src"][0].item(), 0)
        self.assertEqual(cond4["tool_tgt"][0].item(), 1)


if __name__ == "__main__":
    unittest.main()
