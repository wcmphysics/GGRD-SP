"""Unit tests for Feature-wise Linear Modulation (FiLM) in models."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.cost import SpectralLoss
from models.dataset import SpectrumPairDataset
from models.film import FiLMGenerator, prepare_film_condition_tensor
from models.resnet import ResNet1D
from models.trainer import train_one_epoch
from torch.utils.data import DataLoader


class TestFiLMModulation(unittest.TestCase):
    """Test suite verifying FiLM generator and model conditioning."""

    def test_prepare_film_condition_tensor(self) -> None:
        """Verify conditioning tensor extraction from metadata."""
        meta = [
            {
                "die": 0,
                "tool_index": 0,
                "tool_target_index": 1,
                "die_flux_normalized": -0.5,
                "time_normalized": -1.0,
            },
            {
                "die": 5,
                "tool_index": 0,
                "tool_target_index": 1,
                "die_flux_normalized": 1.25,
                "time_normalized": 1.0,
            },
        ]

        cond = prepare_film_condition_tensor(
            metadata=meta, config={"num_samples": 2}
        )

        self.assertIn("die", cond)
        self.assertIn("tool_src", cond)
        self.assertIn("tool_tgt", cond)
        self.assertIn("die_flux_normalized", cond)
        self.assertIn("time_normalized", cond)

        self.assertEqual(cond["die"][0].item(), 0)
        self.assertEqual(cond["die"][1].item(), 5)
        self.assertEqual(cond["tool_src"][0].item(), 0)
        self.assertEqual(cond["tool_tgt"][0].item(), 1)

        self.assertEqual(cond["die_flux_normalized"].shape, (2, 1))
        self.assertEqual(cond["time_normalized"].shape, (2, 1))
        self.assertAlmostEqual(
            cond["die_flux_normalized"][0, 0].item(), -0.5, places=4
        )
        self.assertAlmostEqual(
            cond["die_flux_normalized"][1, 0].item(), 1.25, places=4
        )
        self.assertAlmostEqual(
            cond["time_normalized"][0, 0].item(), -1.0, places=4
        )
        self.assertAlmostEqual(
            cond["time_normalized"][1, 0].item(), 1.0, places=4
        )

    def test_film_generator_identity_initialization(self) -> None:
        """Verify FiLMGenerator outputs gamma=1.0 and beta=0.0 initially."""
        film_gen = FiLMGenerator(channels_list=[16, 32])
        meta = [
            {
                "die": 2,
                "tool_index": 0,
                "tool_target_index": 1,
                "die_flux_normalized": 0.2,
                "time_normalized": -0.3,
            }
        ]
        cond = prepare_film_condition_tensor(metadata=meta)

        mods = film_gen(cond)
        self.assertEqual(len(mods), 2)

        gamma1, beta1 = mods[0]
        gamma2, beta2 = mods[1]

        self.assertEqual(gamma1.shape, (1, 16))
        self.assertEqual(beta1.shape, (1, 16))
        self.assertEqual(gamma2.shape, (1, 32))
        self.assertEqual(beta2.shape, (1, 32))

        np.testing.assert_allclose(gamma1.detach().numpy(), 1.0, atol=1e-5)
        np.testing.assert_allclose(beta1.detach().numpy(), 0.0, atol=1e-5)
        np.testing.assert_allclose(gamma2.detach().numpy(), 1.0, atol=1e-5)
        np.testing.assert_allclose(beta2.detach().numpy(), 0.0, atol=1e-5)

    def test_resnet_with_film_forward_and_backward(self) -> None:
        """Verify ResNet1D with use_film=True runs forward pass and gradients
        update FiLM parameters.
        """
        n_points = 50
        model = ResNet1D(
            n_points=n_points, config={"resnet_channel": 16, "use_film": True}
        )

        x = torch.randn(4, 1, n_points)
        meta = [
            {
                "die": i,
                "tool_index": 0,
                "tool_target_index": 1,
                "die_flux_normalized": 0.1 * i,
                "time_normalized": -0.2 * i,
            }
            for i in range(4)
        ]
        cond = prepare_film_condition_tensor(
            metadata=meta, config={"num_samples": 4}
        )

        out = model(x, cond=cond)
        self.assertEqual(out.shape, (4, 1, n_points))

        loss = torch.sum(out**2)
        loss.backward()

        self.assertIsNotNone(model.film_gen.die_embedding.weight.grad)
        self.assertIsNotNone(model.film_gen.mlp[0].weight.grad)

    def test_resnet_without_film(self) -> None:
        """Verify ResNet1D without use_film behaves identically as
        unconditioned model.
        """
        n_points = 40
        model = ResNet1D(
            n_points=n_points, config={"resnet_channel": 8, "use_film": False}
        )
        x = torch.ones(2, 1, n_points)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, n_points))

    def test_trainer_integration_with_film(self) -> None:
        """Verify train_one_epoch runs cleanly when model has use_film=True."""
        n_points = 30
        x_raw = np.ones((8, n_points), dtype=np.float32)
        y_raw = np.ones((8, n_points), dtype=np.float32) * 1.05
        meta = [
            {
                "die": i % 9,
                "tool_index": 0,
                "tool_target_index": 1,
                "die_flux_normalized": float(i) * 0.1,
                "time_normalized": float(i) * 0.2,
                "spectrum_index": i,
            }
            for i in range(8)
        ]
        ds = SpectrumPairDataset(x_raw, y_raw, metadata=meta)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        model = ResNet1D(
            n_points=n_points, config={"resnet_channel": 8, "use_film": True}
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = SpectralLoss()

        loss = train_one_epoch(model, loader, optimizer, criterion)
        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))

    def test_residual_unet_with_film(self) -> None:
        """Verify ResidualUNet1D with use_film=True forward and backward."""
        from models.unet import ResidualUNet1D

        n_points = 32
        model = ResidualUNet1D(
            n_points=n_points,
            config={"residual_unet_channel": 8, "depth": 2, "use_film": True},
        )
        x = torch.randn(2, 1, n_points)
        meta = [
            {
                "die": 1,
                "tool_index": 0,
                "tool_target_index": 1,
                "die_flux_normalized": -0.4,
                "time_normalized": 0.5,
            },
            {
                "die": 4,
                "tool_index": 0,
                "tool_target_index": 1,
                "die_flux_normalized": 0.4,
                "time_normalized": -0.5,
            },
        ]
        cond = prepare_film_condition_tensor(
            metadata=meta, config={"num_samples": 2}
        )

        out = model(x, cond=cond)
        self.assertEqual(out.shape, (2, 1, n_points))
        loss = torch.sum(out**2)
        loss.backward()
        self.assertIsNotNone(model.film_gen.die_embedding.weight.grad)

    def test_prepare_film_condition_tensor_dict_and_scalar_edge_cases(self) -> None:
        """Verify handling of tool indices, flags, and DataFrame inputs."""
        cond1 = prepare_film_condition_tensor(
            metadata={
                "die": [0, 1, 2],
                "tool_index": 0,
                "tool_target_index": 1,
                "die_flux_normalized": [0.1, 0.2, 0.3],
                "time_normalized": -0.5,
            },
            config={"num_samples": 3},
        )
        self.assertEqual(cond1["die"].shape, (3,))
        self.assertEqual(cond1["tool_src"].shape, (3,))
        self.assertEqual(cond1["tool_tgt"].shape, (3,))
        self.assertEqual(cond1["die_flux_normalized"].shape, (3, 1))
        self.assertEqual(cond1["time_normalized"].shape, (3, 1))

        # Pandas DataFrame input
        df = pd.DataFrame(
            [
                {
                    "die": 3,
                    "tool_index": 0,
                    "tool_target_index": 1,
                    "die_flux_normalized": 1.5,
                    "time_normalized": -1.2,
                }
            ]
        )
        cond3 = prepare_film_condition_tensor(metadata=df)
        self.assertEqual(cond3["die"][0].item(), 3)
        self.assertAlmostEqual(
            cond3["die_flux_normalized"][0, 0].item(), 1.5, places=4
        )
        self.assertAlmostEqual(
            cond3["time_normalized"][0, 0].item(), -1.2, places=4
        )

    def test_missing_metadata_fields_raise_loud_error(self) -> None:
        """Verify prepare_film_condition_tensor raises ValueError when required
        fields are missing or null.
        """
        base_meta = {
            "die": 0,
            "tool_index": 0,
            "tool_target_index": 1,
            "die_flux_normalized": 0.0,
            "time_normalized": 0.0,
        }
        for field in (
            "tool_index",
            "tool_target_index",
            "die_flux_normalized",
            "time_normalized",
        ):
            bad_meta = {k: v for k, v in base_meta.items() if k != field}
            with self.assertRaises(ValueError) as ctx:
                prepare_film_condition_tensor(metadata=[bad_meta])
            self.assertIn(field, str(ctx.exception))

        with self.assertRaises(ValueError):
            prepare_film_condition_tensor(metadata=None)

    def test_film_generator_feature_toggles_and_dimensions(self) -> None:
        """Verify FiLM input dimension adapts to encode_flux, encode_time, and
        encode_window_energy flags.
        """
        # Flux, time, and sliding window energy on: 4 + 8 + 1 + 1 + 2 = 16
        fg_all = FiLMGenerator(
            [16],
            config={"use_sliding_window": True, "encode_window_energy": True},
        )
        self.assertEqual(fg_all.mlp[0].in_features, 16)

        # Sliding window off: 4 + 8 + 1 + 1 = 14
        fg_default = FiLMGenerator([16])
        self.assertEqual(fg_default.mlp[0].in_features, 14)

        # Flux off, time on: 4 + 8 + 0 + 1 = 13
        fg_no_flux = FiLMGenerator([16], config={"encode_flux": False})
        self.assertEqual(fg_no_flux.mlp[0].in_features, 13)
        cond_no_flux = prepare_film_condition_tensor(
            metadata=[
                {
                    "die": 0,
                    "tool_index": 0,
                    "tool_target_index": 1,
                    "time_normalized": 0.5,
                }
            ],
            config={"encode_flux": False},
        )
        self.assertNotIn("die_flux_normalized", cond_no_flux)
        self.assertEqual(len(fg_no_flux(cond_no_flux)), 1)

        # Both flux and time off: 4 + 8 = 12
        fg_neither = FiLMGenerator(
            [16], config={"encode_flux": False, "encode_time": False}
        )
        self.assertEqual(fg_neither.mlp[0].in_features, 12)
        cond_neither = prepare_film_condition_tensor(
            metadata=[{"die": 0, "tool_index": 0, "tool_target_index": 1}],
            config={"encode_flux": False, "encode_time": False},
        )
        self.assertEqual(len(fg_neither(cond_neither)), 1)

    def test_prepare_film_condition_tensor_with_window_energy(self) -> None:
        """Verify window energy bounds are scaled by energy_scale_ev."""
        meta = [
            {
                "die": 0,
                "tool_index": 0,
                "tool_target_index": 1,
                "die_flux_normalized": 0.0,
                "time_normalized": 0.0,
            }
        ]
        we = np.array([[100.0, 106.0]])

        cond_default = prepare_film_condition_tensor(
            metadata=meta,
            window_energy=we,
        )
        self.assertIn("window_energy", cond_default)
        self.assertAlmostEqual(
            cond_default["window_energy"][0, 0].item(), 0.1, places=4
        )
        self.assertAlmostEqual(
            cond_default["window_energy"][0, 1].item(), 0.106, places=4
        )

        cond_custom = prepare_film_condition_tensor(
            metadata=meta,
            window_energy=we,
            config={"energy_scale_ev": 500.0},
        )
        self.assertAlmostEqual(
            cond_custom["window_energy"][0, 0].item(), 0.2, places=4
        )
        self.assertAlmostEqual(
            cond_custom["window_energy"][0, 1].item(), 0.212, places=4
        )

    def test_spectrum_patch_dataset_yields_window_energy(self) -> None:
        """Verify SpectrumPatchDataset extracts window energy bounds."""
        from models.dataset import SpectrumPatchDataset

        n_pts = 30
        x = np.ones((2, n_pts), dtype=np.float32)
        y = np.ones((2, n_pts), dtype=np.float32)
        energy = np.linspace(70.0, 100.0, n_pts, dtype=np.float32)
        full_ds = SpectrumPairDataset(
            x,
            y,
            energy=np.tile(energy, (2, 1)),
            metadata=[
                {
                    "die": 0,
                    "tool_index": 0,
                    "tool_target_index": 1,
                    "die_flux_normalized": 0.0,
                    "time_normalized": 0.0,
                }
            ]
            * 2,
        )
        patch_ds = SpectrumPatchDataset(full_ds, window_size=10, stride=5)
        self.assertGreater(len(patch_ds), 0)
        item = patch_ds[0]
        self.assertIn("window_energy", item)
        self.assertEqual(item["window_energy"].shape, (2,))
        self.assertAlmostEqual(
            item["window_energy"][0].item(), energy[0], places=3
        )
        self.assertAlmostEqual(
            item["window_energy"][1].item(), energy[9], places=3
        )

    def test_resnet_with_window_energy_film_forward_backward(self) -> None:
        """Verify ResNet1D forward and backward pass with window energy FiLM."""
        patch_len = 10
        model = ResNet1D(
            n_points=patch_len,
            config={
                "resnet_channel": 16,
                "use_film": True,
                "use_sliding_window": True,
                "encode_window_energy": True,
                "energy_scale_ev": 1000.0,
            },
        )
        x = torch.randn(2, 1, patch_len)
        meta = [
            {
                "die": 0,
                "tool_index": 0,
                "tool_target_index": 1,
                "die_flux_normalized": 0.1,
                "time_normalized": -0.1,
            }
        ] * 2
        we = torch.tensor([[70.0, 80.0], [75.0, 85.0]], dtype=torch.float32)
        cond = prepare_film_condition_tensor(
            metadata=meta,
            window_energy=we,
            config={"num_samples": 2, "energy_scale_ev": 1000.0},
        )
        out = model(x, cond=cond)
        self.assertEqual(out.shape, (2, 1, patch_len))
        loss = torch.sum(out**2)
        loss.backward()
        self.assertIsNotNone(model.film_gen.mlp[0].weight.grad)

    def test_predict_spectra_and_sliding_window_with_film(self) -> None:
        """Verify predict_spectra and evaluate_sliding_window with FiLM active."""
        from models.inference import predict_spectra
        from models.trainer import evaluate_sliding_window

        n_pts = 30
        x = np.ones((4, n_pts), dtype=np.float32) * 0.5
        y = np.ones((4, n_pts), dtype=np.float32) * 0.6
        energy = np.tile(
            np.linspace(70.0, 100.0, n_pts, dtype=np.float32), (4, 1)
        )
        meta_records = [
            {
                "spectrum_index": i,
                "measurement_id": f"NMG_M_J4_{i:05d}",
                "tool": "J4",
                "tool_index": 0,
                "tool_target": "H1",
                "tool_target_index": 1,
                "die": i % 2,
                "region": "Al2p",
                "split": "test",
                "die_flux_normalized": 0.1 * i,
                "time_normalized": -0.2 * i,
            }
            for i in range(4)
        ]
        val_ds = SpectrumPairDataset(x, y, energy=energy, metadata=meta_records)
        sw_model = ResNet1D(
            n_points=10,
            config={
                "resnet_channel": 8,
                "kernel_size": 3,
                "use_film": True,
                "use_sliding_window": True,
                "encode_window_energy": True,
            },
        )
        sw_loss = evaluate_sliding_window(
            sw_model,
            val_ds,
            SpectralLoss(),
            config={"window_size": 10, "stride": 5},
        )
        self.assertIsInstance(sw_loss, float)
        self.assertFalse(np.isnan(sw_loss))

        pred_i, pred_e, pred_df = predict_spectra(
            models={"Al2p": (sw_model, 10, 5)},
            data=(x, energy, pd.DataFrame(meta_records)),
            config={"source_tool": "J4", "target_tool": "H1"},
        )
        self.assertEqual(pred_i.shape, (4, n_pts))
        self.assertEqual(pred_e.shape, (4, n_pts))
        self.assertIn("die_flux_normalized", pred_df.columns)
        self.assertIn("time_normalized", pred_df.columns)


if __name__ == "__main__":
    unittest.main()
