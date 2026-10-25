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
from models.resnet import REQUIRED_RESNET_KEYS, ResNet1D
from models.trainer import train_one_epoch
from torch.utils.data import DataLoader

_DEFAULT_RESNET_TEST_CONFIG = {
    "resnet_channel": 16,
    "kernel_size": 3,
    "dropout": 0.0,
    "use_batch_norm": True,
    "padding_mode": "reflect",
    "use_film": False,
}


class TestFiLMModulation(unittest.TestCase):
    """Test suite verifying FiLM generator and model conditioning."""

    def test_prepare_film_condition_tensor(self) -> None:
        """Verify conditioning tensor extraction from metadata."""
        meta = [
            {
                "die": 0,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": -0.5,
                "time_standardized": -1.0,
            },
            {
                "die": 5,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": 1.25,
                "time_standardized": 1.0,
            },
        ]

        cond = prepare_film_condition_tensor(
            metadata=meta, config={"num_samples": 2}
        )

        self.assertIn("die", cond)
        self.assertIn("source_tool_index", cond)
        self.assertIn("target_tool_index", cond)
        self.assertIn("total_flux_single_die_standardized", cond)
        self.assertIn("time_standardized", cond)

        self.assertEqual(cond["die"][0].item(), 0)
        self.assertEqual(cond["die"][1].item(), 5)
        self.assertEqual(cond["source_tool_index"][0].item(), 0)
        self.assertEqual(cond["target_tool_index"][0].item(), 1)

        self.assertEqual(cond["total_flux_single_die_standardized"].shape, (2, 1))
        self.assertEqual(cond["time_standardized"].shape, (2, 1))
        self.assertAlmostEqual(
            cond["total_flux_single_die_standardized"][0, 0].item(), -0.5, places=4
        )
        self.assertAlmostEqual(
            cond["total_flux_single_die_standardized"][1, 0].item(), 1.25, places=4
        )
        self.assertAlmostEqual(
            cond["time_standardized"][0, 0].item(), -1.0, places=4
        )
        self.assertAlmostEqual(
            cond["time_standardized"][1, 0].item(), 1.0, places=4
        )

    def test_film_generator_identity_initialization(self) -> None:
        """Verify FiLMGenerator outputs gamma=1.0 and beta=0.0 initially."""
        film_gen = FiLMGenerator(channels_list=[16, 32])
        meta = [
            {
                "die": 2,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": 0.2,
                "time_standardized": -0.3,
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
            n_points,
            config={
                **_DEFAULT_RESNET_TEST_CONFIG,
                "resnet_channel": 16,
                "use_film": True,
            },
        )

        x = torch.randn(4, 1, n_points)
        meta = [
            {
                "die": i,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": 0.1 * i,
                "time_standardized": -0.2 * i,
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
            n_points,
            config={
                **_DEFAULT_RESNET_TEST_CONFIG,
                "resnet_channel": 8,
                "use_film": False,
            },
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
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": float(i) * 0.1,
                "time_standardized": float(i) * 0.2,
                "spectrum_index": i,
            }
            for i in range(8)
        ]
        ds = SpectrumPairDataset(x_raw, y_raw, metadata=meta)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        model = ResNet1D(
            n_points,
            config={
                **_DEFAULT_RESNET_TEST_CONFIG,
                "resnet_channel": 8,
                "use_film": True,
            },
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
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": -0.4,
                "time_standardized": 0.5,
            },
            {
                "die": 4,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": 0.4,
                "time_standardized": -0.5,
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
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": [0.1, 0.2, 0.3],
                "time_standardized": -0.5,
            },
            config={"num_samples": 3},
        )
        self.assertEqual(cond1["die"].shape, (3,))
        self.assertEqual(cond1["source_tool_index"].shape, (3,))
        self.assertEqual(cond1["target_tool_index"].shape, (3,))
        self.assertEqual(cond1["total_flux_single_die_standardized"].shape, (3, 1))
        self.assertEqual(cond1["time_standardized"].shape, (3, 1))

        # Pandas DataFrame input
        df = pd.DataFrame(
            [
                {
                    "die": 3,
                    "source_tool_index": 0,
                    "target_tool_index": 1,
                    "total_flux_single_die_standardized": 1.5,
                    "time_standardized": -1.2,
                }
            ]
        )
        cond3 = prepare_film_condition_tensor(metadata=df)
        self.assertEqual(cond3["die"][0].item(), 3)
        self.assertAlmostEqual(
            cond3["total_flux_single_die_standardized"][0, 0].item(), 1.5, places=4
        )
        self.assertAlmostEqual(
            cond3["time_standardized"][0, 0].item(), -1.2, places=4
        )

    def test_missing_metadata_fields_raise_loud_error(self) -> None:
        """Verify prepare_film_condition_tensor raises ValueError when required
        fields are missing or null.
        """
        base_meta = {
            "die": 0,
            "source_tool_index": 0,
            "target_tool_index": 1,
            "total_flux_single_die_standardized": 0.0,
            "time_standardized": 0.0,
        }
        for field in (
            "source_tool_index",
            "target_tool_index",
            "total_flux_single_die_standardized",
            "time_standardized",
        ):
            bad_meta = {k: v for k, v in base_meta.items() if k != field}
            with self.assertRaises(ValueError) as ctx:
                prepare_film_condition_tensor(metadata=[bad_meta])
            self.assertIn(field, str(ctx.exception))

        with self.assertRaises(ValueError):
            prepare_film_condition_tensor(metadata=None)

    def test_film_generator_feature_toggles_and_dimensions(self) -> None:
        """Verify FiLM input dimension adapts to feature encoding toggles."""
        # Flux, time, and energy on (with sliding window on):
        # die (4) + 2*tool (8) + flux (1) + time (1) + energy (2) = 16
        fg_all = FiLMGenerator(
            [16],
            config={"use_sliding_window": True, "encode_energy": True},
        )
        self.assertEqual(fg_all.mlp[0].in_features, 16)

        # Energy on with sliding window off: 4 + 8 + 1 + 1 + 2 = 16
        fg_energy_sw_off = FiLMGenerator(
            [16],
            config={"use_sliding_window": False, "encode_energy": True},
        )
        self.assertEqual(fg_energy_sw_off.mlp[0].in_features, 16)

        # Default (energy off): 4 + 8 + 1 + 1 = 14
        fg_default = FiLMGenerator([16])
        self.assertEqual(fg_default.mlp[0].in_features, 14)

        # Flux off, time on: 4 + 8 + 0 + 1 = 13
        fg_no_flux = FiLMGenerator([16], config={"encode_flux": False})
        self.assertEqual(fg_no_flux.mlp[0].in_features, 13)
        cond_no_flux = prepare_film_condition_tensor(
            metadata=[
                {
                    "die": 0,
                    "source_tool_index": 0,
                    "target_tool_index": 1,
                    "time_standardized": 0.5,
                }
            ],
            config={"encode_flux": False},
        )
        self.assertNotIn("total_flux_single_die_standardized", cond_no_flux)
        self.assertEqual(len(fg_no_flux(cond_no_flux)), 1)

        # Both flux and time off: 4 + 8 = 12
        fg_neither = FiLMGenerator(
            [16], config={"encode_flux": False, "encode_time": False}
        )
        self.assertEqual(fg_neither.mlp[0].in_features, 12)
        cond_neither = prepare_film_condition_tensor(
            metadata=[{"die": 0, "source_tool_index": 0, "target_tool_index": 1}],
            config={"encode_flux": False, "encode_time": False},
        )
        self.assertEqual(len(fg_neither(cond_neither)), 1)

        # Die encoding off: 0 + 8 + 1 + 1 = 10
        fg_no_die = FiLMGenerator([16], config={"encode_die": False})
        self.assertEqual(fg_no_die.mlp[0].in_features, 10)
        self.assertIsNone(fg_no_die.die_embedding)
        cond_no_die = prepare_film_condition_tensor(
            metadata=[
                {
                    "source_tool_index": 0,
                    "target_tool_index": 1,
                    "total_flux_single_die_standardized": 0.5,
                    "time_standardized": -0.2,
                }
            ],
            config={"encode_die": False},
        )
        self.assertNotIn("die", cond_no_die)
        mods_no_die = fg_no_die(cond_no_die)
        self.assertEqual(len(mods_no_die), 1)

        # Tool encoding off: 4 + 0 + 1 + 1 = 6
        fg_no_tool = FiLMGenerator([16], config={"encode_tool": False})
        self.assertEqual(fg_no_tool.mlp[0].in_features, 6)
        self.assertIsNone(fg_no_tool.tool_embedding)
        cond_no_tool = prepare_film_condition_tensor(
            metadata=[
                {
                    "die": 3,
                    "total_flux_single_die_standardized": 0.5,
                    "time_standardized": -0.2,
                }
            ],
            config={"encode_tool": False},
        )
        self.assertNotIn("source_tool_index", cond_no_tool)
        self.assertNotIn("target_tool_index", cond_no_tool)
        mods_no_tool = fg_no_tool(cond_no_tool)
        self.assertEqual(len(mods_no_tool), 1)

    def test_film_generator_raises_when_all_encodings_disabled(self) -> None:
        """Verify FiLMGenerator raises ValueError when all encodings are False."""
        all_off_cfg = {
            "encode_die": False,
            "encode_tool": False,
            "encode_flux": False,
            "encode_time": False,
            "encode_region": False,
            "encode_relative_intensity": False,
            "encode_elements": False,
            "encode_energy": False,
        }
        with self.assertRaises(ValueError) as ctx:
            FiLMGenerator([16], config=all_off_cfg)
        self.assertIn("all feature encodings are disabled", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            ResNet1D(
                20,
                config={
                    **_DEFAULT_RESNET_TEST_CONFIG,
                    "use_film": True,
                    **all_off_cfg,
                },
            )
        self.assertIn("all feature encodings are disabled", str(ctx2.exception))

    def test_regional_energy_encoding_sliding_window_off(self) -> None:
        """Verify energy encoding works when sliding window is turned off."""
        n_pts = 25
        x = np.ones((2, n_pts), dtype=np.float32)
        y = np.ones((2, n_pts), dtype=np.float32)
        energy = np.linspace(150.0, 175.0, n_pts, dtype=np.float32)
        pair_ds = SpectrumPairDataset(
            x,
            y,
            energy=np.tile(energy, (2, 1)),
            metadata=[
                {
                    "die": 1,
                    "source_tool_index": 0,
                    "target_tool_index": 1,
                    "total_flux_single_die_standardized": 0.0,
                    "time_standardized": 0.0,
                }
            ]
            * 2,
        )
        item = pair_ds[0]
        self.assertIn("energy_bounds", item)
        self.assertAlmostEqual(item["energy_bounds"][0].item(), 150.0, places=3)
        self.assertAlmostEqual(item["energy_bounds"][1].item(), 175.0, places=3)

        model = ResNet1D(
            n_pts,
            config={
                **_DEFAULT_RESNET_TEST_CONFIG,
                "resnet_channel": 8,
                "use_film": True,
                "use_sliding_window": False,
                "encode_energy": True,
                "energy_scale_ev": 1000.0,
            },
        )
        self.assertTrue(model.film_gen.encode_energy)

        # Test forward pass with energy bounds
        meta = [
            {
                "die": 1,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": 0.0,
                "time_standardized": 0.0,
            }
        ] * 2
        cond = prepare_film_condition_tensor(
            metadata=meta,
            energy_bounds=torch.tensor([[150.0, 175.0], [150.0, 175.0]]),
            config={"num_samples": 2, "energy_scale_ev": 1000.0},
        )
        self.assertIn("energy_bounds", cond)
        self.assertAlmostEqual(cond["energy_bounds"][0, 0].item(), 0.15, places=4)
        self.assertAlmostEqual(cond["energy_bounds"][0, 1].item(), 0.175, places=4)

        out = model(torch.as_tensor(x).unsqueeze(1), cond=cond)
        self.assertEqual(out.shape, (2, 1, n_pts))

        # Test predict_spectra in non-sliding-window mode
        # with encode_energy=True.
        from models.inference import predict_spectra

        meta_df = pd.DataFrame(
            [
                {
                    "spectrum_index": i,
                    "measurement_id": f"NMG_M_J4_{i:05d}",
                    "tool": "J4",
                    "source_tool_index": 0,
                    "target_tool": "H1",
                    "target_tool_index": 1,
                    "die": i,
                    "region": "Si2p",
                    "split": "test",
                    "total_flux_single_die_standardized": 0.1 * i,
                    "time_standardized": 0.0,
                }
                for i in range(2)
            ]
        )
        pred_i, pred_e, pred_df = predict_spectra(
            models={"Si2p": model},
            data=(x, np.tile(energy, (2, 1)), meta_df),
            config={"source_tool": "J4", "target_tool": "H1", "use_sliding_window": False},
        )
        self.assertEqual(pred_i.shape, (2, n_pts))
        self.assertEqual(pred_e.shape, (2, n_pts))

    def test_prepare_film_condition_tensor_with_energy_bounds(self) -> None:
        """Verify window energy bounds are scaled by energy_scale_ev."""
        meta = [
            {
                "die": 0,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": 0.0,
                "time_standardized": 0.0,
            }
        ]
        we = np.array([[100.0, 106.0]])

        cond_default = prepare_film_condition_tensor(
            metadata=meta,
            energy_bounds=we,
        )
        self.assertIn("energy_bounds", cond_default)
        self.assertAlmostEqual(
            cond_default["energy_bounds"][0, 0].item(), 0.1, places=4
        )
        self.assertAlmostEqual(
            cond_default["energy_bounds"][0, 1].item(), 0.106, places=4
        )

        cond_custom = prepare_film_condition_tensor(
            metadata=meta,
            energy_bounds=we,
            config={"energy_scale_ev": 500.0},
        )
        self.assertAlmostEqual(
            cond_custom["energy_bounds"][0, 0].item(), 0.2, places=4
        )
        self.assertAlmostEqual(
            cond_custom["energy_bounds"][0, 1].item(), 0.212, places=4
        )

    def test_spectrum_patch_dataset_yields_energy_bounds(self) -> None:
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
                    "source_tool_index": 0,
                    "target_tool_index": 1,
                    "total_flux_single_die_standardized": 0.0,
                    "time_standardized": 0.0,
                }
            ]
            * 2,
        )
        patch_ds = SpectrumPatchDataset(full_ds, window_size=10, stride=5)
        self.assertGreater(len(patch_ds), 0)
        item = patch_ds[0]
        self.assertIn("energy_bounds", item)
        self.assertEqual(item["energy_bounds"].shape, (2,))
        self.assertAlmostEqual(
            item["energy_bounds"][0].item(), energy[0], places=3
        )
        self.assertAlmostEqual(
            item["energy_bounds"][1].item(), energy[9], places=3
        )

    def test_resnet_with_energy_bounds_film_forward_backward(self) -> None:
        """Verify ResNet1D forward and backward pass with window energy FiLM."""
        patch_len = 10
        model = ResNet1D(
            patch_len,
            config={
                **_DEFAULT_RESNET_TEST_CONFIG,
                "resnet_channel": 16,
                "use_film": True,
                "use_sliding_window": True,
                "encode_energy": True,
                "energy_scale_ev": 1000.0,
            },
        )
        x = torch.randn(2, 1, patch_len)
        meta = [
            {
                "die": 0,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": 0.1,
                "time_standardized": -0.1,
            }
        ] * 2
        we = torch.tensor([[70.0, 80.0], [75.0, 85.0]], dtype=torch.float32)
        cond = prepare_film_condition_tensor(
            metadata=meta,
            energy_bounds=we,
            config={"num_samples": 2, "energy_scale_ev": 1000.0},
        )
        out = model(x, cond=cond)
        self.assertEqual(out.shape, (2, 1, patch_len))
        loss = torch.sum(out**2)
        loss.backward()
        self.assertIsNotNone(model.film_gen.mlp[0].weight.grad)

    def test_predict_spectra_and_sliding_window_with_film(self) -> None:
        """Verify predict_spectra and evaluate_sliding_window with FiLM
        active.
        """
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
                "source_tool_index": 0,
                "target_tool": "H1",
                "target_tool_index": 1,
                "die": i % 2,
                "region": "Al2p",
                "split": "test",
                "total_flux_single_die_standardized": 0.1 * i,
                "time_standardized": -0.2 * i,
            }
            for i in range(4)
        ]
        val_ds = SpectrumPairDataset(x, y, energy=energy, metadata=meta_records)
        sw_model = ResNet1D(
            10,
            config={
                **_DEFAULT_RESNET_TEST_CONFIG,
                "resnet_channel": 8,
                "kernel_size": 3,
                "use_film": True,
                "use_sliding_window": True,
                "encode_energy": True,
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
        self.assertIn("total_flux_single_die_standardized", pred_df.columns)
        self.assertIn("time_standardized", pred_df.columns)

    def test_film_region_encoding(self) -> None:
        """Verify region index extraction and FiLM conditioning with regions."""
        meta = [
            {
                "die": 0,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "region_index": 2,
                "total_flux_single_die_standardized": 0.0,
                "time_standardized": 0.0,
            },
            {
                "die": 1,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "region_index": 11,
                "total_flux_single_die_standardized": 0.0,
                "time_standardized": 0.0,
            },
        ]
        cond = prepare_film_condition_tensor(
            meta, config={"encode_region": True, "num_samples": 2}
        )
        self.assertIn("region", cond)
        self.assertEqual(cond["region"].tolist(), [2, 11])

        # Test FiLMGenerator with region embedding
        film_gen = FiLMGenerator(
            channels_list=[16],
            config={
                "encode_region": True,
                "num_regions": 16,
                "region_embed_dim": 4,
            },
        )
        modulations = film_gen(cond)
        self.assertEqual(len(modulations), 1)
        gamma, beta = modulations[0]
        self.assertEqual(gamma.shape, (2, 16))
        self.assertEqual(beta.shape, (2, 16))
        # Initial gamma is 1.0, initial beta is 0.0
        torch.testing.assert_close(gamma, torch.ones_like(gamma))
        torch.testing.assert_close(beta, torch.zeros_like(beta))

    def test_film_relative_intensity_encoding(self) -> None:
        """Verify relative regional intensity condition extraction and FiLM."""
        meta = [
            {
                "die": 0,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": 0.0,
                "time_standardized": 0.0,
                "relative_intensity_single_region": 0.85,
            },
            {
                "die": 1,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": 0.0,
                "time_standardized": 0.0,
                "relative_intensity_single_region": 1.30,
            },
        ]
        cond = prepare_film_condition_tensor(
            meta, config={"encode_relative_intensity": True, "num_samples": 2}
        )
        self.assertIn("relative_intensity_single_region", cond)
        self.assertEqual(cond["relative_intensity_single_region"].shape, (2, 1))
        self.assertAlmostEqual(
            cond["relative_intensity_single_region"][0, 0].item(), 0.85, places=4
        )
        self.assertAlmostEqual(
            cond["relative_intensity_single_region"][1, 0].item(), 1.30, places=4
        )

        film_gen = FiLMGenerator(
            channels_list=[16],
            config={"encode_relative_intensity": True},
        )
        modulations = film_gen(cond)
        self.assertEqual(len(modulations), 1)
        gamma, beta = modulations[0]
        self.assertEqual(gamma.shape, (2, 16))
        self.assertEqual(beta.shape, (2, 16))
        torch.testing.assert_close(gamma, torch.ones_like(gamma))
        torch.testing.assert_close(beta, torch.zeros_like(beta))

    def test_film_element_encoding_dynamic_vocabulary(self) -> None:
        """Verify dynamic element vocabulary multi-hot vector encoding."""
        materials_cfg = {
            "NMG": {"elements": ["Ti", "Al", "C", "O", "Cl", "Si"]},
            "PMG": {"elements": ["Ti", "N", "O", "Cl", "Si"]},
        }
        # Vocab: ["Al", "C", "Cl", "N", "O", "Si", "Ti"] -> 7 elements
        meta = [
            {
                "die": 0,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": 0.0,
                "time_standardized": 0.0,
                "material": "NMG",
            },
            {
                "die": 1,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": 0.0,
                "time_standardized": 0.0,
                "material": "PMG",
            },
        ]
        cond = prepare_film_condition_tensor(
            meta,
            config={
                "encode_elements": True,
                "material_elements": materials_cfg,
                "num_samples": 2,
            },
        )
        self.assertIn("element_multi_hot", cond)
        self.assertEqual(cond["element_multi_hot"].shape, (2, 7))

        # NMG has Al, C, Cl, O, Si, Ti -> indexes 0, 1, 2, 4, 5, 6 (N at 3 is 0)
        expected_nmg = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        # PMG has Cl, N, O, Si, Ti -> idx 2, 3, 4, 5, 6 (Al, C at 0, 1 are 0)
        expected_pmg = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        np.testing.assert_allclose(
            cond["element_multi_hot"][0].numpy(), expected_nmg, atol=1e-5
        )
        np.testing.assert_allclose(
            cond["element_multi_hot"][1].numpy(), expected_pmg, atol=1e-5
        )

        film_gen = FiLMGenerator(
            channels_list=[16],
            config={"encode_elements": True, "element_vocab_size": 7},
        )
        modulations = film_gen(cond)
        self.assertEqual(len(modulations), 1)
        gamma, beta = modulations[0]
        self.assertEqual(gamma.shape, (2, 16))
        self.assertEqual(beta.shape, (2, 16))
        torch.testing.assert_close(gamma, torch.ones_like(gamma))
        torch.testing.assert_close(beta, torch.zeros_like(beta))

    def test_element_encoding_requires_materials_without_fallback(self) -> None:
        """Verify element encoding raises ValueError when materials config is
        missing.
        """
        meta = [
            {
                "die": 0,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": 0.0,
                "time_standardized": 0.0,
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            prepare_film_condition_tensor(
                meta, config={"encode_elements": True}
            )
        self.assertIn("material_elements", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            FiLMGenerator([16], config={"encode_elements": True})
        self.assertIn("material_elements", str(ctx2.exception))

    def test_assemble_prediction_metadata_propagates_metrics(self) -> None:
        """Verify assemble_prediction_metadata propagates all Stage 1
        metrics.
        """
        from models.inference import assemble_prediction_metadata

        source_df = pd.DataFrame(
            [
                {
                    "spectrum_index": 0,
                    "measurement_id": "NMG_M_J4_00000",
                    "die": 0,
                    "region": "Al2p",
                    "total_flux_single_die_standardized": 1.25,
                    "mean_intensity_single_region": 150.0,
                    "mean_intensity_single_die": 120.0,
                    "relative_intensity_single_region": 1.25,
                    "total_flux_single_die": 5000.0,
                    "time_standardized": 0.5,
                }
            ]
        )
        pred_meta = assemble_prediction_metadata(
            source_df,
            {
                "source_tool": "J4",
                "target_tool": "H1",
                "n_points": 100,
            },
        )
        self.assertIn("total_flux_single_die_standardized", pred_meta.columns)
        self.assertIn("mean_intensity_single_region", pred_meta.columns)
        self.assertIn("mean_intensity_single_die", pred_meta.columns)
        self.assertIn("relative_intensity_single_region", pred_meta.columns)
        self.assertIn("total_flux_single_die", pred_meta.columns)
        self.assertEqual(
            pred_meta["total_flux_single_die_standardized"].iloc[0], 1.25
        )
        self.assertEqual(
            pred_meta["mean_intensity_single_region"].iloc[0], 150.0
        )


    def test_film_generator_condition_packing_helpers(self) -> None:
        """Verify get_condition_config and prepare_condition_tensor."""
        fg = FiLMGenerator(
            [16, 32],
            config={
                "encode_die": True,
                "encode_tool": True,
                "encode_flux": True,
                "encode_time": True,
                "encode_energy": True,
            },
        )
        cfg = fg.get_condition_config(num_samples=4, energy_scale_ev=500.0)
        self.assertEqual(cfg["num_samples"], 4)
        self.assertEqual(cfg["energy_scale_ev"], 500.0)
        self.assertTrue(cfg["encode_die"])
        self.assertTrue(cfg["encode_energy"])

        meta = [
            {
                "die": 0,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": 0.5,
                "time_standardized": -0.2,
            }
        ]
        cond = fg.prepare_condition_tensor(
            meta, energy_bounds=np.array([[100.0, 150.0]])
        )
        self.assertIn("die", cond)
        self.assertIn("total_flux_single_die_standardized", cond)
        self.assertIn("energy_bounds", cond)
        self.assertEqual(cond["energy_bounds"].shape, (1, 2))

if __name__ == "__main__":
    unittest.main()
