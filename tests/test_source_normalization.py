"""Unit tests for source-referenced normalization and final zero clamping."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from models.dataset import SpectrumPairDataset, split_session_datasets
from models.inference import predict_sliding_window_spectrum, predict_spectra
from models.resnet import ResNet1D
from models.trainer import train_model_region
from models.unet import UNet1D
from utility.pairing import pair_source_target_spectra
from utility.pseudo_measurement import generate_pseudo_measurements


class TestSourceReferencedNormalization(unittest.TestCase):
    """Test suite verifying source-referenced normalization and zero clamping."""

    def test_spectrum_pair_dataset_source_norm_ratio_preservation(self) -> None:
        """Test that source normalization preserves inter-tool scale ratio (e.g. 1.1x)."""
        x_raw = np.array([[1000.0, 5000.0, 2000.0], [500.0, 2000.0, 1000.0]], dtype=np.float32)
        y_raw = x_raw * 1.15  # Target tool is 15% higher

        ds = SpectrumPairDataset(x_raw, y_raw, config={"normalize_by_source": True})

        # Sample 0: max(x) is 5000
        item0 = ds[0]
        self.assertAlmostEqual(item0["x"].max().item(), 1.0, places=5)
        self.assertAlmostEqual(item0["y"].max().item(), 1.15, places=5)
        self.assertAlmostEqual(float(item0["scale_x"].item()), 5000.0, places=2)

        # Unnormalize helper
        unnorm_y = ds.unnormalize_y(item0["y"], idx=0)
        np.testing.assert_allclose(unnorm_y.numpy(), y_raw[0], rtol=1e-5)

    def test_spectrum_pair_dataset_disabled_normalization(self) -> None:
        """Test that normalize_by_source=False leaves x and y unnormalized."""
        x_raw = np.array([[1000.0, 5000.0, 2000.0]], dtype=np.float32)
        y_raw = x_raw * 1.1

        ds = SpectrumPairDataset(x_raw, y_raw, config={"normalize_by_source": False})
        item = ds[0]
        self.assertAlmostEqual(item["x"].max().item(), 5000.0, places=2)
        self.assertAlmostEqual(item["y"].max().item(), 5500.0, places=2)

    def test_predict_spectra_final_zero_clamping_and_physical_scaling(self) -> None:
        """Test predict_spectra recovers physical target scale and clamps negative values."""
        n_points = 50
        x_intensities = np.full((4, n_points), 10000.0, dtype=np.float32)
        energies = np.linspace(100.0, 110.0, n_points)
        meta_records = [
            {"spectrum_index": i, "measurement_id": f"M_{i}", "tool": "J4", "region": "Ti2p", "die": 0}
            for i in range(4)
        ]
        import pandas as pd
        meta_df = pd.DataFrame(meta_records)

        # Model that outputs -0.2 on first half of points, +1.1 on second half
        class DummyModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = torch.ones_like(x) * 1.1
                out[:, : n_points // 2] = -0.2
                return out

        models = {"Ti2p": DummyModel()}
        pred_i, pred_e, pred_df = predict_spectra(
            models=models,
            data=(x_intensities, np.tile(energies, (4, 1)), meta_df),
            config={"source_tool": "J4", "target_tool": "J5", "clamp_non_negative": True},
        )

        # Non-negativity clamp check
        self.assertTrue(np.all(pred_i >= 0.0))
        # Negative region clamped to 0.0
        np.testing.assert_allclose(pred_i[:, : n_points // 2], 0.0)
        # Positive region restored to physical target scale: 1.1 * 10000 = 11000
        np.testing.assert_allclose(pred_i[:, n_points // 2 :], 11000.0, rtol=1e-4)

    def test_predict_sliding_window_spectrum_final_clamping(self) -> None:
        """Test sliding window averages linearly and clamps at the full spectrum reconstruction."""
        n_points = 60
        spectrum = np.full(n_points, 10000.0, dtype=np.float32)

        # Model returning -0.1 for first 10 points of patch, 1.2 elsewhere
        class DummyPatchModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = torch.ones_like(x) * 1.2
                out[:, :5] = -0.1
                return out

        model = DummyPatchModel()
        reconstructed = predict_sliding_window_spectrum(
            model=model,
            spectrum=spectrum,
            config={
                "window_size": 20,
                "stride": 10,
                "normalize_by_source": True,
                "clamp_non_negative": True,
            },
        )

        # Must be non-negative
        self.assertTrue(np.all(reconstructed >= 0.0))
        # Points away from boundaries should scale to ~1.2 * 10000 = 12000
        self.assertGreater(reconstructed.max(), 10000.0)

    def test_predict_unet_spectra_clamping_and_scaling(self) -> None:
        """Test predict_unet_spectra handles physical scale and zero clamping in full-spectrum mode."""
        n_points = 40
        x_intensities = np.full((2, n_points), 20000.0, dtype=np.float32)
        energies = np.linspace(100.0, 110.0, n_points)
        import pandas as pd
        meta_df = pd.DataFrame([
            {"spectrum_index": 0, "measurement_id": "M_0", "tool": "J4", "region": "Al2p", "die": 0},
            {"spectrum_index": 1, "measurement_id": "M_1", "tool": "J4", "region": "Al2p", "die": 1},
        ])

        class DummyUNet(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = torch.ones_like(x) * 1.15
                out[:, :10] = -0.5
                return out

        models = {"Al2p": DummyUNet()}
        pred_i, _, _ = predict_spectra(
            models=models,
            data=(x_intensities, np.tile(energies, (2, 1)), meta_df),
            config={"source_tool": "J4", "target_tool": "J5", "use_sliding_window": False},
        )

        self.assertTrue(np.all(pred_i >= 0.0))
        np.testing.assert_allclose(pred_i[:, :10], 0.0)
        np.testing.assert_allclose(pred_i[:, 10:], 1.15 * 20000.0, rtol=1e-4)

    def test_training_loss_decreases_with_scale_10000(self) -> None:
        """Verify training on scale 10000 pseudo-spectrum reduces validation loss significantly."""
        scale = 10000.0
        config = {
            "material": "NMG",
            "regions": ["Ti2p"],
            "n_points": 50,
            "measurements_per_tool": {"J4": 6, "J5": 6},
            "tool_offsets": {
                "J4": {"shift_ev": 0.0, "scale": scale},
                "J5": {"shift_ev": 0.2, "scale": scale * 1.1},
            },
            "seed": 42,
        }
        ary_intensity, ary_energy, meta_df = generate_pseudo_measurements(config)
        meta_df = pair_source_target_spectra(meta_df, {"source_tool": "J4", "target_tool": "J5"})
        train_ds, val_ds = split_session_datasets(
            meta_df,
            ary_intensity,
            ary_energy,
            {"region": "Ti2p", "source_tool": "J4", "target_tool": "J5", "normalize_by_source": True},
        )

        from models.cost import NormalizedMSELoss
        untrained_val_loss = NormalizedMSELoss()(val_ds.x, val_ds.y).item()

        torch.manual_seed(42)
        res = train_model_region(
            train_ds,
            val_ds,
            {"model_type": "resnet", "epochs": 10, "verbose": False, "learning_rate": 1e-3, "l2_weight": 1e-4},
        )

        best_val_loss = res["best_val_loss"]
        train_loss_start = res["history"]["train_loss"][0]
        train_loss_end = res["history"]["train_loss"][-1]

        # Validation loss must improve upon the untrained identity baseline
        self.assertLess(best_val_loss, untrained_val_loss)
        # Training loss must decrease noticeably
        self.assertLess(train_loss_end, train_loss_start)

    def test_all_zero_and_negative_noise_floor_edge_cases(self) -> None:
        """Verify handling of all-zero spectra and negative noise floors without NaNs or zero-division."""
        # All-zero source and target
        x_zero = np.zeros((2, 30), dtype=np.float32)
        y_zero = np.zeros((2, 30), dtype=np.float32)
        ds_zero = SpectrumPairDataset(x_zero, y_zero)
        item0 = ds_zero[0]
        self.assertFalse(torch.isnan(item0["x"]).any())
        self.assertFalse(torch.isnan(item0["y"]).any())
        self.assertAlmostEqual(float(item0["scale_x"].item()), 1e-4, places=6)

        # Negative noise floor
        x_neg = np.array([[-100.0, -50.0, -10.0]], dtype=np.float32)
        y_neg = np.array([[-115.0, -57.5, -11.5]], dtype=np.float32)
        ds_neg = SpectrumPairDataset(x_neg, y_neg)
        item_neg = ds_neg[0]
        self.assertAlmostEqual(float(item_neg["scale_x"].item()), 100.0, places=2)
        np.testing.assert_allclose(item_neg["x"].numpy(), [-1.0, -0.5, -0.1], rtol=1e-5)
        np.testing.assert_allclose(item_neg["y"].numpy(), [-1.15, -0.575, -0.115], rtol=1e-5)

    def test_unbiased_sliding_window_overlap_accumulation(self) -> None:
        """Verify that patches are averaged linearly without premature clamping in overlap areas."""
        from utility.patching import reconstruct_from_patches

        # Two overlapping patches: patch 1 predicts -2.0, patch 2 predicts +4.0 in overlap
        patches = np.array([[-2.0] * 10, [4.0] * 10], dtype=np.float32)
        start_indices = [0, 5]
        # Reconstruct without premature clamping
        rec = reconstruct_from_patches(
            patches,
            start_indices,
            original_length=15,
            config={"window_size": 10, "clamp_non_negative": False},
        )
        # Expected overlap average is (-2.0 + 4.0) / 2 = 1.0 (unbiased)
        # If premature clamping had occurred, it would be (0.0 + 4.0) / 2 = 2.0 (biased)
        np.testing.assert_allclose(rec[5:10], 1.0, rtol=1e-5)

    def test_clamp_non_negative_toggle(self) -> None:
        """Verify clamp_non_negative=False preserves negative values in predictions."""
        import pandas as pd
        x_neg = np.array([[-50.0, -20.0, 100.0]], dtype=np.float32)
        energies = np.array([[100.0, 101.0, 102.0]], dtype=np.float32)
        meta_df = pd.DataFrame([
            {"spectrum_index": 0, "measurement_id": "M_0", "tool": "J4", "region": "Al2p", "die": 0}
        ])

        class PassThrough(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        models = {"Al2p": PassThrough()}
        # Unclamped mode
        p_unclamped, _, _ = predict_spectra(
            models=models,
            data=(x_neg, energies, meta_df),
            config={"source_tool": "J4", "target_tool": "H1", "clamp_non_negative": False},
        )
        self.assertLess(p_unclamped[0, 0], 0.0)
        np.testing.assert_allclose(p_unclamped, x_neg, rtol=1e-5)

        # Clamped mode
        p_clamped, _, _ = predict_spectra(
            models=models,
            data=(x_neg, energies, meta_df),
            config={"source_tool": "J4", "target_tool": "H1", "clamp_non_negative": True},
        )
        self.assertGreaterEqual(p_clamped[0, 0], 0.0)
        self.assertEqual(p_clamped[0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()

