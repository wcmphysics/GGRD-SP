"""Unit tests for the sliding window neural network model and pipeline."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from models.baseline_cnn import NormalizedMSELoss, Residual1DCNN
from models.dataset import split_session_datasets
from models.sliding_window import (
    SpectrumPatchDataset,
    calculate_window_points,
    evaluate_sliding_window,
    extract_sliding_windows,
    optimize_sliding_window_hyperparameters,
    predict_sliding_window_spectrum,
    reconstruct_from_patches,
    run_sliding_window_pipeline,
    train_sliding_window_region,
)
from utility.pairing import pair_source_target_spectra
from utility.pseudo_measurement import generate_pseudo_measurements


class TestSlidingWindowOperations(unittest.TestCase):
    """Test cases for window calculation, extraction, and reconstruction."""

    def test_calculate_window_points(self) -> None:
        """Test calculation of window points from binding energy grid."""
        # 100 points spanning 10 eV -> delta_e ~ 0.101 eV/pt
        energy = np.linspace(50.0, 60.0, 100)
        w, s = calculate_window_points(
            energy,
            config={"window_size_ev": 2.0, "sliding_stride_ev": 1.0, "force_odd_window": True},
        )
        self.assertGreater(w, 0)
        self.assertGreater(s, 0)
        self.assertEqual(w % 2, 1)  # odd window
        self.assertLessEqual(s, w)

        # Explicit points override via window_size_points and window_size
        w_exp, s_exp = calculate_window_points(
            energy, config={"window_size_points": 21, "sliding_stride_points": 5}
        )
        self.assertEqual(w_exp, 21)
        self.assertEqual(s_exp, 5)

        w_exp2, s_exp2 = calculate_window_points(
            energy, config={"window_size": 17, "stride": 7}
        )
        self.assertEqual(w_exp2, 17)
        self.assertEqual(s_exp2, 7)

    def test_calculate_window_points_even_length_odd_enforcement(self) -> None:
        """Verify odd window clamping when total points is an even number."""
        energy = np.linspace(50.0, 60.0, 60)  # even total points: 60
        # Request a window size larger than total points (e.g. 70 points or 20 eV)
        w, s = calculate_window_points(
            energy, config={"window_size_points": 70, "force_odd_window": True}
        )
        self.assertEqual(w % 2, 1)
        self.assertLessEqual(w, 60)
        self.assertEqual(w, 59)

    def test_extract_sliding_windows_invalid_stride(self) -> None:
        """Verify that extract_sliding_windows raises ValueError on invalid stride."""
        arr = np.ones(50)
        with self.assertRaises(ValueError):
            extract_sliding_windows(arr, window_size=10, stride=0)
        with self.assertRaises(ValueError):
            extract_sliding_windows(arr, window_size=10, stride=15)

    def test_extract_and_reconstruct_identity(self) -> None:
        """Verify that extracting and immediately reconstructing preserves the original signal."""
        np.random.seed(42)
        n_points = 100
        spectrum = np.sin(np.linspace(0, 3 * np.pi, n_points)) + 5.0
        window_size = 15
        stride = 3

        windows, start_indices = extract_sliding_windows(
            spectrum, window_size=window_size, stride=stride
        )
        self.assertEqual(windows.shape[1], window_size)

        # Reconstruct directly using the extracted windows
        reconstructed = reconstruct_from_patches(
            patches=windows,
            start_indices=start_indices,
            original_length=n_points,
            window_size=window_size,
        )

        self.assertEqual(len(reconstructed), n_points)
        np.testing.assert_allclose(reconstructed, spectrum, rtol=1e-5, atol=1e-5)

    def test_extract_and_reconstruct_unaligned_stride(self) -> None:
        """Verify full coverage and reconstruction when (N - W) is not a multiple of stride."""
        np.random.seed(42)
        n_points = 103  # 103 - 15 = 88, 88 % 7 = 4 != 0
        spectrum = np.cos(np.linspace(0, 4 * np.pi, n_points)) + 10.0
        window_size = 15
        stride = 7

        windows, start_indices = extract_sliding_windows(
            spectrum, window_size=window_size, stride=stride
        )
        # Check right-edge anchoring
        self.assertEqual(start_indices[-1], n_points - window_size)

        reconstructed = reconstruct_from_patches(
            patches=windows,
            start_indices=start_indices,
            original_length=n_points,
            window_size=window_size,
        )
        self.assertEqual(len(reconstructed), n_points)
        np.testing.assert_allclose(reconstructed, spectrum, rtol=1e-5, atol=1e-5)

    def test_extract_and_reconstruct_window_equals_spectrum_length(self) -> None:
        """Verify extraction and reconstruction when window_size equals spectrum length."""
        spectrum = np.linspace(10.0, 20.0, 30)
        windows, start_indices = extract_sliding_windows(
            spectrum, window_size=30, stride=10
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(start_indices, [0])
        recon = reconstruct_from_patches(windows, start_indices, original_length=30)
        np.testing.assert_allclose(recon, spectrum)

    def test_extract_and_reconstruct_stride_one(self) -> None:
        """Verify extraction and reconstruction with maximum overlap (stride = 1)."""
        spectrum = np.sin(np.linspace(0, 2, 20))
        windows, start_indices = extract_sliding_windows(spectrum, window_size=7, stride=1)
        recon = reconstruct_from_patches(windows, start_indices, original_length=20)
        np.testing.assert_allclose(recon, spectrum, rtol=1e-5, atol=1e-5)

    def test_extract_and_reconstruct_stride_equals_window(self) -> None:
        """Verify extraction and reconstruction when stride equals window_size (tiling)."""
        spectrum = np.arange(25, dtype=np.float32)
        windows, start_indices = extract_sliding_windows(spectrum, window_size=5, stride=5)
        self.assertEqual(len(windows), 5)
        recon = reconstruct_from_patches(windows, start_indices, original_length=25)
        np.testing.assert_allclose(recon, spectrum)

    def test_extract_sliding_windows_2d_input_sanitization(self) -> None:
        """Verify that 2D column input (N, 1) is automatically flattened to 1D."""
        spectrum_2d = np.ones((40, 1), dtype=np.float32)
        windows, start_indices = extract_sliding_windows(spectrum_2d, window_size=10, stride=5)
        self.assertEqual(windows.shape, (7, 10))

    def test_extract_sliding_windows_window_larger_than_spectrum(self) -> None:
        """Verify ValueError when window_size exceeds spectrum length."""
        arr = np.ones(20)
        with self.assertRaises(ValueError):
            extract_sliding_windows(arr, window_size=25, stride=5)
        with self.assertRaises(ValueError):
            extract_sliding_windows(arr, window_size=0, stride=5)


class TestSlidingWindowPipeline(unittest.TestCase):
    """Test cases for dataset, training, evaluation, Ax tuning, and root pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate synthetic paired dataset for testing."""
        cls.ary_intensity, cls.ary_energy, cls.meta_df = generate_pseudo_measurements(
            {
                "material": "NMG",
                "regions": ["Al2p", "Ti2p"],
                "n_points": 60,
                "measurements_per_tool": {"J4": 6, "H1": 6},
                "interval_hours_range": (1.0, 4.0),
                "seed": 42,
            }
        )
        cls.meta_df = pair_source_target_spectra(
            cls.meta_df,
            {
                "source_tool": "J4",
                "target_tool": "H1",
                "time_threshold_hours": 12.0,
            },
        )
        cls.train_ds, cls.val_ds = split_session_datasets(
            cls.meta_df,
            cls.ary_intensity,
            cls.ary_energy,
            config={"region": "Al2p", "source_tool": "J4", "target_tool": "H1", "seed": 42},
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
        model = Residual1DCNN(n_points=15, config={"hidden_channels": 8, "kernel_size": 3})
        spectrum = self.train_ds[0]["x"].cpu().numpy()
        pred = predict_sliding_window_spectrum(model, spectrum, window_size=15, stride=5)
        self.assertEqual(len(pred), len(spectrum))

    def test_evaluate_sliding_window_reconstructed(self) -> None:
        """Test full-spectrum reconstructed evaluation loss calculation."""
        model = Residual1DCNN(n_points=15, config={"hidden_channels": 8, "kernel_size": 3})
        crit = NormalizedMSELoss(l2_weight=0.0)
        loss = evaluate_sliding_window(
            model=model,
            full_val_dataset=self.val_ds,
            window_size=15,
            stride=5,
            criterion=crit,
        )
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)

    def test_train_sliding_window_region(self) -> None:
        """Test regional sliding window training."""
        train_res = train_sliding_window_region(
            self.train_ds,
            self.val_ds,
            config={
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
        ax_res = optimize_sliding_window_hyperparameters(
            self.train_ds,
            self.val_ds,
            config={
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

    def test_run_sliding_window_pipeline(self) -> None:
        """Test full sliding window root pipeline execution."""
        pipeline_res = run_sliding_window_pipeline(
            self.meta_df,
            self.ary_intensity,
            self.ary_energy,
            config={
                "regions": ["Al2p"],
                "source_tool": "J4",
                "target_tool": "H1",
                "window_size_ev": 2.0,
                "sliding_stride_ev": 1.0,
                "use_bayesian_opt": False,
                "train_config": {"epochs": 2, "batch_size": 16, "hidden_channels": 8, "kernel_size": 3},
                "predict_source": True,
            },
        )
        self.assertIn("models", pipeline_res)
        self.assertIn("Al2p", pipeline_res["models"])
        self.assertIsNotNone(pipeline_res["predictions"])
        pred_int, pred_eng, pred_df = pipeline_res["predictions"]
        self.assertGreater(len(pred_df), 0)
        self.assertEqual(pred_int.shape[1], 60)

    def test_optimize_sliding_window_invalid_parameters(self) -> None:
        """Verify that Ax optimization raises ValueError on invalid parameters."""
        with self.assertRaises(ValueError):
            optimize_sliding_window_hyperparameters(
                self.train_ds, self.val_ds, config={"kernel_sizes": [4]}  # even
            )
        with self.assertRaises(ValueError):
            optimize_sliding_window_hyperparameters(
                self.train_ds, self.val_ds, config={"lr_bounds": (0.01, 0.001)}  # min > max
            )
        with self.assertRaises(ValueError):
            optimize_sliding_window_hyperparameters(
                self.train_ds, self.val_ds, config={"l2_bounds": (-1e-4, 1e-2)}  # min <= 0
            )

    def test_parameter_packaging_config_dict(self) -> None:
        """Verify predict and evaluate functions with config dictionaries adhering to Rule 7."""
        model = Residual1DCNN(n_points=15, config={"hidden_channels": 8, "kernel_size": 3})
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
