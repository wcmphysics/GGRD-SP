"""Unit tests for utility/patching.py module."""

from __future__ import annotations

import unittest
import numpy as np
import torch

from utility.patching import (
    calculate_window_points,
    extract_sliding_windows,
    reconstruct_from_patches,
)


class TestSpectralPatching(unittest.TestCase):
    """Test suite for spectral patching and geometric reconstruction."""

    def setUp(self) -> None:
        self.energy = np.linspace(100.0, 110.0, 101)  # delta_e = 0.1 eV
        self.spectrum = np.sin(np.linspace(0, 3.14, 101)).astype(np.float32)

    def test_calculate_window_points_from_ev(self) -> None:
        """Verify window size and stride conversion from eV."""
        # 2.0 eV / 0.1 eV = 20 points -> forced odd = 21 points
        # 1.0 eV / 0.1 eV = 10 points
        w_pts, s_pts = calculate_window_points(
            self.energy,
            config={"window_size_ev": 2.0, "sliding_stride_ev": 1.0, "force_odd_window": True},
        )
        self.assertEqual(w_pts, 21)
        self.assertEqual(s_pts, 10)

    def test_calculate_window_points_explicit_points(self) -> None:
        """Verify explicit point override in configuration."""
        w_pts, s_pts = calculate_window_points(
            self.energy,
            config={"window_size_points": 15, "sliding_stride_points": 5},
        )
        self.assertEqual(w_pts, 15)
        self.assertEqual(s_pts, 5)

    def test_calculate_window_points_invalid_inputs(self) -> None:
        """Verify errors on short or non-monotonic energy arrays."""
        with self.assertRaises(ValueError):
            calculate_window_points(np.array([10.0, 11.0]))  # len < 3

        with self.assertRaises(ValueError):
            calculate_window_points(np.array([10.0, 10.0, 10.0]))  # delta_e == 0

    def test_extract_sliding_windows_right_anchoring(self) -> None:
        """Verify 100% spectral coverage and right-edge anchoring."""
        w_size = 21
        stride = 10
        windows, start_indices = extract_sliding_windows(self.spectrum, w_size, stride)

        # Total points = 101. Starts: 0, 10, 20, 30, 40, 50, 60, 70, 80 (covers up to 101)
        self.assertEqual(start_indices[-1], len(self.spectrum) - w_size)
        self.assertEqual(windows.shape[1], w_size)
        self.assertEqual(len(windows), len(start_indices))

    def test_extract_sliding_windows_single_window(self) -> None:
        """When window_size == spectrum length, exactly 1 window is produced."""
        windows, start_indices = extract_sliding_windows(self.spectrum, len(self.spectrum), 1)
        self.assertEqual(len(windows), 1)
        self.assertEqual(start_indices, [0])

    def test_extract_sliding_windows_validation_errors(self) -> None:
        """Verify ValueError on invalid window_size or stride."""
        with self.assertRaises(ValueError):
            extract_sliding_windows(self.spectrum, window_size=0, stride=5)
        with self.assertRaises(ValueError):
            extract_sliding_windows(self.spectrum, window_size=200, stride=5)
        with self.assertRaises(ValueError):
            extract_sliding_windows(self.spectrum, window_size=15, stride=0)
        with self.assertRaises(ValueError):
            extract_sliding_windows(self.spectrum, window_size=15, stride=20)  # stride > window_size

    def test_reconstruct_from_patches_exact(self) -> None:
        """Verify reconstruction from overlapping patches reproduces original values."""
        w_size = 21
        stride = 10
        windows, start_indices = extract_sliding_windows(self.spectrum, w_size, stride)

        reconstructed = reconstruct_from_patches(
            windows,
            start_indices,
            original_length=len(self.spectrum),
            config={"window_size": w_size},
        )

        np.testing.assert_allclose(reconstructed, self.spectrum, atol=1e-5)

    def test_reconstruct_from_patches_tensor(self) -> None:
        """Verify reconstruct_from_patches accepts PyTorch tensors."""
        w_size = 15
        stride = 7
        windows, start_indices = extract_sliding_windows(self.spectrum, w_size, stride)
        windows_tensor = torch.from_numpy(windows)

        reconstructed = reconstruct_from_patches(
            windows_tensor,
            start_indices,
            original_length=len(self.spectrum),
            config={"window_size": w_size},
        )

        np.testing.assert_allclose(reconstructed, self.spectrum, atol=1e-5)

    def test_reconstruct_from_patches_gap_warning(self) -> None:
        """Verify warning when start_indices leaves uncovered gap points."""
        patch = np.ones((1, 5), dtype=np.float32)
        start_indices = [0]  # Only covers 0..5, leaves 5..10 empty
        with self.assertWarns(UserWarning):
            reconstruct_from_patches(patch, start_indices, original_length=10, config={"window_size": 5})


if __name__ == "__main__":
    unittest.main()
