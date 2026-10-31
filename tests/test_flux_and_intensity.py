"""Unit tests for utility.flux_and_intensity module."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from utility.flux_and_intensity import (
    calculate_flux_and_intensity,
    calculate_standardized_time,
    compute_die_total_flux,
    integrate_spectral_flux,
    standardize_metadata_features,
    trapezoid_integral,
)


class TestSpectralIntegration(unittest.TestCase):
    """Test suite for numerical integration and directional spectral flux."""

    def test_trapezoid_integral_functions_and_batching(self) -> None:
        """Verify integration of constant, linear functions, and 2D batched shapes."""
        x_const = np.linspace(0.0, 10.0, 101)
        self.assertAlmostEqual(float(trapezoid_integral(np.full_like(x_const, 5.0), x_const)), 50.0, places=4)

        x_lin = np.linspace(0.0, 5.0, 501)
        self.assertAlmostEqual(float(trapezoid_integral(2.0 * x_lin, x_lin)), 25.0, places=4)

        # Batched integration along last axis
        x_batched = np.linspace(0.0, 2.0, 21)
        y_batched = np.vstack([np.ones_like(x_batched), 2.0 * np.ones_like(x_batched)])
        res = trapezoid_integral(y_batched, x_batched, axis=-1)
        np.testing.assert_allclose(res, [2.0, 4.0], rtol=1e-5)

    def test_spectral_flux_direction_invariance_and_batching(self) -> None:
        """Verify ascending and descending grids yield identical flux and handle 2D arrays."""
        e_asc = np.linspace(280.0, 290.0, 101)
        e_desc = np.linspace(290.0, 280.0, 101)
        intensity = 100.0 * np.exp(-((e_asc - 285.0) ** 2) / 2.0)

        flux_asc = integrate_spectral_flux(intensity, e_asc)
        flux_desc = integrate_spectral_flux(intensity, e_desc)
        self.assertGreater(flux_asc, 0.0)
        self.assertAlmostEqual(flux_asc, flux_desc, places=5)

        # Batched 2D array
        e_batch = np.array([[10.0, 20.0, 30.0], [30.0, 20.0, 10.0]])
        i_batch = np.array([[1.0, 2.0, 1.0], [1.0, 2.0, 1.0]])
        fluxes = integrate_spectral_flux(i_batch, e_batch, axis=-1)
        self.assertEqual(fluxes.shape, (2,))
        self.assertAlmostEqual(float(fluxes[0]), float(fluxes[1]), places=5)


class TestFluxAndStandardization(unittest.TestCase):
    """Test suite for hierarchical flux calculation, time standardization, and leak prevention."""

    def setUp(self) -> None:
        """Prepare minimal deterministic multi-die spectrum dataset."""
        self.x = np.linspace(0.0, 10.0, 11)
        self.ary_energy = np.array([self.x, self.x, self.x, self.x])
        self.ary_intensity = np.array([
            np.full_like(self.x, 2.0),
            np.full_like(self.x, 3.0),
            np.full_like(self.x, 4.0),
            np.full_like(self.x, 6.0),
        ])
        self.df_meta = pd.DataFrame({
            "spectrum_index": [0, 1, 2, 3],
            "measurement_id": ["m1", "m1", "m1", "m1"],
            "die": [1, 1, 2, 2],
            "region": ["C1s", "O1s", "C1s", "O1s"],
            "split": ["train", "train", "train", "train"],
        })

    def test_calculate_flux_and_intensity_hierarchy(self) -> None:
        """Verify hierarchical regional, die, and measurement flux calculations."""
        res = calculate_flux_and_intensity(self.ary_intensity, self.ary_energy, self.df_meta)

        np.testing.assert_allclose(res["mean_intensity_single_region"].to_numpy(), [2.0, 3.0, 4.0, 6.0])
        np.testing.assert_allclose(res["mean_intensity_single_die"].to_numpy(), [2.5, 2.5, 5.0, 5.0])
        np.testing.assert_allclose(res["total_flux_single_region"].to_numpy(), [20.0, 30.0, 40.0, 60.0])
        np.testing.assert_allclose(res["total_flux_single_die"].to_numpy(), [50.0, 50.0, 100.0, 100.0])
        np.testing.assert_allclose(res["total_flux_single_measurement"].to_numpy(), [150.0, 150.0, 150.0, 150.0])
        np.testing.assert_allclose(res["total_flux_single_die_standardized"].to_numpy(), [-1.0, -1.0, 1.0, 1.0])

        # Empty dataframe check
        empty_meta = pd.DataFrame(columns=["spectrum_index", "measurement_id", "die", "region"])
        res_empty = calculate_flux_and_intensity(np.empty((0, 10)), np.empty((0, 10)), empty_meta)
        self.assertIn("total_flux_single_die", res_empty.columns)
        self.assertEqual(len(res_empty), 0)

    def test_metadata_standardization_and_leak_prevention(self) -> None:
        """Verify metadata standardization fits strictly on train split without data leakage."""
        df_meta = pd.DataFrame({
            "measurement_id": ["m1", "m1", "m2", "m2", "m3", "m3"],
            "die": [0, 1, 0, 1, 0, 1],
            "split": ["train", "train", "val", "val", "test", "test"],
            "time": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
                "2026-01-02 00:00:00",
                "2026-01-02 00:00:00",
                "2026-01-03 00:00:00",
                "2026-01-03 00:00:00",
            ],
            "total_flux_single_die": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        })
        res = standardize_metadata_features(df_meta, config={"fit_split": "train"})
        train_flux = res[res["split"] == "train"]["total_flux_single_die_standardized"].to_numpy()
        val_flux = res[res["split"] == "val"]["total_flux_single_die_standardized"].to_numpy()
        test_flux = res[res["split"] == "test"]["total_flux_single_die_standardized"].to_numpy()

        np.testing.assert_allclose(train_flux, [-1.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(val_flux, [3.0, 5.0], atol=1e-5)
        np.testing.assert_allclose(test_flux, [7.0, 9.0], atol=1e-5)

    def test_standardized_time_and_zero_variance_warning(self) -> None:
        """Verify time standardization and warning when sessions share identical timestamps."""
        meta = pd.DataFrame({
            "measurement_id": ["m1", "m1", "m2"],
            "split": ["train", "train", "train"],
            "time": ["2026-01-01 00:00:00", "2026-01-01 00:00:00", "2026-01-01 02:00:00"],
        })
        res = calculate_standardized_time(meta)
        np.testing.assert_allclose(res["time_standardized"].to_numpy(), [-1.0, -1.0, 1.0], rtol=1e-5)

        # Zero std emits warning and falls back to 0.0
        meta_zero = pd.DataFrame({
            "measurement_id": ["m1", "m2"],
            "split": ["train", "train"],
            "time": ["2026-01-01 00:00:00", "2026-01-01 00:00:00"],
        })
        with self.assertWarns(UserWarning):
            res_zero = calculate_standardized_time(meta_zero)
        np.testing.assert_allclose(res_zero["time_standardized"].to_numpy(), [0.0, 0.0], atol=1e-6)

    def test_defensive_error_handling(self) -> None:
        """Table-driven test for missing splits, mismatched inputs, and malformed times."""
        # 1. Missing split KeyError
        meta_no_split = self.df_meta.drop(columns=["split"])
        with self.assertRaises(KeyError):
            calculate_flux_and_intensity(
                self.ary_intensity, self.ary_energy, meta_no_split, config={"fit_split": "train"}
            )
        meta_time_no_split = pd.DataFrame({
            "measurement_id": ["m1", "m2"],
            "time": ["2026-01-01 00:00:00", "2026-01-02 00:00:00"],
        })
        with self.assertRaises(KeyError):
            calculate_standardized_time(meta_time_no_split)

        # 2. Invalid fit split ValueError
        with self.assertRaises(ValueError):
            calculate_flux_and_intensity(
                self.ary_intensity, self.ary_energy, self.df_meta, config={"fit_split": "nonexistent"}
            )
        with self.assertRaises(ValueError):
            calculate_standardized_time(self.df_meta, config={"fit_split": "nonexistent"})

        # 3. Mismatched shapes and bad index
        with self.assertRaises(ValueError):
            calculate_flux_and_intensity(self.ary_intensity[:2], self.ary_energy, self.df_meta)

        bad_meta = self.df_meta.copy()
        bad_meta.loc[0, "spectrum_index"] = 999
        with self.assertRaises(ValueError):
            calculate_flux_and_intensity(self.ary_intensity, self.ary_energy, bad_meta)

        # 4. compute_die_total_flux fast path & error
        f_map = compute_die_total_flux(self.df_meta, self.ary_intensity, self.ary_energy)
        self.assertAlmostEqual(f_map[("m1", 1)], 50.0, places=2)

        nan_meta = pd.DataFrame({
            "measurement_id": ["m1", "m1"],
            "die": [0, 1],
            "spectrum_index": [0, 1],
            "total_flux_single_die": [100.0, np.nan],
        })
        with self.assertRaises(ValueError):
            compute_die_total_flux(nan_meta, self.ary_intensity)

    def test_per_tool_standardization_isolation(self) -> None:
        """Verify standardization statistics are fitted per tool without leakage."""
        meta = pd.DataFrame({
            "measurement_id": ["m1", "m2", "m3", "m4", "m5", "m6"],
            "tool": ["J4", "J4", "J4", "J5", "J5", "J5"],
            "split": ["train", "train", "test", "train", "train", "test"],
            "die": [0, 0, 0, 0, 0, 0],
            "spectrum_index": [0, 1, 2, 3, 4, 5],
            "region": ["Al2p"] * 6,
            "time": [
                "2026-01-01 00:00:00",
                "2026-01-01 02:00:00",
                "2026-01-01 03:00:00",
                "2026-01-01 00:00:00",
                "2026-01-01 04:00:00",
                "2026-01-01 06:00:00",
            ],
            "total_flux_single_die": [100.0, 200.0, 250.0, 1000.0, 3000.0, 4000.0],
        })
        res = standardize_metadata_features(meta)
        self.assertIn("time_standardized", res.columns)
        self.assertIn("total_flux_single_die_standardized", res.columns)
        # Verify J4 test row (m3) uses J4 train stats: (250 - 150) / 50 = 2.0
        self.assertAlmostEqual(res["total_flux_single_die_standardized"].iloc[2], 2.0, places=4)
        # Verify J5 test row (m6) uses J5 train stats: (4000 - 2000) / 1000 = 2.0
        self.assertAlmostEqual(res["total_flux_single_die_standardized"].iloc[5], 2.0, places=4)


if __name__ == "__main__":
    unittest.main()
