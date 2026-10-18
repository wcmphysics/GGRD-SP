"""Unit tests for utility.flux module."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from utility.flux import (
    calculate_flux,
    calculate_normalized_time,
    compute_die_total_flux,
    integrate_spectral_flux,
    trapezoid_integral,
)
from utility.pseudo_measurement import (
    generate_pseudo_measurements,
    get_default_measurement_config,
)


class TestTrapezoidIntegration(unittest.TestCase):
    """Test suite for low-level numerical trapezoidal integration."""

    def test_constant_function_integral(self) -> None:
        """Verify integration of constant function y(x) = c over [a, b]."""
        x = np.linspace(0.0, 10.0, 101)
        y = np.full_like(x, 5.0)
        result = float(trapezoid_integral(y, x))
        self.assertAlmostEqual(result, 50.0, places=5)

    def test_linear_function_integral(self) -> None:
        """Verify integration of linear function y(x) = 2x over [0, 5]."""
        x = np.linspace(0.0, 5.0, 501)
        y = 2.0 * x
        # Exact: [x^2]_0^5 = 25.0
        result = float(trapezoid_integral(y, x))
        self.assertAlmostEqual(result, 25.0, places=4)

    def test_batched_integration(self) -> None:
        """Verify batched integration along axis=-1."""
        x = np.linspace(0.0, 2.0, 21)
        y = np.vstack([np.ones_like(x), 2.0 * np.ones_like(x)])
        res = trapezoid_integral(y, x, axis=-1)
        self.assertEqual(res.shape, (2,))
        self.assertAlmostEqual(float(res[0]), 2.0, places=5)
        self.assertAlmostEqual(float(res[1]), 4.0, places=5)


class TestIntegrateSpectralFlux(unittest.TestCase):
    """Test suite for absolute spectral flux integration."""

    def test_direction_invariance(self) -> None:
        """Verify ascending and descending energy arrays yield equal flux."""
        e_asc = np.linspace(280.0, 290.0, 101)
        e_desc = np.linspace(290.0, 280.0, 101)
        intensity = 100.0 * np.exp(-((e_asc - 285.0) ** 2) / 2.0)

        flux_asc = integrate_spectral_flux(intensity, e_asc)
        flux_desc = integrate_spectral_flux(intensity, e_desc)

        self.assertGreater(flux_asc, 0.0)
        self.assertGreater(flux_desc, 0.0)
        self.assertAlmostEqual(flux_asc, flux_desc, places=5)

    def test_batched_positive_flux(self) -> None:
        """Verify batched 2D inputs return 1D positive array."""
        e = np.array([
            [10.0, 20.0, 30.0],
            [30.0, 20.0, 10.0],
        ])
        i = np.array([
            [1.0, 2.0, 1.0],
            [1.0, 2.0, 1.0],
        ])
        fluxes = integrate_spectral_flux(i, e, axis=-1)
        self.assertEqual(fluxes.shape, (2,))
        self.assertAlmostEqual(float(fluxes[0]), float(fluxes[1]), places=5)
        self.assertGreater(float(fluxes[0]), 0.0)


class TestCalculateFlux(unittest.TestCase):
    """Test suite for hierarchical flux calculation in df_meta."""

    def setUp(self) -> None:
        """Prepare minimal deterministic multi-region spectrum test data."""
        # 1 measurement, 2 dies, 2 regions per die -> 4 spectra
        # Region 0: rect of height 2 over [0, 10] -> area 20
        # Region 1: rect of height 3 over [0, 10] -> area 30
        self.x = np.linspace(0.0, 10.0, 11)
        self.ary_energy = np.array([self.x, self.x, self.x, self.x])
        self.ary_intensity = np.array([
            np.full_like(self.x, 2.0),  # Die 1, Region 0 -> 20.0
            np.full_like(self.x, 3.0),  # Die 1, Region 1 -> 30.0
            np.full_like(self.x, 4.0),  # Die 2, Region 0 -> 40.0
            np.full_like(self.x, 6.0),  # Die 2, Region 1 -> 60.0
        ])
        self.df_meta = pd.DataFrame({
            "spectrum_index": [0, 1, 2, 3],
            "measurement_id": ["m1", "m1", "m1", "m1"],
            "die": [1, 1, 2, 2],
            "region": ["C1s", "O1s", "C1s", "O1s"],
        })

    def test_hierarchical_flux_metrics(self) -> None:
        """Verify exact hierarchical relationships of computed flux columns."""
        df_res = calculate_flux(self.ary_intensity, self.ary_energy, self.df_meta)

        # Expected regional fluxes
        np.testing.assert_allclose(
            df_res["regional_flux"].to_numpy(),
            [20.0, 30.0, 40.0, 60.0],
            rtol=1e-5,
        )

        # Die 1 has regions with flux 20 and 30 -> mean 25, total 50
        # Die 2 has regions with flux 40 and 60 -> mean 50, total 100
        np.testing.assert_allclose(
            df_res["mean_regional_flux"].to_numpy(),
            [25.0, 25.0, 50.0, 50.0],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            df_res["die_flux"].to_numpy(),
            [50.0, 50.0, 100.0, 100.0],
            rtol=1e-5,
        )

        # Measurement m1 has 2 dies with die_flux 50 and 100
        # mean_die_flux = (50 + 100) / 2 = 75.0
        # measurement_flux = 50 + 100 = 150.0
        np.testing.assert_allclose(
            df_res["mean_die_flux"].to_numpy(),
            [75.0, 75.0, 75.0, 75.0],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            df_res["measurement_flux"].to_numpy(),
            [150.0, 150.0, 150.0, 150.0],
            rtol=1e-5,
        )
        # die_flux_normalized for unique dies [50.0, 100.0] -> [-1.0, 1.0]
        np.testing.assert_allclose(
            df_res["die_flux_normalized"].to_numpy(),
            [-1.0, -1.0, 1.0, 1.0],
            rtol=1e-5,
        )

    def test_zero_std_die_flux_emits_warning(self) -> None:
        """Verify warning and 0.0 fallback when all dies have identical flux."""
        identical_intensity = np.full((2, 11), 2.0)
        meta = pd.DataFrame({
            "spectrum_index": [0, 1],
            "measurement_id": ["m1", "m1"],
            "die": [1, 2],
            "region": ["C1s", "C1s"],
        })
        with self.assertWarns(UserWarning):
            res = calculate_flux(identical_intensity, self.ary_energy[:2], meta)
        np.testing.assert_allclose(
            res["die_flux_normalized"].to_numpy(), [0.0, 0.0], atol=1e-6
        )

    def test_df_meta_immutability(self) -> None:
        """Verify original df_meta DataFrame is not mutated in-place."""
        cols_before = list(self.df_meta.columns)
        _ = calculate_flux(self.ary_intensity, self.ary_energy, self.df_meta)
        self.assertEqual(list(self.df_meta.columns), cols_before)

    def test_empty_dataframe(self) -> None:
        """Verify graceful handling of empty metadata DataFrame."""
        empty_meta = pd.DataFrame(
            columns=["spectrum_index", "measurement_id", "die", "region"]
        )
        res = calculate_flux(
            np.empty((0, 10)), np.empty((0, 10)), empty_meta
        )
        for col in [
            "regional_flux",
            "mean_regional_flux",
            "die_flux",
            "mean_die_flux",
            "measurement_flux",
            "die_flux_normalized",
        ]:
            self.assertIn(col, res.columns)
        self.assertEqual(len(res), 0)

    def test_mismatched_intensity_energy_raises(self) -> None:
        """Verify ValueError raised when intensity and energy lengths differ."""
        with self.assertRaises(ValueError):
            calculate_flux(
                self.ary_intensity[:2], self.ary_energy, self.df_meta
            )

    def test_missing_required_columns_raises(self) -> None:
        """Verify ValueError raised when required columns are absent."""
        bad_meta = self.df_meta.drop(columns=["region"])
        with self.assertRaises(ValueError):
            calculate_flux(self.ary_intensity, self.ary_energy, bad_meta)

    def test_out_of_bounds_index_raises(self) -> None:
        """Verify ValueError raised when spectrum_index is out-of-bounds."""
        bad_meta = self.df_meta.copy()
        bad_meta.loc[0, "spectrum_index"] = 999
        with self.assertRaises(ValueError):
            calculate_flux(self.ary_intensity, self.ary_energy, bad_meta)

    def test_negative_index_raises(self) -> None:
        """Verify ValueError raised when spectrum_index has negative value."""
        bad_meta = self.df_meta.copy()
        bad_meta.loc[0, "spectrum_index"] = -1
        with self.assertRaises(ValueError):
            calculate_flux(self.ary_intensity, self.ary_energy, bad_meta)

    def test_non_default_dataframe_index(self) -> None:
        """Verify calculate_flux preserves non-standard DataFrame index."""
        custom_meta = self.df_meta.copy()
        custom_meta.index = pd.Index([10, 42, 100, 255])
        res = calculate_flux(self.ary_intensity, self.ary_energy, custom_meta)
        self.assertEqual(list(res.index), [10, 42, 100, 255])
        self.assertFalse(res["mean_die_flux"].isna().any())

    def test_multi_measurement_isolation(self) -> None:
        """Verify hierarchical metrics correctly isolate multiple sessions."""
        meta = pd.DataFrame({
            "spectrum_index": [0, 1, 2, 3],
            "measurement_id": ["m1", "m1", "m2", "m2"],
            "die": [1, 2, 1, 1],
            "region": ["C1s", "C1s", "C1s", "O1s"],
        })
        res = calculate_flux(self.ary_intensity, self.ary_energy, meta)
        # m1 has die 1 (flux 20) and die 2 (flux 30) -> total 50, mean die 25
        m1_rows = res[res["measurement_id"] == "m1"]
        self.assertTrue((m1_rows["measurement_flux"] == 50.0).all())
        self.assertTrue((m1_rows["mean_die_flux"] == 25.0).all())
        # m2 has 1 die (die 1) with 2 regions (flux 40 and 60) -> total 100
        m2_rows = res[res["measurement_id"] == "m2"]
        self.assertTrue((m2_rows["measurement_flux"] == 100.0).all())
        self.assertTrue((m2_rows["mean_die_flux"] == 100.0).all())

    def test_integration_with_synthetic_pipeline(self) -> None:
        """Verify calculate_flux works with generate_pseudo_measurements."""
        config = get_default_measurement_config()
        config["N_measurement"] = 2
        config["N_die"] = 2
        i_arr, e_arr, meta = generate_pseudo_measurements(config)
        meta_with_flux = calculate_flux(i_arr, e_arr, meta)

        self.assertEqual(len(meta_with_flux), len(meta))
        for col in [
            "regional_flux",
            "mean_regional_flux",
            "die_flux",
            "mean_die_flux",
            "measurement_flux",
        ]:
            self.assertIn(col, meta_with_flux.columns)
            self.assertFalse(meta_with_flux[col].isna().any())
            self.assertTrue((meta_with_flux[col] > 0).all())
        self.assertIn("die_flux_normalized", meta_with_flux.columns)
        self.assertFalse(meta_with_flux["die_flux_normalized"].isna().any())


class TestCalculateNormalizedTime(unittest.TestCase):
    """Test suite for calculate_normalized_time in utility.flux."""

    def test_normalized_time_across_unique_measurements(self) -> None:
        """Verify z-score standardization across unique measurement sessions."""
        # m1 has 2 rows, m2 has 1 row -> unique times are 00:00 and 02:00
        # mean = 01:00, diff = [-3600s, +3600s], std = 3600s -> [-1.0, +1.0]
        meta = pd.DataFrame({
            "measurement_id": ["m1", "m1", "m2"],
            "time": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
                "2026-01-01 02:00:00",
            ],
        })
        res = calculate_normalized_time(meta)
        self.assertIn("time_normalized", res.columns)
        np.testing.assert_allclose(
            res["time_normalized"].to_numpy(),
            [-1.0, -1.0, 1.0],
            rtol=1e-5,
        )

    def test_zero_std_time_emits_warning(self) -> None:
        """Verify warning and 0.0 output when measurements share timestamp."""
        meta = pd.DataFrame({
            "measurement_id": ["m1", "m2"],
            "time": ["2026-01-01 00:00:00", "2026-01-01 00:00:00"],
        })
        with self.assertWarns(UserWarning):
            res = calculate_normalized_time(meta)
        np.testing.assert_allclose(
            res["time_normalized"].to_numpy(), [0.0, 0.0], atol=1e-6
        )

    def test_empty_and_invalid_time_inputs(self) -> None:
        """Verify empty DataFrame handling and ValueError on invalid inputs."""
        empty = pd.DataFrame(columns=["measurement_id", "time"])
        res = calculate_normalized_time(empty)
        self.assertIn("time_normalized", res.columns)
        self.assertEqual(len(res), 0)

        with self.assertRaises(ValueError):
            calculate_normalized_time(pd.DataFrame({"measurement_id": ["m1"]}))

        with self.assertRaises(ValueError):
            calculate_normalized_time(
                pd.DataFrame({"measurement_id": ["m1"], "time": [None]})
            )

        with self.assertRaises(ValueError):
            calculate_normalized_time(
                pd.DataFrame(
                    {"measurement_id": [None], "time": ["2026-01-01 00:00:00"]}
                )
            )

        with self.assertRaises(ValueError):
            calculate_normalized_time(
                pd.DataFrame({"measurement_id": ["m1"], "time": ["not-a-date"]})
            )


class TestComputeDieTotalFlux(unittest.TestCase):
    """Test suite for compute_die_total_flux in utility.flux."""

    def test_nan_die_flux_raises_value_error(self) -> None:
        """Verify ValueError is raised if any row has NaN die_flux when column exists."""
        df_meta = pd.DataFrame({
            "measurement_id": ["m1", "m1", "m2"],
            "die": [0, 1, 0],
            "spectrum_index": [0, 1, 2],
            "die_flux": [100.0, np.nan, 80.0],
        })
        ary_intensity = np.ones((3, 10))
        with self.assertRaises(ValueError) as ctx:
            compute_die_total_flux(df_meta, ary_intensity)
        self.assertIn("invalid or NaN 'die_flux'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
