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
            "split": ["train", "train", "train", "train"],
        })

    def test_hierarchical_flux_and_intensity_metrics(self) -> None:
        """Verify exact hierarchical relationships of flux and intensity columns."""
        df_res = calculate_flux_and_intensity(
            self.ary_intensity, self.ary_energy, self.df_meta
        )

        # Expected regional mean intensities
        # Die 1: region 0 is 2.0, region 1 is 3.0 -> die mean is 2.5
        # Die 2: region 0 is 4.0, region 1 is 6.0 -> die mean is 5.0
        np.testing.assert_allclose(
            df_res["mean_intensity_single_region"].to_numpy(),
            [2.0, 3.0, 4.0, 6.0],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            df_res["mean_intensity_single_die"].to_numpy(),
            [2.5, 2.5, 5.0, 5.0],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            df_res["relative_intensity_single_region"].to_numpy(),
            [2.0 / 2.5, 3.0 / 2.5, 4.0 / 5.0, 6.0 / 5.0],
            rtol=1e-5,
        )

        # Expected regional fluxes
        np.testing.assert_allclose(
            df_res["total_flux_single_region"].to_numpy(),
            [20.0, 30.0, 40.0, 60.0],
            rtol=1e-5,
        )

        # Die 1 has regions with flux 20 and 30 -> mean 25, total 50
        # Die 2 has regions with flux 40 and 60 -> mean 50, total 100
        np.testing.assert_allclose(
            df_res["mean_flux_across_regions_single_die"].to_numpy(),
            [25.0, 25.0, 50.0, 50.0],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            df_res["total_flux_single_die"].to_numpy(),
            [50.0, 50.0, 100.0, 100.0],
            rtol=1e-5,
        )

        # Measurement m1 has 2 dies with die flux 50 and 100
        # mean_flux_across_dies = (50 + 100) / 2 = 75.0
        # total_flux_single_measurement = 50 + 100 = 150.0
        np.testing.assert_allclose(
            df_res["mean_flux_across_dies_single_measurement"].to_numpy(),
            [75.0, 75.0, 75.0, 75.0],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            df_res["total_flux_single_measurement"].to_numpy(),
            [150.0, 150.0, 150.0, 150.0],
            rtol=1e-5,
        )
        # total_flux_single_die_standardized for unique dies:
        # [50.0, 100.0] -> [-1.0, 1.0]
        np.testing.assert_allclose(
            df_res["total_flux_single_die_standardized"].to_numpy(),
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
            "split": ["train", "train"],
        })
        with self.assertWarns(UserWarning):
            res = calculate_flux_and_intensity(
                identical_intensity, self.ary_energy[:2], meta
            )
        np.testing.assert_allclose(
            res["total_flux_single_die_standardized"].to_numpy(),
            [0.0, 0.0],
            atol=1e-6,
        )

    def test_calculate_flux_missing_split_raises_key_error(self) -> None:
        """Verify KeyError raised when fit_split is specified but
        'split' column is missing.
        """
        bad_meta = self.df_meta.drop(columns=["split"])
        with self.assertRaises(KeyError):
            calculate_flux_and_intensity(
                self.ary_intensity,
                self.ary_energy,
                bad_meta,
                config={"fit_split": "train"},
            )

    def test_calculate_flux_invalid_fit_split_raises_value_error(self) -> None:
        """Verify ValueError raised when fit_split is not found in
        df_meta['split'].
        """
        with self.assertRaises(ValueError):
            calculate_flux_and_intensity(
                self.ary_intensity,
                self.ary_energy,
                self.df_meta,
                config={"fit_split": "nonexistent"},
            )

    def test_calculate_flux_unsplit_leaves_die_flux_normalized_nan(self) -> None:
        """Verify total_flux_single_die_standardized is NaN when df_meta has
        no 'split' column.
        """
        bad_meta = self.df_meta.drop(columns=["split"])
        res = calculate_flux_and_intensity(
            self.ary_intensity, self.ary_energy, bad_meta
        )
        self.assertTrue(res["total_flux_single_die_standardized"].isna().all())

    def test_df_meta_immutability(self) -> None:
        """Verify original df_meta DataFrame is not mutated in-place."""
        cols_before = list(self.df_meta.columns)
        _ = calculate_flux_and_intensity(
            self.ary_intensity, self.ary_energy, self.df_meta
        )
        self.assertEqual(list(self.df_meta.columns), cols_before)

    def test_empty_dataframe(self) -> None:
        """Verify graceful handling of empty metadata DataFrame."""
        empty_meta = pd.DataFrame(
            columns=["spectrum_index", "measurement_id", "die", "region"]
        )
        res = calculate_flux_and_intensity(
            np.empty((0, 10)), np.empty((0, 10)), empty_meta
        )
        for col in [
            "mean_intensity_single_region",
            "mean_intensity_single_die",
            "relative_intensity_single_region",
            "total_flux_single_region",
            "mean_flux_across_regions_single_die",
            "total_flux_single_die",
            "mean_flux_across_dies_single_measurement",
            "total_flux_single_measurement",
            "total_flux_single_die_standardized",
        ]:
            self.assertIn(col, res.columns)
        self.assertEqual(len(res), 0)

    def test_mismatched_intensity_energy_raises(self) -> None:
        """Verify ValueError raised when intensity and energy lengths differ."""
        with self.assertRaises(ValueError):
            calculate_flux_and_intensity(
                self.ary_intensity[:2], self.ary_energy, self.df_meta
            )

    def test_missing_required_columns_raises(self) -> None:
        """Verify ValueError raised when required columns are absent."""
        bad_meta = self.df_meta.drop(columns=["region"])
        with self.assertRaises(ValueError):
            calculate_flux_and_intensity(
                self.ary_intensity, self.ary_energy, bad_meta
            )

    def test_out_of_bounds_index_raises(self) -> None:
        """Verify ValueError raised when spectrum_index is out-of-bounds."""
        bad_meta = self.df_meta.copy()
        bad_meta.loc[0, "spectrum_index"] = 999
        with self.assertRaises(ValueError):
            calculate_flux_and_intensity(
                self.ary_intensity, self.ary_energy, bad_meta
            )

    def test_negative_index_raises(self) -> None:
        """Verify ValueError raised when spectrum_index has negative value."""
        bad_meta = self.df_meta.copy()
        bad_meta.loc[0, "spectrum_index"] = -1
        with self.assertRaises(ValueError):
            calculate_flux_and_intensity(
                self.ary_intensity, self.ary_energy, bad_meta
            )

    def test_non_default_dataframe_index(self) -> None:
        """Verify calculate_flux_and_intensity preserves non-standard
        DataFrame index.
        """
        custom_meta = self.df_meta.copy()
        custom_meta.index = pd.Index([10, 42, 100, 255])
        res = calculate_flux_and_intensity(
            self.ary_intensity, self.ary_energy, custom_meta
        )
        self.assertEqual(list(res.index), [10, 42, 100, 255])
        self.assertFalse(res["mean_flux_across_dies_single_measurement"].isna().any())

    def test_multi_measurement_isolation(self) -> None:
        """Verify hierarchical metrics correctly isolate multiple sessions."""
        meta = pd.DataFrame({
            "spectrum_index": [0, 1, 2, 3],
            "measurement_id": ["m1", "m1", "m2", "m2"],
            "die": [1, 2, 1, 1],
            "region": ["C1s", "C1s", "C1s", "O1s"],
        })
        res = calculate_flux_and_intensity(
            self.ary_intensity, self.ary_energy, meta
        )
        # m1 has die 1 (flux 20) and die 2 (flux 30) -> total 50, mean die 25
        m1_rows = res[res["measurement_id"] == "m1"]
        self.assertTrue((m1_rows["total_flux_single_measurement"] == 50.0).all())
        self.assertTrue((m1_rows["mean_flux_across_dies_single_measurement"] == 25.0).all())
        # m2 has 1 die (die 1) with 2 regions (flux 40 and 60) -> total 100
        m2_rows = res[res["measurement_id"] == "m2"]
        self.assertTrue((m2_rows["total_flux_single_measurement"] == 100.0).all())
        self.assertTrue((m2_rows["mean_flux_across_dies_single_measurement"] == 100.0).all())

    def test_integration_with_synthetic_pipeline(self) -> None:
        """Verify calculate_flux_and_intensity works with synthetic data."""
        config = get_default_measurement_config()
        config["N_measurement"] = 2
        config["N_die"] = 2
        i_arr, e_arr, meta = generate_pseudo_measurements(config)
        meta["split"] = "train"
        meta_with_flux = calculate_flux_and_intensity(i_arr, e_arr, meta)

        self.assertEqual(len(meta_with_flux), len(meta))
        for col in [
            "mean_intensity_single_region",
            "mean_intensity_single_die",
            "relative_intensity_single_region",
            "total_flux_single_region",
            "mean_flux_across_regions_single_die",
            "total_flux_single_die",
            "mean_flux_across_dies_single_measurement",
            "total_flux_single_measurement",
            "total_flux_single_die_standardized",
        ]:
            self.assertIn(col, meta_with_flux.columns)
            self.assertFalse(meta_with_flux[col].isna().any())
        self.assertTrue((meta_with_flux["total_flux_single_region"] > 0).all())


class TestCalculateStandardizedTime(unittest.TestCase):
    """Test suite for calculate_standardized_time in utility.flux."""

    def test_standardized_time_across_unique_measurements(self) -> None:
        """Verify z-score standardization across unique measurement sessions."""
        # m1 has 2 rows, m2 has 1 row -> unique times are 00:00 and 02:00
        # mean = 01:00, diff = [-3600s, +3600s], std = 3600s -> [-1.0, +1.0]
        meta = pd.DataFrame({
            "measurement_id": ["m1", "m1", "m2"],
            "split": ["train", "train", "train"],
            "time": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
                "2026-01-01 02:00:00",
            ],
        })
        res = calculate_standardized_time(meta)
        self.assertIn("time_standardized", res.columns)
        np.testing.assert_allclose(
            res["time_standardized"].to_numpy(),
            [-1.0, -1.0, 1.0],
            rtol=1e-5,
        )

    def test_zero_std_time_emits_warning(self) -> None:
        """Verify warning and 0.0 output when measurements share timestamp."""
        meta = pd.DataFrame({
            "measurement_id": ["m1", "m2"],
            "split": ["train", "train"],
            "time": ["2026-01-01 00:00:00", "2026-01-01 00:00:00"],
        })
        with self.assertWarns(UserWarning):
            res = calculate_standardized_time(meta)
        np.testing.assert_allclose(
            res["time_standardized"].to_numpy(), [0.0, 0.0], atol=1e-6
        )

    def test_missing_split_raises_key_error(self) -> None:
        """Verify KeyError is raised when df_meta lacks 'split' column."""
        meta = pd.DataFrame({
            "measurement_id": ["m1", "m2"],
            "time": ["2026-01-01 00:00:00", "2026-01-02 00:00:00"],
        })
        with self.assertRaises(KeyError):
            calculate_standardized_time(meta)

    def test_invalid_fit_split_raises_value_error(self) -> None:
        """Verify ValueError is raised when fit_split is absent from
        df_meta['split'].
        """
        meta = pd.DataFrame({
            "measurement_id": ["m1", "m2"],
            "split": ["val", "val"],
            "time": ["2026-01-01 00:00:00", "2026-01-02 00:00:00"],
        })
        with self.assertRaises(ValueError):
            calculate_standardized_time(meta, config={"fit_split": "train"})

    def test_empty_and_invalid_time_inputs(self) -> None:
        """Verify empty DataFrame handling and ValueError on invalid inputs."""
        empty = pd.DataFrame(columns=["measurement_id", "time"])
        res = calculate_standardized_time(empty)
        self.assertIn("time_standardized", res.columns)
        self.assertEqual(len(res), 0)

        with self.assertRaises(ValueError):
            calculate_standardized_time(
                pd.DataFrame({"measurement_id": ["m1"], "split": ["train"]})
            )

        with self.assertRaises(ValueError):
            calculate_standardized_time(
                pd.DataFrame({
                    "measurement_id": ["m1"],
                    "split": ["train"],
                    "time": [None],
                })
            )

        with self.assertRaises(ValueError):
            calculate_standardized_time(
                pd.DataFrame({
                    "measurement_id": [None],
                    "split": ["train"],
                    "time": ["2026-01-01 00:00:00"],
                })
            )

        with self.assertRaises(ValueError):
            calculate_standardized_time(
                pd.DataFrame({
                    "measurement_id": ["m1"],
                    "split": ["train"],
                    "time": ["not-a-date"],
                })
            )


class TestComputeDieTotalFlux(unittest.TestCase):
    """Test suite for compute_die_total_flux in utility.flux_and_intensity."""

    def test_nan_die_flux_raises_value_error(self) -> None:
        """Verify ValueError raised if row has NaN total_flux_single_die."""
        df_meta = pd.DataFrame({
            "measurement_id": ["m1", "m1", "m2"],
            "die": [0, 1, 0],
            "spectrum_index": [0, 1, 2],
            "total_flux_single_die": [100.0, np.nan, 80.0],
        })
        ary_intensity = np.ones((3, 10))
        with self.assertRaises(ValueError) as ctx:
            compute_die_total_flux(df_meta, ary_intensity)
        self.assertIn("invalid or NaN 'total_flux_single_die'", str(ctx.exception))

    def test_existing_column_fast_path(self) -> None:
        """Verify compute_die_total_flux directly uses total_flux_single_die
        if present.
        """
        df_meta = pd.DataFrame({
            "measurement_id": ["m1", "m1", "m2"],
            "die": [0, 1, 0],
            "spectrum_index": [0, 1, 2],
            "total_flux_single_die": [100.0, 50.0, 80.0],
        })
        ary_intensity = np.ones((3, 10))
        f_map = compute_die_total_flux(df_meta, ary_intensity)
        self.assertEqual(f_map[("m1", 0)], 100.0)
        self.assertEqual(f_map[("m1", 1)], 50.0)
        self.assertEqual(f_map[("m2", 0)], 80.0)


class TestStandardizeMetadataFeatures(unittest.TestCase):
    """Test suite for split-aware metadata standardization without data
    leakage.
    """

    def setUp(self) -> None:
        self.df_meta = pd.DataFrame({
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

    def test_standardize_on_train_split_only(self) -> None:
        """Verify mean and std are fit on train split only."""
        res = standardize_metadata_features(
            self.df_meta, config={"fit_split": "train"}
        )
        self.assertIn("time_standardized", res.columns)
        self.assertIn("total_flux_single_die_standardized", res.columns)

        # Train dies are 10.0 and 20.0 -> mean 15.0, std 5.0
        train_flux = res[res["split"] == "train"]["total_flux_single_die_standardized"].to_numpy()
        np.testing.assert_allclose(train_flux, [-1.0, 1.0], atol=1e-5)

        # Val dies (30.0, 40.0) -> (30-15)/5 = 3.0, (40-15)/5 = 5.0
        val_flux = res[res["split"] == "val"]["total_flux_single_die_standardized"].to_numpy()
        np.testing.assert_allclose(val_flux, [3.0, 5.0], atol=1e-5)

        # Test dies (50.0, 60.0) -> (50-15)/5 = 7.0, (60-15)/5 = 9.0
        test_flux = res[res["split"] == "test"]["total_flux_single_die_standardized"].to_numpy()
        np.testing.assert_allclose(test_flux, [7.0, 9.0], atol=1e-5)

    def test_calculate_standardized_time_with_fit_split(self) -> None:
        """Verify calculate_standardized_time fits timestamps strictly on
        train sessions.
        """
        df = pd.DataFrame({
            "measurement_id": ["m1", "m2", "m3"],
            "split": ["train", "train", "test"],
            "time": [
                "2026-01-01 00:00:00",
                "2026-01-03 00:00:00",
                "2026-01-05 00:00:00",
            ],
        })
        res = calculate_standardized_time(df, config={"fit_split": "train"})
        # Train timestamps: day 0 and day 2 -> mean day 1.
        # Diffs are -1 day and +1 day; std across train is 1 day (86400s).
        train_t = res[res["split"] == "train"]["time_standardized"].to_numpy()
        np.testing.assert_allclose(train_t, [-1.0, 1.0], atol=1e-5)

        # Test timestamp day 4 -> (4 - 1)/1 = 3.0
        test_t = res[res["split"] == "test"]["time_standardized"].to_numpy()
        np.testing.assert_allclose(test_t, [3.0], atol=1e-5)

    def test_missing_split_raises_key_error(self) -> None:
        """Verify KeyError raised when df_meta lacks 'split' column."""
        bad_meta = self.df_meta.drop(columns=["split"])
        with self.assertRaises(KeyError):
            standardize_metadata_features(bad_meta)

    def test_invalid_fit_split_raises_value_error(self) -> None:
        """Verify ValueError raised when fit_split does not exist in
        df_meta['split'].
        """
        with self.assertRaises(ValueError):
            standardize_metadata_features(
                self.df_meta, config={"fit_split": "nonexistent"}
            )


if __name__ == "__main__":
    unittest.main()
