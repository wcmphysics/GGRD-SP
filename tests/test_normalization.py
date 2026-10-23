"""Unit tests for Stage 2 spectral normalization."""

from __future__ import annotations

import unittest
import warnings
import numpy as np
import pandas as pd

from utility.normalization import normalize_spectral_data


class TestNormalization(unittest.TestCase):
    """Test suite for normalize_spectral_data."""

    def setUp(self) -> None:
        """Create sample spectral arrays and metadata DataFrame."""
        n_points = 20
        self.ary_energy = np.tile(np.linspace(10.0, 20.0, n_points), (4, 1))
        self.ary_intensity = np.array(
            [
                np.full(n_points, 100.0, dtype=np.float32),
                np.full(n_points, 200.0, dtype=np.float32),
                np.full(n_points, 300.0, dtype=np.float32),
                np.full(n_points, 400.0, dtype=np.float32),
            ],
            dtype=np.float32,
        )
        self.df_meta = pd.DataFrame(
            [
                {
                    "spectrum_index": 0,
                    "measurement_id": "M1",
                    "die": 0,
                    "region": "Al2p",
                    "mean_intensity_single_die": 150.0,
                },
                {
                    "spectrum_index": 1,
                    "measurement_id": "M1",
                    "die": 0,
                    "region": "Ti2p",
                    "mean_intensity_single_die": 150.0,
                },
                {
                    "spectrum_index": 2,
                    "measurement_id": "M2",
                    "die": 1,
                    "region": "Al2p",
                    "mean_intensity_single_die": 350.0,
                },
                {
                    "spectrum_index": 3,
                    "measurement_id": "M2",
                    "die": 1,
                    "region": "Ti2p",
                    "mean_intensity_single_die": 350.0,
                },
            ]
        )

    def test_normalize_by_total_flux_single_die(self) -> None:
        """Verify by_total_flux_single_die mode divides each region by die flux."""
        norm_i, norm_e, meta = normalize_spectral_data(
            self.ary_intensity,
            self.ary_energy,
            self.df_meta,
            config={"normalization_mode": "by_total_flux_single_die"},
        )
        self.assertEqual(norm_i.shape, self.ary_intensity.shape)
        np.testing.assert_array_equal(norm_e, self.ary_energy)

        # Die 0 total flux: area(0) + area(1) = (100 * 10) + (200 * 10) = 3000
        expected_0 = 100.0 / 3000.0
        np.testing.assert_allclose(norm_i[0], expected_0, rtol=1e-5)

    def test_normalize_by_mean_intensity_single_die(self) -> None:
        """Verify by_mean_intensity_single_die divides each region by die mean."""
        norm_i, norm_e, meta = normalize_spectral_data(
            self.ary_intensity,
            self.ary_energy,
            self.df_meta,
            config={"normalization_mode": "by_mean_intensity_single_die"},
        )
        self.assertEqual(norm_i.shape, self.ary_intensity.shape)
        # Die 0: 100 / 150 = 2/3, 200 / 150 = 4/3
        np.testing.assert_allclose(norm_i[0], 100.0 / 150.0, rtol=1e-5)
        np.testing.assert_allclose(norm_i[1], 200.0 / 150.0, rtol=1e-5)
        # Die 1: 300 / 350, 400 / 350
        np.testing.assert_allclose(norm_i[2], 300.0 / 350.0, rtol=1e-5)
        np.testing.assert_allclose(norm_i[3], 400.0 / 350.0, rtol=1e-5)

    def test_normalize_by_mean_intensity_single_die_missing_column_raises(
        self,
    ) -> None:
        """Verify ValueError is raised if mean_intensity_single_die missing."""
        df_no_col = self.df_meta.drop(columns=["mean_intensity_single_die"])
        with self.assertRaises(ValueError) as ctx:
            normalize_spectral_data(
                self.ary_intensity,
                self.ary_energy,
                df_no_col,
                config={"normalization_mode": "by_mean_intensity_single_die"},
            )
        self.assertIn("mean_intensity_single_die", str(ctx.exception))

    def test_normalize_by_regional_max_intensity(self) -> None:
        """Verify by_regional_max_intensity normalizes each spectrum to peak 1.0."""
        norm_i, norm_e, meta = normalize_spectral_data(
            self.ary_intensity,
            self.ary_energy,
            self.df_meta,
            config={"normalization_mode": "by_regional_max_intensity"},
        )
        for idx in range(4):
            self.assertAlmostEqual(float(norm_i[idx].max()), 1.0, places=5)

    def test_normalize_no_normalization(self) -> None:
        """Verify no_normalization mode leaves intensity unscaled."""
        norm_i, norm_e, meta = normalize_spectral_data(
            self.ary_intensity,
            self.ary_energy,
            self.df_meta,
            config={"normalization_mode": "no_normalization"},
        )
        self.assertEqual(norm_i.shape, self.ary_intensity.shape)
        np.testing.assert_array_equal(norm_i, self.ary_intensity)
        np.testing.assert_array_equal(norm_e, self.ary_energy)
        pd.testing.assert_frame_equal(meta, self.df_meta)

    def test_normalize_by_regional_max_intensity_of_source(self) -> None:
        """Verify by_regional_max_intensity_of_source divides paired source and target
        by source peak while leaving unpaired spectra unscaled.
        """
        paired_meta = pd.DataFrame(
            [
                {
                    "spectrum_index": 0,
                    "measurement_id": "M1",
                    "die": 0,
                    "region": "Al2p",
                    "tool": "J4",
                    "spectrum_index_target": 2,
                },
                {
                    "spectrum_index": 1,
                    "measurement_id": "M1",
                    "die": 0,
                    "region": "Ti2p",
                    "tool": "J4",
                    "spectrum_index_target": None,
                },
                {
                    "spectrum_index": 2,
                    "measurement_id": "M2",
                    "die": 0,
                    "region": "Al2p",
                    "tool": "H1",
                    "spectrum_index_target": None,
                },
                {
                    "spectrum_index": 3,
                    "measurement_id": "M2",
                    "die": 0,
                    "region": "Ti2p",
                    "tool": "H1",
                    "spectrum_index_target": None,
                },
            ]
        )
        norm_i, norm_e, meta = normalize_spectral_data(
            self.ary_intensity,
            self.ary_energy,
            paired_meta,
            config={"normalization_mode": "by_regional_max_intensity_of_source"},
        )
        # Source 0 (peak 100): 100 / 100 = 1.0
        np.testing.assert_allclose(norm_i[0], 1.0, rtol=1e-5)
        # Paired Target 2 (orig 300): 300 / 100 = 3.0
        np.testing.assert_allclose(norm_i[2], 3.0, rtol=1e-5)
        # Unpaired Source 1: left unscaled at 200.0
        np.testing.assert_allclose(norm_i[1], 200.0, rtol=1e-5)
        # Unpaired Target 3: left unscaled at 400.0
        np.testing.assert_allclose(norm_i[3], 400.0, rtol=1e-5)

    def test_normalize_scale_less_than_eps_emits_warning(self) -> None:
        """Verify warning is emitted and scale clamped when scale <= eps."""
        zero_intensity = np.zeros_like(self.ary_intensity)
        df_zero = self.df_meta.copy()
        df_zero["mean_intensity_single_die"] = 0.0

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            norm_i, _, _ = normalize_spectral_data(
                zero_intensity,
                self.ary_energy,
                df_zero,
                config={
                    "normalization_mode": "by_mean_intensity_single_die",
                    "eps": 1e-6,
                },
            )
            self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))
            self.assertTrue(
                any("<= eps" in str(w.message) for w in caught)
            )

    def test_invalid_normalization_mode_raises(self) -> None:
        """Verify ValueError is raised for unrecognized normalization_mode."""
        with self.assertRaises(ValueError) as ctx:
            normalize_spectral_data(
                self.ary_intensity,
                self.ary_energy,
                self.df_meta,
                config={"normalization_mode": "invalid_mode"},
            )
        self.assertIn("invalid_mode", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
