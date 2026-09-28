"""Unit tests for utility/pseudo_measurement.py and utility/visualization.py."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from utility.pseudo_measurement import (
    generate_pseudo_measurements,
    get_default_measurement_config,
    pseudo_voigt,
)
from utility.visualization import (
    plot_regional_spectra,
    plot_tool_comparison,
)


class TestPseudoMeasurement(unittest.TestCase):
    """Test suite for pseudo-measurement generation and XPS specifications."""

    def test_default_shapes_and_types(self) -> None:
        """Verify default configuration generates expected shapes and column schema."""
        ary_intensity, ary_energy, meta_df = generate_pseudo_measurements()

        # 20 measurements (10 for J4, 10 for J5) * 9 dies * 5 regions = 900 spectra
        expected_total = 20 * 9 * 5
        expected_points = 100

        self.assertEqual(ary_intensity.shape, (expected_total, expected_points))
        self.assertEqual(ary_energy.shape, (expected_total, expected_points))
        self.assertEqual(len(meta_df), expected_total)

        # Check metadata columns
        expected_cols = {
            "spectrum_index",
            "material",
            "tool",
            "measurement_id",
            "time",
            "die",
            "region",
            "n_points",
        }
        self.assertTrue(expected_cols.issubset(set(meta_df.columns)))
        self.assertFalse(meta_df.isnull().values.any())

    def test_binding_energy_identical_per_region_and_ascending(self) -> None:
        """Verify binding energy arrays for the same region are identical and ascending."""
        ary_intensity, ary_energy, meta_df = generate_pseudo_measurements()

        regions = meta_df["region"].unique()
        for region in regions:
            indices = meta_df[meta_df["region"] == region]["spectrum_index"].values
            first_be = ary_energy[indices[0]]

            # Monotonically ascending
            self.assertTrue(np.all(np.diff(first_be) > 0))

            # Exactly identical across all dies, measurements, and tools
            for idx in indices[1:]:
                np.testing.assert_array_equal(
                    ary_energy[idx],
                    first_be,
                    err_msg=f"Binding energy grid differs for region {region} at index {idx}",
                )

    def test_non_negative_intensities(self) -> None:
        """Verify intensity values are strictly non-negative."""
        ary_intensity, _, _ = generate_pseudo_measurements()
        self.assertTrue(np.all(ary_intensity >= 0.0))

    def test_custom_config_override(self) -> None:
        """Verify custom configurations correctly alter generation parameters."""
        custom_config = {
            "n_points": 50,
            "n_die": 3,
            "measurements_per_tool": {"J4": 2, "J5": 1},
            "regions": ["Al2p", "C1s"],
            "seed": 123,
        }
        ary_intensity, ary_energy, meta_df = generate_pseudo_measurements(custom_config)

        expected_total = (2 + 1) * 3 * 2  # 3 measurements * 3 dies * 2 regions = 18
        self.assertEqual(ary_intensity.shape, (expected_total, 50))
        self.assertEqual(ary_energy.shape, (expected_total, 50))
        self.assertEqual(len(meta_df), expected_total)
        self.assertListEqual(sorted(meta_df["region"].unique()), ["Al2p", "C1s"])

    def test_deterministic_reproducibility(self) -> None:
        """Verify same seed produces identical arrays and timestamps."""
        config = {"seed": 99, "measurements_per_tool": {"J4": 2, "J5": 2}}
        int1, ene1, meta1 = generate_pseudo_measurements(config)
        int2, ene2, meta2 = generate_pseudo_measurements(config)

        np.testing.assert_array_almost_equal(int1, int2)
        np.testing.assert_array_equal(ene1, ene2)
        pd.testing.assert_frame_equal(meta1, meta2)

    def test_tool_intensity_scale_effect(self) -> None:
        """Verify J5 intensity reflects the higher tool scale relative to J4."""
        config = {
            "measurements_per_tool": {"J4": 5, "J5": 5},
            "tool_offsets": {
                "J4": {"shift_ev": 0.0, "scale": 1.0},
                "J5": {"shift_ev": 0.0, "scale": 1.5},
            },
            "die_variation_std": 0.0,
            "noise_relative_std": 0.0,
            "seed": 42,
        }
        ary_intensity, _, meta_df = generate_pseudo_measurements(config)

        j4_idx = meta_df[
            (meta_df["tool"] == "J4") & (meta_df["region"] == "Al2p")
        ]["spectrum_index"].values[0]
        j5_idx = meta_df[
            (meta_df["tool"] == "J5") & (meta_df["region"] == "Al2p")
        ]["spectrum_index"].values[0]

        ratio = np.max(ary_intensity[j5_idx]) / np.max(ary_intensity[j4_idx])
        # Expected ratio should be close to 1.5
        self.assertAlmostEqual(ratio, 1.5, delta=0.05)

    def test_pseudo_voigt_invalid_fwhm_raises(self) -> None:
        """Verify pseudo_voigt rejects non-positive fwhm."""
        e = np.linspace(70, 78, 10)
        with self.assertRaises(ValueError):
            pseudo_voigt(e, {"center": 72.0, "amplitude": 10.0, "fwhm": 0.0})
        with self.assertRaises(ValueError):
            pseudo_voigt(e, {"center": 72.0, "amplitude": 10.0, "fwhm": -1.0})

    def test_negative_measurement_count_raises(self) -> None:
        """Verify validation catches negative measurement count."""
        with self.assertRaises(ValueError):
            generate_pseudo_measurements({"measurements_per_tool": {"J4": 5, "J5": -2}})

    def test_inverted_energy_range_raises(self) -> None:
        """Verify validation catches inverted energy range in region profiles."""
        custom_profiles = {
            "Al2p": {
                "energy_range": (78.0, 70.0),  # Inverted
                "baseline": 10.0,
                "peaks": [{"center": 74.0, "amplitude": 100.0, "fwhm": 1.2}],
            }
        }
        with self.assertRaises(ValueError):
            generate_pseudo_measurements({
                "regions": ["Al2p"],
                "region_profiles": custom_profiles,
            })

    def test_n_points_less_than_two_raises(self) -> None:
        """Verify invalid n_points raises ValueError."""
        with self.assertRaises(ValueError):
            generate_pseudo_measurements({"n_points": 1})
        with self.assertRaises(ValueError):
            generate_pseudo_measurements({"n_points": -5})

    def test_invalid_interval_range_raises(self) -> None:
        """Verify invalid interval range raises ValueError."""
        with self.assertRaises(ValueError):
            generate_pseudo_measurements({"interval_hours_range": (24.0, 4.0)})
        with self.assertRaises(ValueError):
            generate_pseudo_measurements({"interval_hours_range": (-1.0, 5.0)})


class TestVisualization(unittest.TestCase):
    """Test suite for plotting utilities."""

    @classmethod
    def setUpClass(cls) -> None:
        config = {
            "measurements_per_tool": {"J4": 1, "J5": 1},
            "n_die": 2,
            "n_points": 50,
            "seed": 42,
        }
        cls.ary_int, cls.ary_ene, cls.meta_df = generate_pseudo_measurements(config)

    def test_plot_regional_spectra(self) -> None:
        """Test plotting filtered regional spectra."""
        fig, ax = plot_regional_spectra(
            self.ary_ene,
            self.ary_int,
            self.meta_df,
            plot_config={"filters": {"tool": "J4", "region": "Al2p"}},
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        # 1 measurement * 2 dies = 2 lines
        self.assertEqual(len(ax.lines), 2)

    def test_plot_default_filters(self) -> None:
        """Test default plotting without explicit filter displays first session."""
        fig, ax = plot_regional_spectra(
            self.ary_ene,
            self.ary_int,
            self.meta_df,
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        # First measurement has 2 dies * 5 regions = 10 lines max
        self.assertTrue(len(ax.lines) <= 10)

    def test_plot_tool_comparison(self) -> None:
        """Test tool comparison overlay plot."""
        fig, ax = plot_tool_comparison(
            self.ary_ene,
            self.ary_int,
            self.meta_df,
            compare_config={"region": "Al2p", "die": 0},
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        # Should have 2 lines: J4 and J5
        self.assertEqual(len(ax.lines), 2)

    def test_plot_tool_comparison_insufficient_tools_raises(self) -> None:
        """Test comparison with fewer than 2 tools raises ValueError."""
        with self.assertRaises(ValueError):
            plot_tool_comparison(
                self.ary_ene,
                self.ary_int,
                self.meta_df,
                compare_config={"tools": ["J4"]},
            )

    def test_plot_empty_filter_raises(self) -> None:
        """Test filtering with non-existent criteria raises ValueError."""
        with self.assertRaises(ValueError):
            plot_regional_spectra(
                self.ary_ene,
                self.ary_int,
                self.meta_df,
                plot_config={"filters": {"tool": "NonExistentTool"}},
            )


if __name__ == "__main__":
    unittest.main()
