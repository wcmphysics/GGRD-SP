"""Unit tests for utility/quantification.py and Shirley background visualization."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from utility.pseudo_measurement import generate_pseudo_measurements
from utility.quantification import (
    DEFAULT_SCOFIELD_RSF,
    _find_ti2p_endpoints,
    calculate_atomic_percentages,
    calculate_shirley_background,
    get_default_quantification_config,
)
from utility.visualization import plot_shirley_background


class TestQuantification(unittest.TestCase):
    """Test suite for Shirley background subtraction and atomic percentage calculation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate pseudo-measurement data for testing."""
        config = {
            "measurements_per_tool": {"J4": 2, "J5": 2},
            "n_die": 3,
            "n_points": 60,
            "seed": 42,
        }
        cls.ary_int, cls.ary_ene, cls.meta_df = generate_pseudo_measurements(config)

    def test_shirley_flat_spectrum_zero_area(self) -> None:
        """Verify flat spectrum produces flat background and 0 net area."""
        energy = np.linspace(100, 110, 50)
        intensity = np.full(50, 25.0)

        bg, net_area = calculate_shirley_background(energy, intensity)
        np.testing.assert_allclose(bg, intensity, atol=1e-6)
        self.assertAlmostEqual(net_area, 0.0, places=4)

    def test_shirley_synthetic_peak(self) -> None:
        """Verify Shirley background on Gaussian peak step-like baseline."""
        energy = np.linspace(70, 80, 100)
        # Baseline = 10, peak height = 50 at 75 eV
        intensity = 10.0 + 50.0 * np.exp(-0.5 * ((energy - 75.0) / 0.8) ** 2)

        bg, net_area = calculate_shirley_background(energy, intensity)

        self.assertAlmostEqual(bg[0], 10.0, places=2)
        self.assertAlmostEqual(bg[-1], 10.0, places=2)
        self.assertGreater(net_area, 0.0)
        # Background should be <= intensity everywhere
        self.assertTrue(np.all(bg <= intensity + 1e-6))

    def test_ti2p_auto_endpoints(self) -> None:
        """Verify Ti2p auto-endpoints logic and B(E) == I(E) outside active [E1, E2]."""
        # Find a Ti2p spectrum in dataset
        ti2p_row = self.meta_df[self.meta_df["region"] == "Ti2p"].iloc[0]
        idx = int(ti2p_row["spectrum_index"])
        energy = self.ary_ene[idx]
        intensity = self.ary_int[idx]

        idx_1, idx_2 = _find_ti2p_endpoints(energy, intensity, smooth_search=False)
        self.assertTrue(0 <= idx_1 < idx_2 < len(energy))

        cfg = {"region": "Ti2p", "ti2p_auto_endpoints": True}
        bg, net_area = calculate_shirley_background(energy, intensity, cfg)

        self.assertGreater(net_area, 0.0)

        # Outside [idx_1, idx_2], background must equal intensity
        if idx_1 > 0:
            np.testing.assert_array_equal(bg[:idx_1], intensity[:idx_1])
        if idx_2 < len(energy) - 1:
            np.testing.assert_array_equal(bg[idx_2 + 1 :], intensity[idx_2 + 1 :])

    def test_ti2p_smooth_endpoints_search(self) -> None:
        """Verify smoothed minima search flag executes without error."""
        ti2p_row = self.meta_df[self.meta_df["region"] == "Ti2p"].iloc[0]
        idx = int(ti2p_row["spectrum_index"])
        energy = self.ary_ene[idx]
        intensity = self.ary_int[idx]

        idx_1, idx_2 = _find_ti2p_endpoints(energy, intensity, smooth_search=True)
        self.assertTrue(0 <= idx_1 < idx_2 < len(energy))

    def test_atomic_percentage_normalization_and_schema(self) -> None:
        """Verify atomic percentages sum to 100% and output adheres to wide format."""
        meas_id = "NMG_M_J4_00000"
        df_per_die, df_summary = calculate_atomic_percentages(
            self.ary_ene,
            self.ary_int,
            self.meta_df,
            config={"measurement_id": meas_id},
        )

        expected_at_cols = ["Al2p_at%", "Ti2p_at%", "O1s_at%", "C1s_at%", "Cl2p_at%"]
        for col in expected_at_cols:
            self.assertIn(col, df_per_die.columns)
            self.assertIn(col, df_summary.columns)

        # Wide format: 3 dies in test setup -> 3 rows in df_per_die
        self.assertEqual(len(df_per_die), 3)
        self.assertEqual(list(df_per_die["die"]), [0, 1, 2])

        # Sum of atomic percentages for every die must equal 100%
        die_sums = df_per_die[expected_at_cols].sum(axis=1)
        np.testing.assert_allclose(die_sums, 100.0, atol=0.01)

        # Summary check
        self.assertEqual(len(df_summary), 2)  # mean and std
        self.assertListEqual(list(df_summary["metric"]), ["mean", "std"])

    def test_predicted_spectrum_compatibility(self) -> None:
        """Verify compatibility with artificially wrapped predicted metadata and arrays."""
        # Wrap predicted data
        pred_meas_id = "NMG_P_J4J5_00001"
        ary_energy_predicted = self.ary_ene.copy()
        ary_intensity_predicted = self.ary_int.copy()

        # Update meta_df copy to simulate predicted metadata
        meta_df_predicted = self.meta_df.copy()
        meta_df_predicted["measurement_id"] = pred_meas_id
        meta_df_predicted = meta_df_predicted[meta_df_predicted["die"] == 0]

        df_per_die, df_summary = calculate_atomic_percentages(
            ary_energy_predicted,
            ary_intensity_predicted,
            meta_df_predicted,
            config={"measurement_id": pred_meas_id},
        )

        self.assertEqual(len(df_per_die), 1)
        self.assertEqual(df_per_die["measurement_id"].iloc[0], pred_meas_id)
        at_cols = [c for c in df_per_die.columns if c.endswith("_at%")]
        self.assertAlmostEqual(float(df_per_die[at_cols].sum(axis=1).iloc[0]), 100.0, places=2)

    def test_custom_rsf_override(self) -> None:
        """Verify modifying RSF changes calculated atomic percentage accordingly."""
        custom_rsf = dict(DEFAULT_SCOFIELD_RSF)
        # Double the RSF of Al2p -> should roughly halve Al2p atomic percentage
        custom_rsf["Al2p"] = DEFAULT_SCOFIELD_RSF["Al2p"] * 2.0

        df_default, _ = calculate_atomic_percentages(
            self.ary_ene,
            self.ary_int,
            self.meta_df,
            config={"measurement_id": "NMG_M_J4_00000"},
        )
        df_custom, _ = calculate_atomic_percentages(
            self.ary_ene,
            self.ary_int,
            self.meta_df,
            config={"measurement_id": "NMG_M_J4_00000", "rsf_dict": custom_rsf},
        )

        al_default = df_default["Al2p_at%"].iloc[0]
        al_custom = df_custom["Al2p_at%"].iloc[0]

        self.assertLess(al_custom, al_default)

    def test_length_mismatch_raises(self) -> None:
        """Verify length mismatch between energy and intensity raises ValueError."""
        with self.assertRaises(ValueError):
            calculate_shirley_background(np.array([1, 2, 3]), np.array([1, 2]))

    def test_descending_energy_grid(self) -> None:
        """Verify descending energy grid works properly and calculates positive net area."""
        energy_desc = np.linspace(80, 70, 50)  # Descending
        intensity = 10.0 + 30.0 * np.exp(-0.5 * ((energy_desc - 75.0) / 0.8) ** 2)

        bg, net_area = calculate_shirley_background(energy_desc, intensity)
        self.assertGreater(net_area, 0.0)
        self.assertEqual(len(bg), len(intensity))

    def test_non_positive_rsf_raises(self) -> None:
        """Verify non-positive RSF value raises ValueError."""
        with self.assertRaises(ValueError):
            calculate_atomic_percentages(
                self.ary_ene,
                self.ary_int,
                self.meta_df,
                config={"rsf_dict": {"Al2p": -1.0}},
            )

    def test_zero_area_all_peaks(self) -> None:
        """Verify flat/zero spectra safely produce 0% without NaN or ZeroDivisionError."""
        flat_int = np.full_like(self.ary_int, 10.0)
        df_per_die, df_summary = calculate_atomic_percentages(
            self.ary_ene,
            flat_int,
            self.meta_df,
            config={"measurement_id": "NMG_M_J4_00000"},
        )
        at_cols = [c for c in df_per_die.columns if c.endswith("_at%")]
        self.assertTrue((df_per_die[at_cols] == 0.0).all().all())
        self.assertFalse(df_per_die.isna().any().any())


class TestShirleyVisualization(unittest.TestCase):
    """Test suite for Shirley background plotting utilities."""

    @classmethod
    def setUpClass(cls) -> None:
        config = {
            "measurements_per_tool": {"J4": 1, "J5": 1},
            "n_die": 1,
            "n_points": 40,
            "seed": 42,
        }
        cls.ary_int, cls.ary_ene, cls.meta_df = generate_pseudo_measurements(config)

    def tearDown(self) -> None:
        """Close open matplotlib figures after each test."""
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_shirley_single_region(self) -> None:
        """Verify single region Shirley plot returns figure and axes."""
        fig, ax = plot_shirley_background(
            self.ary_ene,
            self.ary_int,
            self.meta_df,
            plot_config={"region": "Ti2p", "die": 0, "show": False},
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertEqual(len(ax.lines), 2)  # raw + background

    def test_plot_shirley_multi_region_grid(self) -> None:
        """Verify multi-panel grid plotting all regions returns figure and axes grid."""
        fig, axes = plot_shirley_background(
            self.ary_ene,
            self.ary_int,
            self.meta_df,
            plot_config={"die": 0, "show": False},
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(axes)


if __name__ == "__main__":
    unittest.main()
