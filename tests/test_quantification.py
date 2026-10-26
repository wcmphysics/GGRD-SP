"""Unit tests for utility/quantification.py and Shirley background subtraction."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from utility.pseudo_measurement import generate_pseudo_measurements
from utility.flux_and_intensity import compute_die_total_flux
from utility.quantification import (
    DEFAULT_SCOFIELD_RSF,
    _find_minima_endpoints,
    calculate_atomic_percentage_split_statistics,
    calculate_atomic_percentages,
    calculate_shirley_background,
    determine_shirley_endpoints,
    format_side_by_side_atomic_percentages,
    get_default_quantification_config,
    load_rsf_table,
)
from visualization import (
    plot_atomic_percentage_distributions,
    plot_atomic_percentage_mae,
    plot_shirley_background,
)


class TestShirleyBackground(unittest.TestCase):
    """Test suite for Shirley background algorithms and endpoint resolution strategies."""

    def setUp(self) -> None:
        """Create standard synthetic regional spectrum with single peak."""
        self.energy = np.linspace(70.0, 80.0, 101)  # 0.1 eV step
        self.intensity = 20.0 + 80.0 * np.exp(-0.5 * ((self.energy - 75.0) / 0.8) ** 2)

    def test_shirley_module_exports_and_shape_invariants(self) -> None:
        """Verify Shirley routines export properly and handle flat / inverted grids."""
        from utility.shirley import (
            calculate_shirley_background as calc_bg,
            determine_shirley_endpoints as det_ep,
        )
        from utility import (
            calculate_shirley_background as u_calc_bg,
            determine_shirley_endpoints as u_det_ep,
        )

        self.assertIs(calc_bg, u_calc_bg)
        self.assertIs(det_ep, u_det_ep)

        # Flat spectrum produces zero net area
        energy_flat = np.linspace(100, 110, 50)
        intensity_flat = np.full(50, 25.0)
        bg_flat, area_flat = calculate_shirley_background(energy_flat, intensity_flat)
        np.testing.assert_allclose(bg_flat, intensity_flat, atol=1e-6)
        self.assertAlmostEqual(area_flat, 0.0, places=4)

        # Descending energy grid works identically
        energy_desc = np.linspace(80, 70, 50)
        intensity_desc = 10.0 + 30.0 * np.exp(-0.5 * ((energy_desc - 75.0) / 0.8) ** 2)
        bg_desc, area_desc = calculate_shirley_background(energy_desc, intensity_desc)
        self.assertGreater(area_desc, 0.0)
        self.assertEqual(len(bg_desc), len(intensity_desc))

        # Length mismatch raises ValueError
        with self.assertRaises(ValueError):
            calculate_shirley_background(np.array([1, 2, 3]), np.array([1, 2]))

    def test_shirley_endpoint_determination_strategies(self) -> None:
        """Verify edge, minima, and direct endpoint resolution with averaging."""
        # 1. Edge strategy
        idx_1, idx_2, i_1, i_2 = determine_shirley_endpoints(
            self.energy, self.intensity, config={"strategy": "edge", "average_width_ev": 0.0}
        )
        self.assertEqual((idx_1, idx_2), (0, len(self.intensity) - 1))
        self.assertAlmostEqual(i_1, self.intensity[0], places=5)
        self.assertAlmostEqual(i_2, self.intensity[-1], places=5)

        # 2. Minima strategy with shoulders
        int_shoulders = self.intensity.copy()
        int_shoulders[:10] += 30.0
        int_shoulders[-10:] += 30.0
        idx_m1, idx_m2, _, _ = determine_shirley_endpoints(
            self.energy, int_shoulders, config={"strategy": "minima", "average_width_ev": 0.0}
        )
        self.assertTrue(0 < idx_m1 < idx_m2 < len(self.intensity) - 1)

        # 3. Direct strategy with endpoints
        idx_d1, idx_d2, i_d1, i_d2 = determine_shirley_endpoints(
            self.energy,
            self.intensity,
            config={"strategy": "direct", "energy_endpoints": (72.0, 78.0), "average_width_ev": 0.0},
        )
        expected_d1 = int(np.argmin(np.abs(self.energy - 72.0)))
        expected_d2 = int(np.argmin(np.abs(self.energy - 78.0)))
        self.assertEqual((idx_d1, idx_d2), (expected_d1, expected_d2))
        self.assertAlmostEqual(i_d1, self.intensity[expected_d1], places=5)
        self.assertAlmostEqual(i_d2, self.intensity[expected_d2], places=5)

        # Reversed order produces identical indices
        idx_r1, idx_r2, _, _ = determine_shirley_endpoints(
            self.energy,
            self.intensity,
            config={"strategy": "direct", "energy_endpoints": (78.0, 72.0), "average_width_ev": 0.0},
        )
        self.assertEqual((idx_d1, idx_d2), (idx_r1, idx_r2))

    def test_shirley_averaging_and_mirror_padding(self) -> None:
        """Verify mirror reflection padding and smoothing around boundary noise."""
        noisy_int = np.full(101, 100.0)
        noisy_int[0] = 50.0  # Noise dip at boundary
        noisy_int[1] = 100.0

        # Without averaging
        _, _, raw_i1, _ = determine_shirley_endpoints(
            self.energy, noisy_int, config={"strategy": "edge", "average_width_ev": 0.0}
        )
        self.assertEqual(raw_i1, 50.0)

        # With averaging
        _, _, avg_i1, _ = determine_shirley_endpoints(
            self.energy, noisy_int, config={"strategy": "edge", "average_width_ev": 1.0}
        )
        self.assertGreater(avg_i1, 90.0)
        self.assertLess(avg_i1, 100.0)

        # Background calculation with B(E) <= I(E)
        bg, net_area = calculate_shirley_background(
            self.energy, self.intensity, config={"strategy": "minima", "average_width_ev": 1.0}
        )
        self.assertTrue(np.all(bg <= self.intensity + 1e-6))
        self.assertGreater(net_area, 0.0)

    def test_shirley_endpoint_error_handling(self) -> None:
        """Table-driven test for invalid endpoint strategies and parameters."""
        error_cases = [
            ("invalid strategy", {"strategy": "INVALID"}, ValueError),
            ("missing direct endpoints", {"strategy": "direct"}, ValueError),
            ("malformed direct tuple", {"strategy": "direct", "energy_endpoints": (72.0,)}, ValueError),
            ("degenerate direct range", {"strategy": "direct", "energy_endpoints": (75.0, 75.0)}, ValueError),
            ("out of bounds direct points", {"strategy": "direct", "energy_endpoints": (10.0, 20.0)}, ValueError),
            ("non-finite points", {"strategy": "direct", "energy_endpoints": (float("nan"), 75.0)}, ValueError),
        ]
        for name, cfg, expected_error in error_cases:
            with self.subTest(case=name):
                with self.assertRaises(expected_error):
                    determine_shirley_endpoints(self.energy, self.intensity, config=cfg)


class TestAtomicPercentageQuantification(unittest.TestCase):
    """Test suite for atomic percentage quantification, RSF loading, and split statistics."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate pseudo-measurement data for quantification tests."""
        config = {
            "measurements_per_tool": {"J4": 2, "J5": 2},
            "n_die": 3,
            "n_points": 60,
            "seed": 42,
        }
        cls.ary_int, cls.ary_ene, cls.meta_df = generate_pseudo_measurements(config)

    def test_calculate_atomic_percentages_normalization_and_overrides(self) -> None:
        """Verify atomic percentages sum to 100%, respect custom RSF, and handle zero area."""
        meas_id = "NMG_M_J4_00000"
        df_per_die, df_summary = calculate_atomic_percentages(
            self.ary_ene,
            self.ary_int,
            self.meta_df,
            config={"measurement_id": meas_id},
        )
        at_cols = [c for c in df_per_die.columns if c.endswith("_atomic_percentage")]
        self.assertEqual(len(df_per_die), 3)
        np.testing.assert_allclose(df_per_die[at_cols].sum(axis=1), 100.0, atol=0.01)
        self.assertEqual(list(df_summary["metric"]), ["mean", "std"])

        # RSF override alters relative atomic percentages
        custom_rsf = dict(DEFAULT_SCOFIELD_RSF)
        custom_rsf["Al2p"] = DEFAULT_SCOFIELD_RSF["Al2p"] * 2.0
        df_custom, _ = calculate_atomic_percentages(
            self.ary_ene,
            self.ary_int,
            self.meta_df,
            config={"measurement_id": meas_id, "rsf_table": custom_rsf},
        )
        self.assertLess(
            df_custom["Al2p_atomic_percentage"].iloc[0],
            df_per_die["Al2p_atomic_percentage"].iloc[0],
        )

        # Flat / zero area spectra safely yield 0.0% without NaNs
        flat_int = np.full_like(self.ary_int, 10.0)
        df_flat, _ = calculate_atomic_percentages(
            self.ary_ene,
            flat_int,
            self.meta_df,
            config={"measurement_id": meas_id},
        )
        self.assertTrue((df_flat[at_cols] == 0.0).all().all())
        self.assertFalse(df_flat.isna().any().any())

    def test_quantification_defensive_errors_and_rsf_loading(self) -> None:
        """Verify error handling on invalid RSF and verify RSF table loading."""
        # Non-positive RSF
        with self.assertRaises(ValueError):
            calculate_atomic_percentages(
                self.ary_ene, self.ary_int, self.meta_df, config={"rsf_table": {"Al2p": -1.0}}
            )
        # Missing RSF table
        with self.assertRaises(KeyError):
            calculate_atomic_percentages(
                self.ary_ene, self.ary_int, self.meta_df, config={"rsf_table": None, "measurement_id": "M1"}
            )

        # load_rsf_table checks
        rsf_table = load_rsf_table()
        self.assertIn("Ti2p", rsf_table)
        loaded = load_rsf_table({"quantification": {"rsf_table": {"C1s": 1.23}}})
        self.assertEqual(loaded["C1s"], 1.23)
        with self.assertRaises(FileNotFoundError):
            load_rsf_table("nonexistent_path_123.yaml")

    def test_split_statistics_and_side_by_side_reporting(self) -> None:
        """Verify split-level statistical aggregation and side-by-side formatting."""
        meta_orig = self.meta_df.copy()
        meta_orig.loc[meta_orig["measurement_id"] == "NMG_M_J4_00000", "target_measurement_id"] = "NMG_M_J5_00000"
        meta_orig.loc[meta_orig["measurement_id"] == "NMG_M_J4_00001", "target_measurement_id"] = "NMG_M_J5_00001"

        meta_pred = meta_orig[meta_orig["tool"] == "J4"].copy()
        meta_pred["source_measurement_id"] = meta_pred["measurement_id"]
        meta_pred["measurement_id"] = meta_pred["measurement_id"].str.replace("_M_J4_", "_P_J4J5_")
        meta_pred["split"] = np.where(meta_pred["source_measurement_id"] == "NMG_M_J4_00000", "train", "test")
        meta_pred["spectrum_index"] = np.arange(len(meta_pred))

        ary_pred = self.ary_int[:len(meta_pred)] * 1.05
        df_samples, df_summary = calculate_atomic_percentage_split_statistics(
            (self.ary_int, self.ary_ene, meta_orig),
            (ary_pred, self.ary_ene[:len(meta_pred)], meta_pred),
            config={"splits": ["train", "test"]},
        )
        self.assertFalse(df_samples.empty)
        self.assertIn("mae", df_summary.columns)

        # Formatted string mode
        df_side = format_side_by_side_atomic_percentages(df_summary, config={"splits": ["train", "test"]})
        self.assertIn("Overall (MAE)", df_side["element"].values)
        self.assertTrue(df_side["mae_train"].str.contains("±").all())

        # Unformatted numerical mode
        df_side_num = format_side_by_side_atomic_percentages(
            df_summary, config={"splits": ["train", "test"], "format_str": False}
        )
        self.assertIn("mae_train", df_side_num.columns)
        self.assertIsInstance(df_side_num["mae_train"].iloc[0], (float, np.floating))

    def test_compute_die_total_flux(self) -> None:
        """Verify compute_die_total_flux integration across regions."""
        n_points = 10
        ary_int = np.ones((4, n_points), dtype=np.float32) * 5.0
        ary_ene = np.tile(np.linspace(10.0, 20.0, n_points, dtype=np.float32), (4, 1))
        df_meta_sample = pd.DataFrame([
            {"spectrum_index": 0, "measurement_id": "M1", "die": 0, "region": "Al2p"},
            {"spectrum_index": 1, "measurement_id": "M1", "die": 0, "region": "Ti2p"},
            {"spectrum_index": 2, "measurement_id": "M1", "die": 1, "region": "Al2p"},
            {"spectrum_index": 3, "measurement_id": "M1", "die": 1, "region": "Ti2p"},
        ])
        flux_map = compute_die_total_flux(df_meta_sample, ary_int, ary_ene)
        self.assertAlmostEqual(flux_map[("M1", 0)], 100.0, places=2)
        self.assertAlmostEqual(flux_map[("M1", 1)], 100.0, places=2)


class TestQuantificationVisualization(unittest.TestCase):
    """Test suite for Shirley overlays, atomic-% distributions, and MAE visualizations."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate test spectral and atomic percentage sample tables."""
        config = {
            "measurements_per_tool": {"J4": 1, "J5": 1},
            "n_die": 1,
            "n_points": 40,
            "seed": 42,
        }
        cls.ary_int, cls.ary_ene, cls.meta_df = generate_pseudo_measurements(config)
        cls.df_samples = pd.DataFrame([
            {
                "split": "train",
                "element": "Al",
                "die": 0,
                "target_atomic_percentage": 30.5,
                "predicted_atomic_percentage": 31.0,
                "diff_atomic_percentage": 0.5,
                "abs_diff_atomic_percentage": 0.5,
            },
            {
                "split": "test",
                "element": "Al",
                "die": 0,
                "target_atomic_percentage": 30.0,
                "predicted_atomic_percentage": 30.2,
                "diff_atomic_percentage": 0.2,
                "abs_diff_atomic_percentage": 0.2,
            },
            {
                "split": "train",
                "element": "Ti",
                "die": 0,
                "target_atomic_percentage": 69.5,
                "predicted_atomic_percentage": 69.0,
                "diff_atomic_percentage": -0.5,
                "abs_diff_atomic_percentage": 0.5,
            },
        ])

    def tearDown(self) -> None:
        """Close open matplotlib figures."""
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_shirley_background(self) -> None:
        """Verify single region and multi-region grid Shirley plots render lines."""
        fig_single, ax_single = plot_shirley_background(
            self.ary_ene, self.ary_int, self.meta_df, plot_config={"region": "Ti2p", "die": 0, "show": False}
        )
        self.assertEqual(len(ax_single.lines), 2)

        fig_multi, axes_multi = plot_shirley_background(
            self.ary_ene, self.ary_int, self.meta_df, plot_config={"die": 0, "show": False}
        )
        self.assertIsNotNone(axes_multi)

    def test_plot_atomic_percentage_distributions_and_mae(self) -> None:
        """Verify distribution and MAE plots generate valid figure panels."""
        fig_dist, (ax_abs, ax_diff) = plot_atomic_percentage_distributions(
            self.df_samples, plot_config={"show": False}
        )
        self.assertIn("Atomic % Distribution", ax_abs.get_title())
        self.assertIn("Difference Distribution", ax_diff.get_title())

        fig_mae, (ax_strip, ax_ecdf) = plot_atomic_percentage_mae(
            self.df_samples, plot_config={"show": False}
        )
        self.assertIn("Sample Absolute Errors", ax_strip.get_title())
        self.assertIn("Empirical Cumulative Error", ax_ecdf.get_title())

    def test_visualization_defensive_error_handling(self) -> None:
        """Table-driven test for empty and invalid DataFrames in visualization routines."""
        invalid_dfs = [
            ("empty dataframe", pd.DataFrame(), ValueError),
            ("missing columns", pd.DataFrame({"split": ["train"]}), KeyError),
        ]
        for name, df, expected_err in invalid_dfs:
            with self.subTest(routine="distributions", case=name):
                with self.assertRaises(expected_err):
                    plot_atomic_percentage_distributions(df)
            with self.subTest(routine="mae", case=name):
                with self.assertRaises(expected_err):
                    plot_atomic_percentage_mae(df)


if __name__ == "__main__":
    unittest.main()
