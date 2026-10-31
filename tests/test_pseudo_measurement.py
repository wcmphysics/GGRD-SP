"""Unit tests for utility/pseudo_measurement.py and spectral visualization."""

from __future__ import annotations

import copy
import unittest
import numpy as np
import pandas as pd

from utility.pseudo_measurement import (
    DEFAULT_REGION_PROFILES,
    _calculate_shirley_step,
    _synthesize_regional_spectrum,
    generate_pseudo_measurements,
    pseudo_voigt,
)
from visualization import (
    plot_regional_spectra,
    plot_tool_comparison,
)


class TestPseudoMeasurementGeneration(unittest.TestCase):
    """Test suite for pseudo-measurement data generation, physics profiles, and invariants."""

    def test_default_shapes_and_invariants(self) -> None:
        """Verify default configuration generates expected shapes, non-negative intensities, and ascending energy."""
        ary_intensity, ary_energy, meta_df = generate_pseudo_measurements()

        # 20 measurements (10 for J4, 10 for J5) * 9 dies * 5 regions = 900 spectra
        expected_total = 20 * 9 * 5
        self.assertEqual(ary_intensity.shape, (expected_total, 100))
        self.assertEqual(ary_energy.shape, (expected_total, 100))
        self.assertEqual(len(meta_df), expected_total)
        self.assertTrue(np.all(ary_intensity >= 0.0))

        # Check ascending energy grid identical across regions
        for region in meta_df["region"].unique():
            indices = meta_df[meta_df["region"] == region]["spectrum_index"].values
            first_be = ary_energy[indices[0]]
            self.assertTrue(np.all(np.diff(first_be) > 0))
            for idx in indices[1:]:
                np.testing.assert_array_equal(ary_energy[idx], first_be)

    def test_tool_offsets_decay_and_die_zones(self) -> None:
        """Verify tool scale ratios, time decay / maintenance jumps, and die illumination hierarchy."""
        config = {
            "measurements_per_tool": {"J4": 15, "J5": 15},
            "interval_hours_range": (36.0, 48.0),
            "maintenance_interval_days": 15.0,
            "die_repeatability_std": 0.0,
            "seed": 42,
        }
        ary_i, _, meta_df = generate_pseudo_measurements(config)

        # Scale tracking columns exist and satisfy product
        computed = meta_df["scale_tool"] * meta_df["scale_time"] * meta_df["scale_die"]
        np.testing.assert_allclose(meta_df["scale_total"], computed, rtol=1e-5)

        # Die zone hierarchy: Center (Die 0) > Mid (Die 5-8) > Edge (Die 1-4)
        j4_dies = meta_df[meta_df["tool"] == "J4"].drop_duplicates(subset=["die"]).set_index("die")
        center_s = j4_dies.loc[0, "scale_die"]
        mid_s = [j4_dies.loc[d, "scale_die"] for d in (5, 6, 7, 8)]
        edge_s = [j4_dies.loc[d, "scale_die"] for d in (1, 2, 3, 4)]
        self.assertTrue(all(center_s > s for s in mid_s))
        self.assertGreater(min(mid_s), max(edge_s))

    def test_region_relative_strengths_and_t7_distribution(self) -> None:
        """Verify regional peak strength proportionality and t7_code grouping."""
        strengths = {"Al2p": 5.0, "Ti2p": 7.0, "O1s": 10.0, "C1s": 8.0, "Cl2p": 1.0}
        cfg = {
            "regions": ["Al2p", "Ti2p", "O1s", "C1s", "Cl2p"],
            "region_relative_strengths": strengths,
            "measurements_per_tool": {"J4": 20, "J5": 20},
            "measurements_per_t7_code": 10,
            "die_variation_std": 0.0,
            "noise_relative_std": 0.0,
            "shirley_background": {"enabled": False},
            "seed": 42,
        }
        ary_i, _, meta_df = generate_pseudo_measurements(cfg)

        # Proportions check
        die0 = meta_df[(meta_df["measurement_id"] == "NMG_M_J4_00000") & (meta_df["die"] == 0)]
        max_vals = {row["region"]: float(np.max(ary_i[int(row["spectrum_index"])])) for _, row in die0.iterrows()}
        self.assertAlmostEqual(max_vals["Ti2p"] / max_vals["Cl2p"], 7.0, places=4)
        self.assertAlmostEqual(max_vals["O1s"] / max_vals["Cl2p"], 10.0, places=4)

        # t7_code sharing across tools (10 per wafer)
        j4_t7 = meta_df[meta_df["tool"] == "J4"].drop_duplicates(subset=["measurement_id"])["t7_code"].tolist()
        self.assertEqual(j4_t7, ["T7_001"] * 10 + ["T7_002"] * 10)

    def test_shirley_background_elevates_high_be_baseline(self) -> None:
        """Verify Shirley background elevates high-BE baseline relative to low-BE."""
        step_ratio = 0.15
        cfg = {
            "measurements_per_tool": {"J4": 1},
            "n_die": 1,
            "regions": ["Ti2p"],
            "noise_relative_std": 0.0,
            "shirley_background": {"enabled": True, "step_ratio": step_ratio},
            "seed": 42,
        }
        ary_i, _, _ = generate_pseudo_measurements(cfg)
        spectrum = ary_i[0]
        # At index 0 (lowest BE), cum_area is 0, so Shirley step is 0.
        # At index -1 (highest BE), cum_area reaches total_area,
        # giving the full step.
        low_be_intensity = spectrum[0]
        high_be_intensity = spectrum[-1]
        self.assertGreater(high_be_intensity, low_be_intensity)
        self.assertGreater(high_be_intensity - low_be_intensity, 0.0)

    def test_shirley_background_monotonic_step(self) -> None:
        """Verify the isolated Shirley background step is monotonically non-decreasing."""
        cfg_enabled = {
            "measurements_per_tool": {"J4": 1},
            "n_die": 1,
            "regions": ["Ti2p"],
            "noise_relative_std": 0.0,
            "shirley_background": {"enabled": True, "step_ratio": 0.10},
            "seed": 42,
        }
        cfg_disabled = {
            "measurements_per_tool": {"J4": 1},
            "n_die": 1,
            "regions": ["Ti2p"],
            "noise_relative_std": 0.0,
            "shirley_background": {"enabled": False},
            "seed": 42,
        }
        ary_enabled, _, _ = generate_pseudo_measurements(cfg_enabled)
        ary_disabled, _, _ = generate_pseudo_measurements(cfg_disabled)

        # Difference isolates the cumulative Shirley step
        shirley_step = ary_enabled[0] - ary_disabled[0]
        step_diffs = np.diff(shirley_step)
        self.assertTrue(np.all(step_diffs >= -1e-12))
        self.assertGreater(shirley_step[-1] - shirley_step[0], 0.0)

    def test_shirley_background_disabled_and_zero_ratio(self) -> None:
        """Verify enabled: false and step_ratio: 0.0 produce identical spectra."""
        cfg_false = {
            "measurements_per_tool": {"J4": 1},
            "n_die": 1,
            "noise_relative_std": 0.0,
            "shirley_background": {"enabled": False, "step_ratio": 0.10},
            "seed": 42,
        }
        cfg_zero = {
            "measurements_per_tool": {"J4": 1},
            "n_die": 1,
            "noise_relative_std": 0.0,
            "shirley_background": {"enabled": True, "step_ratio": 0.0},
            "seed": 42,
        }
        ary_false, _, _ = generate_pseudo_measurements(cfg_false)
        ary_zero, _, _ = generate_pseudo_measurements(cfg_zero)
        np.testing.assert_allclose(ary_false, ary_zero, atol=1e-12)

    def test_per_region_shirley_step_ratio_override(self) -> None:
        """Verify per-region shirley_step_ratio overrides the global setting."""
        custom_profiles = copy.deepcopy(DEFAULT_REGION_PROFILES)
        custom_profiles["Ti2p"]["shirley_step_ratio"] = 0.0
        custom_profiles["Al2p"]["shirley_step_ratio"] = 0.25

        cfg = {
            "measurements_per_tool": {"J4": 1},
            "n_die": 1,
            "regions": ["Al2p", "Ti2p", "Cl2p"],
            "region_profiles": custom_profiles,
            "noise_relative_std": 0.0,
            "shirley_background": {"enabled": True, "step_ratio": 0.10},
            "seed": 42,
        }
        ary_i, _, meta_df = generate_pseudo_measurements(cfg)

        cfg_dis = {
            "measurements_per_tool": {"J4": 1},
            "n_die": 1,
            "regions": ["Al2p", "Ti2p", "Cl2p"],
            "region_profiles": custom_profiles,
            "noise_relative_std": 0.0,
            "shirley_background": {"enabled": False},
            "seed": 42,
        }
        ary_dis, _, _ = generate_pseudo_measurements(cfg_dis)

        for i, row in meta_df.iterrows():
            region = row["region"]
            step = ary_i[i] - ary_dis[i]
            if region == "Ti2p":
                np.testing.assert_allclose(step, 0.0, atol=1e-12)
            elif region == "Al2p":
                self.assertGreater(step[-1] - step[0], 0.0)
            elif region == "Cl2p":
                self.assertGreater(step[-1] - step[0], 0.0)

    def test_shirley_background_descending_energy_grid(self) -> None:
        """Verify Shirley background correctly elevates high BE when energy is descending."""
        e_asc = np.linspace(50.0, 60.0, 100)
        e_desc = e_asc[::-1]
        profile = {
            "peaks": [{"center": 55.0, "amplitude": 100.0, "fwhm": 1.2, "eta": 0.3}],
            "baseline": 0.0,
        }
        params = {
            "tool_shift": 0.0,
            "total_scale": 1.0,
            "noise_relative_std": 0.0,
            "shirley_background": {"enabled": True, "step_ratio": 0.10},
        }
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        spec_asc = _synthesize_regional_spectrum(e_asc, profile, params, rng1)
        spec_desc = _synthesize_regional_spectrum(e_desc, profile, params, rng2)
        np.testing.assert_allclose(spec_desc, spec_asc[::-1], atol=1e-12)
        self.assertGreater(spec_desc[0] - spec_desc[-1], 0.0)

    def test_shirley_background_zero_peak_and_small_grid(self) -> None:
        """Verify Shirley background step safely returns zero for degenerate inputs."""
        e_grid = np.linspace(50.0, 60.0, 100)
        zero_peak = np.zeros_like(e_grid)
        step_zero = _calculate_shirley_step(e_grid, zero_peak, 0.10)
        np.testing.assert_allclose(step_zero, 0.0)

        step_single = _calculate_shirley_step(
            np.array([50.0]), np.array([10.0]), 0.10
        )
        np.testing.assert_allclose(step_single, 0.0)

    def test_per_region_shirley_enabled_override(self) -> None:
        """Verify regional shirley_enabled: False disables background for that region."""
        custom_profiles = copy.deepcopy(DEFAULT_REGION_PROFILES)
        custom_profiles["Ti2p"]["shirley_enabled"] = False

        cfg = {
            "measurements_per_tool": {"J4": 1},
            "n_die": 1,
            "regions": ["Ti2p", "Cl2p"],
            "region_profiles": custom_profiles,
            "noise_relative_std": 0.0,
            "shirley_background": {"enabled": True, "step_ratio": 0.10},
            "seed": 42,
        }
        ary_i, _, meta_df = generate_pseudo_measurements(cfg)

        cfg_dis = {
            "measurements_per_tool": {"J4": 1},
            "n_die": 1,
            "regions": ["Ti2p", "Cl2p"],
            "region_profiles": custom_profiles,
            "noise_relative_std": 0.0,
            "shirley_background": {"enabled": False},
            "seed": 42,
        }
        ary_dis, _, _ = generate_pseudo_measurements(cfg_dis)

        ti_idx = meta_df[meta_df["region"] == "Ti2p"]["spectrum_index"].values[0]
        cl_idx = meta_df[meta_df["region"] == "Cl2p"]["spectrum_index"].values[0]

        np.testing.assert_allclose(ary_i[ti_idx], ary_dis[ti_idx], atol=1e-12)
        self.assertGreater(ary_i[cl_idx][-1] - ary_dis[cl_idx][-1], 0.0)

    def test_defensive_error_handling(self) -> None:
        """Table-driven test for invalid generator configuration parameters."""
        error_configs = [
            ("negative measurements", {"measurements_per_tool": {"J4": -1}}),
            ("invalid n_points", {"n_points": 1}),
            ("invalid intervals", {"interval_hours_range": (24.0, 4.0)}),
            ("non-positive t7 per code", {"measurements_per_t7_code": 0}),
            ("missing relative strengths", {"region_relative_strengths": None}),
            ("negative relative strength", {"region_relative_strengths": {"Ti2p": -1.0}}),
            ("negative shirley step ratio", {"shirley_background": {"step_ratio": -0.1}}),
            ("excessive shirley step ratio", {"shirley_background": {"step_ratio": 1.5}}),
            ("invalid shirley enabled type", {"shirley_background": {"enabled": "yes"}}),
            ("invalid shirley type", {"shirley_background": 123}),
            (
                "negative regional shirley step",
                {"region_profiles": {"Ti2p": {"shirley_step_ratio": -0.05}}},
            ),
            (
                "invalid regional shirley_enabled",
                {"region_profiles": {"Ti2p": {"shirley_enabled": "not_bool"}}},
            ),
        ]
        for name, cfg in error_configs:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    generate_pseudo_measurements(cfg)

        with self.assertRaises(ValueError):
            pseudo_voigt(np.linspace(70, 78, 10), {"center": 72.0, "amplitude": 10.0, "fwhm": 0.0})


class TestSpectralVisualization(unittest.TestCase):
    """Test suite for pseudo-measurement plotting and tool comparison."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate pseudo-measurement data for visualization."""
        config = {
            "measurements_per_tool": {"J4": 1, "J5": 1},
            "n_die": 2,
            "n_points": 50,
            "seed": 42,
        }
        cls.ary_int, cls.ary_ene, cls.meta_df = generate_pseudo_measurements(config)

    def tearDown(self) -> None:
        """Close open matplotlib figures."""
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_regional_spectra_and_tool_comparison(self) -> None:
        """Verify regional spectra and tool comparison plots render correct line counts."""
        fig_r, ax_r = plot_regional_spectra(
            self.ary_ene,
            self.ary_int,
            self.meta_df,
            plot_config={"filters": {"tool": "J4", "region": "Al2p"}},
        )
        self.assertEqual(len(ax_r.lines), 2)

        fig_t, ax_t = plot_tool_comparison(
            self.ary_ene,
            self.ary_int,
            self.meta_df,
            compare_config={"region": "Al2p", "die": 0},
        )
        self.assertEqual(len(ax_t.lines), 2)

        # Defensive errors on invalid tools or empty filters
        with self.assertRaises(ValueError):
            plot_tool_comparison(self.ary_ene, self.ary_int, self.meta_df, compare_config={"tools": ["J4"]})
        with self.assertRaises(ValueError):
            plot_regional_spectra(self.ary_ene, self.ary_int, self.meta_df, plot_config={"filters": {"tool": "BAD"}})


if __name__ == "__main__":
    unittest.main()
