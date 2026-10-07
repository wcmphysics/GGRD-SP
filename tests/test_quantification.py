"""Unit tests for utility/quantification.py and Shirley background visualization."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from utility.pseudo_measurement import generate_pseudo_measurements
from utility.quantification import (
    DEFAULT_SCOFIELD_RSF,
    _find_minima_endpoints,
    calculate_atomic_percentage_split_statistics,
    calculate_atomic_percentages,
    calculate_shirley_background,
    determine_shirley_endpoints,
    format_side_by_side_atomic_percentages,
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

    def test_minima_endpoints_and_inactive_bounds(self) -> None:
        """Verify minima endpoints logic and B(E) == I(E) outside active [E1, E2]."""
        # Find a Ti2p spectrum in dataset
        ti2p_row = self.meta_df[self.meta_df["region"] == "Ti2p"].iloc[0]
        idx = int(ti2p_row["spectrum_index"])
        energy = self.ary_ene[idx]
        intensity = self.ary_int[idx]

        idx_1, idx_2 = _find_minima_endpoints(energy, intensity, smooth_search=False)
        self.assertTrue(0 <= idx_1 < idx_2 < len(energy))

        cfg = {"region": "Ti2p", "strategy": "minima"}
        bg, net_area = calculate_shirley_background(energy, intensity, cfg)

        self.assertGreater(net_area, 0.0)

        # Outside [idx_1, idx_2], background must equal intensity
        if idx_1 > 0:
            np.testing.assert_array_equal(bg[:idx_1], intensity[:idx_1])
        if idx_2 < len(energy) - 1:
            np.testing.assert_array_equal(bg[idx_2 + 1 :], intensity[idx_2 + 1 :])

    def test_smooth_endpoints_search(self) -> None:
        """Verify smoothed minima search flag executes without error."""
        ti2p_row = self.meta_df[self.meta_df["region"] == "Ti2p"].iloc[0]
        idx = int(ti2p_row["spectrum_index"])
        energy = self.ary_ene[idx]
        intensity = self.ary_int[idx]

        idx_1, idx_2 = _find_minima_endpoints(energy, intensity, smooth_search=True)
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


class TestEndpointStrategiesAndAveraging(unittest.TestCase):
    """Test suite for systematic Shirley endpoint determination strategies and mirror-padded averaging."""

    def setUp(self) -> None:
        # Create synthetic regional spectrum with 2 peaks and known minima
        self.energy = np.linspace(70.0, 80.0, 101)  # 0.1 eV step
        # Baseline = 20.0, main peak at 75 eV
        self.intensity = 20.0 + 80.0 * np.exp(-0.5 * ((self.energy - 75.0) / 0.8) ** 2)

    def test_strategy_edge_selects_boundaries(self) -> None:
        """Strategy 'edge' must select index 0 and index N - 1."""
        idx_1, idx_2, i_1, i_2 = determine_shirley_endpoints(
            self.energy,
            self.intensity,
            config={"strategy": "edge", "average_width_ev": 0.0},
        )
        self.assertEqual(idx_1, 0)
        self.assertEqual(idx_2, len(self.intensity) - 1)
        self.assertAlmostEqual(i_1, self.intensity[0], places=5)
        self.assertAlmostEqual(i_2, self.intensity[-1], places=5)

    def test_strategy_minima_directional_search(self) -> None:
        """Strategy 'minima' searches for directional minima away from peak maximum."""
        # Add side shoulders so minima are not at the boundaries
        int_with_shoulders = self.intensity.copy()
        int_with_shoulders[:10] += 30.0  # High at low energy edge
        int_with_shoulders[-10:] += 30.0  # High at high energy edge

        idx_1, idx_2, i_1, i_2 = determine_shirley_endpoints(
            self.energy,
            int_with_shoulders,
            config={"strategy": "minima", "average_width_ev": 0.0},
        )
        # Minima should be away from the edges
        self.assertGreater(idx_1, 0)
        self.assertLess(idx_2, len(self.intensity) - 1)
        self.assertLess(idx_1, idx_2)

    def test_default_strategy_is_minima_and_default_average_is_1ev(self) -> None:
        """Default configuration must use strategy='minima' and average_width_ev=1.0."""
        default_cfg = get_default_quantification_config()
        self.assertEqual(default_cfg["default_endpoint_strategy"], "minima")
        self.assertEqual(default_cfg["default_endpoint_average_ev"], 1.0)

        # Calling determine_shirley_endpoints with empty config adopts these defaults
        idx_1, idx_2, i_1, i_2 = determine_shirley_endpoints(self.energy, self.intensity, config={})
        self.assertLess(idx_1, idx_2)
        # Check boundary intensities are averaged
        self.assertAlmostEqual(i_1, 20.0, places=2)
        self.assertAlmostEqual(i_2, 20.0, places=2)

    def test_mirror_reflection_padding_averages_boundary_noise(self) -> None:
        """Window extending outside data must use reflection padding without edge repetition."""
        # Create flat baseline with isolated noise spike at boundary index 0
        noisy_int = np.full(101, 100.0)
        noisy_int[0] = 50.0  # Noise dip at boundary
        noisy_int[1] = 100.0
        noisy_int[2] = 100.0

        # With 0 eV averaging, i_1 is 50.0
        _, _, raw_i1, _ = determine_shirley_endpoints(
            self.energy, noisy_int, config={"strategy": "edge", "average_width_ev": 0.0}
        )
        self.assertEqual(raw_i1, 50.0)

        # With 1.0 eV averaging, the centered window [-0.5 eV, +0.5 eV] reflects neighbors
        # and smooths out the spike
        _, _, avg_i1, _ = determine_shirley_endpoints(
            self.energy, noisy_int, config={"strategy": "edge", "average_width_ev": 1.0}
        )
        self.assertGreater(avg_i1, 90.0)
        self.assertLess(avg_i1, 100.0)

    def test_per_region_endpoint_configuration_overrides(self) -> None:
        """Per-region configuration must allow distinct strategies and averaging widths."""
        cfg = {
            "default_endpoint_strategy": "minima",
            "default_endpoint_average_ev": 1.0,
            "region_endpoint_config": {
                "Al2p": {"strategy": "edge", "average_ev": 0.5},
                "Ti2p": {"strategy": "minima", "average_ev": 2.0},
            },
        }

        # For Al2p, should resolve to strategy='edge' and average=0.5
        al_cfg = dict(cfg, region="Al2p")
        idx_1_al, idx_2_al, _, _ = determine_shirley_endpoints(self.energy, self.intensity, al_cfg)
        self.assertEqual(idx_1_al, 0)
        self.assertEqual(idx_2_al, len(self.intensity) - 1)

        # For unconfigured region (e.g. O1s), should fall back to default 'minima'
        o_cfg = dict(cfg, region="O1s")
        idx_1_o, idx_2_o, _, _ = determine_shirley_endpoints(self.energy, self.intensity, o_cfg)
        self.assertLess(idx_1_o, idx_2_o)

    def test_invalid_strategy_raises_value_error(self) -> None:
        """Invalid strategy string must raise an informative ValueError."""
        with self.assertRaises(ValueError) as ctx:
            determine_shirley_endpoints(
                self.energy, self.intensity, config={"strategy": "INVALID_STRATEGY"}
            )
        self.assertIn("Supported strategies are: 'minima', 'edge', 'direct'", str(ctx.exception))

    def test_shirley_background_with_edge_and_minima_strategies(self) -> None:
        """Calculate Shirley background using both strategies produces valid outputs with B(E) <= I(E)."""
        bg_edge, area_edge = calculate_shirley_background(
            self.energy, self.intensity, config={"strategy": "edge", "average_width_ev": 0.5}
        )
        self.assertTrue(np.all(bg_edge <= self.intensity + 1e-6))
        self.assertGreater(area_edge, 0.0)

        bg_min, area_min = calculate_shirley_background(
            self.energy, self.intensity, config={"strategy": "minima", "average_width_ev": 1.0}
        )
        self.assertTrue(np.all(bg_min <= self.intensity + 1e-6))
        self.assertGreater(area_min, 0.0)

    def test_strategy_direct_resolves_points_and_averages(self) -> None:
        """Strategy 'direct' resolves user-specified energy points to nearest indices."""
        cfg = {"strategy": "direct", "points": (72.0, 78.0), "average_width_ev": 0.0}
        idx_1, idx_2, i_1, i_2 = determine_shirley_endpoints(self.energy, self.intensity, config=cfg)
        expected_idx_1 = int(np.argmin(np.abs(self.energy - 72.0)))
        expected_idx_2 = int(np.argmin(np.abs(self.energy - 78.0)))
        self.assertEqual(idx_1, expected_idx_1)
        self.assertEqual(idx_2, expected_idx_2)
        self.assertAlmostEqual(i_1, self.intensity[expected_idx_1], places=5)
        self.assertAlmostEqual(i_2, self.intensity[expected_idx_2], places=5)

    def test_strategy_direct_with_mirror_padding_averaging(self) -> None:
        """Strategy 'direct' with averaging applies mirror reflection padding correctly."""
        cfg = {"strategy": "direct", "points": (70.0, 80.0), "average_width_ev": 1.0}
        idx_1, idx_2, i_1, i_2 = determine_shirley_endpoints(self.energy, self.intensity, config=cfg)
        self.assertEqual(idx_1, 0)
        self.assertEqual(idx_2, len(self.intensity) - 1)
        self.assertAlmostEqual(i_1, 20.0, places=2)
        self.assertAlmostEqual(i_2, 20.0, places=2)

    def test_strategy_direct_per_region_override(self) -> None:
        """Strategy 'direct' can be configured per-region in region_endpoint_config."""
        cfg = {
            "region": "Ti2p",
            "region_endpoint_config": {
                "Ti2p": {"strategy": "direct", "points": (73.0, 77.0), "average_ev": 0.5},
            },
        }
        idx_1, idx_2, _, _ = determine_shirley_endpoints(self.energy, self.intensity, config=cfg)
        self.assertEqual(idx_1, int(np.argmin(np.abs(self.energy - 73.0))))
        self.assertEqual(idx_2, int(np.argmin(np.abs(self.energy - 77.0))))

    def test_strategy_direct_missing_points_raises(self) -> None:
        """Strategy 'direct' without points parameter must raise informative ValueError."""
        with self.assertRaises(ValueError) as ctx:
            determine_shirley_endpoints(self.energy, self.intensity, config={"strategy": "direct"})
        self.assertIn("no endpoint coordinates", str(ctx.exception))

    def test_strategy_direct_invalid_points_format_raises(self) -> None:
        """Strategy 'direct' with malformed points parameter must raise informative ValueError."""
        with self.assertRaises(ValueError) as ctx:
            determine_shirley_endpoints(
                self.energy, self.intensity, config={"strategy": "direct", "points": (72.0,)}
            )
        self.assertIn("expected tuple or list of 2 numbers", str(ctx.exception))

    def test_strategy_direct_degenerate_points_raises(self) -> None:
        """Strategy 'direct' with identical energy points resolving to same index must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            determine_shirley_endpoints(
                self.energy, self.intensity, config={"strategy": "direct", "points": (75.0, 75.0)}
            )
        self.assertIn("degenerate index range", str(ctx.exception))

    def test_strategy_direct_out_of_bounds_points_raises(self) -> None:
        """Strategy 'direct' with points outside spectral range must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            determine_shirley_endpoints(
                self.energy, self.intensity, config={"strategy": "direct", "points": (10.0, 20.0)}
            )
        self.assertIn("outside spectral energy range", str(ctx.exception))

    def test_strategy_direct_non_finite_points_raises(self) -> None:
        """Strategy 'direct' with NaN or Inf coordinates must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            determine_shirley_endpoints(
                self.energy, self.intensity, config={"strategy": "direct", "points": (float("nan"), 75.0)}
            )
        self.assertIn("non-finite points coordinates", str(ctx.exception))

    def test_strategy_direct_reversed_order_supported(self) -> None:
        """Strategy 'direct' handles descending or ascending coordinate order consistently."""
        idx_1a, idx_2a, i_1a, i_2a = determine_shirley_endpoints(
            self.energy, self.intensity, config={"strategy": "direct", "points": (72.0, 78.0), "average_width_ev": 0.0}
        )
        idx_1b, idx_2b, i_1b, i_2b = determine_shirley_endpoints(
            self.energy, self.intensity, config={"strategy": "direct", "points": (78.0, 72.0), "average_width_ev": 0.0}
        )
        self.assertEqual(idx_1a, idx_1b)
        self.assertEqual(idx_2a, idx_2b)
        self.assertEqual(i_1a, i_1b)
        self.assertEqual(i_2a, i_2b)

    def test_shirley_background_with_direct_strategy(self) -> None:
        """Calculate Shirley background using direct strategy produces valid outputs with B(E) <= I(E)."""
        bg_dir, area_dir = calculate_shirley_background(
            self.energy,
            self.intensity,
            config={"strategy": "direct", "points": (72.0, 78.0), "average_width_ev": 1.0},
        )
        self.assertTrue(np.all(bg_dir <= self.intensity + 1e-6))
        self.assertGreater(area_dir, 0.0)

    def test_per_region_average_width_none_disables_averaging(self) -> None:
        """Explicit average_width_ev=None in region_endpoint_config must disable averaging."""
        noisy_int = self.intensity.copy()
        noisy_int[0] = 50.0  # Spike at boundary 0
        cfg = {
            "region": "Al2p",
            "default_endpoint_average_ev": 1.0,
            "region_endpoint_config": {"Al2p": {"strategy": "edge", "average_width_ev": None}},
        }
        _, _, i_1, _ = determine_shirley_endpoints(self.energy, noisy_int, config=cfg)
        # Should return raw intensity without averaging
        self.assertEqual(i_1, 50.0)


class TestAtomicPercentageSplitStatistics(unittest.TestCase):
    """Test suite for atomic percentage split statistics and side-by-side formatting."""

    @classmethod
    def setUpClass(cls) -> None:
        config = {
            "measurements_per_tool": {"J4": 2, "J5": 2},
            "n_die": 3,
            "n_points": 60,
            "seed": 42,
        }
        cls.ary_int, cls.ary_ene, cls.meta_df = generate_pseudo_measurements(config)

    def test_calculate_atomic_percentage_split_statistics(self) -> None:
        """Verify atomic percentage statistics are calculated across train and test splits."""
        # Build mock original and predicted datasets
        meta_orig = self.meta_df.copy()
        # Pair measurement M_J4_00000 with M_J5_00000 and M_J4_00001 with M_J5_00001
        meta_orig.loc[meta_orig["measurement_id"] == "NMG_M_J4_00000", "measurement_id_target"] = "NMG_M_J5_00000"
        meta_orig.loc[meta_orig["measurement_id"] == "NMG_M_J4_00001", "measurement_id_target"] = "NMG_M_J5_00001"

        meta_pred = meta_orig[meta_orig["tool"] == "J4"].copy()
        meta_pred["source_measurement_id"] = meta_pred["measurement_id"]
        meta_pred["measurement_id"] = meta_pred["measurement_id"].str.replace("_M_J4_", "_P_J4J5_")
        meta_pred["split"] = np.where(meta_pred["source_measurement_id"] == "NMG_M_J4_00000", "train", "test")
        meta_pred["spectrum_index"] = np.arange(len(meta_pred))

        # Use ary_int with slight shift for predicted
        ary_pred = self.ary_int[:len(meta_pred)] * 1.05

        df_samples, df_summary = calculate_atomic_percentage_split_statistics(
            (self.ary_int, self.ary_ene, meta_orig),
            (ary_pred, self.ary_ene[:len(meta_pred)], meta_pred),
            config={"splits": ["train", "test"]},
        )

        self.assertFalse(df_samples.empty)
        self.assertFalse(df_summary.empty)
        self.assertEqual(set(df_summary["split"].unique()), {"train", "test"})
        self.assertIn("element", df_summary.columns)
        self.assertIn("target_mean", df_summary.columns)
        self.assertIn("pred_mean", df_summary.columns)
        self.assertIn("diff_mean", df_summary.columns)
        self.assertIn("mae", df_summary.columns)

        # Check side-by-side formatting
        df_side = format_side_by_side_atomic_percentages(df_summary, config={"splits": ["train", "test"]})
        self.assertFalse(df_side.empty)
        self.assertIn("element", df_side.columns)
        self.assertIn("pred_train", df_side.columns)
        self.assertIn("pred_test", df_side.columns)
        self.assertIn("mae_train", df_side.columns)
        self.assertIn("mae_test", df_side.columns)
        self.assertIn("Overall (MAE)", df_side["element"].values)
        self.assertTrue(df_side["mae_train"].str.contains("±").all())
        self.assertTrue(df_side["mae_test"].str.contains("±").all())

        # Check with include_overall=False
        df_side_no_overall = format_side_by_side_atomic_percentages(
            df_summary, config={"splits": ["train", "test"], "include_overall": False}
        )
        self.assertNotIn("Overall (MAE)", df_side_no_overall["element"].values)

        # Check unformatted numerical mode (format_str=False)
        df_side_num = format_side_by_side_atomic_percentages(
            df_summary, config={"splits": ["train", "test"], "format_str": False}
        )
        self.assertIn("mae_train", df_side_num.columns)
        self.assertIn("mae_std_train", df_side_num.columns)
        self.assertIn("target_mean_train", df_side_num.columns)
        self.assertIn("Overall (MAE)", df_side_num["element"].values)
        overall_row = df_side_num[df_side_num["element"] == "Overall (MAE)"].iloc[0]
        self.assertIsInstance(overall_row["mae_train"], float)
        self.assertIsInstance(overall_row["mae_std_train"], float)
        self.assertTrue(np.isnan(overall_row["target_mean_train"]))

        # Check empty DataFrame handling
        self.assertTrue(format_side_by_side_atomic_percentages(pd.DataFrame()).empty)

    def test_split_statistics_empty_splits_warns(self) -> None:
        """Verify warning when requested split is not present in data."""
        meta_orig = self.meta_df.copy()
        meta_orig["measurement_id_target"] = "NMG_M_J5_00000"
        meta_pred = meta_orig[meta_orig["tool"] == "J4"].copy()
        meta_pred["source_measurement_id"] = meta_pred["measurement_id"]
        meta_pred["measurement_id"] = "NMG_P_J4J5_00000"
        meta_pred["split"] = "train"
        meta_pred["spectrum_index"] = np.arange(len(meta_pred))

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df_samples, df_summary = calculate_atomic_percentage_split_statistics(
                (self.ary_int, self.ary_ene, meta_orig),
                (self.ary_int[:len(meta_pred)], self.ary_ene[:len(meta_pred)], meta_pred),
                config={"splits": ["non_existent_split"]},
            )
            self.assertTrue(df_samples.empty)
            self.assertTrue(df_summary.empty)
            self.assertTrue(any("No paired sessions found" in str(item.message) for item in w))


class TestAtomicPercentageVisualization(unittest.TestCase):
    """Test suite for atomic percentage distribution and MAE visualization."""

    def setUp(self) -> None:
        # Create a synthetic df_samples table
        self.df_samples = pd.DataFrame([
            {
                "split": "train",
                "element": "Al",
                "die": 0,
                "target_at%": 30.5,
                "pred_at%": 31.0,
                "diff_at%": 0.5,
                "abs_diff_at%": 0.5,
            },
            {
                "split": "train",
                "element": "Al",
                "die": 1,
                "target_at%": 29.5,
                "pred_at%": 29.8,
                "diff_at%": 0.3,
                "abs_diff_at%": 0.3,
            },
            {
                "split": "test",
                "element": "Al",
                "die": 0,
                "target_at%": 30.0,
                "pred_at%": 30.2,
                "diff_at%": 0.2,
                "abs_diff_at%": 0.2,
            },
            {
                "split": "train",
                "element": "Ti",
                "die": 0,
                "target_at%": 69.5,
                "pred_at%": 69.0,
                "diff_at%": -0.5,
                "abs_diff_at%": 0.5,
            },
            {
                "split": "test",
                "element": "Ti",
                "die": 0,
                "target_at%": 70.0,
                "pred_at%": 69.8,
                "diff_at%": -0.2,
                "abs_diff_at%": 0.2,
            },
        ])

    def test_plot_atomic_percentage_distributions_success(self) -> None:
        """Verify successful generation of 2-panel figure with box and stripplots."""
        from utility.visualization import plot_atomic_percentage_distributions

        fig, (ax_abs, ax_diff) = plot_atomic_percentage_distributions(
            self.df_samples,
            plot_config={
                "title": "Atomic Percentage Distributions",
                "show": False,
            },
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax_abs)
        self.assertIsNotNone(ax_diff)

        # Check titles
        self.assertIn("Atomic % Distribution: Target vs. Predicted", ax_abs.get_title())
        self.assertIn("Atomic % Difference Distribution (Pred - Target)", ax_diff.get_title())

        # Check axis labels
        self.assertEqual(ax_abs.get_xlabel(), "Element")
        self.assertEqual(ax_diff.get_xlabel(), "Element")
        self.assertIn("Percentage", ax_abs.get_ylabel())
        self.assertIn("Difference", ax_diff.get_ylabel())

        # Check that ax_diff contains the zero reference line
        lines = [line.get_ydata() for line in ax_diff.get_lines()]
        has_zero_line = any(np.allclose(y, 0.0) for y in lines if len(y) > 0)
        self.assertTrue(has_zero_line)

    def test_plot_atomic_percentage_distributions_filter(self) -> None:
        """Verify filtering by splits and elements works properly."""
        from utility.visualization import plot_atomic_percentage_distributions

        fig, (ax_abs, ax_diff) = plot_atomic_percentage_distributions(
            self.df_samples,
            plot_config={
                "splits": ["train"],
                "elements": ["Al"],
                "show": False,
            },
        )
        self.assertIsNotNone(fig)

    def test_plot_atomic_percentage_distributions_empty_raises(self) -> None:
        """Verify ValueError when df_at_samples is empty."""
        from utility.visualization import plot_atomic_percentage_distributions

        with self.assertRaises(ValueError):
            plot_atomic_percentage_distributions(pd.DataFrame())

    def test_plot_atomic_percentage_distributions_missing_cols_raises(self) -> None:
        """Verify KeyError when required columns are missing."""
        from utility.visualization import plot_atomic_percentage_distributions

        with self.assertRaises(KeyError):
            plot_atomic_percentage_distributions(pd.DataFrame({"split": ["train"], "dummy": [1]}))

    def test_plot_atomic_percentage_distributions_filter_empty_raises(self) -> None:
        """Verify ValueError when no records match filter."""
        from utility.visualization import plot_atomic_percentage_distributions

        with self.assertRaises(ValueError):
            plot_atomic_percentage_distributions(
                self.df_samples,
                plot_config={"splits": ["non_existent_split"]},
            )

    def test_plot_atomic_percentage_distributions_custom_axes(self) -> None:
        """Verify passing existing axes tuple into plot_config works cleanly."""
        import matplotlib.pyplot as plt
        from utility.visualization import plot_atomic_percentage_distributions

        fig, (ax1, ax2) = plt.subplots(1, 2)
        returned_fig, (ret_ax1, ret_ax2) = plot_atomic_percentage_distributions(
            self.df_samples,
            plot_config={"ax": (ax1, ax2), "show": False},
        )
        self.assertEqual(ret_ax1, ax1)
        self.assertEqual(ret_ax2, ax2)
        plt.close(fig)

    def test_plot_atomic_percentage_mae_from_samples(self) -> None:
        """Verify plot_atomic_percentage_mae generates 2-panel figure with strip and ECDF plots."""
        import matplotlib.pyplot as plt
        from utility.visualization import plot_atomic_percentage_mae

        fig, (ax_strip, ax_ecdf) = plot_atomic_percentage_mae(
            self.df_samples,
            plot_config={
                "title": "Atomic Percentage MAE Test",
                "show": False,
            },
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax_strip)
        self.assertIsNotNone(ax_ecdf)

        # Check titles
        self.assertIn("Sample Absolute Errors & Mean (MAE)", ax_strip.get_title())
        self.assertIn("Empirical Cumulative Error (ECDF)", ax_ecdf.get_title())

        # Check labels
        self.assertEqual(ax_strip.get_xlabel(), "Element")
        self.assertIn("Absolute Error", ax_strip.get_ylabel())
        self.assertIn("Absolute Error", ax_ecdf.get_xlabel())
        self.assertIn("Cumulative Percentage", ax_ecdf.get_ylabel())

        # Check xtick labels include elements and Overall on ax_strip
        xticklabels = [t.get_text() for t in ax_strip.get_xticklabels()]
        self.assertIn("Al", xticklabels)
        self.assertIn("Ti", xticklabels)
        self.assertIn("Overall", xticklabels)
        plt.close(fig)

    def test_plot_atomic_percentage_mae_without_overall(self) -> None:
        """Verify include_overall=False excludes Overall category."""
        import matplotlib.pyplot as plt
        from utility.visualization import plot_atomic_percentage_mae

        fig, (ax_strip, ax_ecdf) = plot_atomic_percentage_mae(
            self.df_samples,
            plot_config={"include_overall": False, "show": False},
        )
        xticklabels = [t.get_text() for t in ax_strip.get_xticklabels()]
        self.assertIn("Al", xticklabels)
        self.assertIn("Ti", xticklabels)
        self.assertNotIn("Overall", xticklabels)
        plt.close(fig)

    def test_plot_atomic_percentage_mae_filters(self) -> None:
        """Verify filtering by split and elements."""
        import matplotlib.pyplot as plt
        from utility.visualization import plot_atomic_percentage_mae

        fig, (ax_strip, ax_ecdf) = plot_atomic_percentage_mae(
            self.df_samples,
            plot_config={
                "splits": ["train"],
                "elements": ["Al"],
                "include_overall": True,
                "show": False,
            },
        )
        xticklabels = [t.get_text() for t in ax_strip.get_xticklabels()]
        self.assertIn("Al", xticklabels)
        self.assertNotIn("Ti", xticklabels)
        self.assertIn("Overall", xticklabels)
        plt.close(fig)

    def test_plot_atomic_percentage_mae_error_handling(self) -> None:
        """Verify error handling on invalid or empty DataFrames."""
        from utility.visualization import plot_atomic_percentage_mae

        # Empty DataFrame
        with self.assertRaises(ValueError):
            plot_atomic_percentage_mae(pd.DataFrame())

        # Missing required columns
        with self.assertRaises(KeyError):
            plot_atomic_percentage_mae(pd.DataFrame({"split": ["train"], "diff": [0.1]}))

        # Missing abs_diff_at% and diff_at%
        with self.assertRaises(KeyError):
            plot_atomic_percentage_mae(pd.DataFrame({"split": ["train"], "element": ["Al"]}))

        # Filter yielding empty DataFrame
        with self.assertRaises(ValueError):
            plot_atomic_percentage_mae(
                self.df_samples,
                plot_config={"splits": ["invalid_split"]},
            )

    def test_plot_atomic_percentage_mae_custom_axis(self) -> None:
        """Verify passing an existing pair of matplotlib Axes objects."""
        import matplotlib.pyplot as plt
        from utility.visualization import plot_atomic_percentage_mae

        fig, (custom_ax1, custom_ax2) = plt.subplots(1, 2, figsize=(12, 5))
        returned_fig, (ret_ax1, ret_ax2) = plot_atomic_percentage_mae(
            self.df_samples,
            plot_config={"ax": (custom_ax1, custom_ax2), "show": False},
        )
        self.assertEqual(ret_ax1, custom_ax1)
        self.assertEqual(ret_ax2, custom_ax2)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()




