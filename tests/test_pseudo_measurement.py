"""Unit tests for utility/pseudo_measurement.py, material_profiles.py, and plots."""

import copy
import unittest
import numpy as np
import pandas as pd

from utility.material_profiles import (
    BUILTIN_MATERIAL_PROFILES,
    resolve_material_region_profile,
)
from utility.pseudo_measurement import (
    _calculate_shirley_step,
    _synthesize_regional_spectrum,
    generate_pseudo_measurements,
    get_default_measurement_config,
    pseudo_voigt,
)
from visualization import (
    plot_regional_spectra,
    plot_tool_comparison,
)


def _single_material_cfg(
    *,
    material: str = "TiAlC",
    recipe: str = "RCP_NMG_A",
    active_regions: list[str] | None = None,
    recipe_regions: dict[str, tuple[float, float]] | None = None,
    tool_counts: dict[str, int] | None = None,
    **overrides: object,
) -> dict[str, object]:
    """Build a compact canonical data_generation config for unit tests."""
    regs = active_regions or ["Al2p", "Ti2p", "O1s", "C1s", "Cl2p"]
    default_bounds = {
        "Al2p": (70.0, 78.0),
        "Ti2p": (452.0, 470.0),
        "O1s": (526.0, 536.0),
        "C1s": (280.0, 292.0),
        "Cl2p": (196.0, 206.0),
        "N1s": (394.0, 404.0),
        "Si2p": (97.0, 107.0),
        "Hf4f": (13.0, 23.0),
    }
    rcp_map = recipe_regions or {r: default_bounds[r] for r in regs}
    counts = tool_counts if tool_counts is not None else {"J4": 2, "J5": 2}
    cfg: dict[str, object] = {
        "materials": {material: {"regions": regs}},
        "recipes": {recipe: rcp_map},
        "material_recipes": {material: {recipe: counts}},
    }
    cfg.update(overrides)
    return cfg


class TestPseudoMeasurementGeneration(unittest.TestCase):
    """Test suite for multi-material, recipe-driven pseudo-measurement generation."""

    def test_default_shapes_and_invariants(self) -> None:
        """Verify default multi-material config generates exact shapes and grids."""
        cfg = get_default_measurement_config()
        ary_intensity, ary_energy, meta_df = generate_pseudo_measurements()

        expected_total = sum(
            sum(tool_counts.values()) * cfg["n_die"] * len(cfg["recipes"][rcp])
            for rcp_map in cfg["material_recipes"].values()
            for rcp, tool_counts in rcp_map.items()
        )
        self.assertEqual(ary_intensity.shape, (expected_total, 100))
        self.assertEqual(ary_energy.shape, (expected_total, 100))
        self.assertEqual(len(meta_df), expected_total)
        self.assertTrue(np.all(ary_intensity >= 0.0))
        self.assertIn("recipe", meta_df.columns)
        self.assertEqual(set(meta_df["material"].unique()), {"TiAlC", "TiN", "HfO2"})

        # Ascending energy grid must be identical within each (recipe, region)
        for (recipe, region), grp in meta_df.groupby(["recipe", "region"]):
            indices = grp["spectrum_index"].to_numpy(dtype=int)
            first_be = ary_energy[indices[0]]
            self.assertTrue(np.all(np.diff(first_be) > 0))
            for idx in indices[1:]:
                np.testing.assert_array_equal(ary_energy[idx], first_be)

    def test_material_specific_peak_physics(self) -> None:
        """Verify TiAlC vs TiN vs HfO2 exhibit distinct chemical-state peaks."""
        cfg = {
            "materials": {
                "TiAlC": {"regions": ["Ti2p", "C1s", "O1s"]},
                "TiN": {"regions": ["Ti2p", "N1s", "C1s", "O1s"]},
                "HfO2": {"regions": ["Hf4f", "O1s", "C1s"]},
            },
            "recipes": {
                "RCP_TI": {
                    "Ti2p": (451.0, 469.0),
                    "C1s": (280.0, 292.0),
                    "O1s": (526.0, 536.0),
                },
                "RCP_HF": {
                    "Hf4f": (13.0, 23.0),
                    "O1s": (526.0, 536.0),
                    "C1s": (280.0, 292.0),
                },
            },
            "material_recipes": {
                "TiAlC": {"RCP_TI": {"J4": 1}},
                "TiN": {"RCP_TI": {"J4": 1}},
                "HfO2": {"RCP_HF": {"J4": 1}},
            },
            "n_die": 1,
            "n_points": 200,
            "die_repeatability_std": 0.0,
            "noise_relative_std": 0.0,
            "shirley_background": {"enabled": False},
            "seed": 42,
        }
        ary_i, ary_e, meta = generate_pseudo_measurements(cfg)

        # Ti2p: TiAlC has Ti-Al/Ti-C near 453.8-454.8 eV; TiN peaks at ~455.2 eV
        idx_tialc_ti = int(
            meta[(meta["material"] == "TiAlC") & (meta["region"] == "Ti2p")][
                "spectrum_index"
            ].iloc[0]
        )
        idx_tin_ti = int(
            meta[(meta["material"] == "TiN") & (meta["region"] == "Ti2p")][
                "spectrum_index"
            ].iloc[0]
        )
        be_ti = ary_e[idx_tialc_ti]
        peak_e_tialc = float(be_ti[np.argmax(ary_i[idx_tialc_ti])])
        peak_e_tin = float(be_ti[np.argmax(ary_i[idx_tin_ti])])
        self.assertLess(peak_e_tialc, peak_e_tin)

        # C1s: TiAlC has Ti-C carbide peak near 281.8 eV; TiN and HfO2 do not
        idx_tialc_c = int(
            meta[(meta["material"] == "TiAlC") & (meta["region"] == "C1s")][
                "spectrum_index"
            ].iloc[0]
        )
        idx_tin_c = int(
            meta[(meta["material"] == "TiN") & (meta["region"] == "C1s")][
                "spectrum_index"
            ].iloc[0]
        )
        be_c = ary_e[idx_tialc_c]
        carbide_idx = int(np.argmin(np.abs(be_c - 281.8)))
        self.assertGreater(
            ary_i[idx_tialc_c, carbide_idx] / np.max(ary_i[idx_tialc_c]),
            ary_i[idx_tin_c, carbide_idx] / np.max(ary_i[idx_tin_c]) * 3.0,
        )

        # Hf4f: HfO2 has strong doublet peaks near 17.0 and 18.7 eV
        idx_hf = int(
            meta[(meta["material"] == "HfO2") & (meta["region"] == "Hf4f")][
                "spectrum_index"
            ].iloc[0]
        )
        be_hf = ary_e[idx_hf]
        self.assertAlmostEqual(float(be_hf[np.argmax(ary_i[idx_hf])]), 17.0, delta=0.3)

    def test_absent_region_generates_scaled_background_noise(self) -> None:
        """Verify probing a region absent from a material produces flat continuum + noise."""
        cfg = _single_material_cfg(
            material="HfO2",
            recipe="RCP_HK_A",
            active_regions=["Hf4f", "O1s"],
            recipe_regions={
                "Hf4f": (13.0, 23.0),
                "O1s": (526.0, 536.0),
                "Ti2p": (452.0, 470.0),
            },
            tool_counts={"J4": 1, "J5": 1},
            tool_offsets={
                "J4": {"shift_ev": 0.0, "scale": 1000.0},
                "J5": {"shift_ev": 0.5, "scale": 2000.0},
            },
            n_die=1,
            n_points=100,
            die_repeatability_std=0.0,
            noise_relative_std=0.0,
            background_relative_strength=0.5,
            shirley_background={"enabled": True, "step_ratio": 0.15},
            seed=42,
        )
        ary_i, _, meta = generate_pseudo_measurements(cfg)
        j4_bg_idx = int(
            meta[(meta["tool"] == "J4") & (meta["region"] == "Ti2p")][
                "spectrum_index"
            ].iloc[0]
        )
        j5_bg_idx = int(
            meta[(meta["tool"] == "J5") & (meta["region"] == "Ti2p")][
                "spectrum_index"
            ].iloc[0]
        )
        # With zero noise, absent region is flat at 0.5 * total_scale (no Shirley step)
        j4_scale = float(meta.loc[j4_bg_idx, "scale_total"])
        j5_scale = float(meta.loc[j5_bg_idx, "scale_total"])
        np.testing.assert_allclose(ary_i[j4_bg_idx], 0.5 * j4_scale, rtol=1e-6)
        np.testing.assert_allclose(ary_i[j5_bg_idx], 0.5 * j5_scale, rtol=1e-6)

    def test_recipe_energy_ranges_and_per_group_tool_counts(self) -> None:
        """Verify distinct recipe energy windows, asymmetric counts, and 5-part IDs."""
        cfg = {
            "materials": {"TiAlC": {"regions": ["Ti2p", "O1s"]}},
            "recipes": {
                "RCP_NMG_A": {"Ti2p": (452.0, 470.0), "O1s": (526.0, 536.0)},
                "RCP_NMG_B": {"Ti2p": (450.0, 466.0)},
            },
            "material_recipes": {
                "TiAlC": {
                    "RCP_NMG_A": {"J4": 3, "J5": 2},
                    "RCP_NMG_B": {"J4": 2, "J5": 0},
                }
            },
            "n_die": 1,
            "n_points": 80,
            "seed": 42,
        }
        ary_i, ary_e, meta = generate_pseudo_measurements(cfg)
        # (3+2)*2 + (2+0)*1 = 12 spectra
        self.assertEqual(len(meta), 12)
        self.assertEqual(ary_i.shape, (12, 80))

        ids_a_j4 = sorted(
            meta[(meta["recipe"] == "RCP_NMG_A") & (meta["tool"] == "J4")][
                "measurement_id"
            ].unique()
        )
        self.assertEqual(
            ids_a_j4,
            [
                "TiAlC-RCP_NMG_A-M-J4-00000",
                "TiAlC-RCP_NMG_A-M-J4-00001",
                "TiAlC-RCP_NMG_A-M-J4-00002",
            ],
        )
        # Recipe B has 0 J5 sessions
        self.assertTrue(
            meta[(meta["recipe"] == "RCP_NMG_B") & (meta["tool"] == "J5")].empty
        )
        # Same region Ti2p has different energy bounds per recipe
        idx_a = int(
            meta[(meta["recipe"] == "RCP_NMG_A") & (meta["region"] == "Ti2p")][
                "spectrum_index"
            ].iloc[0]
        )
        idx_b = int(
            meta[(meta["recipe"] == "RCP_NMG_B") & (meta["region"] == "Ti2p")][
                "spectrum_index"
            ].iloc[0]
        )
        self.assertAlmostEqual(float(ary_e[idx_a, 0]), 452.0)
        self.assertAlmostEqual(float(ary_e[idx_a, -1]), 470.0)
        self.assertAlmostEqual(float(ary_e[idx_b, 0]), 450.0)
        self.assertAlmostEqual(float(ary_e[idx_b, -1]), 466.0)

    def test_tool_offsets_decay_and_die_zones(self) -> None:
        """Verify tool scale ratios, time decay / maintenance jumps, and die hierarchy."""
        config = _single_material_cfg(
            tool_counts={"J4": 15, "J5": 15},
            interval_hours_range=(36.0, 48.0),
            maintenance_interval_days=15.0,
            die_repeatability_std=0.0,
            seed=42,
        )
        _, _, meta_df = generate_pseudo_measurements(config)

        computed = meta_df["scale_tool"] * meta_df["scale_time"] * meta_df["scale_die"]
        np.testing.assert_allclose(meta_df["scale_total"], computed, rtol=1e-5)

        j4_dies = (
            meta_df[meta_df["tool"] == "J4"]
            .drop_duplicates(subset=["die"])
            .set_index("die")
        )
        center_s = j4_dies.loc[0, "scale_die"]
        mid_s = [j4_dies.loc[d, "scale_die"] for d in (5, 6, 7, 8)]
        edge_s = [j4_dies.loc[d, "scale_die"] for d in (1, 2, 3, 4)]
        self.assertTrue(all(center_s > s for s in mid_s))
        self.assertGreater(min(mid_s), max(edge_s))

    def test_region_relative_strengths_and_t7_distribution(self) -> None:
        """Verify regional peak strength proportionality and t7_code grouping."""
        strengths = {"Al2p": 5.0, "Ti2p": 7.0, "O1s": 10.0, "C1s": 8.0, "Cl2p": 1.0}
        cfg = _single_material_cfg(
            tool_counts={"J4": 20, "J5": 20},
            measurements_per_t7_code=10,
            die_variation_std=0.0,
            noise_relative_std=0.0,
            shirley_background={"enabled": False},
            seed=42,
        )
        cfg["materials"]["TiAlC"]["relative_strengths"] = strengths
        ary_i, _, meta_df = generate_pseudo_measurements(cfg)

        die0 = meta_df[
            (meta_df["measurement_id"] == "TiAlC-RCP_NMG_A-M-J4-00000")
            & (meta_df["die"] == 0)
        ]
        max_vals = {
            row["region"]: float(np.max(ary_i[int(row["spectrum_index"])]))
            for _, row in die0.iterrows()
        }
        self.assertAlmostEqual(max_vals["Ti2p"] / max_vals["Cl2p"], 7.0, delta=0.05)
        self.assertAlmostEqual(max_vals["O1s"] / max_vals["Cl2p"], 10.0, delta=0.05)

        j4_t7 = (
            meta_df[meta_df["tool"] == "J4"]
            .drop_duplicates(subset=["measurement_id"])["t7_code"]
            .tolist()
        )
        self.assertEqual(j4_t7, ["T7_001"] * 10 + ["T7_002"] * 10)

    def test_shirley_background_invariants_and_overrides(self) -> None:
        """Verify Shirley elevation, monotonicity, disablement, and per-region overrides."""
        base = _single_material_cfg(
            active_regions=["Ti2p", "Cl2p"],
            tool_counts={"J4": 1},
            n_die=1,
            noise_relative_std=0.0,
            seed=42,
        )
        ary_en, _, meta_en = generate_pseudo_measurements(
            {**base, "shirley_background": {"enabled": True, "step_ratio": 0.15}}
        )
        ary_dis, _, _ = generate_pseudo_measurements(
            {**base, "shirley_background": {"enabled": False, "step_ratio": 0.15}}
        )
        ary_zero, _, _ = generate_pseudo_measurements(
            {**base, "shirley_background": {"enabled": True, "step_ratio": 0.0}}
        )
        np.testing.assert_allclose(ary_dis, ary_zero, atol=1e-12)

        ti_idx = int(meta_en[meta_en["region"] == "Ti2p"]["spectrum_index"].iloc[0])
        step = ary_en[ti_idx] - ary_dis[ti_idx]
        self.assertTrue(np.all(np.diff(step) >= -1e-12))
        self.assertGreater(step[-1] - step[0], 0.0)

        # Per-region shirley_enabled override inside material spec
        custom_cfg = copy.deepcopy(base)
        ti_profile = resolve_material_region_profile("TiAlC", "Ti2p")
        ti_profile["shirley_enabled"] = False
        custom_cfg["materials"]["TiAlC"]["region_profiles"] = {"Ti2p": ti_profile}
        ary_ov, _, meta_ov = generate_pseudo_measurements(
            {**custom_cfg, "shirley_background": {"enabled": True, "step_ratio": 0.10}}
        )
        ary_ov_dis, _, _ = generate_pseudo_measurements(
            {**custom_cfg, "shirley_background": {"enabled": False}}
        )
        ti_ov = int(meta_ov[meta_ov["region"] == "Ti2p"]["spectrum_index"].iloc[0])
        cl_ov = int(meta_ov[meta_ov["region"] == "Cl2p"]["spectrum_index"].iloc[0])
        np.testing.assert_allclose(ary_ov[ti_ov], ary_ov_dis[ti_ov], atol=1e-12)
        self.assertGreater(ary_ov[cl_ov][-1] - ary_ov_dis[cl_ov][-1], 0.0)

    def test_shirley_descending_and_degenerate_grids(self) -> None:
        """Verify Shirley helper handles descending energy and degenerate inputs."""
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
        spec_asc = _synthesize_regional_spectrum(
            e_asc, profile, params, np.random.default_rng(42)
        )
        spec_desc = _synthesize_regional_spectrum(
            e_desc, profile, params, np.random.default_rng(42)
        )
        np.testing.assert_allclose(spec_desc, spec_asc[::-1], atol=1e-12)
        np.testing.assert_allclose(
            _calculate_shirley_step(e_asc, np.zeros_like(e_asc), 0.10), 0.0
        )
        np.testing.assert_allclose(
            _calculate_shirley_step(np.array([50.0]), np.array([10.0]), 0.10), 0.0
        )

    def test_defensive_error_handling(self) -> None:
        """Table-driven test for invalid generator configuration parameters."""
        error_configs = [
            ("negative count", _single_material_cfg(tool_counts={"J4": -1})),
            ("all zero counts", _single_material_cfg(tool_counts={"J4": 0, "J5": 0})),
            ("invalid n_points", _single_material_cfg(n_points=1)),
            ("invalid intervals", _single_material_cfg(interval_hours_range=(24.0, 4.0))),
            ("non-positive t7", _single_material_cfg(measurements_per_t7_code=0)),
            ("non-positive bg strength", _single_material_cfg(background_relative_strength=0.0)),
            ("hyphen in material", _single_material_cfg(material="Ti-AlC")),
            ("hyphen in recipe", _single_material_cfg(recipe="RCP-NMG-A")),
            ("underscore in tool", _single_material_cfg(tool_counts={"J_4": 1})),
            ("negative shirley step", _single_material_cfg(shirley_background={"step_ratio": -0.1})),
            ("excessive shirley step", _single_material_cfg(shirley_background={"step_ratio": 1.5})),
            ("invalid shirley enabled", _single_material_cfg(shirley_background={"enabled": "yes"})),
            ("unknown material region", _single_material_cfg(material="TiN", active_regions=["Al2p"])),
        ]
        for name, cfg in error_configs:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    generate_pseudo_measurements(cfg)

        with self.assertRaises(ValueError):
            pseudo_voigt(
                np.linspace(70, 78, 10),
                {"center": 72.0, "amplitude": 10.0, "fwhm": 0.0},
            )


class TestSpectralVisualization(unittest.TestCase):
    """Test suite for pseudo-measurement plotting and tool comparison."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate pseudo-measurement data for visualization."""
        config = _single_material_cfg(
            tool_counts={"J4": 1, "J5": 1},
            n_die=2,
            n_points=50,
            seed=42,
        )
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

        with self.assertRaises(ValueError):
            plot_tool_comparison(
                self.ary_ene, self.ary_int, self.meta_df, compare_config={"tools": ["J4"]}
            )
        with self.assertRaises(ValueError):
            plot_regional_spectra(
                self.ary_ene, self.ary_int, self.meta_df, plot_config={"filters": {"tool": "BAD"}}
            )


if __name__ == "__main__":
    unittest.main()
