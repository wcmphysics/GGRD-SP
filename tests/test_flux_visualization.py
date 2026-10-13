"""Unit tests for total flux and averaged intensity timeline visualization."""

from __future__ import annotations

import unittest

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Non-interactive backend for testing

import numpy as np
import pandas as pd

from utility.pairing import pair_source_target_spectra
from utility.pseudo_measurement import generate_pseudo_measurements
from visualization.visualization import (
    _derive_tool_die_shades,
    plot_flux_and_averaged_intensity_vs_time,
)


class TestFluxAndIntensityVisualization(unittest.TestCase):
    """Test suite for plot_flux_and_averaged_intensity_vs_time."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate synthetic spectra dataset with paired tools."""
        cls.ary_intensity, cls.ary_energy, cls.meta_df = generate_pseudo_measurements(
            {
                "material": "NMG",
                "regions": ["Al2p", "Ti2p"],
                "n_die": 9,
                "n_points": 50,
                "measurements_per_tool": {"J4": 4, "J5": 4},
                "interval_hours_range": (1.0, 4.0),
                "seed": 42,
            }
        )
        cls.meta_df = pair_source_target_spectra(
            cls.meta_df,
            {
                "source_tool": "J4",
                "target_tool": "J5",
                "time_threshold_hours": 12.0,
            },
        )

    def test_derive_tool_die_shades(self) -> None:
        """Test 3-tier colors from black, red, and custom colors."""
        black_shades = _derive_tool_die_shades("black")
        self.assertEqual(black_shades["center"], "#000000")
        self.assertEqual(black_shades["donut"], "#555555")
        self.assertEqual(black_shades["edge"], "#999999")

        red_shades = _derive_tool_die_shades("red")
        self.assertEqual(red_shades["center"], "#8b0000")
        self.assertEqual(red_shades["donut"], "#d62728")
        self.assertEqual(red_shades["edge"], "#f4a582")

        blue_shades = _derive_tool_die_shades("tab:blue")
        self.assertIn("center", blue_shades)
        self.assertIn("donut", blue_shades)
        self.assertIn("edge", blue_shades)

    def test_default_plot_all_regions_and_all_dies(self) -> None:
        """Test default invocation across all regions with all dies plotted."""
        fig, (ax_flux, ax_intensity) = plot_flux_and_averaged_intensity_vs_time(
            self.ary_intensity,
            self.ary_energy,
            self.meta_df,
            plot_config={"source_tool": "J4", "target_tool": "J5", "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax_flux, plt.Axes)
        self.assertIsInstance(ax_intensity, plt.Axes)

        # Upper and lower panels should contain scatter collections
        self.assertGreater(len(ax_flux.collections), 0)
        self.assertGreater(len(ax_intensity.collections), 0)

        # Default is no connecting lines
        self.assertEqual(len(ax_flux.lines), 0)
        self.assertEqual(len(ax_intensity.lines), 0)

        # Labels
        self.assertIn("Total Flux", ax_flux.get_ylabel())
        self.assertIn("Averaged Intensity", ax_intensity.get_ylabel())
        self.assertEqual(ax_intensity.get_xlabel(), "Measurement Time")

        # Verify legend labels and 3-tier colors:
        # source (black) and target (red)
        leg = ax_flux.get_legend()
        self.assertIsNotNone(leg)
        leg_texts = [t.get_text() for t in leg.get_texts()]
        self.assertIn("J4 - Center (Die 0)", leg_texts)
        self.assertIn("J4 - Donut (Dies 5-8)", leg_texts)
        self.assertIn("J4 - Edge (Dies 1-4)", leg_texts)
        self.assertIn("J5 - Center (Die 0)", leg_texts)
        self.assertIn("J5 - Donut (Dies 5-8)", leg_texts)
        self.assertIn("J5 - Edge (Dies 1-4)", leg_texts)

        marker_colors = [h.get_markerfacecolor() for h in leg.legend_handles]
        self.assertEqual(
            marker_colors,
            ["#000000", "#555555", "#999999", "#8b0000", "#d62728", "#f4a582"],
        )
        plt.close(fig)

    def test_single_region_filter(self) -> None:
        """Test filtering by a specific spectral region."""
        fig, (ax_flux, ax_intensity) = plot_flux_and_averaged_intensity_vs_time(
            self.ary_intensity,
            self.ary_energy,
            self.meta_df,
            plot_config={"show": False},
            region="Ti2p",
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIn("Ti2p", ax_flux.get_ylabel())
        self.assertIn("Ti2p", ax_intensity.get_ylabel())
        plt.close(fig)

    def test_connect_lines_enabled(self) -> None:
        """Test chronological connecting lines when connect_lines=True."""
        fig, (ax_flux, ax_intensity) = plot_flux_and_averaged_intensity_vs_time(
            self.ary_intensity,
            self.ary_energy,
            self.meta_df,
            plot_config={"connect_lines": True, "show": False},
        )
        self.assertGreater(len(ax_flux.lines), 0)
        self.assertGreater(len(ax_intensity.lines), 0)
        plt.close(fig)

    def test_show_lines_alias_and_precedence(self) -> None:
        """Test show_lines alias and region argument precedence."""
        # Explicit kwarg takes precedence over config
        fig, (ax_flux, ax_intensity) = plot_flux_and_averaged_intensity_vs_time(
            self.ary_intensity,
            self.ary_energy,
            self.meta_df,
            plot_config={"show_lines": True, "region": "Al2p", "show": False},
            region="Ti2p",
        )
        self.assertGreater(len(ax_flux.lines), 0)
        self.assertIn("Ti2p", ax_flux.get_ylabel())
        plt.close(fig)

    def test_mixed_die_types_with_connecting_lines(self) -> None:
        """Test robust sorting when die column contains mixed int and str."""
        mixed_meta = self.meta_df.copy()
        mixed_meta["die"] = mixed_meta["die"].map(
            lambda d: f"die_{d}" if d in [1, 2] else d
        )
        fig, (ax_flux, ax_intensity) = plot_flux_and_averaged_intensity_vs_time(
            self.ary_intensity,
            self.ary_energy,
            mixed_meta,
            plot_config={"connect_lines": True, "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertGreater(len(ax_flux.lines), 0)
        plt.close(fig)

    def test_single_die_dataset(self) -> None:
        """Test dataset containing only a single wafer die."""
        single_die_meta = self.meta_df[self.meta_df["die"] == 0].copy()
        fig, (ax_flux, ax_intensity) = plot_flux_and_averaged_intensity_vs_time(
            self.ary_intensity,
            self.ary_energy,
            single_die_meta,
            plot_config={"connect_lines": True, "show": False},
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertGreater(len(ax_flux.collections), 0)
        plt.close(fig)

    def test_custom_axes_injection(self) -> None:
        """Test passing pre-created matplotlib Axes."""
        ext_fig, (ext_ax1, ext_ax2) = plt.subplots(2, 1, figsize=(8, 6))
        ret_fig, (ret_ax1, ret_ax2) = plot_flux_and_averaged_intensity_vs_time(
            self.ary_intensity,
            self.ary_energy,
            self.meta_df,
            plot_config={"axes": (ext_ax1, ext_ax2), "show": False},
        )
        self.assertIs(ret_fig, ext_fig)
        self.assertIs(ret_ax1, ext_ax1)
        self.assertIs(ret_ax2, ext_ax2)
        plt.close(ext_fig)

    def test_validation_errors(self) -> None:
        """Test validation and error handling for invalid arguments."""
        # ary_energy is None
        with self.assertRaises(ValueError):
            plot_flux_and_averaged_intensity_vs_time(
                self.ary_intensity,
                None,  # type: ignore[arg-type]
                self.meta_df,
            )

        # Mismatched array shapes
        with self.assertRaises(ValueError):
            plot_flux_and_averaged_intensity_vs_time(
                self.ary_intensity,
                self.ary_energy[:-1],
                self.meta_df,
            )

        # Empty metadata
        with self.assertRaises(ValueError):
            plot_flux_and_averaged_intensity_vs_time(
                self.ary_intensity,
                self.ary_energy,
                pd.DataFrame(),
            )

        # Missing required metadata column
        incomplete_df = self.meta_df.drop(columns=["tool"])
        with self.assertRaises(KeyError):
            plot_flux_and_averaged_intensity_vs_time(
                self.ary_intensity,
                self.ary_energy,
                incomplete_df,
            )

        # Non-existent region
        with self.assertRaises(ValueError):
            plot_flux_and_averaged_intensity_vs_time(
                self.ary_intensity,
                self.ary_energy,
                self.meta_df,
                region="NON_EXISTENT_REGION",
            )


if __name__ == "__main__":
    unittest.main()
