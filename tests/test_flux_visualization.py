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
from visualization.plot_timeline import (
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
                "material": "all",
            },
        )

    def tearDown(self) -> None:
        """Close open matplotlib figures."""
        plt.close("all")

    def test_plot_flux_and_intensity_options(self) -> None:
        """Verify plotting across default, regional filter, and connecting lines."""
        # 1. Default all regions
        fig, (ax_flux, ax_intensity) = plot_flux_and_averaged_intensity_vs_time(
            self.ary_intensity,
            self.ary_energy,
            self.meta_df,
            plot_config={"source_tool": "J4", "target_tool": "J5", "show": False},
        )
        self.assertGreater(len(ax_flux.collections), 0)
        self.assertGreater(len(ax_intensity.collections), 0)
        self.assertIn("Total Flux", ax_flux.get_ylabel())

        # 2. Single region with connecting lines
        fig_conn, (ax_f2, _) = plot_flux_and_averaged_intensity_vs_time(
            self.ary_intensity,
            self.ary_energy,
            self.meta_df,
            plot_config={"region": "Ti2p", "connect_lines": True, "show": False},
        )
        self.assertGreater(len(ax_f2.lines), 0)
        self.assertIn("Ti2p", ax_f2.get_ylabel())

        # 3. Custom axes injection
        ext_fig, (ext_ax1, ext_ax2) = plt.subplots(2, 1, figsize=(8, 6))
        ret_fig, _ = plot_flux_and_averaged_intensity_vs_time(
            self.ary_intensity,
            self.ary_energy,
            self.meta_df,
            plot_config={"axes": (ext_ax1, ext_ax2), "show": False},
        )
        self.assertIs(ret_fig, ext_fig)

    def test_defensive_error_handling(self) -> None:
        """Table-driven test for invalid inputs, missing columns, and bad regions."""
        with self.assertRaises(ValueError):
            plot_flux_and_averaged_intensity_vs_time(self.ary_intensity, None, self.meta_df)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            plot_flux_and_averaged_intensity_vs_time(self.ary_intensity, self.ary_energy[:-1], self.meta_df)

        with self.assertRaises(ValueError):
            plot_flux_and_averaged_intensity_vs_time(self.ary_intensity, self.ary_energy, pd.DataFrame())

        with self.assertRaises(KeyError):
            plot_flux_and_averaged_intensity_vs_time(
                self.ary_intensity, self.ary_energy, self.meta_df.drop(columns=["tool"])
            )

        with self.assertRaises(ValueError):
            plot_flux_and_averaged_intensity_vs_time(
                self.ary_intensity, self.ary_energy, self.meta_df, plot_config={"region": "BAD_REGION"}
            )


if __name__ == "__main__":
    unittest.main()
