"""Unit tests for utility/pairing.py and pairing timeline visualization."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from utility.pairing import pair_source_target_spectra
from utility.pseudo_measurement import generate_pseudo_measurements
from visualization import plot_pairing_timeline


class TestPairing(unittest.TestCase):
    """Test suite for 1-to-1 source-to-target spectrum pairing and constraints."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate realistic pseudo-measurement dataset for pairing tests."""
        config = {
            "measurements_per_tool": {"J4": 6, "J5": 6},
            "n_die": 3,
            "n_points": 50,
            "seed": 42,
        }
        cls.ary_int, cls.ary_ene, cls.meta_df = generate_pseudo_measurements(config)
        cls.pairing_cfg = {
            "source_tool": "J4",
            "target_tool": "J5",
            "time_threshold_hours": 12.0,
            "material": "all",
        }

    def test_default_pairing_columns_and_invariants(self) -> None:
        """Verify appended metadata columns, 1-to-1 mapping, and die/region consistency."""
        paired_df = pair_source_target_spectra(self.meta_df, config=self.pairing_cfg)

        expected_new_cols = {
            "target_tool",
            "target_measurement_id",
            "target_spectrum_index",
            "target_time",
            "time_diff_hours",
        }
        self.assertTrue(expected_new_cols.issubset(set(paired_df.columns)))

        src_mask = paired_df["tool"] == "J4"
        self.assertTrue((paired_df.loc[src_mask, "target_tool"] == "J5").all())
        self.assertTrue(paired_df.loc[~src_mask, "target_tool"].isna().all())

        # Check strictly 1-to-1 session mapping and threshold adherence
        paired_sessions = (
            paired_df[src_mask & paired_df["target_measurement_id"].notna()][
                ["measurement_id", "target_measurement_id", "time_diff_hours"]
            ]
            .drop_duplicates(subset=["measurement_id"])
        )
        self.assertEqual(
            len(paired_sessions["target_measurement_id"]),
            paired_sessions["target_measurement_id"].nunique(),
        )
        self.assertTrue((paired_sessions["time_diff_hours"] <= 12.0).all())

        # Die and region correspondence
        paired_rows = paired_df[src_mask & paired_df["target_spectrum_index"].notna()]
        first_row = paired_rows.iloc[0]
        tgt_row = self.meta_df.loc[int(first_row["target_spectrum_index"])]
        self.assertEqual(tgt_row["tool"], "J5")
        self.assertEqual(tgt_row["die"], first_row["die"])
        self.assertEqual(tgt_row["region"], first_row["region"])

    def test_greedy_and_optimal_collision_scenarios(self) -> None:
        """Verify greedy and optimal assignment resolution under competing sessions."""
        t0 = datetime(2026, 1, 1, 10, 0, 0)
        df_compete = pd.DataFrame([
            {"spectrum_index": 0, "tool": "J4", "measurement_id": "M_J4_1", "time": t0, "die": 0, "region": "Al2p"},
            {"spectrum_index": 1, "tool": "J4", "measurement_id": "M_J4_2", "time": t0 + timedelta(hours=1), "die": 0, "region": "Al2p"},
            {"spectrum_index": 2, "tool": "J5", "measurement_id": "M_J5_1", "time": t0 + timedelta(hours=2), "die": 0, "region": "Al2p"},
        ])

        # Optimal method assigns closest session 1-to-1
        paired_opt = pair_source_target_spectra(
            df_compete, config={**self.pairing_cfg, "time_threshold_hours": 5.0}
        )
        assigned = paired_opt[paired_opt["tool"] == "J4"]["target_measurement_id"].dropna().tolist()
        self.assertEqual(len(assigned), 1)
        self.assertEqual(assigned[0], "M_J5_1")

        # Greedy method
        paired_greedy = pair_source_target_spectra(
            self.meta_df, config={**self.pairing_cfg, "method": "greedy"}
        )
        src_mask = paired_greedy["tool"] == "J4"
        greedy_sessions = (
            paired_greedy[src_mask & paired_greedy["target_measurement_id"].notna()][
                ["measurement_id", "target_measurement_id"]
            ]
            .drop_duplicates(subset=["measurement_id"])
        )
        self.assertEqual(
            len(greedy_sessions["target_measurement_id"]),
            greedy_sessions["target_measurement_id"].nunique(),
        )

    def test_material_and_t7_code_isolation(self) -> None:
        """Verify pairing isolates cross-material sessions and respects t7_code."""
        t0 = datetime(2026, 1, 1, 10, 0, 0)
        df_multi = pd.DataFrame([
            {"spectrum_index": 0, "material": "MatA", "t7_code": "T1", "tool": "J4", "measurement_id": "M1", "time": t0, "die": 0, "region": "R1"},
            {"spectrum_index": 1, "material": "MatB", "t7_code": "T2", "tool": "J5", "measurement_id": "M2", "time": t0 + timedelta(hours=1), "die": 0, "region": "R1"},
            {"spectrum_index": 2, "material": "MatA", "t7_code": "T1", "tool": "J5", "measurement_id": "M3", "time": t0 + timedelta(hours=3), "die": 0, "region": "R1"},
        ])
        paired = pair_source_target_spectra(df_multi, config={**self.pairing_cfg, "time_threshold_hours": 10.0})
        src_row = paired[paired["measurement_id"] == "M1"].iloc[0]
        # M1 must pair with M3 (same material & t7_code), skipping closer M2
        self.assertEqual(src_row["target_measurement_id"], "M3")
        self.assertEqual(src_row["target_t7_code"], "T1")

    def test_defensive_error_handling(self) -> None:
        """Table-driven test for invalid pairing parameters and missing columns."""
        error_cases = [
            ("same source and target", {**self.pairing_cfg, "source_tool": "J4", "target_tool": "J4"}, ValueError),
            ("negative threshold", {**self.pairing_cfg, "time_threshold_hours": -1.0}, ValueError),
            ("invalid method", {**self.pairing_cfg, "method": "unknown"}, ValueError),
            ("none material", {**self.pairing_cfg, "material": None}, ValueError),
        ]
        for name, cfg, expected_err in error_cases:
            with self.subTest(case=name):
                with self.assertRaises(expected_err):
                    pair_source_target_spectra(self.meta_df, config=cfg)

        with self.assertRaises(KeyError):
            pair_source_target_spectra(self.meta_df.drop(columns=["tool"]), config=self.pairing_cfg)

    def test_hungarian_unavailable_raises_runtime_error(self) -> None:
        """Verify RuntimeError is raised when Hungarian matcher is unavailable."""
        from unittest.mock import patch
        with patch("utility.pairing.linear_sum_assignment", None):
            with self.assertRaises(RuntimeError):
                pair_source_target_spectra(
                    self.meta_df,
                    config={**self.pairing_cfg, "method": "optimal"},
                )


class TestPairingVisualization(unittest.TestCase):
    """Test suite for timeline visualization and label staggering."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate paired dataset for timeline visualization."""
        config = {
            "measurements_per_tool": {"J4": 3, "J5": 3},
            "n_die": 2,
            "n_points": 30,
            "seed": 42,
        }
        _, _, meta = generate_pseudo_measurements(config)
        cls.paired_df = pair_source_target_spectra(
            meta,
            config={
                "source_tool": "J4",
                "target_tool": "J5",
                "time_threshold_hours": 12.0,
                "material": "all",
            },
        )

    def tearDown(self) -> None:
        """Close open matplotlib figures."""
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_pairing_timeline_and_staggering(self) -> None:
        """Verify timeline plot renders valid axes, labels, and handles string dates."""
        fig, ax = plot_pairing_timeline(
            self.paired_df,
            plot_config={"show": False, "annotate_measurements": True, "stagger_levels": 2},
        )
        self.assertIsNotNone(fig)
        self.assertEqual(len(ax.get_yticklabels()), 2)

        # String target time compatibility
        df_str = self.paired_df.copy()
        df_str["target_time"] = df_str["target_time"].astype(str)
        fig_str, _ = plot_pairing_timeline(df_str, plot_config={"show": False})
        self.assertIsNotNone(fig_str)

        with self.assertRaises(KeyError):
            plot_pairing_timeline(pd.DataFrame({"dummy": [1, 2]}))


if __name__ == "__main__":
    unittest.main()
