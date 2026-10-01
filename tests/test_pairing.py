"""Unit tests for utility/pairing.py and pairing timeline visualization."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from utility.pairing import (
    get_default_pairing_config,
    pair_source_target_spectra,
)
from utility.pseudo_measurement import generate_pseudo_measurements
from utility.visualization import plot_pairing_timeline


class TestPairing(unittest.TestCase):
    """Test suite for 1-to-1 source-to-target spectrum pairing."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate a realistic dataset for testing pairing."""
        config = {
            "measurements_per_tool": {"J4": 6, "J5": 6},
            "n_die": 3,
            "n_points": 50,
            "seed": 42,
        }
        cls.ary_int, cls.ary_ene, cls.meta_df = generate_pseudo_measurements(config)

    def test_default_pairing_columns_and_invariants(self) -> None:
        """Verify appended columns, strictly 1-to-1 matching, and threshold adherence."""
        paired_df = pair_source_target_spectra(self.meta_df)

        expected_new_cols = {
            "tool_target",
            "measurement_id_target",
            "spectrum_index_target",
            "time_target",
            "time_diff_hours",
        }
        self.assertTrue(expected_new_cols.issubset(set(paired_df.columns)))

        # Source rows must have tool_target == "J5"
        src_mask = paired_df["tool"] == "J4"
        self.assertTrue((paired_df.loc[src_mask, "tool_target"] == "J5").all())

        # Non-source rows must have tool_target NA/None
        non_src_mask = paired_df["tool"] != "J4"
        self.assertTrue(paired_df.loc[non_src_mask, "tool_target"].isna().all())

        # Check strictly 1-to-1 session mapping among paired source sessions
        paired_src_sessions = (
            paired_df[src_mask & paired_df["measurement_id_target"].notna()][
                ["measurement_id", "measurement_id_target", "time_diff_hours"]
            ]
            .drop_duplicates(subset=["measurement_id"])
        )

        # No duplicate target sessions
        self.assertEqual(
            len(paired_src_sessions["measurement_id_target"]),
            paired_src_sessions["measurement_id_target"].nunique(),
            "Violation: multiple source sessions mapped to the same target session!",
        )

        # All paired differences must be <= threshold (12.0 hours)
        self.assertTrue(
            (paired_src_sessions["time_diff_hours"] <= 12.0).all(),
            "Violation: time difference exceeds threshold!",
        )

        # Verify t7_code_target matches t7_code for all paired source sessions
        if "t7_code" in paired_df.columns and "t7_code_target" in paired_df.columns:
            paired_src = paired_df[src_mask & paired_df["measurement_id_target"].notna()]
            self.assertTrue(
                (paired_src["t7_code"] == paired_src["t7_code_target"]).all(),
                "Violation: paired source and target measurements have different t7_code!",
            )

    def test_die_and_region_consistency(self) -> None:
        """Verify paired spectrum_index_target preserves exact die and region correspondence."""
        paired_df = pair_source_target_spectra(self.meta_df)
        paired_rows = paired_df[
            (paired_df["tool"] == "J4") & paired_df["spectrum_index_target"].notna()
        ]

        self.assertFalse(paired_rows.empty)

        for _, src_row in paired_rows.iterrows():
            tgt_idx = int(src_row["spectrum_index_target"])
            tgt_row = self.meta_df.loc[tgt_idx]

            self.assertEqual(tgt_row["tool"], "J5")
            self.assertEqual(tgt_row["measurement_id"], src_row["measurement_id_target"])
            self.assertEqual(tgt_row["die"], src_row["die"])
            self.assertEqual(tgt_row["region"], src_row["region"])

    def test_greedy_method(self) -> None:
        """Verify greedy pairing method produces valid 1-to-1 assignments."""
        paired_df = pair_source_target_spectra(self.meta_df, config={"method": "greedy"})
        src_mask = paired_df["tool"] == "J4"

        paired_sessions = (
            paired_df[src_mask & paired_df["measurement_id_target"].notna()][
                ["measurement_id", "measurement_id_target"]
            ]
            .drop_duplicates(subset=["measurement_id"])
        )

        self.assertEqual(
            len(paired_sessions["measurement_id_target"]),
            paired_sessions["measurement_id_target"].nunique(),
        )

    def test_zero_matches_under_tight_threshold(self) -> None:
        """Verify small threshold produces 0 pairs while maintaining tool_target for source rows."""
        paired_df = pair_source_target_spectra(
            self.meta_df,
            config={"time_threshold_hours": 0.0001},
        )
        src_mask = paired_df["tool"] == "J4"

        # tool_target must still be 'J5'
        self.assertTrue((paired_df.loc[src_mask, "tool_target"] == "J5").all())
        # All target measurement IDs should be NA
        self.assertTrue(paired_df.loc[src_mask, "measurement_id_target"].isna().all())
        self.assertTrue(paired_df.loc[src_mask, "spectrum_index_target"].isna().all())

    def test_synthetic_collision_scenario(self) -> None:
        """Verify 1-to-1 mapping when two source sessions compete for the same closest target."""
        # Create 2 source measurements and 1 target measurement
        t0 = datetime(2026, 1, 1, 10, 0, 0)
        t_src1 = t0  # diff = 2h to tgt
        t_src2 = t0 + timedelta(hours=1)  # diff = 1h to tgt (closer)
        t_tgt = t0 + timedelta(hours=2)

        data = [
            # Source 1
            {
                "spectrum_index": 0,
                "tool": "J4",
                "measurement_id": "M_J4_01",
                "time": t_src1,
                "die": 0,
                "region": "Al2p",
            },
            # Source 2
            {
                "spectrum_index": 1,
                "tool": "J4",
                "measurement_id": "M_J4_02",
                "time": t_src2,
                "die": 0,
                "region": "Al2p",
            },
            # Target
            {
                "spectrum_index": 2,
                "tool": "J5",
                "measurement_id": "M_J5_01",
                "time": t_tgt,
                "die": 0,
                "region": "Al2p",
            },
        ]
        df = pd.DataFrame(data)

        # Optimal pairing
        paired_opt = pair_source_target_spectra(df, config={"time_threshold_hours": 5.0})
        assigned_targets = (
            paired_opt[paired_opt["tool"] == "J4"]["measurement_id_target"].dropna().tolist()
        )

        # Target can only be assigned ONCE (1-to-1)
        self.assertEqual(len(assigned_targets), 1)
        self.assertEqual(assigned_targets[0], "M_J5_01")

        # The other source row remains unassigned but retains tool_target = 'J5'
        unassigned_row = paired_opt[
            (paired_opt["tool"] == "J4") & paired_opt["measurement_id_target"].isna()
        ]
        self.assertEqual(len(unassigned_row), 1)
        self.assertEqual(unassigned_row["tool_target"].iloc[0], "J5")

    def test_invalid_configs_raise(self) -> None:
        """Verify validation errors for bad arguments."""
        with self.assertRaises(ValueError):
            pair_source_target_spectra(self.meta_df, config={"source_tool": "J4", "target_tool": "J4"})
        with self.assertRaises(ValueError):
            pair_source_target_spectra(self.meta_df, config={"time_threshold_hours": -2.0})
        with self.assertRaises(ValueError):
            pair_source_target_spectra(self.meta_df, config={"method": "invalid_method"})

    def test_missing_required_columns_raises(self) -> None:
        """Verify KeyError when missing required metadata columns."""
        bad_df = self.meta_df.drop(columns=["tool"])
        with self.assertRaises(KeyError):
            pair_source_target_spectra(bad_df)

    def test_filtered_non_range_index_preserves_alignment(self) -> None:
        """Verify pairing works correctly on sliced/filtered DataFrames with non-RangeIndex."""
        # Sliced DataFrame with arbitrary indices
        sliced_df = self.meta_df.iloc[15:45].copy()
        paired_df = pair_source_target_spectra(sliced_df)

        self.assertEqual(len(paired_df), len(sliced_df))
        self.assertListEqual(list(paired_df.index), list(sliced_df.index))
        # Ensure source rows have tool_target populated without NaN index mismatch
        src_mask = paired_df["tool"] == "J4"
        if src_mask.any():
            self.assertTrue((paired_df.loc[src_mask, "tool_target"] == "J5").all())

    def test_multi_material_isolation(self) -> None:
        """Verify measurements are only paired within the same material."""
        t0 = datetime(2026, 1, 1, 10, 0, 0)
        data = [
            # Material A on J4
            {
                "spectrum_index": 0,
                "material": "MaterialA",
                "tool": "J4",
                "measurement_id": "M_J4_A",
                "time": t0,
                "die": 0,
                "region": "Reg1",
            },
            # Material B on J5 (closer in time to MaterialA, diff = 1h)
            {
                "spectrum_index": 1,
                "material": "MaterialB",
                "tool": "J5",
                "measurement_id": "M_J5_B",
                "time": t0 + timedelta(hours=1),
                "die": 0,
                "region": "Reg1",
            },
            # Material A on J5 (further in time to MaterialA, diff = 3h)
            {
                "spectrum_index": 2,
                "material": "MaterialA",
                "tool": "J5",
                "measurement_id": "M_J5_A",
                "time": t0 + timedelta(hours=3),
                "die": 0,
                "region": "Reg1",
            },
        ]
        df = pd.DataFrame(data)
        paired = pair_source_target_spectra(df, config={"time_threshold_hours": 10.0})

        # M_J4_A should pair with M_J5_A (same material), NOT M_J5_B (different material)
        src_row = paired[paired["measurement_id"] == "M_J4_A"].iloc[0]
        self.assertEqual(src_row["measurement_id_target"], "M_J5_A")
        self.assertEqual(src_row["spectrum_index_target"], 2)

    def test_t7_code_isolation(self) -> None:
        """Verify measurements are only paired with target sessions sharing the same t7_code."""
        t0 = datetime(2026, 1, 1, 10, 0, 0)
        data = [
            # T7_001 on J4
            {
                "spectrum_index": 0,
                "material": "MaterialA",
                "t7_code": "T7_001",
                "tool": "J4",
                "measurement_id": "M_J4_1",
                "time": t0,
                "die": 0,
                "region": "Reg1",
            },
            # T7_002 on J5 (closer in time to M_J4_1 with diff = 1h, but different t7_code)
            {
                "spectrum_index": 1,
                "material": "MaterialA",
                "t7_code": "T7_002",
                "tool": "J5",
                "measurement_id": "M_J5_2",
                "time": t0 + timedelta(hours=1),
                "die": 0,
                "region": "Reg1",
            },
            # T7_001 on J5 (further in time with diff = 3h, but matching t7_code)
            {
                "spectrum_index": 2,
                "material": "MaterialA",
                "t7_code": "T7_001",
                "tool": "J5",
                "measurement_id": "M_J5_1",
                "time": t0 + timedelta(hours=3),
                "die": 0,
                "region": "Reg1",
            },
        ]
        df = pd.DataFrame(data)
        paired = pair_source_target_spectra(df, config={"time_threshold_hours": 10.0})

        # M_J4_1 must pair with M_J5_1 (same t7_code), NOT M_J5_2 (different t7_code despite closer time)
        src_row = paired[paired["measurement_id"] == "M_J4_1"].iloc[0]
        self.assertEqual(src_row["measurement_id_target"], "M_J5_1")
        self.assertEqual(src_row["spectrum_index_target"], 2)
        self.assertEqual(src_row["t7_code_target"], "T7_001")

        # When match_t7_code is disabled, closer session M_J5_2 is selected
        paired_unconstrained = pair_source_target_spectra(
            df, config={"time_threshold_hours": 10.0, "match_t7_code": False}
        )
        src_unconstrained = paired_unconstrained[
            paired_unconstrained["measurement_id"] == "M_J4_1"
        ].iloc[0]
        self.assertEqual(src_unconstrained["measurement_id_target"], "M_J5_2")

    def test_timezone_aware_timestamps(self) -> None:
        """Verify pairing and timeline visualization handle timezone-aware timestamps."""
        data = [
            {
                "spectrum_index": 0,
                "tool": "J4",
                "measurement_id": "M1",
                "time": "2026-01-01 10:00:00+08:00",
                "die": 0,
                "region": "Al2p",
            },
            {
                "spectrum_index": 1,
                "tool": "J5",
                "measurement_id": "M2",
                "time": "2026-01-01 02:00:00+00:00",  # Same instant in UTC
                "die": 0,
                "region": "Al2p",
            },
        ]
        df = pd.DataFrame(data)
        paired = pair_source_target_spectra(df)
        src_row = paired[paired["tool"] == "J4"].iloc[0]
        self.assertEqual(src_row["measurement_id_target"], "M2")
        self.assertAlmostEqual(src_row["time_diff_hours"], 0.0, places=3)

        # Ensure plot also works with tz-aware timestamps
        fig, ax = plot_pairing_timeline(paired, plot_config={"show": False})
        self.assertIsNotNone(fig)


class TestPairingVisualization(unittest.TestCase):
    """Test suite for pairing timeline visualization."""

    @classmethod
    def setUpClass(cls) -> None:
        config = {
            "measurements_per_tool": {"J4": 4, "J5": 4},
            "n_die": 2,
            "n_points": 30,
            "seed": 42,
        }
        cls.ary_int, cls.ary_ene, meta = generate_pseudo_measurements(config)
        cls.paired_df = pair_source_target_spectra(meta)

    def test_plot_pairing_timeline(self) -> None:
        """Verify timeline plot renders valid figure and axes."""
        fig, ax = plot_pairing_timeline(self.paired_df, plot_config={"show": False})
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertEqual(len(ax.get_yticklabels()), 2)

    def test_plot_pairing_timeline_missing_cols_raises(self) -> None:
        """Verify KeyError if required columns missing."""
        with self.assertRaises(KeyError):
            plot_pairing_timeline(pd.DataFrame({"dummy": [1, 2]}))


if __name__ == "__main__":
    unittest.main()
