"""Unit tests for Stage 4 measurement session splitting."""

import unittest
import pandas as pd

from utility.splitting import split_measurement_sessions


class TestSplitting(unittest.TestCase):
    """Test suite for split_measurement_sessions."""

    def setUp(self) -> None:
        """Create paired metadata DataFrame with 6 paired source sessions and 1 unpaired."""
        records: list[dict[str, object]] = []
        for i in range(6):
            meas_src = f"SRC_{i}"
            meas_tgt = f"TGT_{i}"
            for die in range(2):
                for reg in ("Al2p", "Ti2p"):
                    records.append(
                        {
                            "spectrum_index": len(records),
                            "measurement_id": meas_src,
                            "tool": "J4",
                            "die": die,
                            "region": reg,
                            "target_measurement_id": meas_tgt,
                            "target_tool": "J5",
                        }
                    )
        # Add 1 unpaired measurement session
        for die in range(2):
            records.append(
                {
                    "spectrum_index": len(records),
                    "measurement_id": "SRC_UNPAIRED",
                    "tool": "J4",
                    "die": die,
                    "region": "Al2p",
                    "target_measurement_id": None,
                    "target_tool": None,
                }
            )
        self.df_meta = pd.DataFrame(records)

    def test_random_splitting(self) -> None:
        """Verify random splitting correctly assigns train, val, test, and none splits."""
        cfg = {
            "method": "random",
            "source_tool": "J4",
            "target_tool": "J5",
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "seed": 42,
        }
        df_split, mapping, splits = split_measurement_sessions(self.df_meta, cfg)

        self.assertIn("split", df_split.columns)
        self.assertEqual(len(mapping), 6)

        # Unpaired session must have split == 'none'
        unpaired_rows = df_split[df_split["measurement_id"] == "SRC_UNPAIRED"]
        self.assertTrue((unpaired_rows["split"] == "none").all())

        # All dies and regions of each measurement session must share the same split
        for meas_id, group in df_split.groupby("measurement_id"):
            self.assertEqual(group["split"].nunique(), 1)

        self.assertTrue(len(splits["train"]) > 0)
        self.assertTrue(len(splits["val"]) > 0)
        self.assertTrue(len(splits["test"]) > 0)

    def test_direct_assignment(self) -> None:
        """Verify direct assignment explicitly maps specified sessions."""
        cfg = {
            "method": "direct",
            "source_tool": "J4",
            "target_tool": "J5",
            "direct_assignment": {
                "train": ["SRC_0", "SRC_1", "SRC_2"],
                "val": ["SRC_3"],
                "test": ["SRC_4", "SRC_5"],
            },
        }
        df_split, mapping, splits = split_measurement_sessions(self.df_meta, cfg)
        self.assertEqual(splits["train"], {"SRC_0", "SRC_1", "SRC_2"})
        self.assertEqual(splits["val"], {"SRC_3"})
        self.assertEqual(splits["test"], {"SRC_4", "SRC_5"})

        row_0 = df_split[df_split["measurement_id"] == "SRC_0"]
        self.assertTrue((row_0["split"] == "train").all())

    def test_direct_assignment_overlap_raises(self) -> None:
        """Verify ValueError when direct assignment contains overlapping sessions."""
        cfg = {
            "method": "direct",
            "source_tool": "J4",
            "target_tool": "J5",
            "direct_assignment": {
                "train": ["SRC_0", "SRC_1"],
                "val": ["SRC_1"],  # Overlap with train
                "test": ["SRC_2", "SRC_3", "SRC_4", "SRC_5"],
            },
        }
        with self.assertRaises(ValueError):
            split_measurement_sessions(self.df_meta, cfg)

    def test_direct_assignment_incomplete_coverage_raises(self) -> None:
        """Verify ValueError when direct assignment misses paired sessions."""
        cfg = {
            "method": "direct",
            "source_tool": "J4",
            "target_tool": "J5",
            "direct_assignment": {
                "train": ["SRC_0", "SRC_1"],
                "val": ["SRC_2"],
                "test": ["SRC_3"],  # Misses SRC_4 and SRC_5
            },
        }
        with self.assertRaises(ValueError):
            split_measurement_sessions(self.df_meta, cfg)


def _build_code_name_meta(code_counts: dict[str, int]) -> pd.DataFrame:
    """Build paired + unpaired metadata with code-prefixed measurement IDs."""
    records: list[dict[str, object]] = []
    for code, count in code_counts.items():
        for i in range(count):
            meas_src = f"{code}_M_J4_{i:05d}"
            meas_tgt = f"{code}_M_J5_{i:05d}"
            for die in range(2):
                for reg in ("Al2p", "Ti2p"):
                    records.append(
                        {
                            "spectrum_index": len(records),
                            "measurement_id": meas_src,
                            "tool": "J4",
                            "die": die,
                            "region": reg,
                            "target_measurement_id": meas_tgt,
                            "target_tool": "J5",
                        }
                    )
                    records.append(
                        {
                            "spectrum_index": len(records),
                            "measurement_id": meas_tgt,
                            "tool": "J5",
                            "die": die,
                            "region": reg,
                            "target_measurement_id": None,
                            "target_tool": None,
                        }
                    )
    records.append(
        {
            "spectrum_index": len(records),
            "measurement_id": "PMGA_M_J4_UNPAIRED",
            "tool": "J4",
            "die": 0,
            "region": "Al2p",
            "target_measurement_id": None,
            "target_tool": None,
        }
    )
    return pd.DataFrame(records)


class TestCodeNameSplitting(unittest.TestCase):
    """Test suite for method='code_name' session splitting."""

    def test_pmga_train_val_and_pmgb_test(self) -> None:
        """Verify PMGA maps to train/val and PMGB maps exclusively to test."""
        df_meta = _build_code_name_meta({"PMGA": 5, "PMGB": 3})
        cfg = {
            "method": "code_name",
            "source_tool": "J4",
            "target_tool": "J5",
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "seed": 42,
            "code_name_assignment": {
                "train": ["PMGA"],
                "val": ["PMGA"],
                "test": ["PMGB"],
            },
        }
        df_split, _, splits = split_measurement_sessions(df_meta, cfg)

        self.assertTrue(all(s.startswith("PMGA_") for s in splits["train"]))
        self.assertTrue(all(s.startswith("PMGA_") for s in splits["val"]))
        self.assertEqual(len(splits["train"]), 4)
        self.assertEqual(len(splits["val"]), 1)
        self.assertEqual(
            splits["test"],
            {"PMGB_M_J4_00000", "PMGB_M_J4_00001", "PMGB_M_J4_00002"},
        )

        # Target sessions inherit source split; unpaired stays 'none'
        pmgb_tgt = df_split[df_split["measurement_id"].str.startswith("PMGB_M_J5_")]
        self.assertTrue((pmgb_tgt["split"] == "test").all())
        unpaired = df_split[df_split["measurement_id"] == "PMGA_M_J4_UNPAIRED"]
        self.assertTrue((unpaired["split"] == "none").all())

    def test_stratified_multi_code_and_zero_test_ratio(self) -> None:
        """Verify per-code stratification and test_ratio=0.0 support."""
        df_meta = _build_code_name_meta({"PMGA": 4, "NMG": 4, "PMGB": 2})
        cfg = {
            "method": "code_name",
            "source_tool": "J4",
            "target_tool": "J5",
            "train_ratio": 0.75,
            "val_ratio": 0.25,
            "test_ratio": 0.0,
            "seed": 7,
            "code_name_assignment": {
                "train": ["PMGA", "NMG"],
                "val": ["PMGA", "NMG"],
                "test": ["PMGB"],
            },
        }
        _, _, splits = split_measurement_sessions(df_meta, cfg)

        for code in ("PMGA_", "NMG_"):
            self.assertTrue(any(s.startswith(code) for s in splits["train"]))
            self.assertTrue(any(s.startswith(code) for s in splits["val"]))
        self.assertEqual(len(splits["test"]), 2)

    def test_three_way_and_non_train_two_way_splits(self) -> None:
        """Verify 3-split, 2-way val/test, and seed reproducibility."""
        df_meta = _build_code_name_meta({"PMGA": 6, "PMGB": 4})
        cfg = {
            "method": "code_name",
            "source_tool": "J4",
            "target_tool": "J5",
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "seed": 123,
            "code_name_assignment": {
                "train": ["PMGA"],
                "val": ["PMGA", "PMGB"],
                "test": ["PMGA", "PMGB"],
            },
        }
        _, _, splits_a = split_measurement_sessions(df_meta, cfg)
        _, _, splits_b = split_measurement_sessions(df_meta, cfg)
        self.assertEqual(splits_a, splits_b)
        self.assertEqual(len(splits_a["train"] & {f"PMGB_M_J4_{i:05d}" for i in range(4)}), 0)
        self.assertTrue(any(s.startswith("PMGB_") for s in splits_a["val"]))
        self.assertTrue(any(s.startswith("PMGB_") for s in splits_a["test"]))

    def test_prefix_safety_distinct_tokens(self) -> None:
        """Verify PMG and PMGA are treated as distinct exact tokens."""
        df_meta = _build_code_name_meta({"PMG": 2, "PMGA": 2})
        cfg = {
            "method": "code_name",
            "source_tool": "J4",
            "target_tool": "J5",
            "code_name_assignment": {"train": ["PMG"], "test": ["PMGA"]},
        }
        _, _, splits = split_measurement_sessions(df_meta, cfg)
        self.assertEqual(splits["train"], {"PMG_M_J4_00000", "PMG_M_J4_00001"})
        self.assertEqual(splits["test"], {"PMGA_M_J4_00000", "PMGA_M_J4_00001"})

    def test_code_name_error_cases(self) -> None:
        """Verify ValueError on unassigned, unknown, or invalid configs."""
        df_meta = _build_code_name_meta({"PMGA": 3, "PMGB": 2})

        # Unassigned code in data (PMGB missing)
        with self.assertRaisesRegex(ValueError, "unassigned code name"):
            split_measurement_sessions(
                df_meta,
                {
                    "method": "code_name",
                    "source_tool": "J4",
                    "target_tool": "J5",
                    "code_name_assignment": {"train": ["PMGA"], "test": ["PMGA"]},
                },
            )

        # Missing train or eval split in code_name_assignment
        with self.assertRaisesRegex(ValueError, "in 'val' or 'test'"):
            split_measurement_sessions(
                df_meta,
                {
                    "method": "code_name",
                    "source_tool": "J4",
                    "target_tool": "J5",
                    "code_name_assignment": {"train": ["PMGA", "PMGB"]},
                },
            )

        # Unknown code in config (PMGC not in data)
        with self.assertRaisesRegex(ValueError, "unknown code name"):
            split_measurement_sessions(
                df_meta,
                {
                    "method": "code_name",
                    "source_tool": "J4",
                    "target_tool": "J5",
                    "code_name_assignment": {"train": ["PMGA"], "test": ["PMGB", "PMGC"]},
                },
            )

        # Insufficient sessions for multi-split code
        df_small = _build_code_name_meta({"PMGA": 1, "PMGB": 2})
        with self.assertRaisesRegex(ValueError, "need >= 2"):
            split_measurement_sessions(
                df_small,
                {
                    "method": "code_name",
                    "source_tool": "J4",
                    "target_tool": "J5",
                    "code_name_assignment": {"train": ["PMGA"], "val": ["PMGA"], "test": ["PMGB"]},
                },
            )

        # Zero ratio on active multi-split code
        with self.assertRaisesRegex(ValueError, "positive ratio"):
            split_measurement_sessions(
                df_meta,
                {
                    "method": "code_name",
                    "source_tool": "J4",
                    "target_tool": "J5",
                    "train_ratio": 0.7,
                    "val_ratio": 0.0,
                    "test_ratio": 0.3,
                    "code_name_assignment": {"train": ["PMGA"], "val": ["PMGA"], "test": ["PMGB"]},
                },
            )


if __name__ == "__main__":
    unittest.main()
