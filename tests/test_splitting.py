"""Unit tests for Stage 4 measurement session splitting."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
