"""Unit tests for utility.preprocessing module."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from utility.preprocessing import preprocess


class TestPreprocessing(unittest.TestCase):
    """Test cases for the preprocess utility function."""

    def setUp(self) -> None:
        """Create synthetic spectrum data and metadata."""
        self.n_samples = 4
        self.n_points = 50
        self.ary_energy = np.tile(
            np.linspace(100.0, 150.0, self.n_points, dtype=np.float32),
            (self.n_samples, 1),
        )
        self.ary_intensity = np.ones(
            (self.n_samples, self.n_points), dtype=np.float32
        ) * 10.0

        self.df_meta = pd.DataFrame([
            {
                "spectrum_index": 0,
                "tool": "J4",
                "target_tool": "J5",
                "measurement_id": "M1",
                "die": 0,
                "region": "Al2p",
                "time": "2026-01-01T00:00:00",
                "split": "train",
            },
            {
                "spectrum_index": 1,
                "tool": "J4",
                "target_tool": "J5",
                "measurement_id": "M1",
                "die": 0,
                "region": "Ti2p",
                "time": "2026-01-01T00:00:00",
                "split": "train",
            },
            {
                "spectrum_index": 2,
                "tool": "J4",
                "target_tool": "J5",
                "measurement_id": "M2",
                "die": 1,
                "region": "Al2p",
                "time": "2026-01-01T06:00:00",
                "split": "val",
            },
            {
                "spectrum_index": 3,
                "tool": "J4",
                "target_tool": "J5",
                "measurement_id": "M3",
                "die": 1,
                "region": "Ti2p",
                "time": "2026-01-01T12:00:00",
                "split": "test",
            },
        ])

    def test_preprocess_physical_metrics_and_mappings(self) -> None:
        """Verify physical metrics, tool mappings, and region mappings are created."""
        df_out, tool_map, reg_map = preprocess(
            self.ary_intensity, self.ary_energy, self.df_meta
        )

        # Check physical metrics
        self.assertIn("total_flux_single_region", df_out.columns)
        self.assertIn("mean_intensity_single_region", df_out.columns)
        self.assertIn("total_flux_single_die", df_out.columns)
        self.assertIn("mean_intensity_single_die", df_out.columns)

        # Check categorical tool mappings
        self.assertIn("source_tool_index", df_out.columns)
        self.assertIn("target_tool_index", df_out.columns)
        self.assertIn("J4", tool_map)
        self.assertIn("J5", tool_map)
        self.assertEqual(df_out["source_tool_index"].iloc[0], tool_map["J4"])
        self.assertEqual(df_out["target_tool_index"].iloc[0], tool_map["J5"])

        # Check region mappings
        self.assertIn("region_index", df_out.columns)
        self.assertIn("Al2p", reg_map)
        self.assertIn("Ti2p", reg_map)

    def test_preprocess_feature_standardization_on_train_split(self) -> None:
        """Verify continuous features are standardized when split is present."""
        df_out, _, _ = preprocess(
            self.ary_intensity, self.ary_energy, self.df_meta
        )
        self.assertIn("time_standardized", df_out.columns)
        self.assertIn("total_flux_single_die_standardized", df_out.columns)

    def test_preprocess_registers_into_config(self) -> None:
        """Verify config dictionaries are updated with mappings."""
        config = {
            "training": {},
            "prediction": {},
        }
        df_out, tool_map, reg_map = preprocess(
            self.ary_intensity, self.ary_energy, self.df_meta, config=config
        )
        self.assertEqual(config["training"]["num_tools"], len(tool_map))
        self.assertEqual(config["training"]["region_index_mapping"], reg_map)
        self.assertEqual(config["prediction"]["region_index_mapping"], reg_map)

    def test_preprocess_without_target_tool_or_split(self) -> None:
        """Verify robust execution when target_tool or split are absent."""
        df_minimal = self.df_meta[
            ["spectrum_index", "tool", "measurement_id", "die", "region"]
        ].copy()
        df_out, tool_map, reg_map = preprocess(
            self.ary_intensity, self.ary_energy, df_minimal
        )
        self.assertIn("source_tool_index", df_out.columns)
        self.assertNotIn("target_tool_index", df_out.columns)
        self.assertNotIn("time_standardized", df_out.columns)
        self.assertEqual(len(tool_map), 1)

    def test_preprocess_missing_required_columns(self) -> None:
        """Verify ValueError is raised when required columns are missing."""
        for col in ["spectrum_index", "measurement_id", "die", "region", "tool"]:
            df_missing = self.df_meta.drop(columns=[col])
            with self.assertRaises(ValueError) as ctx:
                preprocess(self.ary_intensity, self.ary_energy, df_missing)
            self.assertIn("missing required columns", str(ctx.exception))

    def test_preprocess_custom_fit_split(self) -> None:
        """Verify continuous feature standardization uses custom fit_split."""
        df_out, _, _ = preprocess(
            self.ary_intensity,
            self.ary_energy,
            self.df_meta,
            config={"fit_split": "val"},
        )
        self.assertIn("time_standardized", df_out.columns)
        self.assertIn("total_flux_single_die_standardized", df_out.columns)


if __name__ == "__main__":
    unittest.main()
