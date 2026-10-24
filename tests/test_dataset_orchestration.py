"""Unit tests for dataset orchestration module."""

import unittest
import numpy as np
import pandas as pd

from models.dataset_orchestration import (
    create_regional_datasets,
    create_unified_dataset,
)


class TestDatasetOrchestration(unittest.TestCase):
    """Test suite for regional and unified dataset orchestration."""

    def setUp(self) -> None:
        """Create synthetic spectral datasets and metadata."""
        n_spectra = 24
        n_points = 50
        self.n_points = n_points
        self.ary_intensity = np.random.uniform(
            0.1, 1.0, (n_spectra, n_points)
        ).astype(np.float32)
        self.ary_energy = np.tile(
            np.linspace(100.0, 150.0, n_points, dtype=np.float32),
            (n_spectra, 1),
        )

        records = []
        n_half = n_spectra // 2
        for i in range(n_half):
            split = "train" if i < 6 else ("val" if i < 9 else "test")
            records.append(
                {
                    "spectrum_index": i,
                    "measurement_id": f"M_src_{i // 2}",
                    "tool": "J4",
                    "source_tool_index": 0,
                    "target_tool": "J5",
                    "target_tool_index": 1,
                    "target_measurement_id": f"M_tgt_{i // 2}",
                    "target_spectrum_index": i + n_half,
                    "die": i % 3,
                    "region": "Al2p" if i % 2 == 0 else "Ti2p",
                    "region_index": 0 if i % 2 == 0 else 1,
                    "split": split,
                    "total_flux_single_die_standardized": 0.0,
                    "time_standardized": 0.0,
                    "material": "NMG",
                }
            )
        for i in range(n_half, n_spectra):
            src_i = i - n_half
            split = "train" if src_i < 6 else ("val" if src_i < 9 else "test")
            records.append(
                {
                    "spectrum_index": i,
                    "measurement_id": f"M_tgt_{src_i // 2}",
                    "tool": "J5",
                    "source_tool_index": 1,
                    "die": src_i % 3,
                    "region": "Al2p" if src_i % 2 == 0 else "Ti2p",
                    "region_index": 0 if src_i % 2 == 0 else 1,
                    "split": split,
                    "total_flux_single_die_standardized": 0.0,
                    "time_standardized": 0.0,
                    "material": "NMG",
                }
            )
        self.df_meta = pd.DataFrame(records)

    def test_create_regional_datasets_full_spectrum(self) -> None:
        """Verify regional datasets creation without sliding windows."""
        res = create_regional_datasets(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={
                "source_tool": "J4",
                "target_tool": "J5",
                "use_sliding_window": False,
            },
        )
        self.assertIn("Al2p", res)
        self.assertIn("Ti2p", res)
        for reg in ["Al2p", "Ti2p"]:
            bundle = res[reg]
            self.assertEqual(bundle["full_spectrum_length"], self.n_points)
            self.assertEqual(bundle["window_size_points"], self.n_points)
            self.assertGreater(len(bundle["dataset_train"]), 0)
            self.assertGreater(len(bundle["dataset_val"]), 0)

    def test_create_regional_datasets_with_sliding_window(self) -> None:
        """Verify regional datasets creation with sliding windows."""
        res = create_regional_datasets(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={
                "source_tool": "J4",
                "target_tool": "J5",
                "use_sliding_window": True,
                "window_size_ev": 10.0,
                "sliding_stride_ev": 5.0,
            },
        )
        for reg in ["Al2p", "Ti2p"]:
            bundle = res[reg]
            self.assertLess(bundle["window_size_points"], self.n_points)
            self.assertGreater(len(bundle["dataset_train"]), 0)

    def test_create_unified_dataset(self) -> None:
        """Verify unified multi-region dataset construction."""
        bundle = create_unified_dataset(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={
                "source_tool": "J4",
                "target_tool": "J5",
                "use_sliding_window": False,
            },
        )
        self.assertIn("dataset_train", bundle)
        self.assertIn("dataset_val", bundle)
        self.assertIn("dataset_test", bundle)
        self.assertEqual(bundle["full_spectrum_length"], self.n_points)
        self.assertEqual(len(bundle["regions"]), 2)


if __name__ == "__main__":
    unittest.main()
