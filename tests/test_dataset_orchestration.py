"""Unit tests for dataset orchestration module."""

import unittest
import numpy as np
import pandas as pd

from models.dataset import (
    SpectrumPairDataset,
    SpectrumPatchDataset,
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


class TestPatchConcatenate(unittest.TestCase):
    """Test suite for SpectrumPatchDataset.concatenate operations."""

    def _patches(self, window: int, stride: int) -> SpectrumPatchDataset:
        data = np.arange(60, dtype=np.float32).reshape(2, 30)
        return SpectrumPatchDataset(
            SpectrumPairDataset(data, data), window_size=window, stride=stride
        )

    def test_empty_list_raises(self) -> None:
        """Concatenating empty list of datasets must raise ValueError."""
        with self.assertRaisesRegex(ValueError, "empty list"):
            SpectrumPatchDataset.concatenate([])

    def test_geometry_mismatch_raises(self) -> None:
        """Datasets with different windows must not be merged."""
        with self.assertRaisesRegex(ValueError, "must share"):
            SpectrumPatchDataset.concatenate(
                [self._patches(5, 2), self._patches(7, 2)]
            )

    def test_lengths_add_up(self) -> None:
        """The merged length is the sum of component dataset lengths."""
        first, second = self._patches(5, 2), self._patches(5, 2)
        merged = SpectrumPatchDataset.concatenate([first, second])
        self.assertEqual(len(merged), len(first) + len(second))
        self.assertEqual(merged[len(first)]["x"].shape[-1], 5)

    def test_metadata_presence_mismatch_raises(self) -> None:
        """Concatenating with mismatched metadata presence must raise."""
        ds_no_meta = self._patches(5, 2)
        data = np.arange(60, dtype=np.float32).reshape(2, 30)
        pair_with_meta = SpectrumPairDataset(
            data, data, metadata=[{"region": "Al2p"}, {"region": "Ti2p"}]
        )
        ds_with_meta = SpectrumPatchDataset(
            pair_with_meta, window_size=5, stride=2
        )
        with self.assertRaisesRegex(ValueError, "per-patch metadata"):
            SpectrumPatchDataset.concatenate([ds_no_meta, ds_with_meta])


if __name__ == "__main__":
    unittest.main()
