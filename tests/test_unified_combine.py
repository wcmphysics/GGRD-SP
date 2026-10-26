"""Unit tests for merging regional bundles into one unified bundle."""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import torch

import models
from models.dataset import SpectrumPairDataset, SpectrumPatchDataset
from models.orchestration import _combine_regional_bundles
try:
    from test_unified_model import _create_mock_spectral_data
except ModuleNotFoundError:
    from tests.test_unified_model import _create_mock_spectral_data


def _regional(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build regional bundles from the shared mock data."""
    ary_intensity, ary_energy, df_meta = _create_mock_spectral_data()
    return models.create_regional_datasets(
        df_meta,
        ary_intensity,
        ary_energy,
        config={"source_tool": "J4", "target_tool": "J5", **config},
    )


class TestCombineRegionalBundles(unittest.TestCase):
    """Shared-window enforcement and patch merging."""

    def test_merges_per_region_patches(self) -> None:
        """The unified train set is exactly the per-region patches merged."""
        regional = _regional(
            {"use_sliding_window": True, "window_size": 11, "stride": 5}
        )
        unified = _combine_regional_bundles(regional)
        train_ds = unified["dataset_train"]
        self.assertIsInstance(train_ds, SpectrumPatchDataset)
        self.assertEqual((train_ds.window_size, train_ds.stride), (11, 5))
        self.assertEqual(unified["window_size_points"], 11)
        self.assertEqual(unified["sliding_stride_points"], 5)

        parts = [b["dataset_train"] for b in regional.values()]
        torch.testing.assert_close(train_ds.x, torch.cat([p.x for p in parts]))
        torch.testing.assert_close(train_ds.y, torch.cat([p.y for p in parts]))
        expected_regions = [m["region"] for p in parts for m in p.meta_list]
        self.assertEqual(
            [m["region"] for m in train_ds.meta_list], expected_regions
        )
        self.assertEqual(len(train_ds.meta_list), len(train_ds))

    def test_full_spectrum_mode_unchanged(self) -> None:
        """Without sliding window the merged full-spectrum set is used."""
        unified = _combine_regional_bundles(
            _regional({"use_sliding_window": False})
        )
        self.assertIsInstance(unified["dataset_train"], SpectrumPairDataset)
        self.assertEqual(unified["window_size_points"], 40)
        self.assertEqual(unified["full_spectrum_length"], 40)

    def test_differing_window_raises(self) -> None:
        """Per-region window overrides must raise and name the regions."""
        regional = _regional(
            {
                "use_sliding_window": True,
                "window_size": 11,
                "stride": 5,
                "region_configs": {"Ti2p": {"window_size": 15}},
            }
        )
        with self.assertRaisesRegex(ValueError, "sliding window size") as ctx:
            _combine_regional_bundles(regional)
        self.assertIn("'Ti2p': 15", str(ctx.exception))

    def test_differing_stride_raises(self) -> None:
        """Per-region stride overrides must raise."""
        regional = _regional(
            {
                "use_sliding_window": True,
                "window_size": 11,
                "stride": 5,
                "region_configs": {"O1s": {"stride": 3}},
            }
        )
        with self.assertRaisesRegex(ValueError, "sliding stride"):
            _combine_regional_bundles(regional)

    def test_mixed_sliding_window_raises(self) -> None:
        """Sliding window on for some regions and off for others must raise."""
        regional = _regional(
            {
                "use_sliding_window": True,
                "window_size": 11,
                "stride": 5,
                "region_configs": {"Ti2p": {"use_sliding_window": False}},
            }
        )
        with self.assertRaisesRegex(ValueError, r"off for \['Ti2p'\]"):
            _combine_regional_bundles(regional)

    def test_differing_n_points_raises(self) -> None:
        """Regions with different spectrum lengths must raise clearly."""

        def _bundle(n_points: int) -> dict[str, Any]:
            data = np.ones((2, n_points), dtype=np.float32)
            pair = SpectrumPairDataset(data, data)
            return {"dataset_train": pair, "dataset_val": pair}

        with self.assertRaisesRegex(ValueError, "spectrum length"):
            _combine_regional_bundles({"Al2p": _bundle(40), "Ti2p": _bundle(50)})

    def test_empty_regional_datasets_raises(self) -> None:
        """Combining an empty dict of regional datasets must raise."""
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            _combine_regional_bundles({})


class TestPatchConcatenate(unittest.TestCase):
    """SpectrumPatchDataset.concatenate validation."""

    def _patches(self, window: int, stride: int) -> SpectrumPatchDataset:
        data = np.arange(60, dtype=np.float32).reshape(2, 30)
        return SpectrumPatchDataset(
            SpectrumPairDataset(data, data), window_size=window, stride=stride
        )

    def test_empty_list_raises(self) -> None:
        """Concatenating nothing must raise."""
        with self.assertRaisesRegex(ValueError, "empty list"):
            SpectrumPatchDataset.concatenate([])

    def test_geometry_mismatch_raises(self) -> None:
        """Datasets with different windows must not be merged."""
        with self.assertRaisesRegex(ValueError, "must share"):
            SpectrumPatchDataset.concatenate(
                [self._patches(5, 2), self._patches(7, 2)]
            )

    def test_lengths_add_up(self) -> None:
        """The merged length is the sum of the parts."""
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
