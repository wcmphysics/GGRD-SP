"""Unit tests for strict FiLM region-index conditioning."""

from __future__ import annotations

import unittest
import torch

from models.film import FiLMGenerator, prepare_film_condition_tensor

_BASE_CFG = {
    "num_dies": 3,
    "num_tools": 2,
    "num_regions": 4,
    "num_elements": 2,
    "encode_die": False,
    "encode_tool": False,
    "encode_flux": False,
    "encode_time": False,
    "encode_energy": False,
    "encode_region": True,
    "encode_relative_intensity": False,
    "encode_elements": False,
    "hidden_dim": 16,
}


def _meta(region_vec: object) -> list[dict]:
    """Build a one-sample metadata list with the given region one-hot value."""
    return [{"region_one_hot": region_vec}]


class TestPrepareRegionCondition(unittest.TestCase):
    """Region parsing in prepare_film_condition_tensor."""

    def test_missing_region_one_hot_raises(self) -> None:
        """A null region_one_hot must raise."""
        with self.assertRaises(ValueError):
            prepare_film_condition_tensor(_meta(None), config=_BASE_CFG)

    def test_absent_region_columns_raise(self) -> None:
        """Metadata lacking region_one_hot must raise."""
        meta = [{}]
        with self.assertRaises(ValueError):
            prepare_film_condition_tensor(meta, config=_BASE_CFG)

    def test_region_vector_mapped(self) -> None:
        """A region_one_hot vector is extracted into torch.float32 tensor."""
        cond = prepare_film_condition_tensor(
            _meta([0.0, 1.0, 0.0, 0.0]),
            config=_BASE_CFG,
        )
        self.assertEqual(cond["region_one_hot"].tolist(), [[0.0, 1.0, 0.0, 0.0]])

    def test_region_skipped_when_not_encoded(self) -> None:
        """With encode_region=False the region is neither parsed nor packed."""
        cfg = {**_BASE_CFG, "encode_region": False, "encode_flux": True}
        meta = [{"total_flux_single_die_standardized": 0.5}]
        cond = prepare_film_condition_tensor(meta, config=cfg)
        self.assertNotIn("region_one_hot", cond)


class TestFiLMGeneratorRegion(unittest.TestCase):
    """Region handling in FiLMGenerator.forward."""

    def setUp(self) -> None:
        self.film_gen = FiLMGenerator(
            channels_list=[8],
            config=_BASE_CFG,
        )
        self.cond = prepare_film_condition_tensor(
            _meta([0.0, 1.0, 0.0, 0.0]), config=_BASE_CFG
        )

    def test_missing_region_key_raises(self) -> None:
        """A missing 'region_one_hot' entry must raise KeyError."""
        cond = {k: v for k, v in self.cond.items() if k != "region_one_hot"}
        with self.assertRaisesRegex(KeyError, "expects 'region_one_hot'"):
            self.film_gen(cond)

    def test_forward_with_region_one_hot(self) -> None:
        """Modulations compute correctly with region one-hot."""
        mods = self.film_gen(self.cond)
        self.assertEqual(len(mods), 1)
        gamma, beta = mods[0]
        self.assertEqual(gamma.shape, (1, 8))
        self.assertEqual(beta.shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
