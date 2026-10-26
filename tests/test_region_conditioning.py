"""Unit tests for strict FiLM region-index conditioning."""

from __future__ import annotations

import unittest

import torch

from models.film import FiLMGenerator, prepare_film_condition_tensor


def _meta(region_value: object, key: str = "region_index") -> list[dict]:
    """Build a one-sample metadata list with the given region value."""
    return [
        {
            "die": 0,
            "source_tool_index": 0,
            "target_tool_index": 1,
            "total_flux_single_die_standardized": 0.0,
            "time_standardized": 0.0,
            key: region_value,
        }
    ]


class TestPrepareRegionCondition(unittest.TestCase):
    """Region parsing in prepare_film_condition_tensor."""

    def test_missing_region_index_raises(self) -> None:
        """A null region_index must raise instead of defaulting to 0."""
        with self.assertRaisesRegex(ValueError, "missing or null"):
            prepare_film_condition_tensor(
                _meta(None), config={"encode_region": True}
            )

    def test_absent_region_columns_raise(self) -> None:
        """Metadata lacking both region keys must raise."""
        meta = _meta(0)
        del meta[0]["region_index"]
        with self.assertRaisesRegex(ValueError, "missing or null"):
            prepare_film_condition_tensor(meta, config={"encode_region": True})

    def test_unmapped_region_name_raises(self) -> None:
        """A region name absent from the mapping must raise."""
        with self.assertRaisesRegex(ValueError, "Invalid region value 'Xx9z'"):
            prepare_film_condition_tensor(
                _meta("Xx9z", key="region"),
                config={
                    "encode_region": True,
                    "region_index_mapping": {"Al2p": 0},
                },
            )

    def test_negative_region_index_raises(self) -> None:
        """A negative region index must raise instead of clamping."""
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            prepare_film_condition_tensor(
                _meta(-1), config={"encode_region": True}
            )

    def test_region_name_mapped(self) -> None:
        """A region name is mapped through region_index_mapping."""
        cond = prepare_film_condition_tensor(
            _meta("Ti2p", key="region"),
            config={
                "encode_region": True,
                "region_index_mapping": {"Al2p": 0, "Ti2p": 1},
            },
        )
        self.assertEqual(cond["region"].tolist(), [1])

    def test_region_skipped_when_not_encoded(self) -> None:
        """With encode_region=False the region is neither parsed nor packed."""
        cond = prepare_film_condition_tensor(
            _meta("Xx9z", key="region"), config={"encode_region": False}
        )
        self.assertNotIn("region", cond)


class TestFiLMGeneratorRegion(unittest.TestCase):
    """Region handling in FiLMGenerator.forward."""

    def setUp(self) -> None:
        self.film_gen = FiLMGenerator(
            channels_list=[8],
            config={"encode_region": True, "num_regions": 4},
        )
        self.cond = prepare_film_condition_tensor(
            _meta(1), config={"encode_region": True}
        )

    def test_missing_region_key_raises(self) -> None:
        """A missing 'region' entry must raise KeyError."""
        cond = {k: v for k, v in self.cond.items() if k != "region"}
        with self.assertRaisesRegex(KeyError, "expects 'region'"):
            self.film_gen(cond)

    def test_out_of_range_region_raises(self) -> None:
        """An index beyond num_regions must raise instead of clamping."""
        cond = {**self.cond, "region": torch.tensor([4])}
        with self.assertRaisesRegex(ValueError, r"outside the embedding range"):
            self.film_gen(cond)

    def test_out_of_range_2d_region_raises(self) -> None:
        """A 2D region tensor beyond num_regions must raise without TypeError."""
        cond = {**self.cond, "region": torch.tensor([[4]])}
        with self.assertRaisesRegex(ValueError, r"outside the embedding range"):
            self.film_gen(cond)

    def test_valid_region_runs(self) -> None:
        """A valid region index produces modulations of the right shape."""
        gamma, beta = self.film_gen(self.cond)[0]
        self.assertEqual(tuple(gamma.shape), (1, 8))
        self.assertEqual(tuple(beta.shape), (1, 8))


if __name__ == "__main__":
    unittest.main()
