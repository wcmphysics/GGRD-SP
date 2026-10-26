"""Unit tests for Feature-wise Linear Modulation (FiLM) in models/film.py."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
import torch

from models.film import FiLMGenerator, prepare_film_condition_tensor


class TestFiLMModulation(unittest.TestCase):
    """Test suite verifying FiLM condition preparation and modulation generator.
    """

    def setUp(self) -> None:
        self.materials_cfg = {
            "NMG": {"elements": ["Ti", "Al", "C", "O", "Cl", "Si"]},
            "PMG": {"elements": ["Ti", "N", "O", "Cl", "Si"]},
        }
        self.sample_meta = [
            {
                "die": 0,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": -0.5,
                "time_standardized": -1.0,
                "region_index": 2,
                "relative_intensity_single_region": 0.85,
                "material": "NMG",
            },
            {
                "die": 5,
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": 1.25,
                "time_standardized": 1.0,
                "region_index": 11,
                "relative_intensity_single_region": 1.30,
                "material": "PMG",
            },
        ]

    def test_prepare_film_condition_tensor_formats_and_features(self) -> None:
        """Verify feature extraction across dict, list, and DataFrame inputs."""
        # Test with list of dicts and energy bounds
        we = np.array([[100.0, 106.0], [105.0, 111.0]])
        cfg = {
            "num_samples": 2,
            "encode_region": True,
            "encode_relative_intensity": True,
            "encode_elements": True,
            "material_elements": self.materials_cfg,
            "energy_scale_ev": 1000.0,
        }
        cond = prepare_film_condition_tensor(
            metadata=self.sample_meta,
            energy_bounds=we,
            config=cfg,
        )

        self.assertEqual(cond["die"].tolist(), [0, 5])
        self.assertEqual(cond["source_tool_index"].tolist(), [0, 0])
        self.assertEqual(cond["target_tool_index"].tolist(), [1, 1])
        self.assertEqual(cond["region"].tolist(), [2, 11])
        self.assertEqual(cond["total_flux_single_die_standardized"].shape, (2, 1))
        self.assertEqual(cond["time_standardized"].shape, (2, 1))
        self.assertEqual(cond["relative_intensity_single_region"].shape, (2, 1))
        self.assertEqual(cond["energy_bounds"].shape, (2, 2))
        self.assertAlmostEqual(cond["energy_bounds"][0, 0].item(), 0.1, places=4)
        self.assertEqual(cond["element_multi_hot"].shape, (2, 7))

        # Test DataFrame and column mapping
        df_meta = pd.DataFrame(self.sample_meta)
        cond_df = prepare_film_condition_tensor(metadata=df_meta, config=cfg)
        self.assertEqual(cond_df["die"].tolist(), [0, 5])

        # Test scalar/dict input format
        cond_dict = prepare_film_condition_tensor(
            metadata={
                "die": [0, 1],
                "source_tool_index": 0,
                "target_tool_index": 1,
                "total_flux_single_die_standardized": 0.0,
                "time_standardized": 0.0,
            },
            config={"num_samples": 2},
        )
        self.assertEqual(len(cond_dict["die"]), 2)
        self.assertEqual(len(cond_dict["source_tool_index"]), 2)

    def test_missing_metadata_fields_raise(self) -> None:
        """Verify missing mandatory metadata keys or elements config raise
        error.
        """
        # Missing total_flux_single_die_standardized
        bad_meta = [{"die": 0, "source_tool_index": 0, "target_tool_index": 1}]
        with self.assertRaises(ValueError):
            prepare_film_condition_tensor(bad_meta)

        # Missing materials configuration when encode_elements is True
        with self.assertRaises(ValueError):
            prepare_film_condition_tensor(
                self.sample_meta, config={"encode_elements": True}
            )

    def test_film_generator_identity_initialization(self) -> None:
        """Verify FiLMGenerator outputs gamma=1.0 and beta=0.0 initially."""
        film_gen = FiLMGenerator(channels_list=[16, 32])
        cond = prepare_film_condition_tensor(metadata=self.sample_meta[:1])

        mods = film_gen(cond)
        self.assertEqual(len(mods), 2)
        gamma1, beta1 = mods[0]
        gamma2, beta2 = mods[1]

        self.assertEqual(gamma1.shape, (1, 16))
        self.assertEqual(beta1.shape, (1, 16))
        self.assertEqual(gamma2.shape, (1, 32))
        self.assertEqual(beta2.shape, (1, 32))

        torch.testing.assert_close(gamma1, torch.ones_like(gamma1))
        torch.testing.assert_close(beta1, torch.zeros_like(beta1))
        torch.testing.assert_close(gamma2, torch.ones_like(gamma2))
        torch.testing.assert_close(beta2, torch.zeros_like(beta2))

    def test_film_generator_feature_toggles_and_dimensions(self) -> None:
        """Verify feature toggle permutations compute correct input dimension.
        """
        base_cond = prepare_film_condition_tensor(
            metadata=self.sample_meta,
            energy_bounds=np.array([[100.0, 106.0], [105.0, 111.0]]),
            config={
                "encode_region": True,
                "encode_relative_intensity": True,
                "encode_elements": True,
                "material_elements": self.materials_cfg,
            },
        )

        test_cases = [
            # die(4) + tools(2*4) + flux(1) + time(1) = 14
            ({"encode_die": True, "encode_tool": True, "encode_flux": True, "encode_time": True}, 14),
            # region(4) + rel_intensity(1) + energy(2) = 7
            ({
                "encode_die": False,
                "encode_tool": False,
                "encode_flux": False,
                "encode_time": False,
                "encode_region": True,
                "num_regions": 16,
                "region_embed_dim": 4,
                "encode_relative_intensity": True,
                "encode_energy": True,
            }, 7),
            # elements(7) only = 7
            ({
                "encode_die": False,
                "encode_tool": False,
                "encode_flux": False,
                "encode_time": False,
                "encode_elements": True,
                "element_vocab_size": 7,
            }, 7),
        ]

        for cfg_override, expected_dim in test_cases:
            with self.subTest(dim=expected_dim):
                fg = FiLMGenerator(channels_list=[8], config=cfg_override)
                self.assertEqual(fg.mlp[0].in_features, expected_dim)
                mods = fg(base_cond)
                self.assertEqual(mods[0][0].shape, (2, 8))

        # Raising error when all encodings are disabled
        with self.assertRaises(ValueError):
            FiLMGenerator(
                [8],
                config={
                    "encode_die": False,
                    "encode_tool": False,
                    "encode_flux": False,
                    "encode_time": False,
                    "encode_energy": False,
                    "encode_region": False,
                    "encode_relative_intensity": False,
                    "encode_elements": False,
                },
            )

    def test_condition_packing_helpers_and_modulation_application(self) -> None:
        """Verify helper configuration extractors and modulation application."""
        fg = FiLMGenerator(
            [16],
            config={
                "encode_die": True,
                "encode_tool": True,
                "encode_flux": True,
                "encode_time": True,
                "encode_energy": True,
            },
        )
        cfg = fg.get_condition_config(num_samples=2, energy_scale_ev=500.0)
        self.assertEqual(cfg["num_samples"], 2)
        self.assertEqual(cfg["energy_scale_ev"], 500.0)
        self.assertTrue(cfg["encode_die"])
        self.assertTrue(cfg["encode_energy"])

        cond = fg.prepare_condition_tensor(
            self.sample_meta, energy_bounds=np.array([[100.0, 150.0], [120.0, 170.0]])
        )
        self.assertIn("die", cond)
        self.assertIn("energy_bounds", cond)

        # Verify modulation application gamma * h + beta
        h = torch.randn(2, 16, 20)
        gamma, beta = fg(cond)[0]
        h_modulated = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)
        self.assertEqual(h_modulated.shape, (2, 16, 20))
        # Initially gamma=1, beta=0, so h_modulated == h
        torch.testing.assert_close(h_modulated, h)


if __name__ == "__main__":
    unittest.main()
