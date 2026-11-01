"""Unit tests for Feature-wise Linear Modulation (FiLM) in models/film.py."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
import torch

from models.film import FiLMGenerator, prepare_film_condition_tensor


class TestFiLMModulation(unittest.TestCase):
    """Test suite verifying FiLM condition preparation and modulation generator."""

    def setUp(self) -> None:
        self.sample_meta = [
            {
                "die_one_hot": [1.0, 0.0, 0.0],
                "source_tool_one_hot": [1.0, 0.0],
                "target_tool_one_hot": [0.0, 1.0],
                "region_one_hot": [0.0, 1.0, 0.0, 0.0],
                "element_multi_hot": [1.0, 1.0, 0.0],
                "total_flux_single_die_standardized": -0.5,
                "time_standardized": -1.0,
                "relative_intensity_single_region": 0.85,
            },
            {
                "die_one_hot": [0.0, 0.0, 1.0],
                "source_tool_one_hot": [1.0, 0.0],
                "target_tool_one_hot": [0.0, 1.0],
                "region_one_hot": [0.0, 0.0, 0.0, 1.0],
                "element_multi_hot": [0.0, 1.0, 1.0],
                "total_flux_single_die_standardized": 1.25,
                "time_standardized": 1.0,
                "relative_intensity_single_region": 1.30,
            },
        ]
        self.base_cfg = {
            "num_dies": 3,
            "num_tools": 2,
            "num_regions": 4,
            "num_elements": 3,
            "encode_die": True,
            "encode_tool": True,
            "encode_flux": True,
            "encode_time": True,
            "encode_energy": True,
            "encode_region": True,
            "encode_relative_intensity": True,
            "encode_elements": True,
            "energy_scale_ev": 1000.0,
            "hidden_dim": 32,
        }

    def test_prepare_film_condition_tensor_formats_and_features(self) -> None:
        """Verify feature extraction across dict, list, and DataFrame inputs."""
        we = np.array([[100.0, 106.0], [105.0, 111.0]])
        cfg = {**self.base_cfg, "num_samples": 2}
        cond = prepare_film_condition_tensor(
            metadata=self.sample_meta,
            energy_bounds=we,
            config=cfg,
        )

        self.assertEqual(cond["die_one_hot"].shape, (2, 3))
        self.assertEqual(cond["source_tool_one_hot"].shape, (2, 2))
        self.assertEqual(cond["target_tool_one_hot"].shape, (2, 2))
        self.assertEqual(cond["region_one_hot"].shape, (2, 4))
        self.assertEqual(cond["element_multi_hot"].shape, (2, 3))
        self.assertEqual(cond["total_flux_single_die_standardized"].shape, (2, 1))
        self.assertEqual(cond["time_standardized"].shape, (2, 1))
        self.assertEqual(cond["relative_intensity_single_region"].shape, (2, 1))
        self.assertEqual(cond["energy_bounds"].shape, (2, 2))
        self.assertAlmostEqual(cond["energy_bounds"][0, 0].item(), 0.1, places=4)

        # Test DataFrame format
        df_meta = pd.DataFrame(self.sample_meta)
        cond_df = prepare_film_condition_tensor(
            metadata=df_meta, energy_bounds=we, config=cfg
        )
        self.assertEqual(cond_df["die_one_hot"].tolist(), cond["die_one_hot"].tolist())

    def test_missing_metadata_fields_raise(self) -> None:
        """Verify missing mandatory metadata keys or unpaired rows raise error."""
        cfg = {**self.base_cfg, "num_samples": 1}

        # Missing die_one_hot
        bad_meta = [{k: v for k, v in self.sample_meta[0].items() if k != "die_one_hot"}]
        with self.assertRaises(ValueError):
            prepare_film_condition_tensor(bad_meta, config=cfg)

        # Unpaired row with target_tool_one_hot = None
        unpaired_meta = [{**self.sample_meta[0], "target_tool_one_hot": None}]
        with self.assertRaises(ValueError):
            prepare_film_condition_tensor(unpaired_meta, config=cfg)

        # Missing energy bounds when encode_energy is True
        with self.assertRaises(ValueError):
            prepare_film_condition_tensor(
                self.sample_meta, energy_bounds=None, config=cfg
            )

        # Missing required config flag without default
        with self.assertRaises(KeyError):
            prepare_film_condition_tensor(
                self.sample_meta,
                config={k: v for k, v in cfg.items() if k != "encode_die"},
            )

    def test_film_generator_identity_initialization(self) -> None:
        """Verify FiLMGenerator outputs gamma=1.0 and beta=0.0 initially."""
        film_gen = FiLMGenerator(channels_list=[16, 32], config=self.base_cfg)
        cond = prepare_film_condition_tensor(
            metadata=self.sample_meta[:1],
            energy_bounds=np.array([[100.0, 150.0]]),
            config=self.base_cfg,
        )

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
        """Verify feature toggle permutations compute correct input dimension."""
        # die(3) + tools(2*2) + flux(1) + time(1) + rel_int(1) + region(4) + elem(3) + energy(2) = 19
        fg_all = FiLMGenerator(channels_list=[8], config=self.base_cfg)
        self.assertEqual(fg_all.in_dim, 19)
        self.assertEqual(fg_all.mlp[0].in_features, 19)

        # Toggle test: only die(3) + tools(4) = 7
        subset_cfg = {
            **self.base_cfg,
            "encode_flux": False,
            "encode_time": False,
            "encode_energy": False,
            "encode_region": False,
            "encode_relative_intensity": False,
            "encode_elements": False,
        }
        fg_sub = FiLMGenerator(channels_list=[8], config=subset_cfg)
        self.assertEqual(fg_sub.in_dim, 7)
        self.assertEqual(fg_sub.mlp[0].in_features, 7)

        # Raising error when all encodings are disabled
        all_off_cfg = {
            **self.base_cfg,
            "encode_die": False,
            "encode_tool": False,
            "encode_flux": False,
            "encode_time": False,
            "encode_energy": False,
            "encode_region": False,
            "encode_relative_intensity": False,
            "encode_elements": False,
        }
        with self.assertRaises(ValueError):
            FiLMGenerator([8], config=all_off_cfg)

    def test_condition_packing_helpers_and_modulation_application(self) -> None:
        """Verify helper configuration extractors and modulation application."""
        fg = FiLMGenerator([16], config=self.base_cfg)
        cfg = fg.get_condition_config(num_samples=2, energy_scale_ev=500.0)
        self.assertEqual(cfg["num_samples"], 2)
        self.assertEqual(cfg["energy_scale_ev"], 500.0)
        self.assertTrue(cfg["encode_die"])
        self.assertTrue(cfg["encode_energy"])

        cond = fg.prepare_condition_tensor(
            self.sample_meta, energy_bounds=np.array([[100.0, 150.0], [120.0, 170.0]])
        )
        self.assertIn("die_one_hot", cond)
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
