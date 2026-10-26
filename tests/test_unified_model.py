"""Unit tests for Stage 7 unified multi-region neural network architecture."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
import torch.nn as nn

import models
from utility.pseudo_measurement import generate_pseudo_measurements


def _create_mock_spectral_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create reproducible synthetic spectral data with 3 regions."""
    regions = ["Al2p", "Ti2p", "O1s"]
    n_meas = 6
    n_dies = 3
    n_pts = 40
    rows = []
    idx = 0

    for m_idx in range(n_meas):
        tool = "J4" if m_idx < 3 else "J5"
        m_id = f"NMG_M_{tool}_{m_idx:05d}"
        for d in range(n_dies):
            for r_idx, reg in enumerate(regions):
                is_train = m_idx in (0, 3)
                split_val = "train" if is_train else ("val" if m_idx in (1, 4) else "test")
                rows.append({
                    "spectrum_index": idx,
                    "measurement_id": m_id,
                    "tool": tool,
                    "die": d,
                    "region": reg,
                    "region_index": r_idx,
                    "split": split_val,
                })
                idx += 1

    df_meta = pd.DataFrame(rows)
    total_spectra = len(df_meta)
    rng = np.random.RandomState(42)
    ary_intensity = rng.uniform(
        10.0, 100.0, size=(total_spectra, n_pts)
    ).astype(np.float32)
    ary_energy = np.tile(
        np.linspace(70.0, 80.0, n_pts, dtype=np.float32), (total_spectra, 1)
    )

    # Add 1-to-1 pairing mapping
    tgt_map = {
        "NMG_M_J4_00000": "NMG_M_J5_00003",
        "NMG_M_J4_00001": "NMG_M_J5_00004",
        "NMG_M_J4_00002": "NMG_M_J5_00005",
    }
    tgt_tool_map = {m: "J5" for m in tgt_map}

    df_meta["target_measurement_id"] = df_meta["measurement_id"].map(tgt_map)
    df_meta["target_tool"] = df_meta["measurement_id"].map(tgt_tool_map)
    df_meta["source_tool_index"] = df_meta["tool"].map({"J4": 0, "J5": 1})
    df_meta["target_tool_index"] = df_meta["target_tool"].map(
        {"J4": 0, "J5": 1}
    )

    # Fill target spectrum indices
    tgt_lookup: dict[tuple[str, int, str], int] = {
        (str(r["measurement_id"]), int(r["die"]), str(r["region"])): int(
            r["spectrum_index"]
        )
        for _, r in df_meta[df_meta["tool"] == "J5"].iterrows()
    }
    tgt_indices = []
    for _, r in df_meta.iterrows():
        tgt_mid = r["target_measurement_id"]
        if pd.notna(tgt_mid):
            tgt_indices.append(
                tgt_lookup.get((str(tgt_mid), int(r["die"]), str(r["region"])))
            )
        else:
            tgt_indices.append(None)
    df_meta["target_spectrum_index"] = tgt_indices
    df_meta["total_flux_single_die_standardized"] = 0.0
    df_meta["time_standardized"] = 0.0

    return ary_intensity, ary_energy, df_meta


class TestUnifiedModel(unittest.TestCase):
    """Test suite for unified multi-region training and dataset bundling."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ary_intensity, cls.ary_energy, cls.df_meta = (
            _create_mock_spectral_data()
        )

    def test_create_unified_dataset_modes(self) -> None:
        """Verify unified dataset creation in full-spectrum and sliding-window
        modes.
        """
        # 1. Full spectrum mode
        bundle_full = models.create_unified_dataset(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={"source_tool": "J4", "target_tool": "J5", "use_sliding_window": False},
        )
        self.assertEqual(bundle_full["full_spectrum_length"], 40)
        self.assertEqual(len(bundle_full["dataset_train"]), 9)

        # 2. Sliding window mode
        bundle_sw = models.create_unified_dataset(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={
                "source_tool": "J4",
                "target_tool": "J5",
                "use_sliding_window": True,
                "window_size": 11,
                "stride": 5,
            },
        )
        self.assertGreater(len(bundle_sw["dataset_train"]), 9)
        self.assertEqual(bundle_sw["dataset_train"][0]["x"].shape[-1], 11)

    def test_unified_model_training_architectures(self) -> None:
        """Verify unified model training across ResNet, UNet, and DeepLabV3."""
        architectures = [
            (
                "resnet",
                {
                    "resnet_channel": 8,
                    "kernel_size": 3,
                    "dropout": 0.0,
                    "use_batch_norm": True,
                    "padding_mode": "reflect",
                    "use_film": True,
                },
                {"use_film": True, "encode_region": True, "num_regions": 8, "region_embed_dim": 4},
            ),
            ("residual_unet", {"depth": 2, "residual_unet_channel": 8, "kernel_size": 3}, None),
            ("unet", {"depth": 2, "unet_channel": 8, "kernel_size": 3}, None),
            ("deeplabv3", {"deeplabv3_backbone_channel": 8, "deeplabv3_aspp_channel": 8, "kernel_size": 3}, None),
        ]

        regional_datasets = models.create_regional_datasets(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={"source_tool": "J4", "target_tool": "J5", "use_sliding_window": False},
        )

        for arch_name, train_cfg, film_cfg in architectures:
            with self.subTest(arch=arch_name):
                pipeline_cfg = {
                    "model_type": arch_name,
                    "unified_model": True,
                    "use_film": film_cfg is not None,
                    "train_config": {"epochs": 1, "batch_size": 4, "verbose": False, **train_cfg},
                }
                if film_cfg is not None:
                    pipeline_cfg["film_config"] = film_cfg

                results = models.perform_model_training_and_bayesian_optimization(
                    regional_datasets, models.SpectralLoss(), config=pipeline_cfg
                )
                self.assertIn("all", results["models"])
                single_model = results["models"]["all"]
                self.assertIsInstance(single_model, nn.Module)
                self.assertIs(results["models"]["Al2p"], single_model)
                self.assertIs(results["models"]["Ti2p"], single_model)

    def test_dual_mode_per_region_preservation(self) -> None:
        """Verify setting unified_model=False preserves independent regional
        models.
        """
        regional_datasets = models.create_regional_datasets(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={"source_tool": "J4", "target_tool": "J5", "use_sliding_window": False},
        )
        results = models.perform_model_training_and_bayesian_optimization(
            regional_datasets,
            models.SpectralLoss(),
            config={
                "model_type": "resnet",
                "unified_model": False,
                "use_film": False,
                "train_config": {
                    "epochs": 1,
                    "batch_size": 4,
                    "resnet_channel": 8,
                    "kernel_size": 3,
                    "dropout": 0.0,
                    "use_batch_norm": True,
                    "padding_mode": "reflect",
                    "use_film": False,
                    "verbose": False,
                },
            },
        )
        self.assertNotIn("all", results["models"])
        self.assertIsNot(results["models"]["Al2p"], results["models"]["Ti2p"])

    def test_unified_model_bayesian_optimization(self) -> None:
        """Verify Stage 7 Bayesian optimization in unified mode."""
        regional_datasets = models.create_regional_datasets(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={"source_tool": "J4", "target_tool": "J5", "use_sliding_window": False},
        )
        results = models.perform_model_training_and_bayesian_optimization(
            regional_datasets,
            models.SpectralLoss(),
            bayesian_opt_config={
                "num_trials": 2,
                "epochs_per_trial": 1,
                "kernel_sizes": [3],
                "resnet_channels": [8],
                "seed": 42,
            },
            config={
                "model_type": "resnet",
                "unified_model": True,
                "use_bayesian_opt": True,
                "train_config": {
                    "dropout": 0.0,
                    "use_batch_norm": True,
                    "padding_mode": "reflect",
                    "use_film": False,
                    "verbose": False,
                },
            },
        )
        self.assertIn("all", results["models"])
        self.assertIn("best_parameters", results["bayesian_opt_results"]["all"])


if __name__ == "__main__":
    unittest.main()
