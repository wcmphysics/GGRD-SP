"""Unit tests for unified multi-region spectral model training and inference.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import models
import utility


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

    df_meta["measurement_id_target"] = df_meta["measurement_id"].map(tgt_map)
    df_meta["tool_target"] = df_meta["measurement_id"].map(tgt_tool_map)
    df_meta["tool_index"] = df_meta["tool"].map({"J4": 0, "J5": 1})
    df_meta["tool_target_index"] = df_meta["tool_target"].map(
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
        tgt_mid = r["measurement_id_target"]
        if pd.notna(tgt_mid):
            tgt_indices.append(
                tgt_lookup.get((str(tgt_mid), int(r["die"]), str(r["region"])))
            )
        else:
            tgt_indices.append(None)
    df_meta["spectrum_index_target"] = tgt_indices
    df_meta["die_flux_normalized"] = 0.0
    df_meta["time_normalized"] = 0.0

    return ary_intensity, ary_energy, df_meta


class TestUnifiedModel(unittest.TestCase):
    """Test suite verifying unified multi-region model dataset, training,
    and inference.
    """

    def setUp(self) -> None:
        mock_data = _create_mock_spectral_data()
        self.ary_intensity, self.ary_energy, self.df_meta = mock_data

    def test_create_unified_dataset_full_spectrum(self) -> None:
        """Test unified dataset creation in full spectrum mode."""
        bundle = models.create_unified_dataset(
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
        # 1 train session * 3 dies * 3 regions = 9 train samples
        self.assertEqual(len(bundle["dataset_train"]), 9)
        self.assertEqual(len(bundle["dataset_val"]), 9)
        self.assertEqual(len(bundle["dataset_test"]), 9)

        item = bundle["dataset_train"][0]
        self.assertIn("x", item)
        self.assertIn("y", item)
        self.assertIn("meta", item)
        self.assertIn("region", item["meta"])
        self.assertIn("region_index", item["meta"])

    def test_create_unified_dataset_sliding_window(self) -> None:
        """Test unified dataset creation in sliding window patch mode."""
        bundle = models.create_unified_dataset(
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
        # 9 full spectra * multiple patches per spectrum
        self.assertGreater(len(bundle["dataset_train"]), 9)
        patch_item = bundle["dataset_train"][0]
        self.assertEqual(patch_item["x"].shape[-1], 11)
        self.assertIn("meta", patch_item)
        self.assertIn("region", patch_item["meta"])

    def test_unified_model_training_resnet(self) -> None:
        """Test single ResNet model training across all regions."""
        regional_datasets = models.create_regional_datasets(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={
                "source_tool": "J4",
                "target_tool": "J5",
                "use_sliding_window": False,
            },
        )
        criterion = models.SpectralLoss()
        results = models.perform_model_training_and_bayesian_optimization(
            regional_datasets,
            criterion,
            config={
                "model_type": "resnet",
                "unified_model": True,
                "use_film": True,
                "train_config": {
                    "epochs": 2,
                    "batch_size": 4,
                    "learning_rate": 1e-3,
                    "resnet_channel": 8,
                    "kernel_size": 3,
                    "verbose": False,
                },
                "film_config": {
                    "use_film": True,
                    "encode_region": True,
                    "num_regions": 8,
                    "region_embed_dim": 4,
                },
            },
        )

        self.assertIn("all", results["models"])
        single_model = results["models"]["all"]
        self.assertIsInstance(single_model, nn.Module)
        self.assertTrue(getattr(single_model, "use_film", False))

        # Check that individual regions map to the exact same model reference
        self.assertIs(results["models"]["Al2p"], single_model)
        self.assertIs(results["models"]["Ti2p"], single_model)
        self.assertIs(results["models"]["O1s"], single_model)

        # Predict using unified model
        data_orig = (self.ary_intensity, self.ary_energy, self.df_meta)
        pred_i, pred_e, pred_df = models.predict_spectra(
            models=results["models"],
            data=data_orig,
            config={"source_tool": "J4", "target_tool": "J5"},
        )
        self.assertEqual(pred_i.shape[1], 40)
        self.assertEqual(len(pred_df), 27)

    def test_unified_model_training_unet(self) -> None:
        """Test single Residual U-Net training across all regions."""
        regional_datasets = models.create_regional_datasets(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={
                "source_tool": "J4",
                "target_tool": "J5",
                "use_sliding_window": False,
            },
        )
        criterion = models.SpectralLoss()
        results = models.perform_model_training_and_bayesian_optimization(
            regional_datasets,
            criterion,
            config={
                "model_type": "residual_unet",
                "unified_model": True,
                "use_film": False,
                "train_config": {
                    "epochs": 2,
                    "batch_size": 4,
                    "depth": 2,
                    "residual_unet_channel": 8,
                    "kernel_size": 3,
                    "verbose": False,
                },
            },
        )
        self.assertIn("all", results["models"])
        self.assertIs(results["models"]["Al2p"], results["models"]["all"])

    def test_unified_model_training_conventional_unet(self) -> None:
        """Test single Conventional U-Net training across all regions."""
        regional_datasets = models.create_regional_datasets(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={
                "source_tool": "J4",
                "target_tool": "J5",
                "use_sliding_window": False,
            },
        )
        criterion = models.SpectralLoss()
        results = models.perform_model_training_and_bayesian_optimization(
            regional_datasets,
            criterion,
            config={
                "model_type": "unet",
                "unified_model": True,
                "use_film": False,
                "train_config": {
                    "epochs": 2,
                    "batch_size": 4,
                    "depth": 2,
                    "unet_channel": 8,
                    "kernel_size": 3,
                    "verbose": False,
                },
            },
        )
        self.assertIn("all", results["models"])
        self.assertIs(results["models"]["Al2p"], results["models"]["all"])

    def test_dual_mode_per_region_preservation(self) -> None:
        """Test that setting unified_model=False preserves independent
        regional models.
        """
        regional_datasets = models.create_regional_datasets(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={
                "source_tool": "J4",
                "target_tool": "J5",
                "use_sliding_window": False,
            },
        )
        criterion = models.SpectralLoss()
        results = models.perform_model_training_and_bayesian_optimization(
            regional_datasets,
            criterion,
            config={
                "model_type": "resnet",
                "unified_model": False,
                "use_film": False,
                "train_config": {
                    "epochs": 2,
                    "batch_size": 4,
                    "resnet_channel": 8,
                    "kernel_size": 3,
                    "verbose": False,
                },
            },
        )
        # In per-region mode, "all" is NOT created and models are independent
        self.assertNotIn("all", results["models"])
        self.assertIn("Al2p", results["models"])
        self.assertIn("Ti2p", results["models"])
        self.assertIn("O1s", results["models"])
        self.assertIsNot(results["models"]["Al2p"], results["models"]["Ti2p"])

    def test_unified_model_training_deeplabv3(self) -> None:
        """Test single DeepLabV3 training across all regions."""
        regional_datasets = models.create_regional_datasets(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={
                "source_tool": "J4",
                "target_tool": "J5",
                "use_sliding_window": False,
            },
        )
        criterion = models.SpectralLoss()
        results = models.perform_model_training_and_bayesian_optimization(
            regional_datasets,
            criterion,
            config={
                "model_type": "deeplabv3",
                "unified_model": True,
                "use_film": False,
                "train_config": {
                    "epochs": 2,
                    "batch_size": 4,
                    "deeplabv3_backbone_channel": 8,
                    "deeplabv3_aspp_channel": 8,
                    "kernel_size": 3,
                    "verbose": False,
                },
            },
        )
        self.assertIn("all", results["models"])

    def test_unified_model_bayesian_optimization(self) -> None:
        """Test single global Bayesian optimization on unified data."""
        regional_datasets = models.create_regional_datasets(
            self.df_meta,
            self.ary_intensity,
            self.ary_energy,
            config={
                "source_tool": "J4",
                "target_tool": "J5",
                "use_sliding_window": False,
            },
        )
        criterion = models.SpectralLoss()
        results = models.perform_model_training_and_bayesian_optimization(
            regional_datasets,
            criterion,
            config={
                "model_type": "resnet",
                "unified_model": True,
                "use_film": False,
                "train_config": {"epochs": 2, "verbose": False},
            },
            bayesian_opt_config={
                "use_bayesian_opt": True,
                "num_trials": 2,
                "epochs_per_trial": 1,
                "kernel_sizes": [3],
                "resnet_channels": [8],
                "batch_sizes": [4],
            },
        )
        self.assertIn("all", results["bayesian_opt_results"])
        self.assertIn("best_parameters", results["bayesian_opt_results"]["all"])


if __name__ == "__main__":
    unittest.main()
