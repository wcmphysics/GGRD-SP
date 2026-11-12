"""Unit tests verifying architectural refactoring, SpectralDataBundle,
model registry, DatasetMetadata, resolve_subconfig, and configuration resolution.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pydantic import ValidationError

from denoise.common import run_batch_denoise
from models.factory import MODEL_REGISTRY, instantiate_model, register_model
from models.layers import BaseSpectralModel1D
from models.orchestration import _resolve_region_config
from utility.bundle import SpectralDataBundle
from utility.config_schema import (
    PairingConfig,
    QuantificationConfig,
    resolve_subconfig,
)
from utility.preprocessing import DatasetMetadata, PreprocessOutput


class TestSpectralDataBundle(unittest.TestCase):
    """Test SpectralDataBundle container behavior, validation, and unpacking."""

    def setUp(self) -> None:
        self.intensities = np.ones((5, 10), dtype=np.float64)
        self.energies = np.linspace(70, 75, 10)[None, :].repeat(5, axis=0)
        self.df_meta = pd.DataFrame({
            "spectrum_index": range(5),
            "measurement_id": ["m1"] * 5,
            "die": range(5),
            "region": ["Al2p"] * 5,
        })

    def test_valid_bundle_instantiation_and_validation(self) -> None:
        bundle = SpectralDataBundle(self.intensities, self.energies, self.df_meta)
        validated = bundle.validate()
        self.assertIs(validated, bundle)
        self.assertEqual(bundle.ary_intensity.shape, (5, 10))
        self.assertEqual(bundle.ary_energy.shape, (5, 10))
        self.assertEqual(len(bundle.df_meta), 5)

    def test_tuple_unpacking(self) -> None:
        bundle = SpectralDataBundle(self.intensities, self.energies, self.df_meta)
        ary_i, ary_e, df_m = bundle
        self.assertTrue(np.array_equal(ary_i, self.intensities))
        self.assertTrue(np.array_equal(ary_e, self.energies))
        self.assertEqual(len(df_m), 5)

    def test_invalid_dimensions_raise_error(self) -> None:
        bad_intensity = np.ones((5, 10, 2))
        bundle = SpectralDataBundle(bad_intensity, self.energies, self.df_meta)
        with self.assertRaises(ValueError) as ctx:
            bundle.validate()
        self.assertIn("ary_intensity must be 2D", str(ctx.exception))

    def test_shape_mismatch_raises_error(self) -> None:
        bad_energy = np.ones((5, 12))
        bundle = SpectralDataBundle(self.intensities, bad_energy, self.df_meta)
        with self.assertRaises(ValueError) as ctx:
            bundle.validate()
        self.assertIn("Shape mismatch", str(ctx.exception))

    def test_metadata_length_exceeds_rows_raises_error(self) -> None:
        extra_df = pd.DataFrame({"spectrum_index": range(10)})
        bundle = SpectralDataBundle(self.intensities, self.energies, extra_df)
        with self.assertRaises(ValueError) as ctx:
            bundle.validate()
        self.assertIn("Metadata rows 10 exceed intensity rows 5", str(ctx.exception))


class TestDatasetMetadataAndPreprocessOutput(unittest.TestCase):
    """Test typed DatasetMetadata and backwards-compatible PreprocessOutput."""

    def test_dataset_metadata_to_dict(self) -> None:
        meta = DatasetMetadata(
            num_dies=9,
            num_tools=2,
            num_regions=5,
            num_elements=5,
            die_index_mapping={0: 0, 1: 1},
            tool_index_mapping={"J4": 0, "J5": 1},
            region_index_mapping={"Al2p": 0},
            element_index_mapping={"Al": 0},
        )
        as_dict = meta.to_dict()
        self.assertEqual(as_dict["num_dies"], 9)
        self.assertEqual(as_dict["num_tools"], 2)
        self.assertEqual(as_dict["num_regions"], 5)
        self.assertEqual(as_dict["tool_index_mapping"]["J4"], 0)

    def test_preprocess_output_unpacking(self) -> None:
        df = pd.DataFrame({"col": [1, 2]})
        tool_map = {"J4": 0}
        reg_map = {"Al2p": 0}
        meta = DatasetMetadata(1, 1, 1, 1, {}, tool_map, reg_map, {})

        out = PreprocessOutput(df, tool_map, reg_map, meta)
        df_unpacked, t_unpacked, r_unpacked = out
        self.assertIs(df_unpacked, df)
        self.assertEqual(t_unpacked, tool_map)
        self.assertEqual(r_unpacked, reg_map)
        self.assertIs(out.dataset_metadata, meta)
        self.assertIs(out.metadata, meta)


class TestResolveSubconfig(unittest.TestCase):
    """Test resolve_subconfig Pydantic resolver."""

    def test_resolve_default(self) -> None:
        cfg = resolve_subconfig(PairingConfig)
        self.assertEqual(cfg["source_tool"], "J4")
        self.assertEqual(cfg["target_tool"], "J5")
        self.assertEqual(cfg["method"], "optimal")

    def test_resolve_with_partial_override(self) -> None:
        cfg = resolve_subconfig(PairingConfig, {"time_threshold_hours": 6.0})
        self.assertEqual(cfg["time_threshold_hours"], 6.0)
        self.assertEqual(cfg["source_tool"], "J4")

    def test_resolve_with_keyword_override(self) -> None:
        cfg = resolve_subconfig(
            PairingConfig,
            {"time_threshold_hours": 6.0},
            source_tool="J6",
        )
        self.assertEqual(cfg["source_tool"], "J6")
        self.assertEqual(cfg["time_threshold_hours"], 6.0)

    def test_resolve_extra_fields_preserved(self) -> None:
        cfg = resolve_subconfig(QuantificationConfig, {"warn_normalization": False})
        self.assertFalse(cfg["warn_normalization"])

    def test_resolve_invalid_raises_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            resolve_subconfig(PairingConfig, {"time_threshold_hours": -5.0})


class TestModelRegistryAndBaseClass(unittest.TestCase):
    """Test model registration and BaseSpectralModel1D functionality."""

    def test_builtin_models_in_registry(self) -> None:
        for arch in ("resnet", "unet", "deeplabv3", "fpn", "tcn"):
            self.assertIn(arch, MODEL_REGISTRY)

    def test_custom_model_registration_and_instantiation(self) -> None:
        class DummySpectralModel(BaseSpectralModel1D):
            def __init__(self, seq_len: int, config: dict | None = None) -> None:
                super().__init__()
                self.conv = nn.Conv1d(1, 4, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.validate_input(x)
                return self.conv(x)

        register_model("dummy_spectral", DummySpectralModel)
        self.assertIn("dummy_spectral", MODEL_REGISTRY)

        instance = instantiate_model("dummy_spectral", seq_len=50)
        self.assertIsInstance(instance, DummySpectralModel)
        x = torch.randn(2, 1, 50)
        out = instance(x)
        self.assertEqual(out.shape, (2, 4, 50))

        reg = instance.get_l2_regularization()
        self.assertGreater(reg.item(), 0.0)

        with self.assertRaises(ValueError):
            instance(torch.randn(2, 50))


class TestBatchDenoiseAndRegionConfigResolution(unittest.TestCase):
    """Test run_batch_denoise and _resolve_region_config."""

    def test_run_batch_denoise(self) -> None:
        def mock_denoise(energy, intensity, cfg, meta=None):
            return intensity * 0.9, np.full_like(intensity, 0.1), (0.05, 20.0, 1.0)

        energies = np.linspace(70, 75, 20)[None, :].repeat(3, axis=0)
        intensities = np.ones((3, 20))
        metas = [{"id": i} for i in range(3)]

        signals, confs, metrics = run_batch_denoise(
            mock_denoise, energies, intensities, metas
        )
        self.assertEqual(signals.shape, (3, 20))
        self.assertEqual(confs.shape, (3, 20))
        self.assertEqual(metrics.shape, (3, 3))
        self.assertTrue(np.allclose(signals, 0.9))

    def test_resolve_region_config_unified_and_regional(self) -> None:
        cfg = {
            "dataset": {
                "window_size_ev": 5.0,
                "sliding_stride_ev": 1.5,
                "unified_window_size_ev": 4.0,
                "unified_window_stride_ev": 1.2,
                "region_configs": {
                    "Al2p": {"window_size_ev": 3.5},
                },
            }
        }

        # Unified layout ("all")
        sw_unified, _, _ = _resolve_region_config(cfg, "all")
        self.assertEqual(sw_unified["window_size_ev"], 4.0)
        self.assertEqual(sw_unified["sliding_stride_ev"], 1.2)

        # Regional layout with region-specific override
        sw_al2p, _, _ = _resolve_region_config(cfg, "Al2p")
        self.assertEqual(sw_al2p["window_size_ev"], 3.5)
        self.assertEqual(sw_al2p["sliding_stride_ev"], 1.5)

        # Regional layout fallback to dataset window_size_ev
        sw_ti2p, _, _ = _resolve_region_config(cfg, "Ti2p")
        self.assertEqual(sw_ti2p["window_size_ev"], 5.0)
        self.assertEqual(sw_ti2p["sliding_stride_ev"], 1.5)


if __name__ == "__main__":
    unittest.main()
