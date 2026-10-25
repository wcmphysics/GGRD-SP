"""Unit tests for Pydantic configuration schema and validation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any
import unittest

from pydantic import ValidationError
import yaml

from utility.config_loader import load_config
from utility.config_schema import (
    DenoiseSavgolConfig,
    GlobalConfig,
    PipelineConfig,
    SplittingConfig,
    TrainConfig,
)


class TestConfigSchema(unittest.TestCase):
    """Test suite for PipelineConfig Pydantic validation."""

    def test_pipeline_config_from_default_yaml(self) -> None:
        """Verify default config.yaml passes full Pydantic validation."""
        config_path = "config/config.yaml"
        cfg = PipelineConfig.from_yaml(config_path)

        self.assertIsInstance(cfg, PipelineConfig)
        self.assertIn("NMG", cfg.materials)
        self.assertEqual(cfg.global_.source_tool, "J4")
        self.assertEqual(cfg.global_.target_tool, "J5")

        dumped = cfg.to_dict()
        self.assertIn("global", dumped)
        self.assertNotIn("global_", dumped)
        self.assertEqual(dumped["global"]["source_tool"], "J4")

    def test_export_json_schema(self) -> None:
        """Verify JSON Schema export creates valid, non-empty JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_schema.json"
            exported = PipelineConfig.export_json_schema(out_path)
            self.assertTrue(exported.exists())

            with open(exported, "r", encoding="utf-8") as f:
                schema_data = json.load(f)

            self.assertEqual(schema_data.get("title"), "PipelineConfig")
            self.assertIn("properties", schema_data)
            self.assertIn("materials", schema_data["properties"])
            self.assertIn("global", schema_data["properties"])

    def test_kernel_size_odd_validation(self) -> None:
        """Verify TrainConfig rejects even kernel sizes."""
        # Valid odd kernel size
        tc_valid = TrainConfig(kernel_size=9)
        self.assertEqual(tc_valid.kernel_size, 9)

        # Invalid even kernel size
        with self.assertRaises(ValidationError) as ctx:
            TrainConfig(kernel_size=8)
        self.assertIn("kernel_size must be an odd integer", str(ctx.exception))

    def test_train_config_num_layers_validation(self) -> None:
        """Verify TrainConfig enforces num_layers >= 2."""
        tc_valid = TrainConfig(num_layers=3)
        self.assertEqual(tc_valid.num_layers, 3)

        with self.assertRaises(ValidationError):
            TrainConfig(num_layers=1)

    def test_savgol_window_length_odd_validation(self) -> None:
        """Verify DenoiseSavgolConfig rejects even window lengths."""
        # Valid odd window length
        sg_valid = DenoiseSavgolConfig(window_length=15)
        self.assertEqual(sg_valid.window_length, 15)

        # Invalid even window length
        with self.assertRaises(ValidationError) as ctx:
            DenoiseSavgolConfig(window_length=16)
        self.assertIn(
            "window_length must be an odd integer", str(ctx.exception)
        )

    def test_splitting_ratio_sum_validation(self) -> None:
        """Verify SplittingConfig enforces train + val + test == 1.0."""
        # Valid ratios summing to 1.0
        sc_valid = SplittingConfig(
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        self.assertAlmostEqual(
            sc_valid.train_ratio + sc_valid.val_ratio + sc_valid.test_ratio, 1.0
        )

        # Invalid ratios summing to 1.2
        with self.assertRaises(ValidationError) as ctx:
            SplittingConfig(train_ratio=0.6, val_ratio=0.3, test_ratio=0.3)
        self.assertIn("must sum to 1.0", str(ctx.exception))

    def test_normalization_mode_enum_validation(self) -> None:
        """Verify normalization rejects unknown normalization modes."""
        valid_modes = [
            "by_total_flux_single_die",
            "by_regional_max_intensity",
            "by_regional_max_intensity_of_source",
            "by_mean_intensity_single_die",
            "no_normalization",
        ]
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            raw_data: dict[str, Any] = yaml.safe_load(f)

        for mode in valid_modes:
            raw_data["normalization"]["normalization_mode"] = mode
            cfg = PipelineConfig.model_validate(raw_data)
            self.assertEqual(cfg.normalization.normalization_mode, mode)

        raw_data["normalization"]["normalization_mode"] = "invalid_mode"
        with self.assertRaises(ValidationError) as ctx:
            PipelineConfig.model_validate(raw_data)
        self.assertIn("Input should be", str(ctx.exception))

    def test_negative_values_rejected(self) -> None:
        """Verify positive/non-negative constraints on numeric parameters."""
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            raw_data: dict[str, Any] = yaml.safe_load(f)

        # Negative time_threshold_hours
        raw_data["pairing"]["time_threshold_hours"] = -5.0
        with self.assertRaises(ValidationError):
            PipelineConfig.model_validate(raw_data)

    def test_empty_materials_rejected(self) -> None:
        """Verify empty materials dictionary is rejected."""
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            raw_data: dict[str, Any] = yaml.safe_load(f)

        raw_data["materials"] = {}
        with self.assertRaises(ValidationError):
            PipelineConfig.model_validate(raw_data)

    def test_empty_material_elements_rejected(self) -> None:
        """Verify material with empty elements list is rejected."""
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            raw_data: dict[str, Any] = yaml.safe_load(f)

        raw_data["materials"]["NMG"]["elements"] = []
        with self.assertRaises(ValidationError):
            PipelineConfig.model_validate(raw_data)

    def test_load_config_with_validation_success(self) -> None:
        """Verify load_config executes schema validation by default."""
        cfg = load_config("config/config.yaml", validate=True)
        self.assertIsInstance(cfg, dict)
        self.assertIn("global", cfg)
        self.assertIn("material_elements", cfg["film"])

    def test_load_config_validation_failure_on_corrupted_data(self) -> None:
        """Verify load_config raises ValidationError on invalid data."""
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            raw_data: dict[str, Any] = yaml.safe_load(f)

        raw_data["training"]["train_config"]["kernel_size"] = 4  # Even number!

        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.dump(raw_data, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValidationError):
                load_config(tmp_path, validate=True)
            # Should pass when validate=False
            loaded_no_val = load_config(tmp_path, validate=False)
            self.assertEqual(
                loaded_no_val["training"]["train_config"]["kernel_size"], 4
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
