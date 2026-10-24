"""Unit tests for utility.config_loader module."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utility.config_loader import (
    extract_element_vocabulary,
    load_config,
    load_material_elements,
    load_yaml,
    resolve_pipeline_config,
)


class TestConfigLoader(unittest.TestCase):
    """Test suite for YAML loading and element vocabulary extraction."""

    def test_load_yaml_missing_file_raises(self) -> None:
        """Verify FileNotFoundError is raised when file does not exist."""
        with self.assertRaises(FileNotFoundError):
            load_yaml("nonexistent_path/fake_config.yaml")

    def test_load_yaml_invalid_content_raises(self) -> None:
        """Verify ValueError is raised when file content is not a mapping."""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("- item1\n- item2\n")
            temp_path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                load_yaml(temp_path)
            self.assertIn("must parse to a dictionary", str(ctx.exception))
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_yaml_corrupt_syntax_raises(self) -> None:
        """Verify ValueError is raised on corrupt YAML syntax."""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("key: [unclosed list\n")
            temp_path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                load_yaml(temp_path)
            self.assertIn("Failed to parse YAML", str(ctx.exception))
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_material_elements_valid(self) -> None:
        """Verify proper extraction of material element lists."""
        cfg = {
            "materials": {
                "NMG": {"elements": ["Ti", "Al", "C", "O", "Cl", "Si"]},
                "PMG": {"elements": ["Ti", "N", "O", "Cl", "Si"]},
            }
        }
        res = load_material_elements(cfg)
        self.assertEqual(res["NMG"], ["Ti", "Al", "C", "O", "Cl", "Si"])
        self.assertEqual(res["PMG"], ["Ti", "N", "O", "Cl", "Si"])

    def test_load_material_elements_missing_section_raises(self) -> None:
        """Verify KeyError is raised when 'materials' section is missing."""
        with self.assertRaises(KeyError):
            load_material_elements({"other_key": 123})

    def test_load_material_elements_invalid_structure_raises(self) -> None:
        """Verify ValueError is raised when material structure is invalid."""
        with self.assertRaises(ValueError):
            load_material_elements({"materials": []})

        with self.assertRaises(ValueError):
            load_material_elements({"materials": {"NMG": "not a dict"}})

        with self.assertRaises(ValueError):
            load_material_elements({"materials": {"NMG": {"elements": []}}})

        with self.assertRaises(ValueError):
            load_material_elements(
                {"materials": {"NMG": {"elements": ["Ti", ""]}}}
            )

    def test_extract_element_vocabulary_dynamic(self) -> None:
        """Verify dynamic sorted vocabulary extraction supporting >14 elements."""
        materials = {
            "MatA": ["Ti", "Al", "C", "O", "Cl", "Si", "Fe", "Cu"],
            "MatB": ["Zn", "Mo", "W", "Ni", "Ta", "Hf", "Ti"],
        }
        vocab = extract_element_vocabulary(materials)
        # Expected sorted unique set of 14 elements
        expected = sorted([
            "Al", "C", "Cl", "Cu", "Fe", "Hf", "Mo", "Ni", "O", "Si", "Ta",
            "Ti", "W", "Zn"
        ])
        self.assertEqual(vocab, expected)
        self.assertEqual(len(vocab), 14)

    def test_extract_element_vocabulary_empty_raises(self) -> None:
        """Verify ValueError is raised if materials dictionary is empty."""
        with self.assertRaises(ValueError):
            extract_element_vocabulary({})

    def test_load_config_pipeline(self) -> None:
        """Verify config.yaml loads with all required stages."""
        cfg = load_config("config/config.yaml")
        required_stages = [
            "materials",
            "global",
            "data_generation",
            "pairing",
            "normalization",
            "splitting",
            "gp_flux",
            "dataset",
            "cost",
            "noise_analysis",
            "noise_plot",
            "film",
            "bayesian_opt",
            "training",
            "prediction",
            "metrics_report",
        ]
        for stage in required_stages:
            self.assertIn(stage, cfg, f"Stage '{stage}' missing from default config.")

        # Verify materials content
        mat_elements = load_material_elements(cfg)
        self.assertIn("NMG", mat_elements)
        self.assertIn("PMG", mat_elements)
        self.assertEqual(
            mat_elements["NMG"], ["Ti", "Al", "C", "O", "Cl", "Si"]
        )
        self.assertEqual(
            mat_elements["PMG"], ["Ti", "N", "O", "Cl", "Si"]
        )

        # Verify dynamic vocabulary
        vocab = extract_element_vocabulary(mat_elements)
        self.assertEqual(vocab, sorted(["Al", "C", "Cl", "N", "O", "Si", "Ti"]))

        # Verify resolve_pipeline_config resolution inside load_config
        self.assertIn("material_elements", cfg["film"])
        self.assertIn("element_vocab", cfg["film"])
        self.assertEqual(cfg["pairing"]["source_tool"], cfg["global"]["source_tool"])
        self.assertIs(cfg["training"]["film_config"], cfg["film"])
        self.assertIs(cfg["prediction"]["film_config"], cfg["film"])

    def test_resolve_pipeline_config_propagation_and_overrides(self) -> None:
        """Verify global defaults propagate and explicit overrides persist."""
        raw_cfg = {
            "materials": {
                "MatA": {"elements": ["Ti", "O"]},
            },
            "global": {
                "source_tool": "ToolA",
                "target_tool": "ToolB",
                "model_type": "residual_unet",
                "use_sliding_window": True,
                "use_film": True,
                "unified_model": False,
                "example_region": "Ti2p",
                "sample_measurement_id": "Meas_01",
            },
            "data_generation": {
                "regions": ["Ti2p", "O1s"],
            },
            "dataset": {
                "source_tool": "CustomTool",  # Explicit override
            },
            "bayesian_opt": {
                "learning_rate_bounds": [1e-4, 1e-2],
                "l2_bounds": [1e-6, 1e-3],
            },
        }
        resolved = resolve_pipeline_config(raw_cfg)

        # Overridden tool persists in dataset
        self.assertEqual(resolved["dataset"]["source_tool"], "CustomTool")
        # Global propagates to pairing
        self.assertEqual(resolved["pairing"]["source_tool"], "ToolA")
        self.assertEqual(resolved["pairing"]["target_tool"], "ToolB")
        # Global sliding window propagates to dataset
        self.assertTrue(resolved["dataset"]["use_sliding_window"])
        # Metrics report defaults to canonical names
        self.assertIn("peak_error_percentage", resolved["metrics_report"]["metrics"])
        self.assertIn("max_error_percentage", resolved["metrics_report"]["metrics"])
        self.assertNotIn("peak_err%", resolved["metrics_report"]["metrics"])
        # Element vocabulary and materials populated in film
        self.assertEqual(resolved["film"]["element_vocab"], ["O", "Ti"])
        # Training and prediction share film_config reference
        self.assertIs(resolved["training"]["film_config"], resolved["film"])
        self.assertIs(resolved["prediction"]["film_config"], resolved["film"])
        # Modifying film modifies training film_config
        resolved["film"]["test_mutability"] = 123
        self.assertEqual(
            resolved["training"]["film_config"]["test_mutability"], 123
        )
        # Bayesian opt bounds converted to tuples
        self.assertIsInstance(
            resolved["bayesian_opt"]["learning_rate_bounds"], tuple
        )
        self.assertIsInstance(
            resolved["bayesian_opt"]["l2_bounds"], tuple
        )


if __name__ == "__main__":
    unittest.main()
