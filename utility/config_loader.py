"""Configuration loader module for GGRD-SP.

Provides centralized YAML configuration loading, material element parsing,
and dynamic element vocabulary extraction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Parameters
    ----------
    file_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed dictionary from the YAML file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the YAML content is empty or does not parse into a dictionary.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: '{os.path.abspath(path)}'. "
            "Please ensure the file exists."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse YAML configuration file '{path}': {exc}"
        ) from exc

    if not isinstance(content, dict):
        raise ValueError(
            f"Configuration file '{path}' must parse to a dictionary, "
            f"got {type(content).__name__}."
        )

    return content


def load_material_elements(
    config_or_path: dict[str, Any] | str | Path,
) -> dict[str, list[str]]:
    """Extract and validate material element compositions.

    Parameters
    ----------
    config_or_path : dict[str, Any] | str | Path
        Either an already loaded configuration dictionary or a file path to
        a YAML configuration file containing a 'materials' section.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping material names to lists of element symbols.

    Raises
    ------
    KeyError
        If 'materials' section is missing.
    ValueError
        If materials is not a dictionary or any material has invalid elements.
    """
    raw_cfg = (
        load_yaml(config_or_path)
        if isinstance(config_or_path, (str, Path))
        else config_or_path
    )

    if not isinstance(raw_cfg, dict):
        raise ValueError(
            f"Expected dictionary or path, got {type(raw_cfg).__name__}."
        )

    if "materials" not in raw_cfg:
        raise KeyError(
            "Configuration missing required 'materials' section. "
            f"Available top-level keys: {sorted(raw_cfg.keys())}."
        )

    materials = raw_cfg["materials"]
    if not isinstance(materials, dict) or not materials:
        raise ValueError(
            "'materials' must be a non-empty dictionary mapping material names "
            "to element lists."
        )

    validated: dict[str, list[str]] = {}
    for mat_name, mat_info in materials.items():
        if not isinstance(mat_info, dict) or "elements" not in mat_info:
            raise ValueError(
                f"Material '{mat_name}' must be a dictionary with an 'elements' "
                "key."
            )
        elements = mat_info["elements"]
        if not isinstance(elements, (list, tuple)) or not elements:
            raise ValueError(
                f"Material '{mat_name}' must have a non-empty list of elements, "
                f"got {elements!r}."
            )
        for elem in elements:
            if not isinstance(elem, str) or not elem.strip():
                raise ValueError(
                    f"Material '{mat_name}' contains invalid element {elem!r}; "
                    "all elements must be non-empty strings."
                )
        validated[str(mat_name)] = [str(elem).strip() for elem in elements]

    return validated


def extract_element_vocabulary(
    material_elements: dict[str, Any],
) -> list[str]:
    """Dynamically extract the sorted unique element vocabulary from materials.

    Parameters
    ----------
    material_elements : dict[str, Any]
        Dictionary mapping material names to element lists or material
        definitions containing an 'elements' list.

    Returns
    -------
    list[str]
        Alphabetically sorted list of all unique element symbols.
    """
    if not material_elements:
        raise ValueError("material_elements dictionary must not be empty.")

    if "materials" in material_elements and isinstance(material_elements["materials"], dict):
        material_elements = load_material_elements(material_elements)

    vocab_set: set[str] = set()
    for mat_name, mat_val in material_elements.items():
        if isinstance(mat_val, dict) and "elements" in mat_val:
            elems = mat_val["elements"]
        elif isinstance(mat_val, (list, tuple, set)):
            elems = mat_val
        else:
            raise ValueError(
                f"Material '{mat_name}' has invalid element specification: {mat_val!r}."
            )
        for elem in elems:
            if not isinstance(elem, str) or not elem.strip():
                raise ValueError(
                    f"Material '{mat_name}' contains invalid element {elem!r}."
                )
            vocab_set.add(str(elem).strip())

    return sorted(vocab_set)


def load_default_config(
    file_path: str | Path = "config/default_config.yaml",
) -> dict[str, Any]:
    """Load default pipeline configuration from config/default_config.yaml.

    Parameters
    ----------
    file_path : str | Path, optional
        Path to default configuration YAML (default:
        'config/default_config.yaml').

    Returns
    -------
    dict[str, Any]
        Top-level pipeline configuration dictionary.
    """
    return load_yaml(file_path)
