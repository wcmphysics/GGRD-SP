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


def resolve_pipeline_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve and inject cross-section configuration defaults.

    Extracts materials and element vocabulary and propagates global defaults
    (tools, sliding window settings, regions, FiLM config) into individual stage
    configuration dictionaries without mutating explicit overrides.

    Parameters
    ----------
    config : dict[str, Any]
        Raw configuration dictionary loaded from YAML.

    Returns
    -------
    dict[str, Any]
        Resolved configuration dictionary with fully populated stage configs.
    """
    resolved = dict(config)

    # 1. Resolve material elements and element vocabulary
    if "materials" in resolved and isinstance(resolved["materials"], dict):
        materials = load_material_elements(resolved)
        element_vocab = extract_element_vocabulary(materials)
        film_sec = resolved.setdefault("film", {})
        if isinstance(film_sec, dict):
            film_sec.setdefault("material_elements", materials)
            film_sec.setdefault("element_vocab", element_vocab)

    # 2. Extract global settings
    global_cfg = resolved.get("global", {})
    source_tool = str(global_cfg.get("source_tool", "J4"))
    target_tool = str(global_cfg.get("target_tool", "J5"))
    model_type = str(global_cfg.get("model_type", "resnet"))
    use_sliding_window = bool(global_cfg.get("use_sliding_window", False))
    use_film = bool(global_cfg.get("use_film", False))
    unified_model = bool(global_cfg.get("unified_model", True))
    example_region = str(global_cfg.get("example_region", "Ti2p"))
    sample_measurement_id = str(
        global_cfg.get("sample_measurement_id", "NMG_M_J4_00000")
    )

    # 3. Extract regions
    data_gen = resolved.get("data_generation", {})
    regions = data_gen.get("regions", ["Al2p", "Ti2p", "O1s", "C1s", "Cl2p"])

    # 4. Inject into pairing
    pairing = resolved.setdefault("pairing", {})
    if isinstance(pairing, dict):
        pairing.setdefault("source_tool", source_tool)
        pairing.setdefault("target_tool", target_tool)

    # 5. Inject into splitting
    splitting = resolved.setdefault("splitting", {})
    if isinstance(splitting, dict):
        splitting.setdefault("source_tool", source_tool)
        splitting.setdefault("target_tool", target_tool)

    # 6. Inject into dataset
    dataset = resolved.setdefault("dataset", {})
    if isinstance(dataset, dict):
        dataset.setdefault("source_tool", source_tool)
        dataset.setdefault("target_tool", target_tool)
        dataset.setdefault("use_sliding_window", use_sliding_window)
        dataset.setdefault("regions", regions)

    # 7. Inject into noise_analysis and noise_plot
    noise_analysis = resolved.setdefault("noise_analysis", {})
    if isinstance(noise_analysis, dict):
        noise_analysis.setdefault("regions", [example_region])

    noise_plot = resolved.setdefault("noise_plot", {})
    if isinstance(noise_plot, dict):
        noise_plot.setdefault(
            "example",
            {
                "measurement_id": sample_measurement_id,
                "region": example_region,
            },
        )

    # 8. Inject into film
    film = resolved.setdefault("film", {})
    if isinstance(film, dict):
        film.setdefault("use_film", use_film)
        film.setdefault("num_regions", len(regions))

    # 9. Format bayesian_opt bounds
    bo = resolved.setdefault("bayesian_opt", {})
    if isinstance(bo, dict):
        lr_b = bo.get("learning_rate_bounds")
        if isinstance(lr_b, (list, tuple)):
            bo["learning_rate_bounds"] = tuple(lr_b)
        l2_b = bo.get("l2_bounds")
        if isinstance(l2_b, (list, tuple)):
            bo["l2_bounds"] = tuple(l2_b)

    # 10. Inject into training
    training = resolved.setdefault("training", {})
    if isinstance(training, dict):
        training.setdefault("model_type", model_type)
        training.setdefault("unified_model", unified_model)
        training.setdefault("use_film", film.get("use_film", use_film))
        training.setdefault("film_config", film)

    # 11. Inject into prediction
    prediction = resolved.setdefault("prediction", {})
    if isinstance(prediction, dict):
        prediction.setdefault("source_tool", source_tool)
        prediction.setdefault("target_tool", target_tool)
        prediction.setdefault("film_config", film)

    # 12. Inject into metrics_report
    cost = resolved.get("cost", {})
    norm_by_max = cost.get("normalized_by_target_max", True)
    metrics_rep = resolved.setdefault("metrics_report", {})
    if isinstance(metrics_rep, dict):
        metrics_rep.setdefault("normalized_by_target_max", norm_by_max)
        mse_metric = "MSE_normalized_by_target" if norm_by_max else "MSE"
        existing_metrics = list(
            metrics_rep.get("metrics", ["peak_err%", "max%err"])
        )
        if mse_metric not in existing_metrics and "MSE" not in existing_metrics:
            existing_metrics = [mse_metric, *existing_metrics]
        metrics_rep["metrics"] = existing_metrics

    # 13. Inject into visualization
    vis = resolved.setdefault("visualization", {})
    if isinstance(vis, dict):
        vis.setdefault("source_tool", source_tool)
        vis.setdefault("target_tool", target_tool)
        vis.setdefault("example_region", example_region)
        vis.setdefault("sample_measurement_id", sample_measurement_id)
        vis.setdefault("use_sliding_window", use_sliding_window)
        vis.setdefault("noise_plot_config", noise_plot)

    return resolved


def load_default_config(
    file_path: str | Path = "config/default_config.yaml",
) -> dict[str, Any]:
    """Load and resolve default pipeline configuration.

    Parameters
    ----------
    file_path : str | Path, optional
        Path to default configuration YAML (default:
        'config/default_config.yaml').

    Returns
    -------
    dict[str, Any]
        Fully resolved top-level pipeline configuration dictionary.
    """
    raw = load_yaml(file_path)
    return resolve_pipeline_config(raw)
