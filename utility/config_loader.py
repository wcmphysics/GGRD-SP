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


def _set_if_none(d: dict[str, Any], key: str, value: Any) -> None:
    """Set d[key] to value if key is not present or has a None value."""
    if d.get(key) is None:
        d[key] = value


def _inject_spatial_temporal_settings(
    resolved: dict[str, Any],
    global_cfg: dict[str, Any],
    source_tool: str,
    target_tool: str,
    material: str,
) -> None:
    """Inject global pairing and splitting configuration defaults."""
    pairing = resolved.setdefault("pairing", {})
    if isinstance(pairing, dict):
        _set_if_none(pairing, "source_tool", source_tool)
        _set_if_none(pairing, "target_tool", target_tool)
        _set_if_none(pairing, "material", material)

    splitting = resolved.setdefault("splitting", {})
    if isinstance(splitting, dict):
        _set_if_none(splitting, "source_tool", source_tool)
        _set_if_none(splitting, "target_tool", target_tool)


def _inject_preprocessing_and_denoise_settings(
    resolved: dict[str, Any],
    global_cfg: dict[str, Any],
    example_region: str,
    sample_meas_id: str,
) -> None:
    """Inject noise analysis, denoise, and noise plotting defaults."""
    source_tool = str(global_cfg.get("source_tool", "J4"))
    target_tool = str(global_cfg.get("target_tool", "J5"))
    apply_denoise = bool(global_cfg.get("apply_denoise", False))
    data_gen = resolved.get("data_generation", {})
    material = str(
        global_cfg.get("material", data_gen.get("material", "NMG"))
        if isinstance(data_gen, dict)
        else global_cfg.get("material", "NMG")
    )

    # Propagate noise analysis and regional defaults.
    noise_analysis = resolved.setdefault("noise_analysis", {})
    if isinstance(noise_analysis, dict):
        _set_if_none(noise_analysis, "regions", [example_region])

    # Inject default tools and targeted elements into denoise stage.
    denoise_cfg = resolved.setdefault("denoise", {})
    if isinstance(denoise_cfg, dict):
        _set_if_none(denoise_cfg, "apply_denoise", apply_denoise)
        _set_if_none(denoise_cfg, "source_tool", source_tool)
        _set_if_none(denoise_cfg, "target_tool", target_tool)
        _set_if_none(denoise_cfg, "material", material)
        _set_if_none(denoise_cfg, "regions", ["Cl2p", "Ti2p"])

    # Configure noise plot sample measurement and region references.
    noise_plot = resolved.setdefault("noise_plot", {})
    if isinstance(noise_plot, dict):
        noise_plot.setdefault(
            "example",
            {
                "measurement_id": sample_meas_id,
                "region": example_region,
            },
        )


def _inject_model_and_training_settings(
    resolved: dict[str, Any],
    global_cfg: dict[str, Any],
    model_type: str,
    unified_model: bool,
    regions: list[str],
) -> None:
    """Inject dataset, FiLM, Bayesian optimization, and training defaults."""
    source_tool = str(global_cfg.get("source_tool", "J4"))
    target_tool = str(global_cfg.get("target_tool", "J5"))
    use_sliding_window = bool(global_cfg.get("use_sliding_window", False))
    use_film = bool(global_cfg.get("use_film", False))

    # Propagate dataset tools, sliding window, and regions settings.
    dataset = resolved.setdefault("dataset", {})
    if isinstance(dataset, dict):
        _set_if_none(dataset, "source_tool", source_tool)
        _set_if_none(dataset, "target_tool", target_tool)
        _set_if_none(dataset, "use_sliding_window", use_sliding_window)
        _set_if_none(dataset, "regions", regions)

    # Inject FiLM generator defaults and region counts.
    film = resolved.setdefault("film", {})
    if isinstance(film, dict):
        film.setdefault("use_film", use_film)
        film.setdefault("num_regions", len(regions))

    # Convert Bayesian optimization hyperparameter bounds to tuples.
    bo = resolved.setdefault("bayesian_opt", {})
    if isinstance(bo, dict):
        lr_b = bo.get("learning_rate_bounds")
        if isinstance(lr_b, (list, tuple)):
            bo["learning_rate_bounds"] = tuple(lr_b)
        l2_b = bo.get("l2_bounds")
        if isinstance(l2_b, (list, tuple)):
            bo["l2_bounds"] = tuple(l2_b)

    cost = resolved.get("cost", {})
    training = resolved.setdefault("training", {})
    if isinstance(training, dict):
        training.setdefault("model_type", model_type)
        training.setdefault("unified_model", unified_model)
        training.setdefault("use_film", film.get("use_film", use_film))
        training.setdefault("film_config", film)
        if isinstance(cost, dict) and "l2_weight" in cost:
            train_cfg = training.setdefault("train_config", {})
            if isinstance(train_cfg, dict):
                train_cfg.setdefault("l2_weight", cost["l2_weight"])

    prediction = resolved.setdefault("prediction", {})
    if isinstance(prediction, dict):
        prediction.setdefault("source_tool", source_tool)
        prediction.setdefault("target_tool", target_tool)
        prediction.setdefault("film_config", film)


def _inject_evaluation_settings(
    resolved: dict[str, Any],
    global_cfg: dict[str, Any],
) -> None:
    """Inject metrics, quantification, and visualization configuration
    defaults.
    """
    source_tool = str(global_cfg.get("source_tool", "J4"))
    target_tool = str(global_cfg.get("target_tool", "J5"))
    example_region = str(global_cfg.get("example_region", "Ti2p"))
    sample_meas_id = str(
        global_cfg.get("sample_measurement_id", "NMG_M_J4_00000")
    )
    use_sliding_window = bool(global_cfg.get("use_sliding_window", False))
    data_gen = resolved.get("data_generation", {})
    material = str(
        global_cfg.get("material", data_gen.get("material", "NMG"))
        if isinstance(data_gen, dict)
        else global_cfg.get("material", "NMG")
    )

    # Configure metrics report and target normalization flags.
    cost = resolved.get("cost", {})
    norm_by_max = (
        cost.get("normalized_by_target_max", True)
        if isinstance(cost, dict)
        else True
    )
    metrics_rep = resolved.setdefault("metrics_report", {})
    if isinstance(metrics_rep, dict):
        metrics_rep.setdefault("normalized_by_target_max", norm_by_max)
        mse_metric = "MSE_normalized_by_target" if norm_by_max else "MSE"
        existing_metrics = list(
            metrics_rep.get(
                "metrics",
                [
                    "peak_error_percentage",
                    "chebyshev_error_peak_normalized_percentage",
                ],
            )
        )
        if mse_metric not in existing_metrics and "MSE" not in existing_metrics:
            existing_metrics = [mse_metric, *existing_metrics]
        metrics_rep["metrics"] = existing_metrics

    # Propagate visualization stage global tools, sample IDs, and sliding window.
    noise_plot = resolved.setdefault("noise_plot", {})
    vis = resolved.setdefault("visualization", {})
    if isinstance(vis, dict):
        vis.setdefault("source_tool", source_tool)
        vis.setdefault("target_tool", target_tool)
        vis.setdefault("example_region", example_region)
        vis.setdefault("sample_measurement_id", sample_meas_id)
        vis.setdefault("use_sliding_window", use_sliding_window)
        vis.setdefault("noise_plot_config", noise_plot)

    # Propagate material RSF tables into quantification and downstream stages.
    quant = resolved.setdefault("quantification", {})
    denoise_cfg = resolved.get("denoise", {})
    if isinstance(quant, dict):
        quant.setdefault("material", material)
        rsf_tbl = quant.get("rsf_table")
        if rsf_tbl and isinstance(denoise_cfg, dict):
            denoise_cfg.setdefault("rsf_table", rsf_tbl)
        if rsf_tbl and isinstance(metrics_rep, dict):
            metrics_quant = metrics_rep.setdefault("quantification_config", {})
            if isinstance(metrics_quant, dict):
                metrics_quant.setdefault("rsf_table", rsf_tbl)


def _resolve_materials_and_vocab(resolved: dict[str, Any]) -> None:
    """Populate material elements and vocabulary in film config if present."""
    if "materials" in resolved and isinstance(resolved["materials"], dict):
        materials = load_material_elements(resolved)
        film_sec = resolved.setdefault("film", {})
        if isinstance(film_sec, dict):
            film_sec.setdefault("material_elements", materials)
            film_sec.setdefault(
                "element_vocab", extract_element_vocabulary(materials)
            )


def _resolve_material_and_regions(
    resolved: dict[str, Any], global_cfg: dict[str, Any]
) -> tuple[str, list[str]]:
    """Determine global material and regions list for the pipeline."""
    data_gen = resolved.setdefault("data_generation", {})
    material = str(
        global_cfg.get("material", data_gen.get("material", "NMG"))
        if isinstance(data_gen, dict)
        else global_cfg.get("material", "NMG")
    )
    if isinstance(data_gen, dict):
        data_gen.setdefault("material", material)
        regions = data_gen.get(
            "regions", ["Al2p", "Ti2p", "O1s", "C1s", "Cl2p"]
        )
    else:
        regions = ["Al2p", "Ti2p", "O1s", "C1s", "Cl2p"]
    return material, regions


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
    _resolve_materials_and_vocab(resolved)

    global_cfg = resolved.get("global", {})
    source_tool = str(global_cfg.get("source_tool", "J4"))
    target_tool = str(global_cfg.get("target_tool", "J5"))
    material, regions = _resolve_material_and_regions(resolved, global_cfg)

    _inject_spatial_temporal_settings(
        resolved, global_cfg, source_tool, target_tool, material
    )
    _inject_preprocessing_and_denoise_settings(
        resolved,
        global_cfg,
        str(global_cfg.get("example_region", "Ti2p")),
        str(global_cfg.get("sample_measurement_id", "NMG_M_J4_00000")),
    )
    _inject_model_and_training_settings(
        resolved,
        global_cfg,
        str(global_cfg.get("model_type", "resnet")),
        bool(global_cfg.get("unified_model", True)),
        regions,
    )
    _inject_evaluation_settings(resolved, global_cfg)

    return resolved


def load_config(
    file_path: str | Path = "config/config.yaml",
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Load, validate, and resolve pipeline configuration.

    Parameters
    ----------
    file_path : str | Path, optional
        Path to configuration YAML (default:
        'config/config.yaml').
    validate : bool, optional
        Whether to validate configuration using Pydantic schema
        (default: True).

    Returns
    -------
    dict[str, Any]
        Fully resolved top-level pipeline configuration dictionary.
    """
    raw = load_yaml(file_path)
    if validate:
        from utility.config_schema import PipelineConfig

        validated = PipelineConfig.model_validate(raw)
        raw = validated.to_dict()
    return resolve_pipeline_config(raw)
