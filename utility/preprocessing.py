"""Preprocessing module for spectral metadata and conditioning features.

Consolidates physical flux and intensity integration, train-split continuous
feature standardization, and canonical categorical tool/region index mapping.
"""

from __future__ import annotations

import re
from typing import Any
import numpy as np
import pandas as pd

from utility.flux_and_intensity import (
    calculate_flux_and_intensity,
    standardize_metadata_features,
)


def _parse_element_from_region(region_name: str) -> str:
    """Extract chemical element symbol from region string (e.g. 'Ti2p' -> 'Ti')."""
    match = re.match(r"^([A-Z][a-z]?)", str(region_name).strip())
    if match:
        return match.group(1)
    return str(region_name).strip()


def _make_one_hot(idx: int | None, length: int) -> list[float] | None:
    """Create a float one-hot vector or return None if index is None or negative."""
    if idx is None or not (0 <= idx < length):
        return None
    vec = [0.0] * length
    vec[idx] = 1.0
    return vec


def preprocess(
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """Preprocess spectral metadata and conditioning features.

    Calculates physical regional and die-level flux/intensity metrics,
    establishes canonical zero-based integer index mappings for dies, tools,
    regions, and elements, pre-computes categorical one-hot / multi-hot vectors
    directly in df_meta, and standardizes continuous features on the training
    partition without data leakage.

    Parameters
    ----------
    ary_intensity : np.ndarray
        2D array of measured regional spectrum intensities of shape
        (N, N_points).
    ary_energy : np.ndarray
        2D array of binding energies in eV of shape (N, N_points).
    df_meta : pd.DataFrame
        Metadata DataFrame containing at least 'tool', 'measurement_id',
        'die', and 'region'.
    config : dict[str, Any] | None, optional
        Pipeline configuration dictionary. When provided, automatically
        registers categorical counts and mappings into config sections.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int], dict[str, int]]
        - df_meta: Enriched metadata DataFrame with physical metrics,
          categorical indices and one-hot vectors ('die_one_hot',
          'source_tool_one_hot', 'target_tool_one_hot', 'region_one_hot',
          'element_multi_hot'), and continuous standardized features.
        - tool_index_mapping: Dictionary mapping tool names to integer IDs.
        - region_index_mapping: Dictionary mapping region names to integer IDs.
    """
    cfg = config or {}
    required_cols = ["spectrum_index", "measurement_id", "die", "region", "tool"]
    missing_cols = [c for c in required_cols if c not in df_meta.columns]
    if missing_cols:
        raise ValueError(
            f"df_meta missing required columns for preprocessing: {missing_cols}"
        )

    df_res = df_meta.copy()

    # 1. Calculate physical flux and intensity metrics
    df_res = calculate_flux_and_intensity(
        ary_intensity,
        ary_energy,
        df_res,
        config={"standardize_die_flux": False},
    )

    # 2. Establish fixed global die index mapping and one-hot vectors
    unique_dies = sorted(df_res["die"].dropna().unique())
    die_index_mapping = {die_val: idx for idx, die_val in enumerate(unique_dies)}
    num_dies = len(die_index_mapping)
    df_res["die_index"] = df_res["die"].map(die_index_mapping)
    df_res["die_one_hot"] = [
        _make_one_hot(int(x) if pd.notna(x) else None, num_dies)
        for x in df_res["die_index"]
    ]

    # 3. Establish fixed global tool index mapping from df_meta['tool']
    tools_set = set(df_res["tool"].dropna().astype(str).unique())
    if "target_tool" in df_res.columns:
        tools_set.update(df_res["target_tool"].dropna().astype(str).unique())
    unique_tools = sorted(tools_set)
    tool_index_mapping = {
        tool_name: idx for idx, tool_name in enumerate(unique_tools)
    }
    num_tools = len(tool_index_mapping)
    df_res["source_tool_index"] = df_res["tool"].map(tool_index_mapping)
    df_res["source_tool_one_hot"] = [
        _make_one_hot(int(x) if pd.notna(x) else None, num_tools)
        for x in df_res["source_tool_index"]
    ]

    if "target_tool" in df_res.columns:
        df_res["target_tool_index"] = df_res["target_tool"].map(tool_index_mapping)
        df_res["target_tool_one_hot"] = [
            _make_one_hot(int(x) if pd.notna(x) else None, num_tools)
            for x in df_res["target_tool_index"]
        ]

    # 4. Establish fixed global region index mapping and one-hot vectors
    unique_regions = sorted(df_res["region"].dropna().astype(str).unique())
    region_index_mapping = {
        region_name: idx for idx, region_name in enumerate(unique_regions)
    }
    num_regions = len(region_index_mapping)
    df_res["region_index"] = df_res["region"].map(region_index_mapping)
    df_res["region_one_hot"] = [
        _make_one_hot(int(x) if pd.notna(x) else None, num_regions)
        for x in df_res["region_index"]
    ]

    # 5. Establish fixed global element mapping and measurement-level
    # element_multi_hot
    unique_elements = sorted({
        _parse_element_from_region(r) for r in unique_regions
    })
    element_index_mapping = {
        elem_name: idx for idx, elem_name in enumerate(unique_elements)
    }
    num_elements = len(element_index_mapping)

    # Map measurement_id to its element multi-hot vector
    meas_elements: dict[Any, list[float]] = {}
    for meas_id, group in df_res.groupby("measurement_id"):
        m_elems = {
            _parse_element_from_region(r)
            for r in group["region"].dropna().unique()
        }
        vec = [1.0 if elem in m_elems else 0.0 for elem in unique_elements]
        meas_elements[meas_id] = vec
    df_res["element_multi_hot"] = df_res["measurement_id"].map(meas_elements)

    # 6. Standardize continuous metadata features if split is present
    if "split" in df_res.columns and df_res["split"].notna().any():
        fit_split = str(
            cfg.get("fit_split")
            or (
                cfg.get("preprocessing", {}).get("fit_split")
                if isinstance(cfg.get("preprocessing"), dict)
                else None
            )
            or "train"
        )
        df_res = standardize_metadata_features(
            df_res, config={"fit_split": fit_split}
        )

    # 7. Inject resolved mappings and dynamic dimensions into configs
    film_updates = {
        "num_dies": num_dies,
        "num_tools": num_tools,
        "num_regions": num_regions,
        "num_elements": num_elements,
        "region_index_mapping": region_index_mapping,
        "element_index_mapping": element_index_mapping,
        "tool_index_mapping": tool_index_mapping,
        "die_index_mapping": die_index_mapping,
    }
    for sec_key in ("training", "prediction", "film"):
        if sec_key in cfg and isinstance(cfg[sec_key], dict):
            cfg[sec_key].update(film_updates)

    return df_res, tool_index_mapping, region_index_mapping
