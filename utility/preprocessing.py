"""Preprocessing module for spectral metadata and conditioning features.

Consolidates physical flux and intensity integration, train-split continuous
feature standardization, and canonical categorical tool/region index mapping.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from utility.flux_and_intensity import (
    calculate_flux_and_intensity,
    standardize_metadata_features,
)


def preprocess(
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """Preprocess spectral metadata and conditioning features.

    Calculates physical regional and die-level flux/intensity metrics,
    establishes canonical zero-based integer index mappings for tools
    and regions, and standardizes continuous features on the training
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
        registers 'num_tools' and 'region_index_mapping' into
        config['training'] and config['prediction'].

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int], dict[str, int]]
        - df_meta: Enriched metadata DataFrame with physical metrics,
          categorical indices ('source_tool_index', 'target_tool_index',
          'region_index'), and continuous standardized features
          ('time_standardized', 'total_flux_single_die_standardized'
          if 'split' column is present).
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
    df_res = calculate_flux_and_intensity(ary_intensity, ary_energy, df_res)

    # 2. Establish fixed tool index mapping
    source_tools = set(df_res["tool"].dropna().astype(str).unique())
    target_tools = (
        set(df_res["target_tool"].dropna().astype(str).unique())
        if "target_tool" in df_res.columns
        else set()
    )
    unique_tools = sorted(source_tools | target_tools)
    tool_index_mapping = {
        tool_name: idx for idx, tool_name in enumerate(unique_tools)
    }
    df_res["source_tool_index"] = df_res["tool"].map(tool_index_mapping)
    if "target_tool" in df_res.columns:
        df_res["target_tool_index"] = df_res["target_tool"].map(
            tool_index_mapping
        )

    # 3. Establish fixed region index mapping
    unique_regions = sorted(df_res["region"].dropna().astype(str).unique())
    region_index_mapping = {
        region_name: idx for idx, region_name in enumerate(unique_regions)
    }
    df_res["region_index"] = df_res["region"].map(region_index_mapping)

    # 4. Standardize continuous metadata features if split is present
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

    # 5. Inject resolved mappings into training and prediction configs
    if "training" in cfg and isinstance(cfg["training"], dict):
        cfg["training"]["num_tools"] = len(tool_index_mapping)
        cfg["training"]["region_index_mapping"] = region_index_mapping
    if "prediction" in cfg and isinstance(cfg["prediction"], dict):
        cfg["prediction"]["region_index_mapping"] = region_index_mapping

    return df_res, tool_index_mapping, region_index_mapping
