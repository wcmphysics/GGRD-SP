"""Preprocessing module for spectral metadata and conditioning features.

Consolidates physical flux and intensity integration, train-split continuous
feature standardization, and canonical categorical tool/region index mapping.
"""

from dataclasses import dataclass
import re
from typing import Any, Self
import numpy as np
import pandas as pd

from utility.flux_and_intensity import (
    calculate_flux_and_intensity,
    standardize_metadata_features,
)


@dataclass
class DatasetMetadata:
    """Typed container for categorical dimensions and index mappings."""

    num_dies: int
    num_tools: int
    num_regions: int
    num_elements: int
    die_index_mapping: dict[int, int]
    tool_index_mapping: dict[str, int]
    region_index_mapping: dict[str, int]
    element_index_mapping: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata mappings to dictionary for FiLM/model configuration."""
        return {
            "num_dies": self.num_dies,
            "num_tools": self.num_tools,
            "num_regions": self.num_regions,
            "num_elements": self.num_elements,
            "die_index_mapping": self.die_index_mapping,
            "tool_index_mapping": self.tool_index_mapping,
            "region_index_mapping": self.region_index_mapping,
            "element_index_mapping": self.element_index_mapping,
        }


class PreprocessOutput(tuple):
    """3-tuple of (df_meta, tool_index_mapping, region_index_mapping) with dataset_metadata."""

    dataset_metadata: DatasetMetadata
    metadata: DatasetMetadata

    def __new__(
        cls,
        df_meta: pd.DataFrame,
        tool_index_mapping: dict[str, int],
        region_index_mapping: dict[str, int],
        dataset_metadata: DatasetMetadata,
    ) -> Self:
        obj = super().__new__(
            cls, (df_meta, tool_index_mapping, region_index_mapping)
        )
        obj.dataset_metadata = dataset_metadata
        obj.metadata = dataset_metadata
        return obj


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


def _compute_die_mappings(df_res: pd.DataFrame) -> tuple[dict[int, int], int]:
    """Establish fixed global die index mapping and one-hot vectors."""
    unique_dies = sorted(df_res["die"].dropna().unique())
    die_index_mapping = {die_val: idx for idx, die_val in enumerate(unique_dies)}
    num_dies = len(die_index_mapping)
    df_res["die_index"] = df_res["die"].map(die_index_mapping)
    df_res["die_one_hot"] = [
        _make_one_hot(int(x) if pd.notna(x) else None, num_dies)
        for x in df_res["die_index"]
    ]
    return die_index_mapping, num_dies


def _compute_tool_mappings(df_res: pd.DataFrame) -> tuple[dict[str, int], int]:
    """Establish fixed global tool index mapping from tool and target_tool."""
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
    return tool_index_mapping, num_tools


def _compute_region_mappings(
    df_res: pd.DataFrame,
) -> tuple[dict[str, int], int, list[str]]:
    """Establish fixed global region index mapping and one-hot vectors."""
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
    return region_index_mapping, num_regions, unique_regions


def _extract_film_material_elements(cfg: dict[str, Any]) -> dict[str, set[str]]:
    """Extract material -> element set from film.material_elements."""
    film_cfg = cfg.get("film") if isinstance(cfg.get("film"), dict) else {}
    raw_mat_elems = film_cfg.get("material_elements")
    if not isinstance(raw_mat_elems, dict):
        return {}
    return {
        str(mat): {str(e).strip() for e in elems}
        for mat, elems in raw_mat_elems.items()
        if isinstance(elems, (list, tuple, set))
    }


def _extract_material_element_map(cfg: dict[str, Any]) -> dict[str, set[str]]:
    """Extract material -> element mapping from pipeline configuration."""
    from_film = _extract_film_material_elements(cfg)
    if from_film:
        return from_film
    data_gen = (
        cfg.get("data_generation")
        if isinstance(cfg.get("data_generation"), dict)
        else {}
    )
    raw_mats = data_gen.get("materials")
    if not isinstance(raw_mats, dict):
        return {}
    return {
        str(mat): {_parse_element_from_region(r) for r in spec["regions"]}
        for mat, spec in raw_mats.items()
        if isinstance(spec, dict) and isinstance(spec.get("regions"), list)
    }


def _resolve_measurement_elements(
    group: pd.DataFrame,
    has_material: bool,
    mat_elem_map: dict[str, set[str]],
) -> set[str]:
    """Determine element set for one measurement session."""
    mat_val = str(group["material"].iloc[0]) if has_material else ""
    if mat_val and mat_val in mat_elem_map:
        return mat_elem_map[mat_val]
    return {
        _parse_element_from_region(r)
        for r in group["region"].dropna().unique()
    }


def _compute_element_mappings(
    df_res: pd.DataFrame,
    unique_regions: list[str],
    cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, int], int]:
    """Establish global element mapping and measurement element_multi_hot."""
    mat_elem_map = _extract_material_element_map(cfg or {})
    elem_set = {_parse_element_from_region(r) for r in unique_regions}
    for elems in mat_elem_map.values():
        elem_set.update(elems)
    unique_elements = sorted(elem_set)
    element_index_mapping = {
        elem_name: idx for idx, elem_name in enumerate(unique_elements)
    }
    num_elements = len(element_index_mapping)

    has_material = "material" in df_res.columns
    meas_elements: dict[Any, list[float]] = {}
    for meas_id, group in df_res.groupby("measurement_id"):
        m_elems = _resolve_measurement_elements(
            group, has_material, mat_elem_map
        )
        meas_elements[meas_id] = [
            1.0 if elem in m_elems else 0.0 for elem in unique_elements
        ]
    df_res["element_multi_hot"] = df_res["measurement_id"].map(meas_elements)
    return element_index_mapping, num_elements


def _standardize_if_split_present(
    df_res: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Standardize continuous metadata features if split column has valid values."""
    if "split" not in df_res.columns or not df_res["split"].notna().any():
        return df_res
    fit_split = str(
        cfg.get("fit_split")
        or (
            cfg.get("preprocessing", {}).get("fit_split")
            if isinstance(cfg.get("preprocessing"), dict)
            else None
        )
        or "train"
    )
    return standardize_metadata_features(
        df_res, config={"fit_split": fit_split}
    )


def preprocess(
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> PreprocessOutput:
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

    # 2. Establish fixed global categorical index mappings and vectors
    die_index_mapping, num_dies = _compute_die_mappings(df_res)
    tool_index_mapping, num_tools = _compute_tool_mappings(df_res)
    region_index_mapping, num_regions, unique_regions = _compute_region_mappings(
        df_res
    )
    element_index_mapping, num_elements = _compute_element_mappings(
        df_res, unique_regions, cfg
    )

    # 3. Standardize continuous metadata features if split is present
    df_res = _standardize_if_split_present(df_res, cfg)

    # 4. Construct typed dataset metadata
    dataset_metadata = DatasetMetadata(
        num_dies=num_dies,
        num_tools=num_tools,
        num_regions=num_regions,
        num_elements=num_elements,
        die_index_mapping=die_index_mapping,
        tool_index_mapping=tool_index_mapping,
        region_index_mapping=region_index_mapping,
        element_index_mapping=element_index_mapping,
    )

    # 5. Backward compatibility: update configs if provided
    film_updates = dataset_metadata.to_dict()
    for sec_key in ("training", "prediction", "film"):
        if sec_key in cfg and isinstance(cfg[sec_key], dict):
            cfg[sec_key].update(film_updates)

    return PreprocessOutput(
        df_res, tool_index_mapping, region_index_mapping, dataset_metadata
    )
