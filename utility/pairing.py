"""Spectrum pairing module for GGRD-SP.

Provides 1-to-1 matching between source tool and target tool measurement
sessions within a user-defined time threshold, fulfilling Stage 2 of AGENTS.md.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment
except (ImportError, OSError):
    linear_sum_assignment = None


def get_default_pairing_config() -> dict[str, Any]:
    """Return the default configuration dictionary for spectrum pairing.

    Returns
    -------
    dict[str, Any]
        Dictionary packaging default pairing options:
        - 'source_tool': 'J4'
        - 'target_tool': 'J5'
        - 'time_threshold_hours': 12.0
        - 'method': 'optimal' ('optimal' for Hungarian, 'greedy' for nearest)
        - 'material': 'all' (pairs within each material separately)
        - 'match_t7_code': True (if True and 't7_code' in metadata, only pair
          measurements with identical t7_code)
    """
    return {
        "source_tool": "J4",
        "target_tool": "J5",
        "time_threshold_hours": 12.0,
        "method": "optimal",
        "material": "all",
        "match_t7_code": True,
    }


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    """Safely convert a Series to datetime64 handling mixed formats/timezones.

    Parameters
    ----------
    series : pd.Series
        Series containing datetime strings, objects, or timestamps.

    Returns
    -------
    pd.Series
        Converted datetime Series.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        return pd.to_datetime(series, format="mixed")
    except ValueError:
        return pd.to_datetime(series, format="mixed", utc=True)


def _validate_pairing_config(config: dict[str, Any]) -> None:
    """Validate configuration parameters for spectrum pairing.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration options to validate.

    Raises
    ------
    ValueError
        If source/target tools are invalid, threshold is non-positive,
        or an unsupported method is specified.
    """
    source_tool = config.get("source_tool")
    target_tool = config.get("target_tool")
    if not source_tool or not target_tool:
        raise ValueError("Both 'source_tool' and 'target_tool' must be specified.")
    if source_tool == target_tool:
        raise ValueError(
            f"source_tool and target_tool cannot be identical ('{source_tool}')."
        )

    threshold = config.get("time_threshold_hours", 12.0)
    if threshold is None or threshold <= 0:
        raise ValueError(
            f"time_threshold_hours must be a positive float, got {threshold}"
        )

    method = config.get("method", "optimal")
    if method not in ("optimal", "greedy"):
        raise ValueError(
            f"method must be either 'optimal' or 'greedy', got '{method}'"
        )

    if "material" not in config:
        raise KeyError(
            "Missing required configuration key 'material' in pairing config."
        )
    material = config["material"]
    if material is None:
        raise ValueError(
            "'material' in pairing config cannot be None. "
            "Set 'material': 'all' to pair within each material separately, "
            "or specify a specific material string."
        )
    if not isinstance(material, str) or not material.strip():
        raise ValueError(
            f"'material' in pairing config must be a non-empty string or 'all', "
            f"got {material!r}."
        )

    match_t7 = config.get("match_t7_code", True)
    if not isinstance(match_t7, bool):
        raise ValueError(f"match_t7_code must be a boolean, got {match_t7}")


def _extract_tool_sessions(
    df_pool: pd.DataFrame,
    tool_name: str,
) -> pd.DataFrame:
    """Return unique chronological measurement sessions for a specific tool."""
    return (
        df_pool[df_pool["tool"] == tool_name][["measurement_id", "time"]]
        .drop_duplicates(subset=["measurement_id"])
        .sort_values("time")
    )


def _compute_cost_matrix(
    src_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute pairwise absolute time delta matrix in hours between sessions.

    Parameters
    ----------
    src_df : pd.DataFrame
        Source sessions DataFrame containing 'measurement_id' and 'time'.
    tgt_df : pd.DataFrame
        Target sessions DataFrame containing 'measurement_id' and 'time'.

    Returns
    -------
    tuple[np.ndarray, list[str], list[str]]
        (cost_matrix_hours, src_ids, tgt_ids)
    """
    src_times = pd.to_datetime(src_df["time"], utc=True).values[:, None]
    tgt_times = pd.to_datetime(tgt_df["time"], utc=True).values[None, :]
    delta_hours = (
        np.abs(src_times - tgt_times) / np.timedelta64(1, "h")
    ).astype(np.float64)
    return delta_hours, list(src_df["measurement_id"]), list(tgt_df["measurement_id"])


def _match_sessions_optimal(
    src_ids: list[str],
    tgt_ids: list[str],
    cost_matrix: np.ndarray,
    threshold: float,
) -> dict[str, tuple[str, float]]:
    """Perform globally optimal 1-to-1 matching via the Hungarian algorithm."""
    if cost_matrix.size == 0:
        return {}

    if linear_sum_assignment is None:
        warnings.warn(
            "scipy.optimize.linear_sum_assignment is unavailable; falling back "
            "to greedy pairing algorithm.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _match_sessions_greedy(src_ids, tgt_ids, cost_matrix, threshold)

    # Assign prohibitive penalty cost to pairs exceeding threshold
    penalty = 1e9
    masked_cost = np.where(cost_matrix <= threshold, cost_matrix, penalty)
    row_ind, col_ind = linear_sum_assignment(masked_cost)

    return {
        src_ids[r]: (tgt_ids[c], float(cost_matrix[r, c]))
        for r, c in zip(row_ind, col_ind)
        if float(cost_matrix[r, c]) <= threshold
    }


def _match_sessions_greedy(
    src_ids: list[str],
    tgt_ids: list[str],
    cost_matrix: np.ndarray,
    threshold: float,
) -> dict[str, tuple[str, float]]:
    """Perform greedy 1-to-1 matching by smallest time difference first."""
    if cost_matrix.size == 0:
        return {}

    valid_indices = np.argwhere(cost_matrix <= threshold)
    if valid_indices.size == 0:
        return {}

    # Sort candidates by cost ascending
    costs = cost_matrix[valid_indices[:, 0], valid_indices[:, 1]]
    order = np.argsort(costs)

    pairs: dict[str, tuple[str, float]] = {}
    used_src: set[int] = set()
    used_tgt: set[int] = set()

    for idx in order:
        i, j = int(valid_indices[idx, 0]), int(valid_indices[idx, 1])
        if i not in used_src and j not in used_tgt:
            used_src.add(i)
            used_tgt.add(j)
            pairs[src_ids[i]] = (tgt_ids[j], float(costs[idx]))

    return pairs


def _find_all_session_pairs(
    result_df: pd.DataFrame,
    source_tool: str,
    target_tool: str,
    threshold_hours: float,
    matcher: Any,
    materials: list[str | None],
    has_t7: bool,
) -> dict[str, tuple[str, float]]:
    """Find 1-to-1 session pairs across materials and t7 code partitions.

    Parameters
    ----------
    result_df : pd.DataFrame
        Metadata DataFrame containing session information.
    source_tool : str
        Source tool identifier.
    target_tool : str
        Target tool identifier.
    threshold_hours : float
        Maximum allowable absolute time difference between paired sessions.
    matcher : Any
        Callable matching function (_match_sessions_optimal or greedy).
    materials : list[str | None]
        List of materials to process sequentially.
    has_t7 : bool
        Whether to enforce identical t7_code for paired sessions.

    Returns
    -------
    dict[str, tuple[str, float]]
        Mapping of source_measurement_id -> (target_measurement_id, diff_hours).
    """
    all_session_pairs: dict[str, tuple[str, float]] = {}

    for mat in materials:
        df_mat = (
            result_df[result_df["material"] == mat]
            if mat is not None
            else result_df
        )

        if has_t7:
            # Enforce that only measurements with the same t7_code pair together
            t7_groups = list(df_mat["t7_code"].dropna().unique())
            if df_mat["t7_code"].isna().any():
                t7_groups.append(None)
        else:
            t7_groups = [None]

        for t7_val in t7_groups:
            if t7_val is not None:
                df_pool = df_mat[df_mat["t7_code"] == t7_val]
            elif has_t7:
                df_pool = df_mat[df_mat["t7_code"].isna()]
            else:
                df_pool = df_mat

            src_sessions_df = _extract_tool_sessions(df_pool, source_tool)
            tgt_sessions_df = _extract_tool_sessions(df_pool, target_tool)
            if src_sessions_df.empty or tgt_sessions_df.empty:
                continue

            cost_matrix, src_ids, tgt_ids = _compute_cost_matrix(
                src_sessions_df, tgt_sessions_df
            )
            all_session_pairs.update(
                matcher(src_ids, tgt_ids, cost_matrix, threshold_hours)
            )

    return all_session_pairs


def _assemble_paired_columns(
    result_df: pd.DataFrame,
    all_session_pairs: dict[str, tuple[str, float]],
    source_tool: str,
    target_tool: str,
    threshold_hours: float,
    has_t7: bool,
) -> pd.DataFrame:
    """Populate and append paired target columns to the result DataFrame.

    Parameters
    ----------
    result_df : pd.DataFrame
        Copied metadata DataFrame.
    all_session_pairs : dict[str, tuple[str, float]]
        Mapping of source session ID to (target session ID, diff_hours).
    source_tool : str
        Source tool identifier.
    target_tool : str
        Target tool identifier.
    threshold_hours : float
        Pairing threshold for warning message formatting.
    has_t7 : bool
        Whether t7_code is present in metadata and used for pairing.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with target columns populated.
    """
    # Build target lookup: (measurement_id, die, region) -> (spec_idx, time)
    tgt_spectra = result_df[result_df["tool"] == target_tool]
    tgt_lookup: dict[tuple[str, int, str], tuple[int, pd.Timestamp]] = {
        (str(r["measurement_id"]), int(r["die"]), str(r["region"])): (
            int(r["spectrum_index"]),
            pd.Timestamp(r["time"]),
        )
        for _, r in tgt_spectra.iterrows()
    }
    tgt_t7_lookup: dict[str, Any] = (
        dict(zip(tgt_spectra["measurement_id"].astype(str), tgt_spectra["t7_code"]))
        if "t7_code" in result_df.columns
        else {}
    )

    n_rows = len(result_df)
    col_tool = [
        target_tool if t == source_tool else None for t in result_df["tool"]
    ]
    col_meas: list[str | None] = [None] * n_rows
    col_spec: list[int | None] = [None] * n_rows
    col_time: list[pd.Timestamp | None] = [None] * n_rows
    col_diff: list[float | None] = [None] * n_rows
    col_t7: list[str | None] = [None] * n_rows

    src_mask = result_df["tool"] == source_tool
    if src_mask.any() and not all_session_pairs:
        warnings.warn(
            f"No session pairs could be formed between source tool "
            f"'{source_tool}' and target tool '{target_tool}' within time "
            f"threshold of {threshold_hours} hours. All target metadata "
            "columns will remain null."
        )

    for row_idx, (_, row) in enumerate(result_df.iterrows()):
        if row["tool"] != source_tool:
            continue
        src_meas = str(row["measurement_id"])
        if src_meas not in all_session_pairs:
            continue
        tgt_meas, diff_h = all_session_pairs[src_meas]
        key = (tgt_meas, int(row["die"]), str(row["region"]))
        if key not in tgt_lookup:
            continue

        col_meas[row_idx] = tgt_meas
        col_spec[row_idx], col_time[row_idx] = tgt_lookup[key]
        col_diff[row_idx] = diff_h
        if "t7_code" in result_df.columns:
            col_t7[row_idx] = tgt_t7_lookup.get(tgt_meas)

    idx = result_df.index
    result_df["target_tool"] = pd.Series(col_tool, index=idx, dtype="object")
    result_df["target_measurement_id"] = pd.Series(
        col_meas, index=idx, dtype="object"
    )
    result_df["target_spectrum_index"] = pd.Series(
        col_spec, index=idx, dtype="Int64"
    )
    result_df["target_time"] = pd.to_datetime(pd.Series(col_time, index=idx))
    result_df["time_diff_hours"] = pd.Series(
        col_diff, index=idx, dtype="float64"
    )
    if "t7_code" in result_df.columns:
        result_df["target_t7_code"] = pd.Series(
            col_t7, index=idx, dtype="object"
        )

    return result_df


def _prepare_pairing_setup(
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any], list[str | None], bool, Any]:
    """Validate inputs, resolve config, and prepare materials list."""
    if config is None or "material" not in config:
        raise KeyError(
            "Missing required configuration key 'material' in pairing config."
        )
    if config["material"] is None:
        raise ValueError(
            "'material' in pairing config cannot be None. "
            "Set 'material': 'all' to pair within each material separately, "
            "or specify a specific material string."
        )

    merged_config = get_default_pairing_config()
    merged_config.update(config)
    _validate_pairing_config(merged_config)

    required_cols = {
        "tool",
        "measurement_id",
        "time",
        "die",
        "region",
        "spectrum_index",
    }
    missing_cols = required_cols - set(df_meta.columns)
    if missing_cols:
        raise KeyError(
            f"df_meta is missing required columns: {sorted(missing_cols)}"
        )

    result_df = df_meta.copy()
    result_df["time"] = _safe_to_datetime(result_df["time"])
    has_t7 = "t7_code" in result_df.columns and bool(
        merged_config.get("match_t7_code", True)
    )
    matcher = (
        _match_sessions_optimal
        if merged_config["method"] == "optimal"
        else _match_sessions_greedy
    )

    material_filter = str(merged_config["material"]).strip()
    if material_filter.lower() == "all":
        materials: list[str | None] = (
            list(result_df["material"].dropna().unique())
            if "material" in result_df.columns
            else [None]
        )
    else:
        materials = [material_filter]

    return result_df, merged_config, materials, has_t7, matcher


def pair_source_target_spectra(
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Pair source tool spectra to target tool spectra with 1-to-1 matching.

    Fulfills Stage 2 of AGENTS.md:
    - Performs session-level 1-to-1 matching between source and target tools.
    - Pairs within each specimen material separately.
    - If t7_code is present, enforces identical t7_code for paired sessions.
    - Adheres to a configurable time threshold (default 12 hours).
    - Appends target pairing information to the metadata DataFrame:
      'target_tool', 'target_measurement_id', 'target_spectrum_index',
      'target_time', 'time_diff_hours', and 'target_t7_code' (if present).
    - For source tool rows, 'target_tool' is always populated with the target
      tool name, even if no matching target session was found.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata DataFrame containing: 'tool', 'measurement_id', 'time',
        'die', 'region', and 'spectrum_index'.
    config : dict[str, Any]
        Configuration dictionary. Must contain required key 'material'
        ('all' to pair each material separately, or a specific material string).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with pairing columns appended, retaining original index.

    Raises
    ------
    KeyError
        If required columns are missing from df_meta.
    ValueError
        If configuration parameters are invalid.
    """
    result_df, cfg, materials, has_t7, matcher = _prepare_pairing_setup(
        df_meta, config
    )
    source_tool = cfg["source_tool"]
    target_tool = cfg["target_tool"]
    threshold_hours = float(cfg["time_threshold_hours"])

    all_session_pairs = _find_all_session_pairs(
        result_df,
        source_tool,
        target_tool,
        threshold_hours,
        matcher,
        materials,
        has_t7,
    )
    return _assemble_paired_columns(
        result_df,
        all_session_pairs,
        source_tool,
        target_tool,
        threshold_hours,
        has_t7,
    )
