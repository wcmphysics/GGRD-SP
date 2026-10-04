"""Spectrum pairing module for GGRD-SP.

Provides 1-to-1 matching between source tool and target tool measurement sessions
within a user-defined time threshold, fulfilling Part 2 of GEMINI.md.
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
        - 'method': 'optimal' ('optimal' for Hungarian algorithm, 'greedy' for nearest-first)
        - 'material': None (optional string; if None, pairs within each material separately)
        - 'match_t7_code': True (if True and 't7_code' in metadata, only pair measurements with identical t7_code)
    """
    return {
        "source_tool": "J4",
        "target_tool": "J5",
        "time_threshold_hours": 12.0,
        "method": "optimal",
        "material": None,
        "match_t7_code": True,
    }


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    """Safely convert a pandas Series to datetime64 handling mixed formats and timezones.

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

    match_t7 = config.get("match_t7_code", True)
    if not isinstance(match_t7, bool):
        raise ValueError(f"match_t7_code must be a boolean, got {match_t7}")


def _compute_cost_matrix(
    src_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute pairwise absolute time delta matrix in hours between source and target sessions.

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
    delta_hours = (np.abs(src_times - tgt_times) / np.timedelta64(1, "h")).astype(np.float64)
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
            "scipy.optimize.linear_sum_assignment is unavailable; falling back to greedy pairing algorithm.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _match_sessions_greedy(src_ids, tgt_ids, cost_matrix, threshold)

    # Assign prohibitive penalty cost to pairs exceeding threshold
    penalty = 1e9
    masked_cost = np.where(cost_matrix <= threshold, cost_matrix, penalty)

    row_ind, col_ind = linear_sum_assignment(masked_cost)

    pairs: dict[str, tuple[str, float]] = {}
    for r, c in zip(row_ind, col_ind):
        diff = float(cost_matrix[r, c])
        if diff <= threshold:
            pairs[src_ids[r]] = (tgt_ids[c], diff)

    return pairs


def _match_sessions_greedy(
    src_ids: list[str],
    tgt_ids: list[str],
    cost_matrix: np.ndarray,
    threshold: float,
) -> dict[str, tuple[str, float]]:
    """Perform greedy 1-to-1 matching based on smallest time difference first."""
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


def pair_source_target_spectra(
    meta_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Pair source tool spectra to target tool spectra enforcing 1-to-1 session matching.

    Fulfills Part 2 of GEMINI.md:
    - Performs session-level 1-to-1 matching between source and target measurements.
    - Pairs within each specimen material separately.
    - If t7_code is present, enforces that only measurements with the same t7_code can pair together.
    - Adheres to a configurable time threshold (default 12 hours).
    - Appends target pairing information to the metadata DataFrame:
      'tool_target', 'measurement_id_target', 'spectrum_index_target',
      'time_target', 'time_diff_hours', and 't7_code_target' (if t7_code present).
    - For source tool rows, 'tool_target' is always populated with the target tool name,
      even if no matching target session was found.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Metadata DataFrame containing: 'tool', 'measurement_id', 'time',
        'die', 'region', and 'spectrum_index'.
    config : dict[str, Any] | None, optional
        Configuration dictionary. If None, default settings from
        `get_default_pairing_config()` are used. Custom keys will override defaults.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with pairing columns appended, retaining original index.

    Raises
    ------
    KeyError
        If required columns are missing from meta_df.
    ValueError
        If configuration parameters are invalid.
    """
    merged_config = get_default_pairing_config()
    if config is not None:
        merged_config.update(config)

    _validate_pairing_config(merged_config)

    source_tool: str = merged_config["source_tool"]
    target_tool: str = merged_config["target_tool"]
    threshold_hours: float = float(merged_config["time_threshold_hours"])
    method: str = merged_config["method"]
    material_filter: str | None = merged_config.get("material")
    match_t7_code: bool = bool(merged_config.get("match_t7_code", True))

    required_cols = {"tool", "measurement_id", "time", "die", "region", "spectrum_index"}
    missing_cols = required_cols - set(meta_df.columns)
    if missing_cols:
        raise KeyError(f"meta_df is missing required columns: {sorted(missing_cols)}")

    result_df = meta_df.copy()
    result_df["time"] = _safe_to_datetime(result_df["time"])
    has_t7 = "t7_code" in result_df.columns and match_t7_code

    # Determine materials to pair over
    if "material" in result_df.columns:
        if material_filter:
            materials_to_process = [material_filter]
        else:
            materials_to_process = list(result_df["material"].dropna().unique())
    else:
        materials_to_process = [None]

    # session_pairs: mapping of (src_meas_id) -> (tgt_meas_id, diff_hours)
    all_session_pairs: dict[str, tuple[str, float]] = {}

    for mat in materials_to_process:
        if mat is not None:
            mat_mask = result_df["material"] == mat
            df_mat = result_df[mat_mask]
        else:
            df_mat = result_df

        if has_t7:
            # Enforce that only measurements with the same t7_code can pair together
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

            src_sessions_df = (
                df_pool[df_pool["tool"] == source_tool][["measurement_id", "time"]]
                .drop_duplicates(subset=["measurement_id"])
                .sort_values("time")
            )
            tgt_sessions_df = (
                df_pool[df_pool["tool"] == target_tool][["measurement_id", "time"]]
                .drop_duplicates(subset=["measurement_id"])
                .sort_values("time")
            )

            if not src_sessions_df.empty and not tgt_sessions_df.empty:
                cost_matrix, src_ids, tgt_ids = _compute_cost_matrix(src_sessions_df, tgt_sessions_df)
                if method == "optimal":
                    mat_pairs = _match_sessions_optimal(src_ids, tgt_ids, cost_matrix, threshold_hours)
                else:
                    mat_pairs = _match_sessions_greedy(src_ids, tgt_ids, cost_matrix, threshold_hours)
                all_session_pairs.update(mat_pairs)

    # Build target lookup index: (measurement_id, die, region) -> (spectrum_index, time)
    tgt_spectra = result_df[result_df["tool"] == target_tool]
    tgt_lookup: dict[tuple[str, int, str], tuple[int, pd.Timestamp]] = {
        (str(r["measurement_id"]), int(r["die"]), str(r["region"])): (int(r["spectrum_index"]), pd.Timestamp(r["time"]))
        for _, r in tgt_spectra.iterrows()
    }
    tgt_t7_lookup: dict[str, Any] = (
        dict(zip(tgt_spectra["measurement_id"].astype(str), tgt_spectra["t7_code"]))
        if "t7_code" in result_df.columns
        else {}
    )

    # Prepare target columns aligning with result_df
    n_rows = len(result_df)
    col_tool = [target_tool if t == source_tool else None for t in result_df["tool"]]
    col_meas: list[str | None] = [None] * n_rows
    col_spec: list[int | None] = [None] * n_rows
    col_time: list[pd.Timestamp | None] = [None] * n_rows
    col_diff: list[float | None] = [None] * n_rows
    col_t7: list[str | None] = [None] * n_rows

    if all_session_pairs:
        for row_idx, (_, row) in enumerate(result_df.iterrows()):
            if row["tool"] == source_tool:
                src_meas = str(row["measurement_id"])
                if src_meas in all_session_pairs:
                    tgt_meas, diff_h = all_session_pairs[src_meas]
                    key = (tgt_meas, int(row["die"]), str(row["region"]))
                    if key in tgt_lookup:
                        col_meas[row_idx] = tgt_meas
                        col_spec[row_idx], col_time[row_idx] = tgt_lookup[key]
                        col_diff[row_idx] = diff_h
                        if "t7_code" in result_df.columns:
                            col_t7[row_idx] = tgt_t7_lookup.get(tgt_meas)

    result_df["tool_target"] = pd.Series(col_tool, index=result_df.index, dtype="object")
    result_df["measurement_id_target"] = pd.Series(col_meas, index=result_df.index, dtype="object")
    result_df["spectrum_index_target"] = pd.Series(col_spec, index=result_df.index, dtype="Int64")
    result_df["time_target"] = pd.to_datetime(pd.Series(col_time, index=result_df.index))
    result_df["time_diff_hours"] = pd.Series(col_diff, index=result_df.index, dtype="float64")
    if "t7_code" in result_df.columns:
        result_df["t7_code_target"] = pd.Series(col_t7, index=result_df.index, dtype="object")

    return result_df
