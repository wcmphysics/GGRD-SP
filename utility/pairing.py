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
    from scipy.optimize._lsap import linear_sum_assignment
except (ImportError, OSError):
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


def _match_sessions_optimal(
    src_ids: list[str],
    tgt_ids: list[str],
    cost_matrix: np.ndarray,
    threshold: float,
) -> dict[str, tuple[str, float]]:
    """Perform globally optimal 1-to-1 matching via the Hungarian algorithm.

    Parameters
    ----------
    src_ids : list[str]
        List of source measurement IDs.
    tgt_ids : list[str]
        List of target measurement IDs.
    cost_matrix : np.ndarray
        2D array of pairwise absolute time differences in hours, shape (M, N).
    threshold : float
        Maximum allowed time difference in hours.

    Returns
    -------
    dict[str, tuple[str, float]]
        Mapping of source_measurement_id -> (target_measurement_id, time_diff_hours).
    """
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
    """Perform greedy 1-to-1 matching based on smallest time difference first.

    Parameters
    ----------
    src_ids : list[str]
        List of source measurement IDs.
    tgt_ids : list[str]
        List of target measurement IDs.
    cost_matrix : np.ndarray
        2D array of pairwise absolute time differences in hours, shape (M, N).
    threshold : float
        Maximum allowed time difference in hours.

    Returns
    -------
    dict[str, tuple[str, float]]
        Mapping of source_measurement_id -> (target_measurement_id, time_diff_hours).
    """
    if cost_matrix.size == 0:
        return {}

    candidates: list[tuple[float, int, int]] = []
    m, n = cost_matrix.shape
    for i in range(m):
        for j in range(n):
            diff = float(cost_matrix[i, j])
            if diff <= threshold:
                candidates.append((diff, i, j))

    candidates.sort(key=lambda item: item[0])

    pairs: dict[str, tuple[str, float]] = {}
    used_src: set[int] = set()
    used_tgt: set[int] = set()

    for diff, i, j in candidates:
        if i not in used_src and j not in used_tgt:
            used_src.add(i)
            used_tgt.add(j)
            pairs[src_ids[i]] = (tgt_ids[j], diff)

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

            src_ids = list(src_sessions_df["measurement_id"])
            tgt_ids = list(tgt_sessions_df["measurement_id"])

            if src_ids and tgt_ids:
                # Convert timestamps to UTC datetime64 to compute absolute hour differences
                src_times = pd.to_datetime(src_sessions_df["time"], utc=True)
                tgt_times = pd.to_datetime(tgt_sessions_df["time"], utc=True)

                delta_time = np.abs(src_times.values[:, None] - tgt_times.values[None, :])
                cost_matrix = delta_time / np.timedelta64(1, "h")

                if method == "optimal":
                    mat_pairs = _match_sessions_optimal(src_ids, tgt_ids, cost_matrix, threshold_hours)
                else:
                    mat_pairs = _match_sessions_greedy(src_ids, tgt_ids, cost_matrix, threshold_hours)

                all_session_pairs.update(mat_pairs)

    # Build target lookup index: (measurement_id, die, region) -> (spectrum_index, time)
    tgt_spectra = result_df[result_df["tool"] == target_tool]
    tgt_lookup: dict[tuple[str, int, str], tuple[int, pd.Timestamp]] = {}
    tgt_t7_lookup: dict[str, Any] = {}
    for _, row in tgt_spectra.iterrows():
        key = (str(row["measurement_id"]), int(row["die"]), str(row["region"]))
        tgt_lookup[key] = (int(row["spectrum_index"]), pd.Timestamp(row["time"]))
    if "t7_code" in result_df.columns:
        for _, row in tgt_spectra.drop_duplicates(subset=["measurement_id"]).iterrows():
            tgt_t7_lookup[str(row["measurement_id"])] = row.get("t7_code")

    # Prepare column vectors with size matching result_df length
    n_rows = len(result_df)
    col_tool_target: list[str | None] = [None] * n_rows
    col_meas_target: list[str | None] = [None] * n_rows
    col_spec_target: list[int | None] = [None] * n_rows
    col_time_target: list[pd.Timestamp | None] = [None] * n_rows
    col_diff_hours: list[float | None] = [None] * n_rows
    col_t7_target: list[str | None] = [None] * n_rows

    # Iterate using integer index to avoid label-indexing bugs on sliced/filtered DataFrames
    for row_idx, (_, row) in enumerate(result_df.iterrows()):
        current_tool = row["tool"]

        if current_tool == source_tool:
            # Per user requirement: tool_target is always target_tool for source rows
            col_tool_target[row_idx] = target_tool
            src_meas = str(row["measurement_id"])

            if src_meas in all_session_pairs:
                tgt_meas, diff_h = all_session_pairs[src_meas]
                die = int(row["die"])
                region = str(row["region"])
                lookup_key = (tgt_meas, die, region)

                if lookup_key in tgt_lookup:
                    tgt_spec_idx, tgt_time = tgt_lookup[lookup_key]
                    col_meas_target[row_idx] = tgt_meas
                    col_spec_target[row_idx] = tgt_spec_idx
                    col_time_target[row_idx] = tgt_time
                    col_diff_hours[row_idx] = diff_h
                    if "t7_code" in result_df.columns:
                        col_t7_target[row_idx] = tgt_t7_lookup.get(tgt_meas)

    # Assign columns to result_df aligning explicitly with result_df.index
    result_df["tool_target"] = pd.Series(
        col_tool_target, index=result_df.index, dtype="object"
    )
    result_df["measurement_id_target"] = pd.Series(
        col_meas_target, index=result_df.index, dtype="object"
    )
    result_df["spectrum_index_target"] = pd.Series(
        col_spec_target, index=result_df.index, dtype="Int64"
    )
    result_df["time_target"] = pd.to_datetime(
        pd.Series(col_time_target, index=result_df.index)
    )
    result_df["time_diff_hours"] = pd.Series(
        col_diff_hours, index=result_df.index, dtype="float64"
    )
    if "t7_code" in result_df.columns:
        result_df["t7_code_target"] = pd.Series(
            col_t7_target, index=result_df.index, dtype="object"
        )

    return result_df
