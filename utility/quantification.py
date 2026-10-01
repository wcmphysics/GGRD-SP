"""Quantification and Shirley background subtraction module for GGRD-SP.

Provides Shirley background subtraction, peak area integration, and atomic
percentage calculation fulfilling Part 5 of GEMINI.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Backward compatibility for trapezoidal numerical integration:
# In older NumPy versions (< 2.0), this function is named numpy.trapz.
# In NumPy >= 2.0, it was renamed to numpy.trapezoid.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]


# Standard Scofield relative sensitivity factors (RSF) for Al-Ka X-ray excitation
DEFAULT_SCOFIELD_RSF: dict[str, float] = {
    "C1s": 1.000,
    "Al2p": 0.537,
    "Ti2p": 7.910,
    "O1s": 2.930,
    "Cl2p": 2.285,
}


def get_default_quantification_config() -> dict[str, Any]:
    """Return default configuration options for atomic percentage quantification.

    Returns
    -------
    dict[str, Any]
        Dictionary packaging quantification parameters:
        - 'material': 'NMG'
        - 'measurement_id': None (if None, calculates for all sessions in meta_df)
        - 'rsf_dict': dict mapping region name to relative sensitivity factor
        - 'ti2p_auto_endpoints': True (search for low/high BE minima around Ti2p peak)
        - 'ti2p_smooth_endpoints_search': False (optional smoothing for minima search)
        - 'energy_limits': None (optional dict[str, tuple[float, float]])
        - 'max_iter': 50
        - 'tol': 1e-5
    """
    return {
        "material": "NMG",
        "measurement_id": None,
        "rsf_dict": dict(DEFAULT_SCOFIELD_RSF),
        "ti2p_auto_endpoints": True,
        "ti2p_smooth_endpoints_search": False,
        "energy_limits": None,
        "max_iter": 50,
        "tol": 1e-5,
    }


def _find_ti2p_endpoints(
    energy: np.ndarray,
    intensity: np.ndarray,
    smooth_search: bool = False,
) -> tuple[int, int]:
    """Determine low and high BE endpoints for Ti2p via directional minima search.

    Procedure:
    1. Locate maximum intensity index in the regional spectrum.
    2. Search in the high BE direction for the lowest intensity -> E2 index.
    3. Search in the low BE direction for the lowest intensity -> E1 index.

    Parameters
    ----------
    energy : np.ndarray
        Binding energy array (assumed ascending).
    intensity : np.ndarray
        Intensity array.
    smooth_search : bool, optional
        Whether to apply a 3-point moving average filter with edge-padding.
        Default is False.

    Returns
    -------
    tuple[int, int]
        (idx_e1, idx_e2) where idx_e1 <= idx_e2.
    """
    n_pts = len(intensity)
    if n_pts < 2:
        return 0, n_pts - 1

    if smooth_search and n_pts >= 3:
        # Use edge padding to prevent artificial boundary drop-off from zero-padding
        kernel = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        padded = np.pad(intensity, 1, mode="edge")
        i_search = np.convolve(padded, kernel, mode="valid")
    else:
        i_search = intensity

    idx_max = int(np.argmax(i_search))

    # Low BE direction: indices 0 to idx_max (lower energies)
    idx_e1 = int(np.argmin(i_search[: idx_max + 1]))

    # High BE direction: indices idx_max to n_pts - 1 (higher energies)
    idx_e2 = idx_max + int(np.argmin(i_search[idx_max:]))

    if idx_e1 >= idx_e2:
        # Fallback to endpoints if indices degenerate
        return 0, n_pts - 1

    return idx_e1, idx_e2


def calculate_shirley_background(
    energy: np.ndarray,
    intensity: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, float]:
    """Calculate the Shirley background and net integrated area for a spectrum.

    Parameters
    ----------
    energy : np.ndarray
        1D array of binding energies.
    intensity : np.ndarray
        1D array of measured intensities.
    config : dict[str, Any] | None, optional
        Dictionary bundling options:
        - 'region' (str): Name of the spectral region (e.g. 'Ti2p').
        - 'ti2p_auto_endpoints' (bool): Whether to auto-detect Ti2p endpoints (default True).
        - 'ti2p_smooth_endpoints_search' (bool): Smooth search array (default False).
        - 'energy_limits' (tuple[float, float] | None): Custom (e_min, e_max) limits.
        - 'max_iter' (int): Maximum Shirley iterations (default 50).
        - 'tol' (float): Convergence threshold (default 1e-5).

    Returns
    -------
    tuple[np.ndarray, float]
        - background: 1D array of the same shape as intensity. Outside active [E1, E2],
          background equals intensity B(E) = I(E).
        - net_area: Integrated net peak area integral(I(E) - B(E)) dE.

    Raises
    ------
    ValueError
        If energy and intensity arrays have mismatched lengths.
    """
    if len(energy) != len(intensity):
        raise ValueError(
            f"energy and intensity arrays must have matching lengths, got {len(energy)} vs {len(intensity)}"
        )

    cfg = config or {}
    region: str = cfg.get("region", "")
    ti2p_auto: bool = cfg.get("ti2p_auto_endpoints", True)
    ti2p_smooth: bool = cfg.get("ti2p_smooth_endpoints_search", False)
    custom_limits: tuple[float, float] | None = cfg.get("energy_limits")
    max_iter: int = int(cfg.get("max_iter", 50))
    tol: float = float(cfg.get("tol", 1e-5))

    n_pts = len(intensity)
    if n_pts < 2:
        return intensity.copy(), 0.0

    # Ensure internal calculations operate on ascending energy order
    is_descending = energy[0] > energy[-1]
    if is_descending:
        e_work = energy[::-1]
        i_work = intensity[::-1]
    else:
        e_work = energy
        i_work = intensity

    # Determine endpoint indices [idx_1, idx_2]
    # Priority: explicit custom_limits take precedence over auto-detection
    if custom_limits is not None:
        e_min, e_max = min(custom_limits), max(custom_limits)
        valid_indices = np.where((e_work >= e_min) & (e_work <= e_max))[0]
        if len(valid_indices) >= 2:
            idx_1, idx_2 = int(valid_indices[0]), int(valid_indices[-1])
        else:
            idx_1, idx_2 = 0, n_pts - 1
    elif region == "Ti2p" and ti2p_auto:
        idx_1, idx_2 = _find_ti2p_endpoints(e_work, i_work, smooth_search=ti2p_smooth)
    else:
        idx_1, idx_2 = 0, n_pts - 1

    # Extract sub-interval
    e_sub = e_work[idx_1 : idx_2 + 1]
    i_sub = i_work[idx_1 : idx_2 + 1]

    # Initial linear background connecting endpoints
    i_1 = float(i_sub[0])
    i_2 = float(i_sub[-1])
    e_span = float(e_sub[-1] - e_sub[0])

    if e_span == 0.0 or len(e_sub) < 2:
        b_sub = i_sub.copy()
        net_area = 0.0
    else:
        # Linear initialization
        b_sub = i_1 + (i_2 - i_1) * (e_sub - e_sub[0]) / e_span

        # Iterative Shirley background calculation
        for _ in range(max_iter):
            peak = np.maximum(0.0, i_sub - b_sub)
            total_peak_area = float(np.trapz(peak, e_sub))  # type: ignore[attr-defined]

            if total_peak_area <= 0.0:
                break

            # Cumulative area from e_sub[0] to each e_sub[k]
            dx = np.diff(e_sub)
            trap_steps = 0.5 * (peak[:-1] + peak[1:]) * dx
            cum_area = np.concatenate(([0.0], np.cumsum(trap_steps)))

            b_new = i_1 + (i_2 - i_1) * (cum_area / total_peak_area)
            max_change = float(np.max(np.abs(b_new - b_sub)))
            b_sub = b_new

            if max_change < tol:
                break

        net_area = float(np.trapz(np.maximum(0.0, i_sub - b_sub), e_sub))  # type: ignore[attr-defined]

    # Construct full background array: B(E) = I(E) outside [E1, E2]
    b_work = i_work.copy()
    b_work[idx_1 : idx_2 + 1] = b_sub

    # If original input was descending, flip background back to match original order
    if is_descending:
        background = b_work[::-1]
    else:
        background = b_work

    return background, max(0.0, net_area)


def calculate_atomic_percentages(
    ary_energy: np.ndarray,
    ary_intensity: np.ndarray,
    meta_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate atomic percentages for all elements in wide format (per-die and summary).

    Fulfills Part 5 of GEMINI.md:
    - Calculates net peak area via Shirley background subtraction for each region.
    - Supports Ti2p auto-endpoint detection based on directional minima search.
    - Normalizes net areas by Relative Sensitivity Factors (RSF).
    - Returns per-die atomic percentages and summary statistics across dies.

    Parameters
    ----------
    ary_energy : np.ndarray
        2D array of binding energies, shape (N_total, N_points).
    ary_intensity : np.ndarray
        2D array of intensities, shape (N_total, N_points).
    meta_df : pd.DataFrame
        Metadata DataFrame describing spectra rows.
    config : dict[str, Any] | None, optional
        Dictionary bundling options:
        - 'measurement_id' (str | None): Filter to specific measurement session.
        - 'material' (str | None): Target material (default 'NMG').
        - 'rsf_dict' (dict[str, float]): Element RSF mapping.
        - 'ti2p_auto_endpoints' (bool): Auto endpoints for Ti2p (default True).
        - 'ti2p_smooth_endpoints_search' (bool): Smooth search for Ti2p minima (default False).
        - 'energy_limits' (dict[str, tuple[float, float]] | None): Custom energy limits.
        - 'max_iter' (int): Shirley max iterations (default 50).
        - 'tol' (float): Shirley tolerance (default 1e-5).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - df_per_die: Wide DataFrame with per-die atomic percentages.
        - df_summary: DataFrame with mean and std across dies for each measurement.

    Raises
    ------
    ValueError
        If no matching data is found, required columns are missing, or RSF is non-positive.
    """
    cfg = get_default_quantification_config()
    if config is not None:
        cfg.update(config)

    target_meas: str | None = cfg.get("measurement_id")
    material: str | None = cfg.get("material")
    rsf_dict: dict[str, float] = cfg.get("rsf_dict", DEFAULT_SCOFIELD_RSF)
    ti2p_auto: bool = cfg.get("ti2p_auto_endpoints", True)
    ti2p_smooth: bool = cfg.get("ti2p_smooth_endpoints_search", False)
    energy_limits: dict[str, tuple[float, float]] | None = cfg.get("energy_limits")
    max_iter: int = int(cfg.get("max_iter", 50))
    tol: float = float(cfg.get("tol", 1e-5))

    required_cols = {"spectrum_index", "measurement_id", "die", "region"}
    missing = required_cols - set(meta_df.columns)
    if missing:
        raise ValueError(f"meta_df is missing required columns: {sorted(missing)}")

    filtered_df = meta_df.copy()
    if target_meas is not None:
        filtered_df = filtered_df[filtered_df["measurement_id"] == target_meas]
    if material is not None and "material" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["material"] == material]

    if filtered_df.empty:
        raise ValueError(
            f"No spectra found matching criteria: measurement_id={target_meas}, material={material}"
        )

    # Group by measurement session
    meas_groups = filtered_df.groupby("measurement_id", sort=False)

    per_die_records: list[dict[str, Any]] = []

    for meas_id, m_df in meas_groups:
        mat_val = m_df["material"].iloc[0] if "material" in m_df.columns else "NMG"
        die_groups = m_df.groupby("die", sort=True)

        for die_idx, d_df in die_groups:
            # Calculate scaled area for each region
            scaled_areas: dict[str, float] = {}

            for _, row in d_df.iterrows():
                region = str(row["region"])
                spec_idx = int(row["spectrum_index"])
                e_arr = ary_energy[spec_idx]
                i_arr = ary_intensity[spec_idx]

                shirley_cfg = {
                    "region": region,
                    "ti2p_auto_endpoints": ti2p_auto,
                    "ti2p_smooth_endpoints_search": ti2p_smooth,
                    "energy_limits": energy_limits.get(region) if energy_limits else None,
                    "max_iter": max_iter,
                    "tol": tol,
                }
                _, net_area = calculate_shirley_background(e_arr, i_arr, shirley_cfg)

                rsf = float(rsf_dict.get(region, 1.0))
                if rsf <= 0.0:
                    raise ValueError(f"RSF for region '{region}' must be positive, got {rsf}")

                scaled_areas[region] = max(0.0, net_area) / rsf

            total_scaled_area = sum(scaled_areas.values())

            # Assemble row in wide format
            record: dict[str, Any] = {
                "measurement_id": meas_id,
                "material": mat_val,
                "die": int(die_idx),
            }

            for reg, scaled_a in scaled_areas.items():
                if total_scaled_area > 0.0:
                    at_pct = (scaled_a / total_scaled_area) * 100.0
                else:
                    at_pct = 0.0
                record[f"{reg}_at%"] = round(at_pct, 4)

            per_die_records.append(record)

    df_per_die = pd.DataFrame(per_die_records)

    # Calculate summary statistics (mean and std across dies per measurement session)
    at_cols = [c for c in df_per_die.columns if c.endswith("_at%")]

    summary_records: list[dict[str, Any]] = []
    for meas_id, m_df in df_per_die.groupby("measurement_id", sort=False):
        mat_val = m_df["material"].iloc[0]

        mean_row = {"measurement_id": meas_id, "material": mat_val, "metric": "mean"}
        std_row = {"measurement_id": meas_id, "material": mat_val, "metric": "std"}

        for col in at_cols:
            mean_row[col] = round(float(m_df[col].mean()), 4)
            std_row[col] = round(float(m_df[col].std(ddof=1)), 4) if len(m_df) > 1 else 0.0

        summary_records.append(mean_row)
        summary_records.append(std_row)

    df_summary = pd.DataFrame(summary_records)

    return df_per_die, df_summary
