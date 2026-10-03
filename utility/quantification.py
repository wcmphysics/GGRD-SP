"""Quantification and Shirley background subtraction module for GGRD-SP.

Provides Shirley background subtraction, peak area integration, and atomic
percentage calculation fulfilling Part 5 of GEMINI.md.
"""

from __future__ import annotations

from typing import Any
import warnings

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
        - 'default_endpoint_strategy': 'minima' ('minima', 'edge', or 'direct')
        - 'default_endpoint_average_ev': 1.0 (average intensity over 1.0 eV window with mirror padding)
        - 'region_endpoint_config': None (optional dict mapping region to strategy/average/points overrides)
        - 'smooth_endpoints_search': False (optional smoothing for minima search)
        - 'max_iter': 50
        - 'tol': 1e-5
    """
    return {
        "material": "NMG",
        "measurement_id": None,
        "rsf_dict": dict(DEFAULT_SCOFIELD_RSF),
        "default_endpoint_strategy": "minima",
        "default_endpoint_average_ev": 1.0,
        "region_endpoint_config": None,
        "smooth_endpoints_search": False,
        "max_iter": 50,
        "tol": 1e-5,
    }


def _calculate_endpoint_intensity(
    energy: np.ndarray,
    intensity: np.ndarray,
    idx: int,
    average_width_ev: float | None = 1.0,
) -> float:
    """Calculate boundary intensity averaged over an energy window centered at idx with mirror padding.

    If the averaging window extends outside available spectral data, reflection padding
    without edge repetition (matching CNN ReflectionPad1d / np.pad mode='reflect') is used.

    Parameters
    ----------
    energy : np.ndarray
        Binding energy array (ascending order).
    intensity : np.ndarray
        Measured intensity array.
    idx : int
        Index of the endpoint (0 <= idx < len(intensity)).
    average_width_ev : float | None, optional
        Averaging window width in eV. If None or <= 0.0, raw intensity at idx is returned.
        Default is 1.0 eV.

    Returns
    -------
    float
        Averaged intensity value.
    """
    if average_width_ev is None or average_width_ev <= 0.0 or len(intensity) < 2:
        return float(intensity[idx])

    half_w = float(average_width_ev) / 2.0
    de = float(np.mean(np.abs(np.diff(energy))))
    if de <= 0.0:
        return float(intensity[idx])

    n_half = max(1, int(round(half_w / de)))
    # Mirror reflection padding without edge repetition (CNN ReflectionPad1d)
    pad_len = min(n_half, len(intensity) - 1)
    if pad_len < 1:
        return float(intensity[idx])

    padded_i = np.pad(intensity, pad_len, mode="reflect")
    e_left = energy[0] - np.arange(pad_len, 0, -1) * de
    e_right = energy[-1] + np.arange(1, pad_len + 1) * de
    padded_e = np.concatenate([e_left, energy, e_right])

    target_e = float(energy[idx])
    mask = (padded_e >= target_e - half_w - 1e-9) & (padded_e <= target_e + half_w + 1e-9)
    if not np.any(mask):
        return float(intensity[idx])
    return float(np.mean(padded_i[mask]))


def _find_minima_endpoints(
    energy: np.ndarray,
    intensity: np.ndarray,
    smooth_search: bool = False,
    region: str = "",
) -> tuple[int, int]:
    """Determine low and high BE endpoints via directional minima search from peak maximum.

    Procedure:
    1. Locate maximum intensity index in the regional spectrum.
    2. Search in the high BE direction for the lowest intensity -> E2 index.
    3. Search in the low BE direction for the lowest intensity -> E1 index.
    If E1 >= E2:
        Fallback to full spectrum bounds [0, n_pts - 1] with an informative warning.

    Parameters
    ----------
    energy : np.ndarray
        Binding energy array (ascending order).
    intensity : np.ndarray
        Intensity array.
    smooth_search : bool, optional
        Whether to apply a 3-point moving average filter before minima search.
    region : str, optional
        Region identifier for warning messages.

    Returns
    -------
    tuple[int, int]
        (idx_1, idx_2) where idx_1 < idx_2.
    """
    n_pts = len(intensity)
    if n_pts < 2:
        return 0, max(0, n_pts - 1)

    if smooth_search and n_pts >= 3:
        kernel = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        padded = np.pad(intensity, 1, mode="edge")
        i_search = np.convolve(padded, kernel, mode="valid")
    else:
        i_search = intensity

    # Flat spectrum edge case
    if np.max(i_search) == np.min(i_search):
        return 0, n_pts - 1

    idx_max = int(np.argmax(i_search))
    idx_1 = int(np.argmin(i_search[: idx_max + 1]))
    idx_2 = idx_max + int(np.argmin(i_search[idx_max:]))

    if idx_1 >= idx_2:
        region_label = f" for '{region}'" if region else ""
        warnings.warn(
            f"Degenerate minima{region_label} endpoints detected (idx_1={idx_1} >= idx_2={idx_2}). "
            f"Falling back to full spectrum bounds [0, {n_pts - 1}]."
        )
        return 0, n_pts - 1

    return idx_1, idx_2


def determine_shirley_endpoints(
    energy: np.ndarray,
    intensity: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[int, int, float, float]:
    """Determine start and end index and boundary baseline intensities for Shirley background.

    Supports three strategies:
    1. 'minima' (or 'special'): Directional minima search on both sides of peak maximum (Default).
    2. 'edge' (or 'bounds'): Boundary points of the spectral region [0, N-1].
    3. 'direct' (or 'manual', 'custom', 'points'): Directly user-specified start and end points
       (e_start, e_end) in eV.

    All strategies support optional boundary intensity averaging over an energy window of width W eV
    centered at each endpoint, utilizing CNN mirror reflection padding without edge repetition
    (ReflectionPad1d / np.pad mode='reflect') when the window extends outside available spectral data.

    Parameters
    ----------
    energy : np.ndarray
        Binding energy array (ascending order).
    intensity : np.ndarray
        Measured intensity array.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'region' (str): Name of spectral region (e.g. 'Ti2p', 'Al2p').
        - 'strategy' | 'endpoint_strategy' (str): Strategy ('minima', 'edge', or 'direct').
          Default is 'minima'.
        - 'points' | 'direct_points' (tuple[float, float] | None): Energy coordinates (e_start, e_end)
          for the 'direct' strategy.
        - 'average_width_ev' | 'average_ev' (float | None): Window width in eV for intensity averaging
          (default 1.0 eV, 0.0 or None to disable).
        - 'smooth_search' (bool): Smooth search curve before directional minima search (default False).
        - 'region_endpoint_config' (dict[str, dict[str, Any]] | None): Per-region overrides, e.g.:
          {'Al2p': {'strategy': 'edge', 'average_ev': 0.5},
           'Ti2p': {'strategy': 'direct', 'points': (453.0, 467.0), 'average_ev': 1.0}}.

    Returns
    -------
    tuple[int, int, float, float]
        (idx_1, idx_2, i_1, i_2) where:
        - idx_1, idx_2: Start and end indices (idx_1 < idx_2).
        - i_1, i_2: Baseline intensities at idx_1 and idx_2 after optional averaging.
    """
    cfg = dict(config or {})
    region = str(cfg.get("region", ""))
    n_pts = len(intensity)
    if n_pts < 2:
        val = float(intensity[0]) if n_pts == 1 else 0.0
        return 0, max(0, n_pts - 1), val, val

    # Per-region configuration override
    per_region_map = cfg.get("region_endpoint_config") or {}
    region_cfg = per_region_map.get(region, {}) if isinstance(per_region_map, dict) else {}

    # 1. Strategy resolution: Per-region > Specific user strategy > Top-level mappings > Default
    strategy = None
    if "strategy" in region_cfg:
        strategy = region_cfg["strategy"]
    elif "endpoint_strategy" in region_cfg:
        strategy = region_cfg["endpoint_strategy"]

    if strategy is None:
        strat_dict = cfg.get("endpoint_strategies") or cfg.get("strategies")
        if isinstance(strat_dict, dict) and region in strat_dict:
            strategy = strat_dict[region]

    if strategy is None:
        strategy = cfg.get("strategy", cfg.get("endpoint_strategy"))

    if strategy is None:
        strategy = cfg.get("default_endpoint_strategy", "minima")

    # 2. Averaging width resolution: Per-region (including None/0.0) > Top-level mappings > Defaults
    if "average_width_ev" in region_cfg:
        avg_ev = region_cfg["average_width_ev"]
    elif "average_ev" in region_cfg:
        avg_ev = region_cfg["average_ev"]
    else:
        avg_dict = cfg.get("endpoint_averages") or cfg.get("average_evs")
        if isinstance(avg_dict, dict) and region in avg_dict:
            avg_ev = avg_dict[region]
        else:
            avg_ev = cfg.get(
                "average_width_ev",
                cfg.get("average_ev", cfg.get("endpoint_average_ev", cfg.get("default_endpoint_average_ev", 1.0))),
            )

    # 3. Smoothing flag resolution: Per-region > General smoothing
    if "smooth_search" in region_cfg:
        smooth_search = bool(region_cfg["smooth_search"])
    elif "smooth_search" in cfg:
        smooth_search = bool(cfg["smooth_search"])
    elif "smooth_endpoints_search" in cfg:
        smooth_search = bool(cfg["smooth_endpoints_search"])
    else:
        smooth_search = False

    norm_strategy = str(strategy).lower().strip()

    # 4. Strategy execution
    if norm_strategy in ("minima", "directional_minima", "special"):
        idx_1, idx_2 = _find_minima_endpoints(energy, intensity, smooth_search=smooth_search, region=region)
    elif norm_strategy in ("edge", "bounds", "edge_points", "full"):
        idx_1, idx_2 = 0, n_pts - 1
    elif norm_strategy in ("direct", "manual", "custom", "points"):
        # Resolve direct points from per-region config or top-level config
        direct_pts = (
            region_cfg.get("points")
            or region_cfg.get("direct_points")
            or region_cfg.get("energy_limits")
            or region_cfg.get("limits")
        )
        if direct_pts is None:
            pts_dict = cfg.get("points") or cfg.get("direct_points") or cfg.get("energy_limits")
            if isinstance(pts_dict, dict):
                direct_pts = pts_dict.get(region)
            elif isinstance(pts_dict, (tuple, list)):
                direct_pts = pts_dict

        if direct_pts is None:
            raise ValueError(
                f"Strategy '{strategy}' specified for region '{region}', but no endpoint coordinates "
                f"('points': (e_start, e_end)) were provided in config."
            )
        if not isinstance(direct_pts, (tuple, list)) or len(direct_pts) != 2:
            raise ValueError(
                f"Invalid points format for region '{region}': expected tuple or list of 2 numbers (e_start, e_end), "
                f"got {direct_pts}"
            )

        p1, p2 = float(direct_pts[0]), float(direct_pts[1])
        if not (np.isfinite(p1) and np.isfinite(p2)):
            raise ValueError(
                f"Invalid non-finite points coordinates for region '{region}': got ({p1}, {p2})"
            )

        e_min, e_max = float(min(energy[0], energy[-1])), float(max(energy[0], energy[-1]))
        p_low, p_high = min(p1, p2), max(p1, p2)
        if p_high <= e_min or p_low >= e_max:
            raise ValueError(
                f"Direct points ({p1}, {p2}) lie entirely outside spectral energy range "
                f"[{e_min:.2f}, {e_max:.2f}] for region '{region}'."
            )

        idx_a = int(np.argmin(np.abs(energy - p1)))
        idx_b = int(np.argmin(np.abs(energy - p2)))
        idx_1, idx_2 = min(idx_a, idx_b), max(idx_a, idx_b)
        if idx_1 >= idx_2:
            raise ValueError(
                f"Direct points ({p1}, {p2}) resolve to degenerate index range [idx_1={idx_1}, idx_2={idx_2}] "
                f"on energy span [{energy[0]:.2f}, {energy[-1]:.2f}] for region '{region}'. "
                f"Endpoints must map to distinct spectral indices."
            )
    else:
        raise ValueError(
            f"Unknown endpoint strategy '{strategy}' for region '{region}'. "
            f"Supported strategies are: 'minima', 'edge', 'direct'."
        )

    i_1 = _calculate_endpoint_intensity(energy, intensity, idx_1, average_width_ev=avg_ev)
    i_2 = _calculate_endpoint_intensity(energy, intensity, idx_2, average_width_ev=avg_ev)

    return idx_1, idx_2, i_1, i_2


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
        - 'region' (str): Name of the spectral region (e.g. 'Ti2p', 'Al2p').
        - 'strategy' | 'endpoint_strategy' (str): 'minima', 'edge', or 'direct' (default 'minima').
        - 'points' (tuple[float, float] | None): Energy endpoints (e_start, e_end) for 'direct' strategy.
        - 'average_width_ev' (float | None): Averaging width in eV (default 1.0).
        - 'region_endpoint_config' (dict | None): Per-region overrides.
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

    # Determine endpoints and boundary intensities via unified strategy resolver
    idx_1, idx_2, i_1, i_2 = determine_shirley_endpoints(e_work, i_work, config=cfg)

    # Extract sub-interval
    e_sub = e_work[idx_1 : idx_2 + 1]
    i_sub = i_work[idx_1 : idx_2 + 1]

    # Initial linear background connecting endpoints
    e_span = float(e_sub[-1] - e_sub[0])

    if e_span == 0.0 or len(e_sub) < 2:
        b_sub = i_sub.copy()
        net_area = 0.0
    else:
        # Linear initialization using effective boundary intensities
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

        # Physical safeguard: background cannot exceed measured spectrum intensity
        b_sub = np.minimum(b_sub, i_sub)
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
    - Supports systematic Shirley endpoint determination ('minima', 'edge', or 'direct') across all regions.
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
        - 'default_endpoint_strategy' (str): 'minima', 'edge', or 'direct' (default 'minima').
        - 'default_endpoint_average_ev' (float | None): Default averaging width in eV (default 1.0).
        - 'region_endpoint_config' (dict | None): Per-region overrides.
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

                shirley_cfg = dict(cfg)
                shirley_cfg["region"] = region
                _, net_area = calculate_shirley_background(e_arr, i_arr, shirley_cfg)

                if region not in rsf_dict:
                    warnings.warn(
                        f"Region '{region}' not found in rsf_dict. Defaulting RSF to 1.0. "
                        "Atomic percentages may be uncalibrated without specific sensitivity factors."
                    )
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
