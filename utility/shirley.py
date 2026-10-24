"""Shirley background subtraction and spectral endpoint determination.

This module provides specialized algorithms for XPS Shirley background
subtraction, including boundary endpoint determination strategies
(minima, edge, direct), mirror reflection padding for baseline averaging,
and iterative Shirley convergence routines.
"""

from typing import Any
import warnings
import numpy as np

from utility.flux_and_intensity import trapezoid_integral

__all__ = [
    "determine_shirley_endpoints",
    "calculate_shirley_background",
]


def _calculate_endpoint_intensity(
    energy: np.ndarray,
    intensity: np.ndarray,
    idx: int,
    average_width_ev: float | None = 1.0,
) -> float:
    """Calculate boundary intensity averaged over a mirror-padded eV window.

    If the averaging window extends outside available spectral data, reflection
    padding without edge repetition (matching CNN ReflectionPad1d / np.pad
    mode='reflect') is used.

    Parameters
    ----------
    energy : np.ndarray
        Binding energy array (ascending order).
    intensity : np.ndarray
        Measured intensity array.
    idx : int
        Index of the endpoint (0 <= idx < len(intensity)).
    average_width_ev : float | None, optional
        Averaging window width in eV. If None or <= 0.0, raw intensity at idx
        is returned. Default is 1.0 eV.

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
    pad_len = min(n_half, len(intensity) - 1)
    if pad_len < 1:
        return float(intensity[idx])

    padded_i = np.pad(intensity, pad_len, mode="reflect")
    e_left = energy[0] - np.arange(pad_len, 0, -1) * de
    e_right = energy[-1] + np.arange(1, pad_len + 1) * de
    padded_e = np.concatenate([e_left, energy, e_right])

    target_e = float(energy[idx])
    mask = (padded_e >= target_e - half_w - 1e-9) & (
        padded_e <= target_e + half_w + 1e-9
    )
    if not np.any(mask):
        return float(intensity[idx])
    return float(np.mean(padded_i[mask]))


def _find_minima_endpoints(
    energy: np.ndarray,
    intensity: np.ndarray,
    smooth_search: bool = False,
    region: str = "",
) -> tuple[int, int]:
    """Determine low and high BE endpoints via directional minima search.

    Procedure:
    1. Locate maximum intensity index in the regional spectrum.
    2. Search in the high BE direction for the lowest intensity -> E2 index.
    3. Search in the low BE direction for the lowest intensity -> E1 index.
    If E1 >= E2:
        Fallback to full spectrum bounds [0, n_pts - 1] with a warning.

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
    del energy
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
            f"Degenerate minima{region_label} endpoints detected "
            f"(idx_1={idx_1} >= idx_2={idx_2}). "
            f"Falling back to full spectrum bounds [0, {n_pts - 1}]."
        )
        return 0, n_pts - 1

    return idx_1, idx_2


def _resolve_direct_endpoints(
    energy: np.ndarray,
    cfg: dict[str, Any],
    region_cfg: dict[str, Any],
    strategy: str,
) -> tuple[int, int]:
    """Resolve and validate direct energy endpoints to index bounds."""
    region = str(cfg.get("region", ""))
    direct_pts = region_cfg.get(
        "energy_endpoints", cfg.get("energy_endpoints")
    )
    if isinstance(direct_pts, dict):
        direct_pts = direct_pts.get(region)

    if direct_pts is None:
        raise ValueError(
            f"Strategy '{strategy}' specified for region '{region}', but no "
            "endpoint coordinates ('energy_endpoints': (e_start, e_end)) "
            "were provided in config."
        )
    if not isinstance(direct_pts, (tuple, list)) or len(direct_pts) != 2:
        raise ValueError(
            f"Invalid energy_endpoints format for region '{region}': expected "
            f"tuple or list of 2 numbers (e_start, e_end), got {direct_pts}"
        )

    p1, p2 = float(direct_pts[0]), float(direct_pts[1])
    if not (np.isfinite(p1) and np.isfinite(p2)):
        raise ValueError(
            f"Invalid non-finite energy_endpoints coordinates for region "
            f"'{region}': got ({p1}, {p2})"
        )

    e_min = float(min(energy[0], energy[-1]))
    e_max = float(max(energy[0], energy[-1]))
    if max(p1, p2) <= e_min or min(p1, p2) >= e_max:
        raise ValueError(
            f"Direct energy_endpoints ({p1}, {p2}) lie entirely outside "
            f"spectral energy range [{e_min:.2f}, {e_max:.2f}] for region "
            f"'{region}'."
        )

    idx_a = int(np.argmin(np.abs(energy - p1)))
    idx_b = int(np.argmin(np.abs(energy - p2)))
    idx_1, idx_2 = min(idx_a, idx_b), max(idx_a, idx_b)
    if idx_1 >= idx_2:
        raise ValueError(
            f"Direct energy_endpoints ({p1}, {p2}) resolve to degenerate "
            f"index range [idx_1={idx_1}, idx_2={idx_2}] on energy span "
            f"[{energy[0]:.2f}, {energy[-1]:.2f}] for region '{region}'. "
            "Endpoints must map to distinct spectral indices."
        )
    return idx_1, idx_2


def determine_shirley_endpoints(
    energy: np.ndarray,
    intensity: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[int, int, float, float]:
    """Determine endpoint indices and baseline intensities for Shirley baseline.

    Supports three strategies:
    1. 'minima': Directional minima search on both sides of peak maximum.
    2. 'edge': Boundary points of the spectral region [0, N-1].
    3. 'direct': Directly user-specified start and end points in eV.

    All strategies support optional boundary intensity averaging over an energy
    window centered at each endpoint with mirror reflection padding.

    Parameters
    ----------
    energy : np.ndarray
        Binding energy array (ascending order).
    intensity : np.ndarray
        Measured intensity array.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'region' (str): Name of spectral region (e.g. 'Ti2p', 'Al2p').
        - 'strategy' (str): Strategy ('minima', 'edge', or 'direct').
        - 'energy_endpoints' (tuple[float, float] | None): Energy
          coordinates (e_start, e_end) for the 'direct' strategy.
        - 'average_width_ev' (float | None): Window width in eV
          for intensity averaging (default 1.0 eV).
        - 'smooth_search' (bool): Smooth curve before minima search.
        - 'region_endpoint_config' (dict[str, dict[str, Any]] | None):
          Per-region overrides.

    Returns
    -------
    tuple[int, int, float, float]
        (idx_1, idx_2, i_1, i_2) where:
        - idx_1, idx_2: Start and end indices (idx_1 < idx_2).
        - i_1, i_2: Baseline intensities at idx_1 and idx_2 after averaging.
    """
    cfg = dict(config or {})
    region = str(cfg.get("region", ""))
    n_pts = len(intensity)
    if n_pts < 2:
        val = float(intensity[0]) if n_pts == 1 else 0.0
        return 0, max(0, n_pts - 1), val, val

    per_region_map = cfg.get("region_endpoint_config") or {}
    region_cfg = (
        per_region_map.get(region, {})
        if isinstance(per_region_map, dict)
        else {}
    )

    strategy = region_cfg.get(
        "strategy",
        cfg.get("strategy", cfg.get("default_endpoint_strategy", "minima")),
    )

    if "average_width_ev" in region_cfg:
        avg_ev = region_cfg["average_width_ev"]
    elif "average_width_ev" in cfg:
        avg_ev = cfg["average_width_ev"]
    else:
        avg_ev = cfg.get("default_endpoint_average_ev", 1.0)

    smooth_search = bool(
        region_cfg.get("smooth_search", cfg.get("smooth_search", False))
    )

    norm_strategy = str(strategy).lower().strip()
    if norm_strategy == "minima":
        idx_1, idx_2 = _find_minima_endpoints(
            energy, intensity, smooth_search=smooth_search, region=region
        )
    elif norm_strategy == "edge":
        idx_1, idx_2 = 0, n_pts - 1
    elif norm_strategy == "direct":
        idx_1, idx_2 = _resolve_direct_endpoints(
            energy, cfg, region_cfg, str(strategy)
        )
    else:
        raise ValueError(
            f"Unknown endpoint strategy '{strategy}' for region '{region}'. "
            "Supported strategies are: 'minima', 'edge', 'direct'."
        )

    i_1 = _calculate_endpoint_intensity(
        energy, intensity, idx_1, average_width_ev=avg_ev
    )
    i_2 = _calculate_endpoint_intensity(
        energy, intensity, idx_2, average_width_ev=avg_ev
    )
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
        - 'strategy' (str): 'minima', 'edge', or 'direct' (default 'minima').
        - 'energy_endpoints' (tuple[float, float] | None): Energy endpoints
          for 'direct'.
        - 'average_width_ev' (float | None): Averaging width in eV (def 1.0).
        - 'region_endpoint_config' (dict | None): Per-region overrides.
        - 'max_iter' (int): Maximum Shirley iterations (default 50).
        - 'tol' (float): Convergence threshold (default 1e-5).

    Returns
    -------
    tuple[np.ndarray, float]
        - background: 1D array of the same shape as intensity. Outside active
          [E1, E2], background equals intensity B(E) = I(E).
        - net_area: Integrated net peak area integral(I(E) - B(E)) dE.

    Raises
    ------
    ValueError
        If energy and intensity arrays have mismatched lengths.
    """
    if len(energy) != len(intensity):
        raise ValueError(
            f"energy and intensity arrays must have matching lengths, "
            f"got {len(energy)} vs {len(intensity)}"
        )

    cfg = config or {}
    max_iter = int(cfg.get("max_iter", 50))
    tol = float(cfg.get("tol", 1e-5))

    if len(intensity) < 2:
        return intensity.copy(), 0.0

    # Ensure internal calculations operate on ascending energy order
    is_descending = energy[0] > energy[-1]
    e_work = energy[::-1] if is_descending else energy
    i_work = intensity[::-1] if is_descending else intensity

    idx_1, idx_2, i_1, i_2 = determine_shirley_endpoints(
        e_work, i_work, config=cfg
    )
    e_sub = e_work[idx_1 : idx_2 + 1]
    i_sub = i_work[idx_1 : idx_2 + 1]
    e_span = float(e_sub[-1] - e_sub[0])

    if e_span == 0.0 or len(e_sub) < 2:
        b_sub = i_sub.copy()
        net_area = 0.0
    else:
        # Linear initialization using effective boundary intensities
        b_sub = i_1 + (i_2 - i_1) * (e_sub - e_sub[0]) / e_span
        dx = np.diff(e_sub)

        for _ in range(max_iter):
            peak = np.maximum(0.0, i_sub - b_sub)
            total_peak_area = float(trapezoid_integral(peak, e_sub))
            if total_peak_area <= 0.0:
                break

            trap_steps = 0.5 * (peak[:-1] + peak[1:]) * dx
            cum_area = np.concatenate(([0.0], np.cumsum(trap_steps)))
            b_new = i_1 + (i_2 - i_1) * (cum_area / total_peak_area)
            max_change = float(np.max(np.abs(b_new - b_sub)))
            b_sub = b_new
            if max_change < tol:
                break

        # Physical safeguard: background cannot exceed measured intensity
        b_sub = np.minimum(b_sub, i_sub)
        net_area = float(trapezoid_integral(np.maximum(0.0, i_sub - b_sub), e_sub))

    # Construct full background array: B(E) = I(E) outside [E1, E2]
    b_work = i_work.copy()
    b_work[idx_1 : idx_2 + 1] = b_sub
    background = b_work[::-1] if is_descending else b_work

    return background, max(0.0, net_area)
