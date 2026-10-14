"""Spectral flux calculation module for GGRD-SP.

Provides unified numerical integration across energy grids and calculates
hierarchical flux metrics across regions, dies, and measurement sessions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def trapezoid_integral(
    y: np.ndarray,
    x: np.ndarray,
    axis: int = -1,
) -> float | np.ndarray:
    """Compute trapezoidal numerical integration robustly across NumPy versions.

    Parameters
    ----------
    y : np.ndarray
        Array of values to integrate (e.g. intensity).
    x : np.ndarray
        Array of sample points corresponding to the y values (e.g. energy).
    axis : int, optional
        Axis along which to integrate (default -1).

    Returns
    -------
    float | np.ndarray
        The definite integral computed via the trapezoidal rule.
    """
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return trapz_fn(y, x, axis=axis)


def integrate_spectral_flux(
    intensity: np.ndarray,
    energy: np.ndarray,
    axis: int = -1,
) -> float | np.ndarray:
    """Calculate absolute integrated spectral flux |∫ I(E) dE| over energy.

    Taking the absolute value guarantees positive flux regardless of whether
    binding energy values are stored in ascending or descending order.

    Parameters
    ----------
    intensity : np.ndarray
        1D or batched array of spectral intensities.
    energy : np.ndarray
        1D or batched array of binding energies.
    axis : int, optional
        Axis along which to integrate (default -1).

    Returns
    -------
    float | np.ndarray
        Absolute integrated spectral flux (scalar float if 1D input).
    """
    val = np.abs(trapezoid_integral(intensity, energy, axis=axis))
    return float(val) if np.ndim(val) == 0 else val


def calculate_flux(
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute hierarchical spectral fluxes for regions, dies, and sessions.

    Computes integrated flux on raw spectra and appends five hierarchical
    columns:
    1. 'regional_flux': flux of a region of a die from a measurement.
    2. 'mean_regional_flux': average 'regional_flux' among regions of that die.
    3. 'die_flux': total sum of 'regional_flux' across all regions of that die.
    4. 'mean_die_flux': average 'die_flux' among unique dies in measurement.
    5. 'measurement_flux': total sum of 'die_flux' in that measurement.

    Parameters
    ----------
    ary_intensity : np.ndarray
        2D array of measured spectral intensities of shape (N_total, N_points).
    ary_energy : np.ndarray
        2D array of binding energies of shape (N_total, N_points).
    df_meta : pd.DataFrame
        Metadata DataFrame containing 'spectrum_index', 'measurement_id',
        'die', and 'region'.
    config : dict[str, Any] | None, optional
        Optional configuration dictionary (reserved for future extensions).

    Returns
    -------
    pd.DataFrame
        New copy of df_meta with the 5 appended flux columns.

    Raises
    ------
    ValueError
        If inputs have mismatched lengths, out-of-bounds indices, or missing
        required columns.
    """
    del config  # Reserved for future options
    if len(ary_intensity) != len(ary_energy):
        raise ValueError(
            f"ary_intensity and ary_energy must have same row count, "
            f"got {len(ary_intensity)} vs {len(ary_energy)}."
        )

    required_cols = ["spectrum_index", "measurement_id", "die", "region"]
    missing_cols = [c for c in required_cols if c not in df_meta.columns]
    if missing_cols:
        raise ValueError(
            f"df_meta missing required columns for flux calculation: {missing_cols}"
        )

    df_res = df_meta.copy()
    if df_res.empty:
        for col in [
            "regional_flux",
            "mean_regional_flux",
            "die_flux",
            "mean_die_flux",
            "measurement_flux",
        ]:
            df_res[col] = pd.Series(dtype=np.float32)
        return df_res

    indices = df_res["spectrum_index"].astype(int).to_numpy()
    n_total = len(ary_intensity)
    if indices.min() < 0 or indices.max() >= n_total:
        raise ValueError(
            f"spectrum_index in df_meta contains out-of-bounds values: "
            f"[{indices.min()}, {indices.max()}] for array length {n_total}."
        )

    # 1. Compute regional_flux for each row via vectorized integration
    sub_intensity = ary_intensity[indices]
    sub_energy = ary_energy[indices]
    df_res["regional_flux"] = integrate_spectral_flux(
        sub_intensity, sub_energy, axis=-1
    ).astype(np.float32)

    # 2 & 3. Group by (measurement_id, die) for regional mean and die total
    die_group = df_res.groupby(["measurement_id", "die"])["regional_flux"]
    df_res["mean_regional_flux"] = die_group.transform("mean").astype(np.float32)
    df_res["die_flux"] = die_group.transform("sum").astype(np.float32)

    # 4 & 5. Group unique dies by measurement_id for mean die and total flux
    die_summary = df_res.drop_duplicates(subset=["measurement_id", "die"])
    meas_mean_die = die_summary.groupby("measurement_id")["die_flux"].mean()
    meas_total_die = die_summary.groupby("measurement_id")["die_flux"].sum()

    df_res["mean_die_flux"] = (
        df_res["measurement_id"].map(meas_mean_die).astype(np.float32)
    )
    df_res["measurement_flux"] = (
        df_res["measurement_id"].map(meas_total_die).astype(np.float32)
    )

    return df_res
