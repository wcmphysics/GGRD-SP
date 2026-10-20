"""Spectral flux calculation module for GGRD-SP.

Provides unified numerical integration across energy grids and calculates
hierarchical flux metrics across regions, dies, and measurement sessions.
"""

from __future__ import annotations

from typing import Any
import warnings

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

    Computes integrated flux on raw spectra and appends six columns:
    1. 'regional_flux': flux of a region of a die from a measurement.
    2. 'mean_regional_flux': average 'regional_flux' among regions of that die.
    3. 'die_flux': total sum of 'regional_flux' across all regions of that die.
    4. 'mean_die_flux': average 'die_flux' among unique dies in measurement.
    5. 'measurement_flux': total sum of 'die_flux' in that measurement.
    6. 'die_flux_normalized': z-score standardized 'die_flux' across unique
       dies.

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
        Optional configuration dictionary:
        - 'fit_split' (str | None): Split name (e.g. 'train') to compute mean
          and std of die_flux for z-score standardization. If specified,
          df_meta must contain 'split'. Default is None.

    Returns
    -------
    pd.DataFrame
        New copy of df_meta with the 6 appended flux columns.

    Raises
    ------
    KeyError
        If fit_split is specified but 'split' column is missing from df_meta.
    ValueError
        If inputs have mismatched lengths, out-of-bounds indices, missing
        required columns, or specified fit_split is not found.
    """
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
            "die_flux_normalized",
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

    # 4, 5 & 6. Group unique dies for measurement metrics and z-score flux
    die_summary = df_res.drop_duplicates(subset=["measurement_id", "die"])
    meas_mean_die = die_summary.groupby("measurement_id")["die_flux"].mean()
    meas_total_die = die_summary.groupby("measurement_id")["die_flux"].sum()

    df_res["mean_die_flux"] = (
        df_res["measurement_id"].map(meas_mean_die).astype(np.float32)
    )
    df_res["measurement_flux"] = (
        df_res["measurement_id"].map(meas_total_die).astype(np.float32)
    )

    cfg = config or {}
    fit_split = cfg.get("fit_split")
    if fit_split is not None:
        if "split" not in df_res.columns:
            raise KeyError(
                "df_meta must contain 'split' column to compute 'die_flux_normalized'. "
                "Please run split_measurement_sessions before standardizing die flux."
            )
        if not (df_res["split"] == fit_split).any():
            raise ValueError(
                f"Specified fit_split '{fit_split}' was not found in df_meta['split']. "
                f"Available splits: {list(df_res['split'].dropna().unique())}."
            )
        die_fit = die_summary[die_summary["split"] == fit_split]
        die_flux_vals = die_fit["die_flux"].to_numpy(dtype=np.float64)
        flux_mean = float(np.mean(die_flux_vals))
        flux_std = float(np.std(die_flux_vals))
        if flux_std == 0.0:
            warnings.warn(
                f"Standard deviation of die_flux across '{fit_split}' dies is 0.0; "
                "setting die_flux_normalized to 0.0."
            )
            df_res["die_flux_normalized"] = np.float32(0.0)
        else:
            df_res["die_flux_normalized"] = (
                (df_res["die_flux"] - flux_mean) / flux_std
            ).astype(np.float32)
    elif "split" in df_res.columns and (df_res["split"] == "train").any():
        die_fit = die_summary[die_summary["split"] == "train"]
        die_flux_vals = die_fit["die_flux"].to_numpy(dtype=np.float64)
        flux_mean = float(np.mean(die_flux_vals))
        flux_std = float(np.std(die_flux_vals))
        if flux_std == 0.0:
            warnings.warn(
                "Standard deviation of die_flux across 'train' dies is 0.0; "
                "setting die_flux_normalized to 0.0."
            )
            df_res["die_flux_normalized"] = np.float32(0.0)
        else:
            df_res["die_flux_normalized"] = (
                (df_res["die_flux"] - flux_mean) / flux_std
            ).astype(np.float32)
    else:
        df_res["die_flux_normalized"] = pd.Series(
            np.nan, index=df_res.index, dtype=np.float32
        )

    return df_res


def calculate_normalized_time(
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Standardize measurement time across unique measurement sessions.

    Computes the time difference between each measurement timestamp and the
    mean timestamp across unique measurements (optionally fitting mean and std
    strictly on the training split defined by config['fit_split']).

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata DataFrame containing 'measurement_id' and 'time'.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'fit_split' (str): Split name (e.g. 'train') to compute mean
          and std on. Default is 'train'.

    Returns
    -------
    pd.DataFrame
        New copy of df_meta with appended 'time_normalized' column.

    Raises
    ------
    KeyError
        If 'split' column is missing from df_meta.
    ValueError
        If required columns are missing, fit_split is not found, or timestamps
        are null/invalid.
    """
    missing_cols = [
        c for c in ("measurement_id", "time") if c not in df_meta.columns
    ]
    if missing_cols:
        raise ValueError(
            f"df_meta missing required columns for time normalization: {missing_cols}"
        )

    df_res = df_meta.copy()
    if df_res.empty:
        df_res["time_normalized"] = pd.Series(dtype=np.float32)
        return df_res

    if "split" not in df_res.columns:
        raise KeyError(
            "df_meta must contain 'split' column to compute 'time_normalized'. "
            "Please run split_measurement_sessions before standardizing measurement time."
        )

    cfg = config or {}
    fit_split = cfg.get("fit_split", "train")
    if not (df_res["split"] == fit_split).any():
        raise ValueError(
            f"Specified fit_split '{fit_split}' was not found in df_meta['split']. "
            f"Available splits: {list(df_res['split'].dropna().unique())}."
        )

    if df_res["measurement_id"].isna().any() or df_res["time"].isna().any():
        raise ValueError(
            "Columns 'measurement_id' and 'time' in df_meta must not contain "
            "null values when computing time_normalized."
        )

    try:
        all_ts = pd.to_datetime(df_res["time"])
    except (ValueError, TypeError) as err:
        raise ValueError(
            f"Failed to parse timestamps in df_meta['time']: {err}"
        ) from err

    meas_summary = df_res.drop_duplicates(subset=["measurement_id"])
    meas_fit = meas_summary[meas_summary["split"] == fit_split]

    meas_ts = pd.to_datetime(meas_fit["time"])
    mean_ts = meas_ts.mean()

    meas_diff_sec = (meas_ts - mean_ts).dt.total_seconds().to_numpy(dtype=np.float64)
    all_diff_sec = (all_ts - mean_ts).dt.total_seconds().to_numpy(dtype=np.float64)
    std_sec = float(np.std(meas_diff_sec))

    if std_sec == 0.0:
        warnings.warn(
            f"Standard deviation of measurement time differences across '{fit_split}' "
            "sessions is 0.0; setting time_normalized to 0.0."
        )
        df_res["time_normalized"] = np.float32(0.0)
    else:
        df_res["time_normalized"] = (all_diff_sec / std_sec).astype(np.float32)

    return df_res


def standardize_metadata_features(
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Standardize metadata features ('time_normalized', 'die_flux_normalized')
    strictly on the training split to prevent data leakage.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata DataFrame containing 'measurement_id', 'die', and optionally
        'time', 'die_flux', and 'split'.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'fit_split' (str): Split name (e.g. 'train') to compute mean
          and std on. Default is 'train'.

    Returns
    -------
    pd.DataFrame
        New copy of df_meta with standardized 'time_normalized' and
        'die_flux_normalized' columns.

    Raises
    ------
    KeyError
        If 'split' column is missing from df_meta.
    ValueError
        If fit_split is not found in df_meta['split'].
    """
    df_res = df_meta.copy()
    if df_res.empty:
        return df_res

    if "split" not in df_res.columns:
        raise KeyError(
            "df_meta must contain 'split' column to standardize metadata features. "
            "Please run split_measurement_sessions before standardizing metadata features."
        )

    cfg = config or {}
    fit_split = cfg.get("fit_split", "train")
    if not (df_res["split"] == fit_split).any():
        raise ValueError(
            f"Specified fit_split '{fit_split}' was not found in df_meta['split']. "
            f"Available splits: {list(df_res['split'].dropna().unique())}."
        )

    if "time" in df_res.columns:
        df_res = calculate_normalized_time(df_res, config={"fit_split": fit_split})

    if (
        "die_flux" in df_res.columns
        and "die" in df_res.columns
        and "measurement_id" in df_res.columns
    ):
        die_summary = df_res.drop_duplicates(subset=["measurement_id", "die"])
        die_fit = die_summary[die_summary["split"] == fit_split]
        die_flux_vals = die_fit["die_flux"].to_numpy(dtype=np.float64)
        flux_mean = float(np.mean(die_flux_vals))
        flux_std = float(np.std(die_flux_vals))
        if flux_std == 0.0:
            warnings.warn(
                f"Standard deviation of die_flux across '{fit_split}' dies is 0.0; "
                "setting die_flux_normalized to 0.0."
            )
            df_res["die_flux_normalized"] = np.float32(0.0)
        else:
            df_res["die_flux_normalized"] = (
                (df_res["die_flux"] - flux_mean) / flux_std
            ).astype(np.float32)

    return df_res


def compute_die_total_flux(
    df_meta: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray | None = None,
) -> dict[tuple[str, int], float]:
    """Calculate total integrated spectral flux for (measurement_id, die).

    If 'die_flux' is already present in df_meta, it is retrieved directly.
    Otherwise, fluxes are computed by numerical integration across regional
    spectra.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata DataFrame containing 'measurement_id', 'die', and
        'spectrum_index'.
    ary_intensity : np.ndarray
        2D intensity array of shape (N_total, N_points).
    ary_energy : np.ndarray | None, optional
        2D binding energy array of shape (N_total, N_points). If provided,
        integrates over eV.

    Returns
    -------
    dict[tuple[str, int], float]
        Dictionary mapping (measurement_id, die) to total integrated flux.
    """
    if "die_flux" in df_meta.columns and not df_meta["die_flux"].isna().all():
        sub = df_meta.drop_duplicates(subset=["measurement_id", "die"])
        invalid_rows = [
            (str(r.measurement_id), int(r.die))
            for r in sub.itertuples()
            if pd.isna(r.die_flux) or not np.isfinite(float(r.die_flux))
        ]
        if invalid_rows:
            sample_invalid = invalid_rows[:5]
            raise ValueError(
                f"df_meta contains invalid or NaN 'die_flux' values for "
                f"{len(invalid_rows)} (measurement_id, die) pairs: "
                f"{sample_invalid}. All rows must have valid numerical "
                "die_flux if the column is present."
            )
        return {
            (str(r.measurement_id), int(r.die)): max(float(r.die_flux), 1e-4)
            for r in sub.itertuples()
        }

    flux_map: dict[tuple[str, int], float] = {}
    for (meas_id, die), group in df_meta.groupby(["measurement_id", "die"]):
        total_flux = 0.0
        for _, row in group.iterrows():
            idx = int(row["spectrum_index"])
            if idx < 0 or idx >= len(ary_intensity):
                continue
            y_arr = ary_intensity[idx]
            if ary_energy is not None and idx < len(ary_energy):
                x_arr = ary_energy[idx]
                area = float(integrate_spectral_flux(y_arr, x_arr))
            else:
                area = float(np.sum(y_arr))
            total_flux += area
        flux_map[(str(meas_id), int(die))] = max(total_flux, 1e-4)

    return flux_map
