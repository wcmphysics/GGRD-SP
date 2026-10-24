"""Spectral flux and intensity calculation module for GGRD-SP.

Provides unified numerical integration across energy grids and calculates
hierarchical flux and mean intensity metrics across regions, dies, and
measurement sessions.
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


def calculate_flux_and_intensity(
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute hierarchical spectral fluxes and mean intensities.

    Computes integrated flux and mean intensity metrics on raw spectra and
    appends nine columns to df_meta:
    1. 'mean_intensity_single_region': arithmetic mean intensity of that region.
    2. 'mean_intensity_single_die': mean intensity across regions for that die.
    3. 'relative_intensity_single_region': ratio of regional to
       die mean intensity.
    4. 'total_flux_single_region': flux of a region of a die from a measurement.
    5. 'mean_flux_across_regions_single_die': average regional flux
       among regions.
    6. 'total_flux_single_die': total sum of regional flux across that die.
    7. 'mean_flux_across_dies_single_measurement': average die flux in session.
    8. 'total_flux_single_measurement': total sum of die flux in that session.
    9. 'total_flux_single_die_standardized': z-score standardized die flux.

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
          and std of total_flux_single_die for z-score standardization.
          Default is None.

    Returns
    -------
    pd.DataFrame
        New copy of df_meta with the 9 appended flux and intensity columns.

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
    appended_cols = [
        "mean_intensity_single_region",
        "mean_intensity_single_die",
        "relative_intensity_single_region",
        "total_flux_single_region",
        "mean_flux_across_regions_single_die",
        "total_flux_single_die",
        "mean_flux_across_dies_single_measurement",
        "total_flux_single_measurement",
        "total_flux_single_die_standardized",
    ]
    if df_res.empty:
        for col in appended_cols:
            df_res[col] = pd.Series(dtype=np.float32)
        return df_res

    indices = df_res["spectrum_index"].astype(int).to_numpy()
    n_total = len(ary_intensity)
    if indices.min() < 0 or indices.max() >= n_total:
        raise ValueError(
            f"spectrum_index in df_meta contains out-of-bounds values: "
            f"[{indices.min()}, {indices.max()}] for array length {n_total}."
        )

    sub_intensity = ary_intensity[indices]
    sub_energy = ary_energy[indices]

    # 1. Mean intensity calculations
    df_res["mean_intensity_single_region"] = np.mean(
        sub_intensity, axis=-1
    ).astype(np.float32)

    die_group_int = df_res.groupby(["measurement_id", "die"])[
        "mean_intensity_single_region"
    ]
    df_res["mean_intensity_single_die"] = die_group_int.transform(
        "mean"
    ).astype(np.float32)

    # Relative regional intensity: regional mean / die mean
    denom_intensity = np.maximum(
        df_res["mean_intensity_single_die"].to_numpy(dtype=np.float32), 1e-12
    )
    df_res["relative_intensity_single_region"] = (
        df_res["mean_intensity_single_region"] / denom_intensity
    ).astype(np.float32)

    # 2. Flux calculations
    df_res["total_flux_single_region"] = integrate_spectral_flux(
        sub_intensity, sub_energy, axis=-1
    ).astype(np.float32)

    die_group_flux = df_res.groupby(["measurement_id", "die"])[
        "total_flux_single_region"
    ]
    df_res["mean_flux_across_regions_single_die"] = die_group_flux.transform(
        "mean"
    ).astype(np.float32)
    df_res["total_flux_single_die"] = die_group_flux.transform("sum").astype(
        np.float32
    )

    die_summary = df_res.drop_duplicates(subset=["measurement_id", "die"])
    meas_mean_die = die_summary.groupby("measurement_id")[
        "total_flux_single_die"
    ].mean()
    meas_total_die = die_summary.groupby("measurement_id")[
        "total_flux_single_die"
    ].sum()

    df_res["mean_flux_across_dies_single_measurement"] = (
        df_res["measurement_id"].map(meas_mean_die).astype(np.float32)
    )
    df_res["total_flux_single_measurement"] = (
        df_res["measurement_id"].map(meas_total_die).astype(np.float32)
    )

    cfg = config or {}
    if not bool(cfg.get("standardize_die_flux", True)):
        return df_res

    fit_split = cfg.get("fit_split")
    if fit_split is not None:
        if "split" not in df_res.columns:
            raise KeyError(
                "df_meta must contain 'split' column to compute "
                "'total_flux_single_die_standardized'. Please run "
                "split_measurement_sessions before standardizing die flux."
            )
        if not (df_res["split"] == fit_split).any():
            raise ValueError(
                f"Specified fit_split '{fit_split}' was not found in df_meta['split']. "
                f"Available splits: {list(df_res['split'].dropna().unique())}."
            )
        _standardize_die_flux_in_place(df_res, str(fit_split))
    elif "split" in df_res.columns and (df_res["split"] == "train").any():
        _standardize_die_flux_in_place(df_res, "train")
    else:
        df_res["total_flux_single_die_standardized"] = pd.Series(
            np.nan, index=df_res.index, dtype=np.float32
        )

    return df_res


def _standardize_die_flux_in_place(
    df_res: pd.DataFrame,
    fit_split: str,
) -> None:
    """Standardize total_flux_single_die in-place using fit_split dies."""
    die_summary = df_res.drop_duplicates(subset=["measurement_id", "die"])
    die_fit = die_summary[die_summary["split"] == fit_split]
    die_flux_vals = die_fit["total_flux_single_die"].to_numpy(dtype=np.float64)
    flux_mean = float(np.mean(die_flux_vals))
    flux_std = float(np.std(die_flux_vals))
    if flux_std == 0.0:
        warnings.warn(
            f"Standard deviation of total_flux_single_die across '{fit_split}' "
            "dies is 0.0; setting total_flux_single_die_standardized to 0.0."
        )
        df_res["total_flux_single_die_standardized"] = np.float32(0.0)
    else:
        df_res["total_flux_single_die_standardized"] = (
            (df_res["total_flux_single_die"] - flux_mean) / flux_std
        ).astype(np.float32)


def calculate_standardized_time(
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Standardize measurement time across unique measurement sessions.

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
        New copy of df_meta with appended 'time_standardized' column.

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
            f"df_meta missing required columns for time standardization: {missing_cols}"
        )

    df_res = df_meta.copy()
    if df_res.empty:
        df_res["time_standardized"] = pd.Series(dtype=np.float32)
        return df_res

    if "split" not in df_res.columns:
        raise KeyError(
            "df_meta must contain 'split' column to compute 'time_standardized'. "
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
            "null values when computing time_standardized."
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
            "sessions is 0.0; setting time_standardized to 0.0."
        )
        df_res["time_standardized"] = np.float32(0.0)
    else:
        df_res["time_standardized"] = (all_diff_sec / std_sec).astype(np.float32)

    return df_res


def standardize_metadata_features(
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Standardize metadata features ('time_standardized' and
    'total_flux_single_die_standardized') on train split without data leakage.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata DataFrame containing 'measurement_id', 'die', and optionally
        'time', 'total_flux_single_die', and 'split'.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'fit_split' (str): Split name (e.g. 'train') to compute mean
          and std on. Default is 'train'.

    Returns
    -------
    pd.DataFrame
        New copy of df_meta with standardized 'time_standardized' and
        'total_flux_single_die_standardized' columns.

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
        df_res = calculate_standardized_time(
            df_res, config={"fit_split": fit_split}
        )

    if {"total_flux_single_die", "die", "measurement_id"}.issubset(
        df_res.columns
    ):
        _standardize_die_flux_in_place(df_res, str(fit_split))

    return df_res


def compute_die_total_flux(
    df_meta: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray | None = None,
) -> dict[tuple[str, int], float]:
    """Calculate total integrated spectral flux for (measurement_id, die).

    If 'total_flux_single_die' is already present in df_meta, it is
    retrieved directly. Otherwise, fluxes are computed by numerical
    integration across regional spectra.

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
    if (
        "total_flux_single_die" in df_meta.columns
        and not df_meta["total_flux_single_die"].isna().all()
    ):
        sub = df_meta.drop_duplicates(subset=["measurement_id", "die"])
        invalid_rows = [
            (str(r.measurement_id), int(r.die))
            for r in sub.itertuples()
            if pd.isna(r.total_flux_single_die)
            or not np.isfinite(float(r.total_flux_single_die))
        ]
        if invalid_rows:
            raise ValueError(
                "df_meta contains invalid or NaN 'total_flux_single_die' "
                f"values for {len(invalid_rows)} (measurement_id, die) "
                f"pairs: {invalid_rows[:5]}. All rows must have valid "
                "numerical total_flux_single_die if the column is present."
            )
        return {
            (str(r.measurement_id), int(r.die)): max(
                float(r.total_flux_single_die), 1e-4
            )
            for r in sub.itertuples()
        }

    valid_mask = (df_meta["spectrum_index"] >= 0) & (
        df_meta["spectrum_index"] < len(ary_intensity)
    )
    sub_df = df_meta[valid_mask]
    if sub_df.empty:
        return {
            (str(m), int(d)): 1e-4
            for m, d in df_meta[["measurement_id", "die"]]
            .drop_duplicates()
            .itertuples(index=False)
        }

    idx = sub_df["spectrum_index"].to_numpy(dtype=int)
    if ary_energy is not None and len(ary_energy) > int(idx.max()):
        areas = np.asarray(
            integrate_spectral_flux(ary_intensity[idx], ary_energy[idx], axis=-1),
            dtype=np.float64,
        )
    else:
        areas = np.sum(ary_intensity[idx], axis=1, dtype=np.float64)

    grouped = (
        sub_df.assign(_area=areas)
        .groupby(["measurement_id", "die"], sort=False)["_area"]
        .sum()
    )
    return {
        (str(m), int(d)): max(float(val), 1e-4)
        for (m, d), val in grouped.items()
    }
