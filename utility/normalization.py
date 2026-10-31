"""Spectral normalization utilities for GGRD-SP."""

from __future__ import annotations

from typing import Any
import warnings
import numpy as np
import pandas as pd

from utility.flux_and_intensity import compute_die_total_flux

_VALID_NORMALIZATION_MODES = (
    "by_total_flux_single_die",
    "by_regional_max_intensity",
    "by_regional_max_intensity_of_source",
    "by_mean_intensity_single_die",
    "no_normalization",
)


def _clamp_scales(
    scales: np.ndarray,
    eps: float,
    mode: str,
) -> np.ndarray:
    """Clamp scale vector to eps and emit a warning if any scale <= eps."""
    low_mask = scales <= eps
    if np.any(low_mask):
        warnings.warn(
            f"Normalization scale <= eps ({eps:.4e}) encountered in "
            f"{int(np.sum(low_mask))} spectra for mode '{mode}'. "
            "Clamping scale to eps.",
            UserWarning,
            stacklevel=3,
        )
    return np.maximum(scales, eps).astype(np.float32)[:, None]


def normalize_spectral_data(
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Normalize spectral intensity arrays.

    Parameters
    ----------
    ary_intensity : np.ndarray
        2D intensity array of shape (N_total, N_points).
    ary_energy : np.ndarray
        2D binding energy array of shape (N_total, N_points).
    df_meta : pd.DataFrame
        Metadata DataFrame containing 'measurement_id', 'die', and
        'spectrum_index' (and 'target_spectrum_index' for
        'by_regional_max_intensity_of_source').
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'normalization_mode': 'by_total_flux_single_die' (default),
          'by_regional_max_intensity', 'by_regional_max_intensity_of_source',
          'by_mean_intensity_single_die', or 'no_normalization'.
        - 'eps': minimum positive floor (default 1e-6).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Tuple of (ary_intensity_normalized, ary_energy, df_meta).

    Raises
    ------
    ValueError
        If normalization_mode is unknown, or required metadata columns are
        missing.
    """
    cfg = config or {}
    norm_mode = str(
        cfg.get("normalization_mode", "by_total_flux_single_die")
    ).lower()
    eps = float(cfg.get("eps", 1e-6))

    if norm_mode not in _VALID_NORMALIZATION_MODES:
        raise ValueError(
            f"Unknown normalization_mode '{norm_mode}'. "
            f"Supported options are {list(_VALID_NORMALIZATION_MODES)}."
        )

    df_meta = df_meta.copy()
    ary_norm = ary_intensity.astype(np.float32, copy=True)
    if norm_mode == "no_normalization":
        df_meta["normalization_scale"] = 1.0
        df_meta.attrs["normalization_mode"] = norm_mode
        return ary_norm, ary_energy, df_meta

    n_total = len(ary_norm)
    scales_record = np.ones(n_total, dtype=np.float64)
    valid_df = df_meta[
        (df_meta["spectrum_index"] >= 0) & (df_meta["spectrum_index"] < n_total)
    ]
    if valid_df.empty:
        df_meta["normalization_scale"] = 1.0
        df_meta.attrs["normalization_mode"] = norm_mode
        return ary_norm, ary_energy, df_meta
    idx = valid_df["spectrum_index"].to_numpy(dtype=int)

    if norm_mode == "by_total_flux_single_die":
        flux_map = compute_die_total_flux(df_meta, ary_intensity, ary_energy)
        raw_scales = np.array(
            [
                float(flux_map.get((str(m), int(d)), 0.0))
                for m, d in zip(valid_df["measurement_id"], valid_df["die"])
            ],
            dtype=np.float64,
        )
        scales = _clamp_scales(raw_scales, eps, norm_mode)
        ary_norm[idx] /= scales
        scales_record[idx] = scales.ravel()

    elif norm_mode == "by_mean_intensity_single_die":
        if "mean_intensity_single_die" not in df_meta.columns:
            raise ValueError(
                "normalization_mode='by_mean_intensity_single_die' requires "
                "'mean_intensity_single_die' in df_meta. Please run "
                "calculate_flux_and_intensity before normalization."
            )
        raw_scales = valid_df["mean_intensity_single_die"].to_numpy(
            dtype=np.float64
        )
        scales = _clamp_scales(raw_scales, eps, norm_mode)
        ary_norm[idx] /= scales
        scales_record[idx] = scales.ravel()

    elif norm_mode == "by_regional_max_intensity":
        raw_scales = np.max(np.abs(ary_intensity[idx]), axis=1)
        scales = _clamp_scales(raw_scales, eps, norm_mode)
        ary_norm[idx] /= scales
        scales_record[idx] = scales.ravel()

    elif norm_mode == "by_regional_max_intensity_of_source":
        if "target_spectrum_index" not in df_meta.columns:
            raise ValueError(
                "normalization_mode='by_regional_max_intensity_of_source' "
                "requires paired metadata containing 'target_spectrum_index'. "
                "Run pairing before normalization."
            )
        paired = valid_df[valid_df["target_spectrum_index"].notna()]
        paired = paired[
            (paired["target_spectrum_index"] >= 0)
            & (paired["target_spectrum_index"] < n_total)
        ]
        if not paired.empty:
            src_idx = paired["spectrum_index"].to_numpy(dtype=int)
            tgt_idx = paired["target_spectrum_index"].to_numpy(dtype=int)
            raw_scales = np.max(np.abs(ary_intensity[src_idx]), axis=1)
            scales = _clamp_scales(raw_scales, eps, norm_mode)
            ary_norm[src_idx] /= scales
            ary_norm[tgt_idx] /= scales
            scales_record[src_idx] = scales.ravel()
            scales_record[tgt_idx] = scales.ravel()

    df_meta["normalization_scale"] = df_meta["spectrum_index"].map(
        lambda si: scales_record[int(si)] if 0 <= int(si) < n_total else 1.0
    ).astype(np.float64)
    df_meta.attrs["normalization_mode"] = norm_mode

    return ary_norm, ary_energy, df_meta
