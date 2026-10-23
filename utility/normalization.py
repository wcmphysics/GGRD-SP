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


def _check_and_clamp_scale(
    scale: float,
    eps: float,
    context: str,
) -> float:
    """Check scale against eps, issuing a warning if scale <= eps."""
    if scale <= eps:
        warnings.warn(
            f"Normalization scale {scale:.4e} for {context} is <= eps "
            f"({eps:.4e}). Intensity values may be near zero or unphysical. "
            "Clamping scale to eps.",
            UserWarning,
            stacklevel=3,
        )
        return eps
    return scale


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
        'spectrum_index' (and 'spectrum_index_target' for
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

    ary_norm = ary_intensity.astype(np.float32, copy=True)
    if norm_mode == "no_normalization":
        return ary_norm, ary_energy, df_meta

    n_total = len(ary_norm)

    if norm_mode == "by_total_flux_single_die":
        flux_map = compute_die_total_flux(df_meta, ary_intensity, ary_energy)
        for _, row in df_meta.iterrows():
            idx = int(row["spectrum_index"])
            if 0 <= idx < n_total:
                key = (str(row["measurement_id"]), int(row["die"]))
                raw_scale = float(flux_map.get(key, 0.0))
                ctx = f"session '{row['measurement_id']}', die {row['die']}"
                scale = _check_and_clamp_scale(raw_scale, eps, ctx)
                ary_norm[idx] = ary_norm[idx] / scale

    elif norm_mode == "by_mean_intensity_single_die":
        if "mean_intensity_single_die" not in df_meta.columns:
            raise ValueError(
                "normalization_mode='by_mean_intensity_single_die' requires "
                "'mean_intensity_single_die' in df_meta. Please run "
                "calculate_flux_and_intensity before normalization."
            )
        for _, row in df_meta.iterrows():
            idx = int(row["spectrum_index"])
            if 0 <= idx < n_total:
                raw_scale = float(row["mean_intensity_single_die"])
                ctx = f"spectrum index {idx} (session '{row['measurement_id']}', die {row['die']})"
                scale = _check_and_clamp_scale(raw_scale, eps, ctx)
                ary_norm[idx] = ary_norm[idx] / scale

    elif norm_mode == "by_regional_max_intensity":
        for _, row in df_meta.iterrows():
            idx = int(row["spectrum_index"])
            if 0 <= idx < n_total:
                raw_scale = float(np.max(np.abs(ary_intensity[idx])))
                ctx = f"spectrum index {idx} (region '{row.get('region', 'unknown')}')"
                scale = _check_and_clamp_scale(raw_scale, eps, ctx)
                ary_norm[idx] = ary_norm[idx] / scale

    elif norm_mode == "by_regional_max_intensity_of_source":
        if "spectrum_index_target" not in df_meta.columns:
            raise ValueError(
                "normalization_mode='by_regional_max_intensity_of_source' "
                "requires paired metadata containing 'spectrum_index_target'. "
                "Run pairing before normalization."
            )
        paired_mask = df_meta["spectrum_index_target"].notna()
        for _, row in df_meta[paired_mask].iterrows():
            src_idx = int(row["spectrum_index"])
            tgt_idx = int(row["spectrum_index_target"])
            if 0 <= src_idx < n_total and 0 <= tgt_idx < n_total:
                raw_scale = float(np.max(np.abs(ary_intensity[src_idx])))
                ctx = f"source spectrum index {src_idx} (region '{row.get('region', 'unknown')}')"
                scale = _check_and_clamp_scale(raw_scale, eps, ctx)
                ary_norm[src_idx] = ary_intensity[src_idx] / scale
                ary_norm[tgt_idx] = ary_intensity[tgt_idx] / scale

    return ary_norm, ary_energy, df_meta
