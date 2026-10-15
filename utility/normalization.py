"""Spectral normalization utilities for GGRD-SP."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


from utility.flux import compute_die_total_flux


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
        'source_regional_max').
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'normalization_mode': 'total_flux' (default), 'regional_max',
          'source_regional_max', or 'no_normalization'.
        - 'eps': minimum positive floor (default 1e-6).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Tuple of (ary_intensity_normalized, ary_energy, df_meta).
    """
    cfg = config or {}
    norm_mode = str(cfg.get("normalization_mode", "total_flux")).lower()
    eps = float(cfg.get("eps", 1e-6))

    ary_norm = ary_intensity.astype(np.float32, copy=True)
    n_total = len(ary_norm)

    if norm_mode == "total_flux":
        flux_map = compute_die_total_flux(df_meta, ary_intensity, ary_energy)
        for _, row in df_meta.iterrows():
            idx = int(row["spectrum_index"])
            if 0 <= idx < n_total:
                scale = max(
                    flux_map.get((str(row["measurement_id"]), int(row["die"])), eps),
                    eps,
                )
                ary_norm[idx] = ary_norm[idx] / scale

    elif norm_mode == "regional_max":
        for _, row in df_meta.iterrows():
            idx = int(row["spectrum_index"])
            if 0 <= idx < n_total:
                peak = float(np.max(np.abs(ary_intensity[idx])))
                scale = max(peak, eps)
                ary_norm[idx] = ary_norm[idx] / scale

    elif norm_mode == "source_regional_max":
        if "spectrum_index_target" not in df_meta.columns:
            raise ValueError(
                "normalization_mode='source_regional_max' requires paired "
                "metadata containing 'spectrum_index_target'. Run pairing "
                "before normalization."
            )
        paired_mask = df_meta["spectrum_index_target"].notna()
        for _, row in df_meta[paired_mask].iterrows():
            src_idx = int(row["spectrum_index"])
            tgt_idx = int(row["spectrum_index_target"])
            if 0 <= src_idx < n_total and 0 <= tgt_idx < n_total:
                src_peak = float(np.max(np.abs(ary_intensity[src_idx])))
                scale = max(src_peak, eps)
                ary_norm[src_idx] = ary_intensity[src_idx] / scale
                ary_norm[tgt_idx] = ary_intensity[tgt_idx] / scale

    elif norm_mode == "no_normalization":
        pass
    else:
        raise ValueError(
            f"Unknown normalization_mode '{norm_mode}'. "
            "Supported options are 'total_flux', 'regional_max', "
            "'source_regional_max', and 'no_normalization'."
        )

    return ary_norm, ary_energy, df_meta
