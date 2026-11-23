"""Source-tool train-only RMS spectral intensity standardization utilities."""

from typing import Any
import warnings

import numpy as np
import pandas as pd

_ACTIVE_SPLITS: tuple[str, ...] = ("train", "val", "test")
_RMS_COLUMN: str = "intensity_standardization_rms"


def _validate_intensity_array(ary_intensity: np.ndarray, context: str) -> None:
    """Validate that ary_intensity is a 2D NumPy array."""
    if ary_intensity.ndim != 2:
        raise ValueError(
            f"{context} requires a 2D ary_intensity array; "
            f"got shape {ary_intensity.shape}."
        )


def _validate_required_columns(
    df_meta: pd.DataFrame,
    required: tuple[str, ...],
    context: str,
) -> None:
    """Validate that required columns exist in df_meta."""
    missing = [col for col in required if col not in df_meta.columns]
    if missing:
        raise ValueError(
            f"{context} requires columns {list(required)} in df_meta; "
            f"missing {missing}."
        )


def _validate_unique_spectrum_indices(
    df_meta: pd.DataFrame,
    valid_mask: pd.Series,
    context: str,
) -> None:
    """Validate that valid spectrum_index values in df_meta are unique."""
    valid_indices = df_meta.loc[valid_mask, "spectrum_index"]
    dup_mask = valid_indices.duplicated()
    if bool(dup_mask.any()):
        dup_vals = sorted(valid_indices[dup_mask].unique().tolist())
        raise ValueError(
            f"{context} requires unique valid 'spectrum_index' values in "
            f"df_meta; found duplicate indices {dup_vals[:10]}."
        )


def _parse_fit_config(config: dict[str, Any] | None) -> tuple[str, float]:
    """Validate and extract (source_tool, eps) from standardization config."""
    cfg = config or {}
    eps = float(cfg.get("eps", 1e-6))
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError(
            f"Standardization 'eps' must be a finite positive float; "
            f"got {eps!r}."
        )

    source_tool = cfg.get("source_tool")
    if not isinstance(source_tool, str) or not source_tool.strip():
        raise ValueError(
            "Intensity standardization requires a non-empty 'source_tool' "
            f"string in config; got {source_tool!r}."
        )
    return source_tool.strip(), eps


def _extract_recorded_rms(df: pd.DataFrame) -> float:
    """Extract and validate the scalar RMS factor from a metadata DataFrame."""
    # Ensure the recorded RMS column exists and select active rows.
    if _RMS_COLUMN not in df.columns:
        raise ValueError(
            f"Inverse intensity standardization requires '{_RMS_COLUMN}' in "
            "df_meta or reference_df_meta; column was not found."
        )
    if "split" in df.columns:
        subset = df[df["split"].isin(_ACTIVE_SPLITS)]
        if subset.empty:
            raise ValueError(
                "No rows with split in ('train', 'val', 'test') found when "
                f"reading '{_RMS_COLUMN}' for inverse standardization."
            )
    else:
        subset = df
        if subset.empty:
            raise ValueError(
                f"Cannot extract '{_RMS_COLUMN}' from an empty metadata "
                "DataFrame."
            )

    # Validate that all recorded RMS values are finite, positive, and equal.
    values = subset[_RMS_COLUMN].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Non-finite '{_RMS_COLUMN}' values encountered; expected finite "
            "positive scalars."
        )
    if np.any(values <= 0.0):
        raise ValueError(
            f"Non-positive '{_RMS_COLUMN}' values encountered; expected "
            f"positive scalars, got min={float(np.min(values)):.4e}."
        )
    rms = float(values[0])
    if not np.allclose(values, rms, rtol=1e-6, atol=1e-12):
        raise ValueError(
            f"Inconsistent '{_RMS_COLUMN}' values across standardized rows; "
            f"expected a single scalar RMS, got range "
            f"[{float(np.min(values)):.4e}, {float(np.max(values)):.4e}]."
        )
    return rms


def fit_intensity_standardization(
    ary_intensity: np.ndarray,
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> float:
    """Fit scalar RMS intensity scale strictly on source-tool training rows.

    Parameters
    ----------
    ary_intensity : np.ndarray
        2D normalized intensity array of shape (N_total, N_points).
    df_meta : pd.DataFrame
        Metadata DataFrame containing 'split', 'tool', and 'spectrum_index'.
    config : dict[str, Any] | None, optional
        Standardization configuration containing 'source_tool' and optional
        'eps' (default 1e-6).

    Returns
    -------
    float
        Positive scalar RMS standardization factor (>= eps).

    Raises
    ------
    ValueError
        If required columns or source_tool are missing, eps <= 0, no matching
        source-tool training rows exist, or computed RMS is non-finite.
    """
    # Validate config, intensity shape, and required metadata columns.
    source_tool, eps = _parse_fit_config(config)
    _validate_intensity_array(ary_intensity, "fit_intensity_standardization")
    _validate_required_columns(
        df_meta,
        ("split", "tool", "spectrum_index"),
        "fit_intensity_standardization",
    )

    # Filter strictly to source-tool training rows to prevent data leakage.
    n_total = len(ary_intensity)
    valid_mask = (df_meta["spectrum_index"] >= 0) & (
        df_meta["spectrum_index"] < n_total
    )
    _validate_unique_spectrum_indices(
        df_meta, valid_mask, "fit_intensity_standardization"
    )
    train_src_mask = (
        valid_mask
        & (df_meta["split"] == "train")
        & (df_meta["tool"].astype(str) == source_tool)
    )
    train_src_df = df_meta[train_src_mask]
    if train_src_df.empty:
        raise ValueError(
            f"No training spectra found for source_tool={source_tool!r} with "
            "split == 'train' and valid spectrum_index."
        )

    # Compute scalar RMS in float64 across all points of source training rows.
    idx = train_src_df["spectrum_index"].to_numpy(dtype=int)
    x_train = ary_intensity[idx].astype(np.float64, copy=False)
    if x_train.size == 0:
        raise ValueError(
            f"Training intensity slice is empty for source_tool={source_tool!r}."
        )
    raw_rms = float(np.sqrt(np.mean(np.square(x_train))))
    if not np.isfinite(raw_rms):
        raise ValueError(
            f"Computed non-finite RMS ({raw_rms!r}) for "
            f"source_tool={source_tool!r}."
        )
    if raw_rms <= eps:
        warnings.warn(
            f"Intensity standardization RMS ({raw_rms:.4e}) <= eps "
            f"({eps:.4e}) for source_tool={source_tool!r}. "
            "Clamping RMS to eps.",
            UserWarning,
            stacklevel=2,
        )
        return eps
    return raw_rms


def standardize_spectral_intensity(
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Apply scale-only RMS standardization to normalized spectral intensities.

    Fits the scalar RMS strictly on source-tool training rows and divides all
    source and target spectra with split in ('train', 'val', 'test') by RMS,
    leaving unpaired rows (split == 'none') untouched.

    Parameters
    ----------
    ary_intensity : np.ndarray
        2D normalized intensity array of shape (N_total, N_points).
    ary_energy : np.ndarray
        2D binding energy array of shape (N_total, N_points).
    df_meta : pd.DataFrame
        Metadata DataFrame containing 'split', 'tool', and 'spectrum_index'.
    config : dict[str, Any] | None, optional
        Standardization configuration ('enabled', 'eps', 'source_tool').

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Tuple of (ary_intensity_standardized, ary_energy, df_meta_out) with
        'intensity_standardization_rms' recorded in df_meta_out.
    """
    # Validate input shape and short-circuit when standardization is disabled.
    _validate_intensity_array(ary_intensity, "standardize_spectral_intensity")
    cfg = config or {}
    enabled = bool(cfg.get("enabled", False))
    df_out = df_meta.copy()
    ary_out = ary_intensity.astype(np.float32, copy=True)

    if not enabled:
        df_out[_RMS_COLUMN] = np.float64(1.0)
        df_out.attrs["intensity_standardization_enabled"] = False
        return ary_out, ary_energy, df_out

    # Fit scalar RMS on source-train rows and scale all active split spectra.
    rms = fit_intensity_standardization(ary_intensity, df_meta, config=cfg)
    df_out[_RMS_COLUMN] = np.float64(1.0)

    n_total = len(ary_out)
    active_mask = (
        df_out["split"].isin(_ACTIVE_SPLITS)
        & (df_out["spectrum_index"] >= 0)
        & (df_out["spectrum_index"] < n_total)
    )
    idx = df_out.loc[active_mask, "spectrum_index"].to_numpy(dtype=int)
    if idx.size > 0:
        ary_out[idx] /= np.float32(rms)
        df_out.loc[active_mask, _RMS_COLUMN] = np.float64(rms)

    df_out.attrs["intensity_standardization_enabled"] = True
    return ary_out, ary_energy, df_out


def inverse_standardize_spectral_intensity(
    ary_intensity: np.ndarray,
    df_meta: pd.DataFrame,
    reference_df_meta: pd.DataFrame | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Invert RMS intensity standardization using recorded metadata factor.

    Parameters
    ----------
    ary_intensity : np.ndarray
        2D standardized intensity array (e.g., Stage 10 model predictions).
    df_meta : pd.DataFrame
        Metadata DataFrame for ary_intensity containing 'spectrum_index'.
    reference_df_meta : pd.DataFrame | None, optional
        Reference metadata DataFrame holding 'intensity_standardization_rms'
        (e.g., Stage 5 df_meta when inverting Stage 10 predictions).

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        Tuple of (ary_intensity_unstandardized, df_meta_out) with
        'intensity_standardization_rms' recorded on df_meta_out.
    """
    # Validate input shape and extract recorded scalar RMS factor.
    _validate_intensity_array(
        ary_intensity, "inverse_standardize_spectral_intensity"
    )
    source_meta = reference_df_meta if reference_df_meta is not None else df_meta
    rms = _extract_recorded_rms(source_meta)

    df_out = df_meta.copy()
    ary_inv = ary_intensity.astype(np.float32, copy=True)
    if df_out.empty:
        df_out[_RMS_COLUMN] = pd.Series(dtype=np.float64)
        return ary_inv, df_out

    # Validate spectrum indices and select rows to rescale back to I_norm.
    _validate_required_columns(
        df_out,
        ("spectrum_index",),
        "inverse_standardize_spectral_intensity",
    )
    n_total = len(ary_inv)
    valid_mask = (df_out["spectrum_index"] >= 0) & (
        df_out["spectrum_index"] < n_total
    )
    _validate_unique_spectrum_indices(
        df_out, valid_mask, "inverse_standardize_spectral_intensity"
    )
    if "split" not in df_out.columns:
        active_mask = valid_mask
    elif reference_df_meta is not None:
        # Predicted metadata may use 'unassigned'/'unpaired'; skip only 'none'.
        active_mask = valid_mask & (df_out["split"] != "none")
    else:
        active_mask = valid_mask & df_out["split"].isin(_ACTIVE_SPLITS)

    # Multiply active rows by RMS and record the factor on df_meta_out.
    df_out[_RMS_COLUMN] = np.float64(1.0)
    idx = df_out.loc[active_mask, "spectrum_index"].to_numpy(dtype=int)
    if idx.size > 0:
        ary_inv[idx] *= np.float32(rms)
        df_out.loc[active_mask, _RMS_COLUMN] = np.float64(rms)

    return ary_inv, df_out
