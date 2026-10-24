"""Shared validation, SNR calculation, and selection utilities for denoise."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")
_SNR_DEFINITIONS: tuple[str, ...] = ("range", "std")
_VALID_METHODS: tuple[str, ...] = ("savgol", "wavelet", "fourier", "gp")
_VALID_APPLY_TO: tuple[str, ...] = ("both", "source", "target")


def _validate_spectrum(
    energy: np.ndarray,
    intensity: np.ndarray,
    label: str = "spectrum",
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Validate one 1D regional spectrum and return array and scale statistics.

    Parameters
    ----------
    energy : np.ndarray
        1D array of binding energies in eV.
    intensity : np.ndarray
        1D array of spectral intensities.
    label : str, optional
        Diagnostic label used in error messages (default 'spectrum').

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float, float, float]
        Tuple of (energy, intensity, point_spacing, energy_range,
        intensity_scale).
    """
    energy_arr = np.asarray(energy, dtype=np.float64)
    intensity_arr = np.asarray(intensity, dtype=np.float64)
    if (
        energy_arr.ndim != 1
        or energy_arr.shape != intensity_arr.shape
        or energy_arr.size < 5
    ):
        raise ValueError(
            f"{label}: energy and intensity must be 1D arrays of equal length "
            f">= 5, got shapes {energy_arr.shape} and {intensity_arr.shape}."
        )
    if not (np.isfinite(energy_arr).all() and np.isfinite(intensity_arr).all()):
        raise ValueError(f"{label}: non-finite values in energy or intensity.")
    if np.unique(energy_arr).size != energy_arr.size:
        raise ValueError(
            f"{label}: duplicate energy values are not supported."
        )
    point_spacing = float(np.median(np.abs(np.diff(np.sort(energy_arr)))))
    energy_range = float(energy_arr.max() - energy_arr.min())
    intensity_scale = float(np.max(np.abs(intensity_arr)))
    if point_spacing <= 0 or energy_range <= 0 or intensity_scale <= 0:
        raise ValueError(
            f"{label}: degenerate spectrum (point spacing={point_spacing}, "
            f"energy range={energy_range}, max |intensity|={intensity_scale})."
        )
    return (
        energy_arr,
        intensity_arr,
        point_spacing,
        energy_range,
        intensity_scale,
    )


def summarize_denoised_spectrum(
    intensity: np.ndarray,
    denoised_signal: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, float, float]:
    """Calculate confidence band, noise sigma, and SNR from denoised signal.

    Parameters
    ----------
    intensity : np.ndarray
        1D array of measured intensities.
    denoised_signal : np.ndarray
        1D array of filtered/denoised signal intensities.
    config : dict[str, Any] | None, optional
        Configuration dictionary supporting:
        - 'snr_definition': 'range' (default) or 'std'.
        - 'confidence_z': z-score multiplier for confidence band (default 1.96).

    Returns
    -------
    tuple[np.ndarray, float, float]
        Tuple of (confidence_half_width_curve, noise_sigma, snr).
    """
    cfg = config or {}
    snr_definition = str(cfg.get("snr_definition", "range"))
    if snr_definition not in _SNR_DEFINITIONS:
        raise ValueError(
            f"Invalid snr_definition={snr_definition!r}; "
            f"choose one of {_SNR_DEFINITIONS}."
        )
    confidence_z = float(cfg.get("confidence_z", 1.96))
    if confidence_z <= 0:
        raise ValueError(f"confidence_z must be > 0, got {confidence_z}.")

    residual = np.asarray(intensity, dtype=np.float64) - np.asarray(
        denoised_signal, dtype=np.float64
    )
    ddof = 1 if residual.size > 1 else 0
    noise_sigma = float(np.std(residual, ddof=ddof))
    safe_sigma = max(noise_sigma, 1e-12)

    if snr_definition == "range":
        signal_strength = float(
            np.max(denoised_signal) - np.min(denoised_signal)
        )
    else:
        signal_strength = float(np.std(denoised_signal))

    snr = float(signal_strength / safe_sigma) if noise_sigma > 0 else np.inf
    confidence_curve = np.full_like(
        denoised_signal, confidence_z * noise_sigma, dtype=np.float64
    )
    return confidence_curve, noise_sigma, snr


def _validate_meta_indexing(df_meta: pd.DataFrame, required: set[str]) -> None:
    """Validate required columns and spectrum_index integrity in df_meta."""
    missing = required - set(df_meta.columns)
    if missing:
        raise ValueError(
            f"df_meta is missing required columns: {sorted(missing)}."
        )
    if not df_meta.index.is_unique:
        raise ValueError("df_meta.index must be unique for noise analysis.")
    if not pd.api.types.is_integer_dtype(df_meta["spectrum_index"]):
        raise ValueError(
            f"df_meta['spectrum_index'] must be integer, got dtype "
            f"{df_meta['spectrum_index'].dtype}."
        )
    if df_meta["spectrum_index"].duplicated().any():
        duplicated = df_meta.loc[
            df_meta["spectrum_index"].duplicated(), "spectrum_index"
        ]
        raise ValueError(
            f"Duplicate spectrum_index values in df_meta: "
            f"{duplicated.unique()[:5]}."
        )


def _select_spectra(
    df_meta: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Select df_meta rows to analyse by split, die, and region."""
    _validate_meta_indexing(df_meta, {"spectrum_index", "die", "region"})
    unpaired_split = cfg.get("unpaired_split")
    if unpaired_split is not None and unpaired_split not in _VALID_SPLITS:
        raise ValueError(
            f"unpaired_split must be None or one of {_VALID_SPLITS}, "
            f"got {unpaired_split!r}."
        )
    if "split" in df_meta.columns:
        unpaired = df_meta["split"].isna() | (df_meta["split"] == "none")
        invalid = df_meta.loc[
            ~unpaired & ~df_meta["split"].isin(_VALID_SPLITS), "split"
        ]
        if not invalid.empty:
            raise ValueError(
                f"Invalid split labels {sorted(invalid.unique().astype(str))}; "
                f"valid labels: {_VALID_SPLITS} or missing (unpaired)."
            )
        split = df_meta["split"].astype(object).where(~unpaired, None)
    elif unpaired_split is None:
        raise ValueError(
            "df_meta has no 'split' column and unpaired_split is None, so no "
            "spectra can be selected. Run splitting first or set unpaired_split."
        )
    else:
        split = pd.Series([None] * len(df_meta), index=df_meta.index, dtype=object)

    if unpaired_split is not None:
        split = split.where(split.notna(), unpaired_split)

    splits = cfg.get("splits") or list(_VALID_SPLITS)
    bad_splits = [s for s in splits if s not in _VALID_SPLITS]
    if bad_splits:
        raise ValueError(
            f"Invalid splits {bad_splits}; valid splits: {_VALID_SPLITS}."
        )
    selected = df_meta[split.isin(splits)]
    for column, key in (("die", "dies"), ("region", "regions")):
        if cfg.get(key) is None:
            continue
        absent = set(cfg[key]) - set(selected[column].unique())
        if absent:
            raise ValueError(
                f"{key.capitalize()} {sorted(absent)} not found in splits "
                f"{splits}; available {key}: "
                f"{sorted(selected[column].unique())}."
            )
        selected = selected[selected[column].isin(cfg[key])]
    if selected.empty:
        raise ValueError(
            f"No spectra selected (splits={splits}, dies={cfg.get('dies')}, "
            f"regions={cfg.get('regions')}, unpaired_split={unpaired_split})."
        )
    return selected


def _select_denoise_rows(
    df_meta: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Select df_meta rows to denoise by tool scope and target regions."""
    _validate_meta_indexing(
        df_meta, {"spectrum_index", "tool", "die", "region"}
    )
    apply_to = str(cfg.get("apply_to", "both")).lower()
    if apply_to not in _VALID_APPLY_TO:
        raise ValueError(
            f"Invalid apply_to={apply_to!r}; choose one of {_VALID_APPLY_TO}."
        )

    source_tool = str(cfg.get("source_tool", "J4"))
    target_tool = str(cfg.get("target_tool", "J5"))
    if apply_to == "source":
        allowed_tools = {source_tool}
    elif apply_to == "target":
        allowed_tools = {target_tool}
    else:
        allowed_tools = {source_tool, target_tool}

    selected = df_meta[df_meta["tool"].astype(str).isin(allowed_tools)]
    regions = cfg.get("regions")
    if regions is not None:
        if isinstance(regions, str) or len(regions) == 0:
            raise ValueError(
                f"'regions' must be None or a non-empty list, got {regions!r}."
            )
        available_regions = set(df_meta["region"].astype(str).unique())
        missing_regions = set(regions) - available_regions
        if missing_regions:
            raise ValueError(
                f"Configured denoise regions {sorted(missing_regions)} not "
                f"found in df_meta; available: {sorted(available_regions)}."
            )
        selected = selected[selected["region"].isin(regions)]

    return selected
