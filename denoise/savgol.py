"""Savitzky-Golay polynomial filtering for regional XPS spectra."""

import copy
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from denoise.common import (
    _SNR_DEFINITIONS,
    _validate_spectrum,
    run_batch_denoise,
    summarize_denoised_spectrum,
)
from utility.config_schema import DenoiseSavgolConfig, resolve_subconfig

_VALID_BOUNDARY_MODES: tuple[str, ...] = (
    "interp",
    "mirror",
    "nearest",
    "constant",
    "wrap",
)


def get_default_savgol_config() -> dict[str, Any]:
    """Return default configuration for Savitzky-Golay spectral filtering.

    Returns
    -------
    dict[str, Any]
        Default configuration dictionary from DenoiseSavgolConfig schema.
    """
    return resolve_subconfig(DenoiseSavgolConfig)


def _resolve_savgol_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and merge user Savitzky-Golay configuration with defaults."""
    if config is not None and config.get("_resolved"):
        return config
    cfg = resolve_subconfig(DenoiseSavgolConfig, config)
    if cfg["window_length"] % 2 == 0:
        raise ValueError(
            f"window_length must be an odd integer >= 3, got {cfg['window_length']!r}."
        )
    cfg["_resolved"] = True
    return cfg


def _compute_odd_window_points(
    num_points: int,
    point_spacing: float,
    cfg: dict[str, Any],
) -> int:
    """Resolve an odd window length bounded by spectrum length and order."""
    if cfg["window_length_ev"] is not None:
        raw_points = int(round(float(cfg["window_length_ev"]) / point_spacing))
        window_points = max(3, raw_points)
        if window_points % 2 == 0:
            window_points += 1
    else:
        window_points = int(cfg["window_length"])

    max_odd = num_points if num_points % 2 == 1 else num_points - 1
    window_points = min(window_points, max_odd)
    if window_points <= cfg["polynomial_order"]:
        raise ValueError(
            f"Resolved Savitzky-Golay window_length ({window_points}) must be "
            f"strictly greater than polynomial_order "
            f"({cfg['polynomial_order']}) for spectrum of length {num_points}."
        )
    return window_points


def denoise_spectrum_savgol(
    energy: np.ndarray,
    intensity: np.ndarray,
    config: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Denoise one 1D regional spectrum using a Savitzky-Golay filter.

    Parameters
    ----------
    energy : np.ndarray
        1D array of binding energies in eV.
    intensity : np.ndarray
        1D array of measured spectral intensities.
    config : dict[str, Any] | None, optional
        Savitzky-Golay filter configuration (see get_default_savgol_config).
    meta : dict[str, Any] | None, optional
        Optional metadata dictionary for error reporting.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, tuple[float, float, float]]
        Tuple of (denoised_signal, confidence_curve,
        (noise_sigma, snr, effective_window_ev)).
    """
    cfg = _resolve_savgol_config(config)
    label = f"spectrum meta={dict(meta)}" if meta is not None else "spectrum"
    energy_arr, intensity_arr, point_spacing, _, _ = _validate_spectrum(
        energy, intensity, label
    )
    window_points = _compute_odd_window_points(
        energy_arr.size, point_spacing, cfg
    )
    denoised_signal = np.asarray(
        savgol_filter(
            intensity_arr,
            window_length=window_points,
            polyorder=cfg["polynomial_order"],
            mode=cfg["boundary_mode"],
        ),
        dtype=np.float64,
    )
    confidence_curve, noise_sigma, snr = summarize_denoised_spectrum(
        intensity_arr, denoised_signal, cfg
    )
    effective_window_ev = float(window_points * point_spacing)
    return (
        denoised_signal,
        confidence_curve,
        (noise_sigma, snr, effective_window_ev),
    )


def denoise_batch_savgol(
    energies: np.ndarray,
    intensities: np.ndarray,
    metas: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Denoise a batch of 1D spectra using Savitzky-Golay polynomial filtering.

    Parameters
    ----------
    energies : np.ndarray
        2D array of binding energies of shape (B, N_points).
    intensities : np.ndarray
        2D array of measured intensities of shape (B, N_points).
    metas : list[dict[str, Any]]
        List of metadata dictionaries for each spectrum in the batch.
    config : dict[str, Any] | None, optional
        Savitzky-Golay filter configuration dictionary.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (signal_curves (B, N), confidence_curves (B, N),
        summary_metrics (B, 3)).
    """
    cfg = _resolve_savgol_config(config)
    return run_batch_denoise(
        denoise_spectrum_savgol, energies, intensities, metas, cfg
    )
