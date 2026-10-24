"""Savitzky-Golay polynomial filtering for regional XPS spectra."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from denoise.common import (
    _SNR_DEFINITIONS,
    _validate_spectrum,
    summarize_denoised_spectrum,
)

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
        Default configuration dictionary:
        - 'window_length': Odd integer window length in points (default 11).
        - 'window_length_ev': Optional window width in eV (default None).
        - 'polynomial_order': Degree of local polynomial fit (default 3).
        - 'boundary_mode': Edge padding mode for savgol_filter
          (default 'interp').
        - 'snr_definition': 'range' or 'std' (default 'range').
        - 'confidence_z': Confidence band multiplier (default 1.96).
    """
    return {
        "window_length": 11,
        "window_length_ev": None,
        "polynomial_order": 3,
        "boundary_mode": "interp",
        "snr_definition": "range",
        "confidence_z": 1.96,
    }


def _resolve_savgol_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and merge user Savitzky-Golay configuration with defaults."""
    if config is not None and config.get("_resolved"):
        return config
    cfg = get_default_savgol_config()
    user_cfg = copy.deepcopy(config or {})
    for key in user_cfg:
        if key not in cfg:
            raise ValueError(
                f"Unknown Savitzky-Golay config key '{key}'. "
                f"Valid keys: {sorted(cfg)}"
            )
    cfg.update(user_cfg)

    polynomial_order = cfg["polynomial_order"]
    if (
        isinstance(polynomial_order, bool)
        or not isinstance(polynomial_order, int)
        or polynomial_order < 1
    ):
        raise ValueError(
            f"polynomial_order must be an integer >= 1, "
            f"got {polynomial_order!r}."
        )
    window_length = cfg["window_length"]
    if (
        isinstance(window_length, bool)
        or not isinstance(window_length, int)
        or window_length < 3
        or window_length % 2 == 0
    ):
        raise ValueError(
            f"window_length must be an odd integer >= 3, "
            f"got {window_length!r}."
        )
    window_length_ev = cfg["window_length_ev"]
    if window_length_ev is not None and (
        isinstance(window_length_ev, bool)
        or not isinstance(window_length_ev, (int, float))
        or window_length_ev <= 0
    ):
        raise ValueError(
            f"window_length_ev must be None or > 0, got {window_length_ev!r}."
        )
    if cfg["boundary_mode"] not in _VALID_BOUNDARY_MODES:
        raise ValueError(
            f"Invalid boundary_mode={cfg['boundary_mode']!r}; "
            f"choose one of {_VALID_BOUNDARY_MODES}."
        )
    if cfg["snr_definition"] not in _SNR_DEFINITIONS:
        raise ValueError(
            f"Invalid snr_definition={cfg['snr_definition']!r}; "
            f"choose one of {_SNR_DEFINITIONS}."
        )
    if cfg["confidence_z"] <= 0:
        raise ValueError(
            f"confidence_z must be > 0, got {cfg['confidence_z']}."
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
    num_spectra, num_points = energies.shape
    signal_curves = np.empty((num_spectra, num_points), dtype=np.float64)
    confidence_curves = np.empty((num_spectra, num_points), dtype=np.float64)
    summary_metrics = np.empty((num_spectra, 3), dtype=np.float64)

    for idx, meta in enumerate(metas):
        (
            signal_curves[idx],
            confidence_curves[idx],
            summary_metrics[idx],
        ) = denoise_spectrum_savgol(
            energies[idx], intensities[idx], cfg, meta=meta
        )
    return signal_curves, confidence_curves, summary_metrics
