"""Fourier and zero-phase low-pass filtering (DCT and Butterworth filtfilt)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy.fft import dct, idct
from scipy.signal import butter, filtfilt

from denoise.common import (
    _SNR_DEFINITIONS,
    _validate_spectrum,
    run_batch_denoise,
    summarize_denoised_spectrum,
)
from utility.config_schema import DenoiseFourierConfig, resolve_subconfig

_VALID_FILTER_TYPES: tuple[str, ...] = ("dct", "butterworth")
_VALID_ROLLOFF_TYPES: tuple[str, ...] = (
    "butterworth",
    "gaussian",
    "brickwall",
)
_VALID_PADDING_MODES: tuple[str, ...] = ("even", "odd", "constant")


def get_default_fourier_config() -> dict[str, Any]:
    """Return default configuration for Fourier/low-pass spectral filtering.

    Returns
    -------
    dict[str, Any]
        Default configuration dictionary from DenoiseFourierConfig schema.
    """
    return resolve_subconfig(DenoiseFourierConfig)


def _resolve_fourier_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and merge user Fourier/low-pass configuration with defaults."""
    if config is not None and config.get("_resolved"):
        return config
    cfg = resolve_subconfig(DenoiseFourierConfig, config)
    if cfg["rolloff_type"] not in _VALID_ROLLOFF_TYPES:
        raise ValueError(
            f"Invalid rolloff_type={cfg['rolloff_type']!r}; "
            f"choose one of {_VALID_ROLLOFF_TYPES}."
        )
    if cfg["padding_mode"] not in _VALID_PADDING_MODES:
        raise ValueError(
            f"Invalid padding_mode={cfg['padding_mode']!r}; "
            f"choose one of {_VALID_PADDING_MODES}."
        )
    cfg["_resolved"] = True
    return cfg


def _apply_dct_lowpass(
    intensity_arr: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Apply type-II Discrete Cosine Transform low-pass frequency filtering."""
    num_points = intensity_arr.size
    dct_coeffs = dct(intensity_arr, type=2, norm="ortho")
    normalized_freqs = np.arange(num_points, dtype=np.float64) / float(
        max(num_points - 1, 1)
    )
    cutoff_ratio = float(cfg["cutoff_ratio"])
    rolloff_type = cfg["rolloff_type"]

    if rolloff_type == "butterworth":
        filter_order = int(cfg["filter_order"])
        freq_mask = 1.0 / (
            1.0 + (normalized_freqs / cutoff_ratio) ** (2 * filter_order)
        )
    elif rolloff_type == "gaussian":
        freq_mask = np.exp(-0.5 * (normalized_freqs / cutoff_ratio) ** 2)
    else:
        freq_mask = (normalized_freqs <= cutoff_ratio).astype(np.float64)

    filtered_coeffs = dct_coeffs * freq_mask
    return np.asarray(
        idct(filtered_coeffs, type=2, norm="ortho"), dtype=np.float64
    )


def _apply_butterworth_filtfilt(
    intensity_arr: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Apply zero-phase forward-backward Butterworth low-pass filtering."""
    num_coefficients, den_coefficients = butter(
        N=int(cfg["filter_order"]),
        Wn=float(cfg["cutoff_ratio"]),
        btype="low",
        analog=False,
    )
    pad_length = min(
        intensity_arr.size - 1, 3 * max(len(num_coefficients), len(den_coefficients))
    )
    return np.asarray(
        filtfilt(
            num_coefficients,
            den_coefficients,
            intensity_arr,
            padtype=cfg["padding_mode"],
            padlen=pad_length,
        ),
        dtype=np.float64,
    )


def denoise_spectrum_fourier(
    energy: np.ndarray,
    intensity: np.ndarray,
    config: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Denoise one 1D regional spectrum using Fourier/Butterworth low-pass.

    Parameters
    ----------
    energy : np.ndarray
        1D array of binding energies in eV.
    intensity : np.ndarray
        1D array of measured spectral intensities.
    config : dict[str, Any] | None, optional
        Fourier low-pass configuration (see get_default_fourier_config).
    meta : dict[str, Any] | None, optional
        Optional metadata dictionary for error reporting.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, tuple[float, float, float]]
        Tuple of (denoised_signal, confidence_curve,
        (noise_sigma, snr, cutoff_wavelength_ev)).
    """
    cfg = _resolve_fourier_config(config)
    label = f"spectrum meta={dict(meta)}" if meta is not None else "spectrum"
    _, intensity_arr, point_spacing, _, _ = _validate_spectrum(
        energy, intensity, label
    )

    if cfg["filter_type"] == "dct":
        denoised_signal = _apply_dct_lowpass(intensity_arr, cfg)
    else:
        denoised_signal = _apply_butterworth_filtfilt(intensity_arr, cfg)

    confidence_curve, noise_sigma, snr = summarize_denoised_spectrum(
        intensity_arr, denoised_signal, cfg
    )
    cutoff_wavelength_ev = float(
        (2.0 * point_spacing) / float(cfg["cutoff_ratio"])
    )
    return (
        denoised_signal,
        confidence_curve,
        (noise_sigma, snr, cutoff_wavelength_ev),
    )


def denoise_batch_fourier(
    energies: np.ndarray,
    intensities: np.ndarray,
    metas: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Denoise a batch of 1D spectra using Fourier/Butterworth low-pass.

    Parameters
    ----------
    energies : np.ndarray
        2D array of binding energies of shape (B, N_points).
    intensities : np.ndarray
        2D array of measured intensities of shape (B, N_points).
    metas : list[dict[str, Any]]
        List of metadata dictionaries for each spectrum in the batch.
    config : dict[str, Any] | None, optional
        Fourier low-pass configuration dictionary.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (signal_curves (B, N), confidence_curves (B, N),
        summary_metrics (B, 3)).
    """
    cfg = _resolve_fourier_config(config)
    return run_batch_denoise(
        denoise_spectrum_fourier, energies, intensities, metas, cfg
    )
