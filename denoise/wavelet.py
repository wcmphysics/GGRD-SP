"""Wavelet shrinkage denoising (Stationary and Discrete Wavelet Transforms)."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pywt

from denoise.common import (
    _SNR_DEFINITIONS,
    _validate_spectrum,
    summarize_denoised_spectrum,
)

_VALID_TRANSFORM_TYPES: tuple[str, ...] = ("stationary", "discrete")
_VALID_THRESHOLD_MODES: tuple[str, ...] = ("soft", "hard")
_VALID_THRESHOLD_RULES: tuple[str, ...] = ("universal", "bayes")


def get_default_wavelet_config() -> dict[str, Any]:
    """Return default configuration for wavelet shrinkage denoising.

    Returns
    -------
    dict[str, Any]
        Default configuration dictionary:
        - 'transform_type': 'stationary' (default) or 'discrete'.
        - 'wavelet_name': Mother wavelet family name (default 'sym4').
        - 'decomposition_level': Decomposition depth >= 1 (default 2).
        - 'threshold_mode': 'soft' (default) or 'hard'.
        - 'threshold_rule': 'universal' (default) or 'bayes'.
        - 'threshold_scale': Positive multiplier on threshold (default 1.0).
        - 'boundary_mode': Signal padding mode for discrete transform
          (default 'symmetric').
        - 'snr_definition': 'range' or 'std' (default 'range').
        - 'confidence_z': Confidence band multiplier (default 1.96).
    """
    return {
        "transform_type": "stationary",
        "wavelet_name": "sym4",
        "decomposition_level": 2,
        "threshold_mode": "soft",
        "threshold_rule": "universal",
        "threshold_scale": 1.0,
        "boundary_mode": "symmetric",
        "snr_definition": "range",
        "confidence_z": 1.96,
    }


def _resolve_wavelet_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and merge user wavelet configuration with defaults."""
    if config is not None and config.get("_resolved"):
        return config
    cfg = get_default_wavelet_config()
    user_cfg = copy.deepcopy(config or {})
    for key in user_cfg:
        if key not in cfg:
            raise ValueError(
                f"Unknown wavelet config key '{key}'. "
                f"Valid keys: {sorted(cfg)}"
            )
    cfg.update(user_cfg)

    if cfg["transform_type"] not in _VALID_TRANSFORM_TYPES:
        raise ValueError(
            f"Invalid transform_type={cfg['transform_type']!r}; "
            f"choose one of {_VALID_TRANSFORM_TYPES}."
        )
    try:
        pywt.Wavelet(str(cfg["wavelet_name"]))
    except ValueError as exc:
        raise ValueError(
            f"Invalid wavelet_name={cfg['wavelet_name']!r}: {exc}"
        ) from exc

    decomposition_level = cfg["decomposition_level"]
    if decomposition_level is not None and (
        isinstance(decomposition_level, bool)
        or not isinstance(decomposition_level, int)
        or decomposition_level < 1
    ):
        raise ValueError(
            f"decomposition_level must be None or an integer >= 1, "
            f"got {decomposition_level!r}."
        )
    if cfg["threshold_mode"] not in _VALID_THRESHOLD_MODES:
        raise ValueError(
            f"Invalid threshold_mode={cfg['threshold_mode']!r}; "
            f"choose one of {_VALID_THRESHOLD_MODES}."
        )
    if cfg["threshold_rule"] not in _VALID_THRESHOLD_RULES:
        raise ValueError(
            f"Invalid threshold_rule={cfg['threshold_rule']!r}; "
            f"choose one of {_VALID_THRESHOLD_RULES}."
        )
    threshold_scale = cfg["threshold_scale"]
    if (
        isinstance(threshold_scale, bool)
        or not isinstance(threshold_scale, (int, float))
        or threshold_scale <= 0
    ):
        raise ValueError(
            f"threshold_scale must be > 0, got {threshold_scale!r}."
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


def _estimate_mad_noise_sigma(finest_detail_coeffs: np.ndarray) -> float:
    """Estimate noise std from finest wavelet detail coefficients via MAD."""
    median_val = float(np.median(finest_detail_coeffs))
    median_abs_deviation = float(
        np.median(np.abs(finest_detail_coeffs - median_val))
    )
    return median_abs_deviation / 0.6745


def _compute_level_threshold(
    detail_coeffs: np.ndarray,
    estimated_noise_sigma: float,
    num_points: int,
    threshold_rule: str,
    threshold_scale: float,
) -> float:
    """Compute shrinkage threshold for one decomposition level."""
    if estimated_noise_sigma <= 0.0:
        return 0.0
    if threshold_rule == "universal":
        base_threshold = estimated_noise_sigma * math.sqrt(
            2.0 * math.log(max(num_points, 2))
        )
    else:
        detail_variance = float(np.mean(detail_coeffs**2))
        signal_variance = max(detail_variance - estimated_noise_sigma**2, 0.0)
        if signal_variance <= 1e-16:
            base_threshold = float(np.max(np.abs(detail_coeffs)))
        else:
            base_threshold = (estimated_noise_sigma**2) / math.sqrt(
                signal_variance
            )
    return float(threshold_scale * base_threshold)


def _denoise_stationary_wavelet(
    intensity_arr: np.ndarray,
    wavelet_object: pywt.Wavelet,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, int]:
    """Apply Stationary Wavelet Transform (SWT) shrinkage with padding."""
    num_points = intensity_arr.size
    requested_level = cfg["decomposition_level"] or 2
    divisor = 2**requested_level
    remainder = num_points % divisor
    pad_total = (divisor - remainder) % divisor
    if num_points + pad_total < divisor * 2:
        pad_total += divisor
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    padded_intensity = np.pad(
        intensity_arr, (pad_left, pad_right), mode="symmetric"
    )
    max_swt_level = pywt.swt_max_level(padded_intensity.size)
    effective_level = min(requested_level, max(1, max_swt_level))

    swt_coeffs = pywt.swt(
        padded_intensity, wavelet_object, level=effective_level
    )
    finest_detail_coeffs = swt_coeffs[-1][1]
    estimated_noise_sigma = _estimate_mad_noise_sigma(finest_detail_coeffs)

    thresholded_coeffs: list[tuple[np.ndarray, np.ndarray]] = []
    for approx_coeffs, detail_coeffs in swt_coeffs:
        threshold_value = _compute_level_threshold(
            detail_coeffs,
            estimated_noise_sigma,
            num_points,
            cfg["threshold_rule"],
            float(cfg["threshold_scale"]),
        )
        shrunk_detail = pywt.threshold(
            detail_coeffs,
            value=threshold_value,
            mode=cfg["threshold_mode"],
        )
        thresholded_coeffs.append((approx_coeffs, shrunk_detail))

    reconstructed = pywt.iswt(thresholded_coeffs, wavelet_object)
    denoised_signal = np.asarray(
        reconstructed[pad_left : pad_left + num_points], dtype=np.float64
    )
    return denoised_signal, effective_level


def _denoise_discrete_wavelet(
    intensity_arr: np.ndarray,
    wavelet_object: pywt.Wavelet,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, int]:
    """Apply standard Discrete Wavelet Transform (DWT) shrinkage."""
    num_points = intensity_arr.size
    max_dwt_level = pywt.dwt_max_level(num_points, wavelet_object.dec_len)
    requested_level = cfg["decomposition_level"] or max(1, max_dwt_level)
    effective_level = min(requested_level, max(1, max_dwt_level))

    coeffs = pywt.wavedec(
        intensity_arr,
        wavelet_object,
        mode=cfg["boundary_mode"],
        level=effective_level,
    )
    finest_detail_coeffs = coeffs[-1]
    estimated_noise_sigma = _estimate_mad_noise_sigma(finest_detail_coeffs)

    thresholded_coeffs = [coeffs[0]]
    for detail_coeffs in coeffs[1:]:
        threshold_value = _compute_level_threshold(
            detail_coeffs,
            estimated_noise_sigma,
            num_points,
            cfg["threshold_rule"],
            float(cfg["threshold_scale"]),
        )
        shrunk_detail = pywt.threshold(
            detail_coeffs,
            value=threshold_value,
            mode=cfg["threshold_mode"],
        )
        thresholded_coeffs.append(shrunk_detail)

    reconstructed = pywt.waverec(
        thresholded_coeffs, wavelet_object, mode=cfg["boundary_mode"]
    )
    denoised_signal = np.asarray(reconstructed[:num_points], dtype=np.float64)
    return denoised_signal, effective_level


def denoise_spectrum_wavelet(
    energy: np.ndarray,
    intensity: np.ndarray,
    config: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Denoise one 1D regional spectrum using wavelet coefficient shrinkage.

    Parameters
    ----------
    energy : np.ndarray
        1D array of binding energies in eV.
    intensity : np.ndarray
        1D array of measured spectral intensities.
    config : dict[str, Any] | None, optional
        Wavelet shrinkage configuration (see get_default_wavelet_config).
    meta : dict[str, Any] | None, optional
        Optional metadata dictionary for error reporting.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, tuple[float, float, float]]
        Tuple of (denoised_signal, confidence_curve,
        (noise_sigma, snr, coarse_scale_ev)).
    """
    cfg = _resolve_wavelet_config(config)
    label = f"spectrum meta={dict(meta)}" if meta is not None else "spectrum"
    energy_arr, intensity_arr, point_spacing, _, _ = _validate_spectrum(
        energy, intensity, label
    )
    wavelet_object = pywt.Wavelet(str(cfg["wavelet_name"]))

    if cfg["transform_type"] == "stationary":
        denoised_signal, effective_level = _denoise_stationary_wavelet(
            intensity_arr, wavelet_object, cfg
        )
    else:
        denoised_signal, effective_level = _denoise_discrete_wavelet(
            intensity_arr, wavelet_object, cfg
        )

    confidence_curve, noise_sigma, snr = summarize_denoised_spectrum(
        intensity_arr, denoised_signal, cfg
    )
    coarse_scale_ev = float((2**effective_level) * point_spacing)
    return (
        denoised_signal,
        confidence_curve,
        (noise_sigma, snr, coarse_scale_ev),
    )


def denoise_batch_wavelet(
    energies: np.ndarray,
    intensities: np.ndarray,
    metas: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Denoise a batch of 1D spectra using wavelet coefficient shrinkage.

    Parameters
    ----------
    energies : np.ndarray
        2D array of binding energies of shape (B, N_points).
    intensities : np.ndarray
        2D array of measured intensities of shape (B, N_points).
    metas : list[dict[str, Any]]
        List of metadata dictionaries for each spectrum in the batch.
    config : dict[str, Any] | None, optional
        Wavelet shrinkage configuration dictionary.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (signal_curves (B, N), confidence_curves (B, N),
        summary_metrics (B, 3)).
    """
    cfg = _resolve_wavelet_config(config)
    num_spectra, num_points = energies.shape
    signal_curves = np.empty((num_spectra, num_points), dtype=np.float64)
    confidence_curves = np.empty((num_spectra, num_points), dtype=np.float64)
    summary_metrics = np.empty((num_spectra, 3), dtype=np.float64)

    for idx, meta in enumerate(metas):
        (
            signal_curves[idx],
            confidence_curves[idx],
            summary_metrics[idx],
        ) = denoise_spectrum_wavelet(
            energies[idx], intensities[idx], cfg, meta=meta
        )
    return signal_curves, confidence_curves, summary_metrics
