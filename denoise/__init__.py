"""Spectral noise analysis and denoising package for GGRD-SP."""

from __future__ import annotations

from denoise.common import summarize_denoised_spectrum
from denoise.fourier import (
    denoise_batch_fourier,
    denoise_spectrum_fourier,
    get_default_fourier_config,
)
from denoise.gp import (
    SpectrumExactGPModel,
    create_mean_and_confidence_curves,
    denoise_batch_gp,
    denoise_spectrum_gp,
    fit_regional_spectrum_by_gp,
    get_default_gp_denoise_config,
    trace_gp_hyperparameter_convergence,
)
from denoise.orchestration import (
    compare_denoised_atomic_percentages,
    denoise_regional_spectra,
    denoise_single_spectrum,
    get_default_denoise_config,
    get_default_noise_analysis_config,
    perform_spectral_noise_analysis,
)
from denoise.savgol import (
    denoise_batch_savgol,
    denoise_spectrum_savgol,
    get_default_savgol_config,
)
from denoise.wavelet import (
    denoise_batch_wavelet,
    denoise_spectrum_wavelet,
    get_default_wavelet_config,
)

__all__ = [
    # Common
    "summarize_denoised_spectrum",
    # Savitzky-Golay
    "get_default_savgol_config",
    "denoise_spectrum_savgol",
    "denoise_batch_savgol",
    # Wavelet
    "get_default_wavelet_config",
    "denoise_spectrum_wavelet",
    "denoise_batch_wavelet",
    # Fourier / Butterworth
    "get_default_fourier_config",
    "denoise_spectrum_fourier",
    "denoise_batch_fourier",
    # Gaussian Process
    "SpectrumExactGPModel",
    "get_default_gp_denoise_config",
    "fit_regional_spectrum_by_gp",
    "create_mean_and_confidence_curves",
    "trace_gp_hyperparameter_convergence",
    "denoise_spectrum_gp",
    "denoise_batch_gp",
    # Orchestration
    "get_default_noise_analysis_config",
    "perform_spectral_noise_analysis",
    "get_default_denoise_config",
    "denoise_single_spectrum",
    "denoise_regional_spectra",
    "compare_denoised_atomic_percentages",
]
