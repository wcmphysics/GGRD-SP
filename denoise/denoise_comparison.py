"""Standalone figures explaining Fourier, Savgol and wavelet filtering."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt

from denoise.fourier_comparison import plot_fourier_comparison
from denoise.savgol import denoise_spectrum_savgol
from denoise.wavelet import (
    _compute_level_threshold,
    _estimate_mad_noise_sigma,
    denoise_spectrum_wavelet,
)


def _select_spectrum(data: tuple, selector: dict) -> tuple:
    """Select exactly one spectrum and discard unused array padding."""
    intensity, energy, metadata = data
    selected = metadata
    for key in ("measurement_id", "die", "region"):
        selected = selected[selected[key] == selector[key]]
    if len(selected) != 1:
        raise ValueError(
            f"Denoise plot spectrum {selector} matched {len(selected)} rows; "
            "expected exactly one."
        )
    row = selected.iloc[0]
    index, length = row["spectrum_index"], row["n_points"]
    return energy[index, :length], intensity[index, :length]


def _plot_savgol(energy: np.ndarray, raw: np.ndarray, config: dict, path: Path) -> None:
    """Show local polynomial fits and their full-spectrum results."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[1].plot(energy, raw, color="black", alpha=0.5, label="Raw")
    largest_window = min(max(item["window_length"] for item in config),
                         len(raw) if len(raw) % 2 else len(raw) - 1)
    half = largest_window // 2
    center = int(np.clip(np.argmax(raw), half, len(raw) - half - 1))
    axes[0].plot(energy[center-half:center+half+1], raw[center-half:center+half+1],
                 "o-", color="black", alpha=0.5, label="Raw")
    for settings in config:
        filtered, _, _ = denoise_spectrum_savgol(energy, raw, settings)
        window = min(settings["window_length"], len(raw) if len(raw) % 2 else len(raw) - 1)
        radius = window // 2
        section = slice(center - radius, center + radius + 1)
        offsets = energy[section] - energy[center]
        coefficients = np.polynomial.polynomial.polyfit(offsets, raw[section],
                                                        settings["polynomial_order"])
        label = f"window={window}, order={settings['polynomial_order']}"
        line, = axes[0].plot(energy[section], np.polynomial.polynomial.polyval(offsets, coefficients),
                             label=label)
        axes[0].scatter(energy[center], filtered[center], color=line.get_color(), zorder=4)
        axes[1].plot(energy, filtered, color=line.get_color(), label=label)
    axes[0].axvline(energy[center], color="0.5", linestyle=":")
    axes[0].set(title="Local polynomial fit (center marked)", xlabel="Binding energy (eV)",
                ylabel="Intensity")
    axes[1].set(title="Smoothed spectrum", xlabel="Binding energy (eV)", ylabel="Intensity")
    for axis in axes:
        axis.legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _stationary_details(raw: np.ndarray, settings: dict) -> tuple:
    """Calculate the actual finest stationary-wavelet detail and threshold."""
    level = settings["decomposition_level"]
    divisor = 2 ** level
    padding = (divisor - len(raw) % divisor) % divisor
    if len(raw) + padding < divisor * 2:
        padding += divisor
    left = padding // 2
    padded = np.pad(raw, (left, padding - left), mode="symmetric")
    details = pywt.swt(padded, settings["wavelet_name"], level=level)[-1][1]
    sigma = _estimate_mad_noise_sigma(details)
    threshold = _compute_level_threshold(details, sigma, len(raw), settings["threshold_rule"],
                                         settings["threshold_scale"])
    reduced = pywt.threshold(details, threshold, mode=settings["threshold_mode"])
    return details[left:left + len(raw)], reduced[left:left + len(raw)], threshold


def _plot_wavelet(energy: np.ndarray, raw: np.ndarray, config: list, path: Path) -> None:
    """Show finest detail thresholding for each stationary wavelet setting."""
    figure = plt.figure(figsize=(14, max(5, 2.7 * len(config))), constrained_layout=True)
    grid = figure.add_gridspec(len(config), 2)
    spectrum_axis = figure.add_subplot(grid[:, 1])
    spectrum_axis.plot(energy, raw, color="black", alpha=0.5, label="Raw")
    for index, settings in enumerate(config):
        detail, reduced, threshold = _stationary_details(raw, settings)
        filtered, _, _ = denoise_spectrum_wavelet(energy, raw, settings)
        label = (f"{settings['wavelet_name']}, level={settings['decomposition_level']}, "
                 f"{settings['threshold_rule']}, scale={settings['threshold_scale']:g}, "
                 f"{settings['threshold_mode']}")
        line, = spectrum_axis.plot(energy, filtered, label=label)
        axis = figure.add_subplot(grid[index, 0])
        axis.plot(energy, detail, color="0.65", label="Before")
        axis.plot(energy, reduced, color=line.get_color(), label="After")
        axis.axhline(threshold, color="black", linestyle=":", label="Threshold")
        axis.axhline(-threshold, color="black", linestyle=":")
        axis.set(title=label + " — finest detail", ylabel="Coefficient")
        if index == len(config) - 1:
            axis.set_xlabel("Binding energy (eV)")
        axis.legend(loc="upper right")
    spectrum_axis.set(title="Smoothed spectrum", xlabel="Binding energy (eV)", ylabel="Intensity")
    spectrum_axis.legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_denoise_comparison(data: tuple, config: dict) -> None:
    """Save separate figures comparing three denoising methods."""
    energy, raw = _select_spectrum(data, config["spectrum"])
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    plot_fourier_comparison(data, {"spectrum": config["spectrum"],
                                  "filters": config["fourier"],
                                  "output_path": str(output / "fourier_comparison.png")})
    _plot_savgol(energy, raw, config["savgol"], output / "savgol_comparison.png")
    _plot_wavelet(energy, raw, config["wavelet"], output / "wavelet_comparison.png")
