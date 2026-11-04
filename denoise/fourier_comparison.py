"""Standalone comparison of Fourier denoising settings."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct
from scipy.signal import butter, freqz

from denoise.fourier import denoise_spectrum_fourier


def plot_fourier_comparison(data: tuple, config: dict) -> None:
    """Save a comparison of filter responses and smoothed spectra."""
    intensity, energy, metadata = data
    selector = config["spectrum"]
    matches = metadata
    for key in ("measurement_id", "die", "region"):
        matches = matches[matches[key] == selector[key]]
    if len(matches) != 1:
        raise ValueError(
            f"Fourier plot spectrum {selector} matched {len(matches)} rows; "
            "expected exactly one."
        )
    index = matches.iloc[0]["spectrum_index"]
    num_points = matches.iloc[0]["n_points"]
    raw = intensity[index, :num_points]
    energies = energy[index, :num_points]
    frequencies = np.arange(len(raw)) / max(len(raw) - 1, 1)
    coefficients = np.abs(dct(raw, type=2, norm="ortho"))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    coefficient_axis = axes[0].twinx()
    coefficient_axis.plot(frequencies, coefficients, color="0.7", alpha=0.7,
                          label="DCT magnitude")
    axes[1].plot(energies, raw, color="black", alpha=0.6, label="Raw")

    # Match each response with its independently filtered spectrum.
    for settings in config["filters"]:
        cutoff = settings["cutoff_ratio"]
        order = settings["filter_order"]
        kind = settings["filter_type"]
        if kind == "dct":
            response = 1 / (1 + (frequencies / cutoff) ** (2 * order))
        else:
            numerator, denominator = butter(order, cutoff)
            _, transfer = freqz(numerator, denominator, worN=frequencies * np.pi)
            response = np.abs(transfer) ** 2
        label = f"{kind}: cutoff={cutoff:g}, order={order}"
        line, = axes[0].plot(frequencies, response, label=label)
        axes[0].axvline(cutoff, color=line.get_color(), linestyle=":", alpha=0.6)
        filtered, _, _ = denoise_spectrum_fourier(energies, raw, settings)
        axes[1].plot(energies, filtered, color=line.get_color(), label=label)
    axes[0].set(title="Gain (Butterworth ignores edge effects)",
                xlabel="Frequency / Nyquist", ylabel="Filter gain", ylim=(0, 1.05))
    coefficient_axis.set_ylabel("DCT coefficient magnitude")
    axes[1].set(xlabel="Binding energy (eV)", ylabel="Intensity")
    axes[0].legend()
    coefficient_axis.legend(loc="lower right")
    axes[1].legend()
    output = Path(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
