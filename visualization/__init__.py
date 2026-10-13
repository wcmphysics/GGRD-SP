"""Visualization module for GGRD-SP spectral analysis and model diagnostics."""

from __future__ import annotations

from visualization.visualization import (
    plot_atomic_percentage_distributions,
    plot_atomic_percentage_mae,
    plot_flux_and_averaged_intensity_vs_time,
    plot_max_intensity_vs_time,
    plot_normalized_max_intensity_vs_time,
    plot_pairing_timeline,
    plot_prediction_comparison,
    plot_regional_spectra,
    plot_shirley_background,
    plot_sliding_window_slices,
    plot_tool_comparison,
    plot_training_history,
)

__all__ = [
    "plot_atomic_percentage_distributions",
    "plot_atomic_percentage_mae",
    "plot_flux_and_averaged_intensity_vs_time",
    "plot_max_intensity_vs_time",
    "plot_normalized_max_intensity_vs_time",
    "plot_pairing_timeline",
    "plot_prediction_comparison",
    "plot_regional_spectra",
    "plot_shirley_background",
    "plot_sliding_window_slices",
    "plot_tool_comparison",
    "plot_training_history",
]
