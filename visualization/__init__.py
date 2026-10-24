"""Visualization module for GGRD-SP spectral analysis and model diagnostics."""

from __future__ import annotations

from visualization.dispatcher import (
    DEFAULT_PLOT_TOGGLES,
    dispatch_visualizations,
)
from visualization.plot_evaluation import (
    plot_prediction_comparison,
    plot_regional_spectrum_distribution,
)
from visualization.plot_noise_analysis import (
    plot_gp_hyperparameter_convergence,
    plot_spectral_noise_analysis,
)
from visualization.plot_quantification import (
    plot_atomic_percentage_distributions,
    plot_atomic_percentage_mae,
    plot_shirley_background,
)
from visualization.plot_spectral import (
    plot_regional_spectra,
    plot_sliding_window_slices,
    plot_tool_comparison,
)
from visualization.plot_timeline import (
    plot_flux_and_averaged_intensity_vs_time,
    plot_gp_die_flux_timeline,
    plot_max_intensity_vs_time,
    plot_normalized_max_intensity_vs_time,
    plot_pairing_timeline,
)
from visualization.plot_training import plot_training_history

__all__ = [
    "DEFAULT_PLOT_TOGGLES",
    "dispatch_visualizations",
    "plot_gp_hyperparameter_convergence",
    "plot_spectral_noise_analysis",
    "plot_atomic_percentage_distributions",
    "plot_atomic_percentage_mae",
    "plot_flux_and_averaged_intensity_vs_time",
    "plot_gp_die_flux_timeline",
    "plot_max_intensity_vs_time",
    "plot_normalized_max_intensity_vs_time",
    "plot_pairing_timeline",
    "plot_prediction_comparison",
    "plot_regional_spectra",
    "plot_regional_spectrum_distribution",
    "plot_shirley_background",
    "plot_sliding_window_slices",
    "plot_tool_comparison",
    "plot_training_history",
]
