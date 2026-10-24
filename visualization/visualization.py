"""Visualization façade re-exporting domain plotting submodules for backward
compatibility."""

from __future__ import annotations

from visualization.common import (
    _check_meta,
    _derive_tool_die_shades,
    _finalize_plot,
    _finish_fig,
    _prepare_canvas,
    _setup_subplot_grid,
)
from visualization.plot_evaluation import (
    plot_prediction_comparison,
    plot_regional_spectrum_distribution,
)
from visualization.plot_quantification import (
    _filter_atomic_df,
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
    _calculate_staggered_levels,
    _render_intensity_time_series,
    plot_flux_and_averaged_intensity_vs_time,
    plot_gp_die_flux_timeline,
    plot_max_intensity_vs_time,
    plot_normalized_max_intensity_vs_time,
    plot_pairing_timeline,
)
from visualization.plot_training import plot_training_history

__all__ = [
    # Common
    "_prepare_canvas",
    "_setup_subplot_grid",
    "_finalize_plot",
    "_finish_fig",
    "_check_meta",
    "_derive_tool_die_shades",
    # Spectral
    "plot_regional_spectra",
    "plot_tool_comparison",
    "plot_sliding_window_slices",
    # Timeline
    "_calculate_staggered_levels",
    "_render_intensity_time_series",
    "plot_pairing_timeline",
    "plot_max_intensity_vs_time",
    "plot_normalized_max_intensity_vs_time",
    "plot_flux_and_averaged_intensity_vs_time",
    "plot_gp_die_flux_timeline",
    # Training
    "plot_training_history",
    # Evaluation
    "plot_prediction_comparison",
    "plot_regional_spectrum_distribution",
    # Quantification
    "plot_shirley_background",
    "_filter_atomic_df",
    "plot_atomic_percentage_distributions",
    "plot_atomic_percentage_mae",
]
