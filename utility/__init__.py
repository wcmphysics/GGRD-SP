"""Utility module for GGRD-SP spectral processing and synthetic data generation."""

from utility.evaluation import calculate_prediction_metrics, format_side_by_side_metrics
from utility.pairing import (
    get_default_pairing_config,
    pair_source_target_spectra,
)
from utility.patching import (
    calculate_window_points,
    extract_sliding_windows,
    reconstruct_from_patches,
)
from utility.pseudo_measurement import (
    generate_pseudo_measurements,
    get_default_measurement_config,
)
from utility.quantification import (
    DEFAULT_SCOFIELD_RSF,
    calculate_atomic_percentage_split_statistics,
    calculate_atomic_percentages,
    calculate_shirley_background,
    determine_shirley_endpoints,
    format_side_by_side_atomic_percentages,
    get_default_quantification_config,
)
from utility.visualization import (
    plot_max_intensity_vs_time,
    plot_normalized_max_intensity_vs_time,
    plot_pairing_timeline,
    plot_prediction_comparison,
    plot_regional_spectra,
    plot_shirley_background,
    plot_sliding_window_slices,
    plot_tool_comparison,
    plot_training_history,
    plot_atomic_percentage_distributions,
)

__all__ = [
    "generate_pseudo_measurements",
    "get_default_measurement_config",
    "pair_source_target_spectra",
    "get_default_pairing_config",
    "calculate_shirley_background",
    "determine_shirley_endpoints",
    "calculate_atomic_percentages",
    "calculate_atomic_percentage_split_statistics",
    "format_side_by_side_atomic_percentages",
    "get_default_quantification_config",
    "DEFAULT_SCOFIELD_RSF",
    "calculate_window_points",
    "extract_sliding_windows",
    "reconstruct_from_patches",
    "plot_regional_spectra",
    "plot_tool_comparison",
    "plot_pairing_timeline",
    "plot_max_intensity_vs_time",
    "plot_normalized_max_intensity_vs_time",
    "plot_shirley_background",
    "plot_training_history",
    "plot_prediction_comparison",
    "calculate_prediction_metrics",
    "format_side_by_side_metrics",
    "plot_sliding_window_slices",
    "plot_atomic_percentage_distributions",
]
