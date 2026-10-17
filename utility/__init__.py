"""Utility module for GGRD-SP spectral processing and synthetic data.
"""

from utility.evaluation import calculate_prediction_metrics, format_side_by_side_metrics
from utility.flux import (
    calculate_flux,
    calculate_normalized_time,
    compute_die_total_flux,
    integrate_spectral_flux,
    trapezoid_integral,
)
from utility.normalization import normalize_spectral_data
from utility.pairing import (
    get_default_pairing_config,
    pair_source_target_spectra,
)
from utility.splitting import split_measurement_sessions
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
__all__ = [
    "generate_pseudo_measurements",
    "get_default_measurement_config",
    "calculate_flux",
    "calculate_normalized_time",
    "integrate_spectral_flux",
    "trapezoid_integral",
    "normalize_spectral_data",
    "pair_source_target_spectra",
    "get_default_pairing_config",
    "split_measurement_sessions",
    "calculate_shirley_background",
    "compute_die_total_flux",
    "determine_shirley_endpoints",
    "calculate_atomic_percentages",
    "calculate_atomic_percentage_split_statistics",
    "format_side_by_side_atomic_percentages",
    "get_default_quantification_config",
    "DEFAULT_SCOFIELD_RSF",
    "calculate_window_points",
    "extract_sliding_windows",
    "reconstruct_from_patches",
    "calculate_prediction_metrics",
    "format_side_by_side_metrics",
]
