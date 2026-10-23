"""Utility module for GGRD-SP spectral processing and synthetic data.
"""

from utility.config_loader import (
    extract_element_vocabulary,
    load_default_config,
    load_material_elements,
    load_yaml,
)
from utility.evaluation import (
    calculate_prediction_metrics,
    format_side_by_side_metrics,
)
from utility.flux_and_intensity import (
    calculate_flux_and_intensity,
    calculate_normalized_time,
    compute_die_total_flux,
    integrate_spectral_flux,
    standardize_metadata_features,
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
    generate_tool_timeline,
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
    # Config loading
    "load_yaml",
    "load_material_elements",
    "extract_element_vocabulary",
    "load_default_config",
    # Data generation
    "generate_pseudo_measurements",
    "generate_tool_timeline",
    "get_default_measurement_config",
    # Flux & Intensity
    "calculate_flux_and_intensity",
    "calculate_normalized_time",
    "standardize_metadata_features",
    "integrate_spectral_flux",
    "trapezoid_integral",
    "compute_die_total_flux",
    # Normalization & Pairing
    "normalize_spectral_data",
    "pair_source_target_spectra",
    "get_default_pairing_config",
    "split_measurement_sessions",
    # Quantification
    "calculate_shirley_background",
    "determine_shirley_endpoints",
    "calculate_atomic_percentages",
    "calculate_atomic_percentage_split_statistics",
    "format_side_by_side_atomic_percentages",
    "get_default_quantification_config",
    "DEFAULT_SCOFIELD_RSF",
    # Patching
    "calculate_window_points",
    "extract_sliding_windows",
    "reconstruct_from_patches",
    # Metrics
    "calculate_prediction_metrics",
    "format_side_by_side_metrics",
]
