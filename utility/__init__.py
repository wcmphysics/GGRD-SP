"""Utility module for GGRD-SP spectral processing and synthetic data.
"""

from utility.config_loader import (
    extract_element_vocabulary,
    load_config,
    load_material_elements,
    load_yaml,
    resolve_pipeline_config,
)
from utility.config_schema import PipelineConfig
from utility.evaluation import (
    calculate_prediction_metrics,
    format_side_by_side_metrics,
    match_prediction_source_target_indices,
)
from utility.flux_and_intensity import (
    calculate_flux_and_intensity,
    calculate_standardized_time,
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
from utility.preprocessing import preprocess
from utility.reporting import format_split_comparison_table
from utility.shirley import (
    calculate_shirley_background,
    determine_shirley_endpoints,
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
    format_side_by_side_atomic_percentages,
    get_default_quantification_config,
    load_rsf_table,
)

__all__ = [
    # Config loading
    "PipelineConfig",
    "load_yaml",
    "load_material_elements",
    "extract_element_vocabulary",
    "load_config",
    "resolve_pipeline_config",
    # Data generation
    "generate_pseudo_measurements",
    "generate_tool_timeline",
    "get_default_measurement_config",
    # Flux & Intensity
    "calculate_flux_and_intensity",
    "calculate_standardized_time",
    "standardize_metadata_features",
    "integrate_spectral_flux",
    "trapezoid_integral",
    "compute_die_total_flux",
    # Normalization & Pairing
    "normalize_spectral_data",
    "pair_source_target_spectra",
    "get_default_pairing_config",
    "split_measurement_sessions",
    "preprocess",
    "format_split_comparison_table",
    # Quantification
    "calculate_shirley_background",
    "determine_shirley_endpoints",
    "calculate_atomic_percentages",
    "calculate_atomic_percentage_split_statistics",
    "format_side_by_side_atomic_percentages",
    "get_default_quantification_config",
    "load_rsf_table",
    "DEFAULT_SCOFIELD_RSF",
    # Patching
    "calculate_window_points",
    "extract_sliding_windows",
    "reconstruct_from_patches",
    # Metrics
    "calculate_prediction_metrics",
    "format_side_by_side_metrics",
    "match_prediction_source_target_indices",
]
