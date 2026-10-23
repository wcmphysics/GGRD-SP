"""Visualization dispatcher module for GGRD-SP.

Provides a unified dispatch function to selectively render diagnostic and
performance plots according to boolean configuration toggles.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from visualization.plot_noise_analysis import (
    plot_gp_hyperparameter_convergence,
    plot_spectral_noise_analysis,
)
from visualization.visualization import (
    plot_atomic_percentage_distributions,
    plot_atomic_percentage_mae,
    plot_flux_and_averaged_intensity_vs_time,
    plot_gp_die_flux_timeline,
    plot_max_intensity_vs_time,
    plot_normalized_max_intensity_vs_time,
    plot_pairing_timeline,
    plot_prediction_comparison,
    plot_regional_spectrum_distribution,
    plot_shirley_background,
    plot_sliding_window_slices,
    plot_tool_comparison,
    plot_training_history,
)

DEFAULT_PLOT_TOGGLES: dict[str, bool] = {
    "gp_die_flux_timeline": True,
    "pairing_timeline": True,
    "flux_and_averaged_intensity": False,
    "max_intensity": False,
    "normalized_max_intensity": False,
    "tool_comparison": False,
    "shirley_background": False,
    "spectral_noise_analysis": True,
    "gp_hyperparameter_convergence": True,
    "training_history": True,
    "prediction_comparison": True,
    "regional_spectrum_distribution": True,
    "atomic_percentage_distributions": True,
    "atomic_percentage_mae": True,
    "sliding_window_slices": True,
}


def dispatch_visualizations(
    pipeline_data: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Execute enabled diagnostic plots based on configuration flags.

    Parameters
    ----------
    pipeline_data : dict[str, Any]
        Packaged pipeline artifacts and data containing:
        - 'df_meta': metadata DataFrame.
        - 'ary_intensity': 2D measured intensity array.
        - 'ary_energy': 2D binding energy array.
        - 'data_original': tuple of (intensity, energy, df_meta).
        - 'data_predicted': tuple of (pred_intensity, energy, pred_meta).
        - 'training_results': model training histories dictionary.
        - 'df_atomic_percentage_samples': atomic percentage DataFrame.
        - 'gp_bundle': GP flux model bundle (optional).
        - 'gp_flux_config': GP flux configuration (optional).
        - 'noise_analysis': tuple of (signal, noise) arrays (optional).
    config : dict[str, Any] | None, optional
        Visualization configuration dictionary containing:
        - 'show_plots' (bool): master switch to display plots (default True).
        - 'plots' (dict[str, bool]): individual plot enablement toggles.
        - 'source_tool' (str): source tool identifier.
        - 'target_tool' (str): target tool identifier.
        - 'example_region' (str): region for diagnostic single-region plots.
        - 'sample_measurement_id' (str): measurement ID for single-sample plots.
        - 'die' (int): die index (default 0).
        - 'session_splits' (dict | list): split session mapping.
        - 'use_sliding_window' (bool): whether sliding window is active.
        - 'noise_plot_config' (dict): config for spectral noise plot.

    Returns
    -------
    list[str]
        List of plot identifier strings that were successfully rendered.
    """
    cfg = config or {}
    show_plots = bool(cfg.get("show_plots", True))
    if not show_plots:
        return []

    plot_toggles = {**DEFAULT_PLOT_TOGGLES, **(cfg.get("plots") or {})}
    rendered: list[str] = []

    source_tool = str(cfg.get("source_tool", "J4"))
    target_tool = str(cfg.get("target_tool", "J5"))
    example_region = str(cfg.get("example_region", "Al2p"))
    sample_measurement_id = cfg.get("sample_measurement_id")
    die = int(cfg.get("die", 0))
    session_splits = cfg.get("session_splits")
    use_sw = bool(cfg.get("use_sliding_window", False))

    df_meta = pipeline_data.get("df_meta")
    ary_intensity = pipeline_data.get("ary_intensity")
    ary_energy = pipeline_data.get("ary_energy")
    data_orig = pipeline_data.get("data_original")
    data_pred = pipeline_data.get("data_predicted")
    training_results = pipeline_data.get("training_results") or {}
    df_atomic = pipeline_data.get("df_atomic_percentage_samples")

    # 1. GP Die Flux Timeline
    if plot_toggles.get("gp_die_flux_timeline"):
        gp_bundle = pipeline_data.get("gp_bundle")
        gp_cfg = pipeline_data.get("gp_flux_config") or cfg.get(
            "gp_flux_config"
        )
        if gp_bundle is not None and gp_cfg is not None:
            plot_gp_die_flux_timeline(gp_bundle, gp_cfg)
            rendered.append("gp_die_flux_timeline")

    # 2. Pairing Timeline
    if plot_toggles.get("pairing_timeline") and df_meta is not None:
        plot_pairing_timeline(
            df_meta,
            plot_config={
                "source_tool": source_tool,
                "target_tool": target_tool,
            },
        )
        rendered.append("pairing_timeline")

    # 3. Flux and Averaged Intensity vs Time
    if (
        plot_toggles.get("flux_and_averaged_intensity")
        and ary_intensity is not None
        and ary_energy is not None
        and df_meta is not None
    ):
        plot_flux_and_averaged_intensity_vs_time(
            ary_intensity,
            ary_energy,
            df_meta,
            plot_config={
                "source_tool": source_tool,
                "target_tool": target_tool,
            },
        )
        rendered.append("flux_and_averaged_intensity")

    # 4. Max Intensity vs Time
    if (
        plot_toggles.get("max_intensity")
        and ary_intensity is not None
        and df_meta is not None
    ):
        plot_max_intensity_vs_time(
            ary_intensity,
            df_meta,
            region=example_region,
            die=die,
            plot_config={"hue": "tool"},
        )
        rendered.append("max_intensity")

    # 5. Normalized Max Intensity vs Time
    if (
        plot_toggles.get("normalized_max_intensity")
        and ary_intensity is not None
        and df_meta is not None
    ):
        plot_normalized_max_intensity_vs_time(
            ary_intensity,
            df_meta,
            ary_energy=ary_energy,
            region=example_region,
            die=die,
            plot_config={
                "hue": "tool",
                "normalization_mode": "die_total_flux",
            },
        )
        rendered.append("normalized_max_intensity")

    # 6. Tool Comparison
    if (
        plot_toggles.get("tool_comparison")
        and ary_intensity is not None
        and ary_energy is not None
        and df_meta is not None
    ):
        plot_tool_comparison(
            ary_energy,
            ary_intensity,
            df_meta,
            compare_config={
                "tools": (source_tool, target_tool),
                "region": example_region,
                "die": die,
            },
        )
        rendered.append("tool_comparison")

    # 7. Shirley Background
    if (
        plot_toggles.get("shirley_background")
        and ary_intensity is not None
        and ary_energy is not None
        and df_meta is not None
        and sample_measurement_id is not None
    ):
        plot_shirley_background(
            ary_energy,
            ary_intensity,
            df_meta,
            plot_config={
                "measurement_id": sample_measurement_id,
                "die": die,
            },
        )
        rendered.append("shirley_background")

    # 8. Spectral Noise Analysis
    if plot_toggles.get("spectral_noise_analysis"):
        noise_tuple = pipeline_data.get("noise_analysis")
        if (
            noise_tuple is not None
            and ary_intensity is not None
            and ary_energy is not None
            and df_meta is not None
        ):
            sig, noise = noise_tuple
            noise_plot_cfg = cfg.get("noise_plot_config") or {}
            plot_spectral_noise_analysis(
                (ary_intensity, ary_energy, df_meta, sig, noise),
                noise_plot_cfg,
            )
            rendered.append("spectral_noise_analysis")

    # 9. GP Hyperparameter Convergence
    if plot_toggles.get("gp_hyperparameter_convergence") and data_orig:
        plot_gp_hyperparameter_convergence(
            data_orig,
            plot_config={
                "measurement_id": sample_measurement_id,
                "die": die,
                "save_path": "gp_hyperparameter_convergence.png",
            },
        )
        rendered.append("gp_hyperparameter_convergence")

    # 10. Training History
    if plot_toggles.get("training_history") and "histories" in training_results:
        plot_training_history(training_results["histories"])
        rendered.append("training_history")

    # 11. Prediction Comparison
    if (
        plot_toggles.get("prediction_comparison")
        and data_orig is not None
        and data_pred is not None
    ):
        plot_prediction_comparison(
            data_orig,
            data_pred,
            config={"session_splits": session_splits, "die": die},
        )
        rendered.append("prediction_comparison")

    # 12. Regional Spectrum Distribution
    if (
        plot_toggles.get("regional_spectrum_distribution")
        and data_orig is not None
        and data_pred is not None
    ):
        plot_regional_spectrum_distribution(
            data_orig,
            data_pred,
            config={
                "split": "test",
                "save_path": "regional_spectrum_distribution.png",
            },
        )
        rendered.append("regional_spectrum_distribution")

    # 13. Atomic Percentage Distributions
    if plot_toggles.get("atomic_percentage_distributions") and df_atomic is not None:
        plot_atomic_percentage_distributions(df_atomic)
        rendered.append("atomic_percentage_distributions")

    # 14. Atomic Percentage MAE
    if plot_toggles.get("atomic_percentage_mae") and df_atomic is not None:
        plot_atomic_percentage_mae(df_atomic)
        rendered.append("atomic_percentage_mae")

    # 15. Sliding Window Slices
    if (
        plot_toggles.get("sliding_window_slices")
        and use_sw
        and df_meta is not None
        and ary_intensity is not None
        and ary_energy is not None
    ):
        ex_row = df_meta[
            (df_meta["region"] == example_region)
            & (df_meta["tool"] == source_tool)
        ]
        if not ex_row.empty:
            spec_idx = int(ex_row.iloc[0]["spectrum_index"])
            plot_sliding_window_slices(
                ary_intensity[spec_idx],
                ary_energy[spec_idx],
                plot_config={"region": example_region},
            )
            rendered.append("sliding_window_slices")

    return rendered
