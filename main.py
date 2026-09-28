"""Main entry point for GGRD-SP project."""

from __future__ import annotations

import matplotlib.pyplot as plt

from models import format_predicted_measurement_id, run_baseline_pipeline
from utility import (
    calculate_atomic_percentages,
    calculate_prediction_metrics,
    generate_pseudo_measurements,
    pair_source_target_spectra,
    plot_pairing_timeline,
    plot_prediction_comparison,
    plot_regional_spectra,
    plot_shirley_background,
    plot_tool_comparison,
    plot_training_history,
)


def main() -> None:
    """Execute pseudo-measurement generation, pairing, quantification, and visualization."""
    # =========================================================================
    # GLOBAL SETUP & CONFIGURATION (Easily tune all parameters here)
    # =========================================================================
    source_tool = "J4"
    target_tool = "H1"
    example_region = "Ti2p"
    sample_meas_id = "M_J4_000"  # session for at% calculation and comparison

    # Part 1: Pseudo-measurement generation configuration
    pseudo_config = {
        "material": "NMG",
        "regions": ["Al2p", "Ti2p", "O1s", "C1s", "Cl2p"],
        "n_points": 100,
        "measurements_per_tool": {source_tool: 15, target_tool: 20},
        "interval_hours_range": (4.0, 24.0),
        "tool_offsets": {
            source_tool: {"shift_ev": 0.0, "scale": 1.0},
            target_tool: {"shift_ev": 0.2, "scale": 1.1},
        },
        "die_variation_std": 0.03,
        "noise_relative_std": 0.015,
        "seed": None,
    }

    # Part 2: Source-to-target 1-to-1 pairing configuration
    pairing_config = {
        "source_tool": source_tool,
        "target_tool": target_tool,
        "time_threshold_hours": 8.0,
        "method": "optimal",
        "material": None,
    }

    # Part 3: Baseline Neural Network & Ax Bayesian Optimization configuration
    baseline_config = {
        "regions": ["Al2p", "Ti2p", "O1s", "C1s", "Cl2p"],
        "source_tool": source_tool,
        "target_tool": target_tool,
        "val_ratio": 0.2,
        "seed": None,
        "use_bayesian_opt": False,  # Set to True to enable Ax Bayesian hyperparameter optimization
        "bayesian_opt_config": {
            "num_trials": 5,
            "epochs_per_trial": 15,
            "kernel_sizes": [3, 5, 13],
            "hidden_channels": [16, 32, 64],
            "lr_bounds": (1e-4, 1e-2),
            "l2_bounds": (1e-6, 1e-2),
            "batch_size": 16,
            "verbose": True,
        },
        "train_config": {
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "hidden_channels": 32,
            "kernel_size": 5,
            "l2_weight": 1e-4,
            "early_stopping_patience": 10,
            "verbose": False,
        },
        "predict_source": True,
    }

    # =========================================================================
    # PART 1: GENERATE PSEUDO-MEASUREMENTS
    # =========================================================================
    print("=== Part 1: Generating Pseudo-Measurements ===")
    ary_intensity, ary_energy, meta_df = generate_pseudo_measurements(pseudo_config)

    print(f"Intensity array shape: {ary_intensity.shape}")
    print(f"Energy array shape:    {ary_energy.shape}")
    print(f"Total spectra:         {len(meta_df)}")
    print(
        f"Measurements per tool: {meta_df.groupby('tool')['measurement_id'].nunique().to_dict()}"
    )

    # =========================================================================
    # PART 2: PAIR SOURCE AND TARGET SPECTRA
    # =========================================================================
    print("\n=== Part 2: Pairing Source and Target Spectra ===")
    meta_df = pair_source_target_spectra(meta_df, pairing_config)

    paired_src = meta_df[
        (meta_df["tool"] == source_tool) & meta_df["measurement_id_target"].notna()
    ]
    total_src_sessions = meta_df[meta_df["tool"] == source_tool]["measurement_id"].nunique()
    paired_src_sessions = paired_src["measurement_id"].nunique()

    print(
        f"Paired measurement sessions / All measurement sessions: {paired_src_sessions}/{total_src_sessions}"
    )
    print("\nUpdated metadata summary with pairing columns:")
    print(meta_df.info())
    print("\nSample paired source rows:")
    cols_to_show = [
        "spectrum_index",
        "tool",
        "measurement_id",
        "die",
        "region",
        "tool_target",
        "measurement_id_target",
        "spectrum_index_target",
        "time_diff_hours",
    ]
    print(paired_src[cols_to_show].head(5))

    # =========================================================================
    # PART 3: BASELINE NEURAL NETWORK TRAINING & PREDICTION
    # =========================================================================
    print("\n=== Part 3: Baseline Neural Network Training & Prediction ===")
    baseline_results = run_baseline_pipeline(
        meta_df,
        ary_intensity,
        ary_energy,
        config=baseline_config,
    )

    val_evaluation = baseline_results["evaluation"]
    print("\nValidation Normalized MSE Loss per region:")
    for reg, loss_val in val_evaluation.items():
        print(f"  {reg:6s}: {loss_val:.6f}")

    ary_intensity_predicted, ary_energy_predicted, meta_df_predicted = baseline_results["predictions"]
    print(f"\nPredicted intensity array shape: {ary_intensity_predicted.shape}")
    print(f"Total predicted spectra:         {len(meta_df_predicted)}")

    # =========================================================================
    # PART 4: MODEL PERFORMANCE OBSERVATION & METRICS
    # =========================================================================
    print("\n=== Part 4: Model Performance Observation & Metrics ===")
    data_orig = (ary_intensity, ary_energy, meta_df)
    data_pred = (ary_intensity_predicted, ary_energy_predicted, meta_df_predicted)

    df_metrics_samples, df_metrics_summary = calculate_prediction_metrics(
        data_orig,
        data_pred,
    )
    print("\nQuantitative prediction metrics summary by region (Ground Truth Target vs. Predicted Target):")
    print(df_metrics_summary.to_string(index=False))

    # =========================================================================
    # PART 5: CALCULATE ATOMIC PERCENTAGES (Shirley Integration)
    # =========================================================================
    print("\n=== Part 5: Calculating Atomic Percentage (Shirley Integration) ===")

    # 1. Measured source tool session
    df_per_die_src, df_summary_src = calculate_atomic_percentages(
        ary_energy,
        ary_intensity,
        meta_df,
        config={"measurement_id": sample_meas_id},
    )

    # 2. Predicted target tool session
    pred_sample_meas_id = format_predicted_measurement_id(sample_meas_id, source_tool, target_tool)
    df_per_die_pred, df_summary_pred = calculate_atomic_percentages(
        ary_energy_predicted,
        ary_intensity_predicted,
        meta_df_predicted,
        config={"measurement_id": pred_sample_meas_id},
    )

    print(f"\n[Source Measured] Atomic percentage per die for session '{sample_meas_id}' (first 3 dies):")
    print(df_per_die_src.head(3).to_string(index=False))
    print(f"\n[Predicted Target] Atomic percentage per die for session '{pred_sample_meas_id}' (first 3 dies):")
    print(df_per_die_pred.head(3).to_string(index=False))

    print(f"\nSummary across all 9 dies for Source '{sample_meas_id}':")
    print(df_summary_src.to_string(index=False))
    print(f"\nSummary across all 9 dies for Predicted '{pred_sample_meas_id}':")
    print(df_summary_pred.to_string(index=False))

    # =========================================================================
    # DEMONSTRATION PLOTS
    # =========================================================================
    print("\nGenerating demonstration plots...")

    # 1. Direct tool comparison between tools for a region (Die 0)
    plot_tool_comparison(
        ary_energy,
        ary_intensity,
        meta_df,
        compare_config={"tools": (source_tool, target_tool), "region": example_region, "die": 0},
    )

    # 2. 1-to-1 Measurement pairing timeline
    plot_pairing_timeline(meta_df, plot_config={"source_tool": source_tool, "target_tool": target_tool})

    # 3. Shirley background subtraction in multi-region grid mode (all 5 regions for Die 0)
    plot_shirley_background(
        ary_energy,
        ary_intensity,
        meta_df,
        plot_config={"measurement_id": sample_meas_id, "die": 0},
    )

    # 4. Training loss history curves across all regions (Part 4)
    plot_training_history(
        baseline_results["histories"],
        plot_config={"title": "Part 4: Baseline Neural Network Training & Validation Loss"},
    )

    # 5. Spectral transfer comparison: Source vs. True Target vs. Predicted Target with Residual (Part 4)
    plot_prediction_comparison(
        data_orig,
        data_pred,
        config={
            "measurement_id": sample_meas_id,
            "region": example_region,
            "die": 0,
            "show_residual": True,
        },
    )

    print("Showing plots (close plot windows to finish execution)...")
    plt.show()


if __name__ == "__main__":
    main()