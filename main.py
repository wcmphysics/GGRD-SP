"""Main entry point for GGRD-SP project."""

from __future__ import annotations

import matplotlib.pyplot as plt

from models import (
    format_predicted_measurement_id,
    run_baseline_pipeline,
    run_sliding_window_pipeline,
    run_unet_pipeline,
)
from utility import (
    calculate_atomic_percentages,
    calculate_prediction_metrics,
    format_side_by_side_metrics,
    generate_pseudo_measurements,
    pair_source_target_spectra,
    plot_pairing_timeline,
    plot_prediction_comparison,
    plot_regional_spectra,
    plot_shirley_background,
    plot_sliding_window_slices,
    plot_tool_comparison,
    plot_training_history,
)


def main() -> None:
    """Execute pseudo-measurement generation, pairing, quantification, and visualization."""
    # =========================================================================
    # GLOBAL SETUP & CONFIGURATION (Easily tune all parameters here)
    # =========================================================================
    # Choose which neural network model architecture to train and evaluate:
    # "baseline": Full regional spectrum ResNet
    # "sliding_window": Local sequence patch-to-patch ResNet with overlap reconstruction
    # "unet": Multi-scale 1D U-Net (supports both full spectrum and sliding window modes)
    model_type = "baseline"  # Options: "baseline", "sliding_window", or "unet"

    source_tool = "J4"
    target_tool = "J5"
    example_region = "Ti2p"
    sample_meas_id = "NMG_M_J4_00000"  # session for at% calculation and comparison
    show_plots = True  # Set to False to skip GUI plot display (useful in non-interactive/headless runs)

    # Part 1: Pseudo-measurement generation configuration
    pseudo_config = {
        "material": "NMG",
        "regions": ["Al2p", "Ti2p", "O1s", "C1s", "Cl2p"],
        "n_points": 100,
        "measurements_per_tool": {source_tool: 25, target_tool: 30},
        "measurements_per_t7_code": 10,
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
        "train_ratio": 0.5,
        "val_ratio": 0.2,
        "test_ratio": 0.3,
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

    # Part 3: Sliding Window Neural Network & Ax Bayesian Optimization configuration
    sliding_window_config = {
        "regions": ["Al2p", "Ti2p", "O1s", "C1s", "Cl2p"],
        "source_tool": source_tool,
        "target_tool": target_tool,
        "train_ratio": 0.5,
        "val_ratio": 0.2,
        "test_ratio": 0.3,
        "seed": None,
        "window_size_ev": 2.0,  # Local window length in eV
        "sliding_stride_ev": 1.0,  # Sliding stride step in eV
        "use_bayesian_opt": False,  # Set to True to enable Ax Bayesian hyperparameter optimization
        "bayesian_opt_config": {
            "num_trials": 5,
            "epochs_per_trial": 15,
            "window_size_choices": [11, 15, 21],
            "kernel_sizes": [3, 5],
            "hidden_channels": [16, 32],
            "lr_bounds": (1e-4, 1e-2),
            "l2_bounds": (1e-6, 1e-2),
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

    # Part 3: 1D U-Net Neural Network & Ax Bayesian Optimization configuration
    unet_config = {
        "use_sliding_window": False,  # True for patch-to-patch mode, False for full regional spectrum mode
        "regions": ["Al2p", "Ti2p", "O1s", "C1s", "Cl2p"],
        "source_tool": source_tool,
        "target_tool": target_tool,
        "train_ratio": 0.5,
        "val_ratio": 0.2,
        "test_ratio": 0.3,
        "seed": None,
        "window_size_ev": 2.0,  # used if use_sliding_window=True
        "sliding_stride_ev": 1.0,  # used if use_sliding_window=True
        "use_bayesian_opt": False,
        "bayesian_opt_config": {
            "num_trials": 5,
            "epochs_per_trial": 15,
            "base_channels": [16, 32],
            "depths": [2, 3],
            "kernel_sizes": [3, 5],
            "lr_bounds": (1e-4, 1e-2),
            "l2_bounds": (1e-6, 1e-2),
            "verbose": True,
        },
        "train_config": {
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "base_channels": 16,
            "depth": 3,
            "kernel_size": 5,
            "residual": True,  # y = x + UNet(x) for stable inter-tool delta transfer
            "l2_weight": 1e-4,
            "early_stopping_patience": 10,
            "verbose": True,
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
    # PART 3: NEURAL NETWORK TRAINING & PREDICTION
    # =========================================================================
    if model_type == "sliding_window":
        print("\n=== Part 3: Sliding Window Neural Network Training & Prediction ===")
        model_results = run_sliding_window_pipeline(
            meta_df,
            ary_intensity,
            ary_energy,
            config=sliding_window_config,
        )
    elif model_type == "baseline":
        print("\n=== Part 3: Baseline Neural Network Training & Prediction ===")
        model_results = run_baseline_pipeline(
            meta_df,
            ary_intensity,
            ary_energy,
            config=baseline_config,
        )
    elif model_type == "unet":
        print("\n=== Part 3: 1D U-Net Neural Network Training & Prediction ===")
        model_results = run_unet_pipeline(
            meta_df,
            ary_intensity,
            ary_energy,
            config=unet_config,
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose 'baseline', 'sliding_window', or 'unet'."
        )

    val_evaluation = model_results["evaluation"]
    test_evaluation = model_results.get("test_evaluation", {})
    print("\nRegional Evaluation Losses:")
    print("  Region | Val Normalized MSE | Test Normalized MSE")
    print("  -------+--------------------+--------------------")
    for reg in sorted(val_evaluation.keys()):
        val_l = val_evaluation.get(reg, float("nan"))
        test_l = test_evaluation.get(reg, float("nan"))
        print(f"  {reg:6s} | {val_l:18.6f} | {test_l:19.6f}")

    ary_intensity_predicted, ary_energy_predicted, meta_df_predicted = model_results["predictions"]
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
    print("\nDetailed quantitative prediction metrics by region and split:")
    print(df_metrics_summary.to_string(index=False))

    df_side_by_side = format_side_by_side_metrics(df_metrics_samples)
    if not df_side_by_side.empty:
        print("\nSide-by-side performance comparison across splits (Train vs. Val vs. Test):")
        print(df_side_by_side.to_string(index=False))

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
    if model_type == "sliding_window":
        model_name_label = "Sliding Window ResNet"
    elif model_type == "unet":
        mode_str = "Sliding Window" if unet_config.get("use_sliding_window") else "Full Spectrum"
        model_name_label = f"1D U-Net ({mode_str})"
    else:
        model_name_label = "Baseline ResNet"

    plot_training_history(
        model_results["histories"],
        plot_config={"title": f"Part 4: {model_name_label} Neural Network Training & Validation Loss"},
    )

    # 5. Sliding window decomposition (Original vs Sliced Spectra) if sliding window chosen
    has_sw_decomp = (model_type == "sliding_window") or (
        model_type == "unet" and unet_config.get("use_sliding_window", False)
    )
    if has_sw_decomp:
        example_subset = meta_df[
            (meta_df["measurement_id"] == sample_meas_id)
            & (meta_df["region"] == example_region)
            & (meta_df["die"] == 0)
        ]
        if not example_subset.empty:
            ex_idx = int(example_subset.iloc[0]["spectrum_index"])
            active_sw_cfg = sliding_window_config if model_type == "sliding_window" else unet_config
            plot_sliding_window_slices(
                ary_intensity[ex_idx],
                ary_energy[ex_idx],
                plot_config={
                    "region": example_region,
                    "window_size_ev": active_sw_cfg.get("window_size_ev", 2.0),
                    "sliding_stride_ev": active_sw_cfg.get("sliding_stride_ev", 1.0),
                    "offset_patches": True,
                    "title": (
                        f"Sliding Window Decomposition - {example_region} "
                        f"(Session {sample_meas_id}, Die 0)"
                    ),
                },
            )

    # 6. Spectral transfer comparison: Source vs. True Target vs. Predicted Target with Residual (Part 4)
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

    if show_plots:
        print("Showing plots (close plot windows to finish execution)...")
        plt.show()


if __name__ == "__main__":
    main()