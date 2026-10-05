"""Main entry point for GGRD-SP project."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from models import (
    format_predicted_measurement_id,
    run_spectral_pipeline,
)
from utility import (
    calculate_atomic_percentage_split_statistics,
    calculate_atomic_percentages,
    calculate_prediction_metrics,
    format_side_by_side_atomic_percentages,
    format_side_by_side_metrics,
    generate_pseudo_measurements,
    pair_source_target_spectra,
    plot_max_intensity_vs_time,
    plot_normalized_max_intensity_vs_time,
    plot_pairing_timeline,
    plot_prediction_comparison,
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
    # "resnet": 1D ResNet (3 CNN layers with mirror padding and shortcut)
    # "residual_unet": 1D Residual U-Net (y = x + UNet(x))
    # "unet": Conventional 1D U-Net without shortcut (y = UNet(x))
    # "deeplabv3": 1D DeepLabV3 (Multi-Grid ResNet backbone + ASPP with global pooling)
    model_type = "resnet"  # Options: "resnet", "residual_unet", "unet", or "deeplabv3"
    use_sliding_window = True  # Toggle available to all models (True for patch data augmentation)

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
            source_tool: {"shift_ev": 0.0, "scale": 10000.0},
            target_tool: {"shift_ev": 1.1, "scale": 14000.0},
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

    # Part 3: Unified Neural Network & Ax Bayesian Optimization configuration
    pipeline_config = {
        "model_type": model_type,
        "use_sliding_window": use_sliding_window,
        "regions": ["Al2p", "Ti2p", "O1s", "C1s", "Cl2p"],
        "source_tool": source_tool,
        "target_tool": target_tool,
        "train_ratio": 0.5,
        "val_ratio": 0.2,
        "test_ratio": 0.3,
        "seed": None,
        "window_size_ev": 3.0,  # Global default window length in eV
        "sliding_stride_ev": 1.0,  # Global default sliding stride step in eV
        "use_bayesian_opt": False,  # Set to True to enable Ax Bayesian hyperparameter optimization
        "bayesian_opt_config": {
            "num_trials": 5,
            "epochs_per_trial": 15,
            "batch_sizes": [16, 32, 64],
            "kernel_sizes": [3, 5, 7],
            "hidden_channels": [16, 32, 64],
            "base_channels": [16, 32],
            "depths": [2, 3],
            "lr_bounds": (1e-4, 1e-2),
            "l2_bounds": (1e-6, 1e-2),
            "verbose": True,
        },
        "train_config": {
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "hidden_channels": 32,
            "base_channels": 16,
            "depth": 3,
            "kernel_size": 5,
            "l2_weight": 1e-6,
            "early_stopping_patience": 10,
            "verbose": True,
        },
        # Per-region configuration overrides (e.g. specialized window size or learning rate)
        "region_configs": {
            "Al2p": {
                "window_size_ev": 3.0,
                "sliding_stride_ev": 1.0,
                "batch_size": 16,
                "learning_rate": 1e-4,
            },
        },
        "predict_source": True,
    }

    # Part 4: Side-by-side performance report configuration
    report_config = {
        "splits": ["train", "test"],
        "metrics": ["norm_mse", "peak_err%", "max%err"],
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
    # PART 3: NEURAL NETWORK TRAINING & PREDICTION (Unified 8-step workflow)
    # =========================================================================
    print(
        f"\n=== Part 3: Neural Network Pipeline (model={model_type}, sliding_window={use_sliding_window}) ==="
    )
    model_results = run_spectral_pipeline(
        data=(ary_intensity, ary_energy, meta_df),
        model_type=model_type,
        use_sliding_window=use_sliding_window,
        config=pipeline_config,
    )

    val_evaluation = model_results["evaluation"]
    test_evaluation = model_results.get("test_evaluation", {})
    print("\nRegional Evaluation Losses:")
    print("  Region | Val Normalized MSE | Test Normalized MSE")
    print("  -------+--------------------+--------------------")
    for reg, val_l in sorted(val_evaluation.items()):
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

    df_side_by_side = format_side_by_side_metrics(df_metrics_samples, config=report_config)
    if not df_side_by_side.empty:
        splits_list = report_config.get("splits", ["train", "test"])
        split_titles = " vs. ".join([s.capitalize() for s in splits_list])
        print(f"\nSide-by-side performance comparison across splits ({split_titles}):")
        print(df_side_by_side.to_string(index=False))

    # =========================================================================
    # PART 5: CALCULATE ATOMIC PERCENTAGES (Shirley Integration)
    # =========================================================================
    print("\n=== Part 5: Calculating Atomic Percentage (Shirley Integration) ===")

    # 1. Statistics across Train and Test dataset splits
    print("\nAtomic percentage statistics across dataset splits (Train vs. Test):")
    df_at_samples, df_at_summary = calculate_atomic_percentage_split_statistics(
        data_orig,
        data_pred,
        config={"splits": ["train", "test"]},
    )

    if not df_at_summary.empty:
        for split_name in ["train", "test"]:
            split_df = df_at_summary[df_at_summary["split"] == split_name]
            if not split_df.empty:
                print(f"\n--- Split: {split_name.upper()} ---")
                display_cols = [
                    "element",
                    "n_samples",
                    "source_mean",
                    "source_std",
                    "target_mean",
                    "target_std",
                    "pred_mean",
                    "pred_std",
                    "diff_mean",
                    "diff_std",
                    "mae",
                ]
                print(split_df[display_cols].to_string(index=False))

        # Side-by-side comparison across splits
        df_at_side_by_side = format_side_by_side_atomic_percentages(
            df_at_summary, config={"splits": ["train", "test"]}
        )
        if not df_at_side_by_side.empty:
            print("\nSide-by-side atomic percentage comparison (Train vs. Test):")
            print(df_at_side_by_side.to_string(index=False))

    # 2. Detailed single session drill-down
    def _print_quant_summary(label: str, meas: str, df_die: pd.DataFrame, df_sum: pd.DataFrame) -> None:
        print(f"\n[{label}] Atomic percentage per die for session '{meas}' (first 3 dies):")
        print(df_die.head(3).to_string(index=False))
        print(f"\nSummary across all dies for {label} '{meas}':")
        print(df_sum.to_string(index=False))

    pred_sample_meas_id = format_predicted_measurement_id(sample_meas_id, source_tool, target_tool)

    df_per_die_src, df_summary_src = calculate_atomic_percentages(
        ary_energy, ary_intensity, meta_df, config={"measurement_id": sample_meas_id}
    )
    df_per_die_pred, df_summary_pred = calculate_atomic_percentages(
        ary_energy_predicted, ary_intensity_predicted, meta_df_predicted,
        config={"measurement_id": pred_sample_meas_id}
    )

    _print_quant_summary("Source Measured", sample_meas_id, df_per_die_src, df_summary_src)
    _print_quant_summary("Predicted Target", pred_sample_meas_id, df_per_die_pred, df_summary_pred)

    # =========================================================================
    # DEMONSTRATION PLOTS
    # =========================================================================
    print("\nGenerating demonstration plots...")

    # 1. Maximum intensity vs. measurement time (hue=tool, die, or region)
    plot_max_intensity_vs_time(
        ary_intensity,
        meta_df,
        plot_config={"hue": "tool", "title": "Maximum Spectral Intensity vs. Measurement Time (hue=tool)"},
    )

    # 2. Maximum intensity normalized by total integrated area vs. measurement time
    plot_normalized_max_intensity_vs_time(
        ary_intensity,
        meta_df,
        ary_energy=ary_energy,
        plot_config={
            "hue": "tool",
            "normalization_mode": "die_total_flux",
            "title": "Normalized Maximum Intensity vs. Measurement Time (hue=tool, die_total_flux)",
        },
    )

    # 3. Direct tool comparison between tools for a region (Die 0)
    plot_tool_comparison(
        ary_energy,
        ary_intensity,
        meta_df,
        compare_config={"tools": (source_tool, target_tool), "region": example_region, "die": 0},
    )

    # 4. 1-to-1 Measurement pairing timeline
    plot_pairing_timeline(meta_df, plot_config={"source_tool": source_tool, "target_tool": target_tool})

    # 5. Shirley background subtraction in multi-region grid mode (all 5 regions for Die 0)
    plot_shirley_background(
        ary_energy,
        ary_intensity,
        meta_df,
        plot_config={"measurement_id": sample_meas_id, "die": 0},
    )

    # 6. Training loss history curves across all regions (Part 4)
    model_labels = {
        "resnet": "1D ResNet",
        "residual_unet": "1D Residual U-Net",
        "unet_residual": "1D Residual U-Net",
        "unet": "Conventional 1D U-Net",
    }
    mode_str = "Sliding Window" if use_sliding_window else "Full Spectrum"
    model_name_label = f"{model_labels.get(model_type, model_type)} ({mode_str})"

    plot_training_history(
        model_results["histories"],
        plot_config={"title": f"Part 4: {model_name_label} Neural Network Training & Validation Loss"},
    )

    # 7. Sliding window decomposition (Original vs Sliced Spectra) if sliding window chosen
    if use_sliding_window:
        example_subset = meta_df[
            (meta_df["measurement_id"] == sample_meas_id)
            & (meta_df["region"] == example_region)
            & (meta_df["die"] == 0)
        ]
        if not example_subset.empty:
            ex_idx = int(example_subset.iloc[0]["spectrum_index"])
            plot_sliding_window_slices(
                ary_intensity[ex_idx],
                ary_energy[ex_idx],
                plot_config={
                    "region": example_region,
                    "window_size_ev": pipeline_config.get("window_size_ev", 2.0),
                    "sliding_stride_ev": pipeline_config.get("sliding_stride_ev", 1.0),
                    "offset_patches": True,
                    "title": (
                        f"Sliding Window Decomposition - {example_region} "
                        f"(Session {sample_meas_id}, Die 0)"
                    ),
                },
            )

    # 8. Spectral transfer comparison: Source vs. Target vs. Transformed across all regions in sub-plots
    # Defaults to test split measurement session; strict color scheme: source=black, target=red, transformed=blue dotted
    plot_prediction_comparison(
        data_orig,
        data_pred,
        config={
            "session_splits": model_results.get("session_splits"),
            "die": 0,
        },
    )

    if show_plots:
        print("Showing plots (close plot windows to finish execution)...")
        plt.show()


if __name__ == "__main__":
    main()