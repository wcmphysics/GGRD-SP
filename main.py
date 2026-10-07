"""Main entry point for GGRD-SP project."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from models import run_spectral_pipeline
from utility import (
    calculate_atomic_percentage_split_statistics,
    calculate_prediction_metrics,
    format_side_by_side_atomic_percentages,
    format_side_by_side_metrics,
    generate_pseudo_measurements,
    pair_source_target_spectra,
    plot_atomic_percentage_distributions,
    plot_atomic_percentage_mae,
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
    use_sliding_window = False  # Toggle available to all models (True for patch data augmentation)

    source_tool = "J4"
    target_tool = "J5"
    example_region = "Ti2p"
    sample_meas_id = "NMG_M_J4_00000"  # session for at% calculation and comparison
    show_plots = "--no-plots" not in sys.argv and "--headless" not in sys.argv  # Set False or pass --no-plots to skip GUI plot display

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
        "use_film": False, 
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
            "use_lr_scheduler": True,
            "lr_reduce_factor": 0.9,
            "lr_scheduler_patience": 10,
            "hidden_channels": 32,
            "base_channels": 16,
            "depth": 3,
            "kernel_size": 9,
            "l2_weight": 1e-6,
            "early_stopping_patience": 20,
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
    # Synthesize multi-die, multi-region regional spectra with physical binding energy
    # calibration shifts and tool-specific intensity scaling factors.
    print("=== Part 1: Generating Pseudo-Measurements ===")
    ary_intensity, ary_energy, meta_df = generate_pseudo_measurements(pseudo_config)
    print(f"Intensity shape: {ary_intensity.shape} | Energy shape: {ary_energy.shape} | Total spectra: {len(meta_df)}")

    # =========================================================================
    # PART 2: PAIR SOURCE AND TARGET SPECTRA
    # =========================================================================
    # Establish strictly 1-to-1 temporal pairings between source and target sessions
    # within the configured time threshold window.
    print("\n=== Part 2: Pairing Source and Target Spectra ===")
    meta_df = pair_source_target_spectra(meta_df, pairing_config)
    src_mask = meta_df["tool"] == source_tool
    n_paired = meta_df.loc[src_mask, "measurement_id_target"].dropna().nunique()
    n_total = meta_df.loc[src_mask, "measurement_id"].nunique()
    print(f"Paired measurement sessions: {n_paired}/{n_total}")

    # =========================================================================
    # PART 3: NEURAL NETWORK PIPELINE (Unified 8-step workflow)
    # =========================================================================
    # Orchestrates train/val/test session splitting, sliding window slicing (if enabled),
    # weighted multi-region normalized MSE loss, model training, and target prediction.
    print(f"\n=== Part 3: Neural Network Pipeline (model={model_type}, sliding_window={use_sliding_window}) ===")
    model_results = run_spectral_pipeline(
        data=(ary_intensity, ary_energy, meta_df),
        model_type=model_type,
        use_sliding_window=use_sliding_window,
        config=pipeline_config,
    )
    ary_intensity_predicted, ary_energy_predicted, meta_df_predicted = model_results["predictions"]
    print(f"Predicted spectra shape: {ary_intensity_predicted.shape} (total: {len(meta_df_predicted)})")

    # =========================================================================
    # PART 4: MODEL PERFORMANCE OBSERVATION & METRICS
    # =========================================================================
    # Evaluate prediction fidelity metrics (Normalized MSE, Peak Error, Max Error)
    # formatted side-by-side across Train and Test splits.
    print("\n=== Part 4: Model Performance Observation & Metrics ===")
    data_orig = (ary_intensity, ary_energy, meta_df)
    data_pred = (ary_intensity_predicted, ary_energy_predicted, meta_df_predicted)

    df_metrics_samples, _ = calculate_prediction_metrics(data_orig, data_pred)
    df_side_by_side = format_side_by_side_metrics(df_metrics_samples, config=report_config)
    if not df_side_by_side.empty:
        print("\nSide-by-side performance comparison across splits (Train vs. Test):")
        print(df_side_by_side.to_string(index=False))

    # =========================================================================
    # PART 5: CALCULATE ATOMIC PERCENTAGES (Shirley Integration)
    # =========================================================================
    # Calculate elemental atomic percentages using Shirley background subtraction
    # scaled by Scofield Relative Sensitivity Factors (RSF).
    # format_side_by_side_atomic_percentages compiles Target, Predicted,
    # Difference, and MAE ± Std side-by-side for all elements and Overall composition.
    print("\n=== Part 5: Calculating Atomic Percentage (Shirley Integration) ===")
    df_at_samples, df_at_summary = calculate_atomic_percentage_split_statistics(
        data_orig,
        data_pred,
        config={"splits": ["train", "test"]},
    )
    df_at_side_by_side = format_side_by_side_atomic_percentages(
        df_at_summary, config={"splits": ["train", "test"]}
    )
    if not df_at_side_by_side.empty:
        print("\nSide-by-side atomic percentage comparison (Train vs. Test):")
        print(df_at_side_by_side.to_string(index=False))

    # =========================================================================
    # DEMONSTRATION DIAGNOSTIC PLOTS
    # =========================================================================
    # Render diagnostics covering data integrity, background subtraction,
    # training loss convergence, spectral overlays, and atomic % errors.
    print("\nGenerating demonstration plots...")

    # 1. Timeline pairing and intensity stability across measurement time
    plot_pairing_timeline(meta_df, plot_config={"source_tool": source_tool, "target_tool": target_tool})
    plot_max_intensity_vs_time(ary_intensity, meta_df, region=example_region, die=0, plot_config={"hue": "tool"})
    plot_normalized_max_intensity_vs_time(
        ary_intensity, meta_df, ary_energy=ary_energy, region=example_region, die=0,
        plot_config={"hue": "tool", "normalization_mode": "die_total_flux"},
    )
    plot_tool_comparison(
        ary_energy, ary_intensity, meta_df,
        compare_config={"tools": (source_tool, target_tool), "region": example_region, "die": 0},
    )

    # 2. Shirley background subtraction in multi-region grid mode (Die 0)
    plot_shirley_background(ary_energy, ary_intensity, meta_df, plot_config={"measurement_id": sample_meas_id, "die": 0})

    # 3. Neural network training & validation loss history
    plot_training_history(model_results["histories"])

    # 4. Optional sliding window patch decomposition (if sliding window enabled)
    if use_sliding_window:
        ex_match = meta_df[(meta_df["measurement_id"] == sample_meas_id) & (meta_df["region"] == example_region) & (meta_df["die"] == 0)]
        if not ex_match.empty:
            ex_idx = int(ex_match.iloc[0]["spectrum_index"])
            plot_sliding_window_slices(ary_intensity[ex_idx], ary_energy[ex_idx], plot_config={"region": example_region})

    # 5. Spectral transfer comparison: Source vs. Target vs. Predicted across regions
    plot_prediction_comparison(data_orig, data_pred, config={"session_splits": model_results.get("session_splits"), "die": 0})

    # 6. Atomic percentage distributions and Mean Absolute Error (MAE)
    if not df_at_samples.empty:
        plot_atomic_percentage_distributions(df_at_samples)
        plot_atomic_percentage_mae(df_at_samples)

    if show_plots:
        print("Showing plots (close plot windows to finish execution)...")
        plt.show()


if __name__ == "__main__":
    main()