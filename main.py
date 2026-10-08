"""Main entry point for GGRD-SP project."""

from __future__ import annotations

import sys
import matplotlib.pyplot as plt

from models import run_spectral_pipeline
import utility as ut


def main() -> None:
    """Execute pseudo-measurement generation, pairing, quantification, and visualization."""
    # =========================================================================
    # GLOBAL SETUP & HYPERPARAMETERS
    # =========================================================================
    # Sequence-to-sequence model architectures:
    # - "resnet": 1D ResNet with 3 CNN layers, mirror padding, and identity shortcut.
    # - "residual_unet": 1D Residual U-Net (y = x + UNet(x)) preserving physical baseline.
    # - "unet": Standard encoder-decoder U-Net without shortcut.
    # - "deeplabv3": Multi-grid ResNet backbone with Atrous Spatial Pyramid Pooling (ASPP).
    model_type = "resnet"
    use_sliding_window = False
    source_tool = "J4"
    target_tool = "J5"
    example_region = "Ti2p"
    sample_meas_id = "NMG_M_J4_00000"
    show_plots = "--no-plots" not in sys.argv and "--headless" not in sys.argv

    # Synthetic XPS data generation configuration:
    # Simulates multi-die physical spectra with binding energy shifts and intensity scaling.
    pseudo_config = {
        "material": "NMG",
        "regions": ["Al2p", "Ti2p", "O1s", "C1s", "Cl2p"],
        "n_points": 100,
        "measurements_per_tool": {source_tool: 25, target_tool: 30},
        "measurements_per_t7_code": 10,
        "interval_hours_range": (4.0, 24.0),
        "die_variation_std": 0.03,
        "noise_relative_std": 0.015,
        "seed": None,
        "tool_offsets": {
            source_tool: {"shift_ev": 0.0, "scale": 10000.0},
            target_tool: {"shift_ev": 1.1, "scale": 14000.0},
        },
    }

    # 1-to-1 temporal pairing configuration:
    # Pairs each source measurement with the closest target measurement within time_threshold_hours.
    pairing_config = {
        "source_tool": source_tool,
        "target_tool": target_tool,
        "time_threshold_hours": 8.0,
        "method": "optimal",
        "material": None,
    }

    # Neural network pipeline configuration:
    # Orchestrates session splitting, multi-region normalized MSE loss, and model training.
    # Ax Bayesian hyperparameter optimization can be activated by setting use_bayesian_opt=True.
    pipeline_config = {
        "model_type": model_type,
        "use_sliding_window": use_sliding_window,
        "use_film": False,
        "use_bayesian_opt": False,
        "regions": pseudo_config["regions"],
        "source_tool": source_tool,
        "target_tool": target_tool,
        "train_ratio": 0.5,
        "val_ratio": 0.2,
        "test_ratio": 0.3,
        "seed": None,
        "window_size_ev": 3.0,
        "sliding_stride_ev": 1.0,
        "predict_source": True,
        "bayesian_opt_config": {
            "num_trials": 5,
            "epochs_per_trial": 15,
            "verbose": True,
            "lr_bounds": (1e-4, 1e-2),
            "l2_bounds": (1e-6, 1e-2),
            "batch_sizes": [16, 32, 64],
            "kernel_sizes": [3, 5, 7],
            "hidden_channels": [16, 32, 64],
            "base_channels": [16, 32],
            "depths": [2, 3],
        },
        "train_config": {
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "depth": 3,
            "kernel_size": 9,
            "l2_weight": 1e-6,
            "use_lr_scheduler": True,
            "lr_reduce_factor": 0.9,
            "lr_scheduler_patience": 10,
            "early_stopping_patience": 20,
            "verbose": True,
            "hidden_channels": 32,
            "base_channels": 16,
        },
        "region_configs": {
            "Al2p": {
                "window_size_ev": 3.0,
                "sliding_stride_ev": 1.0,
                "batch_size": 16,
                "learning_rate": 1e-4,
            }
        },
    }
    report_config = {
        "splits": ["train", "test"],
        "metrics": ["norm_mse", "peak_err%", "max%err"],
    }

    # =========================================================================
    # PARTS 1 & 2: DATA GENERATION & 1-TO-1 TEMPORAL PAIRING
    # =========================================================================
    print("=== Parts 1 & 2: Data Generation & Pairing ===")
    ary_intensity, ary_energy, df_meta = ut.generate_pseudo_measurements(pseudo_config)
    df_meta = ut.pair_source_target_spectra(df_meta, pairing_config)
    source_df = df_meta[df_meta["tool"] == source_tool]
    print(
        f"Paired sessions: {source_df['measurement_id_target'].dropna().nunique()}/{source_df['measurement_id'].nunique()}"
    )
    data_orig = (ary_intensity, ary_energy, df_meta)

    # =========================================================================
    # PART 3: NEURAL NETWORK PIPELINE (Unified 8-step workflow)
    # =========================================================================
    print(f"\n=== Part 3: Neural Pipeline ({model_type}) ===")
    results = run_spectral_pipeline(
        data=data_orig,
        model_type=model_type,
        use_sliding_window=use_sliding_window,
        config=pipeline_config,
    )
    data_pred = results["predictions"]
    print(f"Predicted spectra shape: {data_pred[0].shape} (total: {len(data_pred[2])})")

    # =========================================================================
    # PARTS 4 & 5: PERFORMANCE METRICS & ATOMIC % QUANTIFICATION
    # =========================================================================
    # Evaluate prediction fidelity (Normalized MSE, Peak Error, Max Error) and compute
    # elemental atomic percentages using Shirley background subtraction and Scofield RSF scaling.
    print("\n=== Parts 4 & 5: Performance Metrics & Atomic % Quantification ===")
    df_metrics, _ = ut.calculate_prediction_metrics(data_orig, data_pred)
    df_at_samples, df_at_summary = ut.calculate_atomic_percentage_split_statistics(
        data_orig, data_pred, config=report_config
    )
    print(
        "\nMetrics (Train vs. Test):\n",
        ut.format_side_by_side_metrics(df_metrics, config=report_config).to_string(index=False),
    )
    print(
        "\nAtomic % (Train vs. Test):\n",
        ut.format_side_by_side_atomic_percentages(df_at_summary, config=report_config).to_string(index=False),
    )

    # =========================================================================
    # DEMONSTRATION DIAGNOSTIC PLOTS
    # =========================================================================
    if show_plots:
        print("\nGenerating demonstration plots...")
        for plot_fn in (
            lambda: ut.plot_pairing_timeline(
                df_meta,
                plot_config={"source_tool": source_tool, "target_tool": target_tool},
            ),
            lambda: ut.plot_max_intensity_vs_time(
                ary_intensity,
                df_meta,
                region=example_region,
                die=0,
                plot_config={"hue": "tool"},
            ),
            lambda: ut.plot_normalized_max_intensity_vs_time(
                ary_intensity,
                df_meta,
                ary_energy=ary_energy,
                region=example_region,
                die=0,
                plot_config={"hue": "tool", "normalization_mode": "die_total_flux"},
            ),
            lambda: ut.plot_tool_comparison(
                ary_energy,
                ary_intensity,
                df_meta,
                compare_config={
                    "tools": (source_tool, target_tool),
                    "region": example_region,
                    "die": 0,
                },
            ),
            lambda: ut.plot_shirley_background(
                ary_energy,
                ary_intensity,
                df_meta,
                plot_config={"measurement_id": sample_meas_id, "die": 0},
            ),
            lambda: ut.plot_training_history(results["histories"]),
            lambda: ut.plot_prediction_comparison(
                data_orig,
                data_pred,
                config={"session_splits": results.get("session_splits"), "die": 0},
            ),
            lambda: ut.plot_atomic_percentage_distributions(df_at_samples),
            lambda: ut.plot_atomic_percentage_mae(df_at_samples),
        ):
            plot_fn()
        if use_sliding_window:
            ex = df_meta[
                (df_meta["measurement_id"] == sample_meas_id)
                & (df_meta["region"] == example_region)
                & (df_meta["die"] == 0)
            ]
            if not ex.empty:
                ut.plot_sliding_window_slices(
                    ary_intensity[int(ex.iloc[0]["spectrum_index"])],
                    ary_energy[int(ex.iloc[0]["spectrum_index"])],
                    plot_config={"region": example_region},
                )
        print("Showing plots (close plot windows to finish execution)...")
        plt.show()


if __name__ == "__main__":
    main()