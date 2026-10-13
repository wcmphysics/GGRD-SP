"""Main entry point for GGRD-SP project."""

from __future__ import annotations

import sys
import matplotlib.pyplot as plt

import models
import utility as ut
import visualization as vz


def main() -> None:
    """Execute sequential key stages: generation, normalization, pairing, splitting,
    dataset construction, cost preparation, training, prediction, quantification, and plotting.
    """
    # =========================================================================
    # GLOBAL SETUP & CONFIGURATIONS
    # =========================================================================
    model_type = "resnet"
    use_sliding_window = False
    source_tool, target_tool = "J4", "J5"
    example_region, sample_meas_id = "Ti2p", "NMG_M_J4_00000"
    show_plots = "--no-plots" not in sys.argv and "--headless" not in sys.argv

    # Stage 1: Data generation configuration
    pseudo_cfg = {
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

    # Stage 2: Normalization configuration (by total flux across regions)
    norm_cfg = {"normalization_mode": "total_flux"}

    # Stage 3: 1-to-1 temporal pairing configuration
    pair_cfg = {
        "source_tool": source_tool,
        "target_tool": target_tool,
        "time_threshold_hours": 8.0,
        "method": "optimal",
        "material": None,
    }

    # Stage 4: Train-val-test measurement session splitting configuration
    split_cfg = {
        "method": "random",
        "source_tool": source_tool,
        "target_tool": target_tool,
        "train_ratio": 0.5,
        "val_ratio": 0.2,
        "test_ratio": 0.3,
        "seed": 42,
    }

    # Stage 5: Dataset & patch augmentation configuration
    dataset_cfg = {
        "regions": pseudo_cfg["regions"],
        "source_tool": source_tool,
        "target_tool": target_tool,
        "use_sliding_window": use_sliding_window,
        "window_size_ev": 3.0,
        "sliding_stride_ev": 1.0,
        "region_configs": {
            "Al2p": {"window_size_ev": 3.0, "sliding_stride_ev": 1.0},
        },
    }

    # Stage 6: Cost function configuration
    cost_cfg = {
        "w_mse": 1.0,
        "w_shape": 0.0,
        "l2_weight": 1e-6,
        "cost_type": "normalized_mse",
    }

    # Stage 7: Neural network training and Bayesian optimization configuration
    train_cfg = {
        "model_type": model_type,
        "use_sliding_window": use_sliding_window,
        "use_film": False,
        "use_bayesian_opt": False,
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
            "Al2p": {"batch_size": 16, "learning_rate": 1e-4},
        },
    }

    # Stage 8: Model prediction configuration
    pred_cfg = {
        "source_tool": source_tool,
        "target_tool": target_tool,
        "normalization_mode": "none",
        "clamp_non_negative": True,
    }

    rpt_cfg = {"splits": ["train", "test"], "metrics": ["norm_mse", "peak_err%", "max%err"]}

    # =========================================================================
    # STAGE 1: SPECTRUM DATA READING OR GENERATION
    # =========================================================================
    print("=== Stage 1: Data Generation ===")
    ary_i, ary_e, df_meta = ut.generate_pseudo_measurements(pseudo_cfg)
    print(f"Generated raw spectra: {ary_i.shape[0]} rows, {ary_i.shape[1]} points")

    # =========================================================================
    # STAGE 2: NORMALIZATION OF SPECTRAL DATA
    # =========================================================================
    print("\n=== Stage 2: Normalization ===")
    ary_i, ary_e, df_meta = ut.normalize_spectral_data(ary_i, ary_e, df_meta, norm_cfg)
    print(f"Normalized intensity by {norm_cfg['normalization_mode']}; max={ary_i.max():.4f}")

    # =========================================================================
    # STAGE 3: PAIRING SOURCE AND TARGET SPECTRA
    # =========================================================================
    print("\n=== Stage 3: 1-to-1 Tool Pairing ===")
    df_meta = ut.pair_source_target_spectra(df_meta, pair_cfg)
    src_df = df_meta[df_meta["tool"] == source_tool]
    print(
        f"Paired sessions: {src_df['measurement_id_target'].dropna().nunique()}"
        f"/{src_df['measurement_id'].nunique()}"
    )

    # Establish fixed tool mapping indices
    unique_tools = sorted(
        set(df_meta["tool"].dropna().astype(str).unique())
        | set(df_meta["tool_target"].dropna().astype(str).unique())
    )
    tool_map = {tool_name: idx for idx, tool_name in enumerate(unique_tools)}
    df_meta["tool_index"] = df_meta["tool"].map(tool_map)
    df_meta["tool_target_index"] = df_meta["tool_target"].map(tool_map)
    train_cfg["num_tools"] = len(tool_map)

    # =========================================================================
    # STAGE 4: TRAIN-VAL-TEST DATA SPLITTING
    # =========================================================================
    print("\n=== Stage 4: Train-Val-Test Session Splitting ===")
    df_meta, measurement_mapping, session_splits = ut.split_measurement_sessions(
        df_meta, split_cfg
    )
    print(
        f"Session splits: train={len(session_splits['train'])}, "
        f"val={len(session_splits['val'])}, test={len(session_splits['test'])}"
    )

    # =========================================================================
    # STAGE 5: DATASET AND DATALOADER DEFINITION (With Augmentation)
    # =========================================================================
    print("\n=== Stage 5: Regional Dataset Definition & Patch Augmentation ===")
    regional_datasets = models.create_regional_datasets(
        df_meta, ary_i, ary_e, config=dataset_cfg
    )
    for reg, bnd in regional_datasets.items():
        print(
            f"  [{reg}] train samples={len(bnd['dataset_train'])}, "
            f"val={len(bnd['dataset_val'])}, test={len(bnd['dataset_test'])}"
        )

    # =========================================================================
    # STAGE 6: COST FUNCTION DEFINITION AND PREPARATION
    # =========================================================================
    print("\n=== Stage 6: Cost Function Preparation ===")
    criterion = models.create_cost_function(cost_cfg)
    print(f"Initialized criterion: {criterion.__class__.__name__}")

    # =========================================================================
    # STAGE 7: NEURAL NETWORK TRAINING AND BAYESIAN OPTIMIZATION
    # =========================================================================
    print(f"\n=== Stage 7: Neural Model Training ({model_type}) ===")
    results = models.perform_model_training_and_bayesian_optimization(
        regional_datasets, criterion, config=train_cfg
    )
    print("Model training completed across all regions.")

    # =========================================================================
    # STAGE 8: PREDICTION FROM MODEL
    # =========================================================================
    print("\n=== Stage 8: Transformed Spectrum Prediction ===")
    pred_cfg["session_splits"] = session_splits
    data_orig = (ary_i, ary_e, df_meta)
    data_pred = models.predict_spectra(
        models=results["models"], data=data_orig, config=pred_cfg
    )
    print(f"Predicted spectra shape: {data_pred[0].shape} (total: {len(data_pred[2])})")

    # =========================================================================
    # STAGES 9 & 10: PERFORMANCE METRICS & ATOMIC % QUANTIFICATION
    # =========================================================================
    print("\n=== Stages 9 & 10: Performance Metrics & Atomic % Quantification ===")
    df_metrics, _ = ut.calculate_prediction_metrics(data_orig, data_pred)
    df_at_samples, df_at_summary = ut.calculate_atomic_percentage_split_statistics(
        data_orig, data_pred, config=rpt_cfg
    )
    print(
        "\nMetrics (Train vs. Test):\n",
        ut.format_side_by_side_metrics(df_metrics, config=rpt_cfg).to_string(index=False),
    )
    print(
        "\nAtomic % (Train vs. Test):\n",
        ut.format_side_by_side_atomic_percentages(df_at_summary, config=rpt_cfg).to_string(
            index=False
        ),
    )

    # =========================================================================
    # STAGE 11: DIAGNOSTIC PLOTS
    # =========================================================================
    if show_plots:
        print("\nGenerating demonstration plots...")
        for plot_fn in (
            lambda: vz.plot_pairing_timeline(
                df_meta, plot_config={"source_tool": source_tool, "target_tool": target_tool}
            ),
            lambda: vz.plot_flux_and_averaged_intensity_vs_time(
                ary_i, ary_e, df_meta, plot_config={"source_tool": source_tool, "target_tool": target_tool}
            ),
            lambda: vz.plot_max_intensity_vs_time(
                ary_i, df_meta, region=example_region, die=0, plot_config={"hue": "tool"}
            ),
            lambda: vz.plot_normalized_max_intensity_vs_time(
                ary_i, df_meta, ary_energy=ary_e, region=example_region, die=0,
                plot_config={"hue": "tool", "normalization_mode": "die_total_flux"}
            ),
            lambda: vz.plot_tool_comparison(
                ary_e, ary_i, df_meta,
                compare_config={"tools": (source_tool, target_tool), "region": example_region, "die": 0}
            ),
            lambda: vz.plot_shirley_background(
                ary_e, ary_i, df_meta,
                plot_config={"measurement_id": sample_meas_id, "die": 0}
            ),
            lambda: vz.plot_training_history(results["histories"]),
            lambda: vz.plot_prediction_comparison(
                data_orig, data_pred, config={"session_splits": session_splits, "die": 0}
            ),
            lambda: vz.plot_atomic_percentage_distributions(df_at_samples),
            lambda: vz.plot_atomic_percentage_mae(df_at_samples),
        ):
            plot_fn()
        if use_sliding_window:
            ex = df_meta[
                (df_meta["measurement_id"] == sample_meas_id)
                & (df_meta["region"] == example_region)
                & (df_meta["die"] == 0)
            ]
            if not ex.empty:
                vz.plot_sliding_window_slices(
                    ary_i[int(ex.iloc[0]["spectrum_index"])],
                    ary_e[int(ex.iloc[0]["spectrum_index"])],
                    plot_config={"region": example_region},
                )
        print("Showing plots (close plot windows to finish execution)...")
        plt.show()


if __name__ == "__main__":
    main()