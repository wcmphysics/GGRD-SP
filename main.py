"""Main entry point for GGRD-SP project."""

from __future__ import annotations

import sys
import matplotlib.pyplot as plt

import models
import utility
import visualization


def main() -> None:
    """Execute sequential key stages: generation, normalization, pairing,
    splitting, dataset construction, cost preparation, training, prediction,
    quantification, and plotting.
    """
    # =========================================================================
    # GLOBAL SETUP & CONFIGURATIONS
    # =========================================================================
    model_type = "resnet"
    use_sliding_window = False
    source_tool, target_tool = "J4", "J5"
    example_region = "Ti2p"
    sample_measurement_id = "NMG_M_J4_00000"
    show_plots = "--no-plots" not in sys.argv and "--headless" not in sys.argv

    # Stage 1: Data generation configuration
    data_generation_config = {
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

    # Stage 2: 1-to-1 temporal pairing configuration
    pairing_config = {
        "source_tool": source_tool,
        "target_tool": target_tool,
        "time_threshold_hours": 8.0,
        "method": "optimal",
        "material": None,
    }

    # Stage 3: Normalization configuration (by total flux across regions)
    normalization_config = {"normalization_mode": "source_regional_max"}

    # Stage 4: Train-val-test measurement session splitting configuration
    splitting_config = {
        "method": "random",
        "source_tool": source_tool,
        "target_tool": target_tool,
        "train_ratio": 0.5,
        "val_ratio": 0.2,
        "test_ratio": 0.3,
        "seed": 42,
    }

    # Stage 4B: Gaussian Process die flux prediction configuration
    gp_flux_config = {
        "reference_date": "2026-01-01",
        "unpaired_split": "train",
        "target_column": "die_flux",
        "kernel": {
            "time_continuous": {
                "type": "rbf",
                "initial_lengthscale": 5.0,
            },
            "time_jump": {
                "enabled": True,
                "initial_step_time_days": 10.0,
                "initial_steepness": 2.0,
                "learn_step_time": True,
            },
            "die": {
                "type": "index",
                "num_dies": 9,
                "rank": 2,
            },
            "composition": "product_with_jump",
        },
        "training": {
            "num_iterations": 150,
            "learning_rate": 0.05,
        },
        "plot": {
            "plot_die": 0,
        },
    }

    # Stage 5: Dataset & patch augmentation configuration
    dataset_config = {
        "regions": data_generation_config["regions"],
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
    cost_config = {
        "w_mse": 1.0,
        "w_shape": 0.0,
        "w_gradient": 0.0,
        "l2_weight": 1e-6,
        "normalized_by_target_max": True,
    }

    # Dedicated Feature-wise Linear Modulation (FiLM) configuration
    film_config = {
        "use_film": False,
        "encode_window_energy": True,
        "energy_scale_ev": 1000.0,
        "die_embed_dim": 4,
        "tool_embed_dim": 4,
        "hidden_dim": 32,
    }

    # Stage 7: Neural network training and Bayesian optimization configuration
    training_config = {
        "model_type": model_type,
        "use_sliding_window": use_sliding_window,
        "use_film": film_config["use_film"],
        "film_config": film_config,
        "use_bayesian_opt": False,
        "bayesian_opt_config": {
            "num_trials": 5,
            "epochs_per_trial": 15,
            "verbose": True,
            "learning_rate_bounds": (1e-4, 1e-2),
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
            "use_learning_rate_scheduler": True,
            "learning_rate_reduce_factor": 0.9,
            "learning_rate_scheduler_patience": 10,
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
    prediction_config = {
        "source_tool": source_tool,
        "target_tool": target_tool,
        "normalization_mode": "none",
        "clamp_non_negative": True,
        "film_config": film_config,
    }

    mse_metric_name = (
        "MSE_normalized_by_target"
        if cost_config.get("normalized_by_target_max", True)
        else "MSE"
    )
    metrics_report_config = {
        "splits": ["train", "test"],
        "metrics": [mse_metric_name, "peak_err%", "max%err"],
        "normalized_by_target_max": cost_config.get("normalized_by_target_max", True),
    }

    # =========================================================================
    # STAGE 1: SPECTRUM DATA READING OR GENERATION
    # =========================================================================
    print("=== Stage 1: Data Generation ===")
    ary_intensity, ary_energy, df_meta = utility.generate_pseudo_measurements(
        data_generation_config
    )
    print(
        f"Generated raw spectra: {ary_intensity.shape[0]} rows, "
        f"{ary_intensity.shape[1]} points"
    )

    # Calculate hierarchical flux metrics across regions, dies, and measurements
    df_meta = utility.calculate_flux(ary_intensity, ary_energy, df_meta)
    print(
        f"Calculated spectral fluxes: mean regional={df_meta['regional_flux'].mean():.2f}, "
        f"mean die={df_meta['die_flux'].mean():.2f}, "
        f"mean measurement={df_meta['measurement_flux'].mean():.2f}"
    )

    # =========================================================================
    # STAGE 2: PAIRING SOURCE AND TARGET SPECTRA
    # =========================================================================
    print("\n=== Stage 2: 1-to-1 Tool Pairing ===")
    df_meta = utility.pair_source_target_spectra(df_meta, pairing_config)
    df_source_tool = df_meta[df_meta["tool"] == source_tool]
    print(
        f"Paired sessions: {df_source_tool['measurement_id_target'].dropna().nunique()}"
        f"/{df_source_tool['measurement_id'].nunique()}"
    )

    # Establish fixed tool mapping indices
    unique_tools = sorted(
        set(df_meta["tool"].dropna().astype(str).unique())
        | set(df_meta["tool_target"].dropna().astype(str).unique())
    )
    tool_index_mapping = {
        tool_name: tool_index for tool_index, tool_name in enumerate(unique_tools)
    }
    df_meta["tool_index"] = df_meta["tool"].map(tool_index_mapping)
    df_meta["tool_target_index"] = df_meta["tool_target"].map(tool_index_mapping)
    training_config["num_tools"] = len(tool_index_mapping)

    # =========================================================================
    # STAGE 3: NORMALIZATION OF SPECTRAL DATA
    # =========================================================================
    print("\n=== Stage 3: Normalization ===")
    ary_intensity, ary_energy, df_meta = utility.normalize_spectral_data(
        ary_intensity, ary_energy, df_meta, normalization_config
    )
    print(
        f"Normalized intensity by {normalization_config['normalization_mode']}; "
        f"max={ary_intensity.max():.4f}"
    )

    # =========================================================================
    # STAGE 4: TRAIN-VAL-TEST DATA SPLITTING
    # =========================================================================
    print("\n=== Stage 4: Train-Val-Test Session Splitting ===")
    df_meta, measurement_mapping, session_splits = utility.split_measurement_sessions(
        df_meta, splitting_config
    )
    print(
        f"Session splits: train={len(session_splits['train'])}, "
        f"val={len(session_splits['val'])}, test={len(session_splits['test'])}"
    )

    # =========================================================================
    # STAGE 4B: GAUSSIAN PROCESS DIE FLUX PREDICTION
    # =========================================================================
    print("\n=== Stage 4B: Gaussian Process Die Flux Prediction ===")
    gp_bundle, df_gp_metrics = models.run_gp_flux_pipeline(
        df_meta, gp_flux_config
    )
    print("\nGP Die Flux Error Performance Summary:")
    print(df_gp_metrics.to_string(index=False))

    if show_plots:
        visualization.plot_gp_die_flux_timeline(
            gp_bundle, gp_flux_config
        )

    # =========================================================================
    # STAGE 5: DATASET AND DATALOADER DEFINITION (With Augmentation)
    # =========================================================================
    print("\n=== Stage 5: Regional Dataset Definition & Patch Augmentation ===")
    regional_datasets = models.create_regional_datasets(
        df_meta, ary_intensity, ary_energy, config=dataset_config
    )
    for region_name, dataset_bundle in regional_datasets.items():
        print(
            f"  [{region_name}] train samples={len(dataset_bundle['dataset_train'])}, "
            f"val={len(dataset_bundle['dataset_val'])}, "
            f"test={len(dataset_bundle['dataset_test'])}"
        )

    # =========================================================================
    # STAGE 6: COST FUNCTION DEFINITION AND PREPARATION
    # =========================================================================
    print("\n=== Stage 6: Cost Function Preparation ===")
    criterion = models.SpectralLoss(config=cost_config)
    print(f"Initialized criterion: {criterion.__class__.__name__}")

    # =========================================================================
    # STAGE 7: NEURAL NETWORK TRAINING AND BAYESIAN OPTIMIZATION
    # =========================================================================
    print(f"\n=== Stage 7: Neural Model Training ({model_type}) ===")
    training_results = models.perform_model_training_and_bayesian_optimization(
        regional_datasets, criterion, config=training_config
    )
    print("Model training completed across all regions.")

    # =========================================================================
    # STAGE 8: PREDICTION FROM MODEL
    # =========================================================================
    print("\n=== Stage 8: Transformed Spectrum Prediction ===")
    prediction_config["session_splits"] = session_splits
    data_original = (ary_intensity, ary_energy, df_meta)
    data_predicted = models.predict_spectra(
        models=training_results["models"], data=data_original, config=prediction_config
    )
    print(
        f"Predicted spectra shape: {data_predicted[0].shape} "
        f"(total: {len(data_predicted[2])})"
    )

    # =========================================================================
    # STAGES 9 & 10: PERFORMANCE METRICS & ATOMIC % QUANTIFICATION
    # =========================================================================
    print("\n=== Stages 9 & 10: Performance Metrics & Atomic % Quantification ===")
    df_prediction_metrics, _ = utility.calculate_prediction_metrics(
        data_original, data_predicted, config=metrics_report_config
    )
    df_atomic_percentage_samples, df_atomic_percentage_summary = (
        utility.calculate_atomic_percentage_split_statistics(
            data_original, data_predicted, config=metrics_report_config
        )
    )
    print(
        "\nMetrics (Train vs. Test):\n",
        utility.format_side_by_side_metrics(
            df_prediction_metrics, config=metrics_report_config
        ).to_string(index=False),
    )
    print(
        "\nAtomic % (Train vs. Test):\n",
        utility.format_side_by_side_atomic_percentages(
            df_atomic_percentage_summary, config=metrics_report_config
        ).to_string(index=False),
    )

    # =========================================================================
    # STAGE 11: DIAGNOSTIC PLOTS
    # =========================================================================
    if show_plots:
        print("\nGenerating demonstration plots...")
        for render_plot_function in (
            lambda: visualization.plot_pairing_timeline(
                df_meta,
                plot_config={"source_tool": source_tool, "target_tool": target_tool},
            ),
            lambda: visualization.plot_flux_and_averaged_intensity_vs_time(
                ary_intensity,
                ary_energy,
                df_meta,
                plot_config={"source_tool": source_tool, "target_tool": target_tool},
            ),
            lambda: visualization.plot_max_intensity_vs_time(
                ary_intensity,
                df_meta,
                region=example_region,
                die=0,
                plot_config={"hue": "tool"},
            ),
            lambda: visualization.plot_normalized_max_intensity_vs_time(
                ary_intensity,
                df_meta,
                ary_energy=ary_energy,
                region=example_region,
                die=0,
                plot_config={"hue": "tool", "normalization_mode": "die_total_flux"},
            ),
            lambda: visualization.plot_tool_comparison(
                ary_energy,
                ary_intensity,
                df_meta,
                compare_config={
                    "tools": (source_tool, target_tool),
                    "region": example_region,
                    "die": 0,
                },
            ),
            lambda: visualization.plot_shirley_background(
                ary_energy,
                ary_intensity,
                df_meta,
                plot_config={"measurement_id": sample_measurement_id, "die": 0},
            ),
            lambda: visualization.plot_training_history(training_results["histories"]),
            lambda: visualization.plot_prediction_comparison(
                data_original,
                data_predicted,
                config={"session_splits": session_splits, "die": 0},
            ),
            lambda: visualization.plot_atomic_percentage_distributions(
                df_atomic_percentage_samples
            ),
            lambda: visualization.plot_atomic_percentage_mae(
                df_atomic_percentage_samples
            ),
        ):
            render_plot_function()
        if use_sliding_window:
            example_spectrum_row = df_meta[
                (df_meta["measurement_id"] == sample_measurement_id)
                & (df_meta["region"] == example_region)
                & (df_meta["die"] == 0)
            ]
            if not example_spectrum_row.empty:
                spectrum_index = int(
                    example_spectrum_row.iloc[0]["spectrum_index"]
                )
                visualization.plot_sliding_window_slices(
                    ary_intensity[spectrum_index],
                    ary_energy[spectrum_index],
                    plot_config={"region": example_region},
                )
        print("Showing plots (close plot windows to finish execution)...")
        plt.show()


if __name__ == "__main__":
    main()