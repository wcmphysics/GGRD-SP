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
    
    Availabe model_type: resnet, resnet_5l, resnet_scaled_short,
    residual_unet, unet, deeplabv3
    """
    # =========================================================================
    # GLOBAL SETUP & CONFIGURATIONS (Loaded from config/default_config.yaml)
    # =========================================================================
    config = utility.load_default_config("config/default_config.yaml")
    materials = utility.load_material_elements(config)
    element_vocab = utility.extract_element_vocabulary(materials)

    global_cfg = config.get("global", {})
    model_type = global_cfg.get("model_type", "resnet")
    use_sliding_window = global_cfg.get("use_sliding_window", False)
    use_film = global_cfg.get("use_film", False)
    unified_model = global_cfg.get("unified_model", True)
    source_tool = global_cfg.get("source_tool", "J4")
    target_tool = global_cfg.get("target_tool", "J5")
    example_region = global_cfg.get("example_region", "Ti2p")
    sample_measurement_id = global_cfg.get("sample_measurement_id", "NMG_M_J4_00000")
    show_plots = "--no-plots" not in sys.argv and "--headless" not in sys.argv

    # Stage 1: Data generation configuration
    data_generation_config = config["data_generation"]

    # Stage 2: 1-to-1 temporal pairing configuration
    pairing_config = {
        **config["pairing"],
        "source_tool": source_tool,
        "target_tool": target_tool,
    }

    # Stage 3: Normalization configuration
    normalization_config = config["normalization"]

    # Stage 4: Train-val-test measurement session splitting configuration
    splitting_config = {
        **config["splitting"],
        "source_tool": source_tool,
        "target_tool": target_tool,
    }

    # Stage 4B: Gaussian Process die flux prediction configuration
    gp_flux_config = config["gp_flux"]

    # Stage 5: Dataset & patch augmentation configuration
    dataset_config = {
        **config["dataset"],
        "regions": data_generation_config["regions"],
        "source_tool": source_tool,
        "target_tool": target_tool,
        "use_sliding_window": use_sliding_window,
    }

    # Stage 6: Cost function configuration
    cost_config = config["cost"]

    # Stage 6B: Gaussian Process spectral noise analysis configuration
    noise_analysis_config = {
        **config["noise_analysis"],
        "regions": [example_region],
    }
    noise_plot_config = {
        **config["noise_plot"],
        "example": {
            "measurement_id": sample_measurement_id,
            "region": example_region,
        },
    }

    # Dedicated Feature-wise Linear Modulation (FiLM) configuration
    film_config = {
        **config["film"],
        "use_film": use_film,
        "num_regions": len(data_generation_config["regions"]),
        "material_elements": materials,
        "element_vocab": element_vocab,
    }

    # Stage 7A: Bayesian optimization configuration (plural search keys)
    bayesian_opt_config = {
        **config["bayesian_opt"],
        "learning_rate_bounds": tuple(
            config["bayesian_opt"]["learning_rate_bounds"]
        ),
        "l2_bounds": tuple(config["bayesian_opt"]["l2_bounds"]),
    }

    # Stage 7B: Neural network training configuration (singular keys)
    training_config = {
        "model_type": model_type,
        "unified_model": unified_model,
        "use_film": film_config["use_film"],
        "film_config": film_config,
        "train_config": config["training"]["train_config"],
        "region_configs": config["training"]["region_configs"],
    }

    # Stage 8: Model prediction configuration
    prediction_config = {
        **config["prediction"],
        "source_tool": source_tool,
        "target_tool": target_tool,
        "film_config": film_config,
    }

    mse_metric_name = (
        "MSE_normalized_by_target"
        if cost_config.get("normalized_by_target_max", True)
        else "MSE"
    )
    metrics_report_config = {
        **config["metrics_report"],
        "metrics": [mse_metric_name, *config["metrics_report"]["metrics"]],
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

    # Calculate hierarchical physical flux and intensity metrics
    df_meta = utility.calculate_flux_and_intensity(
        ary_intensity, ary_energy, df_meta
    )
    print(
        f"Calculated spectral metrics: "
        f"mean regional intensity={df_meta['mean_intensity_single_region'].mean():.2f}, "
        f"mean die intensity={df_meta['mean_intensity_single_die'].mean():.2f}, "
        f"mean regional flux={df_meta['total_flux_single_region'].mean():.2f}, "
        f"mean die flux={df_meta['total_flux_single_die'].mean():.2f}"
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

    # Establish fixed region mapping indices
    unique_regions = sorted(df_meta["region"].dropna().astype(str).unique())
    region_index_mapping = {
        region_name: region_index
        for region_index, region_name in enumerate(unique_regions)
    }
    df_meta["region_index"] = df_meta["region"].map(region_index_mapping)
    training_config["region_index_mapping"] = region_index_mapping
    prediction_config["region_index_mapping"] = region_index_mapping

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
    df_meta, _, session_splits = utility.split_measurement_sessions(
        df_meta, splitting_config
    )
    print(
        f"Session splits: train={len(session_splits['train'])}, "
        f"val={len(session_splits['val'])}, test={len(session_splits['test'])}"
    )

    # Standardize time and die flux metadata strictly on the training split
    df_meta = utility.standardize_metadata_features(
        df_meta, config={"fit_split": "train"}
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
    # STAGE 6B: GAUSSIAN PROCESS SPECTRAL NOISE ANALYSIS
    # =========================================================================
    print("\n=== Stage 6B: Gaussian Process Noise Analysis ===")
    ary_spec_analysis_signal, ary_spec_analysis_noise, df_meta = (
        models.perform_spectral_noise_analysis(
            ary_intensity, ary_energy, df_meta, noise_analysis_config
        )
    )
    print(
        f"Analysed {ary_spec_analysis_signal.shape[0]} regional spectra; "
        f"mean noise sigma={df_meta['gp_noise_sigma'].mean():.4g}, "
        f"mean SNR={df_meta['gp_snr'].mean():.4g}"
    )

    # =========================================================================
    # STAGE 7: NEURAL NETWORK TRAINING AND BAYESIAN OPTIMIZATION
    # =========================================================================
    print(f"\n=== Stage 7: Neural Model Training ({model_type}) ===")
    training_results = models.perform_model_training_and_bayesian_optimization(
        regional_datasets,
        criterion,
        config=training_config,
        bayesian_opt_config=bayesian_opt_config,
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
            # lambda: visualization.plot_flux_and_averaged_intensity_vs_time(
            #     ary_intensity,
            #     ary_energy,
            #     df_meta,
            #     plot_config={
            #         "source_tool": source_tool, "target_tool": target_tool
            #     },
            # ),
            # lambda: visualization.plot_max_intensity_vs_time(
            #     ary_intensity,
            #     df_meta,
            #     region=example_region,
            #     die=0,
            #     plot_config={"hue": "tool"},
            # ),
            # lambda: visualization.plot_normalized_max_intensity_vs_time(
            #     ary_intensity,
            #     df_meta,
            #     ary_energy=ary_energy,
            #     region=example_region,
            #     die=0,
            #     plot_config={
            #         "hue": "tool", "normalization_mode": "die_total_flux"
            #     },
            # ),
            # lambda: visualization.plot_tool_comparison(
            #     ary_energy,
            #     ary_intensity,
            #     df_meta,
            #     compare_config={
            #         "tools": (source_tool, target_tool),
            #         "region": example_region,
            #         "die": 0,
            #     },
            # ),
            # lambda: visualization.plot_shirley_background(
            #     ary_energy,
            #     ary_intensity,
            #     df_meta,
            #     plot_config={
            #         "measurement_id": sample_measurement_id, "die": 0
            #     },
            # ),
            lambda: visualization.plot_spectral_noise_analysis(
                (
                    ary_intensity,
                    ary_energy,
                    df_meta,
                    ary_spec_analysis_signal,
                    ary_spec_analysis_noise,
                ),
                noise_plot_config,
            ),
            lambda: visualization.plot_gp_hyperparameter_convergence(
                data_original,
                plot_config={
                    "measurement_id": sample_measurement_id,
                    "die": 0,
                    "save_path": "gp_hyperparameter_convergence.png",
                },
            ),
            lambda: visualization.plot_training_history(training_results["histories"]),
            lambda: visualization.plot_prediction_comparison(
                data_original,
                data_predicted,
                config={"session_splits": session_splits, "die": 0},
            ),
            lambda: visualization.plot_regional_spectrum_distribution(
                data_original,
                data_predicted,
                config={
                    "split": "test",
                    "save_path": "regional_spectrum_distribution.png",
                },
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