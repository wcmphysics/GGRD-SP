"""Main execution pipeline for GGRD-SP.

Orchestrates sequential end-to-end XPS spectral transfer stages:
1. Synthetic spectrum reading/generation and physical metric extraction.
2. Temporal pairing of source and target measurements.
3. Spectrum normalization across dies and regions.
4. Train, validation, and test session-level splitting.
4B. Exact Gaussian Process regression for die-level flux modeling.
5. Regional dataset definition and sliding window patching augmentation.
6. Multi-component spectral loss preparation.
6B. Gaussian Process spectral noise and SNR analysis.
7. Neural network model training and Bayesian hyperparameter optimization.
8. Cross-tool transformed spectrum prediction.
9. Side-by-side metric evaluation across train and test partitions.
10. Shirley background subtraction and atomic percentage quantification.
11. Diagnostic visualization dispatching.
"""

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
    
    Available model_type: resnet, resnet_5l, resnet_scaled_short,
    residual_unet, unet, deeplabv3.
    """
    # =========================================================================
    # GLOBAL SETUP & CONFIGURATIONS (Loaded and resolved from YAML)
    # =========================================================================
    config = utility.load_config("config/config.yaml")
    show_plots = "--no-plots" not in sys.argv and "--headless" not in sys.argv

    # =========================================================================
    # STAGE 1: SPECTRUM DATA READING OR GENERATION
    # =========================================================================
    print("=== Stage 1: Data Generation ===")
    ary_intensity, ary_energy, df_meta = utility.generate_pseudo_measurements(
        config["data_generation"]
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
    df_meta = utility.pair_source_target_spectra(df_meta, config["pairing"])
    source_tool = str(config["pairing"]["source_tool"])
    df_source_tool = df_meta[df_meta["tool"] == source_tool]
    print(
        f"Paired sessions: {df_source_tool['target_measurement_id'].dropna().nunique()}"
        f"/{df_source_tool['measurement_id'].nunique()}"
    )

    # Establish fixed tool mapping indices
    unique_tools = sorted(
        set(df_meta["tool"].dropna().astype(str).unique())
        | set(df_meta["target_tool"].dropna().astype(str).unique())
    )
    tool_index_mapping = {
        tool_name: tool_index for tool_index, tool_name in enumerate(unique_tools)
    }
    df_meta["source_tool_index"] = df_meta["tool"].map(tool_index_mapping)
    df_meta["target_tool_index"] = df_meta["target_tool"].map(tool_index_mapping)
    config["training"]["num_tools"] = len(tool_index_mapping)

    # Establish fixed region mapping indices
    unique_regions = sorted(df_meta["region"].dropna().astype(str).unique())
    region_index_mapping = {
        region_name: region_index
        for region_index, region_name in enumerate(unique_regions)
    }
    df_meta["region_index"] = df_meta["region"].map(region_index_mapping)
    config["training"]["region_index_mapping"] = region_index_mapping
    config["prediction"]["region_index_mapping"] = region_index_mapping

    # =========================================================================
    # STAGE 3: NORMALIZATION OF SPECTRAL DATA
    # =========================================================================
    print("\n=== Stage 3: Normalization ===")
    ary_intensity, ary_energy, df_meta = utility.normalize_spectral_data(
        ary_intensity, ary_energy, df_meta, config["normalization"]
    )
    print(
        f"Normalized intensity by {config['normalization']['normalization_mode']}; "
        f"max={ary_intensity.max():.4f}"
    )

    # =========================================================================
    # STAGE 4: TRAIN-VAL-TEST DATA SPLITTING
    # =========================================================================
    print("\n=== Stage 4: Train-Val-Test Session Splitting ===")
    df_meta, _, session_splits = utility.split_measurement_sessions(
        df_meta, config["splitting"]
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
        df_meta, config["gp_flux"]
    )
    print("\nGP Die Flux Error Performance Summary:")
    print(df_gp_metrics.to_string(index=False))

    if show_plots and config["visualization"].get("plots", {}).get(
        "gp_die_flux_timeline", True
    ):
        visualization.plot_gp_die_flux_timeline(gp_bundle, config["gp_flux"])

    # =========================================================================
    # STAGE 5: DATASET AND DATALOADER DEFINITION (With Augmentation)
    # =========================================================================
    print("\n=== Stage 5: Regional Dataset Definition & Patch Augmentation ===")
    regional_datasets = models.create_regional_datasets(
        df_meta, ary_intensity, ary_energy, config=config["dataset"]
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
    criterion = models.SpectralLoss(config=config["cost"])
    print(f"Initialized criterion: {criterion.__class__.__name__}")

    # =========================================================================
    # STAGE 6B: GAUSSIAN PROCESS SPECTRAL NOISE ANALYSIS
    # =========================================================================
    print("\n=== Stage 6B: Gaussian Process Noise Analysis ===")
    ary_spec_analysis_signal, ary_spec_analysis_noise, df_meta = (
        models.perform_spectral_noise_analysis(
            ary_intensity, ary_energy, df_meta, config["noise_analysis"]
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
    model_type = config["training"]["model_type"]
    print(f"\n=== Stage 7: Neural Model Training ({model_type}) ===")
    training_results = models.perform_model_training_and_bayesian_optimization(
        regional_datasets,
        criterion,
        config=config["training"],
        bayesian_opt_config=config["bayesian_opt"],
    )
    print("Model training completed across all regions.")

    # =========================================================================
    # STAGE 8: PREDICTION FROM MODEL
    # =========================================================================
    print("\n=== Stage 8: Transformed Spectrum Prediction ===")
    config["prediction"]["session_splits"] = session_splits
    data_original = (ary_intensity, ary_energy, df_meta)
    data_predicted = models.predict_spectra(
        models=training_results["models"],
        data=data_original,
        config=config["prediction"],
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
        data_original, data_predicted, config=config["metrics_report"]
    )
    df_atomic_percentage_samples, df_atomic_percentage_summary = (
        utility.calculate_atomic_percentage_split_statistics(
            data_original, data_predicted, config=config["metrics_report"]
        )
    )
    print(
        "\nMetrics (Train vs. Test):\n",
        utility.format_side_by_side_metrics(
            df_prediction_metrics, config=config["metrics_report"]
        ).to_string(index=False),
    )
    print(
        "\nAtomic % (Train vs. Test):\n",
        utility.format_side_by_side_atomic_percentages(
            df_atomic_percentage_summary, config=config["metrics_report"]
        ).to_string(index=False),
    )

    # =========================================================================
    # STAGE 11: DIAGNOSTIC PLOTS (Via Visualization Dispatcher)
    # =========================================================================
    if show_plots:
        print("\nGenerating demonstration plots via dispatcher...")
        pipeline_plot_data = {
            "df_meta": df_meta,
            "ary_intensity": ary_intensity,
            "ary_energy": ary_energy,
            "data_original": data_original,
            "data_predicted": data_predicted,
            "training_results": training_results,
            "df_atomic_percentage_samples": df_atomic_percentage_samples,
            "noise_analysis": (
                ary_spec_analysis_signal,
                ary_spec_analysis_noise,
            ),
        }
        dispatch_cfg = {
            **config["visualization"],
            "show_plots": show_plots,
            "session_splits": session_splits,
            "plots": {
                **config["visualization"].get("plots", {}),
                "gp_die_flux_timeline": False,  # Already plotted in Stage 4
            },
        }
        visualization.dispatch_visualizations(
            pipeline_plot_data, config=dispatch_cfg
        )
        print("Showing plots (close plot windows to finish execution)...")
        plt.show()


if __name__ == "__main__":
    main()
