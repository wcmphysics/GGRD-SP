"""Main execution pipeline for GGRD-SP.

Orchestrates sequential end-to-end XPS spectral transfer stages:
1. Synthetic spectrum reading/generation.
2. Temporal pairing of source and target measurements.
3. Train, validation, and test session-level splitting.
4. Spectral noise and SNR analysis and regional spectrum de-noising.
5. Pre-processing: physical metric calculation, tool and region mapping,
   feature standardization, and spectral normalization.
6. Die flux prediction and baseline drift modeling.
7. Regional dataset definition and sliding window patch augmentation.
8. Multi-component spectral loss preparation.
9. Neural network model training and Bayesian hyperparameter optimization.
10. Cross-tool transformed spectrum prediction.
11. Quantitative model evaluation across train and test partitions.
12. Shirley background subtraction and atomic percentage quantification.
13. Diagnostic visualization dispatching.
"""

from __future__ import annotations

import sys
import matplotlib.pyplot as plt
import torch

import denoise
import models
import utility
import visualization


def main() -> None:
    """Execute sequential key stages: generation, pairing, splitting,
    noise analysis and denoising, preprocessing, die flux prediction, dataset
    construction, cost preparation, training, prediction, evaluation,
    quantification, and visualization.

    Available model_type: resnet, resnet_scaled_short, residual_unet, unet,
    deeplabv3.
    """
    # =========================================================================
    # GLOBAL SETUP & CONFIGURATIONS (Loaded and resolved from YAML)
    # =========================================================================
    config = utility.load_config("config/config.yaml")
    show_plots = "--no-plots" not in sys.argv and "--headless" not in sys.argv

    if hasattr(torch.sparse, "check_sparse_tensor_invariants"):
        torch.sparse.check_sparse_tensor_invariants.enable()

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

    # =========================================================================
    # STAGE 3: TRAIN-VAL-TEST DATA SPLITTING
    # =========================================================================
    print("\n=== Stage 3: Train-Val-Test Session Splitting ===")
    df_meta, _, session_splits = utility.split_measurement_sessions(
        df_meta, config["splitting"]
    )
    print(
        f"Session splits: train={len(session_splits['train'])}, "
        f"val={len(session_splits['val'])}, test={len(session_splits['test'])}"
    )

    # =========================================================================
    # STAGE 4: NOISE ANALYSIS AND DE-NOISING
    # =========================================================================
    print("\n=== Stage 4: Noise Analysis & Regional De-noising ===")
    ary_spec_analysis_signal, ary_spec_analysis_noise, df_meta = (
        denoise.perform_spectral_noise_analysis(
            ary_intensity, ary_energy, df_meta, config["noise_analysis"]
        )
    )
    print(
        f"Analysed {ary_spec_analysis_signal.shape[0]} regional spectra; "
        f"mean noise sigma={df_meta['gp_noise_sigma'].mean():.4g}, "
        f"mean SNR={df_meta['gp_snr'].mean():.4g}"
    )

    ary_intensity, ary_intensity_non_denoised, ary_intensity_denoised = (
        denoise.denoise_regional_spectra(
            ary_intensity, ary_energy, df_meta, config["denoise"]
        )
    )
    df_denoise_atomic_samples, df_denoise_atomic_summary = (
        denoise.compare_denoised_atomic_percentages(
            ary_energy,
            (ary_intensity_non_denoised, ary_intensity_denoised),
            df_meta,
            config["denoise"],
        )
    )
    print(
        f"Denoised regions={config['denoise']['regions']} "
        f"(method={config['denoise']['method']}, "
        f"apply_denoise={config['denoise']['apply_denoise']})"
    )
    if not df_denoise_atomic_summary.empty:
        print(
            "\nDenoised vs. Non-Denoised Atomic % Comparison:\n",
            df_denoise_atomic_summary.to_string(index=False),
        )

    # =========================================================================
    # STAGE 5: PRE-PROCESSING
    # =========================================================================
    print("\n=== Stage 5: Pre-processing ===")
    df_meta, _, _ = utility.preprocess(
        ary_intensity, ary_energy, df_meta, config=config
    )

    print(
        f"Calculated spectral metrics: "
        f"mean regional intensity={df_meta['mean_intensity_single_region'].mean():.2f}, "
        f"mean die intensity={df_meta['mean_intensity_single_die'].mean():.2f}, "
        f"mean regional flux={df_meta['total_flux_single_region'].mean():.2f}, "
        f"mean die flux={df_meta['total_flux_single_die'].mean():.2f}"
    )

    ary_intensity, ary_energy, df_meta = utility.normalize_spectral_data(
        ary_intensity, ary_energy, df_meta, config["normalization"]
    )
    print(
        f"Normalized intensity by {config['normalization']['normalization_mode']}; "
        f"max={ary_intensity.max():.4f}"
    )

    # =========================================================================
    # STAGE 6: DIE FLUX PREDICTION
    # =========================================================================
    print("\n=== Stage 6: Die Flux Prediction ===")
    gp_bundle, df_gp_metrics = models.run_gp_flux_pipeline(
        df_meta, config["gp_flux"]
    )
    print("\nDie Flux Error Performance Summary:")
    print(df_gp_metrics.to_string(index=False))

    # =========================================================================
    # STAGE 7: DATASET AND DATALOADER DEFINITION (With Augmentation)
    # =========================================================================
    print("\n=== Stage 7: Regional Dataset Definition & Patch Augmentation ===")
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
    # STAGE 8: COST FUNCTION DEFINITION AND PREPARATION
    # =========================================================================
    print("\n=== Stage 8: Cost Function Preparation ===")
    criterion = models.SpectralLoss(config=config["cost"])
    print(f"Initialized criterion: {criterion.__class__.__name__}")

    # =========================================================================
    # STAGE 9: NEURAL NETWORK TRAINING AND BAYESIAN OPTIMIZATION
    # =========================================================================
    model_type = config["training"]["model_type"]
    print(f"\n=== Stage 9: Neural Model Training ({model_type}) ===")
    training_results = models.perform_model_training_and_bayesian_optimization(
        regional_datasets,
        criterion,
        config=config["training"],
        bayesian_opt_config=config["bayesian_opt"],
    )
    print("Model training completed across all regions.")

    # =========================================================================
    # STAGE 10: PREDICTION FROM MODEL
    # =========================================================================
    print("\n=== Stage 10: Transformed Spectrum Prediction ===")
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
    # STAGE 11: QUANTITATIVE MODEL EVALUATION
    # =========================================================================
    print("\n=== Stage 11: Quantitative Model Evaluation ===")
    df_prediction_metrics, _ = utility.calculate_prediction_metrics(
        data_original, data_predicted, config=config["metrics_report"]
    )
    print(
        "\nMetrics (Train vs. Test):\n",
        utility.format_side_by_side_metrics(
            df_prediction_metrics, config=config["metrics_report"]
        ).to_string(index=False),
    )

    # =========================================================================
    # STAGE 12: ATOMIC PERCENTAGE QUANTIFICATION
    # =========================================================================
    print("\n=== Stage 12: Atomic % Quantification ===")
    df_atomic_percentage_samples, df_atomic_percentage_summary = (
        utility.calculate_atomic_percentage_split_statistics(
            data_original, data_predicted, config=config["metrics_report"]
        )
    )
    print(
        "\nAtomic % (Train vs. Test):\n",
        utility.format_side_by_side_atomic_percentages(
            df_atomic_percentage_summary, config=config["metrics_report"]
        ).to_string(index=False),
    )

    # =========================================================================
    # STAGE 13: DIAGNOSTIC PLOTS (Via Visualization Dispatcher)
    # =========================================================================
    if show_plots:
        print("\n=== Stage 13: Diagnostic Visualization Dispatching ===")
        print("Generating demonstration plots via dispatcher...")
        pipeline_plot_data = {
            "df_meta": df_meta,
            "ary_intensity": ary_intensity,
            "ary_energy": ary_energy,
            "data_original": data_original,
            "data_predicted": data_predicted,
            "training_results": training_results,
            "df_atomic_percentage_samples": df_atomic_percentage_samples,
            "gp_bundle": gp_bundle,
            "gp_die_flux_bundle": gp_bundle,
            "gp_flux_config": config.get("gp_flux"),
            "noise_analysis": (
                ary_spec_analysis_signal,
                ary_spec_analysis_noise,
            ),
            "denoise_intensity_pair": (
                ary_intensity_non_denoised,
                ary_intensity_denoised,
            ),
            "df_denoise_atomic_samples": df_denoise_atomic_samples,
        }
        dispatch_cfg = {
            **config["visualization"],
            "show_plots": show_plots,
            "session_splits": session_splits,
            "gp_flux_config": config.get("gp_flux"),
        }
        visualization.dispatch_visualizations(
            pipeline_plot_data, config=dispatch_cfg
        )
        print("Showing plots (close plot windows to finish execution)...")
        plt.show()


if __name__ == "__main__":
    main()
