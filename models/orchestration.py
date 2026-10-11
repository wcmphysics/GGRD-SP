"""Main orchestration module executing the unified 8-step spectral
transformation pipeline.

This module coordinates the end-to-end machine learning workflow for
tool-to-tool XPS spectrum transfer across spectral regions:
1. Train / validation / test session partitioning across regions.
2. Sliding window patch sizing and data augmentation preparation.
3. Regional dataset and DataLoader construction.
4. Spectral cost function initialization (MSE + shape + scale + L2).
5. Regional neural network instantiation (ResNet, U-Net, DeepLabV3).
6. Optional Bayesian hyperparameter optimization via Ax.
7. Regional model training, validation, and held-out test evaluation.
8. Standardized full-spectrum prediction across source measurements.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch.nn as nn

from models.bayesian_opt import optimize_model_hyperparameters
from models.cost import CompositeSpectralLoss, NormalizedMSELoss
from models.dataset import (
    SpectrumPatchDataset,
    create_dataloaders,
    partition_measurement_sessions,
    split_session_datasets,
)
from models.deeplabv3 import DeepLabV3
from models.inference import predict_spectra
from models.resnet import ResNet1D
from models.trainer import evaluate, evaluate_sliding_window, train_model_region
from models.unet import ConventionalUNet1D, ResidualUNet1D
from utility.patching import calculate_window_points


def instantiate_model(
    model_type: str,
    seq_len: int,
    config: dict[str, Any] | None = None,
) -> nn.Module:
    """Instantiate a 1D spectral neural network by architecture name
    and input sequence length.

    Parameters
    ----------
    model_type : str
        Architecture name: 'resnet', 'residual_unet', 'unet', or
        'deeplabv3'.
    seq_len : int
        Input sequence length in data points (either full regional
        spectrum length or sliding window patch length).
    config : dict[str, Any] | None, optional
        Dictionary of model architecture hyperparameters (e.g.,
        kernel_size, hidden_channels, base_channels, depth, use_film).

    Returns
    -------
    nn.Module
        Instantiated PyTorch neural network module ready for training.
    """
    model_class_map = {
        "resnet": ResNet1D,
        "residual_unet": ResidualUNet1D,
        "unet": ConventionalUNet1D,
        "deeplabv3": DeepLabV3,
    }
    model_class = model_class_map.get(model_type.lower())
    if model_class is None:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose 'resnet', 'residual_unet', 'unet', or 'deeplabv3'."
        )
    return model_class(n_points=seq_len, config=config or {})


def _resolve_region_config(
    global_config: dict[str, Any],
    region: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve per-region sliding window, training, and Bayesian
    optimization configurations by merging global defaults with
    region-specific overrides.

    Parameters
    ----------
    global_config : dict[str, Any]
        Top-level pipeline configuration dictionary containing global
        defaults and optional 'region_configs' overrides.
    region : str
        Name of the XPS spectral region (e.g., 'Al2p', 'Ti2p').

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
        A 3-tuple of (sliding_window_config, resolved_train_config,
        resolved_bo_config) tailored for the specified region.
    """
    # Extract region-specific override dictionary if configured
    region_configs = global_config.get("region_configs") or global_config.get("region_config") or {}
    if isinstance(region_configs, dict) and region in region_configs:
        region_override = dict(region_configs.get(region, {}))
    else:
        region_override = {}

    # Helper to resolve point-based window/stride settings with region
    # priority over global settings
    def _resolve_points_override(primary_key: str, secondary_key: str) -> int | None:
        override_value = region_override.get(primary_key, region_override.get(secondary_key))
        if override_value is not None:
            return override_value
        return global_config.get(primary_key, global_config.get(secondary_key))

    # 1. Build sliding window calculation configuration
    window_size_points = _resolve_points_override("window_size", "window_size_points")
    stride_points = _resolve_points_override("stride", "sliding_stride_points")
    sliding_window_config = {
        "window_size_ev": region_override.get(
            "window_size_ev", global_config.get("window_size_ev", 2.0),
        ),
        "sliding_stride_ev": region_override.get(
            "sliding_stride_ev", global_config.get("sliding_stride_ev", 1.0),
        ),
        "window_size": window_size_points,
        "stride": stride_points,
        "window_size_points": window_size_points,
        "sliding_stride_points": stride_points,
    }

    # 2. Build regional training configuration by merging global
    # train_config, nested region train_config, and flat overrides
    global_train_config = dict(global_config.get("train_config") or {})
    nested_region_train_config = dict(region_override.get("train_config") or {})
    excluded_keys = {"train_config", "bayesian_opt_config", "window_size_ev", "sliding_stride_ev"}
    flat_region_overrides = {
        key: value for key, value in region_override.items() if key not in excluded_keys
    }
    resolved_train_config = {
        **global_train_config,
        **nested_region_train_config,
        **flat_region_overrides,
    }

    # 3. Build regional Bayesian optimization configuration by merging
    # global bayesian_opt_config, nested overrides, and search bounds
    global_bo_config = dict(global_config.get("bayesian_opt_config") or {})
    nested_region_bo_config = dict(region_override.get("bayesian_opt_config") or {})
    resolved_bo_config = {**global_bo_config, **nested_region_bo_config}
    bo_search_keys = (
        "batch_sizes",
        "kernel_sizes",
        "window_size_choices",
        "lr_bounds",
        "l2_bounds",
        "num_trials",
        "epochs_per_trial",
        "seed",
        "verbose",
        "hidden_channels",
        "base_channels",
        "depths",
        "backbone_channels",
        "aspp_channels",
    )
    for key in bo_search_keys:
        if key in region_override:
            override_val = region_override[key]
            resolved_bo_config[key] = (
                list(override_val) if isinstance(override_val, (list, tuple)) else override_val
            )

    return sliding_window_config, resolved_train_config, resolved_bo_config


def run_spectral_pipeline(
    data: tuple[np.ndarray, np.ndarray, pd.DataFrame] | dict[str, Any] | pd.DataFrame,
    model_type: str = "resnet",
    use_sliding_window: bool = False,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute the unified 8-step end-to-end spectral transformation
    pipeline across all target spectral regions.

    Workflow Steps:
      - Step 1: Partition paired measurements into train, validation,
        and test splits at the measurement-session level so all dies
        and regions from one measurement stay in the same split.
      - Step 2: Resolve per-region sliding window parameters (converting
        eV to data points when needed) and slice training spectra into
        overlapping unpadded patches if sliding window augmentation is
        enabled.
      - Step 3: Build regional train, validation, and test datasets
        with intensity normalization and optional FiLM tool metadata.
      - Step 4: Initialize the regional spectral cost function
        (NormalizedMSELoss or CompositeSpectralLoss with L2 penalty).
      - Step 5: Instantiate the regional 1D neural network architecture
        ('resnet', 'residual_unet', 'unet', or 'deeplabv3').
      - Step 6: Optionally run Ax Bayesian hyperparameter optimization
        to search window size, kernel size, channels, learning rate,
        and L2 regularization weight, rebuilding the patch dataset and
        model if optimal architectural parameters change.
      - Step 7: Train the regional model with early stopping on full
        reconstructed validation spectra, and evaluate final loss on
        the held-out test set.
      - Step 8: Predict transformed target-tool spectra for all source
        measurements across train, validation, and test splits,
        producing a standardized (intensity, energy, metadata) output.

    Parameters
    ----------
    data : tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Canonical 3-tuple of (ary_intensity, ary_energy, meta_df)
        containing regional intensity arrays, binding energy arrays,
        and paired spectrum metadata.
    model_type : str, optional
        Default model architecture name: 'resnet' (default),
        'residual_unet', 'unet', or 'deeplabv3'. Can be overridden
        globally or per region in `config`.
    use_sliding_window : bool, optional
        Whether to apply sliding window patch augmentation by default.
        Can be overridden globally or per region in `config`.
    config : dict[str, Any] | None, optional
        Pipeline configuration dictionary controlling tools, split
        ratios, normalization mode, cost weights, training parameters,
        Bayesian optimization settings, and per-region overrides.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'models': dict mapping each region to its trained model (or
          tuple of (model, window_size, stride) when sliding window is
          active).
        - 'evaluation': dict mapping each region to its best validation
          loss.
        - 'test_evaluation': dict mapping each region to its held-out
          test loss.
        - 'session_splits': dict of ('train', 'val', 'test') session ID
          lists.
        - 'histories': dict mapping each region to its per-epoch
          training and validation loss history.
        - 'bayesian_opt_results': dict mapping each region to its Ax
          Bayesian optimization results (if enabled).
        - 'predictions': canonical 3-tuple of predicted
          (pred_intensity, pred_energy, pred_meta_df) if
          'predict_source' is True, otherwise None.
    """
    # ------------------------------------------------------------------
    # Step 0: Validate input data tuple and unpack global configuration
    # ------------------------------------------------------------------
    if not isinstance(data, (tuple, list)) or len(data) != 3:
        raise ValueError(
            "run_spectral_pipeline requires canonical tuple: (ary_intensity, ary_energy, meta_df)."
        )
    ary_intensity, ary_energy, meta_df = data[0], data[1], data[2]
    if (
        not isinstance(ary_intensity, np.ndarray)
        or not isinstance(ary_energy, np.ndarray)
        or not isinstance(meta_df, pd.DataFrame)
    ):
        raise ValueError(
            "run_spectral_pipeline requires tuple elements of type: (np.array, np.array, pd.DataFrame)."
        )

    pipeline_config = dict(config or {})
    default_model_type = str(pipeline_config.get("model_type", model_type)).lower()
    default_use_sliding_window = bool(
        pipeline_config.get("use_sliding_window", use_sliding_window),
    )
    source_tool = str(pipeline_config.get("source_tool", "J4"))
    target_tool = str(pipeline_config.get("target_tool", "H1"))
    split_ratios = (
        float(pipeline_config.get("train_ratio", 0.5)),
        float(pipeline_config.get("val_ratio", 0.2)),
        float(pipeline_config.get("test_ratio", 0.3)),
    )
    seed = pipeline_config.get("seed", 42)
    verbose = bool(pipeline_config.get("verbose", False))
    default_use_bayesian_opt = bool(pipeline_config.get("use_bayesian_opt", False))
    global_bo_config = dict(pipeline_config.get("bayesian_opt_config", {}))
    if "seed" not in global_bo_config and seed is not None:
        global_bo_config["seed"] = seed
    pipeline_config["bayesian_opt_config"] = global_bo_config

    # Determine which XPS spectral regions to train on
    regions = pipeline_config.get("regions")
    if regions is None:
        paired_mask = meta_df["measurement_id_target"].notna()
        if source_tool:
            paired_mask = paired_mask & (meta_df["tool"] == source_tool)
        regions = sorted(meta_df[paired_mask]["region"].unique().tolist())
    if not regions:
        raise ValueError(
            f"No valid spectral regions found to train for {source_tool=}, {target_tool=}"
        )

    # ------------------------------------------------------------------
    # Step 1: Partition paired measurement sessions into train/val/test
    # ------------------------------------------------------------------
    # Splitting is performed by measurement session ID so all dies and
    # regions from a single physical measurement belong to the same split.
    session_splits = pipeline_config.get("session_splits")
    if session_splits is None:
        partition_config = {
            "source_tool": source_tool,
            "target_tool": target_tool,
            "train_ratio": split_ratios[0],
            "val_ratio": split_ratios[1],
            "test_ratio": split_ratios[2],
            "seed": seed,
        }
        session_splits = partition_measurement_sessions(meta_df, config=partition_config)

    # Initialize containers for per-region outputs
    trained_models: dict[str, Any] = {}
    val_evaluation_results: dict[str, float] = {}
    test_evaluation_results: dict[str, float] = {}
    training_histories: dict[str, dict[str, list[float]]] = {}
    bayesian_opt_results: dict[str, dict[str, Any]] = {}

    # Iterate through each spectral region to train an independent regional model
    for region in regions:
        # Resolve region-specific configuration overrides
        sliding_window_config, region_train_config, region_bo_config = _resolve_region_config(
            pipeline_config, region,
        )
        region_use_sliding_window = bool(
            region_train_config.get("use_sliding_window", default_use_sliding_window),
        )
        region_model_type = str(
            region_train_config.get("model_type", default_model_type),
        ).lower()
        region_use_film = bool(
            region_train_config.get("use_film", pipeline_config.get("use_film", False)),
        )
        region_use_bayesian_opt = bool(
            region_bo_config.get(
                "use_bayesian_opt",
                region_train_config.get("use_bayesian_opt", default_use_bayesian_opt),
            ),
        )

        # --------------------------------------------------------------
        # Step 2 & 3: Build regional full-spectrum datasets & patches
        # --------------------------------------------------------------
        # First construct full-spectrum train, validation, and test
        # datasets for the current region using the shared session splits.
        dataset_split_config = {
            "region": region,
            "source_tool": source_tool,
            "target_tool": target_tool,
            "session_splits": session_splits,
            "return_test": True,
            "normalization_mode": pipeline_config.get(
                "normalization_mode", "source_referenced",
            ),
        }
        train_full_dataset, val_full_dataset, test_full_dataset = split_session_datasets(
            meta_df, ary_intensity, ary_energy, config=dataset_split_config,
        )
        full_spectrum_length = len(train_full_dataset[0]["x"])

        # If sliding window augmentation is enabled, convert window size
        # and stride (in eV or points) into integer point counts and
        # slice the training dataset into overlapping patches.
        # Validation and test datasets remain full spectra so evaluation
        # is always performed on reconstructed full regional spectra.
        if region_use_sliding_window:
            sample_energy_tensor = train_full_dataset[0].get("energy")
            if sample_energy_tensor is not None:
                energy_grid = sample_energy_tensor.cpu().numpy()
            else:
                energy_grid = np.linspace(0.0, 10.0, full_spectrum_length)

            window_size, stride = calculate_window_points(
                energy_grid, config=sliding_window_config,
            )
            window_size = min(full_spectrum_length, max(3, window_size))
            stride = max(1, min(window_size, stride))
            input_sequence_length = window_size
            train_dataset: Any = SpectrumPatchDataset(
                train_full_dataset, window_size=window_size, stride=stride,
            )
        else:
            window_size = full_spectrum_length
            stride = full_spectrum_length
            input_sequence_length = full_spectrum_length
            train_dataset = train_full_dataset

        # --------------------------------------------------------------
        # Step 4: Initialize the spectral cost function (Loss + L2)
        # --------------------------------------------------------------
        # Supports standard NormalizedMSELoss (normalizing by the max of
        # the true target spectrum) or CompositeSpectralLoss combining
        # point-wise MSE, first-derivative shape loss, and area scale loss.
        l2_weight = float(region_train_config.get("l2_weight", 1e-4))
        cost_type = str(
            region_train_config.get(
                "cost_type", pipeline_config.get("cost_type", "normalized_mse"),
            ),
        ).lower()
        loss_kwargs = {
            "l2_weight": l2_weight,
            "region_weights": pipeline_config.get("region_weights"),
        }
        if "composite" in cost_type:
            loss_class = CompositeSpectralLoss
            default_w_mse, default_w_shape = 0.5, 0.5
        else:
            loss_class = NormalizedMSELoss
            default_w_mse, default_w_shape = 1.0, 0.0

        criterion = loss_class(
            w_mse=float(
                region_train_config.get("w_mse", pipeline_config.get("w_mse", default_w_mse)),
            ),
            w_shape=float(
                region_train_config.get(
                    "w_shape", pipeline_config.get("w_shape", default_w_shape),
                ),
            ),
            w_scale=float(
                region_train_config.get("w_scale", pipeline_config.get("w_scale", 0.0)),
            ),
            **loss_kwargs,
        )

        # --------------------------------------------------------------
        # Step 5: Instantiate the regional neural network model
        # --------------------------------------------------------------
        region_train_config.update(
            {
                "model_type": region_model_type,
                "use_sliding_window": region_use_sliding_window,
                "window_size": window_size,
                "stride": stride,
                "n_points": input_sequence_length,
                "use_film": region_use_film,
            }
        )
        model = instantiate_model(
            region_model_type, seq_len=input_sequence_length, config=region_train_config,
        )
        region_train_config.update({"model": model, "criterion": criterion})

        # --------------------------------------------------------------
        # Step 6: Bayesian hyperparameter optimization (if enabled)
        # --------------------------------------------------------------
        # Uses Ax/BoTorch to search optimal architectural and training
        # hyperparameters on the validation set. If a new window_size or
        # architecture parameter is selected, rebuild the patch dataset
        # and re-instantiate the model with the optimal configuration.
        if region_use_bayesian_opt:
            region_bo_config.update(
                {
                    "model_type": region_model_type,
                    "use_sliding_window": region_use_sliding_window,
                    "window_size": window_size,
                    "stride": stride,
                    "criterion": criterion,
                }
            )
            bo_input_train_dataset = (
                train_full_dataset if region_use_sliding_window else train_dataset
            )
            bo_result = optimize_model_hyperparameters(
                bo_input_train_dataset, val_full_dataset, config=region_bo_config,
            )
            bayesian_opt_results[region] = bo_result
            best_parameters = bo_result.get("best_parameters", {})
            if best_parameters:
                region_train_config.update(best_parameters)

                # If Bayesian optimization tuned the sliding window size,
                # update window_size, stride, and regenerate training patches
                if region_use_sliding_window and "window_size" in best_parameters:
                    window_size = int(best_parameters["window_size"])
                    stride = max(1, window_size // 2)
                    input_sequence_length = window_size
                    train_dataset = SpectrumPatchDataset(
                        train_full_dataset, window_size=window_size, stride=stride,
                    )
                    region_train_config.update({"window_size": window_size, "stride": stride})

                # Re-instantiate the model if any architectural hyperparameter
                # was modified during the Bayesian optimization search
                architecture_keys = (
                    "kernel_size",
                    "hidden_channels",
                    "base_channels",
                    "depth",
                    "backbone_channels",
                    "aspp_channels",
                    "window_size",
                )
                if any(key in best_parameters for key in architecture_keys):
                    model = instantiate_model(
                        region_model_type,
                        seq_len=input_sequence_length,
                        config=region_train_config,
                    )
                    region_train_config["model"] = model

        # --------------------------------------------------------------
        # Step 7: Regional model training, validation, & test evaluation
        # --------------------------------------------------------------
        # Train the regional model with early stopping monitored on the
        # validation dataset (using full-spectrum reconstruction when
        # sliding window is enabled).
        if verbose:
            region_train_config["verbose"] = True
        training_result = train_model_region(
            train_dataset, val_full_dataset, config=region_train_config,
        )
        trained_model = training_result["model"]

        # Store either (model, window_size, stride) tuple for sliding
        # window inference, or the standalone model for full-spectrum mode
        if region_use_sliding_window:
            trained_models[region] = (trained_model, window_size, stride)
        else:
            trained_models[region] = trained_model

        val_evaluation_results[region] = float(training_result["best_val_loss"])
        training_histories[region] = training_result["history"]

        # Evaluate the trained regional model on the held-out test split
        if len(test_full_dataset) > 0:
            if region_use_sliding_window:
                test_loss = evaluate_sliding_window(
                    trained_model,
                    test_full_dataset,
                    criterion=criterion,
                    config={"window_size": window_size, "stride": stride},
                )
            else:
                _, test_loader = create_dataloaders(
                    test_full_dataset,
                    test_full_dataset,
                    config={
                        "batch_size": region_train_config.get("batch_size", 16),
                        "shuffle_train": False,
                    },
                )
                test_loss = evaluate(trained_model, test_loader, criterion)
            test_evaluation_results[region] = float(test_loss)

    # ------------------------------------------------------------------
    # Step 8: Predict transformed spectra across all source measurements
    # ------------------------------------------------------------------
    # Apply the trained regional models to transform source-tool spectra
    # into predicted target-tool spectra across train, val, and test sets.
    predictions = None
    if pipeline_config.get("predict_source", True):
        prediction_config = {
            "source_tool": source_tool,
            "target_tool": target_tool,
            "session_splits": session_splits,
            "normalization_mode": pipeline_config.get(
                "normalization_mode", "source_referenced",
            ),
            "clamp_non_negative": True,
        }
        predictions = predict_spectra(
            models=trained_models,
            data=(ary_intensity, ary_energy, meta_df),
            config=prediction_config,
        )

    return {
        "models": trained_models,
        "evaluation": val_evaluation_results,
        "test_evaluation": test_evaluation_results,
        "session_splits": session_splits,
        "histories": training_histories,
        "bayesian_opt_results": bayesian_opt_results,
        "predictions": predictions,
    }
