"""Model training and Bayesian optimization orchestration module for GGRD-SP."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from models.bayesian_opt import optimize_model_hyperparameters
from models.dataset import SpectrumPatchDataset
from models.deeplabv3 import DeepLabV3
from models.resnet import ResNet1D
from models.trainer import train_model_region
from models.unet import ConventionalUNet1D, ResidualUNet1D


def instantiate_model(
    model_type: str,
    seq_len: int,
    config: dict[str, Any] | None = None,
) -> nn.Module:
    """Instantiate a 1D spectral neural network by architecture name and input sequence length.

    Parameters
    ----------
    model_type : str
        Architecture name: 'resnet', 'residual_unet', 'unet', or 'deeplabv3'.
    seq_len : int
        Input sequence length in data points (full regional spectrum or patch length).
    config : dict[str, Any] | None, optional
        Dictionary of model architecture hyperparameters (e.g., kernel_size,
        resnet_channel, residual_unet_channel, unet_channel, depth, use_film).

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
    bayesian_opt_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve per-region sliding window, training, and Bayesian optimization configurations.

    Parameters
    ----------
    global_config : dict[str, Any]
        Top-level configuration dictionary containing global defaults and optional
        'region_configs' overrides.
    region : str
        Name of the XPS spectral region (e.g., 'Al2p', 'Ti2p').
    bayesian_opt_config : dict[str, Any] | None, optional
        Dedicated Bayesian optimization configuration dictionary.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
        3-tuple of (sliding_window_config, resolved_train_config, resolved_bo_config).
    """
    region_configs = global_config.get("region_configs") or global_config.get("region_config") or {}
    region_override = (
        dict(region_configs.get(region, {}))
        if isinstance(region_configs, dict) and region in region_configs
        else {}
    )

    def _resolve_points_override(primary_key: str, secondary_key: str) -> int | None:
        override_value = region_override.get(primary_key, region_override.get(secondary_key))
        if override_value is not None:
            return override_value
        return global_config.get(primary_key, global_config.get(secondary_key))

    window_size_points = _resolve_points_override("window_size", "window_size_points")
    stride_points = _resolve_points_override("stride", "sliding_stride_points")
    sliding_window_config = {
        "window_size_ev": region_override.get(
            "window_size_ev", global_config.get("window_size_ev", 6.0),
        ),
        "sliding_stride_ev": region_override.get(
            "sliding_stride_ev", global_config.get("sliding_stride_ev", 2.0),
        ),
        "window_size": window_size_points,
        "stride": stride_points,
        "window_size_points": window_size_points,
        "sliding_stride_points": stride_points,
    }

    container_keys = {
        "train_config",
        "bayesian_opt_config",
        "film_config",
        "region_configs",
        "region_config",
        "window_size_ev",
        "sliding_stride_ev",
    }
    top_level_train_defaults = {
        key: value for key, value in global_config.items() if key not in container_keys
    }
    global_train_config = dict(global_config.get("train_config") or {})
    nested_region_train_config = dict(region_override.get("train_config") or {})
    flat_region_overrides = {
        key: value for key, value in region_override.items() if key not in container_keys
    }
    resolved_train_config = {
        **top_level_train_defaults,
        **global_train_config,
        **nested_region_train_config,
        **flat_region_overrides,
    }

    base_bo_cfg = (
        bayesian_opt_config
        if bayesian_opt_config is not None
        else (global_config.get("bayesian_opt_config") or {})
    )
    global_bo_config = dict(base_bo_cfg)
    nested_region_bo_config = dict(region_override.get("bayesian_opt_config") or {})
    resolved_bo_config = {**global_bo_config, **nested_region_bo_config}

    return sliding_window_config, resolved_train_config, resolved_bo_config


def perform_model_training_and_bayesian_optimization(
    regional_datasets: dict[str, dict[str, Any]],
    criterion: nn.Module,
    config: dict[str, Any] | None = None,
    bayesian_opt_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train regional spectral neural networks and optimize hyperparameters via Bayesian search.

    Parameters
    ----------
    regional_datasets : dict[str, dict[str, Any]]
        Dictionary mapping region names to their dataset bundles:
        - 'dataset_train': training dataset (SpectrumPatchDataset if sliding window on).
        - 'dataset_val': validation full-spectrum dataset.
        - 'dataset_train_full': unpatched full-spectrum training dataset.
        - 'window_size_points': window size in data points.
        - 'sliding_stride_points': sliding stride in data points.
        - 'full_spectrum_length': full spectrum sequence length.
    criterion : nn.Module
        Spectral loss criterion for model training.
    config : dict[str, Any] | None, optional
        Training configuration dictionary (singular hyperparameter keys):
        - 'model_type': architecture name ('resnet', 'residual_unet', 'unet', 'deeplabv3').
        - 'use_film': whether to enable FiLM tool metadata conditioning (default False).
        - 'train_config': training parameters (epochs, learning_rate, batch_size, etc.).
        - 'region_configs': optional per-region overrides.
        - 'verbose': verbosity toggle (default False).
    bayesian_opt_config : dict[str, Any] | None, optional
        Dedicated Bayesian optimization configuration dictionary (plural search keys):
        - 'use_bayesian_opt': whether to execute Ax Bayesian optimization (default False).
        - 'num_trials', 'epochs_per_trial', 'batch_sizes', 'window_size_choices',
          'kernel_sizes', 'depths', 'resnet_channels', 'residual_unet_channels',
          'unet_channels', 'deeplabv3_backbone_channels', 'deeplabv3_aspp_channels',
          'learning_rate_bounds', 'l2_bounds'.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'models': dict mapping each region to its trained model (or tuple of (model, window_size, stride)).
        - 'val_evaluation': dict mapping each region to its best validation loss.
        - 'histories': dict mapping each region to its epoch training history.
        - 'bayesian_opt_results': dict mapping each region to Ax optimization results.
    """
    pipeline_config = dict(config or {})
    bo_cfg_input = dict(bayesian_opt_config) if bayesian_opt_config is not None else None
    default_model_type = str(pipeline_config.get("model_type", "resnet")).lower()
    default_use_bayesian_opt = bool(
        (bo_cfg_input or {}).get(
            "use_bayesian_opt", pipeline_config.get("use_bayesian_opt", False)
        )
    )
    verbose = bool(pipeline_config.get("verbose", False))

    trained_models: dict[str, Any] = {}
    val_evaluation_results: dict[str, float] = {}
    training_histories: dict[str, dict[str, list[float]]] = {}
    bayesian_opt_results: dict[str, dict[str, Any]] = {}

    for region, bundle in regional_datasets.items():
        dataset_train = bundle["dataset_train"]
        dataset_val = bundle["dataset_val"]
        dataset_train_full = bundle.get("dataset_train_full", dataset_train)
        window_size_points = int(bundle.get("window_size_points", bundle.get("full_spectrum_length", 100)))
        stride_points = int(bundle.get("sliding_stride_points", window_size_points))
        full_spectrum_length = int(bundle.get("full_spectrum_length", window_size_points))

        _, region_train_config, region_bo_config = _resolve_region_config(
            pipeline_config, region, bayesian_opt_config=bo_cfg_input,
        )
        region_model_type = str(
            region_train_config.get("model_type", default_model_type),
        ).lower()
        film_cfg = dict(pipeline_config.get("film_config") or {})
        region_use_film = bool(
            region_train_config.get(
                "use_film",
                film_cfg.get("use_film", pipeline_config.get("use_film", False)),
            ),
        )
        region_use_bayesian_opt = bool(
            region_bo_config.get(
                "use_bayesian_opt",
                region_train_config.get("use_bayesian_opt", default_use_bayesian_opt),
            ),
        )

        is_sw = isinstance(dataset_train, SpectrumPatchDataset)
        input_sequence_length = window_size_points if is_sw else full_spectrum_length

        for k, v in film_cfg.items():
            region_train_config.setdefault(k, v)
        region_train_config["model_type"] = region_model_type
        region_train_config["use_sliding_window"] = is_sw
        region_train_config.setdefault("window_size", window_size_points)
        region_train_config.setdefault("stride", stride_points)
        region_train_config["n_points"] = input_sequence_length
        region_train_config["use_film"] = region_use_film

        if region_use_bayesian_opt:
            bo_run_cfg = {
                **film_cfg,
                **region_bo_config,
                "model_type": region_model_type,
                "use_sliding_window": is_sw,
                "stride": stride_points,
                "use_film": region_use_film,
                "criterion": criterion,
            }
            if is_sw and "window_size_choices" not in bo_run_cfg and "window_size" not in bo_run_cfg and "window_size_choice" not in bo_run_cfg:
                bo_run_cfg["window_size_choices"] = [window_size_points]
            bo_input_train_dataset = dataset_train_full if is_sw else dataset_train
            bo_result = optimize_model_hyperparameters(
                bo_input_train_dataset, dataset_val, config=bo_run_cfg,
            )
            bayesian_opt_results[region] = bo_result
            best_parameters = bo_result.get("best_parameters", {})
            if best_parameters:
                region_train_config.update(best_parameters)

                if is_sw and "window_size" in best_parameters:
                    window_size_points = int(best_parameters["window_size"])
                    stride_points = max(1, window_size_points // 2)
                    input_sequence_length = window_size_points
                    dataset_train = SpectrumPatchDataset(
                        dataset_train_full,
                        window_size=window_size_points,
                        stride=stride_points,
                    )
                    region_train_config.update(
                        {
                            "window_size": window_size_points,
                            "stride": stride_points,
                            "n_points": input_sequence_length,
                        }
                    )

        model = instantiate_model(
            region_model_type, seq_len=input_sequence_length, config=region_train_config,
        )
        region_train_config.update({"model": model, "criterion": criterion})

        if verbose:
            region_train_config["verbose"] = True

        training_result = train_model_region(
            dataset_train, dataset_val, config=region_train_config,
        )
        trained_model = training_result["model"]

        if is_sw:
            trained_models[region] = (trained_model, window_size_points, stride_points)
        else:
            trained_models[region] = trained_model

        val_evaluation_results[region] = float(training_result["best_val_loss"])
        training_histories[region] = training_result["history"]

    return {
        "models": trained_models,
        "val_evaluation": val_evaluation_results,
        "histories": training_histories,
        "bayesian_opt_results": bayesian_opt_results,
    }
