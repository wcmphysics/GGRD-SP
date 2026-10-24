"""Bayesian optimization of neural network hyperparameters using Ax."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import logging
import warnings
from typing import Any, Callable

from ax.service.ax_client import AxClient, ObjectiveProperties
import numpy as np
import torch
from torch.utils.data import Dataset

_SEARCH_SPACE_KEYS = (
    "batch_sizes",
    "window_size_choices",
    "kernel_sizes",
    "depths",
    "resnet_channels",
    "residual_unet_channels",
    "unet_channels",
    "deeplabv3_backbone_channels",
    "deeplabv3_aspp_channels",
    "learning_rate_bounds",
    "l2_bounds",
)

_BO_ONLY_KEYS = {
    *_SEARCH_SPACE_KEYS,
    "parameters",
    "num_trials",
    "epochs_per_trial",
    "experiment_name",
    "use_bayesian_opt",
}

_DISALLOWED_BO_KEYS = {
    "batch_size": "batch_sizes",
    "window_size": "window_size_choices",
    "window_size_choice": "window_size_choices",
    "kernel_size": "kernel_sizes",
    "depth": "depths",
    "resnet_channel": "resnet_channels",
    "residual_unet_channel": "residual_unet_channels",
    "unet_channel": "unet_channels",
    "deeplabv3_backbone_channel": "deeplabv3_backbone_channels",
    "deeplabv3_aspp_channel": "deeplabv3_aspp_channels",
    "hidden_channels": "resnet_channels",
    "hidden_channel": "resnet_channels",
    "base_channels": "residual_unet_channels / unet_channels",
    "base_channel": "residual_unet_channels / unet_channels",
    "backbone_channels": "deeplabv3_backbone_channels",
    "backbone_channel": "deeplabv3_backbone_channels",
    "aspp_channels": "deeplabv3_aspp_channels",
    "aspp_channel": "deeplabv3_aspp_channels",
    "lr_bounds": "learning_rate_bounds",
}

_CHANNEL_SEARCH_KEYS = (
    "resnet_channels",
    "residual_unet_channels",
    "unet_channels",
    "deeplabv3_backbone_channels",
    "deeplabv3_aspp_channels",
)


def _validate_iterable(name: str, values: Any, desc: str) -> list[Any]:
    """Validate that values is a non-None, non-string/dict iterable."""
    if values is None:
        raise ValueError(f"Parameter '{name}' cannot be None; expected {desc}.")
    if isinstance(values, (str, bytes, dict)) or not isinstance(values, Iterable):
        raise TypeError(
            f"Parameter '{name}' must be {desc}, "
            f"got {type(values).__name__}: {values!r}."
        )
    return list(values)


def _choice(
    name: str,
    values: Sequence[int],
    min_val: int | None = None,
    max_val: int | None = None,
) -> dict[str, Any]:
    val_list = _validate_iterable(name, values, "an iterable of integers")
    if not val_list:
        raise ValueError(f"Choice parameter '{name}' must not be empty.")
    for v in val_list:
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(
                f"All values for choice parameter '{name}' must be integers, "
                f"got {v!r} ({type(v).__name__})."
            )
        if min_val is not None and v < min_val:
            raise ValueError(
                f"All values for '{name}' must be >= {min_val}, got {v}"
            )
        if max_val is not None and v > max_val:
            raise ValueError(
                f"All values for '{name}' must be <= {max_val}, got {v}"
            )
    return {
        "name": name,
        "type": "choice",
        "values": val_list,
        "value_type": "int",
        "is_ordered": True,
    }


def _range(name: str, bounds: Sequence[float]) -> dict[str, Any]:
    bound_list = _validate_iterable(
        name, bounds, "a 2-element iterable of floats"
    )
    if len(bound_list) != 2 or any(
        isinstance(b, bool) or not isinstance(b, (int, float)) for b in bound_list
    ):
        raise ValueError(
            f"Range parameter '{name}' must contain exactly 2 numeric bounds, "
            f"got {bounds!r}."
        )
    low, high = float(bound_list[0]), float(bound_list[1])
    if low <= 0 or high <= low:
        raise ValueError(
            f"Invalid {name} bounds: ({low}, {high}), both must be > 0 with min < max"
        )
    return {
        "name": name,
        "type": "range",
        "bounds": [low, high],
        "value_type": "float",
        "log_scale": True,
    }


def run_ax_search(
    train_fn: Callable[[Any, Any, dict[str, Any]], dict[str, Any]],
    train_dataset: Dataset,
    val_dataset: Dataset,
    parameters: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute Bayesian hyperparameter optimization search using Ax.

    Parameters
    ----------
    train_fn : Callable
        Callable with signature (train_ds, val_ds, config=...) returning
        a dict with 'best_val_loss' and optionally 'best_epoch'.
    train_dataset : Dataset
        Training dataset for the target region.
    val_dataset : Dataset
        Validation dataset for the target region.
    parameters : list[dict[str, Any]] | None, optional
        List of parameter specifications formatted for AxClient.
        If None, read from config.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'parameters' (list[dict[str, Any]]): Parameter specs if
          not passed directly.
        - 'num_trials' (int): Total number of trials (default 10).
        - 'epochs_per_trial' (int): Maximum epochs per trial
          (default 20).
        - 'seed' (int | None): Random seed (default 42).
        - 'experiment_name' (str): Experiment name
          (default 'spectral_transfer_optimization').
        - 'verbose' (bool): Whether to display trial progress
          (default False).
        - Extra hyperparameters passed down to train_fn in each trial.

    Returns
    -------
    dict[str, Any]
        Dictionary containing 'best_parameters', 'best_val_loss',
        'trials_data', and 'ax_client'.
    """
    cfg = config or {}
    search_params = (
        parameters if parameters is not None else cfg.get("parameters", [])
    )
    if not search_params:
        raise ValueError("run_ax_search requires 'parameters' list to be specified.")

    num_trials = int(cfg.get("num_trials", 10))
    epochs_per_trial = int(cfg.get("epochs_per_trial", 20))
    seed = cfg.get("seed", 42)
    verbose = bool(cfg.get("verbose", False))
    if not verbose:
        logging.getLogger("ax").setLevel(logging.WARNING)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        ax_client = AxClient(random_seed=seed)
        ax_client.create_experiment(
            name=str(
                cfg.get("experiment_name", "spectral_transfer_optimization")
            ),
            parameters=search_params,
            objectives={"val_loss": ObjectiveProperties(minimize=True)},
        )

    base_train_cfg = {k: v for k, v in cfg.items() if k not in _BO_ONLY_KEYS}
    trials_data: list[dict[str, Any]] = []
    for trial_idx in range(num_trials):
        if seed is not None:
            torch.manual_seed(seed + trial_idx)
        params, trial_id = ax_client.get_next_trial()
        trial_cfg = {
            **base_train_cfg,
            **params,
            "epochs": epochs_per_trial,
            "early_stopping_patience": epochs_per_trial,
            "verbose": False,
        }
        try:
            train_result = train_fn(train_dataset, val_dataset, config=trial_cfg)
            val_loss = float(train_result.get("best_val_loss", float("inf")))
        except Exception as exc:
            warnings.warn(f"Ax trial {trial_id} failed with error: {exc}")
            ax_client.log_trial_failure(trial_index=trial_id)
            continue

        if np.isnan(val_loss) or np.isinf(val_loss):
            ax_client.log_trial_failure(trial_index=trial_id)
            continue

        ax_client.complete_trial(
            trial_index=trial_id, raw_data={"val_loss": val_loss}
        )
        trials_data.append(
            {
                "trial_id": trial_id,
                "parameters": params,
                "val_loss": val_loss,
                "best_epoch": train_result.get("best_epoch", 0),
            }
        )
        if verbose:
            print(
                f"[Ax Trial {trial_idx + 1}/{num_trials}] "
                f"Params: {params} => Val Loss: {val_loss:.6f}"
            )

    best_params: dict[str, Any] = {}
    best_val_loss = float("inf")
    try:
        best_p, metrics = ax_client.get_best_parameters()
        if best_p is not None:
            best_params = best_p
        if (
            metrics
            and isinstance(metrics, (tuple, list))
            and isinstance(metrics[0], dict)
        ):
            best_val_loss = float(metrics[0].get("val_loss", float("inf")))
    except Exception as exc:
        warnings.warn(
            f"ax_client.get_best_parameters() failed ({exc}); "
            "falling back to best recorded trial."
        )

    if (np.isinf(best_val_loss) or np.isnan(best_val_loss)) and trials_data:
        best_trial = min(trials_data, key=lambda t: t["val_loss"])
        best_val_loss = best_trial["val_loss"]
        best_params = best_trial["parameters"]

    return {
        "best_parameters": best_params,
        "best_val_loss": best_val_loss,
        "trials_data": trials_data,
        "ax_client": ax_client,
    }


def optimize_model_hyperparameters(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform Bayesian optimization over model hyperparameters with Ax.

    Supports 'resnet', 'residual_unet', 'unet', and 'deeplabv3'
    architectures, as well as sliding window search (window_size_choices).

    Parameters
    ----------
    train_dataset : Dataset
        Training dataset.
    val_dataset : Dataset
        Validation dataset.
    config : dict[str, Any] | None, optional
        Configuration dictionary (plural keys for search spaces):
        - 'model_type' (str): 'resnet', 'residual_unet', 'unet', or
          'deeplabv3' (default 'resnet').
        - 'use_sliding_window' (bool): Whether sliding window is
          active (default False).
        - 'window_size_choices' (list[int]): Candidate window sizes in
          points when sliding window is active.
        - 'batch_sizes' (list[int]): Candidate batch sizes.
        - 'kernel_sizes' (list[int]): Candidate odd kernel sizes.
        - 'depths' (list[int]): Candidate U-Net depths.
        - 'resnet_channels' (list[int]): Candidate ResNet channels.
        - 'residual_unet_channels' (list[int]): Candidate Residual U-Net
          channels.
        - 'unet_channels' (list[int]): Candidate Conventional U-Net channels.
        - 'deeplabv3_backbone_channels' (list[int]): Candidate DeepLabV3
          backbone channels.
        - 'deeplabv3_aspp_channels' (list[int]): Candidate DeepLabV3 ASPP
          channels.
        - 'learning_rate_bounds' (tuple[float, float]):
          Learning rate search bounds (default (1e-4, 1e-2)).
        - 'l2_bounds' (tuple[float, float]): L2 regularization bounds
          (default (1e-6, 1e-2)).
        - 'num_trials' (int): Total Ax trials (default 10).
        - 'epochs_per_trial' (int): Max epochs per trial (default 20).

    Returns
    -------
    dict[str, Any]
        Dictionary with 'best_parameters', 'best_val_loss',
        'trials_data', and 'ax_client'.
    """
    from models.trainer import train_model_region

    cfg = dict(config or {})
    for invalid_key, canonical_key in _DISALLOWED_BO_KEYS.items():
        if invalid_key in cfg:
            raise KeyError(
                f"Invalid key '{invalid_key}' in Bayesian optimization config; "
                f"use plural key '{canonical_key}'."
            )

    for search_key in _SEARCH_SPACE_KEYS:
        if search_key in cfg and cfg[search_key] is None:
            raise ValueError(
                f"'{search_key}' cannot be None in Bayesian optimization config."
            )

    model_type = str(cfg.get("model_type", "resnet")).lower()
    use_sw = bool(cfg.get("use_sliding_window", False))
    full_len = (
        int(train_dataset[0]["x"].shape[-1]) if len(train_dataset) > 0 else 100
    )

    raw_lr_bounds = cfg.get("learning_rate_bounds", (1e-4, 1e-2))
    lr_range_spec = _range("learning_rate", raw_lr_bounds)
    l2_range_spec = _range("l2_weight", cfg.get("l2_bounds", (1e-6, 1e-2)))

    parameters: list[dict[str, Any]] = []
    if "batch_sizes" in cfg:
        parameters.append(_choice("batch_size", cfg["batch_sizes"], min_val=1))

    window_spec: dict[str, Any] | None = None
    if "window_size_choices" in cfg:
        window_spec = _choice(
            "window_size", cfg["window_size_choices"], min_val=3, max_val=full_len
        )
        if use_sw:
            parameters.append(window_spec)

    for ch_key in (*_CHANNEL_SEARCH_KEYS, "depths"):
        if ch_key in cfg:
            _choice(ch_key[:-1], cfg[ch_key], min_val=1)

    if "deeplab" in model_type:
        arch_specs = [
            ("kernel_size", "kernel_sizes", [3, 5]),
            ("deeplabv3_backbone_channel", "deeplabv3_backbone_channels", [16, 32, 64]),
            ("deeplabv3_aspp_channel", "deeplabv3_aspp_channels", [16, 32, 64]),
        ]
    elif "residual" in model_type and "unet" in model_type:
        arch_specs = [
            ("residual_unet_channel", "residual_unet_channels", [16, 32]),
            ("depth", "depths", [2, 3]),
            ("kernel_size", "kernel_sizes", [3, 5]),
        ]
    elif "unet" in model_type:
        arch_specs = [
            ("unet_channel", "unet_channels", [16, 32]),
            ("depth", "depths", [2, 3]),
            ("kernel_size", "kernel_sizes", [3, 5]),
        ]
    else:
        arch_specs = [
            ("kernel_size", "kernel_sizes", [3, 5, 7]),
            ("resnet_channel", "resnet_channels", [16, 32, 64]),
        ]

    resolved_arch: dict[str, list[int]] = {}
    for singular_name, plural_key, default_vals in arch_specs:
        spec = _choice(
            singular_name, cfg.get(plural_key, default_vals), min_val=1
        )
        resolved_arch[singular_name] = spec["values"]
        parameters.append(spec)

    kernel_vals = resolved_arch["kernel_size"]
    if any(k % 2 == 0 for k in kernel_vals):
        raise ValueError(
            f"All kernel_sizes must be positive odd integers, got {kernel_vals}"
        )

    if window_spec is not None:
        min_seq_len = min(window_spec["values"])
    elif use_sw:
        min_seq_len = min(full_len, 15)
    else:
        min_seq_len = full_len

    if max(kernel_vals) > min_seq_len:
        raise ValueError(
            f"kernel_size ({max(kernel_vals)}) cannot be larger than "
            f"window_size / n_points ({min_seq_len})."
        )
    if "depth" in resolved_arch:
        max_depth = max(resolved_arch["depth"])
        if 2**max_depth > min_seq_len:
            raise ValueError(
                f"window_size / n_points ({min_seq_len}) is too small for "
                f"depth {max_depth}. Requires at least {2**max_depth} points (2**depth)."
            )

    parameters.extend([lr_range_spec, l2_range_spec])

    opt_cfg = dict(cfg)
    opt_cfg.setdefault(
        "experiment_name", f"{model_type}_spectral_transfer_optimization"
    )
    return run_ax_search(
        train_fn=train_model_region,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        parameters=parameters,
        config=opt_cfg,
    )
