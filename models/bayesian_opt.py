"""Bayesian optimization of neural network hyperparameters using Ax."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import logging
import warnings
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from ax.service.ax_client import AxClient, ObjectiveProperties


_BO_ONLY_KEYS = {
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
    "lr_bounds",
    "l2_bounds",
    "parameters",
    "num_trials",
    "epochs_per_trial",
    "experiment_name",
    "use_bayesian_opt",
}


def _choice(name: str, values: Sequence[int]) -> dict[str, Any]:
    if values is None:
        raise ValueError(
            f"Choice parameter '{name}' cannot be None; expected a non-empty iterable of integers."
        )
    if isinstance(values, (str, bytes, dict)) or not isinstance(values, Iterable):
        raise TypeError(
            f"Choice parameter '{name}' must be an iterable of integers, "
            f"got {type(values).__name__}: {values!r}."
        )
    val_list = list(values)
    if not val_list:
        raise ValueError(f"Choice parameter '{name}' must not be empty.")
    for v in val_list:
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(
                f"All values for choice parameter '{name}' must be integers, "
                f"got {v!r} ({type(v).__name__})."
            )
    return {
        "name": name,
        "type": "choice",
        "values": val_list,
        "value_type": "int",
        "is_ordered": True,
    }


def _range(name: str, bounds: Sequence[float]) -> dict[str, Any]:
    if bounds is None:
        raise ValueError(
            f"Range parameter '{name}' cannot be None; expected a 2-element iterable of floats."
        )
    if isinstance(bounds, (str, bytes, dict)) or not isinstance(bounds, Iterable):
        raise TypeError(
            f"Range parameter '{name}' must be a 2-element iterable of floats, "
            f"got {type(bounds).__name__}: {bounds!r}."
        )
    bound_list = list(bounds)
    if len(bound_list) != 2 or any(
        isinstance(b, bool) or not isinstance(b, (int, float)) for b in bound_list
    ):
        raise ValueError(
            f"Range parameter '{name}' must contain exactly 2 numeric bounds, got {bounds!r}."
        )
    return {
        "name": name,
        "type": "range",
        "bounds": [float(bound_list[0]), float(bound_list[1])],
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
    search_params = parameters if parameters is not None else cfg.get("parameters", [])
    if not search_params:
        raise ValueError("run_ax_search requires 'parameters' list to be specified.")

    num_trials, epochs_per_trial = int(cfg.get("num_trials", 10)), int(cfg.get("epochs_per_trial", 20))
    seed, verbose = cfg.get("seed", 42), bool(cfg.get("verbose", False))
    if not verbose:
        logging.getLogger("ax").setLevel(logging.WARNING)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        ax_client = AxClient(random_seed=seed)
        ax_client.create_experiment(
            name=str(cfg.get("experiment_name", "spectral_transfer_optimization")),
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
            if np.isnan(val_loss) or np.isinf(val_loss):
                ax_client.log_trial_failure(trial_index=trial_id)
            else:
                ax_client.complete_trial(trial_index=trial_id, raw_data={"val_loss": val_loss})
                trials_data.append({
                    "trial_id": trial_id,
                    "parameters": params,
                    "val_loss": val_loss,
                    "best_epoch": train_result.get("best_epoch", 0),
                })
                if verbose:
                    print(f"[Ax Trial {trial_idx + 1}/{num_trials}] Params: {params} => Val Loss: {val_loss:.6f}")
        except Exception as exc:
            warnings.warn(f"Ax trial {trial_id} failed with error: {exc}")
            ax_client.log_trial_failure(trial_index=trial_id)

    best_params, best_val_loss = {}, float("inf")
    try:
        best_p, metrics = ax_client.get_best_parameters()
        if best_p is not None:
            best_params = best_p
        if metrics and isinstance(metrics, (tuple, list)) and isinstance(metrics[0], dict):
            best_val_loss = float(metrics[0].get("val_loss", float("inf")))
    except Exception as exc:
        warnings.warn(f"ax_client.get_best_parameters() failed ({exc}); falling back to best recorded trial.")

    if (np.isinf(best_val_loss) or np.isnan(best_val_loss)) and trials_data:
        best_trial = min(trials_data, key=lambda t: t["val_loss"])
        best_val_loss, best_params = best_trial["val_loss"], best_trial["parameters"]

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
        - 'residual_unet_channels' (list[int]): Candidate Residual U-Net channels.
        - 'unet_channels' (list[int]): Candidate Conventional U-Net channels.
        - 'deeplabv3_backbone_channels' (list[int]): Candidate DeepLabV3 backbone channels.
        - 'deeplabv3_aspp_channels' (list[int]): Candidate DeepLabV3 ASPP channels.
        - 'learning_rate_bounds' / 'lr_bounds' (tuple[float, float]):
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
    disallowed_bo_keys = {
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
    }
    for invalid_key, canonical_key in disallowed_bo_keys.items():
        if invalid_key in cfg:
            raise KeyError(
                f"Invalid key '{invalid_key}' in Bayesian optimization config; "
                f"use plural key '{canonical_key}'."
            )

    for search_key in (
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
        "lr_bounds",
        "l2_bounds",
    ):
        if search_key in cfg and cfg[search_key] is None:
            raise ValueError(
                f"'{search_key}' cannot be None in Bayesian optimization config."
            )

    model_type = str(cfg.get("model_type", "resnet")).lower()
    use_sw = bool(cfg.get("use_sliding_window", False))
    full_len = int(train_dataset[0]["x"].shape[-1]) if len(train_dataset) > 0 else 100

    raw_lr_bounds = (
        cfg["learning_rate_bounds"]
        if "learning_rate_bounds" in cfg
        else cfg.get("lr_bounds", (1e-4, 1e-2))
    )
    lr_range_spec = _range("learning_rate", raw_lr_bounds)
    l2_range_spec = _range("l2_weight", cfg.get("l2_bounds", (1e-6, 1e-2)))

    for name, bounds in (
        ("learning_rate_bounds", lr_range_spec["bounds"]),
        ("l2_bounds", l2_range_spec["bounds"]),
    ):
        if bounds[0] <= 0 or bounds[1] <= bounds[0]:
            raise ValueError(f"Invalid {name}: {tuple(bounds)}, both must be > 0 with min < max")

    parameters: list[dict[str, Any]] = []
    if "batch_sizes" in cfg:
        batch_spec = _choice("batch_size", cfg["batch_sizes"])
        for b in batch_spec["values"]:
            if b <= 0:
                raise ValueError(f"All batch_sizes must be positive integers, got {b}")
        parameters.append(batch_spec)

    window_spec: dict[str, Any] | None = None
    if "window_size_choices" in cfg:
        window_spec = _choice("window_size", cfg["window_size_choices"])
        for w in window_spec["values"]:
            if w < 3 or w > full_len:
                raise ValueError(
                    f"All window_size_choices must be integers in [3, {full_len}], got {w}"
                )
        if use_sw:
            parameters.append(window_spec)

    for ch_key in (
        "resnet_channels",
        "residual_unet_channels",
        "unet_channels",
        "deeplabv3_backbone_channels",
        "deeplabv3_aspp_channels",
    ):
        if ch_key in cfg:
            ch_spec = _choice(ch_key[:-1], cfg[ch_key])
            for c in ch_spec["values"]:
                if c <= 0:
                    raise ValueError(f"All {ch_key} must be positive integers, got {c}")

    if "depths" in cfg:
        depth_spec = _choice("depth", cfg["depths"])
        for d_val in depth_spec["values"]:
            if d_val < 1:
                raise ValueError(f"All depths must be positive integers >= 1, got {d_val}")

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
        spec = _choice(singular_name, cfg.get(plural_key, default_vals))
        resolved_arch[singular_name] = spec["values"]
        parameters.append(spec)

    kernel_vals = resolved_arch["kernel_size"]
    for k in kernel_vals:
        if k <= 0 or k % 2 == 0:
            raise ValueError(f"All kernel_sizes must be positive odd integers, got {k}")

    if window_spec is not None:
        min_seq_len = min(window_spec["values"])
    elif use_sw:
        min_seq_len = min(full_len, 15)
    else:
        min_seq_len = full_len

    if max(kernel_vals) > min_seq_len:
        raise ValueError(
            f"kernel_size ({max(kernel_vals)}) cannot be larger than window_size / n_points ({min_seq_len})."
        )
    if "depth" in resolved_arch:
        max_depth = max(resolved_arch["depth"])
        if 2**max_depth > min_seq_len:
            raise ValueError(
                f"window_size / n_points ({min_seq_len}) is too small for depth {max_depth}. "
                f"Requires at least {2**max_depth} points (2**depth)."
            )

    parameters.extend([lr_range_spec, l2_range_spec])

    opt_cfg = dict(cfg)
    opt_cfg.setdefault("experiment_name", f"{model_type}_spectral_transfer_optimization")
    return run_ax_search(
        train_fn=train_model_region,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        parameters=parameters,
        config=opt_cfg,
    )

