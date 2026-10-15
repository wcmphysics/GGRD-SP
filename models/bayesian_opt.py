"""Bayesian optimization of neural network hyperparameters using Ax."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ax.service.ax_client import AxClient, ObjectiveProperties


def _choice(name: str, values: Sequence[int]) -> dict[str, Any]:
    return {"name": name, "type": "choice", "values": list(values), "value_type": "int", "is_ordered": True}


def _range(name: str, bounds: Sequence[float]) -> dict[str, Any]:
    return {"name": name, "type": "range", "bounds": list(map(float, bounds)), "value_type": "float", "log_scale": True}


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

    trials_data: list[dict[str, Any]] = []
    for trial_idx in range(num_trials):
        if seed is not None:
            torch.manual_seed(seed + trial_idx)
        params, trial_id = ax_client.get_next_trial()
        trial_cfg = {
            **cfg, **params, "epochs": epochs_per_trial,
            "early_stopping_patience": epochs_per_trial, "verbose": False,
        }
        try:
            train_result = train_fn(train_dataset, val_dataset, config=trial_cfg)
            val_loss = float(train_result.get("best_val_loss", float("inf")))
            if np.isnan(val_loss) or np.isinf(val_loss):
                ax_client.log_trial_failure(trial_index=trial_id)
            else:
                ax_client.complete_trial(trial_index=trial_id, raw_data={"val_loss": val_loss})
                trials_data.append({
                    "trial_id": trial_id, "parameters": params,
                    "val_loss": val_loss, "best_epoch": train_result.get("best_epoch", 0),
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
        "best_parameters": best_params, "best_val_loss": best_val_loss,
        "trials_data": trials_data, "ax_client": ax_client,
    }


def optimize_model_hyperparameters(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform Bayesian optimization over model hyperparameters with Ax.

    Supports 'resnet', 'residual_unet', 'unet', and 'deeplabv3'
    architectures, as well as sliding window search (window_size).

    Parameters
    ----------
    train_dataset : Dataset
        Training dataset.
    val_dataset : Dataset
        Validation dataset.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'model_type' (str): 'resnet', 'residual_unet', 'unet', or
          'deeplabv3' (default 'resnet').
        - 'use_sliding_window' (bool): Whether sliding window is
          active (default False).
        - 'window_size_choices' (list[int]): Optional window sizes to
          search if sliding window active.
        - Architecture search spaces (kernel_sizes, hidden_channels,
          etc).
        - 'lr_bounds' (tuple[float, float]): Learning rate search bounds
          (default (1e-4, 1e-2)).
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
    model_type = str(cfg.get("model_type", "resnet")).lower()
    learning_rate_bounds: tuple[float, float] = tuple(
        cfg.get("learning_rate_bounds", cfg.get("lr_bounds", (1e-4, 1e-2)))
    )
    l2_bounds: tuple[float, float] = tuple(cfg.get("l2_bounds", (1e-6, 1e-2)))

    for k in cfg.get("kernel_sizes", []):
        if k <= 0 or k % 2 == 0:
            raise ValueError(f"All kernel_sizes must be positive odd integers, got {k}")
    for name, bounds in (
        ("learning_rate_bounds", learning_rate_bounds),
        ("l2_bounds", l2_bounds),
    ):
        if bounds[0] <= 0 or bounds[1] <= bounds[0]:
            raise ValueError(f"Invalid {name}: {bounds}, both must be > 0 with min < max")
    for b in cfg.get("batch_sizes", []):
        if b <= 0:
            raise ValueError(f"All batch_sizes must be positive integers, got {b}")

    parameters: list[dict[str, Any]] = []
    if "batch_sizes" in cfg:
        parameters.append(_choice("batch_size", cfg["batch_sizes"]))
    if bool(cfg.get("use_sliding_window", False)) and "window_size_choices" in cfg:
        parameters.append(_choice("window_size", cfg["window_size_choices"]))

    arch_params = {
        "unet": [("base_channels", [16, 32]), ("depth", [2, 3]), ("kernel_size", [3, 5])],
        "deeplab": [("kernel_size", [3, 5]), ("backbone_channels", [16, 32, 64]), ("aspp_channels", [16, 32, 64])],
    }.get("deeplab" if "deeplab" in model_type else ("unet" if "unet" in model_type else "resnet"),
          [("kernel_size", [3, 5, 7]), ("hidden_channels", [16, 32, 64])])
    key_map = {"depth": "depths", "kernel_size": "kernel_sizes"}
    parameters.extend(_choice(p, cfg.get(key_map.get(p, p), d)) for p, d in arch_params)
    parameters.extend([_range("learning_rate", learning_rate_bounds), _range("l2_weight", l2_bounds)])

    opt_cfg = dict(cfg)
    opt_cfg.setdefault("experiment_name", f"{model_type}_spectral_transfer_optimization")
    return run_ax_search(
        train_fn=train_model_region, train_dataset=train_dataset, val_dataset=val_dataset,
        parameters=parameters, config=opt_cfg,
    )
