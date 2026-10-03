"""Bayesian optimization of neural network hyperparameters using the Ax platform."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from ax.service.ax_client import AxClient, ObjectiveProperties

    AX_AVAILABLE = True
except (ImportError, OSError):
    AX_AVAILABLE = False


def run_ax_search(
    train_fn: Callable[[Any, Any, dict[str, Any]], dict[str, Any]],
    train_dataset: Dataset,
    val_dataset: Dataset,
    parameters: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a standardized Bayesian hyperparameter optimization search using Ax.

    Parameters
    ----------
    train_fn : Callable
        Callable with signature (train_ds, val_ds, config=...) returning a dict with
        'best_val_loss' and optionally 'best_epoch'.
    train_dataset : Dataset
        Training dataset for the target region.
    val_dataset : Dataset
        Validation dataset for the target region.
    parameters : list[dict[str, Any]] | None, optional
        List of parameter specifications formatted for AxClient. If None, read from config.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'parameters' (list[dict[str, Any]]): Parameter specs if not passed directly.
        - 'num_trials' (int): Total number of trials (default 10).
        - 'epochs_per_trial' (int): Maximum epochs per trial (default 20).
        - 'seed' (int | None): Random seed (default 42).
        - 'experiment_name' (str): Experiment name (default 'spectral_transfer_optimization').
        - 'verbose' (bool): Whether to display trial progress (default False).
        - Extra hyperparameters passed down to train_fn in each trial.

    Returns
    -------
    dict[str, Any]
        Dictionary containing 'best_parameters', 'best_val_loss', 'trials_data', and 'ax_client'.
    """
    if not AX_AVAILABLE:
        warnings.warn("Ax platform is not available. Skipping hyperparameter optimization.")
        return {
            "best_parameters": {},
            "best_val_loss": float("nan"),
            "trials_data": [],
            "ax_client": None,
        }

    cfg = config or {}
    search_params = parameters if parameters is not None else cfg.get("parameters", [])
    if not search_params:
        raise ValueError("run_ax_search requires 'parameters' list to be specified.")

    num_trials: int = int(cfg.get("num_trials", 10))
    epochs_per_trial: int = int(cfg.get("epochs_per_trial", 20))
    seed: int | None = cfg.get("seed", 42)
    experiment_name: str = str(cfg.get("experiment_name", "spectral_transfer_optimization"))
    verbose: bool = bool(cfg.get("verbose", False))

    if not verbose:
        logging.getLogger("ax").setLevel(logging.WARNING)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        ax_client = AxClient(random_seed=seed)
        ax_client.create_experiment(
            name=experiment_name,
            parameters=search_params,
            objectives={"val_loss": ObjectiveProperties(minimize=True)},
        )

    trials_data: list[dict[str, Any]] = []

    for trial_idx in range(num_trials):
        if seed is not None:
            torch.manual_seed(seed + trial_idx)

        params, trial_id = ax_client.get_next_trial()

        trial_train_cfg = dict(cfg)
        trial_train_cfg.update(params)
        trial_train_cfg["epochs"] = epochs_per_trial
        trial_train_cfg["early_stopping_patience"] = epochs_per_trial
        trial_train_cfg["verbose"] = False

        try:
            train_result = train_fn(train_dataset, val_dataset, config=trial_train_cfg)
            val_loss = float(train_result.get("best_val_loss", float("inf")))

            if np.isnan(val_loss) or np.isinf(val_loss):
                ax_client.log_trial_failure(trial_index=trial_id)
            else:
                ax_client.complete_trial(trial_index=trial_id, raw_data={"val_loss": val_loss})
                trial_record = {
                    "trial_id": trial_id,
                    "parameters": params,
                    "val_loss": val_loss,
                    "best_epoch": train_result.get("best_epoch", 0),
                }
                trials_data.append(trial_record)

                if verbose:
                    print(f"[Ax Trial {trial_idx + 1}/{num_trials}] Params: {params} => Val Loss: {val_loss:.6f}")
        except Exception as exc:
            warnings.warn(f"Ax trial {trial_id} failed with error: {exc}")
            ax_client.log_trial_failure(trial_index=trial_id)

    best_params: dict[str, Any] = {}
    best_val_loss = float("inf")
    try:
        best_p, metrics = ax_client.get_best_parameters()
        if best_p is not None:
            best_params = best_p
        if metrics is not None and isinstance(metrics, (tuple, list)) and len(metrics) > 0:
            first = metrics[0]
            if isinstance(first, dict) and "val_loss" in first:
                best_val_loss = float(first["val_loss"])
    except Exception:
        pass

    if (np.isinf(best_val_loss) or np.isnan(best_val_loss)) and trials_data:
        sorted_trials = sorted(trials_data, key=lambda t: t["val_loss"])
        best_val_loss = sorted_trials[0]["val_loss"]
        best_params = sorted_trials[0]["parameters"]

    return {
        "best_parameters": best_params,
        "best_val_loss": best_val_loss,
        "trials_data": trials_data,
        "ax_client": ax_client,
    }


def optimize_baseline_hyperparameters(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform Bayesian optimization over baseline CNN hyperparameters using Ax.

    Optimizes kernel size, hidden channels, learning rate, and L2 regularization weight
    to minimize validation loss.
    """
    from models.trainer import train_baseline_region

    cfg = config or {}
    kernel_sizes: list[int] = list(cfg.get("kernel_sizes", [3, 5, 7]))
    hidden_channel_choices: list[int] = list(cfg.get("hidden_channels", [16, 32, 64]))
    lr_bounds: tuple[float, float] = tuple(cfg.get("lr_bounds", (1e-4, 1e-2)))
    l2_bounds: tuple[float, float] = tuple(cfg.get("l2_bounds", (1e-6, 1e-2)))

    # Parameter validation
    for k in kernel_sizes:
        if k <= 0 or k % 2 == 0:
            raise ValueError(f"All kernel_sizes must be positive odd integers, got {k}")
    if lr_bounds[0] <= 0 or lr_bounds[1] <= lr_bounds[0]:
        raise ValueError(f"Invalid lr_bounds: {lr_bounds}, both must be > 0 with min < max")
    if l2_bounds[0] <= 0 or l2_bounds[1] <= l2_bounds[0]:
        raise ValueError(f"Invalid l2_bounds: {l2_bounds}, both must be > 0 with min < max")

    parameters: list[dict[str, Any]] = [
        {
            "name": "kernel_size",
            "type": "choice",
            "values": kernel_sizes,
            "value_type": "int",
            "is_ordered": True,
        },
        {
            "name": "hidden_channels",
            "type": "choice",
            "values": hidden_channel_choices,
            "value_type": "int",
            "is_ordered": True,
        },
        {
            "name": "learning_rate",
            "type": "range",
            "bounds": [float(lr_bounds[0]), float(lr_bounds[1])],
            "value_type": "float",
            "log_scale": True,
        },
        {
            "name": "l2_weight",
            "type": "range",
            "bounds": [float(l2_bounds[0]), float(l2_bounds[1])],
            "value_type": "float",
            "log_scale": True,
        },
    ]

    opt_cfg = dict(cfg)
    opt_cfg["experiment_name"] = cfg.get("experiment_name", "baseline_cnn_optimization")

    return run_ax_search(
        train_fn=train_baseline_region,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        parameters=parameters,
        config=opt_cfg,
    )
