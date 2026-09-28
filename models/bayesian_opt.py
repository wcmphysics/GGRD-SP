"""Bayesian optimization of baseline hyperparameters using the Ax platform."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import torch
from ax.service.ax_client import AxClient, ObjectiveProperties

from models.dataset import SpectrumPairDataset
from models.trainer import train_baseline_region


def optimize_baseline_hyperparameters(
    train_dataset: SpectrumPairDataset,
    val_dataset: SpectrumPairDataset,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform Bayesian optimization over baseline CNN hyperparameters using Ax.

    Optimizes kernel size, hidden channels, learning rate, and L2 regularization weight
    to minimize validation loss.

    Parameters
    ----------
    train_dataset : SpectrumPairDataset
        Training dataset for the target region.
    val_dataset : SpectrumPairDataset
        Validation dataset for the target region.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'num_trials' (int): Total number of Bayesian optimization trials (default 10).
        - 'epochs_per_trial' (int): Training epochs per trial (default 20).
        - 'kernel_sizes' (list[int]): Choice of odd kernel sizes (default [3, 5, 7]).
        - 'hidden_channels' (list[int]): Choice of channel counts (default [16, 32, 64]).
        - 'lr_bounds' (tuple[float, float]): (min, max) for learning rate log-scale (default (1e-4, 1e-2)).
        - 'l2_bounds' (tuple[float, float]): (min, max) for L2 weight log-scale (default (1e-6, 1e-2)).
        - 'batch_size' (int): Batch size (default 16).
        - 'seed' (int | None): Random seed for reproducibility (default 42).
        - 'device' (str | torch.device | None): Computation device.
        - 'verbose' (bool): Whether to print trial progress (default False).

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'best_parameters': Best hyperparameter combination.
        - 'best_val_loss': Lowest validation loss achieved across trials.
        - 'trials_data': Summary of all evaluated trials.
        - 'ax_client': AxClient instance.
    """
    cfg = config or {}
    num_trials: int = int(cfg.get("num_trials", 10))
    epochs_per_trial: int = int(cfg.get("epochs_per_trial", 20))
    kernel_sizes: list[int] = list(cfg.get("kernel_sizes", [3, 5, 7]))
    hidden_channel_choices: list[int] = list(cfg.get("hidden_channels", [16, 32, 64]))
    lr_bounds: tuple[float, float] = tuple(cfg.get("lr_bounds", (1e-4, 1e-2)))
    l2_bounds: tuple[float, float] = tuple(cfg.get("l2_bounds", (1e-6, 1e-2)))
    batch_size: int = int(cfg.get("batch_size", 16))
    seed: int | None = cfg.get("seed", 42)
    device = cfg.get("device")
    verbose: bool = bool(cfg.get("verbose", False))

    # Parameter validation
    for k in kernel_sizes:
        if k <= 0 or k % 2 == 0:
            raise ValueError(f"All kernel_sizes must be positive odd integers, got {k}")
    if lr_bounds[0] <= 0 or lr_bounds[1] <= lr_bounds[0]:
        raise ValueError(f"Invalid lr_bounds: {lr_bounds}, both must be > 0 with min < max")
    if l2_bounds[0] <= 0 or l2_bounds[1] <= l2_bounds[0]:
        raise ValueError(f"Invalid l2_bounds: {l2_bounds}, both must be > 0 with min < max")

    # Suppress verbose Ax logs unless verbose is enabled
    if not verbose:
        logging.getLogger("ax").setLevel(logging.WARNING)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        ax_client = AxClient(random_seed=seed)

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
                "bounds": [lr_bounds[0], lr_bounds[1]],
                "value_type": "float",
                "log_scale": True,
            },
            {
                "name": "l2_weight",
                "type": "range",
                "bounds": [l2_bounds[0], l2_bounds[1]],
                "value_type": "float",
                "log_scale": True,
            },
        ]

        ax_client.create_experiment(
            name="baseline_cnn_optimization",
            parameters=parameters,
            objectives={"val_loss": ObjectiveProperties(minimize=True)},
        )

    trials_data: list[dict[str, Any]] = []

    for trial_idx in range(num_trials):
        if seed is not None:
            torch.manual_seed(seed + trial_idx)

        params, trial_id = ax_client.get_next_trial()

        train_cfg = {
            "kernel_size": int(params["kernel_size"]),
            "hidden_channels": int(params["hidden_channels"]),
            "learning_rate": float(params["learning_rate"]),
            "l2_weight": float(params["l2_weight"]),
            "epochs": epochs_per_trial,
            "batch_size": batch_size,
            "device": device,
            "verbose": False,
        }

        train_result = train_baseline_region(train_dataset, val_dataset, config=train_cfg)
        val_loss = float(train_result["best_val_loss"])

        ax_client.complete_trial(trial_index=trial_id, raw_data={"val_loss": val_loss})

        trial_record = {
            "trial_id": trial_id,
            "parameters": params,
            "val_loss": val_loss,
            "best_epoch": train_result["best_epoch"],
        }
        trials_data.append(trial_record)

        if verbose:
            print(f"[Ax Trial {trial_idx + 1}/{num_trials}] Params: {params} => Val Loss: {val_loss:.6f}")

    best_params, _ = ax_client.get_best_parameters()
    best_val_loss = min(t["val_loss"] for t in trials_data)

    return {
        "best_parameters": best_params,
        "best_val_loss": best_val_loss,
        "trials_data": trials_data,
        "ax_client": ax_client,
    }
