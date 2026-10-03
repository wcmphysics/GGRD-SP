"""Sliding window neural network model and pipeline for sequence-to-sequence spectral transfer."""

from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.baseline_cnn import NormalizedMSELoss, Residual1DCNN
from models.bayesian_opt import run_ax_search
from models.dataset import SpectrumPairDataset
from models.inference import assemble_prediction_metadata
from models.root import run_model_pipeline


from utility.patching import (
    calculate_window_points,
    extract_sliding_windows,
    reconstruct_from_patches,
)


class SpectrumPatchDataset(Dataset):
    """PyTorch Dataset yielding paired sliding window patches extracted from paired spectra.

    Parameters
    ----------
    full_dataset : SpectrumPairDataset
        The full-spectrum paired dataset.
    window_size : int
        Window size in points.
    stride : int
        Stride in points.
    """

    def __init__(
        self,
        full_dataset: SpectrumPairDataset,
        window_size: int,
        stride: int,
    ) -> None:
        self.window_size = window_size
        self.stride = stride

        x_patches_list: list[np.ndarray] = []
        y_patches_list: list[np.ndarray] = []
        max_vals_list: list[float] = []

        for idx in range(len(full_dataset)):
            item = full_dataset[idx]
            x_arr = item["x"].cpu().numpy()
            y_arr = item["y"].cpu().numpy()
            max_y = max(float(np.max(np.abs(y_arr))), 1e-4)

            x_win, _ = extract_sliding_windows(x_arr, window_size, stride)
            y_win, _ = extract_sliding_windows(y_arr, window_size, stride)

            x_patches_list.append(x_win)
            y_patches_list.append(y_win)
            max_vals_list.extend([max_y] * len(x_win))

        if not x_patches_list:
            raise ValueError("No patches could be extracted from the dataset.")

        self.x = torch.from_numpy(np.vstack(x_patches_list).astype(np.float32))
        self.y = torch.from_numpy(np.vstack(y_patches_list).astype(np.float32))
        self.max_vals = torch.tensor(max_vals_list, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "max_val": self.max_vals[idx],
        }


def predict_sliding_window_spectrum(
    model: nn.Module,
    spectrum: np.ndarray,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Transform a full 1D spectrum using a trained sliding window patch model.

    Parameters
    ----------
    model : nn.Module
        Trained Residual1DCNN model expecting input length window_size.
    spectrum : np.ndarray
        1D source spectrum array of length N.
    config : dict[str, Any] | None, optional
        Configuration dictionary containing:
        - 'window_size' (int): Window points.
        - 'stride' (int): Stride points.
        - 'device' (str | torch.device | None): Computation device.
        - 'batch_size' (int): Batch size for patch forward pass (default 64).
    **kwargs : Any
        Optional keyword arguments (e.g. window_size, stride) for backward compatibility.

    Returns
    -------
    np.ndarray
        Reconstructed target spectrum of length N.
    """
    cfg = dict(config or {})
    cfg.update(kwargs)
    w_size = int(cfg.get("window_size", 15))
    s_step = int(cfg.get("stride", max(1, w_size // 2)))

    device = cfg.get("device")
    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    batch_size = int(cfg.get("batch_size", 64))

    windows, start_indices = extract_sliding_windows(
        spectrum, window_size=w_size, stride=s_step
    )

    model.eval()
    pred_patches_list: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i : i + batch_size]).to(device)
            out = model(batch).cpu().numpy()
            pred_patches_list.append(out)

    pred_patches = np.vstack(pred_patches_list)
    reconstructed = reconstruct_from_patches(
        patches=pred_patches,
        start_indices=start_indices,
        original_length=len(np.asarray(spectrum).ravel()),
        config={"window_size": w_size},
    )
    return reconstructed


def evaluate_sliding_window(
    model: nn.Module,
    full_val_dataset: SpectrumPairDataset,
    criterion: NormalizedMSELoss,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Evaluate end-to-end normalized MSE on the reconstructed full spectra.

    Parameters
    ----------
    model : nn.Module
        Trained patch model.
    full_val_dataset : SpectrumPairDataset
        Validation dataset containing full paired spectra.
    criterion : NormalizedMSELoss
        Loss function instance.
    config : dict[str, Any] | None, optional
        Configuration dictionary containing 'window_size', 'stride', and 'device'.
    **kwargs : Any
        Optional keyword arguments (e.g. window_size, stride, device) for backward compatibility.

    Returns
    -------
    float
        Average normalized MSE across all reconstructed validation spectra.
    """
    cfg = dict(config or {})
    cfg.update(kwargs)
    w_size = int(cfg.get("window_size", 15))
    s_step = int(cfg.get("stride", max(1, w_size // 2)))

    device = cfg.get("device")
    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    if len(full_val_dataset) == 0:
        raise ValueError("full_val_dataset is empty.")

    total_loss = 0.0
    model.eval()

    predict_cfg = {"device": device, "window_size": w_size, "stride": s_step}

    for idx in range(len(full_val_dataset)):
        item = full_val_dataset[idx]
        x_arr = item["x"].cpu().numpy()
        y_true = item["y"].unsqueeze(0).to(device)

        reconstructed_arr = predict_sliding_window_spectrum(
            model, x_arr, config=predict_cfg
        )
        y_pred = torch.from_numpy(reconstructed_arr).unsqueeze(0).to(device)

        loss = criterion(y_pred, y_true, model=None)
        total_loss += loss.item()

    return total_loss / len(full_val_dataset)


def train_sliding_window_region(
    train_full_ds: SpectrumPairDataset,
    val_full_ds: SpectrumPairDataset,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a 1D Residual CNN using sliding window patches with full-spectrum evaluation.

    Parameters
    ----------
    train_full_ds : SpectrumPairDataset
        Full-spectrum training dataset.
    val_full_ds : SpectrumPairDataset
        Full-spectrum validation dataset.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'window_size' | 'window_size_points' (int): Window points.
        - 'stride' | 'sliding_stride_points' (int): Stride points.
        - 'hidden_channels' (int): Conv hidden channels (default 32).
        - 'kernel_size' (int): Conv kernel size, must be odd and <= window_size (default 5).
        - 'learning_rate' (float): Adam learning rate (default 1e-3).
        - 'weight_decay' (float): Optimizer weight decay (default 0.0).
        - 'l2_weight' (float): L2 regularization weight on conv weights (default 1e-4).
        - 'epochs' (int): Maximum epochs (default 40).
        - 'batch_size' (int): Batch size for patch DataLoader (default 32).
        - 'early_stopping_patience' (int): Early stopping patience epochs (default 10).
        - 'device' (str | torch.device | None): Computation device.
        - 'verbose' (bool): Whether to log progress (default False).

    Returns
    -------
    dict[str, Any]
        Dictionary with 'model', 'best_val_loss', 'best_epoch', 'history', 'window_size', 'stride'.
    """
    if len(train_full_ds) == 0:
        raise ValueError("Cannot train on empty train_full_ds.")
    if len(val_full_ds) == 0:
        raise ValueError("Cannot evaluate on empty val_full_ds.")

    cfg = config or {}

    # Check if explicit window_size or points were provided in config
    explicit_w = cfg.get("window_size", cfg.get("window_size_points"))
    explicit_s = cfg.get("stride", cfg.get("sliding_stride_points"))

    sample_energy = train_full_ds[0].get("energy")
    if explicit_w is not None:
        window_size = int(explicit_w)
        stride = int(explicit_s) if explicit_s is not None else max(1, window_size // 2)
        stride = max(1, min(window_size, stride))
    elif sample_energy is not None:
        energy_np = sample_energy.cpu().numpy()
        window_size, stride = calculate_window_points(energy_np, config=cfg)
    else:
        window_size = 15
        stride = 7

    # Ensure kernel_size <= window_size
    kernel_size = int(cfg.get("kernel_size", 5))
    if kernel_size > window_size:
        kernel_size = window_size if window_size % 2 == 1 else window_size - 1

    hidden_channels = int(cfg.get("hidden_channels", 32))
    lr = float(cfg.get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    l2_weight = float(cfg.get("l2_weight", 1e-4))
    epochs = int(cfg.get("epochs", 40))
    batch_size = int(cfg.get("batch_size", 32))
    patience = int(cfg.get("early_stopping_patience", 10))
    verbose = bool(cfg.get("verbose", False))

    device_str = cfg.get("device")
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device_str, str):
        device = torch.device(device_str)
    else:
        device = device_str

    # Create patch dataset for training
    train_patch_ds = SpectrumPatchDataset(
        train_full_ds, window_size=window_size, stride=stride
    )
    patch_loader = DataLoader(train_patch_ds, batch_size=batch_size, shuffle=True)

    # Instantiate model with n_points = window_size
    model_cfg = {
        "hidden_channels": hidden_channels,
        "kernel_size": kernel_size,
        "dropout": float(cfg.get("dropout", 0.0)),
        "use_batch_norm": bool(cfg.get("use_batch_norm", True)),
    }
    model = Residual1DCNN(n_points=window_size, config=model_cfg).to(device)
    criterion = NormalizedMSELoss(l2_weight=l2_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, Any] | None = None
    patience_counter = 0

    eval_cfg = {"device": device, "window_size": window_size, "stride": stride}

    for epoch in range(1, epochs + 1):
        # Training loop on patches with full-spectrum peak normalization
        model.train()
        epoch_patch_loss = 0.0
        n_batches = 0
        for batch in patch_loader:
            x_b = batch["x"].to(device)
            y_b = batch["y"].to(device)
            max_v = batch["max_val"].to(device)

            optimizer.zero_grad()
            y_pred = model(x_b)
            loss = criterion(y_pred, y_b, model=model, max_val=max_v)
            loss.backward()
            optimizer.step()

            epoch_patch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_patch_loss / max(1, n_batches)

        # Validation on full reconstructed spectra
        val_reconstructed_loss = evaluate_sliding_window(
            model=model,
            full_val_dataset=val_full_ds,
            criterion=criterion,
            config=eval_cfg,
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_reconstructed_loss)

        if val_reconstructed_loss < best_val_loss:
            best_val_loss = val_reconstructed_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % 10 == 0 or epoch == epochs):
            print(
                f"[Sliding Window W={window_size}, S={stride}] Epoch {epoch:3d}/{epochs:3d} "
                f"- Train Loss: {avg_train_loss:.6f}, Val Loss: {val_reconstructed_loss:.6f}"
            )

        if patience_counter >= patience:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best epoch: {best_epoch}, best val_loss: {best_val_loss:.6f})"
                )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
        "window_size": window_size,
        "stride": stride,
        "config": cfg,
    }


def optimize_sliding_window_hyperparameters(
    train_full_ds: SpectrumPairDataset,
    val_full_ds: SpectrumPairDataset,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform Bayesian optimization over sliding window model hyperparameters using Ax.

    Optimizes window size, kernel size, hidden channels, learning rate, and L2
    regularization weight against reconstructed full-spectrum validation loss.

    Parameters
    ----------
    train_full_ds : SpectrumPairDataset
        Training full-spectrum dataset.
    val_full_ds : SpectrumPairDataset
        Validation full-spectrum dataset.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'num_trials' (int): Total Bayesian optimization trials (default 8).
        - 'epochs_per_trial' (int): Training epochs per trial (default 15).
        - 'window_size_choices' (list[int]): Window size options in points (default [11, 15, 21]).
        - 'kernel_sizes' (list[int]): Conv kernel sizes (default [3, 5]).
        - 'hidden_channels' (list[int]): Hidden channels (default [16, 32]).
        - 'lr_bounds' (tuple[float, float]): (min, max) for learning rate log-scale (default (1e-4, 1e-2)).
        - 'l2_bounds' (tuple[float, float]): (min, max) for L2 weight log-scale (default (1e-6, 1e-2)).
        - 'seed' (int | None): Random seed for reproducibility (default 42).
        - 'device' (str | torch.device | None): Computation device.
        - 'verbose' (bool): Whether to log progress (default False).

    Returns
    -------
    dict[str, Any]
        Dictionary containing best parameters, lowest reconstructed validation loss,
        trials data, and ax_client.
    """
    cfg = config or {}
    num_trials: int = int(cfg.get("num_trials", 8))
    epochs_per_trial: int = int(cfg.get("epochs_per_trial", 15))
    window_size_choices: list[int] = list(cfg.get("window_size_choices", [11, 15, 21]))
    kernel_sizes: list[int] = list(cfg.get("kernel_sizes", [3, 5]))
    hidden_channels: list[int] = list(cfg.get("hidden_channels", [16, 32]))
    lr_bounds: tuple[float, float] = tuple(cfg.get("lr_bounds", (1e-4, 1e-2)))
    l2_bounds: tuple[float, float] = tuple(cfg.get("l2_bounds", (1e-6, 1e-2)))
    seed: int | None = cfg.get("seed", 42)
    device = cfg.get("device")
    verbose: bool = bool(cfg.get("verbose", False))

    for k in kernel_sizes:
        if k <= 0 or k % 2 == 0:
            raise ValueError(f"All kernel_sizes must be positive odd integers, got {k}")
    if lr_bounds[0] <= 0 or lr_bounds[1] <= lr_bounds[0]:
        raise ValueError(f"Invalid lr_bounds: {lr_bounds}, both must be > 0 with min < max")
    if l2_bounds[0] <= 0 or l2_bounds[1] <= l2_bounds[0]:
        raise ValueError(f"Invalid l2_bounds: {l2_bounds}, both must be > 0 with min < max")

    parameters: list[dict[str, Any]] = [
        {
            "name": "window_size",
            "type": "choice",
            "values": window_size_choices,
            "value_type": "int",
            "is_ordered": True,
        },
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
            "values": hidden_channels,
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

    def _train_adapter(
        train_ds: Any,
        val_ds: Any,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        c = dict(config or {})
        win_size = int(c["window_size"])
        k_size = int(c["kernel_size"])
        if k_size > win_size:
            c["kernel_size"] = win_size if win_size % 2 == 1 else win_size - 1
        c["stride"] = max(1, win_size // 2)
        return train_sliding_window_region(train_ds, val_ds, config=c)

    opt_cfg = dict(cfg)
    opt_cfg["experiment_name"] = cfg.get("experiment_name", "sliding_window_optimization")

    return run_ax_search(
        train_fn=_train_adapter,
        train_dataset=train_full_ds,
        val_dataset=val_full_ds,
        parameters=parameters,
        config=opt_cfg,
    )


def predict_sliding_window_spectra(
    models: dict[str, tuple[nn.Module, int, int]],
    data: tuple[np.ndarray, np.ndarray, pd.DataFrame] | dict[str, Any],
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate transformed predictions across source measurements using sliding window models.

    Parameters
    ----------
    models : dict[str, tuple[nn.Module, int, int]]
        Mapping of region name to tuple of (trained_model, window_size, stride).
    data : tuple[np.ndarray, np.ndarray, pd.DataFrame] | dict[str, Any]
        Original measured data (ary_intensity, ary_energy, meta_df).
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'source_tool' (str): Source tool identifier (default 'J4').
        - 'target_tool' (str): Target tool identifier (default 'H1').
        - 'measurement_ids' (list[str] | None): Specific source measurements to predict.
        - 'device' (str | torch.device | None): Computation device.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame]
        (ary_intensity_predicted, ary_energy_predicted, meta_df_predicted)
    """
    if isinstance(data, (tuple, list)):
        ary_intensity, ary_energy, meta_df = data[0], data[1], data[2]
    elif isinstance(data, dict):
        ary_intensity = data["ary_intensity"]
        ary_energy = data["ary_energy"]
        meta_df = data["meta_df"]
    else:
        raise TypeError(f"Expected data to be tuple or dict, got {type(data)}")

    cfg = config or {}
    source_tool: str = cfg.get("source_tool", "J4")
    target_tool: str = cfg.get("target_tool", "H1")
    measurement_ids: list[str] | None = cfg.get("measurement_ids")

    condition = meta_df["tool"] == source_tool
    if measurement_ids is not None:
        condition = condition & meta_df["measurement_id"].isin(measurement_ids)

    source_df = meta_df[condition].copy()
    if source_df.empty:
        raise ValueError(f"No source spectra found for tool='{source_tool}' with criteria: {cfg}")

    n_samples = len(source_df)
    n_points = ary_intensity.shape[1]

    predicted_intensities = np.zeros((n_samples, n_points), dtype=np.float32)
    predicted_energies = np.zeros((n_samples, n_points), dtype=np.float32)
    predicted_meta_records: list[dict[str, Any]] = []

    source_indices = source_df["spectrum_index"].astype(int).to_numpy()
    predicted_energies[:] = ary_energy[source_indices]

    for new_idx, (_, row) in enumerate(source_df.iterrows()):
        region = str(row["region"])
        orig_idx = int(row["spectrum_index"])

        x_raw = ary_intensity[orig_idx]

        if region in models:
            model_tuple = models[region]
            if isinstance(model_tuple, (tuple, list)):
                model, w_size, s_step = model_tuple[0], model_tuple[1], model_tuple[2]
            else:
                model, w_size, s_step = model_tuple, 15, 7

            reconstructed = predict_sliding_window_spectrum(
                model=model,
                spectrum=x_raw,
                config=dict(cfg, window_size=w_size, stride=s_step),
            )
            predicted_intensities[new_idx] = reconstructed
        else:
            warnings.warn(f"Region '{region}' not in models; falling back to identity pass-through.")
            predicted_intensities[new_idx] = x_raw.copy()

    meta_df_predicted = assemble_prediction_metadata(
        source_df=source_df,
        source_tool=source_tool,
        target_tool=target_tool,
        n_points=n_points,
    )
    return predicted_intensities, predicted_energies, meta_df_predicted


def run_sliding_window_pipeline(
    meta_df: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the end-to-end sliding window modeling pipeline."""
    cfg = config or {}

    def _prep_sw_cfg(train_ds: Any, val_ds: Any, region_cfg: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
        c = dict(region_cfg)
        for k in (
            "window_size_ev",
            "sliding_stride_ev",
            "window_size",
            "window_size_points",
            "stride",
            "sliding_stride_points",
        ):
            if k in cfg and k not in c:
                c[k] = cfg[k]
        return train_ds, val_ds, c

    hooks = {
        "bo_fn": optimize_sliding_window_hyperparameters,
        "dataset_prep_fn": _prep_sw_cfg,
        "model_record_fn": lambda res, r_cfg: (res["model"], res["window_size"], res["stride"]),
    }
    return run_model_pipeline(
        data=(ary_intensity, ary_energy, meta_df),
        train_region_fn=train_sliding_window_region,
        predict_fn=predict_sliding_window_spectra,
        config=config,
        hooks=hooks,
    )
