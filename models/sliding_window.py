"""Sliding window neural network model and pipeline for sequence-to-sequence spectral transfer."""

from __future__ import annotations

import copy
import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ax.service.ax_client import AxClient, ObjectiveProperties
from torch.utils.data import DataLoader, Dataset

from models.baseline_cnn import NormalizedMSELoss, Residual1DCNN
from models.dataset import SpectrumPairDataset, split_session_datasets
from models.inference import format_predicted_measurement_id


def calculate_window_points(
    energy: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Calculate window size and sliding stride in data points from binding energy grid.

    Parameters
    ----------
    energy : np.ndarray
        1D array of binding energies for the spectral region.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'window_size_ev' (float): Desired window width in eV (default 2.0).
        - 'sliding_stride_ev' (float): Desired sliding stride in eV (default 1.0).
        - 'window_size_points' | 'window_size' (int | None): Explicit window size in points.
        - 'sliding_stride_points' | 'stride' (int | None): Explicit stride in points.
        - 'force_odd_window' (bool): Ensure window size is an odd integer (default True).

    Returns
    -------
    tuple[int, int]
        (window_size_points, sliding_stride_points)
    """
    cfg = config or {}
    w_pts = cfg.get("window_size_points", cfg.get("window_size"))
    s_pts = cfg.get("sliding_stride_points", cfg.get("stride"))

    n_total = len(energy)
    if n_total < 3:
        raise ValueError(f"Energy array length must be >= 3, got {n_total}")

    delta_e = float(np.mean(np.abs(np.diff(energy))))
    if delta_e <= 0.0:
        raise ValueError("Energy spacing delta_e must be strictly positive.")

    # Determine window size in points
    if w_pts is not None:
        window_size = int(w_pts)
    else:
        window_size_ev = float(cfg.get("window_size_ev", 2.0))
        window_size = int(round(window_size_ev / delta_e))

    force_odd = bool(cfg.get("force_odd_window", True))
    window_size = min(n_total, window_size)
    if force_odd and window_size % 2 == 0:
        window_size = window_size - 1 if window_size == n_total else window_size + 1

    window_size = max(3, window_size)

    # Determine sliding stride in points
    if s_pts is not None:
        stride = int(s_pts)
    else:
        stride_ev = float(cfg.get("sliding_stride_ev", 1.0))
        stride = int(round(stride_ev / delta_e))

    stride = max(1, min(window_size, stride))

    return window_size, stride


def extract_sliding_windows(
    spectrum: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, list[int]]:
    """Slice a 1D regional spectrum into overlapping window patches without padding.

    Extracts valid sub-sequences [start : start + window_size] directly from the
    unpadded spectrum. Anchors the final window to the right edge (N - window_size)
    if not aligned by stride, guaranteeing complete 100% spectral coverage using
    only real physical measurements without artificial boundary padding.

    Parameters
    ----------
    spectrum : np.ndarray
        1D array of spectral intensities of length N (squeezed/raveled if 2D).
    window_size : int
        Number of points in each window (W).
    stride : int
        Step size in points between consecutive window starts (S).

    Returns
    -------
    tuple[np.ndarray, list[int]]
        - windows: 2D array of shape (N_windows, window_size).
        - start_indices: List of starting indices in the spectrum.

    Raises
    ------
    ValueError
        If window_size < 1, window_size > len(spectrum), stride < 1, or stride > window_size.
    """
    spectrum_1d = np.asarray(spectrum).ravel()
    n_points = len(spectrum_1d)
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if window_size > n_points:
        raise ValueError(
            f"window_size ({window_size}) cannot exceed spectrum length ({n_points})"
        )
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if stride > window_size:
        raise ValueError(
            f"stride ({stride}) cannot exceed window_size ({window_size}) "
            "as it creates reconstruction gaps."
        )

    start_indices: list[int] = []
    current_idx = 0
    while current_idx + window_size <= n_points:
        start_indices.append(current_idx)
        current_idx += stride

    # Ensure the rightmost edge of the spectrum is fully covered
    last_possible = n_points - window_size
    if start_indices and start_indices[-1] < last_possible:
        start_indices.append(last_possible)

    windows_list = [spectrum_1d[j : j + window_size] for j in start_indices]
    windows = np.vstack(windows_list).astype(np.float32)

    return windows, start_indices


def reconstruct_from_patches(
    patches: np.ndarray | torch.Tensor,
    start_indices: list[int],
    original_length: int,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Reconstruct a full 1D spectrum by accumulating and averaging overlapping predicted patches.

    Bundles optional parameters into config dictionary adhering to project interface rules.

    Parameters
    ----------
    patches : np.ndarray | torch.Tensor
        2D array of predicted window patches of shape (N_windows, window_size).
    start_indices : list[int]
        Starting index of each window in the spectrum sequence.
    original_length : int
        Length of the target spectrum (N).
    config : dict[str, Any] | None, optional
        Configuration dictionary containing optional 'window_size'.
    **kwargs : Any
        Optional keyword arguments (e.g. window_size) for backward compatibility.

    Returns
    -------
    np.ndarray
        Reconstructed 1D spectrum of length original_length.
    """
    cfg = dict(config or {})
    cfg.update(kwargs)

    if isinstance(patches, torch.Tensor):
        patches_np = patches.detach().cpu().numpy()
    else:
        patches_np = np.asarray(patches)
    patches_np = np.atleast_2d(patches_np)

    w_size = int(cfg.get("window_size", patches_np.shape[1]))

    accum = np.zeros(original_length, dtype=np.float32)
    counts = np.zeros(original_length, dtype=np.float32)

    for patch, j in zip(patches_np, start_indices):
        accum[j : j + w_size] += patch
        counts[j : j + w_size] += 1.0

    return accum / np.maximum(counts, 1.0)


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

    if not verbose:
        logging.getLogger("ax").setLevel(logging.WARNING)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        ax_client = AxClient(random_seed=seed)

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
            name="sliding_window_optimization",
            parameters=parameters,
            objectives={"val_loss": ObjectiveProperties(minimize=True)},
        )

    trials_data: list[dict[str, Any]] = []

    for trial_idx in range(num_trials):
        if seed is not None:
            torch.manual_seed(seed + trial_idx)

        params, trial_id = ax_client.get_next_trial()
        win_size = int(params["window_size"])
        k_size = int(params["kernel_size"])
        if k_size > win_size:
            k_size = win_size if win_size % 2 == 1 else win_size - 1

        train_cfg = {
            "window_size": win_size,
            "stride": max(1, win_size // 2),
            "kernel_size": k_size,
            "hidden_channels": int(params["hidden_channels"]),
            "learning_rate": float(params["learning_rate"]),
            "l2_weight": float(params["l2_weight"]),
            "epochs": epochs_per_trial,
            "device": device,
            "verbose": False,
        }

        train_res = train_sliding_window_region(train_full_ds, val_full_ds, config=train_cfg)
        val_loss = float(train_res["best_val_loss"])

        ax_client.complete_trial(trial_index=trial_id, raw_data={"val_loss": val_loss})

        trials_data.append({
            "trial_id": trial_id,
            "parameters": params,
            "val_loss": val_loss,
            "best_epoch": train_res["best_epoch"],
        })

        if verbose:
            print(
                f"[Ax Sliding Window Trial {trial_idx + 1}/{num_trials}] "
                f"Params: {params} => Val Loss: {val_loss:.6f}"
            )

    best_params, _ = ax_client.get_best_parameters()
    best_val_loss = min(t["val_loss"] for t in trials_data)

    return {
        "best_parameters": best_params,
        "best_val_loss": best_val_loss,
        "trials_data": trials_data,
        "ax_client": ax_client,
    }


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
        orig_meas_id = str(row["measurement_id"])

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

        pred_meas_id = format_predicted_measurement_id(orig_meas_id, source_tool, target_tool)
        meta_rec = {
            "spectrum_index": new_idx,
            "material": row.get("material", "NMG"),
            "tool": target_tool,
            "measurement_id": pred_meas_id,
            "die": row.get("die", 0),
            "region": region,
            "n_points": n_points,
            "time": row.get("time"),
            "source_tool": source_tool,
            "source_measurement_id": orig_meas_id,
            "source_spectrum_index": orig_idx,
            "is_predicted": True,
        }
        predicted_meta_records.append(meta_rec)

    meta_df_predicted = pd.DataFrame(predicted_meta_records)
    return predicted_intensities, predicted_energies, meta_df_predicted


def run_sliding_window_pipeline(
    meta_df: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the end-to-end sliding window modeling pipeline.

    Orchestrates:
    1. Session-level dataset splitting per spectral region.
    2. Dynamic computation of window size and stride from binding energy resolution.
    3. Optional Ax Bayesian hyperparameter optimization on reconstructed full spectra.
    4. Model instantiation and training using sliding window patches.
    5. Evaluation directly on the reconstructed full spectrum.
    6. Prediction generation producing standardized output containers:
       (ary_intensity_predicted, ary_energy_predicted, meta_df_predicted).

    Parameters
    ----------
    meta_df : pd.DataFrame
        Metadata DataFrame containing source-to-target pairing information.
    ary_intensity : np.ndarray
        2D array of measured intensities.
    ary_energy : np.ndarray
        2D array of binding energies.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'regions' (list[str] | None): Regions to train (default auto-detected).
        - 'source_tool' (str): Source tool identifier (default 'J4').
        - 'target_tool' (str): Target tool identifier (default 'H1').
        - 'val_ratio' (float): Fraction of sessions reserved for validation (default 0.2).
        - 'seed' (int | None): Random seed for reproducibility (default 42).
        - 'window_size_ev' (float): Desired window width in eV (default 2.0).
        - 'sliding_stride_ev' (float): Desired sliding stride in eV (default 1.0).
        - 'use_bayesian_opt' (bool): Whether to perform Ax Bayesian optimization (default False).
        - 'bayesian_opt_config' (dict[str, Any] | None): Configuration for Ax optimization.
        - 'train_config' (dict[str, Any] | None): Training hyperparameters.
        - 'predict_source' (bool): Whether to run inference on source data (default True).
        - 'verbose' (bool): Whether to print progress (default False).

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'models': Dictionary mapping region -> (model, window_size, stride).
        - 'evaluation': Dictionary of validation losses per region (evaluated on full spectra).
        - 'histories': Training history per region.
        - 'bayesian_opt_results': Results from Ax (if use_bayesian_opt=True).
        - 'predictions': Tuple of (ary_intensity_predicted, ary_energy_predicted, meta_df_predicted).
    """
    cfg = config or {}
    source_tool: str = cfg.get("source_tool", "J4")
    target_tool: str = cfg.get("target_tool", "H1")
    val_ratio: float = float(cfg.get("val_ratio", 0.2))
    seed: int | None = cfg.get("seed", 42)
    use_bo: bool = bool(cfg.get("use_bayesian_opt", False))
    bo_cfg: dict[str, Any] = dict(cfg.get("bayesian_opt_config", {}))
    train_cfg: dict[str, Any] = dict(cfg.get("train_config", {}))
    predict_source: bool = bool(cfg.get("predict_source", True))
    verbose: bool = bool(cfg.get("verbose", False))

    if "seed" not in bo_cfg and seed is not None:
        bo_cfg["seed"] = seed

    for k in (
        "window_size_ev",
        "sliding_stride_ev",
        "window_size",
        "window_size_points",
        "stride",
        "sliding_stride_points",
    ):
        if k in cfg and k not in train_cfg:
            train_cfg[k] = cfg[k]

    regions: list[str] | None = cfg.get("regions")
    if regions is None:
        paired_mask = meta_df["measurement_id_target"].notna()
        if source_tool:
            paired_mask = paired_mask & (meta_df["tool"] == source_tool)
        regions = sorted(meta_df[paired_mask]["region"].unique().tolist())

    if not regions:
        raise ValueError(
            f"No valid spectral regions found to train for "
            f"source_tool='{source_tool}', target_tool='{target_tool}'"
        )

    models: dict[str, tuple[nn.Module, int, int]] = {}
    eval_results: dict[str, float] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    bo_results: dict[str, dict[str, Any]] = {}

    for region in regions:
        if verbose:
            print(f"\n--- [Sliding Window] Processing Region: {region} ({source_tool} -> {target_tool}) ---")

        split_cfg = {
            "region": region,
            "source_tool": source_tool,
            "target_tool": target_tool,
            "val_ratio": val_ratio,
            "seed": seed,
        }
        train_ds, val_ds = split_session_datasets(
            meta_df, ary_intensity, ary_energy, config=split_cfg
        )

        region_train_cfg = dict(train_cfg)

        if use_bo:
            if verbose:
                print(f"Running Ax Bayesian Optimization for {region}...")
            bo_res = optimize_sliding_window_hyperparameters(train_ds, val_ds, config=bo_cfg)
            bo_results[region] = bo_res
            best_params = bo_res["best_parameters"]
            region_train_cfg.update(best_params)
            if verbose:
                print(f"Ax optimal params for {region}: {best_params}")

        # Train sliding window model
        train_res = train_sliding_window_region(train_ds, val_ds, config=region_train_cfg)
        models[region] = (train_res["model"], train_res["window_size"], train_res["stride"])
        eval_results[region] = train_res["best_val_loss"]
        histories[region] = train_res["history"]

        if verbose:
            print(
                f"Region {region} training complete (W={train_res['window_size']}, S={train_res['stride']}). "
                f"Best Reconstructed Val Loss: {train_res['best_val_loss']:.6f}"
            )

    predictions = None
    if predict_source:
        pred_cfg = {
            "source_tool": source_tool,
            "target_tool": target_tool,
        }
        predictions = predict_sliding_window_spectra(
            models=models,
            data=(ary_intensity, ary_energy, meta_df),
            config=pred_cfg,
        )

    return {
        "models": models,
        "evaluation": eval_results,
        "histories": histories,
        "bayesian_opt_results": bo_results,
        "predictions": predictions,
    }
