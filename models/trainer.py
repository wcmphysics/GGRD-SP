"""Unified training and evaluation routines for spectral transformation models."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.cost import CompositeSpectralLoss, NormalizedMSELoss, format_loss_log10
from models.dataset import SpectrumPairDataset, SpectrumPatchDataset, create_dataloaders
from models.deeplabv3 import DeepLabV3
from models.film import prepare_film_condition_tensor
from models.inference import predict_sliding_window_spectrum
from models.resnet import ResNet1D
from models.unet import ConventionalUNet1D, ResidualUNet1D, UNet1D



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: NormalizedMSELoss,
    config: dict[str, Any] | None = None,
) -> float:
    """Train the model for a single epoch.

    Parameters
    ----------
    model : nn.Module
        The neural network model.
    loader : DataLoader
        DataLoader for the training dataset (batches of full spectra or patches).
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    criterion : NormalizedMSELoss
        Loss function instance.
    config : dict[str, Any] | None, optional
        Configuration dictionary containing 'device' (default auto from model).

    Returns
    -------
    float
        Average training loss across all batches.
    """
    cfg = config or {}
    device = cfg.get("device")
    if device is None:
        params = list(model.parameters())
        device = params[0].device if params else torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        max_v = batch.get("max_val")
        if max_v is not None and isinstance(max_v, torch.Tensor):
            max_v = max_v.to(device)

        cond = None
        if getattr(model, "use_film", False):
            cond = prepare_film_condition_tensor(
                metadata=batch.get("meta"),
                scale_x=batch.get("scale_x"),
                device=device,
                num_samples=len(x),
            )

        optimizer.zero_grad()
        y_pred = model(x, cond=cond) if cond is not None else model(x)
        loss = criterion(y_pred, y, model=model, max_val=max_v)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Training dataloader is empty.")

    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: NormalizedMSELoss,
    config: dict[str, Any] | None = None,
) -> float:
    """Evaluate model reconstruction loss on validation dataset without L2 penalty.

    Parameters
    ----------
    model : nn.Module
        The neural network model.
    loader : DataLoader
        DataLoader for the validation dataset.
    criterion : NormalizedMSELoss
        Loss function instance.
    config : dict[str, Any] | None, optional
        Configuration dictionary containing 'device' (default auto from model).

    Returns
    -------
    float
        Average validation loss across all batches.
    """
    cfg = config or {}
    device = cfg.get("device")
    if device is None:
        params = list(model.parameters())
        device = params[0].device if params else torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            max_v = batch.get("max_val")
            if max_v is not None and isinstance(max_v, torch.Tensor):
                max_v = max_v.to(device)

            cond = None
            if getattr(model, "use_film", False):
                cond = prepare_film_condition_tensor(
                    metadata=batch.get("meta"),
                    scale_x=batch.get("scale_x"),
                    device=device,
                    num_samples=len(x),
                )

            y_pred = model(x, cond=cond) if cond is not None else model(x)
            loss = criterion(y_pred, y, model=None, max_val=max_v)
            total_loss += loss.item()
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Validation dataloader is empty.")

    return total_loss / num_batches


def evaluate_sliding_window(
    model: nn.Module,
    full_val_dataset: SpectrumPairDataset,
    criterion: NormalizedMSELoss | None = None,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Evaluate end-to-end normalized MSE directly on reconstructed full spectra.

    Parameters
    ----------
    model : nn.Module
        Trained patch model.
    full_val_dataset : SpectrumPairDataset
        Validation dataset containing full paired spectra.
    criterion : NormalizedMSELoss | None, optional
        Loss function instance (default NormalizedMSELoss()).
    config : dict[str, Any] | None, optional
        Configuration dictionary containing 'window_size', 'stride', and 'device'.
    **kwargs : Any
        Optional keyword arguments for backward compatibility.

    Returns
    -------
    float
        Average normalized MSE across all reconstructed validation spectra.
    """
    if criterion is None:
        criterion = NormalizedMSELoss()
    cfg = dict(config or {})
    cfg.update(kwargs)
    w_size = int(cfg.get("window_size", 15))
    s_step = int(cfg.get("stride", max(1, w_size // 2)))

    device = cfg.get("device")
    if device is None:
        params = list(model.parameters())
        device = params[0].device if params else torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if len(full_val_dataset) == 0:
        raise ValueError("full_val_dataset is empty.")

    total_loss = 0.0
    model.eval()

    predict_cfg = {
        "device": device,
        "window_size": w_size,
        "stride": s_step,
        "normalize_by_source": False,  # dataset items are already normalized
        "clamp_non_negative": True,
    }

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


def train_model_region(
    train_dataset: SpectrumPairDataset,
    val_dataset: SpectrumPairDataset,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a neural network model for a single spectral region.

    Unifies ResNet, Residual U-Net, and conventional U-Net training, with
    optional sliding window augmentation and log10 loss progress logging.

    Parameters
    ----------
    train_dataset : SpectrumPairDataset
        Training dataset for this region.
    val_dataset : SpectrumPairDataset
        Validation dataset for this region.
    config : dict[str, Any] | None, optional
        Configuration dictionary containing:
        - 'model_type' (str): Architecture 'resnet', 'residual_unet', or 'unet' (default 'resnet').
        - 'use_sliding_window' (bool): Whether to use sliding window patch training (default False).
        - 'window_size' (int): Window points if sliding window active (default 15).
        - 'stride' (int): Stride points if sliding window active (default window_size // 2).
        - 'epochs' (int): Maximum training epochs (default 50).
        - 'batch_size' (int): Batch size (default 16).
        - 'learning_rate' (float): Optimizer learning rate (default 1e-3).
        - 'l2_weight' (float): L2 regularization weight (default 1e-4).
        - 'early_stopping_patience' (int): Patience epochs before stopping (default 10).
        - 'verbose' (bool): Whether to print progress in log10 scale (default False).
        - 'device' (str | torch.device | None): Device to use.
        - 'model' (nn.Module | None): Pre-instantiated model instance if available.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'model': Trained model (weights restored to lowest validation loss epoch).
        - 'best_val_loss': Lowest validation loss achieved.
        - 'best_epoch': Epoch index with lowest validation loss.
        - 'history': Dictionary of 'train_loss' and 'val_loss' histories.
        - 'config': Resolved configuration dictionary.
    """
    if len(train_dataset) == 0:
        raise ValueError("Cannot train on an empty train_dataset.")
    if len(val_dataset) == 0:
        raise ValueError("Cannot evaluate on an empty val_dataset.")

    cfg = config or {}
    model_type: str = str(cfg.get("model_type", cfg.get("model", "resnet"))).lower()
    use_sw: bool = bool(cfg.get("use_sliding_window", cfg.get("use_sw", False)))
    epochs: int = int(
        cfg.get("epochs", cfg.get("epoch", cfg.get("num_epochs", cfg.get("n_epochs", 50))))
    )
    batch_size: int = int(cfg.get("batch_size", cfg.get("batch", cfg.get("batchsize", 16))))
    lr: float = float(cfg.get("learning_rate", cfg.get("lr", 1e-3)))
    weight_decay: float = float(cfg.get("weight_decay", 0.0))
    l2_weight: float = float(cfg.get("l2_weight", cfg.get("l2", cfg.get("l2_reg", 1e-4))))
    patience: int = int(
        cfg.get(
            "early_stopping_patience",
            cfg.get("patience", cfg.get("early_stop_patience", 10)),
        )
    )
    verbose: bool = bool(cfg.get("verbose", False))

    # Synchronize canonical and alias keys in cfg dictionary
    cfg["epochs"] = epochs
    cfg["batch_size"] = batch_size
    cfg["learning_rate"] = lr
    cfg["l2_weight"] = l2_weight
    cfg["early_stopping_patience"] = patience
    cfg["model_type"] = model_type
    cfg["use_sliding_window"] = use_sw

    device_str = cfg.get("device")
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device_str, str):
        device = torch.device(device_str)
    else:
        device = device_str

    # Determine sequence length & sliding window setup
    sample_item = train_dataset[0]
    full_len = len(sample_item["x"])

    if use_sw:
        if isinstance(train_dataset, SpectrumPatchDataset):
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            w_size = train_dataset.window_size
            s_step = train_dataset.stride
            seq_len = w_size
            val_loader = None
        else:
            w_size = int(cfg.get("window_size", cfg.get("window_size_points", 15)))
            s_step = int(cfg.get("stride", cfg.get("sliding_stride_points", max(1, w_size // 2))))
            w_size = min(full_len, max(3, w_size))
            s_step = max(1, min(w_size, s_step))
            seq_len = w_size

            train_patch_ds = SpectrumPatchDataset(train_dataset, window_size=w_size, stride=s_step)
            train_loader = DataLoader(train_patch_ds, batch_size=batch_size, shuffle=True)
            val_loader = None
    else:
        seq_len = full_len
        w_size = full_len
        s_step = full_len
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, config={"batch_size": batch_size, "shuffle_train": True}
        )

    # Instantiate model if not provided
    model: nn.Module | None = cfg.get("model")
    if model is None:
        if model_type == "resnet":
            model = ResNet1D(n_points=seq_len, config=cfg).to(device)
        elif model_type in ("residual_unet", "unet_residual"):
            model = ResidualUNet1D(n_points=seq_len, config=cfg).to(device)
        elif model_type == "unet":
            model = ConventionalUNet1D(n_points=seq_len, config=cfg).to(device)
        elif model_type in ("deeplabv3", "deeplab", "deeplabv3_1d"):
            model = DeepLabV3(n_points=seq_len, config=cfg).to(device)
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Choose 'resnet', 'residual_unet', 'unet', or 'deeplabv3'."
            )
    else:
        model = model.to(device)

    criterion: nn.Module | None = cfg.get("criterion")
    if criterion is None:
        cost_type = str(cfg.get("cost_type", cfg.get("loss_type", "normalized_mse"))).lower()
        loss_cfg = dict(cfg)
        loss_cfg.setdefault("l2_weight", l2_weight)
        if cost_type in ("composite", "composite_spectral"):
            criterion = CompositeSpectralLoss(config=loss_cfg)
        else:
            criterion = NormalizedMSELoss(config=loss_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, Any] | None = None
    patience_counter = 0

    run_cfg = {"device": device}
    eval_cfg = {"device": device, "window_size": w_size, "stride": s_step}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config=run_cfg)

        if use_sw:
            val_loss = evaluate_sliding_window(model, val_dataset, criterion, config=eval_cfg)
        else:
            assert val_loader is not None
            val_loss = evaluate(model, val_loader, criterion, config=run_cfg)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % 10 == 0 or epoch == epochs or epoch == 1):
            tr_str = format_loss_log10(train_loss)
            val_str = format_loss_log10(val_loss)
            print(
                f"Epoch {epoch:3d}/{epochs:3d} - Train Loss: {tr_str}, Val Loss: {val_str}"
            )

        if patience_counter >= patience:
            if verbose:
                b_str = format_loss_log10(best_val_loss)
                print(
                    f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, best val_loss: {b_str})"
                )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    resolved_cfg = dict(cfg)
    resolved_cfg.update({"window_size": w_size, "stride": s_step, "seq_len": seq_len})

    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
        "config": resolved_cfg,
        "window_size": w_size,
        "stride": s_step,
    }

