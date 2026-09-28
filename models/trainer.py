"""Training and evaluation routines for baseline 1D Residual CNN."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.baseline_cnn import NormalizedMSELoss, Residual1DCNN
from models.dataset import SpectrumPairDataset, create_dataloaders


def train_one_epoch(
    model: Residual1DCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: NormalizedMSELoss,
    config: dict[str, Any] | None = None,
) -> float:
    """Train the model for a single epoch.

    Parameters
    ----------
    model : Residual1DCNN
        The neural network model.
    loader : DataLoader
        DataLoader for the training dataset.
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
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y, model=model)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Training dataloader is empty.")

    return total_loss / num_batches


def evaluate(
    model: Residual1DCNN,
    loader: DataLoader,
    criterion: NormalizedMSELoss,
    config: dict[str, Any] | None = None,
) -> float:
    """Evaluate model reconstruction loss on validation dataset without L2 penalty.

    Parameters
    ----------
    model : Residual1DCNN
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
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_pred = model(x)
            # Pass model=None so evaluation strictly measures out-of-sample normalized MSE
            loss = criterion(y_pred, y, model=None)
            total_loss += loss.item()
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Validation dataloader is empty.")

    return total_loss / num_batches


def train_baseline_region(
    train_dataset: SpectrumPairDataset,
    val_dataset: SpectrumPairDataset,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a baseline Residual1DCNN for a single spectral region.

    Parameters
    ----------
    train_dataset : SpectrumPairDataset
        Training dataset.
    val_dataset : SpectrumPairDataset
        Validation dataset.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'n_points' (int | None): Sequence length (default auto-detected from dataset).
        - 'hidden_channels' (int): Feature channels (default 32).
        - 'kernel_size' (int): Conv kernel size, must be odd (default 5).
        - 'dropout' (float): Dropout probability (default 0.0).
        - 'use_batch_norm' (bool): Whether to use BatchNorm1d (default True).
        - 'l2_weight' (float): Regularization weight on model weights (default 1e-4).
        - 'learning_rate' (float): Optimizer learning rate (default 1e-3).
        - 'weight_decay' (float): Optimizer weight decay (default 0.0).
        - 'batch_size' (int): Batch size (default 16).
        - 'epochs' (int): Maximum training epochs (default 50).
        - 'early_stopping_patience' (int): Patience epochs before stopping (default 10).
        - 'device' (str | torch.device | None): Device to use (default CPU/CUDA auto).
        - 'verbose' (bool): Whether to print progress each epoch (default False).

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'model': Trained Residual1DCNN model (with weights restored to best epoch).
        - 'best_val_loss': Best validation loss achieved.
        - 'best_epoch': Epoch index with lowest validation loss.
        - 'history': Dictionary of 'train_loss' and 'val_loss' histories.
        - 'config': Resolved configuration used during training.
    """
    if len(train_dataset) == 0:
        raise ValueError("Cannot train on an empty train_dataset.")
    if len(val_dataset) == 0:
        raise ValueError("Cannot evaluate on an empty val_dataset.")

    cfg = config or {}
    sample_item = train_dataset[0]
    n_points = int(cfg.get("n_points", len(sample_item["x"])))

    epochs: int = int(cfg.get("epochs", 50))
    batch_size: int = int(cfg.get("batch_size", 16))
    lr: float = float(cfg.get("learning_rate", 1e-3))
    weight_decay: float = float(cfg.get("weight_decay", 0.0))
    l2_weight: float = float(cfg.get("l2_weight", 1e-4))
    patience: int = int(cfg.get("early_stopping_patience", 10))
    verbose: bool = bool(cfg.get("verbose", False))

    device_str = cfg.get("device")
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device_str, str):
        device = torch.device(device_str)
    else:
        device = device_str

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        config={"batch_size": batch_size, "shuffle_train": True},
    )

    # Initialize model, loss, optimizer
    model = Residual1DCNN(n_points=n_points, config=cfg).to(device)
    criterion = NormalizedMSELoss(l2_weight=l2_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, Any] | None = None
    patience_counter = 0

    run_cfg = {"device": device}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config=run_cfg)
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

        if verbose and (epoch % 10 == 0 or epoch == epochs):
            print(f"Epoch {epoch:3d}/{epochs:3d} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, best val_loss: {best_val_loss:.6f})")
            break

    # Restore best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
        "config": cfg,
    }
