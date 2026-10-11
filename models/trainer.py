"""Unified training and evaluation routines for spectral
transformation models.
"""

from __future__ import annotations

import copy, math, warnings
from typing import Any

import torch, torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.cost import CompositeSpectralLoss, NormalizedMSELoss, format_loss_log10
from models.dataset import SpectrumPairDataset, SpectrumPatchDataset, create_dataloaders, default_spectrum_collate
from models.deeplabv3 import DeepLabV3
from models.film import prepare_film_condition_tensor
from models.inference import predict_sliding_window_spectrum
from models.resnet import ResNet1D
from models.unet import ConventionalUNet1D, ResidualUNet1D


def _batch_cond(
    model: nn.Module, batch: dict[str, Any], dev: torch.device, n: int | None = None,
) -> dict[str, Any] | None:
    if not getattr(model, "use_film", False):
        return None
    num_samples = n if n is not None else len(batch["x"])
    return prepare_film_condition_tensor(
        batch.get("meta"), scale_x=batch.get("scale_x"), device=dev, num_samples=num_samples,
    )


def _run_epoch(
    model: nn.Module, loader: DataLoader, criterion: NormalizedMSELoss,
    device: torch.device, optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, num_batches = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for b in loader:
            x, y = b["x"].to(device), b["y"].to(device)
            mv = b.get("max_val")
            max_v = mv.to(device) if isinstance(mv, torch.Tensor) else None
            cond = _batch_cond(model, b, device)
            if is_train:
                optimizer.zero_grad()
            yp = model(x, cond=cond) if cond is not None else model(x)
            loss = criterion(yp, y, model=model if is_train else None, max_val=max_v)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            num_batches += 1
    if num_batches == 0:
        raise ValueError(f"{'Training' if is_train else 'Validation'} dataloader is empty.")
    return total_loss / num_batches


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None = None,
    criterion: NormalizedMSELoss | None = None, config: dict[str, Any] | None = None, **kwargs: Any,
) -> float:
    """Train the model for a single epoch.

    Parameters
    ----------
    model : nn.Module
        The neural network model.
    loader : DataLoader
        DataLoader for the training dataset.
    optimizer : torch.optim.Optimizer | None, optional
        Optimizer instance (or pass via config['optimizer']).
    criterion : NormalizedMSELoss | None, optional
        Loss function instance (or pass via config['criterion']).
    config : dict[str, Any] | None, optional
        Configuration dictionary containing 'optimizer', 'criterion',
        'device'.

    Returns
    -------
    float
        Average training loss across all batches.
    """
    cfg = {**(config or {}), **kwargs}
    opt = optimizer or cfg.get("optimizer")
    crit = criterion or cfg.get("criterion")
    if opt is None or crit is None:
        raise ValueError("train_one_epoch requires 'optimizer' and 'criterion'.")
    dev = cfg.get("device") or next((p.device for p in model.parameters()), "cpu")
    return _run_epoch(model, loader, crit, torch.device(dev), opt)


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: NormalizedMSELoss, config: dict[str, Any] | None = None,
) -> float:
    """Evaluate model reconstruction loss on validation dataset without
    L2 penalty.

    Parameters
    ----------
    model : nn.Module
        The neural network model.
    loader : DataLoader
        DataLoader for the validation dataset.
    criterion : NormalizedMSELoss
        Loss function instance.
    config : dict[str, Any] | None, optional
        Configuration dictionary containing 'device'.

    Returns
    -------
    float
        Average validation loss across all batches.
    """
    dev = (config or {}).get("device") or next((p.device for p in model.parameters()), "cpu")
    return _run_epoch(model, loader, criterion, torch.device(dev))


def evaluate_sliding_window(
    model: nn.Module, full_val_dataset: SpectrumPairDataset, criterion: NormalizedMSELoss | None = None,
    config: dict[str, Any] | None = None, **kwargs: Any,
) -> float:
    """Evaluate end-to-end normalized MSE directly on reconstructed
    full spectra.

    Parameters
    ----------
    model : nn.Module
        Trained patch model.
    full_val_dataset : SpectrumPairDataset
        Validation dataset containing full paired spectra.
    criterion : NormalizedMSELoss | None, optional
        Loss function instance (default NormalizedMSELoss()).
    config : dict[str, Any] | None, optional
        Configuration dictionary containing 'window_size', 'stride',
        'device'.

    Returns
    -------
    float
        Average normalized MSE across all reconstructed validation
        spectra.
    """
    if len(full_val_dataset) == 0:
        raise ValueError("full_val_dataset is empty.")
    crit, cfg = criterion or NormalizedMSELoss(), {**(config or {}), **kwargs}
    w_size = int(cfg.get("window_size", 15))
    s_step = int(cfg.get("stride", max(1, w_size // 2)))
    dev = cfg.get("device") or next((p.device for p in model.parameters()), "cpu")
    device = torch.device(dev) if isinstance(dev, str) else dev

    model.eval()
    p_cfg = dict(device=device, window_size=w_size, stride=s_step, normalization_mode="none", clamp_non_negative=True)
    total = 0.0

    for idx in range(len(full_val_dataset)):
        it = full_val_dataset[idx]
        rec = predict_sliding_window_spectrum(
            model, it["x"].cpu().numpy(), config=p_cfg, cond=_batch_cond(model, it, device, 1),
        )
        yp = torch.from_numpy(rec).float().unsqueeze(0).to(device)
        total += crit(yp, it["y"].unsqueeze(0).to(device), model=None).item()

    return total / len(full_val_dataset)


def train_model_region(
    train_dataset: SpectrumPairDataset, val_dataset: SpectrumPairDataset | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a neural network model for a single spectral region.

    Parameters
    ----------
    train_dataset : SpectrumPairDataset
        Training dataset for this region.
    val_dataset : SpectrumPairDataset | None
        Validation dataset for this region.
    config : dict[str, Any] | None, optional
        Configuration dictionary containing training hyperparameters.

    Returns
    -------
    dict[str, Any]
        Dictionary containing trained model, best_val_loss, best_epoch,
        history, and resolved configuration.
    """
    if len(train_dataset) == 0:
        raise ValueError("Cannot train on an empty train_dataset.")
    has_val: bool = val_dataset is not None and len(val_dataset) > 0

    cfg = dict(config or {})
    m_type = str(cfg.get("model_type", cfg.get("model", "resnet"))).lower()
    use_sw = bool(cfg.get("use_sliding_window", cfg.get("use_sw", False)))
    epochs = int(cfg.get("epochs", cfg.get("epoch", cfg.get("num_epochs", 50))))
    batch_size = int(cfg.get("batch_size", cfg.get("batch", 16)))
    lr, weight_decay = float(cfg.get("learning_rate", cfg.get("lr", 1e-3))), float(cfg.get("weight_decay", 0.0))
    l2_weight, verbose = float(cfg.get("l2_weight", cfg.get("l2", 1e-4))), bool(cfg.get("verbose", False))

    use_lr_sched = bool(cfg.get("use_lr_scheduler", cfg.get("lr_scheduler", True)))
    lr_factor = float(cfg.get("lr_reduce_factor", cfg.get("lr_factor", 0.9)))
    lr_pat = int(cfg.get("lr_scheduler_patience", cfg.get("lr_patience", 10)))
    min_lr = float(cfg.get("min_lr", cfg.get("lr_min", lr / 1000.0)))

    if not (0.0 < lr_factor < 1.0):
        raise ValueError(f"lr_reduce_factor must be strictly between 0 and 1, got {lr_factor}")
    if lr_pat < 1:
        raise ValueError(f"lr_scheduler_patience must be an integer >= 1, got {lr_pat}")
    if min_lr < 0.0:
        raise ValueError(f"min_lr cannot be negative, got {min_lr}")

    patience = int(cfg.get("early_stopping_patience", cfg.get("patience", (lr_pat * 2) if use_lr_sched else 10)))
    if use_lr_sched and ("early_stopping_patience" in cfg or "patience" in cfg) and patience <= lr_pat < epochs:
        warnings.warn(
            f"early_stopping_patience ({patience}) <= lr_scheduler_patience ({lr_pat}) while epochs > patience.",
            UserWarning, stacklevel=2,
        )

    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    full_len = len(train_dataset[0]["x"])
    if use_sw:
        is_p = isinstance(train_dataset, SpectrumPatchDataset)
        w_size = train_dataset.window_size if is_p else min(full_len, max(3, int(cfg.get("window_size", 15))))
        s_step = train_dataset.stride if is_p else max(1, min(w_size, int(cfg.get("stride", max(1, w_size // 2)))))
        train_ds = train_dataset if is_p else SpectrumPatchDataset(train_dataset, window_size=w_size, stride=s_step)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=default_spectrum_collate)
        seq_len, val_loader = w_size, None
    else:
        seq_len, w_size, s_step = full_len, full_len, full_len
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset if has_val else train_dataset,
            config={"batch_size": batch_size, "shuffle_train": True},
        )
        if not has_val:
            val_loader = None

    model = cfg.get("model")
    if model is None:
        m_map = dict(resnet=ResNet1D, residual_unet=ResidualUNet1D, unet=ConventionalUNet1D, deeplabv3=DeepLabV3)
        is_res, is_dl = "residual" in m_type, "deeplab" in m_type
        clean_m = "residual_unet" if is_res else ("deeplabv3" if is_dl else m_type.replace("_1d", ""))
        if clean_m not in m_map:
            raise ValueError(f"Unknown model_type '{m_type}'.")
        model = m_map[clean_m](n_points=seq_len, config=cfg)
    model = model.to(device)

    criterion = cfg.get("criterion")
    if criterion is None:
        cost_t = str(cfg.get("cost_type", cfg.get("loss_type", "normalized_mse"))).lower()
        loss_cls = CompositeSpectralLoss if "composite" in cost_t else NormalizedMSELoss
        criterion = loss_cls(config={**cfg, "l2_weight": l2_weight})
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=lr_pat, min_lr=min_lr)
        if use_lr_sched else None
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best_loss, best_epoch, best_state, pat_count = float("inf"), 0, None, 0
    run_cfg, eval_cfg = {"device": device}, {"device": device, "window_size": w_size, "stride": s_step}

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, config=run_cfg)
        val_loss = (
            (evaluate_sliding_window(model, val_dataset, criterion, config=eval_cfg) if use_sw
             else evaluate(model, val_loader, criterion, config=run_cfg)) if has_val else float("nan")
        )
        mon_loss = val_loss if has_val else tr_loss

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if math.isnan(mon_loss) or math.isinf(mon_loss):
            warnings.warn(f"Training diverged at epoch {epoch} with NaN/Inf loss ({mon_loss}). Stopping early.")
            break

        if mon_loss < best_loss:
            best_loss, best_epoch, best_state, pat_count = mon_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            pat_count += 1

        if scheduler is not None:
            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(mon_loss)
            cur_lr = optimizer.param_groups[0]["lr"]
            if cur_lr < prev_lr:
                if verbose:
                    lbl = "Val Loss" if has_val else "Train Loss"
                    print(f"Epoch {epoch:3d}: ReduceLROnPlateau reduced LR from {prev_lr:.2e} to {cur_lr:.2e} ({lbl}).")
                pat_count = 0

        if verbose and (epoch % 10 == 0 or epoch in (1, epochs)):
            tr_str = format_loss_log10(tr_loss)
            v_str = format_loss_log10(val_loss) if has_val else f"{tr_str} (No validation set; monitoring Train Loss)"
            print(f"Epoch {epoch:3d}/{epochs:3d} - Train Loss: {tr_str}, Val Loss: {v_str}")

        if pat_count >= patience:
            if verbose:
                t_str = "best val_loss" if has_val else "best train_loss"
                print(
                    f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, {t_str}: "
                    f"{format_loss_log10(best_loss)})"
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    cfg.update({
        "epochs": epochs, "batch_size": batch_size, "learning_rate": lr, "l2_weight": l2_weight,
        "early_stopping_patience": patience, "model_type": m_type, "use_sliding_window": use_sw,
        "use_lr_scheduler": use_lr_sched, "lr_reduce_factor": lr_factor, "lr_scheduler_patience": lr_pat,
        "min_lr": min_lr, "window_size": w_size, "stride": s_step, "seq_len": seq_len,
    })
    return {
        "model": model, "best_val_loss": best_loss, "best_epoch": best_epoch,
        "history": history, "config": cfg, "window_size": w_size, "stride": s_step,
    }
