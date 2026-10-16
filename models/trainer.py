"""Unified training and evaluation routines for spectral
transformation models.
"""

from __future__ import annotations

import copy, math, warnings
from typing import Any

import torch, torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.cost import SpectralLoss, format_loss_log10
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
    energy_scale_ev = float(getattr(model, "energy_scale_ev", 1000.0))
    return prepare_film_condition_tensor(
        batch.get("meta"),
        scale_x=batch.get("scale_x"),
        window_energy=batch.get("window_energy"),
        device=dev,
        num_samples=num_samples,
        config={"energy_scale_ev": energy_scale_ev},
    )


def _run_epoch(
    model: nn.Module, loader: DataLoader, criterion: SpectralLoss,
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
    criterion: SpectralLoss | None = None, config: dict[str, Any] | None = None, **kwargs: Any,
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
    criterion : SpectralLoss | None, optional
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
    model: nn.Module, loader: DataLoader, criterion: SpectralLoss, config: dict[str, Any] | None = None,
) -> float:
    """Evaluate model reconstruction loss on validation dataset without
    L2 penalty.

    Parameters
    ----------
    model : nn.Module
        The neural network model.
    loader : DataLoader
        DataLoader for the validation dataset.
    criterion : SpectralLoss
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
    model: nn.Module, full_val_dataset: SpectrumPairDataset, criterion: SpectralLoss | None = None,
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
    criterion : SpectralLoss | None, optional
        Loss function instance (default SpectralLoss()).
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
    crit, cfg = criterion or SpectralLoss(), {**(config or {}), **kwargs}
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
            model,
            it["x"].cpu().numpy().squeeze(),
            config=p_cfg,
            cond=_batch_cond(model, it, device, 1),
            energy=it.get("energy"),
        )
        yp = torch.from_numpy(rec).float().view(1, 1, -1).to(device)
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
    disallowed_aliases = {
        "lr": "learning_rate",
        "batch": "batch_size",
        "batch_sizes": "batch_size",
        "window_size_choices": "window_size",
        "kernel_sizes": "kernel_size",
        "depths": "depth",
        "resnet_channels": "resnet_channel",
        "residual_unet_channels": "residual_unet_channel",
        "unet_channels": "unet_channel",
        "deeplabv3_backbone_channels": "deeplabv3_backbone_channel",
        "deeplabv3_aspp_channels": "deeplabv3_aspp_channel",
        "hidden_channels": "resnet_channel",
        "hidden_channel": "resnet_channel",
        "base_channels": "residual_unet_channel / unet_channel",
        "base_channel": "residual_unet_channel / unet_channel",
        "backbone_channels": "deeplabv3_backbone_channel",
        "backbone_channel": "deeplabv3_backbone_channel",
        "aspp_channels": "deeplabv3_aspp_channel",
        "aspp_channel": "deeplabv3_aspp_channel",
        "epoch": "epochs",
        "num_epochs": "epochs",
        "l2": "l2_weight",
        "use_sw": "use_sliding_window",
        "lr_scheduler": "use_learning_rate_scheduler",
        "lr_factor": "learning_rate_reduce_factor",
        "lr_patience": "learning_rate_scheduler_patience",
        "lr_min": "min_learning_rate",
        "patience": "early_stopping_patience",
    }
    for alias, canonical in disallowed_aliases.items():
        if alias in cfg:
            raise KeyError(
                f"Missing required canonical key '{canonical}'. "
                f"Found deprecated or plural alias '{alias}'."
            )

    model_type = str(cfg.get("model_type", "resnet")).lower()
    use_sw = bool(cfg.get("use_sliding_window", False))

    raw_epochs = cfg.get("epochs", 50)
    if isinstance(raw_epochs, bool) or not isinstance(raw_epochs, int) or raw_epochs <= 0:
        raise ValueError(f"epochs must be a positive integer, got {raw_epochs!r}")
    epochs = raw_epochs

    raw_batch_size = cfg.get("batch_size", 16)
    if isinstance(raw_batch_size, bool) or not isinstance(raw_batch_size, int) or raw_batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {raw_batch_size!r}")
    batch_size = raw_batch_size

    for int_key in (
        "window_size_choice",
        "window_size",
        "stride",
        "depth",
        "resnet_channel",
        "residual_unet_channel",
        "unet_channel",
        "deeplabv3_backbone_channel",
        "deeplabv3_aspp_channel",
    ):
        if int_key in cfg:
            raw_val = cfg[int_key]
            if isinstance(raw_val, bool) or not isinstance(raw_val, int) or raw_val < 1:
                raise ValueError(f"{int_key} must be a positive integer, got {raw_val!r}")

    learning_rate = float(cfg.get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    l2_weight = float(cfg.get("l2_weight", 1e-4))
    verbose = bool(cfg.get("verbose", False))

    use_lr_sched = bool(
        cfg.get(
            "use_learning_rate_scheduler",
            cfg.get("use_lr_scheduler", True),
        )
    )
    learning_rate_reduce_factor = float(
        cfg.get(
            "learning_rate_reduce_factor",
            cfg.get("lr_reduce_factor", 0.9),
        )
    )
    learning_rate_scheduler_patience = int(
        cfg.get(
            "learning_rate_scheduler_patience",
            cfg.get("lr_scheduler_patience", 10),
        )
    )
    min_learning_rate = float(
        cfg.get(
            "min_learning_rate",
            cfg.get("min_lr", learning_rate / 1000.0),
        )
    )

    if not (0.0 < learning_rate_reduce_factor < 1.0):
        raise ValueError(
            f"learning_rate_reduce_factor must be strictly between 0 and 1, got {learning_rate_reduce_factor}"
        )
    if learning_rate_scheduler_patience < 1:
        raise ValueError(
            f"learning_rate_scheduler_patience must be an integer >= 1, got {learning_rate_scheduler_patience}"
        )
    if min_learning_rate < 0.0:
        raise ValueError(
            f"min_learning_rate cannot be negative, got {min_learning_rate}"
        )

    patience = int(
        cfg.get(
            "early_stopping_patience",
            (learning_rate_scheduler_patience * 2) if use_lr_sched else 10,
        )
    )
    if (
        use_lr_sched
        and "early_stopping_patience" in cfg
        and patience <= learning_rate_scheduler_patience < epochs
    ):
        warnings.warn(
            f"early_stopping_patience ({patience}) <= "
            f"learning_rate_scheduler_patience ({learning_rate_scheduler_patience}) "
            f"while epochs > patience.",
            UserWarning,
            stacklevel=2,
        )

    device = torch.device(
        cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    full_len = int(train_dataset[0]["x"].shape[-1])
    explicit_ws = cfg.get("window_size_choice", cfg.get("window_size"))
    if use_sw:
        is_p = isinstance(train_dataset, SpectrumPatchDataset)
        ws_cfg_val = explicit_ws if explicit_ws is not None else 15
        if ws_cfg_val < 3:
            raise ValueError(f"window_size must be >= 3 when sliding window is active, got {ws_cfg_val}")
        w_size = train_dataset.window_size if is_p else min(full_len, ws_cfg_val)
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

    effective_max_k = min(seq_len, explicit_ws) if explicit_ws is not None else seq_len
    if "kernel_size" in cfg:
        raw_k = cfg["kernel_size"]
        if isinstance(raw_k, bool) or not isinstance(raw_k, int) or raw_k <= 0 or raw_k % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {raw_k!r}")
        if raw_k > effective_max_k:
            raise ValueError(
                f"kernel_size ({raw_k}) cannot be larger than window_size / n_points ({effective_max_k})."
            )

    model = cfg.get("model")
    if model is None:
        m_map = {
            "resnet": ResNet1D,
            "residual_unet": ResidualUNet1D,
            "unet": ConventionalUNet1D,
            "deeplabv3": DeepLabV3,
        }
        is_res = "residual" in model_type
        is_dl = "deeplab" in model_type
        clean_m = "residual_unet" if is_res else ("deeplabv3" if is_dl else model_type.replace("_1d", ""))
        if clean_m not in m_map:
            raise ValueError(f"Unknown model_type '{model_type}'.")
        model = m_map[clean_m](n_points=seq_len, config=cfg)
    model = model.to(device)

    criterion = cfg.get("criterion")
    if criterion is None:
        criterion = SpectralLoss(config={**cfg, "l2_weight": l2_weight})
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = (
        ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=learning_rate_reduce_factor,
            patience=learning_rate_scheduler_patience,
            min_lr=min_learning_rate,
        )
        if use_lr_sched
        else None
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }
    history["lr"] = history["learning_rate"]
    best_loss, best_epoch, best_state, pat_count = float("inf"), 0, None, 0
    run_cfg = {"device": device}
    eval_cfg = {"device": device, "window_size": w_size, "stride": s_step}

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, config=run_cfg)
        val_loss = (
            (evaluate_sliding_window(model, val_dataset, criterion, config=eval_cfg) if use_sw
             else evaluate(model, val_loader, criterion, config=run_cfg)) if has_val else float("nan")
        )
        mon_loss = val_loss if has_val else tr_loss

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

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
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "l2_weight": l2_weight,
        "early_stopping_patience": patience,
        "model_type": model_type,
        "use_sliding_window": use_sw,
        "use_learning_rate_scheduler": use_lr_sched,
        "learning_rate_reduce_factor": learning_rate_reduce_factor,
        "learning_rate_scheduler_patience": learning_rate_scheduler_patience,
        "min_learning_rate": min_learning_rate,
        "use_lr_scheduler": use_lr_sched,
        "lr_reduce_factor": learning_rate_reduce_factor,
        "lr_scheduler_patience": learning_rate_scheduler_patience,
        "min_lr": min_learning_rate,
        "window_size": w_size,
        "stride": s_step,
        "seq_len": seq_len,
    })
    return {
        "model": model,
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "history": history,
        "config": cfg,
        "window_size": w_size,
        "stride": s_step,
    }
