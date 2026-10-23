"""Unified training and evaluation routines for spectral
transformation models.
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.cost import SpectralLoss, format_loss_log10
from models.dataset import (
    SpectrumPairDataset,
    SpectrumPatchDataset,
    create_dataloaders,
    default_spectrum_collate,
)
from models.deeplabv3 import DeepLabV3
from models.film import prepare_film_condition_tensor
from models.inference import predict_sliding_window_spectrum
from models.layers import validate_odd_kernel_size, validate_positive_int
from models.resnet import ResNet1D, ResNet5L1D, ResNetScaledShort1D
from models.unet import ConventionalUNet1D, ResidualUNet1D

_MODEL_CLASS_MAP: dict[str, type[nn.Module]] = {
    "resnet": ResNet1D,
    "resnet_5l": ResNet5L1D,
    "resnet_scaled_short": ResNetScaledShort1D,
    "residual_unet": ResidualUNet1D,
    "unet": ConventionalUNet1D,
    "deeplabv3": DeepLabV3,
}

_DISALLOWED_TRAINER_ALIASES: dict[str, str] = {
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
    "use_lr_scheduler": "use_learning_rate_scheduler",
    "lr_scheduler": "use_learning_rate_scheduler",
    "lr_reduce_factor": "learning_rate_reduce_factor",
    "lr_factor": "learning_rate_reduce_factor",
    "lr_scheduler_patience": "learning_rate_scheduler_patience",
    "lr_patience": "learning_rate_scheduler_patience",
    "min_lr": "min_learning_rate",
    "lr_min": "min_learning_rate",
    "patience": "early_stopping_patience",
}

_POSITIVE_INT_CONFIG_KEYS = (
    "window_size_choice",
    "window_size",
    "stride",
    "depth",
    "resnet_channel",
    "residual_unet_channel",
    "unet_channel",
    "deeplabv3_backbone_channel",
    "deeplabv3_aspp_channel",
)


def instantiate_model(
    model_type: str,
    seq_len: int,
    config: dict[str, Any] | None = None,
) -> nn.Module:
    """Instantiate a 1D spectral neural network by architecture name and
    sequence length.

    Parameters
    ----------
    model_type : str
        Architecture name: 'resnet', 'resnet_5L', 'resnet_scaled_short',
        'residual_unet', 'unet', or 'deeplabv3'.
    seq_len : int
        Input sequence length in data points (full regional spectrum or
        patch length).
    config : dict[str, Any] | None, optional
        Dictionary of model architecture hyperparameters (e.g., kernel_size,
        resnet_channel, residual_unet_channel, unet_channel, depth, use_film).

    Returns
    -------
    nn.Module
        Instantiated PyTorch neural network module ready for training.
    """
    model_class = _MODEL_CLASS_MAP.get(model_type.lower())
    if model_class is None:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose 'resnet', 'resnet_5L', "
            "'resnet_scaled_short', 'residual_unet', 'unet', or 'deeplabv3'."
        )
    return model_class(n_points=seq_len, config=config or {})


def _batch_cond(
    model: nn.Module,
    batch: dict[str, Any],
    dev: torch.device,
    n: int | None = None,
) -> dict[str, Any] | None:
    if not getattr(model, "use_film", False):
        return None
    num_samples = n if n is not None else len(batch["x"])
    energy_scale_ev = float(getattr(model, "energy_scale_ev", 1000.0))
    film_gen = getattr(model, "film_gen", None) or getattr(
        model, "shortcut_film_gen", None
    )
    energy_bounds = batch.get("energy_bounds")
    if energy_bounds is None and "energy" in batch:
        e = batch["energy"]
        if isinstance(e, torch.Tensor) and e.dim() >= 2:
            e_flat = e.view(e.shape[0], -1)
            energy_bounds = torch.stack([e_flat[:, 0], e_flat[:, -1]], dim=-1)

    return prepare_film_condition_tensor(
        batch.get("meta"),
        config={
            "num_samples": num_samples,
            "energy_scale_ev": energy_scale_ev,
            "encode_die": bool(getattr(film_gen, "encode_die", True)),
            "encode_tool": bool(getattr(film_gen, "encode_tool", True)),
            "encode_energy": bool(getattr(film_gen, "encode_energy", False)),
            "encode_flux": bool(getattr(film_gen, "encode_flux", True)),
            "encode_time": bool(getattr(film_gen, "encode_time", True)),
            "encode_region": bool(getattr(film_gen, "encode_region", False)),
            "encode_relative_intensity": bool(
                getattr(film_gen, "encode_relative_intensity", False)
            ),
            "encode_elements": bool(
                getattr(film_gen, "encode_elements", False)
            ),
            "element_vocab": getattr(film_gen, "element_vocab", None),
            "material_elements": getattr(film_gen, "material_elements", None),
        },
        device=dev,
        energy_bounds=energy_bounds,
    )


def _resolve_device(
    model: nn.Module, config: dict[str, Any] | None = None
) -> torch.device:
    dev = (config or {}).get("device") or next(
        (p.device for p in model.parameters()), "cpu"
    )
    return torch.device(dev)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: SpectralLoss,
    exec_ctx: dict[str, Any],
) -> float:
    device: torch.device = exec_ctx["device"]
    optimizer: torch.optim.Optimizer | None = exec_ctx.get("optimizer")
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, num_batches = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mv = batch.get("max_val")
            max_v = mv.to(device) if isinstance(mv, torch.Tensor) else None
            cond = _batch_cond(model, batch, device)
            if is_train:
                optimizer.zero_grad()
            yp = model(x, cond=cond) if cond is not None else model(x)
            loss = criterion(
                yp, y, model=model if is_train else None, max_val=max_v
            )
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            num_batches += 1
    if num_batches == 0:
        label = "Training" if is_train else "Validation"
        raise ValueError(f"{label} dataloader is empty.")
    return total_loss / num_batches


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
    criterion: SpectralLoss | None = None,
    config: dict[str, Any] | None = None,
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
    cfg = config or {}
    opt = optimizer or cfg.get("optimizer")
    crit = criterion or cfg.get("criterion")
    if opt is None or crit is None:
        raise ValueError("train_one_epoch requires 'optimizer' and 'criterion'.")
    return _run_epoch(
        model,
        loader,
        crit,
        {"device": _resolve_device(model, cfg), "optimizer": opt},
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: SpectralLoss,
    config: dict[str, Any] | None = None,
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
    return _run_epoch(
        model, loader, criterion, {"device": _resolve_device(model, config)}
    )


def evaluate_sliding_window(
    model: nn.Module,
    full_val_dataset: SpectrumPairDataset,
    criterion: SpectralLoss | None = None,
    config: dict[str, Any] | None = None,
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
    crit = criterion or SpectralLoss()
    cfg = config or {}
    w_size = int(cfg.get("window_size", 15))
    s_step = int(cfg.get("stride", max(1, w_size // 2)))
    device = _resolve_device(model, cfg)

    model.eval()
    p_cfg = {
        "device": device,
        "window_size": w_size,
        "stride": s_step,
        "clamp_non_negative": True,
    }
    total = 0.0

    for idx in range(len(full_val_dataset)):
        item = full_val_dataset[idx]
        rec = predict_sliding_window_spectrum(
            model,
            item["x"].cpu().numpy().squeeze(),
            config=p_cfg,
            cond=_batch_cond(model, item, device, 1),
            energy=item.get("energy"),
        )
        yp = torch.from_numpy(rec).float().view(1, 1, -1).to(device)
        total += crit(yp, item["y"].unsqueeze(0).to(device), model=None).item()

    return total / len(full_val_dataset)


def train_model_region(
    train_dataset: SpectrumPairDataset,
    val_dataset: SpectrumPairDataset | None,
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
    has_val = val_dataset is not None and len(val_dataset) > 0

    cfg = dict(config or {})
    for alias, canonical in _DISALLOWED_TRAINER_ALIASES.items():
        if alias in cfg:
            raise KeyError(
                f"Missing required canonical key '{canonical}'. "
                f"Found deprecated or plural alias '{alias}'."
            )

    model_type = str(cfg.get("model_type", "resnet")).lower()
    use_sw = bool(cfg.get("use_sliding_window", False))
    epochs = validate_positive_int(cfg.get("epochs", 50), "epochs")
    batch_size = validate_positive_int(cfg.get("batch_size", 16), "batch_size")

    for int_key in _POSITIVE_INT_CONFIG_KEYS:
        if int_key in cfg:
            validate_positive_int(cfg[int_key], int_key)

    learning_rate = float(cfg.get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    l2_weight = float(cfg.get("l2_weight", 1e-4))
    verbose = bool(cfg.get("verbose", False))
    relative_learning_rate = float(cfg.get("relative_learning_rate", 0.1))
    if relative_learning_rate <= 0.0:
        raise ValueError(
            "relative_learning_rate must be strictly positive, "
            f"got {relative_learning_rate}"
        )

    use_lr_sched = bool(cfg.get("use_learning_rate_scheduler", True))
    learning_rate_reduce_factor = float(
        cfg.get("learning_rate_reduce_factor", 0.9)
    )
    learning_rate_scheduler_patience = int(
        cfg.get("learning_rate_scheduler_patience", 10)
    )
    min_learning_rate = float(
        cfg.get("min_learning_rate", learning_rate / 10000.0)
    )

    if not (0.0 < learning_rate_reduce_factor < 1.0):
        raise ValueError(
            "learning_rate_reduce_factor must be strictly between 0 and 1, "
            f"got {learning_rate_reduce_factor}"
        )
    if learning_rate_scheduler_patience < 1:
        raise ValueError(
            "learning_rate_scheduler_patience must be an integer >= 1, "
            f"got {learning_rate_scheduler_patience}"
        )
    if min_learning_rate < 0.0:
        raise ValueError(
            f"min_learning_rate cannot be negative, got {min_learning_rate}"
        )
    if learning_rate < min_learning_rate:
        raise ValueError(
            f"Base learning_rate ({learning_rate:.2e}) is lower than the floor "
            f"min_learning_rate ({min_learning_rate:.2e}). "
            "Please increase 'learning_rate' or decrease 'min_learning_rate'."
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
        is_patched = isinstance(train_dataset, SpectrumPatchDataset)
        ws_cfg_val = explicit_ws if explicit_ws is not None else 15
        if ws_cfg_val < 3:
            raise ValueError(
                f"window_size must be >= 3 when sliding window is active, got {ws_cfg_val}"
            )
        w_size = (
            train_dataset.window_size
            if is_patched
            else min(full_len, ws_cfg_val)
        )
        s_step = (
            train_dataset.stride
            if is_patched
            else max(1, min(w_size, int(cfg.get("stride", max(1, w_size // 2)))))
        )
        train_ds = (
            train_dataset
            if is_patched
            else SpectrumPatchDataset(
                train_dataset, window_size=w_size, stride=s_step
            )
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=default_spectrum_collate,
        )
        seq_len, val_loader = w_size, None
    else:
        seq_len = w_size = s_step = full_len
        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset if has_val else train_dataset,
            config={"batch_size": batch_size, "shuffle_train": True},
        )
        if not has_val:
            val_loader = None

    effective_max_k = (
        min(seq_len, explicit_ws) if explicit_ws is not None else seq_len
    )
    if "kernel_size" in cfg:
        validate_odd_kernel_size(cfg["kernel_size"], max_allowed=effective_max_k)

    model = cfg.get("model")
    if model is None:
        model = instantiate_model(model_type, seq_len=seq_len, config=cfg)
    model = model.to(device)

    criterion = cfg.get("criterion") or SpectralLoss(
        config={**cfg, "l2_weight": l2_weight}
    )
    if hasattr(model, "shortcut_film_gen") and model.shortcut_film_gen is not None:
        shortcut_lr = learning_rate * relative_learning_rate
        if shortcut_lr < min_learning_rate:
            raise ValueError(
                f"Initial learning rate of shortcut gain ({shortcut_lr:.2e}) is lower "
                f"than the floor min_learning_rate ({min_learning_rate:.2e}). "
                "Please increase 'learning_rate' or 'relative_learning_rate', "
                "or decrease 'min_learning_rate'."
            )
        shortcut_params = list(model.shortcut_film_gen.parameters())
        shortcut_param_ids = {id(p) for p in shortcut_params}
        base_params = [
            p for p in model.parameters() if id(p) not in shortcut_param_ids
        ]
        param_groups = [
            {"params": base_params, "lr": learning_rate},
            {
                "params": shortcut_params,
                "lr": shortcut_lr,
            },
        ]
        optimizer = torch.optim.Adam(
            param_groups, lr=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
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
    best_loss, best_epoch, best_state, pat_count = float("inf"), 0, None, 0
    run_cfg = {"device": device}
    eval_cfg = {"device": device, "window_size": w_size, "stride": s_step}

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config=run_cfg
        )
        if not has_val:
            val_loss = float("nan")
        elif use_sw:
            val_loss = evaluate_sliding_window(
                model, val_dataset, criterion, config=eval_cfg
            )
        else:
            val_loss = evaluate(model, val_loader, criterion, config=run_cfg)
        mon_loss = val_loss if has_val else tr_loss

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        if math.isnan(mon_loss) or math.isinf(mon_loss):
            warnings.warn(
                f"Training diverged at epoch {epoch} with NaN/Inf loss ({mon_loss}). Stopping early."
            )
            break

        if mon_loss < best_loss:
            best_loss = mon_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            pat_count = 0
        else:
            pat_count += 1

        if scheduler is not None:
            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(mon_loss)
            cur_lr = optimizer.param_groups[0]["lr"]
            if cur_lr < prev_lr:
                if verbose:
                    lbl = "Val Loss" if has_val else "Train Loss"
                    print(
                        f"Epoch {epoch:3d}: ReduceLROnPlateau reduced LR from "
                        f"{prev_lr:.2e} to {cur_lr:.2e} ({lbl})."
                    )
                pat_count = 0

        if verbose and (epoch % 10 == 0 or epoch in (1, epochs)):
            tr_str = format_loss_log10(tr_loss)
            v_str = (
                format_loss_log10(val_loss)
                if has_val
                else f"{tr_str} (No validation set; monitoring Train Loss)"
            )
            print(
                f"Epoch {epoch:3d}/{epochs:3d} - Train Loss: {tr_str}, Val Loss: {v_str}"
            )

        if pat_count >= patience:
            if verbose:
                t_str = "best val_loss" if has_val else "best train_loss"
                print(
                    f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, "
                    f"{t_str}: {format_loss_log10(best_loss)})"
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    cfg.update(
        {
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
            "window_size": w_size,
            "stride": s_step,
            "seq_len": seq_len,
        }
    )
    return {
        "model": model,
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "history": history,
        "config": cfg,
        "window_size": w_size,
        "stride": s_step,
    }
