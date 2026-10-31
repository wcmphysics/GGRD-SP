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
from models.factory import MODEL_CLASS_MAP, instantiate_model
from models.film import prepare_film_condition_tensor
from models.inference import predict_sliding_window_spectrum
from models.layers import validate_odd_kernel_size, validate_positive_int

_MODEL_CLASS_MAP = MODEL_CLASS_MAP

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

    return film_gen.prepare_condition_tensor(
        batch.get("meta"),
        device=dev,
        energy_bounds=energy_bounds,
        energy_scale_ev=energy_scale_ev,
        extra_config={"num_samples": num_samples},
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
                yp, y, model=model if is_train else None, loss_context=max_v
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


def _extract_dataset_tensors(
    dataset: Any,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Extract stacked (x, y) tensors if dataset is homogeneous, else None."""
    try:
        if (
            hasattr(dataset, "x")
            and hasattr(dataset, "y")
            and isinstance(dataset.x, torch.Tensor)
            and isinstance(dataset.y, torch.Tensor)
        ):
            if (
                dataset.x.dim() == 3
                and dataset.y.dim() == 3
                and dataset.x.shape[0] == dataset.y.shape[0]
                and dataset.x.shape[-1] == dataset.y.shape[-1]
            ):
                return dataset.x, dataset.y
        xs = [dataset[i]["x"] for i in range(len(dataset))]
        ys = [dataset[i]["y"] for i in range(len(dataset))]
        stacked_x = torch.stack(xs, dim=0) if xs[0].dim() == 2 else torch.cat(xs, dim=0)
        stacked_y = torch.stack(ys, dim=0) if ys[0].dim() == 2 else torch.cat(ys, dim=0)
        if (
            stacked_x.dim() == 3
            and stacked_y.dim() == 3
            and stacked_x.shape[0] == stacked_y.shape[0]
            and stacked_x.shape[-1] == stacked_y.shape[-1]
        ):
            return stacked_x, stacked_y
        return None
    except (RuntimeError, ValueError, KeyError, TypeError, IndexError):
        return None


def _evaluate_sliding_window_sample_by_sample(
    model: nn.Module,
    full_val_dataset: Any,
    criterion: SpectralLoss,
    config: dict[str, Any],
    device: torch.device,
    window_size: int,
    stride: int,
) -> float:
    """Evaluate sliding-window loss sequentially sample by sample."""
    model.eval()
    p_cfg = {
        "device": device,
        "window_size": window_size,
        "stride": stride,
        "clamp_non_negative": bool(config.get("clamp_non_negative", True)),
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
        total += criterion(
            yp, item["y"].unsqueeze(0).to(device), model=None
        ).item()
    return total / len(full_val_dataset)


def _vectorized_sliding_window_predict(
    model: nn.Module,
    val_x: torch.Tensor,
    window_size: int,
    stride: int,
    device: torch.device,
    clamp_non_negative: bool = True,
    batch_size: int = 64,
) -> torch.Tensor:
    """Predict and reconstruct full spectra using vectorized window batches
    in PyTorch.
    """
    n_samples, _, seq_len = val_x.shape
    start_indices = list(range(0, seq_len - window_size + 1, stride))
    last_possible = seq_len - window_size
    if start_indices and start_indices[-1] < last_possible:
        start_indices.append(last_possible)

    val_x_dev = val_x.to(device)
    patches = torch.stack(
        [val_x_dev[:, :, s : s + window_size] for s in start_indices], dim=1
    )
    num_windows = len(start_indices)
    flat_patches = patches.view(-1, 1, window_size)

    # Evaluate patches through model in batches.
    with torch.no_grad():
        if flat_patches.shape[0] <= batch_size:
            flat_preds = model(flat_patches)
        else:
            flat_preds = torch.cat(
                [
                    model(flat_patches[i : i + batch_size])
                    for i in range(0, flat_patches.shape[0], batch_size)
                ],
                dim=0,
            )

    preds = flat_preds.view(n_samples, num_windows, 1, window_size)
    accum = torch.zeros(
        n_samples, 1, seq_len, device=device, dtype=val_x.dtype
    )
    counts = torch.zeros(1, 1, seq_len, device=device, dtype=val_x.dtype)

    # Accumulate overlapping patch predictions and normalize.
    for k, s in enumerate(start_indices):
        accum[:, :, s : s + window_size] += preds[:, k, :, :]
        counts[:, :, s : s + window_size] += 1.0

    recon = accum / counts.clamp(min=1.0)
    if clamp_non_negative:
        recon = torch.clamp(recon, min=0.0)
    return recon


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
    if w_size < 1:
        raise ValueError(f"window_size must be >= 1, got {w_size}")
    if s_step < 1:
        raise ValueError(f"stride must be >= 1, got {s_step}")
    if s_step > w_size:
        raise ValueError(
            f"stride ({s_step}) cannot exceed window_size ({w_size}) "
            "as it creates reconstruction gaps."
        )

    device = _resolve_device(model, cfg)
    clamp_non_negative = bool(cfg.get("clamp_non_negative", True))
    model.eval()

    # Fall back to sample-by-sample when FiLM is active or data
    # is heterogeneous.
    if getattr(model, "use_film", False):
        return _evaluate_sliding_window_sample_by_sample(
            model, full_val_dataset, crit, cfg, device, w_size, s_step
        )

    tensors = _extract_dataset_tensors(full_val_dataset)
    if tensors is None:
        return _evaluate_sliding_window_sample_by_sample(
            model, full_val_dataset, crit, cfg, device, w_size, s_step
        )

    val_x, val_y = tensors
    seq_len = val_x.shape[-1]
    if w_size > seq_len:
        raise ValueError(
            f"window_size ({w_size}) cannot exceed spectrum length ({seq_len})"
        )

    eval_batch_size = int(cfg.get("batch_size", 64))
    recon = _vectorized_sliding_window_predict(
        model,
        val_x,
        w_size,
        s_step,
        device,
        clamp_non_negative=clamp_non_negative,
        batch_size=eval_batch_size,
    )
    val_y_dev = val_y.to(device)
    loss = crit(recon, val_y_dev, model=None).item()
    return float(loss)


def _validate_training_hyperparameters(
    cfg: dict[str, Any],
    epochs: int,
) -> dict[str, Any]:
    """Validate and return learning rate, scheduler, and patience settings."""
    # Validate integer parameter types and learning rate parameters.
    for int_key in _POSITIVE_INT_CONFIG_KEYS:
        if int_key in cfg:
            validate_positive_int(cfg[int_key], int_key)

    learning_rate = float(cfg.get("learning_rate", 1e-3))
    l2_weight = float(cfg.get("l2_weight", 1e-4))
    relative_learning_rate = float(cfg.get("relative_learning_rate", 0.1))
    if relative_learning_rate <= 0.0:
        raise ValueError(
            "relative_learning_rate must be strictly positive, "
            f"got {relative_learning_rate}"
        )

    # Extract and bounds-check learning rate scheduler hyperparameters.
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

    # Resolve early stopping patience and compare with scheduler patience.
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

    return {
        "lr": learning_rate,
        "l2_weight": l2_weight,
        "use_sched": use_lr_sched,
        "reduce_factor": learning_rate_reduce_factor,
        "sched_patience": learning_rate_scheduler_patience,
        "min_lr": min_learning_rate,
        "patience": patience,
    }


def _setup_loss_criterion(cfg: dict[str, Any], l2_weight: float) -> SpectralLoss:
    """Instantiate or configure the SpectralLoss criterion."""
    passed_criterion = cfg.get("criterion")
    if passed_criterion is not None:
        criterion = copy.copy(passed_criterion)
        if "l2_weight" in cfg and hasattr(criterion, "l2_weight"):
            criterion.l2_weight = l2_weight
        return criterion
    return SpectralLoss(config={**cfg, "l2_weight": l2_weight})


def _setup_training_dataloaders(
    train_dataset: SpectrumPairDataset,
    val_dataset: SpectrumPairDataset | None,
    use_sw: bool,
    cfg: dict[str, Any],
) -> tuple[DataLoader, DataLoader | None, int, int, int]:
    """Configure training and validation DataLoaders and spatial lengths."""
    has_val = val_dataset is not None and len(val_dataset) > 0
    batch_size = validate_positive_int(cfg.get("batch_size", 16), "batch_size")
    full_len = int(train_dataset[0]["x"].shape[-1])
    explicit_ws = cfg.get("window_size_choice", cfg.get("window_size"))

    # Configure sliding window patch dataset or standard whole-spectrum loaders.
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

    # Validate odd kernel size against effective spatial sequence length.
    effective_max_k = (
        min(seq_len, explicit_ws) if explicit_ws is not None else seq_len
    )
    if "kernel_size" in cfg:
        validate_odd_kernel_size(cfg["kernel_size"], max_allowed=effective_max_k)

    return train_loader, val_loader, seq_len, w_size, s_step


def _setup_optimizer_and_scheduler(
    model: nn.Module,
    cfg: dict[str, Any],
    learning_rate: float,
    min_learning_rate: float,
    use_lr_sched: bool,
) -> tuple[torch.optim.Optimizer, ReduceLROnPlateau | None]:
    """Configure Adam optimizer with potential parameter groups and
    scheduler.
    """
    weight_decay = float(cfg.get("weight_decay", 0.0))
    relative_learning_rate = float(cfg.get("relative_learning_rate", 0.1))
    learning_rate_reduce_factor = float(
        cfg.get("learning_rate_reduce_factor", 0.9)
    )
    learning_rate_scheduler_patience = int(
        cfg.get("learning_rate_scheduler_patience", 10)
    )

    # Configure dedicated parameter groups for shortcut FiLM generators.
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

    # Instantiate ReduceLROnPlateau learning rate scheduler if enabled.
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
    return optimizer, scheduler


def _execute_training_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau | None,
    criterion: SpectralLoss,
    val_dataset: SpectrumPairDataset | None,
    exec_cfg: dict[str, Any],
) -> tuple[float, int, dict[str, list[float]]]:
    """Execute training and evaluation loop across all epochs with early
    stopping.
    """
    epochs = int(exec_cfg["epochs"])
    use_sw = bool(exec_cfg["use_sw"])
    has_val = bool(exec_cfg["has_val"])
    patience = int(exec_cfg["patience"])
    device: torch.device = exec_cfg["device"]
    verbose = bool(exec_cfg["verbose"])
    w_size = int(exec_cfg["w_size"])
    s_step = int(exec_cfg["s_step"])

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }
    best_loss, best_epoch, best_state, pat_count = float("inf"), 0, None, 0
    run_cfg = {"device": device}
    eval_cfg = {"device": device, "window_size": w_size, "stride": s_step}

    # Train and evaluate over epochs with dynamic LR and early stopping.
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

    # Restore best checkpoint weights before returning.
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_loss, best_epoch, history


def _package_training_result(
    model: nn.Module,
    best_loss: float,
    best_epoch: int,
    history: dict[str, list[float]],
    cfg: dict[str, Any],
    lr_params: dict[str, Any],
    exec_meta: dict[str, Any],
) -> dict[str, Any]:
    """Collate training outcomes and updated configuration into return
    dictionary.
    """
    # Update config dictionary with resolved training hyperparameters.
    cfg.update(
        {
            "epochs": exec_meta["epochs"],
            "batch_size": exec_meta["batch_size"],
            "learning_rate": lr_params["lr"],
            "l2_weight": lr_params["l2_weight"],
            "early_stopping_patience": lr_params["patience"],
            "model_type": exec_meta["model_type"],
            "use_sliding_window": exec_meta["use_sw"],
            "use_learning_rate_scheduler": lr_params["use_sched"],
            "learning_rate_reduce_factor": lr_params["reduce_factor"],
            "learning_rate_scheduler_patience": lr_params["sched_patience"],
            "min_learning_rate": lr_params["min_lr"],
            "window_size": exec_meta["w_size"],
            "stride": exec_meta["s_step"],
            "seq_len": exec_meta["seq_len"],
        }
    )

    # Package trained model weights, best loss metrics, and histories.
    return {
        "model": model,
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "history": history,
        "config": cfg,
        "window_size": exec_meta["w_size"],
        "stride": exec_meta["s_step"],
    }


def _init_training_config(
    train_dataset: SpectrumPairDataset,
    val_dataset: SpectrumPairDataset | None,
    config: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], torch.device, bool]:
    """Validate datasets and extract training configuration parameters."""
    if len(train_dataset) == 0:
        raise ValueError("Cannot train on an empty train_dataset.")
    has_val = val_dataset is not None and len(val_dataset) > 0
    cfg = dict(config or {})
    epochs = validate_positive_int(cfg.get("epochs", 50), "epochs")
    lr_params = _validate_training_hyperparameters(cfg, epochs)
    dev_str = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    return cfg, lr_params, torch.device(dev_str), has_val


def _build_training_model(
    cfg: dict[str, Any], seq_len: int, device: torch.device
) -> nn.Module:
    """Instantiate or retrieve target model and move to computation device."""
    model_type = str(cfg.get("model_type", "resnet")).lower()
    model = cfg.get("model") or instantiate_model(
        model_type, seq_len=seq_len, config=cfg
    )
    return model.to(device)


def _prepare_training_execution(
    cfg: dict[str, Any],
    lr_params: dict[str, Any],
    device: torch.device,
    use_sw: bool,
    has_val: bool,
    w_size: int,
    s_step: int,
    seq_len: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build execution configuration and metadata dictionaries."""
    exec_cfg = {
        "epochs": cfg.get("epochs", 50),
        "use_sw": use_sw,
        "has_val": has_val,
        "patience": lr_params["patience"],
        "device": device,
        "verbose": bool(cfg.get("verbose", False)),
        "w_size": w_size,
        "s_step": s_step,
    }
    exec_meta = {
        "epochs": cfg.get("epochs", 50),
        "batch_size": cfg.get("batch_size", 16),
        "model_type": str(cfg.get("model_type", "resnet")).lower(),
        "use_sw": use_sw,
        "w_size": w_size,
        "s_step": s_step,
        "seq_len": seq_len,
    }
    return exec_cfg, exec_meta


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
    cfg, lr_params, device, has_val = _init_training_config(
        train_dataset, val_dataset, config
    )
    use_sw = bool(cfg.get("use_sliding_window", False))
    train_loader, val_loader, seq_len, w_size, s_step = _setup_training_dataloaders(
        train_dataset, val_dataset, use_sw, cfg
    )
    model = _build_training_model(cfg, seq_len, device)
    criterion = _setup_loss_criterion(cfg, lr_params["l2_weight"])
    optimizer, scheduler = _setup_optimizer_and_scheduler(
        model, cfg, lr_params["lr"], lr_params["min_lr"], lr_params["use_sched"]
    )

    exec_cfg, exec_meta = _prepare_training_execution(
        cfg, lr_params, device, use_sw, has_val, w_size, s_step, seq_len
    )
    best_loss, best_epoch, history = _execute_training_epochs(
        model, train_loader, val_loader, optimizer, scheduler, criterion, val_dataset, exec_cfg
    )
    return _package_training_result(
        model, best_loss, best_epoch, history, cfg, lr_params, exec_meta
    )
