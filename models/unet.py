"""1D U-Net neural network architecture, training, and pipeline for spectral transformation.

Supports both full regional spectrum mode and sliding window patch mode,
with global residual shortcut for stable inter-tool transfer.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.baseline_cnn import NormalizedMSELoss
from models.bayesian_opt import run_ax_search
from models.dataset import SpectrumPairDataset, create_dataloaders
from models.inference import assemble_prediction_metadata
from models.root import run_model_pipeline
from models.sliding_window import (
    SpectrumPatchDataset,
    evaluate_sliding_window,
    predict_sliding_window_spectrum,
)
from utility.patching import calculate_window_points


class UNetConvBlock1D(nn.Module):
    """Double 1D convolutional block with normalization and non-linear activations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        cfg = dict(config or {})
        cfg.update(kwargs)
        kernel_size: int = int(cfg.get("kernel_size", 5))
        dropout: float = float(cfg.get("dropout", 0.0))
        use_batch_norm: bool = bool(cfg.get("use_batch_norm", True))
        activation: str = str(cfg.get("activation", "relu"))

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        padding = kernel_size // 2
        act_layer = nn.LeakyReLU(0.1, inplace=True) if activation == "leaky_relu" else nn.ReLU(inplace=True)

        layers: list[nn.Module] = []
        # Conv 1
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(act_layer)
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        # Conv 2
        layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU(inplace=True) if activation != "leaky_relu" else nn.LeakyReLU(0.1, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet1D(nn.Module):
    """1D U-Net architecture with multi-scale contracting and expanding paths.

    Features:
    - Multi-scale hierarchical feature extraction.
    - Long skip connections preserving high-frequency peak boundaries.
    - Robust linear interpolation in the decoder supporting arbitrary/odd sequence lengths.
    - Optional global residual connection y = x + UNet(x) for stable inter-tool delta learning.
    - Explicit L2 penalty computation on convolutional weights for NormalizedMSELoss compatibility.

    Parameters
    ----------
    n_points : int
        Nominal number of sequence points (spectral length or patch window length).
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'base_channels' (int): Channels at level 0 (default 16).
        - 'depth' (int): Number of downsampling stages (default 3).
        - 'kernel_size' (int): Conv kernel size, positive odd int (default 5).
        - 'dropout' (float): Dropout probability (default 0.0).
        - 'use_batch_norm' (bool): Whether to use BatchNorm1d (default True).
        - 'residual' (bool): Whether to apply global residual shortcut y = x + delta (default True).
        - 'activation' (str): Activation function 'relu' or 'leaky_relu' (default 'relu').
    """

    def __init__(self, n_points: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        self.n_points = int(n_points)
        self.base_channels = int(cfg.get("base_channels", 16))
        self.depth = int(cfg.get("depth", 3))
        self.kernel_size = int(cfg.get("kernel_size", 5))
        self.dropout = float(cfg.get("dropout", 0.0))
        self.use_batch_norm = bool(cfg.get("use_batch_norm", True))
        self.residual = bool(cfg.get("residual", True))
        self.activation = str(cfg.get("activation", "relu"))

        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        min_points = 2**self.depth
        if self.n_points < min_points:
            raise ValueError(
                f"n_points ({self.n_points}) is too small for depth {self.depth}. "
                f"Requires at least {min_points} points (2**depth)."
            )
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {self.kernel_size}")

        block_cfg = {
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation,
        }

        # Encoder stages
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        current_ch = 1
        ch_list: list[int] = []

        for i in range(self.depth):
            out_ch = self.base_channels * (2**i)
            self.encoders.append(
                UNetConvBlock1D(
                    in_channels=current_ch,
                    out_channels=out_ch,
                    config=block_cfg,
                )
            )
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            ch_list.append(out_ch)
            current_ch = out_ch

        # Bottleneck stage
        bottleneck_ch = self.base_channels * (2**self.depth)
        self.bottleneck = UNetConvBlock1D(
            in_channels=current_ch,
            out_channels=bottleneck_ch,
            config=block_cfg,
        )

        # Decoder stages
        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_current = bottleneck_ch

        for i in reversed(range(self.depth)):
            skip_ch = ch_list[i]
            # 1x1 conv to project channels after linear interpolation
            self.up_convs.append(nn.Conv1d(dec_current, skip_ch, kernel_size=1))
            self.decoders.append(
                UNetConvBlock1D(
                    in_channels=skip_ch * 2,
                    out_channels=skip_ch,
                    config=block_cfg,
                )
            )
            dec_current = skip_ch

        # Final projection to 1 channel
        self.final_conv = nn.Conv1d(dec_current, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 1D U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_points) or (batch_size, 1, n_points).

        Returns
        -------
        torch.Tensor
            Transformed spectrum matching input shape.
        """
        orig_dim = x.dim()
        if orig_dim == 2:
            x_in = x.unsqueeze(1)
        elif orig_dim == 3:
            x_in = x
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        skips: list[torch.Tensor] = []
        feat = x_in

        # Contracting path
        for i in range(self.depth):
            feat = self.encoders[i](feat)
            skips.append(feat)
            feat = self.pools[i](feat)

        # Bottleneck
        feat = self.bottleneck(feat)

        # Expanding path
        for i in range(self.depth):
            skip = skips[-(i + 1)]
            # Direct linear interpolation to target skip sequence length
            feat = F.interpolate(feat, size=skip.shape[-1], mode="linear", align_corners=False)
            feat = self.up_convs[i](feat)

            cat = torch.cat([feat, skip], dim=1)
            feat = self.decoders[i](cat)

        delta = self.final_conv(feat)

        out = (x_in + delta) if self.residual else delta

        if orig_dim == 2:
            return out.squeeze(1)
        return out

    def get_l2_regularization(self) -> torch.Tensor:
        """Compute the sum of squared weights of convolutional layers."""
        device = next(self.parameters()).device
        l2_sum = torch.tensor(0.0, device=device)
        for module in self.modules():
            if isinstance(module, nn.Conv1d) and module.weight.requires_grad:
                l2_sum = l2_sum + torch.sum(module.weight**2)
        return l2_sum


def train_unet_region(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a 1D U-Net model on a spectral dataset.

    Parameters
    ----------
    train_dataset : Dataset
        Dataset providing training samples.
    val_dataset : Dataset
        Dataset providing validation samples.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'epochs' (int): Number of training epochs (default 100).
        - 'batch_size' (int): Batch size (default 16).
        - 'learning_rate' (float): Optimizer learning rate (default 1e-3).
        - 'l2_weight' (float): L2 regularization factor (default 1e-4).
        - 'base_channels' (int): Base feature channels (default 16).
        - 'depth' (int): Number of downsampling levels (default 3).
        - 'kernel_size' (int): Convolution kernel size (default 5).
        - 'dropout' (float): Dropout probability (default 0.0).
        - 'use_batch_norm' (bool): Whether to use BatchNorm1d (default True).
        - 'residual' (bool): Global residual mode (default True).
        - 'early_stopping_patience' (int): Early stopping patience (default 15).
        - 'device' (str | torch.device | None): Computation device.
        - 'verbose' (bool): Whether to print epoch progress (default False).

    Returns
    -------
    dict[str, Any]
        Dictionary containing 'model', 'best_val_loss', 'best_epoch', and 'history'.
    """
    cfg = config or {}
    epochs: int = int(cfg.get("epochs", 100))
    batch_size: int = int(cfg.get("batch_size", 16))
    lr: float = float(cfg.get("learning_rate", 1e-3))
    l2_weight: float = float(cfg.get("l2_weight", 1e-4))
    patience: int = int(cfg.get("early_stopping_patience", 15))
    verbose: bool = bool(cfg.get("verbose", False))

    device_cfg = cfg.get("device")
    if device_cfg is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    # Inspect sample length
    sample = train_dataset[0]
    n_points = int(sample["x"].shape[-1])

    dl_cfg = {
        "batch_size": batch_size,
        "shuffle_train": True,
        "num_workers": int(cfg.get("num_workers", 0)),
    }
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config=dl_cfg)

    model = UNet1D(n_points=n_points, config=cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = NormalizedMSELoss(l2_weight=l2_weight)
    eval_criterion = NormalizedMSELoss(l2_weight=0.0)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, Any] | None = None
    patience_counter = 0

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            max_v = batch["max_val"].to(device) if "max_val" in batch and batch["max_val"] is not None else None

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y, model=model, max_val=max_v)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_train_loss / max(1, n_batches)

        # Validation phase: evaluate on full reconstructed spectrum if val_full_dataset is supplied
        if cfg.get("val_full_dataset") is not None:
            w_size = int(cfg.get("window_size", n_points))
            s_step = int(cfg.get("stride", max(1, w_size // 2)))
            avg_val_loss = evaluate_sliding_window(
                model=model,
                full_val_dataset=cfg["val_full_dataset"],
                criterion=eval_criterion,
                config={"window_size": w_size, "stride": s_step, "device": device},
            )
        else:
            model.eval()
            total_val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    y_pred = model(x)
                    val_loss = eval_criterion(y_pred, y)
                    total_val_loss += val_loss.item()
                    val_batches += 1

            avg_val_loss = total_val_loss / max(1, val_batches)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % 10 == 0 or epoch == epochs):
            print(f"[UNet1D] Epoch {epoch:3d}/{epochs:3d} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if patience_counter >= patience:
            if verbose:
                print(f"[UNet1D] Early stopping at epoch {epoch} (best: {best_val_loss:.6f})")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
        "config": cfg,
    }


def optimize_unet_hyperparameters(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform Bayesian optimization over 1D U-Net hyperparameters using Ax."""
    cfg = config or {}
    base_channel_choices: list[int] = list(cfg.get("base_channels", [16, 32]))
    depth_choices: list[int] = list(cfg.get("depths", [2, 3]))
    kernel_sizes: list[int] = list(cfg.get("kernel_sizes", [3, 5]))
    lr_bounds: tuple[float, float] = tuple(cfg.get("lr_bounds", (1e-4, 1e-2)))
    l2_bounds: tuple[float, float] = tuple(cfg.get("l2_bounds", (1e-6, 1e-2)))

    parameters: list[dict[str, Any]] = [
        {
            "name": "base_channels",
            "type": "choice",
            "values": base_channel_choices,
            "value_type": "int",
            "is_ordered": True,
        },
        {
            "name": "depth",
            "type": "choice",
            "values": depth_choices,
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

    search_cfg = dict(cfg)
    search_cfg.setdefault("experiment_name", "unet_spectral_transfer_optimization")
    search_cfg.setdefault("num_trials", 6)
    search_cfg.setdefault("epochs_per_trial", 15)

    return run_ax_search(
        train_fn=train_unet_region,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        parameters=parameters,
        config=search_cfg,
    )


def predict_unet_spectra(
    models: dict[str, Any],
    data: tuple[np.ndarray, np.ndarray, pd.DataFrame] | dict[str, Any],
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate transformed predictions across source measurements using trained U-Net models."""
    if isinstance(data, (tuple, list)):
        ary_intensity, ary_energy, meta_df = data[0], data[1], data[2]
    else:
        ary_intensity, ary_energy, meta_df = data["ary_intensity"], data["ary_energy"], data["meta_df"]

    cfg = config or {}
    source_tool = str(cfg.get("source_tool", "J4"))
    target_tool = str(cfg.get("target_tool", "H1"))
    use_sw = bool(cfg.get("use_sliding_window", False))

    regions = list(models.keys())
    source_mask = (meta_df["tool"] == source_tool) & (meta_df["region"].isin(regions))
    source_df = meta_df[source_mask].copy()

    if source_df.empty:
        raise ValueError(f"No source spectra found for tool='{source_tool}' with criteria: {cfg}")

    n_pred = len(source_df)
    n_points = ary_intensity.shape[1]
    pred_intensities = np.zeros((n_pred, n_points), dtype=np.float32)
    pred_energies = np.zeros((n_pred, n_points), dtype=np.float32)

    normalize_by_source: bool = bool(cfg.get("normalize_by_source", True))
    clamp_non_negative: bool = bool(cfg.get("clamp_non_negative", True))
    eps: float = float(cfg.get("eps", 1e-4))

    for new_idx, (_, row) in enumerate(source_df.iterrows()):
        reg = str(row["region"])
        orig_idx = int(row["spectrum_index"])
        pred_energies[new_idx] = ary_energy[orig_idx]
        x_raw = ary_intensity[orig_idx]

        if reg in models:
            model_entry = models[reg]
            if isinstance(model_entry, (tuple, list)):
                model, w_size, s_step = model_entry[0], model_entry[1], model_entry[2]
                is_sw = True
            elif use_sw:
                model = model_entry
                w_size = int(cfg.get("window_size", 15))
                s_step = int(cfg.get("stride", max(1, w_size // 2)))
                is_sw = True
            else:
                model = model_entry
                is_sw = False

            if is_sw:
                sw_params = list(model.parameters())
                model_device = sw_params[0].device if sw_params else torch.device("cpu")
                pred_intensities[new_idx] = predict_sliding_window_spectrum(
                    model=model,
                    spectrum=x_raw,
                    config={
                        "window_size": w_size,
                        "stride": s_step,
                        "device": model_device,
                        "normalize_by_source": normalize_by_source,
                        "clamp_non_negative": clamp_non_negative,
                        "eps": eps,
                    },
                )
            else:
                model = model_entry
                model.eval()
                m_params = list(model.parameters())
                model_device = m_params[0].device if m_params else torch.device("cpu")
                if normalize_by_source:
                    scale_x = max(float(np.max(np.abs(x_raw))), eps)
                    x_in = x_raw / scale_x
                else:
                    scale_x = 1.0
                    x_in = x_raw

                with torch.no_grad():
                    x_tensor = torch.from_numpy(x_in.astype(np.float32)).unsqueeze(0).to(model_device)
                    y_pred = model(x_tensor).squeeze(0).cpu().numpy()

                y_pred_phys = y_pred * scale_x if normalize_by_source else y_pred
                if clamp_non_negative:
                    y_pred_phys = np.clip(y_pred_phys, 0.0, None)
                pred_intensities[new_idx] = y_pred_phys
        else:
            warnings.warn(f"No trained model found for region '{reg}'. Copying source intensity.")
            fallback = x_raw.copy()
            if clamp_non_negative:
                fallback = np.clip(fallback, 0.0, None)
            pred_intensities[new_idx] = fallback

    pred_meta_df = assemble_prediction_metadata(
        source_df,
        config={
            "source_tool": source_tool,
            "target_tool": target_tool,
            "n_points": n_points,
            "session_splits": cfg.get("session_splits"),
        },
    )
    return pred_intensities, pred_energies, pred_meta_df


def run_unet_pipeline(
    meta_df: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the end-to-end 1D U-Net spectral transformation pipeline.

    Supports both full regional spectrum mode (default) and sliding window
    patch mode via the 'use_sliding_window' configuration flag.
    """
    cfg = config or {}
    use_sw: bool = bool(cfg.get("use_sliding_window", False))

    hooks: dict[str, Any] = {
        "bo_fn": optimize_unet_hyperparameters,
    }

    if use_sw:
        def _dataset_prep_sw(
            train_full_ds: Any,
            val_full_ds: Any,
            reg_cfg: dict[str, Any],
        ) -> tuple[Any, Any, dict[str, Any]]:
            sample_energy = train_full_ds[0].get("energy")
            if sample_energy is not None:
                e_grid = sample_energy.cpu().numpy()
            else:
                e_grid = np.linspace(0.0, 10.0, train_full_ds[0]["x"].shape[-1])
            sw_calc_cfg = {
                "window_size_ev": cfg.get("window_size_ev", 2.0),
                "sliding_stride_ev": cfg.get("sliding_stride_ev", 1.0),
                "window_size_points": cfg.get("window_size_points"),
                "sliding_stride_points": cfg.get("sliding_stride_points"),
            }
            w_size, s_step = calculate_window_points(e_grid, config=sw_calc_cfg)

            train_ds = SpectrumPatchDataset(train_full_ds, window_size=w_size, stride=s_step)
            val_ds = SpectrumPatchDataset(val_full_ds, window_size=w_size, stride=s_step)

            updated_cfg = dict(reg_cfg)
            updated_cfg["val_full_dataset"] = val_full_ds
            updated_cfg["window_size"] = w_size
            updated_cfg["stride"] = s_step
            return train_ds, val_ds, updated_cfg

        hooks["dataset_prep_fn"] = _dataset_prep_sw
        hooks["model_record_fn"] = lambda res, r_cfg: (res["model"], r_cfg["window_size"], r_cfg["stride"])

    return run_model_pipeline(
        data=(ary_intensity, ary_energy, meta_df),
        train_region_fn=train_unet_region,
        predict_fn=predict_unet_spectra,
        config=config,
        hooks=hooks,
    )
