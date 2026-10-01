"""1D U-Net neural network architecture, training, and pipeline for spectral transformation.

Supports both full regional spectrum mode and sliding window patch mode,
with global residual shortcut for stable inter-tool transfer.
"""

from __future__ import annotations

import copy
import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.baseline_cnn import NormalizedMSELoss
from models.dataset import SpectrumPairDataset, create_dataloaders, split_session_datasets
from models.inference import format_predicted_measurement_id, predict_spectra
from models.sliding_window import (
    SpectrumPatchDataset,
    calculate_window_points,
    evaluate_sliding_window,
    extract_sliding_windows,
    predict_sliding_window_spectrum,
)

try:
    from ax.service.ax_client import AxClient, ObjectiveProperties

    AX_AVAILABLE = True
except (ImportError, OSError):
    AX_AVAILABLE = False


class UNetConvBlock1D(nn.Module):
    """Double 1D convolutional block with normalization and non-linear activations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.0,
        use_batch_norm: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
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
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {self.kernel_size}")

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
                    kernel_size=self.kernel_size,
                    dropout=self.dropout,
                    use_batch_norm=self.use_batch_norm,
                    activation=self.activation,
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
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm,
            activation=self.activation,
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
                    kernel_size=self.kernel_size,
                    dropout=self.dropout,
                    use_batch_norm=self.use_batch_norm,
                    activation=self.activation,
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
            # Linear interpolation upsampling (scale_factor=2)
            feat = F.interpolate(feat, scale_factor=2, mode="linear", align_corners=False)
            feat = self.up_convs[i](feat)

            # Robust length matching in case integer division of odd length produced length mismatch
            if feat.shape[-1] != skip.shape[-1]:
                feat = F.interpolate(feat, size=skip.shape[-1], mode="linear", align_corners=False)

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
        for name, param in self.named_parameters():
            if param.requires_grad and name.endswith(".weight") and "bn" not in name:
                l2_sum = l2_sum + torch.sum(param**2)
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y, model=model)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_train_loss / max(1, n_batches)

        # Validation phase
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
    if not AX_AVAILABLE:
        warnings.warn("Ax platform not available. Skipping hyperparameter optimization.")
        return {"best_parameters": {}, "best_val_loss": float("nan")}

    cfg = config or {}
    num_trials: int = int(cfg.get("num_trials", 6))
    epochs_per_trial: int = int(cfg.get("epochs_per_trial", 15))
    base_channel_choices: list[int] = list(cfg.get("base_channels", [16, 32]))
    depth_choices: list[int] = list(cfg.get("depths", [2, 3]))
    kernel_sizes: list[int] = list(cfg.get("kernel_sizes", [3, 5]))
    lr_bounds: tuple[float, float] = tuple(cfg.get("lr_bounds", (1e-4, 1e-2)))
    l2_bounds: tuple[float, float] = tuple(cfg.get("l2_bounds", (1e-6, 1e-2)))
    seed: int | None = cfg.get("seed", 42)
    verbose: bool = bool(cfg.get("verbose", False))

    if not verbose:
        logging.getLogger("ax").setLevel(logging.WARNING)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        ax_client = AxClient(random_seed=seed)

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

        ax_client.create_experiment(
            name="unet_spectral_transfer_optimization",
            parameters=parameters,
            objectives={"val_loss": ObjectiveProperties(minimize=True)},
        )

        for trial_idx in range(num_trials):
            params, trial_index = ax_client.get_next_trial()
            trial_train_cfg = dict(cfg)
            trial_train_cfg.update(
                {
                    "base_channels": int(params["base_channels"]),
                    "depth": int(params["depth"]),
                    "kernel_size": int(params["kernel_size"]),
                    "learning_rate": float(params["learning_rate"]),
                    "l2_weight": float(params["l2_weight"]),
                    "epochs": epochs_per_trial,
                    "early_stopping_patience": epochs_per_trial,
                    "verbose": False,
                }
            )

            res = train_unet_region(train_dataset, val_dataset, config=trial_train_cfg)
            ax_client.complete_trial(trial_index=trial_index, raw_data={"val_loss": res["best_val_loss"]})

        best_params, metrics = ax_client.get_best_parameters()
        best_val_loss = float(metrics[0]["val_loss"])

        return {
            "best_parameters": best_params,
            "best_val_loss": best_val_loss,
            "ax_client": ax_client,
        }


def run_unet_pipeline(
    meta_df: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the end-to-end 1D U-Net spectral transformation pipeline.

    Supports both full regional spectrum mode (default) and sliding window
    patch mode via the 'use_sliding_window' configuration flag.

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
        - 'use_sliding_window' (bool): Whether to use sliding window patch decomposition (default False).
        - 'regions' (list[str] | None): Regions to train (default auto-detected).
        - 'source_tool' (str): Source tool identifier (default 'J4').
        - 'target_tool' (str): Target tool identifier (default 'H1').
        - 'val_ratio' (float): Fraction of sessions reserved for validation (default 0.2).
        - 'seed' (int | None): Random seed (default 42).
        - 'window_size_ev' (float): Window size in eV for sliding window mode (default 2.0).
        - 'sliding_stride_ev' (float): Stride in eV for sliding window mode (default 1.0).
        - 'use_bayesian_opt' (bool): Whether to run Ax optimization (default False).
        - 'bayesian_opt_config' (dict[str, Any] | None): Ax optimization settings.
        - 'train_config' (dict[str, Any] | None): Hyperparameters for training.
        - 'predict_source' (bool): Whether to run inference on source data (default True).
        - 'verbose' (bool): Verbosity flag (default False).

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'models': Dictionary mapping region -> trained UNet1D (or (model, w_size, stride)).
        - 'evaluation': Validation normalized MSE loss per region.
        - 'histories': Training loss histories per region.
        - 'bayesian_opt_results': Results from Ax (if use_bayesian_opt=True).
        - 'predictions': Tuple of (ary_intensity_predicted, ary_energy_predicted, meta_df_predicted).
    """
    cfg = config or {}
    use_sw: bool = bool(cfg.get("use_sliding_window", False))
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

    # Detect regions if not provided
    regions: list[str] | None = cfg.get("regions")
    if regions is None:
        paired_mask = meta_df["measurement_id_target"].notna()
        if source_tool:
            paired_mask = paired_mask & (meta_df["tool"] == source_tool)
        regions = sorted(meta_df[paired_mask]["region"].unique().tolist())

    if not regions:
        raise ValueError(f"No valid spectral regions found to train for source_tool='{source_tool}', target_tool='{target_tool}'")

    models: dict[str, Any] = {}
    eval_results: dict[str, float] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    bo_results: dict[str, dict[str, Any]] = {}

    for region in regions:
        mode_label = "Sliding Window U-Net" if use_sw else "Full Spectrum U-Net"
        if verbose:
            print(f"\n--- [{mode_label}] Processing Region: {region} ({source_tool} -> {target_tool}) ---")

        split_cfg = {
            "region": region,
            "source_tool": source_tool,
            "target_tool": target_tool,
            "val_ratio": val_ratio,
            "seed": seed,
        }
        train_full_ds, val_full_ds = split_session_datasets(
            meta_df, ary_intensity, ary_energy, config=split_cfg
        )

        region_train_cfg = dict(train_cfg)

        if use_sw:
            # Sliding window mode
            e_grid = ary_energy[train_full_ds.source_indices[0]]
            sw_calc_cfg = {
                "window_size_ev": cfg.get("window_size_ev", 2.0),
                "sliding_stride_ev": cfg.get("sliding_stride_ev", 1.0),
                "window_size_points": cfg.get("window_size_points"),
                "sliding_stride_points": cfg.get("sliding_stride_points"),
            }
            w_size, s_step = calculate_window_points(e_grid, config=sw_calc_cfg)

            if use_bo:
                bo_res = optimize_unet_hyperparameters(train_full_ds, val_full_ds, config=bo_cfg)
                bo_results[region] = bo_res
                if bo_res.get("best_parameters"):
                    region_train_cfg.update(bo_res["best_parameters"])

            train_patches = extract_sliding_windows(train_full_ds, config={"window_size": w_size, "stride": s_step})
            val_patches = extract_sliding_windows(val_full_ds, config={"window_size": w_size, "stride": s_step})
            train_ds: Dataset = SpectrumPatchDataset(train_patches["x_patches"], train_patches["y_patches"])
            val_ds: Dataset = SpectrumPatchDataset(val_patches["x_patches"], val_patches["y_patches"])

            train_res = train_unet_region(train_ds, val_ds, config=region_train_cfg)
            models[region] = (train_res["model"], w_size, s_step)
            eval_results[region] = train_res["best_val_loss"]
            histories[region] = train_res["history"]

        else:
            # Full regional spectrum mode
            if use_bo:
                bo_res = optimize_unet_hyperparameters(train_full_ds, val_full_ds, config=bo_cfg)
                bo_results[region] = bo_res
                if bo_res.get("best_parameters"):
                    region_train_cfg.update(bo_res["best_parameters"])

            train_res = train_unet_region(train_full_ds, val_full_ds, config=region_train_cfg)
            models[region] = train_res["model"]
            eval_results[region] = train_res["best_val_loss"]
            histories[region] = train_res["history"]

    # Generate predictions on source tool data
    predictions = None
    if predict_source:
        source_mask = (meta_df["tool"] == source_tool) & (meta_df["region"].isin(regions))
        source_df = meta_df[source_mask].copy()

        n_pred = len(source_df)
        n_points = ary_intensity.shape[1]
        pred_intensities = np.empty((n_pred, n_points), dtype=np.float64)
        pred_energies = np.empty((n_pred, n_points), dtype=np.float64)
        pred_meta_records: list[dict[str, Any]] = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for new_idx, (_, row) in enumerate(source_df.iterrows()):
            reg = str(row["region"])
            orig_idx = int(row["spectrum_index"])
            orig_meas_id = str(row["measurement_id"])
            pred_meas_id = format_predicted_measurement_id(orig_meas_id, source_tool, target_tool)

            pred_energies[new_idx] = ary_energy[orig_idx]
            x_raw = ary_intensity[orig_idx]

            if reg in models:
                if use_sw:
                    model, w_size, s_step = models[reg]
                    reconstructed = predict_sliding_window_spectrum(
                        model=model,
                        spectrum=x_raw,
                        config={"window_size": w_size, "stride": s_step},
                    )
                    pred_intensities[new_idx] = reconstructed
                else:
                    model = models[reg]
                    model.eval()
                    with torch.no_grad():
                        x_tensor = torch.from_numpy(x_raw.astype(np.float32)).unsqueeze(0).to(device)
                        y_pred = model(x_tensor).squeeze(0).cpu().numpy()
                    pred_intensities[new_idx] = y_pred
            else:
                pred_intensities[new_idx] = x_raw.copy()

            meta_rec = {
                "spectrum_index": new_idx,
                "material": row.get("material", "NMG"),
                "tool": target_tool,
                "measurement_id": pred_meas_id,
                "die": row.get("die", 0),
                "region": reg,
                "n_points": n_points,
                "time": row.get("time"),
                "source_tool": source_tool,
                "source_measurement_id": orig_meas_id,
                "source_spectrum_index": orig_idx,
                "is_predicted": True,
            }
            if "t7_code" in row:
                meta_rec["t7_code"] = row["t7_code"]
            pred_meta_records.append(meta_rec)

        pred_meta_df = pd.DataFrame(pred_meta_records)
        predictions = (pred_intensities, pred_energies, pred_meta_df)

    return {
        "models": models,
        "evaluation": eval_results,
        "histories": histories,
        "bayesian_opt_results": bo_results,
        "predictions": predictions,
    }
