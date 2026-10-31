"""1D U-Net neural network architecture for spectral transformation.

Supports both full regional spectrum mode and sliding window patch mode,
with global residual shortcut for stable inter-tool transfer.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.film import FiLMGenerator
from models.layers import (
    compute_conv1d_l2_reg,
    validate_3d_spectrum_tensor,
    validate_common_model_config,
    validate_odd_kernel_size,
    validate_positive_int,
)


class UNetConvBlock1D(nn.Module):
    """Double 1D convolutional block with normalization and activations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg = config or {}
        kernel_size = validate_odd_kernel_size(int(cfg.get("kernel_size", 5)))
        dropout = float(cfg.get("dropout", 0.0))
        use_batch_norm = bool(cfg.get("use_batch_norm", True))
        activation = str(cfg.get("activation", "relu"))
        padding = kernel_size // 2

        def _make_act() -> nn.Module:
            if activation == "leaky_relu":
                return nn.LeakyReLU(0.1, inplace=True)
            return nn.ReLU(inplace=True)

        layers: list[nn.Module] = []
        for idx, ch_in in enumerate((in_channels, out_channels)):
            layers.append(
                nn.Conv1d(ch_in, out_channels, kernel_size=kernel_size, padding=padding)
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(_make_act())
            if idx == 0 and dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet1D(nn.Module):
    """1D U-Net architecture with multi-scale contracting and expanding paths
    and optional FiLM.

    Features:
    - Multi-scale hierarchical feature extraction.
    - Long skip connections preserving high-frequency peak boundaries.
    - Robust linear interpolation in the decoder supporting arbitrary/odd
      sequence lengths.
    - Optional global residual connection y = x + UNet(x) for stable inter-tool
      delta learning.
    - Optional FiLM feature-wise linear modulation conditioning at the
      bottleneck.
    - Explicit L2 penalty computation on convolutional weights for
      SpectralLoss compatibility.

    Parameters
    ----------
    n_points : int
        Nominal number of sequence points (spectral length or patch window
        length).
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'residual_unet_channel' (int): Channels at level 0 for Residual U-Net
          (default 16).
        - 'unet_channel' (int): Channels at level 0 for Conventional U-Net
          (default 16).
        - 'depth' (int): Number of downsampling stages (default 3).
        - 'kernel_size' (int): Conv kernel size, positive odd int (default 5).
        - 'dropout' (float): Dropout probability (default 0.0).
        - 'use_batch_norm' (bool): Whether to use BatchNorm1d (default True).
        - 'residual' (bool): Whether to apply global residual shortcut
          y = x + delta (default True).
        - 'activation' (str): Activation function 'relu' or 'leaky_relu'
          (default 'relu').
        - 'use_film' (bool): Whether to enable FiLM conditioning at the
          bottleneck (default False).
    """

    def __init__(self, n_points: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        self.n_points = int(n_points)
        max_allowed_k = validate_common_model_config(cfg, self.n_points)


        self.residual = bool(cfg.get("residual", True))
        primary_ch_key = (
            "residual_unet_channel" if self.residual else "unet_channel"
        )
        fallback_ch_key = (
            "unet_channel" if self.residual else "residual_unet_channel"
        )
        raw_ch = cfg.get(primary_ch_key, cfg.get(fallback_ch_key, 16))
        self.base_channels = validate_positive_int(raw_ch, primary_ch_key)
        self.depth = validate_positive_int(cfg.get("depth", 3), "depth")

        min_points = 2**self.depth
        if self.n_points < min_points:
            raise ValueError(
                f"n_points ({self.n_points}) is too small for depth {self.depth}. "
                f"Requires at least {min_points} points (2**depth)."
            )

        self.kernel_size = validate_odd_kernel_size(
            cfg.get("kernel_size", 5), max_allowed=max_allowed_k
        )
        self.dropout = float(cfg.get("dropout", 0.0))
        self.use_batch_norm = bool(cfg.get("use_batch_norm", True))
        self.activation = str(cfg.get("activation", "relu"))
        self.use_film = bool(cfg.get("use_film", False))
        self.energy_scale_ev = float(cfg.get("energy_scale_ev", 1000.0))

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
                    in_channels=current_ch, out_channels=out_ch, config=block_cfg
                )
            )
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            ch_list.append(out_ch)
            current_ch = out_ch

        # Bottleneck stage
        bottleneck_ch = self.base_channels * (2**self.depth)
        self.bottleneck = UNetConvBlock1D(
            in_channels=current_ch, out_channels=bottleneck_ch, config=block_cfg
        )
        self.film_gen = (
            FiLMGenerator(channels_list=[bottleneck_ch], config=cfg)
            if self.use_film
            else None
        )

        # Decoder stages
        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_current = bottleneck_ch

        for skip_ch in reversed(ch_list):
            self.up_convs.append(nn.Conv1d(dec_current, skip_ch, kernel_size=1))
            self.decoders.append(
                UNetConvBlock1D(
                    in_channels=skip_ch * 2, out_channels=skip_ch, config=block_cfg
                )
            )
            dec_current = skip_ch

        self.final_conv = nn.Conv1d(dec_current, 1, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through 1D U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, n_points).
        cond : dict[str, torch.Tensor] | torch.Tensor | None, optional
            Optional metadata conditioning features for FiLM modulation.

        Returns
        -------
        torch.Tensor
            Transformed spectrum of shape (batch_size, 1, n_points).

        Raises
        ------
        ValueError
            If x is not a 3D tensor of shape (batch_size, 1, n_points).
        """
        validate_3d_spectrum_tensor(x)

        skips: list[torch.Tensor] = []
        feat = x

        for encoder, pool in zip(self.encoders, self.pools):
            feat = encoder(feat)
            skips.append(feat)
            feat = pool(feat)

        feat = self.bottleneck(feat)
        if self.use_film and cond is not None and self.film_gen is not None:
            gamma, beta = self.film_gen(cond)[0]
            feat = gamma.unsqueeze(-1) * feat + beta.unsqueeze(-1)

        for skip, up_conv, decoder in zip(
            reversed(skips), self.up_convs, self.decoders
        ):
            feat = F.interpolate(
                feat, size=skip.shape[-1], mode="linear", align_corners=False
            )
            feat = up_conv(feat)
            feat = decoder(torch.cat([feat, skip], dim=1))

        delta = self.final_conv(feat)
        return (x + delta) if self.residual else delta

    def get_l2_regularization(self) -> torch.Tensor:
        """Compute the sum of squared weights of convolutional layers."""
        return compute_conv1d_l2_reg(self)


class ResidualUNet1D(UNet1D):
    """1D U-Net with global residual shortcut: y = x + UNet(x)."""

    def __init__(self, n_points: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(
            n_points=n_points, config={**(config or {}), "residual": True}
        )


class ConventionalUNet1D(UNet1D):
    """Conventional 1D U-Net without global shortcut: y = UNet(x)."""

    def __init__(self, n_points: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(
            n_points=n_points, config={**(config or {}), "residual": False}
        )
