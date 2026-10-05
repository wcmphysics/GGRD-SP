"""1D U-Net neural network architecture, training, and pipeline for spectral transformation.

Supports both full regional spectrum mode and sliding window patch mode,
with global residual shortcut for stable inter-tool transfer.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.film import FiLMGenerator


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
    """1D U-Net architecture with multi-scale contracting and expanding paths and optional FiLM.

    Features:
    - Multi-scale hierarchical feature extraction.
    - Long skip connections preserving high-frequency peak boundaries.
    - Robust linear interpolation in the decoder supporting arbitrary/odd sequence lengths.
    - Optional global residual connection y = x + UNet(x) for stable inter-tool delta learning.
    - Optional FiLM feature-wise linear modulation conditioning at the bottleneck.
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
        - 'use_film' (bool): Whether to enable FiLM conditioning at the bottleneck (default False).
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
        self.use_film = bool(cfg.get("use_film", False))

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

        if self.use_film:
            self.film_gen = FiLMGenerator(channels_list=[bottleneck_ch], config=cfg)
        else:
            self.film_gen = None

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

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through 1D U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_points) or (batch_size, 1, n_points).
        cond : dict[str, torch.Tensor] | torch.Tensor | None, optional
            Optional metadata conditioning features for FiLM modulation.

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
        if self.use_film and cond is not None and self.film_gen is not None:
            gamma, beta = self.film_gen(cond)[0]
            feat = gamma.unsqueeze(-1) * feat + beta.unsqueeze(-1)

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


class ResidualUNet1D(UNet1D):
    """1D U-Net with global residual shortcut: y = x + UNet(x)."""

    def __init__(self, n_points: int, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        cfg["residual"] = True
        super().__init__(n_points=n_points, config=cfg)


class ConventionalUNet1D(UNet1D):
    """Conventional 1D U-Net without global shortcut: y = UNet(x)."""

    def __init__(self, n_points: int, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        cfg["residual"] = False
        super().__init__(n_points=n_points, config=cfg)

