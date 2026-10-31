"""1D Feature Pyramid Network (FPN) for multi-scale XPS spectral transfer."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.film import FiLMGenerator
from models.layers import (
    SafeBatchNorm1d,
    build_conv1d_layer,
    compute_conv1d_l2_reg,
    validate_3d_spectrum_tensor,
    validate_common_model_config,
    validate_odd_kernel_size,
    validate_positive_int,
)


class FPNConvBlock1D(nn.Module):
    """Two-layer 1D convolution block with reflection padding
    and normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.kernel_size = validate_odd_kernel_size(config.get("kernel_size", 3))
        self.use_batch_norm = bool(config.get("use_batch_norm", True))
        self.dropout = float(config.get("dropout", 0.0))
        act_name = str(config.get("activation", "relu")).lower()

        # Build first convolution layer with safe reflection padding.
        self.conv1 = build_conv1d_layer(
            (in_channels, out_channels), kernel_size=self.kernel_size
        )
        self.norm1: nn.Module | None = (
            SafeBatchNorm1d(out_channels) if self.use_batch_norm else None
        )
        self.act1: nn.Module = (
            nn.LeakyReLU(0.2, inplace=True)
            if act_name == "leaky_relu"
            else nn.ReLU(inplace=True)
        )
        self.drop1: nn.Module | None = (
            nn.Dropout(self.dropout) if self.dropout > 0.0 else None
        )

        # Build second convolution layer with safe reflection padding.
        self.conv2 = build_conv1d_layer(
            (out_channels, out_channels), kernel_size=self.kernel_size
        )
        self.norm2: nn.Module | None = (
            SafeBatchNorm1d(out_channels) if self.use_batch_norm else None
        )
        self.act2: nn.Module = (
            nn.LeakyReLU(0.2, inplace=True)
            if act_name == "leaky_relu"
            else nn.ReLU(inplace=True)
        )
        self.drop2: nn.Module | None = (
            nn.Dropout(self.dropout) if self.dropout > 0.0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through two convolutional layers."""
        h = self.conv1(x)
        if self.norm1 is not None:
            h = self.norm1(h)
        h = self.act1(h)
        if self.drop1 is not None:
            h = self.drop1(h)

        h = self.conv2(h)
        if self.norm2 is not None:
            h = self.norm2(h)
        h = self.act2(h)
        if self.drop2 is not None:
            h = self.drop2(h)
        return h


class FPN1D(nn.Module):
    """1D Feature Pyramid Network with multi-scale fusion head for XPS."""

    def __init__(
        self,
        n_points: int,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg = config or {}
        if bool(cfg.get("use_sliding_window", False)):
            raise ValueError(
                "Sliding window mode (use_sliding_window=True) is not supported "
                "for FPN1D. Multi-field-of-view architectures process "
                "full regional spectra directly."
            )

        self.n_points = int(n_points)
        max_allowed_k = validate_common_model_config(cfg, self.n_points)
        self.kernel_size = validate_odd_kernel_size(
            cfg.get("kernel_size", 3), max_allowed=max_allowed_k
        )
        self.depth = validate_positive_int(cfg.get("depth", 3), "depth")
        min_points = 2**self.depth
        if self.n_points < min_points:
            raise ValueError(
                f"n_points ({self.n_points}) is too small for depth {self.depth}. "
                f"Requires at least {min_points} points (2**depth)."
            )

        self.fpn_base_channel = validate_positive_int(
            cfg.get("fpn_base_channel", 16), "fpn_base_channel"
        )
        self.fpn_pyramid_channel = validate_positive_int(
            cfg.get("fpn_pyramid_channel", 32), "fpn_pyramid_channel"
        )
        self.residual = bool(cfg.get("residual", True))
        self.use_batch_norm = bool(cfg.get("use_batch_norm", True))
        self.dropout = float(cfg.get("dropout", 0.0))
        act_name = str(cfg.get("activation", "relu")).lower()
        self.use_film = bool(cfg.get("use_film", False))
        self.energy_scale_ev = float(cfg.get("energy_scale_ev", 1000.0))

        block_cfg = {
            "kernel_size": self.kernel_size,
            "use_batch_norm": self.use_batch_norm,
            "dropout": self.dropout,
            "activation": act_name,
        }

        # Build pyramid pathways and multi-scale fusion head.
        self._build_pyramid_pathways(block_cfg)
        self._build_fusion_head(cfg, act_name)

    def _build_pyramid_pathways(self, block_cfg: dict[str, Any]) -> None:
        """Construct bottom-up encoders, lateral 1x1 convs, and smooth convs."""
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.laterals = nn.ModuleList()
        current_in = 1
        for i in range(self.depth):
            out_ch = self.fpn_base_channel * (2**i)
            if i > 0:
                self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            self.encoders.append(
                FPNConvBlock1D(current_in, out_ch, block_cfg)
            )
            self.laterals.append(
                nn.Conv1d(out_ch, self.fpn_pyramid_channel, kernel_size=1)
            )
            current_in = out_ch

        self.smooth_convs = nn.ModuleList()
        for _ in range(self.depth - 1):
            self.smooth_convs.append(
                build_conv1d_layer(
                    (self.fpn_pyramid_channel, self.fpn_pyramid_channel),
                    kernel_size=self.kernel_size,
                )
            )

    def _build_fusion_head(
        self, cfg: dict[str, Any], act_name: str
    ) -> None:
        """Construct multi-scale fusion head, FiLM generator, and output conv."""
        fused_in_channels = self.depth * self.fpn_pyramid_channel
        self.fuse_conv = build_conv1d_layer(
            (fused_in_channels, self.fpn_pyramid_channel),
            kernel_size=self.kernel_size,
        )
        self.fuse_norm: nn.Module | None = (
            SafeBatchNorm1d(self.fpn_pyramid_channel)
            if self.use_batch_norm
            else None
        )
        self.fuse_act: nn.Module = (
            nn.LeakyReLU(0.2, inplace=True)
            if act_name == "leaky_relu"
            else nn.ReLU(inplace=True)
        )
        self.fuse_drop: nn.Module | None = (
            nn.Dropout(self.dropout) if self.dropout > 0.0 else None
        )
        self.film_gen = (
            FiLMGenerator(channels_list=[self.fpn_pyramid_channel], config=cfg)
            if self.use_film
            else None
        )
        self.head = nn.Conv1d(self.fpn_pyramid_channel, 1, kernel_size=1)

    def _bottom_up_pass(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Compute bottom-up hierarchical feature maps across stages."""
        features: list[torch.Tensor] = []
        feat = x
        for i, encoder in enumerate(self.encoders):
            if i > 0:
                feat = self.pools[i - 1](feat)
            feat = encoder(feat)
            features.append(feat)
        return features

    def _top_down_pass(
        self, features: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Build feature pyramid levels from coarsest to finest."""
        pyramid_levels: list[torch.Tensor] = [None] * self.depth  # type: ignore
        pyramid_levels[-1] = self.laterals[-1](features[-1])

        for i in range(self.depth - 2, -1, -1):
            lat = self.laterals[i](features[i])
            up = F.interpolate(
                pyramid_levels[i + 1],
                size=lat.shape[-1],
                mode="linear",
                align_corners=False,
            )
            pyramid_levels[i] = self.smooth_convs[i](lat + up)
        return pyramid_levels

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through bottom-up encoder, lateral FPN, and
        fusion head.
        """
        validate_3d_spectrum_tensor(x)
        features = self._bottom_up_pass(x)
        pyramid = self._top_down_pass(features)

        # Upsample all pyramid levels to input sequence length and concatenate.
        seq_len = x.shape[-1]
        upsampled = [
            F.interpolate(
                p, size=seq_len, mode="linear", align_corners=False
            )
            for p in pyramid
        ]
        fused = torch.cat(upsampled, dim=1)
        fused = self.fuse_conv(fused)
        if self.fuse_norm is not None:
            fused = self.fuse_norm(fused)
        fused = self.fuse_act(fused)
        if self.fuse_drop is not None:
            fused = self.fuse_drop(fused)

        if self.use_film and cond is not None and self.film_gen is not None:
            gamma, beta = self.film_gen(cond)[0]
            fused = gamma.unsqueeze(-1) * fused + beta.unsqueeze(-1)

        delta = self.head(fused)
        return (x + delta) if self.residual else delta

    def get_l2_regularization(self) -> torch.Tensor:
        """Compute the sum of squared weights of convolutional layers."""
        return compute_conv1d_l2_reg(self)


__all__ = [
    "FPNConvBlock1D",
    "FPN1D",
]
