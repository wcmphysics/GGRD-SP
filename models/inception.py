"""InceptionTime 1D multi-field-of-view neural network for XPS spectra."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from models.film import FiLMGenerator
from models.layers import (
    SafeBatchNorm1d,
    SafeReflectionPad1d,
    build_conv1d_layer,
    compute_conv1d_l2_reg,
    validate_3d_spectrum_tensor,
    validate_common_model_config,
    validate_odd_kernel_size,
    validate_positive_int,
)


class InceptionModule1D(nn.Module):
    """Multi-scale parallel 1D Inception module with bottleneck and pooling."""

    def __init__(
        self,
        in_channels: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.inception_channel = validate_positive_int(
            config.get("inception_channel", 16), "inception_channel"
        )
        self.bottleneck_channels = validate_positive_int(
            config.get("bottleneck_channels", 16), "bottleneck_channels"
        )
        self.kernel_sizes: list[int] = list(
            config.get("kernel_sizes", [3, 7, 15, 31])
        )
        self.use_bottleneck = bool(config.get("use_bottleneck", True))
        self.padding_mode = str(config.get("padding_mode", "reflect"))
        self.use_batch_norm = bool(config.get("use_batch_norm", True))
        self.dropout = float(config.get("dropout", 0.0))
        act_name = str(config.get("activation", "relu")).lower()

        # Build 1x1 bottleneck convolution to reduce channel dimension.
        if self.in_channels > 1 and self.use_bottleneck:
            self.bottleneck: nn.Module | None = nn.Conv1d(
                self.in_channels,
                self.bottleneck_channels,
                kernel_size=1,
                bias=False,
            )
            branch_in = self.bottleneck_channels
        else:
            self.bottleneck = None
            branch_in = self.in_channels

        # Build heterogeneous multi-scale parallel conv branches.
        self.conv_branches = nn.ModuleList()
        for k in self.kernel_sizes:
            self.conv_branches.append(
                build_conv1d_layer(
                    (branch_in, self.inception_channel),
                    kernel_size=k,
                    padding_mode=self.padding_mode,
                )
            )

        # Build parallel max-pooling branch with padding to preserve length.
        self.pool_pad: nn.Module = (
            SafeReflectionPad1d(1)
            if self.padding_mode == "reflect"
            else nn.ConstantPad1d((1, 1), 0.0)
        )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1)
        self.pool_conv = nn.Conv1d(
            self.in_channels,
            self.inception_channel,
            kernel_size=1,
            bias=False,
        )

        total_out = self.inception_channel * (len(self.kernel_sizes) + 1)
        self.out_channels = total_out
        self.norm: nn.Module | None = (
            SafeBatchNorm1d(total_out) if self.use_batch_norm else None
        )
        self.act: nn.Module = (
            nn.LeakyReLU(0.2, inplace=True)
            if act_name == "leaky_relu"
            else nn.ReLU(inplace=True)
        )
        self.drop: nn.Module | None = (
            nn.Dropout(self.dropout) if self.dropout > 0.0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input tensor across all multi-scale parallel branches."""
        x_conv = self.bottleneck(x) if self.bottleneck is not None else x
        branch_feats = [branch(x_conv) for branch in self.conv_branches]

        pool_feat = self.pool_conv(self.max_pool(self.pool_pad(x)))
        branch_feats.append(pool_feat)

        out = torch.cat(branch_feats, dim=1)
        if self.norm is not None:
            out = self.norm(out)
        out = self.act(out)
        if self.drop is not None:
            out = self.drop(out)
        return out


class InceptionResidualBlock1D(nn.Module):
    """Stacked Inception modules with residual identity or 1x1 shortcut."""

    def __init__(
        self,
        in_channels: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.modules_per_block = validate_positive_int(
            config.get("modules_per_block", 2), "modules_per_block"
        )
        self.use_batch_norm = bool(config.get("use_batch_norm", True))
        act_name = str(config.get("activation", "relu")).lower()
        self.act: nn.Module = (
            nn.LeakyReLU(0.2, inplace=True)
            if act_name == "leaky_relu"
            else nn.ReLU(inplace=True)
        )

        self.modules_list = nn.ModuleList()
        current_in = in_channels
        for _ in range(self.modules_per_block):
            module = InceptionModule1D(current_in, config)
            self.modules_list.append(module)
            current_in = module.out_channels
        self.out_channels = current_in

        # Shortcut projection to match channel dimensions if needed.
        if in_channels != self.out_channels:
            shortcut_layers: list[nn.Module] = [
                nn.Conv1d(
                    in_channels, self.out_channels, kernel_size=1, bias=False
                )
            ]
            if self.use_batch_norm:
                shortcut_layers.append(SafeBatchNorm1d(self.out_channels))
            self.shortcut: nn.Module = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stacked Inception modules with shortcut."""
        h = x
        for module in self.modules_list:
            h = module(h)
        return self.act(h + self.shortcut(x))


class InceptionTime1D(nn.Module):
    """InceptionTime 1D deep network for cross-tool spectral transfer."""

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
                "for InceptionTime1D. Multi-field-of-view architectures process "
                "full regional spectra directly."
            )

        self.n_points = int(n_points)
        max_allowed_k = validate_common_model_config(cfg, self.n_points)
        raw_kernel_sizes: Sequence[int] = cfg.get(
            "kernel_sizes", (3, 7, 15, 31)
        )
        self.kernel_sizes = [
            validate_odd_kernel_size(k, max_allowed=max_allowed_k)
            for k in raw_kernel_sizes
        ]
        self.num_blocks = validate_positive_int(
            cfg.get("num_blocks", 3), "num_blocks"
        )
        self.residual = bool(cfg.get("residual", True))
        self.use_film = bool(cfg.get("use_film", False))
        self.energy_scale_ev = float(cfg.get("energy_scale_ev", 1000.0))

        module_cfg = {**cfg, "kernel_sizes": self.kernel_sizes}
        self.blocks = nn.ModuleList()
        current_ch = 1
        for _ in range(self.num_blocks):
            block = InceptionResidualBlock1D(current_ch, module_cfg)
            self.blocks.append(block)
            current_ch = block.out_channels

        self.final_channels = current_ch
        self.film_gen = (
            FiLMGenerator(channels_list=[self.final_channels], config=cfg)
            if self.use_film
            else None
        )
        self.head = nn.Conv1d(self.final_channels, 1, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through InceptionTime blocks and output head."""
        validate_3d_spectrum_tensor(x)
        h = x
        for block in self.blocks:
            h = block(h)

        if self.use_film and cond is not None and self.film_gen is not None:
            gamma, beta = self.film_gen(cond)[0]
            h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)

        delta = self.head(h)
        return (x + delta) if self.residual else delta

    def get_l2_regularization(self) -> torch.Tensor:
        """Compute the sum of squared weights of convolutional layers."""
        return compute_conv1d_l2_reg(self)


__all__ = [
    "InceptionModule1D",
    "InceptionResidualBlock1D",
    "InceptionTime1D",
]
