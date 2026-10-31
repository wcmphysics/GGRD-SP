"""Dilated Temporal Convolutional Network (TCN) 1D for XPS spectra.

Physical Rationale for Non-Causal Symmetric Reflection Padding:
Standard temporal time-series TCN architectures employ causal (unidirectional
left-sided) padding to prevent lookahead into the future. In XPS (X-ray
Photoelectron Spectroscopy), however, the independent axis is binding energy
(eV)—a static, bidirectional physical property rather than an arrow of time.
Spectral features such as photoelectron core-level peaks, spin-orbit split
doublets, satellite peaks, and inelastic background slopes possess bidirectional
continuity and symmetric physical broadening. Using causal padding would
introduce an artificial directional asymmetry, shifting peak centroids along
the binding energy axis and causing boundary artifacts. Therefore, TCN1D
strictly uses non-causal symmetric reflection padding (via SafeReflectionPad1d),
preserving peak alignment and sequence length without boundary distortion.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from models.film import FiLMGenerator
from models.layers import (
    SafeBatchNorm1d,
    SafeReflectionPad1d,
    compute_conv1d_l2_reg,
    validate_3d_spectrum_tensor,
    validate_common_model_config,
    validate_odd_kernel_size,
    validate_positive_int,
)


class TCNResidualBlock1D(nn.Module):
    """Dilated residual block with non-causal symmetric reflection padding.

    In XPS spectra, binding energy is bidirectional. Non-causal symmetric
    reflection padding avoids artificial peak shifts and edge distortion.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.use_batch_norm = bool(config.get("use_batch_norm", True))
        self.dropout = float(config.get("dropout", 0.0))
        act_name = str(config.get("activation", "relu")).lower()

        pad = (self.kernel_size - 1) * self.dilation // 2

        # First dilated convolution branch with symmetric reflection padding.
        self.conv1 = nn.Sequential(
            SafeReflectionPad1d(pad),
            nn.Conv1d(
                self.in_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
            ),
        )
        self.norm1: nn.Module | None = (
            SafeBatchNorm1d(self.out_channels) if self.use_batch_norm else None
        )
        self.act1: nn.Module = (
            nn.LeakyReLU(0.2, inplace=True)
            if act_name == "leaky_relu"
            else nn.ReLU(inplace=True)
        )
        self.drop1: nn.Module | None = (
            nn.Dropout(self.dropout) if self.dropout > 0.0 else None
        )

        # Second dilated convolution branch with symmetric reflection padding.
        self.conv2 = nn.Sequential(
            SafeReflectionPad1d(pad),
            nn.Conv1d(
                self.out_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
            ),
        )
        self.norm2: nn.Module | None = (
            SafeBatchNorm1d(self.out_channels) if self.use_batch_norm else None
        )
        self.act2: nn.Module = (
            nn.LeakyReLU(0.2, inplace=True)
            if act_name == "leaky_relu"
            else nn.ReLU(inplace=True)
        )
        self.drop2: nn.Module | None = (
            nn.Dropout(self.dropout) if self.dropout > 0.0 else None
        )

        # Shortcut projection matching channel dimensions.
        if self.in_channels != self.out_channels:
            shortcut_layers: list[nn.Module] = [
                nn.Conv1d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=1,
                    bias=False,
                )
            ]
            if self.use_batch_norm:
                shortcut_layers.append(SafeBatchNorm1d(self.out_channels))
            self.shortcut: nn.Module = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through two dilated convolutions with shortcut."""
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

        return self.act2(h + self.shortcut(x))


class TCN1D(nn.Module):
    """Dilated Temporal Convolutional Network 1D with symmetric padding."""

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
                "for TCN1D. Multi-field-of-view architectures process "
                "full regional spectra directly."
            )

        self.n_points = int(n_points)
        max_allowed_k = validate_common_model_config(cfg, self.n_points)
        self.kernel_size = validate_odd_kernel_size(
            cfg.get("kernel_size", 5), max_allowed=max_allowed_k
        )
        self.tcn_channel = validate_positive_int(
            cfg.get("tcn_channel", 32), "tcn_channel"
        )
        raw_dilations: Sequence[int] = cfg.get(
            "dilations", (1, 2, 4, 8, 16)
        )
        self.dilations = [validate_positive_int(d, "dilation") for d in raw_dilations]
        self.residual = bool(cfg.get("residual", True))
        self.use_film = bool(cfg.get("use_film", False))
        self.energy_scale_ev = float(cfg.get("energy_scale_ev", 1000.0))

        # Stack dilated residual blocks with exponentially increasing dilation.
        self.blocks = nn.ModuleList()
        current_ch = 1
        for d in self.dilations:
            block = TCNResidualBlock1D(
                in_channels=current_ch,
                out_channels=self.tcn_channel,
                kernel_size=self.kernel_size,
                dilation=d,
                config=cfg,
            )
            self.blocks.append(block)
            current_ch = self.tcn_channel

        self.film_gen = (
            FiLMGenerator(channels_list=[self.tcn_channel], config=cfg)
            if self.use_film
            else None
        )
        self.head = nn.Conv1d(self.tcn_channel, 1, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through stacked dilated blocks and projection head."""
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
    "TCNResidualBlock1D",
    "TCN1D",
]
