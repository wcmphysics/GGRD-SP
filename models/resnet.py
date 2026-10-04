"""1D ResNet neural network architecture for spectral transformation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ResNet1D(nn.Module):
    """Residual 1D CNN with 3 convolutional layers, mirror padding, and identity shortcut.

    Transforms a 1D regional spectrum sequence from source tool to target tool
    while preserving sequence length: y = x + Delta(x).

    Uses reflection (mirror) padding to prevent boundary attenuation artifacts
    at sequence edges.

    Parameters
    ----------
    n_points : int
        Number of spectral data points (length of input sequence or patch window).
    config : dict[str, Any] | None, optional
        Configuration dictionary containing:
        - 'hidden_channels' (int): Feature channels in hidden layers (default 32).
        - 'kernel_size' (int): Convolution kernel size, positive odd int (default 5).
        - 'dropout' (float): Dropout probability between conv layers (default 0.0).
        - 'use_batch_norm' (bool): Whether to use BatchNorm1d in hidden layers (default True).
        - 'padding_mode' (str): Padding mode 'reflect' (mirror) or 'zeros' (default 'reflect').
    """

    def __init__(self, n_points: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        hidden_channels: int = int(cfg.get("hidden_channels", 32))
        kernel_size: int = int(cfg.get("kernel_size", 5))
        dropout: float = float(cfg.get("dropout", 0.0))
        use_batch_norm: bool = bool(cfg.get("use_batch_norm", True))
        padding_mode: str = str(cfg.get("padding_mode", "reflect"))

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )

        pad = kernel_size // 2
        self.n_points = n_points

        def _make_conv_layer(in_ch: int, out_ch: int) -> nn.Module:
            layers: list[nn.Module] = []
            if pad > 0:
                if padding_mode == "reflect" and n_points > pad:
                    layers.append(nn.ReflectionPad1d(pad))
                    layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=0))
                else:
                    layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad))
            else:
                layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=0))
            return nn.Sequential(*layers) if len(layers) > 1 else layers[0]

        layers: list[nn.Module] = []

        # Layer 1: 1 -> hidden_channels
        layers.append(_make_conv_layer(1, hidden_channels))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        # Layer 2: hidden_channels -> hidden_channels
        layers.append(_make_conv_layer(hidden_channels, hidden_channels))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        # Layer 3: hidden_channels -> 1 (pure linear output, no BatchNorm, no activation)
        layers.append(_make_conv_layer(hidden_channels, 1))

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying residual transformation: y = x + Delta(x).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_points) or (batch_size, 1, n_points).

        Returns
        -------
        torch.Tensor
            Transformed spectrum matching input sequence dimensionality.
        """
        orig_dim = x.dim()
        if orig_dim == 2:
            x_in = x.unsqueeze(1)
        elif orig_dim == 3:
            x_in = x
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        delta = self.conv_block(x_in)
        out = x_in + delta

        if orig_dim == 2:
            return out.squeeze(1)
        return out

    def get_l2_regularization(self) -> torch.Tensor:
        """Compute the sum of squared weights (L2 regularization penalty).

        Only penalizes convolutional weight matrices; biases and BatchNorm
        parameters are excluded to preserve baseline intensity offsets.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing sum of squares of convolutional weights.
        """
        device = next(self.parameters()).device
        l2_sum = torch.tensor(0.0, device=device)
        for name, param in self.named_parameters():
            if param.requires_grad and name.endswith(".weight") and "bn" not in name:
                l2_sum = l2_sum + torch.sum(param ** 2)
        return l2_sum

