"""Baseline 1D Residual CNN architecture and custom loss for spectral transformation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class Residual1DCNN(nn.Module):
    """Residual 1D CNN with 3 convolutional layers and an identity shortcut.

    Transforms a 1D regional spectrum sequence from source tool to target tool
    while preserving sequence length: y = x + Delta(x).

    Parameters
    ----------
    n_points : int
        Number of spectral data points (length of input sequence).
    config : dict[str, Any] | None, optional
        Configuration dictionary containing:
        - 'hidden_channels' (int): Feature channels in hidden layers (default 32).
        - 'kernel_size' (int): Convolution kernel size, must be positive odd int (default 5).
        - 'dropout' (float): Dropout probability between conv layers (default 0.0).
        - 'use_batch_norm' (bool): Whether to use BatchNorm1d (default True).
    """

    def __init__(self, n_points: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        hidden_channels: int = int(cfg.get("hidden_channels", 32))
        kernel_size: int = int(cfg.get("kernel_size", 5))
        dropout: float = float(cfg.get("dropout", 0.0))
        use_batch_norm: bool = bool(cfg.get("use_batch_norm", True))

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )

        padding = kernel_size // 2

        layers: list[nn.Module] = []

        # Layer 1: 1 -> hidden_channels
        layers.append(nn.Conv1d(1, hidden_channels, kernel_size=kernel_size, padding=padding))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        # Layer 2: hidden_channels -> hidden_channels
        layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        # Layer 3: hidden_channels -> 1
        layers.append(nn.Conv1d(hidden_channels, 1, kernel_size=kernel_size, padding=padding))

        self.conv_block = nn.Sequential(*layers)
        self.n_points = n_points

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying residual transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_points) or (batch_size, 1, n_points).

        Returns
        -------
        torch.Tensor
            Transformed spectrum of shape (batch_size, n_points).
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


class NormalizedMSELoss(nn.Module):
    """Normalized Mean Squared Error loss with optional L2 weight regularization.

    Normalizes true and predicted spectra by the maximum intensity of the true
    spectrum, then computes MSE:
        loss = MSE(y_pred / max(y_true), y_true / max(y_true)) + l2_weight * L2_norm

    Parameters
    ----------
    l2_weight : float, optional
        Weight for model parameter L2 regularization (default 0.0).
    eps : float, optional
        Small positive floor to prevent division by zero (default 1e-4).
    """

    def __init__(self, l2_weight: float = 0.0, eps: float = 1e-4) -> None:
        super().__init__()
        self.l2_weight = float(l2_weight)
        self.eps = float(eps)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        model: Residual1DCNN | None = None,
        max_val: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the normalized MSE loss.

        Guarantees 2D matching between y_pred and y_true to prevent
        accidental cross-batch broadcasting: (B, 1, L) vs (B, L) -> (B, B, L).

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted intensity of shape (batch_size, n_points) or (batch_size, 1, n_points).
        y_true : torch.Tensor
            Target intensity of shape (batch_size, n_points) or (batch_size, 1, n_points).
        model : Residual1DCNN | None, optional
            The model instance, needed to compute L2 regularization if l2_weight > 0.
        max_val : torch.Tensor | None, optional
            Precomputed normalization maximum tensor. If None, computes max(|y_true|) per sample.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        # Ensure 2D (B, L) shapes to avoid accidental (B, 1, L) - (B, L) -> (B, B, L)
        if y_pred.dim() == 3 and y_pred.size(1) == 1:
            y_pred = y_pred.squeeze(1)
        if y_true.dim() == 3 and y_true.size(1) == 1:
            y_true = y_true.squeeze(1)

        # Compute maximum of absolute true target spectrum per sample along sequence dimension
        if max_val is None:
            max_val, _ = torch.max(torch.abs(y_true), dim=-1, keepdim=True)
        max_val = torch.clamp(max_val, min=self.eps)

        norm_pred = y_pred / max_val
        norm_true = y_true / max_val

        mse_loss = torch.mean((norm_pred - norm_true) ** 2)

        if self.l2_weight > 0.0 and model is not None:
            l2_reg = model.get_l2_regularization()
            return mse_loss + self.l2_weight * l2_reg

        return mse_loss
