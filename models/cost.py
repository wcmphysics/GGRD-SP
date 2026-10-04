"""Cost function definitions for spectral sequence-to-sequence transformation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def format_loss_log10(loss_val: float) -> str:
    """Format a loss value with both raw and log10 scale representations.

    Parameters
    ----------
    loss_val : float
        Numerical loss value.

    Returns
    -------
    str
        Formatted string, e.g. '0.000456 (log10: -3.3410)'.
    """
    if loss_val <= 0.0 or math.isnan(loss_val) or math.isinf(loss_val):
        return f"{loss_val:.6f} (log10: N/A)"
    log_val = math.log10(loss_val)
    return f"{loss_val:.6f} (log10: {log_val:+.4f})"


class NormalizedMSELoss(nn.Module):
    """Normalized Mean Squared Error loss with L2 weight regularization.

    Normalizes true and predicted spectra by the maximum intensity of the true
    spectrum, then computes MSE:
        loss = MSE(y_pred / max(|y_true|), y_true / max(|y_true|)) + l2_weight * L2_penalty

    Parameters
    ----------
    l2_weight : float, optional
        Weight for model convolutional parameter L2 regularization (default 0.0).
    eps : float, optional
        Small positive floor to prevent division by zero (default 1e-4).
    region_weights : dict[str, float] | None, optional
        Optional regional weighting mapping region names to error multipliers (default None).
    """

    def __init__(
        self,
        l2_weight: float = 0.0,
        eps: float = 1e-4,
        region_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.l2_weight = float(l2_weight)
        self.eps = float(eps)
        self.region_weights = region_weights or {}

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        model: nn.Module | None = None,
        max_val: torch.Tensor | None = None,
        region: str | None = None,
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
        model : nn.Module | None, optional
            The model instance, needed to compute L2 regularization if l2_weight > 0.
        max_val : torch.Tensor | None, optional
            Precomputed normalization maximum tensor. If None, computes max(|y_true|) per sample.
        region : str | None, optional
            Name of the spectral region (used for regional weighting if configured).

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        if y_pred.dim() == 3 and y_pred.size(1) == 1:
            y_pred = y_pred.squeeze(1)
        if y_true.dim() == 3 and y_true.size(1) == 1:
            y_true = y_true.squeeze(1)

        if max_val is None:
            max_val, _ = torch.max(torch.abs(y_true), dim=-1, keepdim=True)
        max_val = torch.clamp(max_val, min=self.eps)

        norm_pred = y_pred / max_val
        norm_true = y_true / max_val

        mse_loss = torch.mean((norm_pred - norm_true) ** 2)

        # Apply regional weighting if specified
        if region is not None and region in self.region_weights:
            weight = float(self.region_weights[region])
            mse_loss = mse_loss * weight

        # Add L2 parameter regularization penalty if model provided
        if self.l2_weight > 0.0 and model is not None:
            if hasattr(model, "get_l2_regularization"):
                l2_reg = model.get_l2_regularization()
            else:
                l2_reg = sum(
                    torch.sum(p**2)
                    for name, p in model.named_parameters()
                    if "weight" in name and p.requires_grad and p.dim() > 1
                )
            return mse_loss + self.l2_weight * l2_reg

        return mse_loss
