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
    """Normalized Mean Squared Error loss with optional shape, scale, and L2 regularization.

    Normalizes true and predicted spectra by the maximum intensity of the true
    spectrum, then computes a weighted combination of Normalized MSE, Cosine Shape Loss,
    and Scale Penalty:
        loss = w_mse * L_mse + w_shape * L_shape + w_scale * L_scale + l2_weight * L2_penalty

    Parameters
    ----------
    l2_weight : float, optional
        Weight for model parameter L2 regularization (default 0.0).
    eps : float, optional
        Small positive floor to prevent division by zero (default 1e-4).
    region_weights : dict[str, float] | None, optional
        Optional regional weighting mapping region names to error multipliers (default None).
    w_mse : float, optional
        Weight for normalized MSE loss component (default 1.0).
    w_shape : float, optional
        Weight for scale-invariant cosine shape distance (default 0.0).
    w_scale : float, optional
        Weight for relative peak amplitude scale discrepancy (default 0.0).
    config : dict[str, Any] | None, optional
        Optional configuration dictionary packaging loss weights.
    """

    def __init__(
        self,
        l2_weight: float = 0.0,
        eps: float = 1e-4,
        region_weights: dict[str, float] | None = None,
        w_mse: float = 1.0,
        w_shape: float = 0.0,
        w_scale: float = 0.0,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg = config or {}
        self.l2_weight = float(cfg.get("l2_weight", l2_weight))
        self.eps = float(cfg.get("eps", eps))
        self.region_weights = dict(cfg.get("region_weights") or region_weights or {})
        self.w_mse = float(cfg.get("w_mse", w_mse))
        self.w_shape = float(cfg.get("w_shape", w_shape))
        self.w_scale = float(cfg.get("w_scale", w_scale))

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        model: nn.Module | None = None,
        max_val: torch.Tensor | None = None,
        region: str | None = None,
    ) -> torch.Tensor:
        """Compute the normalized loss.

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

        total_loss = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        # 1. Normalized MSE loss component
        if self.w_mse > 0.0:
            mse_loss = torch.mean((norm_pred - norm_true) ** 2)
            total_loss = total_loss + self.w_mse * mse_loss

        # 2. Scale-invariant Cosine Shape distance component (1 - cos(y_pred, y_true))
        if self.w_shape > 0.0:
            pred_norm = torch.norm(norm_pred, p=2, dim=-1, keepdim=True).clamp(min=self.eps)
            true_norm = torch.norm(norm_true, p=2, dim=-1, keepdim=True).clamp(min=self.eps)
            cosine_sim = torch.sum((norm_pred / pred_norm) * (norm_true / true_norm), dim=-1)
            shape_loss = torch.mean(1.0 - cosine_sim)
            total_loss = total_loss + self.w_shape * shape_loss

        # 3. Peak amplitude scale discrepancy component
        if self.w_scale > 0.0:
            p_max, _ = torch.max(torch.abs(y_pred), dim=-1, keepdim=True)
            scale_ratio = p_max / max_val
            scale_loss = torch.mean((scale_ratio - 1.0) ** 2)
            total_loss = total_loss + self.w_scale * scale_loss

        # Apply regional weighting if specified
        if region is not None and region in self.region_weights:
            weight = float(self.region_weights[region])
            total_loss = total_loss * weight

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
            return total_loss + self.l2_weight * l2_reg

        return total_loss


class CompositeSpectralLoss(NormalizedMSELoss):
    """Composite spectral loss combining scale-invariant cosine shape loss and normalized MSE.

    Parameters
    ----------
    w_shape : float, optional
        Weight for scale-invariant cosine shape distance (default 0.5).
    w_mse : float, optional
        Weight for normalized MSE loss component (default 0.5).
    w_scale : float, optional
        Weight for relative scale error (default 0.0).
    l2_weight : float, optional
        Weight for model parameter L2 regularization (default 0.0).
    eps : float, optional
        Small positive floor to prevent division by zero (default 1e-4).
    region_weights : dict[str, float] | None, optional
        Optional regional weighting mapping region names to error multipliers (default None).
    config : dict[str, Any] | None, optional
        Configuration dictionary packaging loss options.
    """

    def __init__(
        self,
        w_shape: float = 0.5,
        w_mse: float = 0.5,
        w_scale: float = 0.0,
        l2_weight: float = 0.0,
        eps: float = 1e-4,
        region_weights: dict[str, float] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        cfg = dict(config or {})
        cfg.setdefault("w_shape", w_shape)
        cfg.setdefault("w_mse", w_mse)
        cfg.setdefault("w_scale", w_scale)
        cfg.setdefault("l2_weight", l2_weight)
        cfg.setdefault("eps", eps)
        if region_weights is not None:
            cfg.setdefault("region_weights", region_weights)
        super().__init__(config=cfg)
