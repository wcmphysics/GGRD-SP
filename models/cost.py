"""Cost functions for spectral sequence-to-sequence transformation."""

from __future__ import annotations

import math
from typing import Any

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


class SpectralLoss(nn.Module):
    """Spectral loss combining MSE, shape, gradient, and L2 regularization.

    Operates strictly on 3D tensors of standard shape (batch, 1, seq_len).
    Optionally normalizes true and predicted spectra by the maximum intensity
    of the true spectrum, then computes a weighted combination:
        loss = w_mse * L_mse + w_shape * L_shape + w_grad * L_grad + l2 * L2

    Parameters
    ----------
    config : dict[str, Any] | None, optional
        Optional configuration dictionary packaging loss weights.
    w_mse : float, optional
        Weight for mean-squared error component (default 1.0).
    w_shape : float, optional
        Weight for scale-invariant cosine shape distance (default 0.0).
    w_gradient : float, optional
        Weight for intensity gradient MSE component (default 0.0).
    l2_weight : float, optional
        Weight for model parameter L2 regularization (default 0.0).
    eps : float, optional
        Small positive floor to prevent division by zero (default 1e-4).
    region_weights : dict[str, float] | None, optional
        Optional regional weighting mapping region names to error multipliers
        (default None).
    normalized_by_target_max : bool, optional
        Whether to divide intensities by the maximum intensity of the true
        spectrum (default True).
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        w_mse: float = 1.0,
        w_shape: float = 0.0,
        w_gradient: float = 0.0,
        l2_weight: float = 0.0,
        eps: float = 1e-4,
        region_weights: dict[str, float] | None = None,
        normalized_by_target_max: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        cfg = dict(config or {})
        cfg.update(kwargs)
        self.w_mse = float(cfg.get("w_mse", w_mse))
        self.w_shape = float(cfg.get("w_shape", w_shape))
        self.w_gradient = float(cfg.get("w_gradient", w_gradient))
        self.l2_weight = float(cfg.get("l2_weight", l2_weight))
        self.eps = float(cfg.get("eps", eps))
        self.region_weights = dict(cfg.get("region_weights") or region_weights or {})
        self.normalized_by_target_max = bool(
            cfg.get("normalized_by_target_max", normalized_by_target_max)
        )

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        model: nn.Module | None = None,
        max_val: torch.Tensor | None = None,
        region: str | None = None,
    ) -> torch.Tensor:
        """Compute the spectral loss between predicted and ground truth spectra.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted intensity tensor of shape (batch_size, 1, seq_len).
        y_true : torch.Tensor
            Target intensity tensor of shape (batch_size, 1, seq_len).
        model : nn.Module | None, optional
            The model instance, needed to compute L2 regularization if
            l2_weight > 0.
        max_val : torch.Tensor | None, optional
            Precomputed normalization maximum tensor. If None, computes
            max(|y_true|) per sample.
        region : str | None, optional
            Name of the spectral region (used for regional weighting if
            configured).

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.

        Raises
        ------
        ValueError
            If y_pred or y_true does not match shape (batch_size, 1, seq_len).
        """
        if y_pred.dim() != 3 or y_pred.size(1) != 1:
            raise ValueError(
                f"y_pred must have shape (batch, 1, seq_len), got {tuple(y_pred.shape)}"
            )
        if y_true.dim() != 3 or y_true.size(1) != 1:
            raise ValueError(
                f"y_true must have shape (batch, 1, seq_len), got {tuple(y_true.shape)}"
            )

        if self.normalized_by_target_max:
            if max_val is None:
                max_val = torch.amax(torch.abs(y_true), dim=-1, keepdim=True)
            else:
                max_val = max_val.to(device=y_pred.device, dtype=y_pred.dtype)
                while max_val.dim() < y_pred.dim():
                    max_val = max_val.unsqueeze(-1)
            max_val = torch.clamp(max_val, min=self.eps)
            norm_pred = y_pred / max_val
            norm_true = y_true / max_val
        else:
            norm_pred = y_pred
            norm_true = y_true

        total_loss = 0.0 * torch.sum(y_pred)

        # 1. Mean-squared error (MSE) component
        if self.w_mse > 0.0:
            mse_loss = torch.mean((norm_pred - norm_true) ** 2)
            total_loss = total_loss + self.w_mse * mse_loss

        # 2. Scale-invariant Cosine Shape distance component
        if self.w_shape > 0.0:
            pred_norm = torch.linalg.vector_norm(
                norm_pred, dim=-1, keepdim=True
            ).clamp(min=self.eps)
            true_norm = torch.linalg.vector_norm(
                norm_true, dim=-1, keepdim=True
            ).clamp(min=self.eps)
            cosine_sim = torch.sum(
                (norm_pred / pred_norm) * (norm_true / true_norm), dim=-1
            )
            cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
            shape_loss = torch.mean(1.0 - cosine_sim)
            total_loss = total_loss + self.w_shape * shape_loss

        # 3. Intensity gradient MSE component (central difference)
        if self.w_gradient > 0.0:
            grad_pred = torch.gradient(norm_pred, dim=-1)[0]
            grad_true = torch.gradient(norm_true, dim=-1)[0]
            grad_loss = torch.mean((grad_pred - grad_true) ** 2)
            total_loss = total_loss + self.w_gradient * grad_loss

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
