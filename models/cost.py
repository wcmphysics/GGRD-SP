"""Cost functions for spectral sequence-to-sequence transformation."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from models.layers import validate_3d_spectrum_tensor


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
    return f"{loss_val:.6f} (log10: {math.log10(loss_val):+.4f})"


class SpectralLoss(nn.Module):
    """Spectral loss combining MSE, shape, gradient, and L2 regularization.

    Operates strictly on 3D tensors of standard shape (batch, 1, seq_len).
    Optionally normalizes true and predicted spectra by the maximum intensity
    of the true spectrum, then computes a weighted combination:
        loss = w_mse * L_mse + w_shape * L_shape + w_grad * L_grad + l2 * L2

    Parameters
    ----------
    config : dict[str, Any] | None, optional
        Optional configuration dictionary packaging loss weights:
        - 'w_mse' (float): Weight for MSE component (default 1.0).
        - 'w_shape' (float): Weight for cosine shape distance (default 0.0).
        - 'w_gradient' (float): Weight for intensity gradient MSE (default 0.0).
        - 'l2_weight' (float): Weight for parameter L2 penalty (default 0.0).
        - 'eps' (float): Small positive floor (default 1e-4).
        - 'region_weights' (dict[str, float] | None): Regional weight factors.
        - 'normalized_by_target_max' (bool): Divide by max(|y_true|) (True).
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg = config or {}
        self.w_mse = float(cfg.get("w_mse", 1.0))
        self.w_shape = float(cfg.get("w_shape", 0.0))
        self.w_gradient = float(cfg.get("w_gradient", 0.0))
        self.l2_weight = float(cfg.get("l2_weight", 0.0))
        self.eps = float(cfg.get("eps", 1e-4))
        self.region_weights = dict(cfg.get("region_weights") or {})
        self.normalized_by_target_max = bool(
            cfg.get("normalized_by_target_max", True)
        )

    def _normalize_pair(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        max_val: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize predicted and true spectra by target maximum if enabled."""
        if not self.normalized_by_target_max:
            return y_pred, y_true

        if max_val is None:
            scale = torch.amax(torch.abs(y_true), dim=-1, keepdim=True)
        else:
            scale = max_val.to(device=y_pred.device, dtype=y_pred.dtype)
            while scale.dim() < y_pred.dim():
                scale = scale.unsqueeze(-1)
        scale = torch.clamp(scale, min=self.eps)
        return y_pred / scale, y_true / scale

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        model: nn.Module | None = None,
        loss_context: dict[str, Any] | torch.Tensor | None = None,
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
        loss_context : dict[str, Any] | torch.Tensor | None, optional
            Optional context dictionary with 'max_val' (precomputed scale)
            and/or 'region' (str), or a direct max_val tensor.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.

        Raises
        ------
        ValueError
            If y_pred or y_true does not match shape (batch_size, 1, seq_len).
        """
        validate_3d_spectrum_tensor(y_pred, "y_pred")
        validate_3d_spectrum_tensor(y_true, "y_true")

        if isinstance(loss_context, dict):
            max_val = loss_context.get("max_val")
            region = loss_context.get("region")
        else:
            max_val = loss_context
            region = None

        norm_pred, norm_true = self._normalize_pair(y_pred, y_true, max_val)
        total_loss = 0.0 * torch.sum(y_pred)

        # 1. Mean-squared error (MSE) component
        if self.w_mse > 0.0:
            total_loss = total_loss + self.w_mse * torch.mean(
                (norm_pred - norm_true) ** 2
            )

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
            ).clamp(-1.0, 1.0)
            total_loss = total_loss + self.w_shape * torch.mean(1.0 - cosine_sim)

        # 3. Intensity gradient MSE component (central difference)
        if self.w_gradient > 0.0:
            grad_pred = torch.gradient(norm_pred, dim=-1)[0]
            grad_true = torch.gradient(norm_true, dim=-1)[0]
            total_loss = total_loss + self.w_gradient * torch.mean(
                (grad_pred - grad_true) ** 2
            )

        # Apply regional weighting if specified
        if region is not None and region in self.region_weights:
            total_loss = total_loss * float(self.region_weights[region])

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
