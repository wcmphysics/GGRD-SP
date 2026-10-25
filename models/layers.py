"""Shared neural network layers, padding, and validation utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeReflectionPad1d(nn.Module):
    """Dynamically applies reflection padding when sequence length L > pad.

    Falls back gracefully to constant zero padding when L <= pad to prevent
    PyTorch runtime assertion crashes on short sequences or large kernels.

    Parameters
    ----------
    pad : int
        Number of padding elements applied to both boundaries.
    """

    def __init__(self, pad: int) -> None:
        super().__init__()
        self.pad = int(pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply safe symmetric padding along the sequence dimension."""
        if x.shape[-1] > self.pad:
            return F.pad(x, (self.pad, self.pad), mode="reflect")
        return F.pad(x, (self.pad, self.pad), mode="constant", value=0.0)


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that safely handles single-element tensors.

    When batch_size * length <= 1 during training (such as following
    global average pooling when batch_size == 1 or on the trailing batch
    of a dataset), PyTorch's standard BatchNorm raises ValueError
    because sample variance is undefined. This class falls back to
    running statistics (eval mode) without throwing an exception.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch normalization with single-element fallback."""
        if (
            self.training
            and (x.size(0) * x.size(-1) <= 1)
            and self.running_mean is not None
        ):
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )
        return super().forward(x)


def validate_positive_int(val: Any, name: str, min_val: int = 1) -> int:
    """Validate that a value is an integer (not bool) >= min_val."""
    if isinstance(val, bool) or not isinstance(val, int) or val < min_val:
        raise ValueError(f"{name} must be a positive integer, got {val!r}")
    return val


def validate_odd_kernel_size(
    val: Any, max_allowed: int | None = None
) -> int:
    """Validate that kernel_size is a positive odd integer within bounds."""
    if isinstance(val, bool) or not isinstance(val, int) or val <= 0 or val % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {val!r}")
    if max_allowed is not None and val > max_allowed:
        raise ValueError(
            f"kernel_size ({val}) cannot be larger than window_size / n_points ({max_allowed})."
        )
    return val


def validate_common_model_config(
    cfg: dict[str, Any],
    n_points: int,
    disallowed_keys: Iterable[str],
    model_spec: tuple[str, str],
) -> int:
    """Validate disallowed plural/alias keys and return kernel size bound.

    Returns the maximum allowed kernel size bounded by n_points and any
    explicit window_size override.
    """
    model_name, canonical_hint = model_spec
    for invalid_key in disallowed_keys:
        if invalid_key in cfg:
            raise KeyError(
                f"Invalid key '{invalid_key}' in {model_name} config; {canonical_hint}"
            )

    explicit_ws = cfg.get("window_size_choice", cfg.get("window_size"))
    return min(n_points, explicit_ws) if explicit_ws is not None else n_points


def validate_3d_spectrum_tensor(
    x: torch.Tensor, name: str | None = None
) -> None:
    """Ensure input tensor has shape (batch_size, 1, seq_len)."""
    if x.dim() != 3 or x.size(1) != 1:
        if name is not None:
            raise ValueError(
                f"{name} must have shape (batch, 1, seq_len), got {tuple(x.shape)}"
            )
        raise ValueError(
            f"Expected 3D tensor of shape (batch, 1, seq_len), got shape {tuple(x.shape)}"
        )


def build_conv1d_layer(
    channels: tuple[int, int],
    kernel_size: int,
    dilation: int = 1,
    padding_mode: str = "reflect",
) -> nn.Module:
    """Build a 1D conv layer with safe symmetric padding."""
    in_channels, out_channels = channels
    validate_odd_kernel_size(kernel_size)
    if dilation < 1:
        raise ValueError(f"dilation must be a positive integer, got {dilation}")
    pad = (kernel_size - 1) * dilation // 2
    if pad > 0 and padding_mode == "reflect":
        return nn.Sequential(
            SafeReflectionPad1d(pad),
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation),
        )
    return nn.Conv1d(
        in_channels, out_channels, kernel_size, dilation=dilation, padding=pad
    )


def compute_conv1d_l2_reg(module: nn.Module) -> torch.Tensor:
    """Compute sum of squared Conv1d weights for L2 regularization."""
    device = next((p.device for p in module.parameters()), torch.device("cpu"))
    return sum(
        (
            torch.sum(m.weight**2)
            for m in module.modules()
            if isinstance(m, nn.Conv1d) and m.weight.requires_grad
        ),
        torch.tensor(0.0, device=device),
    )

