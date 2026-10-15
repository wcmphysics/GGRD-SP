"""Shared neural network layers and padding utilities for spectral models."""

from __future__ import annotations

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


# Backward-compatible alias
_SafeReflectionPad1d = SafeReflectionPad1d


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


# Backward-compatible alias
_SafeBatchNorm1d = SafeBatchNorm1d
