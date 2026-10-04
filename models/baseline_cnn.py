"""Baseline 1D Residual CNN architecture and custom loss for spectral transformation.

Backward compatibility module re-exporting ResNet1D / Residual1DCNN and NormalizedMSELoss.
"""

from __future__ import annotations

from models.cost import NormalizedMSELoss
from models.resnet import ResNet1D, Residual1DCNN

__all__ = [
    "Residual1DCNN",
    "ResNet1D",
    "NormalizedMSELoss",
]
