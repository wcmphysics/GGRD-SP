"""1D DeepLabV3 neural network architecture for spectral transformation.

Adapts DeepLabv3 principles to 1D sequence-to-sequence spectral
modeling:
1. Multi-Grid 1D ResNet backbone with customizable dilation rates.
2. 1D Atrous Spatial Pyramid Pooling (ASPP) with multiple dilations.
3. Global average pooling branch and batch normalization in ASPP
   to prevent filter degeneracy on boundary spectral features.
4. Sequence length preservation with reflection padding and stride 1.
5. Global residual shortcut y = x + Delta(x) for stable inter-tool
   shift learning.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SafeReflectionPad1d(nn.Module):
    """Dynamically applies reflection padding when length L > pad.

    Falls back gracefully to zero padding when L <= pad to prevent
    PyTorch assertion crashes.
    """

    def __init__(self, pad: int) -> None:
        super().__init__()
        self.pad = int(pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] > self.pad:
            return F.pad(x, (self.pad, self.pad), mode="reflect")
        return F.pad(x, (self.pad, self.pad), mode="constant", value=0.0)


class _SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that safely handles single-element tensors.

    When batch_size * length <= 1 during training (such as following
    global average pooling when batch_size == 1 or on the trailing batch
    of a dataset), PyTorch's standard BatchNorm raises ValueError
    because sample variance is undefined. This class falls back to
    running statistics (eval mode) without throwing an exception.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and (x.size(0) * x.size(-1) <= 1) and self.running_mean is not None:
            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                training=False, momentum=self.momentum, eps=self.eps,
            )
        return super().forward(x)


def _build_conv1d_layer(
    in_channels: int, out_channels: int, kernel_size: int,
    dilation: int = 1, padding_mode: str = "reflect", n_points: int = 100,
) -> nn.Module:
    """Build a 1D conv layer with safe symmetric padding."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    if dilation < 1:
        raise ValueError(f"dilation must be a positive integer, got {dilation}")
    pad = (kernel_size - 1) * dilation // 2
    pad_val = 0 if padding_mode == "reflect" else pad
    conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=pad_val)
    return nn.Sequential(_SafeReflectionPad1d(pad), conv) if (pad > 0 and padding_mode == "reflect") else conv


def _conv_block(
    in_channels: int, out_channels: int, kernel_size: int = 1,
    dilation: int = 1, padding_mode: str = "reflect", n_points: int = 100,
    use_batch_norm: bool = True, dropout: float = 0.0, bias: bool = True,
) -> nn.Sequential:
    """Build a Conv-BN-ReLU sequential block with optional dropout."""
    conv = (
        nn.Conv1d(in_channels, out_channels, 1, bias=bias) if kernel_size == 1
        else _build_conv1d_layer(in_channels, out_channels, kernel_size, dilation, padding_mode, n_points)
    )
    bn = _SafeBatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
    drop = [nn.Dropout(dropout)] if dropout > 0.0 else []
    return nn.Sequential(conv, bn, nn.ReLU(inplace=True), *drop)


class ResidualBlock1D(nn.Module):
    """1D Residual block with dilated convolutions and shortcut."""

    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: int = 1,
        use_batch_norm: bool = True, dropout: float = 0.0,
        padding_mode: str = "reflect", n_points: int = 100,
    ) -> None:
        super().__init__()
        conv_fn = lambda: _build_conv1d_layer(channels, channels, kernel_size, dilation, padding_mode, n_points)
        bn_fn = lambda: _SafeBatchNorm1d(channels) if use_batch_norm else nn.Identity()
        self.conv1, self.bn1 = conv_fn(), bn_fn()
        self.conv2, self.bn2 = conv_fn(), bn_fn()
        self.relu1, self.relu2 = nn.ReLU(inplace=True), nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu1(self.bn1(self.conv1(x))))
        return self.relu2(self.bn2(self.conv2(out)) + x)


class ResNet1DBackbone(nn.Module):
    """ResNet 1D backbone supporting Multi-Grid dilation rates.

    Dynamically adapts the number of residual blocks to the length of
    the user-specified `multi_grid` tuple.
    """

    def __init__(
        self, in_channels: int = 1, out_channels: int = 32,
        multi_grid: Sequence[int] = (1, 2, 4), kernel_size: int = 3,
        use_batch_norm: bool = True, dropout: float = 0.0,
        padding_mode: str = "reflect", n_points: int = 100,
    ) -> None:
        super().__init__()
        if not multi_grid:
            raise ValueError("multi_grid sequence must not be empty.")
        self.stem = _conv_block(in_channels, out_channels, kernel_size, 1, padding_mode, n_points, use_batch_norm)
        self.blocks = nn.ModuleList([
            ResidualBlock1D(out_channels, kernel_size, int(r), use_batch_norm, dropout, padding_mode, n_points)
            for r in multi_grid
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functools.reduce(lambda feat, block: block(feat), self.blocks, self.stem(x))


class ASPP1D(nn.Module):
    """1D Atrous Spatial Pyramid Pooling (ASPP) module.

    Branches:
    1. 1x1 Conv1d + BatchNorm1d + ReLU (pointwise feature map)
    2. Atrous Conv1d branch 1 (rate = aspp_rates[0]) + BN + ReLU
    3. Atrous Conv1d branch 2 (rate = aspp_rates[1]) + BN + ReLU
    4. Atrous Conv1d branch 3 (rate = aspp_rates[2]) + BN + ReLU
    5. 1D Global Avg Pooling + 1x1 Conv + BN + ReLU
    All branches are concatenated and projected to `out_channels`.
    """

    def __init__(
        self, in_channels: int = 32, out_channels: int = 32,
        aspp_rates: Sequence[int] = (2, 4, 6), kernel_size: int = 3,
        use_batch_norm: bool = True, dropout: float = 0.1,
        padding_mode: str = "reflect", n_points: int = 100,
    ) -> None:
        super().__init__()
        if len(aspp_rates) != 3:
            raise ValueError(f"aspp_rates must contain exactly 3 dilation rates, got {aspp_rates}")

        self.branch_1x1 = _conv_block(in_channels, out_channels, use_batch_norm=use_batch_norm, bias=False)
        self.atrous_branches = nn.ModuleList([
            _conv_block(in_channels, out_channels, kernel_size, int(r), padding_mode, n_points, use_batch_norm)
            for r in aspp_rates
        ])
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), *_conv_block(in_channels, out_channels, use_batch_norm=use_batch_norm, bias=False)
        )
        total_in = out_channels * (2 + len(aspp_rates))
        self.project = _conv_block(total_in, out_channels, use_batch_norm=use_batch_norm, dropout=dropout, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_1x1 = self.branch_1x1(x)
        atrous_feats = [branch(x) for branch in self.atrous_branches]
        glob_feat = self.global_pooling(x).expand(-1, -1, x.shape[-1])
        return self.project(torch.cat([feat_1x1, *atrous_feats, glob_feat], dim=1))


class DeepLabV3(nn.Module):
    """1D DeepLabV3 architecture for spectral transformation.

    Features:
    - Multi-Grid ResNet 1D backbone with configurable dilation rates.
    - 1D ASPP module capturing local, intermediate, and wide contexts.
    - Global avg pooling branch preventing degenerate edge filters.
    - Reflection padding and stride 1 maintaining exact resolution.
    - Global residual shortcut y = x + Delta(x) for stable shift.
    - Pure end-to-end differentiable neural net without DenseCRF.
    - Explicit L2 penalty calculation on convolutional weights.

    Parameters
    ----------
    n_points : int
        Number of spectral data points (sequence or patch length).
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'backbone_channels' (int): Hidden channels in ResNet (def 32).
        - 'aspp_channels' (int): Feature channels per ASPP (def 32).
        - 'multi_grid' (tuple[int, ...]): Multi-grid rates
          (default (1, 2, 4)).
        - 'aspp_rates' (tuple[int, int, int]): ASPP rates
          (default (2, 4, 6)).
        - 'kernel_size' (int): Conv kernel size, odd int (default 3).
        - 'dropout' (float): Dropout probability after ASPP (def 0.1).
        - 'use_batch_norm' (bool): Use BatchNorm1d (default True).
        - 'residual' (bool): Apply global residual shortcut (def True).
        - 'padding_mode' (str): Padding mode 'reflect' (default).
    """

    def __init__(self, n_points: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        self.n_points = int(n_points)
        backbone_channels, aspp_channels = int(cfg.get("backbone_channels", 32)), int(cfg.get("aspp_channels", 32))
        multi_grid, aspp_rates = tuple(cfg.get("multi_grid", (1, 2, 4))), tuple(cfg.get("aspp_rates", (2, 4, 6)))
        kernel_size, dropout = int(cfg.get("kernel_size", 3)), float(cfg.get("dropout", 0.1))
        use_batch_norm, padding_mode = bool(cfg.get("use_batch_norm", True)), str(cfg.get("padding_mode", "reflect"))
        self.residual = bool(cfg.get("residual", True))

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        kw = dict(kernel_size=kernel_size, use_batch_norm=use_batch_norm, dropout=dropout, padding_mode=padding_mode)
        self.backbone = ResNet1DBackbone(1, backbone_channels, multi_grid, n_points=self.n_points, **kw)
        self.aspp = ASPP1D(backbone_channels, aspp_channels, aspp_rates, n_points=self.n_points, **kw)
        self.head = nn.Conv1d(aspp_channels, 1, kernel_size=1)

    def forward(
        self, x: torch.Tensor,
        cond: dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transform input spectrum x to target spectrum y.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, n_points).
        cond : dict[str, torch.Tensor] | torch.Tensor | None, optional
            Optional metadata conditioning (unused by default backbone).

        Returns
        -------
        torch.Tensor
            Transformed spectrum of shape (batch_size, 1, n_points).

        Raises
        ------
        ValueError
            If x is not a 3D tensor of shape (batch_size, 1, n_points).
        """
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(
                f"Expected 3D tensor of shape (batch, 1, seq_len), got shape {tuple(x.shape)}"
            )

        delta = self.head(self.aspp(self.backbone(x)))
        return (x + delta) if self.residual else delta

    def get_l2_regularization(self) -> torch.Tensor:
        """Compute sum of squared conv weights for L2 loss penalty.

        Excludes BatchNorm parameters and biases to preserve offsets.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing sum of squares of conv weights.
        """
        device = next(self.parameters()).device
        return sum(
            (torch.sum(m.weight ** 2) for m in self.modules() if isinstance(m, nn.Conv1d) and m.weight.requires_grad),
            torch.tensor(0.0, device=device),
        )
