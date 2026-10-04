"""1D DeepLabV3 neural network architecture for spectral transformation.

Adapts DeepLabv3 principles to 1D sequence-to-sequence spectral modeling:
1. Multi-Grid 1D ResNet backbone with customizable dilation rates.
2. 1D Atrous Spatial Pyramid Pooling (ASPP) with multiple dilation rates.
3. Global average pooling branch and batch normalization in ASPP to prevent
   filter degeneracy on boundary spectral features.
4. Sequence length preservation with reflection padding and stride 1.
5. Global residual shortcut y = x + Delta(x) for stable inter-tool shift learning.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SafeReflectionPad1d(nn.Module):
    """Dynamically applies reflection padding when sequence length L > pad.

    Falls back gracefully to zero padding when L <= pad to prevent PyTorch assertion crashes.
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

    When batch_size * length <= 1 during training (such as following global average
    pooling when batch_size == 1 or on the trailing batch of a dataset), PyTorch's
    standard BatchNorm raises ValueError because sample variance is undefined.
    This class falls back to running statistics (eval mode) without throwing an exception.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and (x.size(0) * x.size(-1) <= 1):
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


def _build_conv1d_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dilation: int = 1,
    padding_mode: str = "reflect",
    n_points: int = 100,
) -> nn.Module:
    """Build a 1D conv layer with safe symmetric padding."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}"
        )
    if dilation < 1:
        raise ValueError(f"dilation must be a positive integer, got {dilation}")

    pad = (kernel_size - 1) * dilation // 2

    if pad == 0:
        return nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    if padding_mode == "reflect":
        return nn.Sequential(
            _SafeReflectionPad1d(pad),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=0,
            ),
        )
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=pad,
    )


class ResidualBlock1D(nn.Module):
    """1D Residual block with dilated convolutions and identity shortcut."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        padding_mode: str = "reflect",
        n_points: int = 100,
    ) -> None:
        super().__init__()
        self.conv1 = _build_conv1d_layer(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding_mode=padding_mode,
            n_points=n_points,
        )
        self.bn1 = _SafeBatchNorm1d(channels) if use_batch_norm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.conv2 = _build_conv1d_layer(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding_mode=padding_mode,
            n_points=n_points,
        )
        self.bn2 = _SafeBatchNorm1d(channels) if use_batch_norm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out + residual)
        return out


class ResNet1DBackbone(nn.Module):
    """ResNet 1D backbone supporting Multi-Grid dilation rates.

    Dynamically adapts the number of residual blocks to the length of the
    user-specified `multi_grid` tuple.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        multi_grid: Sequence[int] = (1, 2, 4),
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        padding_mode: str = "reflect",
        n_points: int = 100,
    ) -> None:
        super().__init__()
        if not multi_grid:
            raise ValueError("multi_grid sequence must not be empty.")

        self.stem = nn.Sequential(
            _build_conv1d_layer(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=1,
                padding_mode=padding_mode,
                n_points=n_points,
            ),
            _SafeBatchNorm1d(out_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        for rate in multi_grid:
            blocks.append(
                ResidualBlock1D(
                    channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=int(rate),
                    use_batch_norm=use_batch_norm,
                    dropout=dropout,
                    padding_mode=padding_mode,
                    n_points=n_points,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        for block in self.blocks:
            out = block(out)
        return out


class ASPP1D(nn.Module):
    """1D Atrous Spatial Pyramid Pooling (ASPP) module.

    Branches:
    1. 1x1 Conv1d + BatchNorm1d + ReLU (pointwise feature transformation)
    2. Atrous Conv1d branch 1 (rate = aspp_rates[0]) + BN + ReLU
    3. Atrous Conv1d branch 2 (rate = aspp_rates[1]) + BN + ReLU
    4. Atrous Conv1d branch 3 (rate = aspp_rates[2]) + BN + ReLU
    5. 1D Global Average Pooling + 1x1 Conv + BN + ReLU (avoids degenerate filters)
    All 5 branches are concatenated and projected down to `out_channels`.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        aspp_rates: Sequence[int] = (2, 4, 6),
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
        padding_mode: str = "reflect",
        n_points: int = 100,
    ) -> None:
        super().__init__()
        if len(aspp_rates) != 3:
            raise ValueError(
                f"aspp_rates must contain exactly 3 dilation rates, got {aspp_rates}"
            )

        # Branch 1: 1x1 convolution
        self.branch_1x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            _SafeBatchNorm1d(out_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # Branches 2, 3, 4: Atrous convolutions
        self.atrous_branches = nn.ModuleList()
        for rate in aspp_rates:
            branch = nn.Sequential(
                _build_conv1d_layer(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=int(rate),
                    padding_mode=padding_mode,
                    n_points=n_points,
                ),
                _SafeBatchNorm1d(out_channels) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            )
            self.atrous_branches.append(branch)

        # Branch 5: 1D Global Average Pooling (safeguard against degenerate filter)
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            _SafeBatchNorm1d(out_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # Projection of all 5 concatenated branches
        total_in_channels = out_channels * (2 + len(aspp_rates))
        self.project = nn.Sequential(
            nn.Conv1d(total_in_channels, out_channels, kernel_size=1, bias=False),
            _SafeBatchNorm1d(out_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-1]
        feat_1x1 = self.branch_1x1(x)

        atrous_feats = [branch(x) for branch in self.atrous_branches]

        glob_feat = self.global_pooling(x)
        glob_feat = glob_feat.expand(-1, -1, seq_len)

        concat_feats = torch.cat([feat_1x1, *atrous_feats, glob_feat], dim=1)
        return self.project(concat_feats)


class DeepLabV3(nn.Module):
    """1D DeepLabV3 architecture for sequence-to-sequence spectral transformation.

    Features:
    - Multi-Grid ResNet 1D backbone with configurable dilation rates.
    - 1D ASPP module capturing local, intermediate, and wide spectral contexts.
    - Global average pooling branch preventing degenerate filters at sequence edges.
    - Reflection padding and stride 1 maintaining exact sequence resolution.
    - Global residual shortcut y = x + Delta(x) for stable inter-tool shift learning.
    - Pure end-to-end differentiable neural network without DenseCRF post-processing.
    - Explicit L2 penalty calculation on convolutional weights for NormalizedMSELoss.

    Parameters
    ----------
    n_points : int
        Number of spectral data points (length of sequence or patch window).
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'backbone_channels' (int): Hidden channels in ResNet backbone (default 32).
        - 'aspp_channels' (int): Feature channels per ASPP branch (default 32).
        - 'multi_grid' (tuple[int, ...]): Multi-grid dilation rates (default (1, 2, 4)).
        - 'aspp_rates' (tuple[int, int, int]): Dilations for ASPP branches (default (2, 4, 6)).
        - 'kernel_size' (int): Conv kernel size, positive odd int (default 3).
        - 'dropout' (float): Dropout probability after ASPP fusion (default 0.1).
        - 'use_batch_norm' (bool): Whether to use BatchNorm1d (default True).
        - 'residual' (bool): Whether to apply global shortcut y = x + Delta(x) (default True).
        - 'padding_mode' (str): Padding mode 'reflect' or 'zeros' (default 'reflect').
    """

    def __init__(self, n_points: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        self.n_points = int(n_points)
        backbone_channels = int(cfg.get("backbone_channels", 32))
        aspp_channels = int(cfg.get("aspp_channels", 32))
        multi_grid = tuple(cfg.get("multi_grid", (1, 2, 4)))
        aspp_rates = tuple(cfg.get("aspp_rates", (2, 4, 6)))
        kernel_size = int(cfg.get("kernel_size", 3))
        dropout = float(cfg.get("dropout", 0.1))
        use_batch_norm = bool(cfg.get("use_batch_norm", True))
        self.residual = bool(cfg.get("residual", True))
        padding_mode = str(cfg.get("padding_mode", "reflect"))

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )

        # 1. ResNet Backbone with Multi-Grid
        self.backbone = ResNet1DBackbone(
            in_channels=1,
            out_channels=backbone_channels,
            multi_grid=multi_grid,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            padding_mode=padding_mode,
            n_points=self.n_points,
        )

        # 2. Atrous Spatial Pyramid Pooling (ASPP)
        self.aspp = ASPP1D(
            in_channels=backbone_channels,
            out_channels=aspp_channels,
            aspp_rates=aspp_rates,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            padding_mode=padding_mode,
            n_points=self.n_points,
        )

        # 3. Final Prediction Head (maps feature channels to 1 intensity channel)
        self.head = nn.Conv1d(aspp_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass transforming input spectrum x to target spectrum y.

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
            if x.shape[1] != 1:
                raise ValueError(
                    f"Expected 3D tensor with 1 channel (batch_size, 1, n_points), got shape {x.shape}"
                )
            x_in = x
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        features = self.backbone(x_in)
        aspp_features = self.aspp(features)
        delta = self.head(aspp_features)

        out = (x_in + delta) if self.residual else delta

        if orig_dim == 2:
            return out.squeeze(1)
        return out

    def get_l2_regularization(self) -> torch.Tensor:
        """Compute sum of squared convolutional weights for L2 loss penalty.

        Excludes BatchNorm parameters and biases to preserve baseline intensity offsets.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing sum of squares of conv weight parameters.
        """
        device = next(self.parameters()).device
        l2_sum = torch.tensor(0.0, device=device)
        for module in self.modules():
            if isinstance(module, nn.Conv1d) and module.weight.requires_grad:
                l2_sum = l2_sum + torch.sum(module.weight ** 2)
        return l2_sum


# Alias for naming consistency
DeepLabV3_1D = DeepLabV3
