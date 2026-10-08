"""1D ResNet neural network architecture for sequence-to-sequence spectral transformation.

Transforms a regional spectrum measured on a source tool into the corresponding
spectrum as if measured on a target tool, fulfilling Part 3 of GEMINI.md.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.film import FiLMGenerator


class ResNet1D(nn.Module):
    """Residual 1D CNN with 3 convolutional layers, mirror padding, identity shortcut, and optional FiLM.

    Transforms a 1D regional spectrum sequence from source tool to target tool
    while preserving sequence length: y = x + Delta(x, cond).

    The identity shortcut preserves the physical spectral baseline, while the 3 CNN
    layers learn the tool-to-tool spectral shift and intensity deformation Delta(x).
    Reflection (mirror) padding prevents boundary attenuation artifacts at spectral edges.

    Parameters
    ----------
    n_points : int
        Number of spectral data points (length of input regional spectrum or patch window).
    config : dict[str, Any] | None, optional
        Configuration dictionary packaging architectural hyperparameters:
        - 'hidden_channels' (int): Feature channels in hidden conv layers (default 32).
        - 'kernel_size' (int): Convolution kernel size, must be positive odd int (default 5).
        - 'dropout' (float): Dropout probability applied between conv layers (default 0.0).
        - 'use_batch_norm' (bool): Whether to include BatchNorm1d in hidden layers (default True).
        - 'padding_mode' (str): Padding mode 'reflect' (mirror) or 'zeros' (default 'reflect').
        - 'use_film' (bool): Whether to enable Feature-wise Linear Modulation conditioning (default False).
    """

    def __init__(self, n_points: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}

        # Parse configuration parameters with project defaults
        hidden_channels = int(cfg.get("hidden_channels", 32))
        kernel_size = int(cfg.get("kernel_size", 5))
        dropout = float(cfg.get("dropout", 0.0))
        use_batch_norm = bool(cfg.get("use_batch_norm", True))
        padding_mode = str(cfg.get("padding_mode", "reflect"))
        self.use_film = bool(cfg.get("use_film", False))
        self.n_points = n_points

        # Validate that kernel size is positive and odd to ensure symmetric padding
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        pad = kernel_size // 2

        # Factory helper to construct conv layer with optional reflection padding
        def _make_conv(in_channels: int, out_channels: int) -> nn.Module:
            if pad > 0 and padding_mode == "reflect" and n_points > pad:
                # Use ReflectionPad1d to avoid boundary attenuation artifacts
                return nn.Sequential(
                    nn.ReflectionPad1d(pad),
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size),
                )
            return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad)

        # 3-layer CNN backbone:
        # Layer 1: expands 1 spectral channel to hidden_channels
        # Layer 2: refines features within hidden_channels
        # Layer 3: projects hidden_channels back to 1 channel (spectral residual Delta)
        self.conv1 = _make_conv(1, hidden_channels)
        self.conv2 = _make_conv(hidden_channels, hidden_channels)
        self.conv3 = _make_conv(hidden_channels, 1)

        # Batch normalization layers for stabilizing hidden representations
        self.bn1 = nn.BatchNorm1d(hidden_channels) if use_batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(hidden_channels) if use_batch_norm else nn.Identity()

        # Non-linear activation and regularization
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Optional FiLM generator for metadata-conditioned modulation (die, tool, drift)
        self.film_gen = (
            FiLMGenerator([hidden_channels, hidden_channels], config=cfg)
            if self.use_film
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass applying residual transformation: y = x + Delta(x, cond).

        Parameters
        ----------
        x : torch.Tensor
            Input spectral tensor of shape (batch_size, n_points) or (batch_size, 1, n_points).
        cond : dict[str, torch.Tensor] | torch.Tensor | None, optional
            Optional metadata conditioning features for FiLM modulation.

        Returns
        -------
        torch.Tensor
            Transformed spectrum matching original input dimensionality.
        """
        orig_dim = x.dim()
        if orig_dim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")
        # Standardize 2D input (B, L) to 3D channel format (B, 1, L)
        x_in = x.unsqueeze(1) if orig_dim == 2 else x

        # Compute FiLM modulation parameters (gamma, beta) if conditioning is enabled
        mods = self.film_gen(cond) if (self.use_film and cond is not None) else None

        # Hidden representation extraction across layers 1 and 2
        h = x_in
        for i, (conv, bn) in enumerate(((self.conv1, self.bn1), (self.conv2, self.bn2))):
            h = bn(conv(h))
            if mods is not None:
                # Apply FiLM affine transformation: gamma * h + beta
                gamma, beta = mods[i]
                h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)
            h = self.drop(self.act(h))

        # Project to residual difference Delta(x) and add physical shortcut: y = x + Delta(x)
        delta = self.conv3(h)
        out = x_in + delta

        # Restore original input dimensionality
        return out.squeeze(1) if orig_dim == 2 else out

    def get_l2_regularization(self) -> torch.Tensor:
        """Compute the sum of squared weights (L2 regularization penalty) for conv layers.

        Biases and BatchNorm parameters are intentionally excluded to avoid penalizing
        baseline intensity shifts and scaling factors.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing sum of squared convolutional weights.
        """
        device = next(self.parameters()).device
        weights = [
            param
            for name, param in self.named_parameters()
            if param.requires_grad and name.endswith(".weight") and "bn" not in name
        ]
        if not weights:
            return torch.tensor(0.0, device=device)
        return sum((param ** 2).sum() for param in weights)
