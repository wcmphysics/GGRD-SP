"""1D ResNet neural network architecture for sequence-to-sequence spectral transformation.

Transforms a regional spectrum measured on a source tool into the corresponding
spectrum as if measured on a target tool, fulfilling Part 3 of GEMINI.md.
"""

from __future__ import annotations

from typing import Any

import torch, torch.nn as nn

from models.film import FiLMGenerator


class ResNet1D(nn.Module):
    """Residual 1D CNN with 3 convolutional layers, mirror padding, identity shortcut, and optional FiLM.

    Transforms a 1D regional spectrum sequence from source tool to target tool
    while preserving sequence length: y = x + Delta(x, cond).

    The identity shortcut preserves the physical spectral baseline, while the 3 CNN
    layers learn the tool-to-tool spectral shift and deformation Delta(x).
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
        effective_config = config or {}

        # Parse configuration parameters with project defaults
        hidden_channels, kernel_size, dropout = int(effective_config.get("hidden_channels", 32)), int(effective_config.get("kernel_size", 5)), float(effective_config.get("dropout", 0.0))
        use_batch_norm, padding_mode, self.use_film, self.n_points = bool(effective_config.get("use_batch_norm", True)), str(effective_config.get("padding_mode", "reflect")), bool(effective_config.get("use_film", False)), n_points

        # Validate that kernel size is positive and odd to ensure symmetric padding
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        padding_size = kernel_size // 2

        # Factory helper to construct conv layer with optional reflection padding
        def _make_conv(in_channels: int, out_channels: int) -> nn.Module:
            return nn.Sequential(nn.ReflectionPad1d(padding_size), nn.Conv1d(in_channels, out_channels, kernel_size)) if (padding_size > 0 and padding_mode == "reflect" and n_points > padding_size) else nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding_size)

        # 3-layer CNN backbone: 1 -> hidden -> hidden -> 1 (spectral residual Delta)
        self.conv1, self.conv2, self.conv3 = _make_conv(1, hidden_channels), _make_conv(hidden_channels, hidden_channels), _make_conv(hidden_channels, 1)

        # Normalization, activation, and dropout layers
        self.bn1, self.bn2 = (nn.BatchNorm1d(hidden_channels) if use_batch_norm else nn.Identity()), (nn.BatchNorm1d(hidden_channels) if use_batch_norm else nn.Identity())
        self.activation, self.dropout = nn.ReLU(inplace=True), nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Optional FiLM generator for metadata-conditioned modulation (die, tool, drift)
        self.film_gen = FiLMGenerator([hidden_channels, hidden_channels], config=effective_config) if self.use_film else None

    def forward(self, x: torch.Tensor, cond: dict[str, torch.Tensor] | torch.Tensor | None = None) -> torch.Tensor:
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
        if x.dim() not in (2, 3):
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")
        # Standardize 2D input (B, L) to 3D channel format (B, 1, L)
        x_in = x.unsqueeze(1) if x.dim() == 2 else x

        # Compute FiLM modulation parameters (gamma, beta) if conditioning is enabled
        film_modulations = self.film_gen(cond) if (self.use_film and cond is not None) else None

        # Hidden representation extraction across layers 1 and 2 with FiLM affine modulation
        hidden_features = x_in
        for layer_index, (conv_layer, batch_norm_layer) in enumerate(((self.conv1, self.bn1), (self.conv2, self.bn2))):
            hidden_features = batch_norm_layer(conv_layer(hidden_features))
            if film_modulations is not None:
                gamma, beta = film_modulations[layer_index]
                hidden_features = gamma.unsqueeze(-1) * hidden_features + beta.unsqueeze(-1)
            hidden_features = self.dropout(self.activation(hidden_features))

        # Project to residual difference Delta(x) and add physical shortcut: y = x + Delta(x)
        return (x_in + self.conv3(hidden_features)).squeeze(1) if x.dim() == 2 else (x_in + self.conv3(hidden_features))

    def get_l2_regularization(self) -> torch.Tensor:
        """Compute the sum of squared weights (L2 regularization penalty) for conv layers.

        Biases and BatchNorm parameters are intentionally excluded to avoid penalizing
        baseline intensity shifts and scaling factors.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing sum of squared convolutional weights.
        """
        weights = [param for name, param in self.named_parameters() if param.requires_grad and name.endswith(".weight") and "bn" not in name]
        return sum((param ** 2).sum() for param in weights) if weights else torch.tensor(0.0, device=next(self.parameters()).device)
