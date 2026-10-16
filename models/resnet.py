"""1D ResNet neural network architecture for sequence-to-sequence spectral transformation.

Transforms a regional spectrum measured on a source tool into the corresponding
spectrum as if measured on a target tool, fulfilling Part 3 of GEMINI.md.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.film import FiLMGenerator
from models.layers import SafeReflectionPad1d


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
        - 'resnet_channel' (int): Feature channels in hidden conv layers (default 32).
        - 'kernel_size' (int): Convolution kernel size, must be positive odd int (default 5).
        - 'dropout' (float): Dropout probability applied between conv layers (default 0.0).
        - 'use_batch_norm' (bool): Whether to include BatchNorm1d in hidden layers (default True).
        - 'padding_mode' (str): Padding mode 'reflect' (mirror) or 'zeros' (default 'reflect').
        - 'use_film' (bool): Whether to enable Feature-wise Linear Modulation conditioning (default False).
    """

    def __init__(self, n_points: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        effective_config = config or {}

        for invalid_key in (
            "resnet_channels",
            "hidden_channels",
            "hidden_channel",
            "kernel_sizes",
            "batch_sizes",
            "window_size_choices",
        ):
            if invalid_key in effective_config:
                raise KeyError(
                    f"Invalid key '{invalid_key}' in ResNet1D config; "
                    "use singular 'resnet_channel' and 'kernel_size' outside Bayesian optimization."
                )

        for int_key in ("batch_size", "window_size", "window_size_choice"):
            if int_key in effective_config:
                raw_v = effective_config[int_key]
                if isinstance(raw_v, bool) or not isinstance(raw_v, int) or raw_v <= 0:
                    raise ValueError(f"{int_key} must be a positive integer, got {raw_v!r}")

        raw_channel = effective_config.get("resnet_channel", 32)
        if isinstance(raw_channel, bool) or not isinstance(raw_channel, int) or raw_channel <= 0:
            raise ValueError(
                f"resnet_channel must be a positive integer, got {raw_channel!r}"
            )
        resnet_channel = raw_channel

        raw_kernel = effective_config.get("kernel_size", 5)
        if isinstance(raw_kernel, bool) or not isinstance(raw_kernel, int):
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {raw_kernel!r}"
            )
        kernel_size = raw_kernel

        dropout = float(effective_config.get("dropout", 0.0))
        use_batch_norm = bool(effective_config.get("use_batch_norm", True))
        padding_mode = str(effective_config.get("padding_mode", "reflect"))
        self.use_film = bool(effective_config.get("use_film", False))
        self.energy_scale_ev = float(effective_config.get("energy_scale_ev", 1000.0))
        self.n_points = int(n_points)

        explicit_ws = effective_config.get("window_size_choice", effective_config.get("window_size"))
        max_allowed_k = min(self.n_points, explicit_ws) if explicit_ws is not None else self.n_points

        # Validate that kernel size is positive, odd, and does not exceed sequence/window length
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if kernel_size > max_allowed_k:
            raise ValueError(
                f"kernel_size ({kernel_size}) cannot be larger than window_size / n_points ({max_allowed_k})."
            )

        padding_size = kernel_size // 2

        # Factory helper to construct conv layer with safe reflection padding
        def _make_conv(in_channels: int, out_channels: int) -> nn.Module:
            if padding_size > 0 and padding_mode == "reflect":
                return nn.Sequential(
                    SafeReflectionPad1d(padding_size),
                    nn.Conv1d(in_channels, out_channels, kernel_size),
                )
            return nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding_size)

        # 3-layer CNN backbone: 1 -> hidden -> hidden -> 1 (spectral residual Delta)
        self.conv1 = _make_conv(1, resnet_channel)
        self.conv2 = _make_conv(resnet_channel, resnet_channel)
        self.conv3 = _make_conv(resnet_channel, 1)

        # Normalization, activation, and dropout layers
        self.bn1 = nn.BatchNorm1d(resnet_channel) if use_batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(resnet_channel) if use_batch_norm else nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Optional FiLM generator for metadata-conditioned modulation (die, tool, drift)
        self.film_gen = (
            FiLMGenerator([resnet_channel, resnet_channel], config=effective_config)
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
            Input spectral tensor of shape (batch_size, 1, n_points).
        cond : dict[str, torch.Tensor] | torch.Tensor | None, optional
            Optional metadata conditioning features for FiLM modulation.

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

        # Compute FiLM modulation parameters (gamma, beta) if conditioning is enabled
        film_modulations = self.film_gen(cond) if (self.use_film and cond is not None) else None

        # Hidden representation extraction across layers 1 and 2 with FiLM affine modulation
        hidden_features = x
        for layer_index, (conv_layer, batch_norm_layer) in enumerate(
            ((self.conv1, self.bn1), (self.conv2, self.bn2))
        ):
            hidden_features = batch_norm_layer(conv_layer(hidden_features))
            if film_modulations is not None:
                gamma, beta = film_modulations[layer_index]
                hidden_features = gamma.unsqueeze(-1) * hidden_features + beta.unsqueeze(-1)
            hidden_features = self.dropout(self.activation(hidden_features))

        # Project to residual difference Delta(x) and add physical shortcut: y = x + Delta(x)
        delta = self.conv3(hidden_features)
        return x + delta

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
        l2_sum = torch.tensor(0.0, device=device)
        for module in self.modules():
            if isinstance(module, nn.Conv1d) and module.weight.requires_grad:
                l2_sum = l2_sum + torch.sum(module.weight**2)
        return l2_sum
