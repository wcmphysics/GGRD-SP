"""1D ResNet neural network architecture for sequence-to-sequence spectral
transformation.

Transforms a regional spectrum measured on a source tool into the corresponding
spectrum as if measured on a target tool, fulfilling Part 3 of GEMINI.md.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.film import FiLMGenerator
from models.layers import (
    build_conv1d_layer,
    compute_conv1d_l2_reg,
    validate_3d_spectrum_tensor,
    validate_common_model_config,
    validate_odd_kernel_size,
    validate_positive_int,
)

_DISALLOWED_RESNET_KEYS = (
    "resnet_channels",
    "hidden_channels",
    "hidden_channel",
    "kernel_sizes",
    "batch_sizes",
    "window_size_choices",
)


class ResNet1D(nn.Module):
    """Residual 1D CNN with 3 convolutional layers, mirror padding, identity
    shortcut, and optional FiLM.

    Transforms a 1D regional spectrum sequence from source tool to target tool
    while preserving sequence length: y = x + Delta(x, cond).

    The identity shortcut preserves the physical spectral baseline, while the
    3 CNN layers learn the tool-to-tool spectral shift and deformation Delta(x).
    Reflection (mirror) padding prevents boundary attenuation artifacts at
    spectral edges.

    Parameters
    ----------
    n_points : int
        Number of spectral data points (length of input regional spectrum or
        patch window).
    config : dict[str, Any] | None, optional
        Configuration dictionary packaging architectural hyperparameters:
        - 'resnet_channel' (int): Feature channels in hidden conv layers
          (default 32).
        - 'kernel_size' (int): Convolution kernel size, must be positive odd
          int (default 5).
        - 'dropout' (float): Dropout probability applied between conv layers
          (default 0.0).
        - 'use_batch_norm' (bool): Whether to include BatchNorm1d in hidden
          layers (default True).
        - 'padding_mode' (str): Padding mode 'reflect' (mirror) or 'zeros'
          (default 'reflect').
        - 'use_film' (bool): Whether to enable Feature-wise Linear Modulation
          conditioning (default False).
    """

    def __init__(self, n_points: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        self.n_points = int(n_points)
        max_allowed_k = validate_common_model_config(
            cfg,
            self.n_points,
            _DISALLOWED_RESNET_KEYS,
            (
                "ResNet1D",
                "use singular 'resnet_channel' and 'kernel_size' outside "
                "Bayesian optimization.",
            ),
        )

        resnet_channel = validate_positive_int(
            cfg.get("resnet_channel", 32), "resnet_channel"
        )
        kernel_size = validate_odd_kernel_size(
            cfg.get("kernel_size", 5), max_allowed=max_allowed_k
        )

        dropout = float(cfg.get("dropout", 0.0))
        use_batch_norm = bool(cfg.get("use_batch_norm", True))
        padding_mode = str(cfg.get("padding_mode", "reflect"))
        self.use_film = bool(cfg.get("use_film", False))
        self.energy_scale_ev = float(cfg.get("energy_scale_ev", 1000.0))

        # 3-layer CNN backbone: 1 -> hidden -> hidden -> 1 (residual Delta)
        self.conv1 = build_conv1d_layer(
            (1, resnet_channel), kernel_size, padding_mode=padding_mode
        )
        self.conv2 = build_conv1d_layer(
            (resnet_channel, resnet_channel),
            kernel_size,
            padding_mode=padding_mode,
        )
        self.conv3 = build_conv1d_layer(
            (resnet_channel, 1), kernel_size, padding_mode=padding_mode
        )

        # Normalization, activation, and dropout layers
        self.bn1 = nn.BatchNorm1d(resnet_channel) if use_batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(resnet_channel) if use_batch_norm else nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Optional FiLM generator for metadata-conditioned modulation
        self.film_gen = (
            FiLMGenerator([resnet_channel, resnet_channel], config=cfg)
            if self.use_film
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply residual transformation: y = x + Delta(x, cond).


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
        validate_3d_spectrum_tensor(x)

        film_modulations = (
            self.film_gen(cond)
            if (self.use_film and cond is not None and self.film_gen is not None)
            else None
        )

        hidden_features = x
        for layer_index, (conv_layer, batch_norm_layer) in enumerate(
            ((self.conv1, self.bn1), (self.conv2, self.bn2))
        ):
            hidden_features = batch_norm_layer(conv_layer(hidden_features))
            if film_modulations is not None:
                gamma, beta = film_modulations[layer_index]
                hidden_features = (
                    gamma.unsqueeze(-1) * hidden_features + beta.unsqueeze(-1)
                )
            hidden_features = self.dropout(self.activation(hidden_features))

        return x + self.conv3(hidden_features)

    def get_l2_regularization(self) -> torch.Tensor:
        """Compute the sum of squared weights for conv layers.

        Biases and BatchNorm parameters are intentionally excluded to avoid
        penalizing baseline intensity shifts and scaling factors.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing sum of squared convolutional weights.
        """
        return compute_conv1d_l2_reg(self)
