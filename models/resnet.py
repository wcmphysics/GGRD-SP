"""1D ResNet neural network architecture for spectral transformation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


from models.film import FiLMGenerator


class ResNet1D(nn.Module):
    """Residual 1D CNN with 3 convolutional layers, mirror padding, identity shortcut, and optional FiLM.

    Transforms a 1D regional spectrum sequence from source tool to target tool
    while preserving sequence length: y = x + Delta(x).

    Uses reflection (mirror) padding to prevent boundary attenuation artifacts
    at sequence edges.

    Parameters
    ----------
    n_points : int
        Number of spectral data points (length of input sequence or patch window).
    config : dict[str, Any] | None, optional
        Configuration dictionary containing:
        - 'hidden_channels' (int): Feature channels in hidden layers (default 32).
        - 'kernel_size' (int): Convolution kernel size, positive odd int (default 5).
        - 'dropout' (float): Dropout probability between conv layers (default 0.0).
        - 'use_batch_norm' (bool): Whether to use BatchNorm1d in hidden layers (default True).
        - 'padding_mode' (str): Padding mode 'reflect' (mirror) or 'zeros' (default 'reflect').
        - 'use_film' (bool): Whether to enable Feature-wise Linear Modulation conditioning (default False).
    """

    def __init__(self, n_points: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        hidden_channels: int = int(cfg.get("hidden_channels", 32))
        kernel_size: int = int(cfg.get("kernel_size", 5))
        dropout: float = float(cfg.get("dropout", 0.0))
        use_batch_norm: bool = bool(cfg.get("use_batch_norm", True))
        padding_mode: str = str(cfg.get("padding_mode", "reflect"))
        self.use_film: bool = bool(cfg.get("use_film", False))

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )

        pad = kernel_size // 2
        self.n_points = n_points

        def _make_conv_layer(in_ch: int, out_ch: int) -> nn.Module:
            layers: list[nn.Module] = []
            if pad > 0:
                if padding_mode == "reflect" and n_points > pad:
                    layers.append(nn.ReflectionPad1d(pad))
                    layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=0))
                else:
                    layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad))
            else:
                layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=0))
            return nn.Sequential(*layers) if len(layers) > 1 else layers[0]

        # Individual layers allowing FiLM modulation between conv/bn and activation
        self.conv1 = _make_conv_layer(1, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels) if use_batch_norm else nn.Identity()
        self.act1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.conv2 = _make_conv_layer(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels) if use_batch_norm else nn.Identity()
        self.act2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.conv3 = _make_conv_layer(hidden_channels, 1)

        if self.use_film:
            self.film_gen = FiLMGenerator(
                channels_list=[hidden_channels, hidden_channels],
                config=cfg,
            )
        else:
            self.film_gen = None

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass applying residual transformation: y = x + Delta(x, cond).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_points) or (batch_size, 1, n_points).
        cond : dict[str, torch.Tensor] | torch.Tensor | None, optional
            Optional metadata conditioning features for FiLM modulation.

        Returns
        -------
        torch.Tensor
            Transformed spectrum matching input sequence dimensionality.
        """
        orig_dim = x.dim()
        if orig_dim == 2:
            x_in = x.unsqueeze(1)
        elif orig_dim == 3:
            x_in = x
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        mods = self.film_gen(cond) if (self.use_film and cond is not None) else None

        # Layer 1
        h = self.conv1(x_in)
        h = self.bn1(h)
        if mods is not None:
            gamma1, beta1 = mods[0]
            h = gamma1.unsqueeze(-1) * h + beta1.unsqueeze(-1)
        h = self.act1(h)
        h = self.drop1(h)

        # Layer 2
        h = self.conv2(h)
        h = self.bn2(h)
        if mods is not None:
            gamma2, beta2 = mods[1]
            h = gamma2.unsqueeze(-1) * h + beta2.unsqueeze(-1)
        h = self.act2(h)
        h = self.drop2(h)

        # Layer 3
        delta = self.conv3(h)
        out = x_in + delta

        if orig_dim == 2:
            return out.squeeze(1)
        return out

    def get_l2_regularization(self) -> torch.Tensor:
        """Compute the sum of squared weights (L2 regularization penalty).

        Only penalizes convolutional weight matrices; biases and BatchNorm
        parameters are excluded to preserve baseline intensity offsets.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing sum of squares of convolutional weights.
        """
        device = next(self.parameters()).device
        l2_sum = torch.tensor(0.0, device=device)
        for name, param in self.named_parameters():
            if param.requires_grad and name.endswith(".weight") and "bn" not in name:
                l2_sum = l2_sum + torch.sum(param ** 2)
        return l2_sum

