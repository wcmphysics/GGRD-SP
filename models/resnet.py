"""1D ResNet neural network architecture for sequence-to-sequence spectral
transformation.

Transforms a regional spectrum measured on a source tool into the corresponding
spectrum as if measured on a target tool, fulfilling Stage 9 of AGENTS.md.
"""

from __future__ import annotations

from typing import Any
import warnings

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

REQUIRED_RESNET_KEYS = (
    "resnet_channel",
    "kernel_size",
    "dropout",
    "use_batch_norm",
    "padding_mode",
    "use_film",
)


def _validate_required_resnet_keys(cfg: dict[str, Any]) -> None:
    """Validate presence of all mandatory ResNet configuration keys.

    Raises KeyError if any required configuration key is missing.
    """
    missing = [key for key in REQUIRED_RESNET_KEYS if key not in cfg]
    if missing:
        raise KeyError(
            f"Missing required ResNet config key(s): {missing}. "
            "Please ensure they are defined in config.yaml under 'training.train_config'."
        )


def _validate_layer_count(num_layers: int) -> int:
    """Validate layer count to ensure at least one hidden and one output conv.

    Raises ValueError if num_layers < 2 (cannot form a residual block).
    """
    if num_layers < 2:
        raise ValueError(f"ResNet1D num_layers must be >= 2, got {num_layers}.")
    return num_layers


def _build_hidden_conv_and_norm_layers(
    num_hidden: int,
    resnet_channel: int,
    kernel_size: int,
    padding_mode: str,
    use_batch_norm: bool,
) -> tuple[nn.ModuleList, nn.ModuleList]:
    """Build paired lists of 1D conv layers and BatchNorm/Identity modules.

    Iterates through hidden layers, mapping 1 input channel to resnet_channel
    at layer 0 and preserving resnet_channel across all subsequent layers.
    """
    hidden_convs = [build_conv1d_layer(channels = (1 if i == 0 else resnet_channel, resnet_channel),
                                       kernel_size = kernel_size,
                                       padding_mode = padding_mode,) 
                    for i in range(num_hidden)]
    hidden_convs = nn.ModuleList(hidden_convs)
        
    hidden_bns = [nn.BatchNorm1d(resnet_channel) if use_batch_norm else nn.Identity()
                  for _ in range(num_hidden)]
    hidden_bns = nn.ModuleList(hidden_bns)
    
    return hidden_convs, hidden_bns


def _build_shortcut_film_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build dedicated FiLM configuration for the scaled shortcut connection.

    Disables energy bounds if global sample attributes (die, tool, flux, time)
    are active, ensuring shortcut gain reflects global tool scaling.
    """
    has_global_condition = any([
        cfg.get("encode_die", True),
        cfg.get("encode_tool", True),
        cfg.get("encode_flux", True),
        cfg.get("encode_time", True),
        cfg.get("encode_region", False),
        cfg.get("encode_relative_intensity", False),
        cfg.get("encode_elements", False),
    ])
    return {
        **cfg,
        "use_sliding_window": False,
        "encode_energy": (
            False
            if has_global_condition
            else bool(cfg.get("encode_energy", False))
        ),
    }


def _extract_and_clamp_shortcut_gain(
    shortcut_mods: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Extract and non-negatively clamp scalar shortcut gain gamma_out.

    Clamps negative gain to zero to prevent unphysical spectrum inversion if
    FiLM outputs a negative scale factor on the identity residual path.
    """
    gamma_out, _ = shortcut_mods[0]
    if not (gamma_out < 0.0).any():
        return gamma_out

    warnings.warn(
        "Negative shortcut gain detected in ResNetScaledShort1D; "
        "clamping gamma_out to 0.0.",
        UserWarning,
        stacklevel=3,
    )
    return torch.clamp(gamma_out, min=0.0)


class ResNet1D(nn.Module):
    """Residual 1D CNN with 3 convolutional layers, mirror padding, identity
    shortcut, and optional FiLM.

    Transforms a 1D regional spectrum sequence from source tool to target tool
    while preserving sequence length: y = x + Delta(x, cond).

    The identity shortcut preserves the physical spectral baseline, while the
    3 CNN layers learn the tool-to-tool spectral shift and deformation Delta(x).
    Reflection (mirror) padding prevents boundary attenuation artifacts at
    spectral edges.

    Structure Overview:
    1. Validates hyperparameter dictionary and kernel bounds against n_points.
    2. Builds hidden 1D conv layers and normalization modules.
    3. Initializes 1x1 output projection conv, activation, and dropout layers.
    4. Instantiates optional FiLM generator for metadata condition modulation.

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
        
        cfg = config if config is not None else {}
        self.n_points = int(n_points)
        
        # validatet config and inputs
        _validate_required_resnet_keys(cfg)
        max_allowed_k = validate_common_model_config(cfg, self.n_points, (), ("ResNet1D", ""),)
        resnet_channel = validate_positive_int(cfg["resnet_channel"], "resnet_channel")
        kernel_size = validate_odd_kernel_size(cfg["kernel_size"], max_allowed=max_allowed_k)
        dropout = float(cfg["dropout"])
        use_batch_norm = bool(cfg["use_batch_norm"])
        padding_mode = str(cfg["padding_mode"])
        self.use_film = bool(cfg["use_film"])
        self.energy_scale_ev = float(cfg.get("energy_scale_ev", 1000.0))
        self.num_layers = _validate_layer_count(int(cfg.get("num_layers", 3)))
        num_hidden = self.num_layers - 1

        # setup neural-layer components (hidden and output CNNs, activation, dropout, film)
        self.hidden_convs, self.hidden_bns = _build_hidden_conv_and_norm_layers(
            num_hidden,
            resnet_channel,
            kernel_size,
            padding_mode,
            use_batch_norm,
        )
        self.conv_out = build_conv1d_layer((resnet_channel, 1), kernel_size, padding_mode=padding_mode)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.film_gen = (FiLMGenerator([resnet_channel] * num_hidden, config=cfg) if self.use_film else None)

    def _get_film_modulations(self, cond: dict[str, torch.Tensor] | torch.Tensor | None
                             ) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        """Compute FiLM modulation parameters from metadata condition vector.

        Returns None early if FiLM is disabled, condition is absent, or the
        generator is uninitialized.
        """
        if not self.use_film or cond is None or self.film_gen is None:
            return None
        return self.film_gen(cond)

    def _forward_delta(self, x: torch.Tensor, 
                       film_modulations: list[tuple[torch.Tensor, torch.Tensor]] | None = None,) -> torch.Tensor:
        """Execute the convolutional residual branch Delta(x)."""
        hidden_features = x
        for layer_index, (conv_layer, batch_norm_layer) in enumerate(zip(self.hidden_convs, self.hidden_bns)):
            hidden_features = batch_norm_layer(conv_layer(hidden_features))
            if film_modulations is not None:
                gamma, beta = film_modulations[layer_index]
                hidden_features = (gamma.unsqueeze(-1) * hidden_features + beta.unsqueeze(-1))
            hidden_features = self.dropout(self.activation(hidden_features))

        return self.conv_out(hidden_features)

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
        film_modulations = self._get_film_modulations(cond)
        return x + self._forward_delta(x, film_modulations)

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


class ResNetScaledShort1D(ResNet1D):
    """Residual 1D CNN with FiLM-scaled shortcut:
    y = gamma_out * x + Delta(x, cond).

    Extends ResNet1D by modulating the shortcut connection with a
    multiplicative gain gamma_out derived via Feature-wise Linear Modulation
    (FiLM). The shortcut gain scales the baseline spectrum x by a
    condition-dependent factor (e.g., tool or flux ratio) while the residual
    branch Delta learns localized spectral shape differences.

    Parameters
    ----------
    n_points : int
        Number of spectral data points (length of input regional spectrum or
        patch window).
    config : dict[str, Any] | None, optional
        Configuration dictionary. Must have 'use_film': True. Supports all
        standard ResNet1D parameters plus:
        - 'film_internal_layers' (bool): Whether to apply FiLM conditioning to
          internal convolutional layers (default True). When False, internal
          layers learn unconditioned Delta(x), and FiLM is only applied to the
          shortcut.
    """

    def __init__(
        self, n_points: int, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(n_points, config)
        cfg = config or {}
        if not self.use_film:
            raise ValueError(
                "ResNetScaledShort1D requires FiLM conditioning. "
                "Please set 'use_film': True in model config."
            )

        self.film_internal_layers = bool(cfg.get("film_internal_layers", True))
        if not self.film_internal_layers:
            self.film_gen = None

        shortcut_cfg = _build_shortcut_film_config(cfg)
        self.shortcut_film_gen = FiLMGenerator([1], config=shortcut_cfg)

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply scaled-shortcut residual transformation:
        y = gamma_out * x + Delta(x, cond).

        Parameters
        ----------
        x : torch.Tensor
            Input spectral tensor of shape (batch_size, 1, n_points).
        cond : dict[str, torch.Tensor] | torch.Tensor | None, optional
            Metadata conditioning features dictionary. Required.

        Returns
        -------
        torch.Tensor
            Transformed spectrum of shape (batch_size, 1, n_points).

        Raises
        ------
        ValueError
            If x is not a 3D tensor or cond is None or not a dict.
        """
        validate_3d_spectrum_tensor(x)
        if cond is None or not isinstance(cond, dict):
            raise ValueError(
                "ResNetScaledShort1D requires a metadata conditioning dictionary "
                f"'cond', got {type(cond).__name__ if cond is not None else 'None'}."
            )

        shortcut_mods = self.shortcut_film_gen(cond)
        gamma_out = _extract_and_clamp_shortcut_gain(shortcut_mods)
        film_modulations = self._get_film_modulations(cond)
        delta = self._forward_delta(x, film_modulations)
        return (gamma_out.unsqueeze(-1) * x) + delta


class ResNet5L1D(ResNet1D):
    """5-layer 1D ResNet: 4 hidden convolutional layers + 1 projection layer.

    Transforms a 1D regional spectrum sequence from source tool to target tool
    while preserving sequence length: y = x + Delta(x, cond).
    Features 4 hidden layers of uniform channel width (resnet_channel), mirror
    padding, identity shortcut, and optional FiLM modulation across all 4
    hidden layers.

    Parameters
    ----------
    n_points : int
        Number of spectral data points (length of input regional spectrum or
        patch window).
    config : dict[str, Any] | None, optional
        Configuration dictionary. Accepts standard ResNet1D hyperparameters.
    """

    def __init__(
        self, n_points: int, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(n_points, {**(config or {}), "num_layers": 5})


__all__ = ["REQUIRED_RESNET_KEYS", "ResNet1D", "ResNet5L1D", "ResNetScaledShort1D"]
