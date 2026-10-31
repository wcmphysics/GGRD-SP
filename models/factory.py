"""Neural network model factory for 1D spectral transformation architectures."""

from __future__ import annotations

from typing import Any
import torch.nn as nn

from models.deeplabv3 import DeepLabV3
from models.fpn import FPN1D
from models.inception import InceptionTime1D
from models.resnet import ResNet1D, ResNetScaledShort1D
from models.tcn import TCN1D
from models.unet import ConventionalUNet1D, ResidualUNet1D

NO_SLIDING_WINDOW_MODEL_TYPES: frozenset[str] = frozenset(
    {"inception_time", "tcn", "fpn"}
)

MODEL_CLASS_MAP: dict[str, type[nn.Module]] = {
    "resnet": ResNet1D,
    "resnet_scaled_short": ResNetScaledShort1D,
    "residual_unet": ResidualUNet1D,
    "unet": ConventionalUNet1D,
    "deeplabv3": DeepLabV3,
    "inception_time": InceptionTime1D,
    "tcn": TCN1D,
    "fpn": FPN1D,
}

__all__ = [
    "MODEL_CLASS_MAP",
    "NO_SLIDING_WINDOW_MODEL_TYPES",
    "instantiate_model",
]


def instantiate_model(
    model_type: str,
    seq_len: int,
    config: dict[str, Any] | None = None,
) -> nn.Module:
    """Instantiate a 1D spectral neural network by architecture name and
    sequence length.

    Parameters
    ----------
    model_type : str
        Architecture name: 'resnet', 'resnet_scaled_short', 'residual_unet',
        'unet', 'deeplabv3', 'inception_time', 'tcn', or 'fpn'.
    seq_len : int
        Input sequence length in data points (full regional spectrum or
        patch length).
    config : dict[str, Any] | None, optional
        Dictionary of model architecture hyperparameters (e.g., kernel_size,
        resnet_channel, residual_unet_channel, unet_channel, depth, use_film).

    Returns
    -------
    nn.Module
        Instantiated PyTorch neural network module ready for training.
    """
    model_class = MODEL_CLASS_MAP.get(model_type.lower())
    if model_class is None:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose 'resnet', "
            "'resnet_scaled_short', 'residual_unet', 'unet', 'deeplabv3', "
            "'inception_time', 'tcn', or 'fpn'."
        )
    return model_class(seq_len, config=config or {})
