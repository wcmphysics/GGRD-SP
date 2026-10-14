"""Neural network models and training pipelines for spectral transfer."""

from __future__ import annotations

from models.bayesian_opt import (
    optimize_model_hyperparameters,
    run_ax_search,
)
from models.cost import (
    SpectralLoss,
    create_cost_function,
    format_loss_log10,
)
from models.dataset import (
    SpectrumPairDataset,
    SpectrumPatchDataset,
    create_dataloaders,
    create_regional_datasets,
    partition_measurement_sessions,
    split_session_datasets,
)
from models.deeplabv3 import DeepLabV3, DeepLabV3_1D
from models.film import FiLMGenerator, prepare_film_condition_tensor
from models.inference import (
    assemble_prediction_metadata,
    format_predicted_measurement_id,
    predict_sliding_window_spectrum,
    predict_spectra,
)
from models.orchestration import (
    instantiate_model,
    perform_model_training_and_bayesian_optimization,
)
from models.resnet import ResNet1D
from models.trainer import (
    evaluate,
    evaluate_sliding_window,
    train_model_region,
    train_one_epoch,
)
from models.unet import (
    ConventionalUNet1D,
    ResidualUNet1D,
    UNet1D,
)

__all__ = [
    # Models
    "ResNet1D",
    "UNet1D",
    "ResidualUNet1D",
    "ConventionalUNet1D",
    "DeepLabV3",
    "DeepLabV3_1D",
    "FiLMGenerator",
    "instantiate_model",
    # Cost
    "SpectralLoss",
    "create_cost_function",
    "format_loss_log10",
    # Conditioning & Datasets
    "prepare_film_condition_tensor",
    "SpectrumPairDataset",
    "SpectrumPatchDataset",
    "create_dataloaders",
    "create_regional_datasets",
    "partition_measurement_sessions",
    "split_session_datasets",
    # Training & Evaluation
    "train_one_epoch",
    "evaluate",
    "evaluate_sliding_window",
    "train_model_region",
    # Optimization
    "run_ax_search",
    "optimize_model_hyperparameters",
    # Inference
    "format_predicted_measurement_id",
    "assemble_prediction_metadata",
    "predict_spectra",
    "predict_sliding_window_spectrum",
    # Orchestration pipelines
    "perform_model_training_and_bayesian_optimization",
]
