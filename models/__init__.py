"""Neural network models and training pipelines for spectral transfer."""

from __future__ import annotations

from models.bayesian_opt import (
    optimize_baseline_hyperparameters,
    optimize_model_hyperparameters,
    optimize_sliding_window_hyperparameters,
    optimize_unet_hyperparameters,
    run_ax_search,
)
from models.cost import NormalizedMSELoss, format_loss_log10
from models.dataset import (
    SpectrumPairDataset,
    SpectrumPatchDataset,
    create_dataloaders,
    partition_measurement_sessions,
    split_session_datasets,
)
from models.inference import (
    assemble_prediction_metadata,
    format_predicted_measurement_id,
    predict_sliding_window_spectrum,
    predict_spectra,
)
from models.orchestration import (
    instantiate_model,
    run_baseline_pipeline,
    run_model_pipeline,
    run_residual_unet_pipeline,
    run_resnet_pipeline,
    run_sliding_window_pipeline,
    run_spectral_pipeline,
    run_unet_pipeline,
)
from models.resnet import ResNet1D, Residual1DCNN
from models.sliding_window import (
    calculate_window_points,
    evaluate_sliding_window,
    extract_sliding_windows,
    predict_sliding_window_spectra,
    reconstruct_from_patches,
    train_sliding_window_region,
)
from models.trainer import (
    evaluate,
    train_baseline_region,
    train_model_region,
    train_one_epoch,
    train_unet_region,
)
from models.unet import (
    ConventionalUNet1D,
    ResidualUNet1D,
    UNet1D,
    predict_unet_spectra,
)

__all__ = [
    # Models
    "ResNet1D",
    "Residual1DCNN",
    "UNet1D",
    "ResidualUNet1D",
    "ConventionalUNet1D",
    "instantiate_model",
    # Cost
    "NormalizedMSELoss",
    "format_loss_log10",
    # Datasets
    "SpectrumPairDataset",
    "SpectrumPatchDataset",
    "create_dataloaders",
    "partition_measurement_sessions",
    "split_session_datasets",
    # Training & Evaluation
    "train_one_epoch",
    "evaluate",
    "evaluate_sliding_window",
    "train_model_region",
    "train_baseline_region",
    "train_sliding_window_region",
    "train_unet_region",
    # Optimization
    "run_ax_search",
    "optimize_model_hyperparameters",
    "optimize_baseline_hyperparameters",
    "optimize_sliding_window_hyperparameters",
    "optimize_unet_hyperparameters",
    # Inference
    "format_predicted_measurement_id",
    "assemble_prediction_metadata",
    "predict_spectra",
    "predict_sliding_window_spectrum",
    "predict_sliding_window_spectra",
    "predict_unet_spectra",
    # Orchestration pipelines
    "run_spectral_pipeline",
    "run_resnet_pipeline",
    "run_residual_unet_pipeline",
    "run_unet_pipeline",
    "run_sliding_window_pipeline",
    "run_baseline_pipeline",
    "run_model_pipeline",
    # Patching utilities
    "calculate_window_points",
    "extract_sliding_windows",
    "reconstruct_from_patches",
]
