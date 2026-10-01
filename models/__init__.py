"""Neural network models and training pipelines for spectral transfer."""

from __future__ import annotations

from models.baseline_cnn import NormalizedMSELoss, Residual1DCNN
from models.bayesian_opt import optimize_baseline_hyperparameters
from models.dataset import (
    SpectrumPairDataset,
    create_dataloaders,
    split_session_datasets,
)
from models.inference import format_predicted_measurement_id, predict_spectra
from models.root import run_baseline_pipeline
from models.sliding_window import (
    SpectrumPatchDataset,
    calculate_window_points,
    evaluate_sliding_window,
    extract_sliding_windows,
    optimize_sliding_window_hyperparameters,
    predict_sliding_window_spectra,
    predict_sliding_window_spectrum,
    reconstruct_from_patches,
    run_sliding_window_pipeline,
    train_sliding_window_region,
)
from models.trainer import evaluate, train_baseline_region, train_one_epoch
from models.unet import (
    UNet1D,
    optimize_unet_hyperparameters,
    run_unet_pipeline,
    train_unet_region,
)

__all__ = [
    "Residual1DCNN",
    "NormalizedMSELoss",
    "SpectrumPairDataset",
    "split_session_datasets",
    "create_dataloaders",
    "train_one_epoch",
    "evaluate",
    "train_baseline_region",
    "optimize_baseline_hyperparameters",
    "format_predicted_measurement_id",
    "predict_spectra",
    "run_baseline_pipeline",
    "calculate_window_points",
    "extract_sliding_windows",
    "reconstruct_from_patches",
    "SpectrumPatchDataset",
    "predict_sliding_window_spectrum",
    "evaluate_sliding_window",
    "train_sliding_window_region",
    "optimize_sliding_window_hyperparameters",
    "predict_sliding_window_spectra",
    "run_sliding_window_pipeline",
    "UNet1D",
    "train_unet_region",
    "optimize_unet_hyperparameters",
    "run_unet_pipeline",
]
