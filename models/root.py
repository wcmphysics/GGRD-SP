"""Root function orchestrating baseline neural network training, hyperparameter search, and prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch.nn as nn

from models.bayesian_opt import optimize_baseline_hyperparameters
from models.dataset import split_session_datasets
from models.inference import predict_spectra
from models.trainer import train_baseline_region


def run_baseline_pipeline(
    meta_df: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the end-to-end baseline modeling pipeline.

    Orchestrates:
    1. Session-level dataset splitting per spectral region.
    2. Optional Bayesian hyperparameter optimization via Ax.
    3. Model instantiation and training using normalized MSE loss and L2 penalty.
    4. Model evaluation on validation sessions (pure reconstruction error).
    5. Sequence-to-sequence prediction producing standardized output containers:
       (ary_intensity_predicted, ary_energy_predicted, meta_df_predicted).

    Parameters
    ----------
    meta_df : pd.DataFrame
        Metadata DataFrame containing source-to-target pairing information.
    ary_intensity : np.ndarray
        2D array of measured intensities.
    ary_energy : np.ndarray
        2D array of binding energies.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'regions' (list[str] | None): Regions to train (default auto-detected from paired data).
        - 'source_tool' (str): Source tool identifier (default 'J4').
        - 'target_tool' (str): Target tool identifier (default 'H1').
        - 'val_ratio' (float): Fraction of sessions reserved for validation (default 0.2).
        - 'seed' (int | None): Random seed for reproducibility (default 42).
        - 'use_bayesian_opt' (bool): Whether to perform Ax Bayesian optimization (default False).
        - 'bayesian_opt_config' (dict[str, Any] | None): Configuration for Ax optimization.
        - 'train_config' (dict[str, Any] | None): Training hyperparameters.
        - 'predict_source' (bool): Whether to run inference on source data (default True).
        - 'verbose' (bool): Whether to print training progress (default False).

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'models': Dictionary mapping region -> trained Residual1DCNN.
        - 'evaluation': Dictionary of validation losses per region.
        - 'histories': Training history per region.
        - 'bayesian_opt_results': Results from Ax (if use_bayesian_opt=True).
        - 'predictions': Tuple of (ary_intensity_predicted, ary_energy_predicted, meta_df_predicted).
    """
    cfg = config or {}
    source_tool: str = cfg.get("source_tool", "J4")
    target_tool: str = cfg.get("target_tool", "H1")
    val_ratio: float = float(cfg.get("val_ratio", 0.2))
    seed: int | None = cfg.get("seed", 42)
    use_bo: bool = bool(cfg.get("use_bayesian_opt", False))
    bo_cfg: dict[str, Any] = dict(cfg.get("bayesian_opt_config", {}))
    train_cfg: dict[str, Any] = dict(cfg.get("train_config", {}))
    predict_source: bool = bool(cfg.get("predict_source", True))
    verbose: bool = bool(cfg.get("verbose", False))

    if "seed" not in bo_cfg and seed is not None:
        bo_cfg["seed"] = seed

    # Detect regions if not explicitly provided
    regions: list[str] | None = cfg.get("regions")
    if regions is None:
        paired_mask = meta_df["measurement_id_target"].notna()
        if source_tool:
            paired_mask = paired_mask & (meta_df["tool"] == source_tool)
        regions = sorted(meta_df[paired_mask]["region"].unique().tolist())

    if not regions:
        raise ValueError(
            f"No valid spectral regions found to train for source_tool='{source_tool}', target_tool='{target_tool}'"
        )

    models: dict[str, nn.Module] = {}
    eval_results: dict[str, float] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    bo_results: dict[str, dict[str, Any]] = {}

    for region in regions:
        if verbose:
            print(f"\n--- Processing Region: {region} ({source_tool} -> {target_tool}) ---")

        split_cfg = {
            "region": region,
            "source_tool": source_tool,
            "target_tool": target_tool,
            "val_ratio": val_ratio,
            "seed": seed,
        }
        train_ds, val_ds = split_session_datasets(
            meta_df, ary_intensity, ary_energy, config=split_cfg
        )

        region_train_cfg = dict(train_cfg)

        if use_bo:
            if verbose:
                print(f"Running Ax Bayesian Optimization for {region}...")
            bo_res = optimize_baseline_hyperparameters(train_ds, val_ds, config=bo_cfg)
            bo_results[region] = bo_res
            best_params = bo_res["best_parameters"]
            region_train_cfg.update(best_params)
            if verbose:
                print(f"Ax optimal params for {region}: {best_params}")

        # Train model
        train_result = train_baseline_region(train_ds, val_ds, config=region_train_cfg)
        models[region] = train_result["model"]
        eval_results[region] = train_result["best_val_loss"]
        histories[region] = train_result["history"]

        if verbose:
            print(f"Region {region} training complete. Best Val Loss: {train_result['best_val_loss']:.6f}")

    # Generate predictions if requested
    predictions = None
    if predict_source:
        pred_cfg = {
            "source_tool": source_tool,
            "target_tool": target_tool,
        }
        predictions = predict_spectra(
            models=models,
            data=(ary_intensity, ary_energy, meta_df),
            config=pred_cfg,
        )

    return {
        "models": models,
        "evaluation": eval_results,
        "histories": histories,
        "bayesian_opt_results": bo_results,
        "predictions": predictions,
    }
