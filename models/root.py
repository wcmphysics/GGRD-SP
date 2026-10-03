"""Root function orchestrating neural network training, hyperparameter search, and prediction."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from models.baseline_cnn import NormalizedMSELoss
from models.dataset import (
    create_dataloaders,
    partition_measurement_sessions,
    split_session_datasets,
)
from models.trainer import evaluate


def run_model_pipeline(
    data: tuple[np.ndarray, np.ndarray, pd.DataFrame] | dict[str, Any] | pd.DataFrame,
    train_region_fn: Callable[[Any, Any, dict[str, Any]], dict[str, Any]],
    predict_fn: Callable[..., Any],
    config: dict[str, Any] | None = None,
    hooks: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute a generalized end-to-end spectral transformation pipeline.

    Unifies session-level partitioning, optional Bayesian hyperparameter optimization,
    regional neural network training, and standardized prediction output.

    Parameters
    ----------
    data : tuple[np.ndarray, np.ndarray, pd.DataFrame] | dict[str, Any] | pd.DataFrame
        Dataset containers. Accepts (ary_intensity, ary_energy, meta_df) tuple,
        a dictionary with 'meta_df', 'ary_intensity', 'ary_energy', or meta_df DataFrame.
    train_region_fn : Callable
        Function training a single region model with signature (train_ds, val_ds, config=...).
    predict_fn : Callable
        Function generating predictions with signature (models=..., data=..., config=...).
    config : dict[str, Any] | None, optional
        Configuration dictionary.
    hooks : dict[str, Any] | None, optional
        Dictionary packaging lifecycle callback hooks:
        - 'bo_fn': Function running Bayesian hyperparameter optimization.
        - 'model_record_fn': Transforms training result into object stored in models dict.
        - 'dataset_prep_fn': Hook to transform train/val datasets before training.
    **kwargs : Any
        Backward-compatibility keyword arguments.

    Returns
    -------
    dict[str, Any]
        Dictionary with 'models', 'evaluation', 'histories', 'bayesian_opt_results', and 'predictions'.
    """
    if isinstance(data, pd.DataFrame):
        meta_df = data
        ary_intensity = kwargs.get("ary_intensity")
        ary_energy = kwargs.get("ary_energy")
    elif isinstance(data, (tuple, list)) and len(data) == 3:
        if isinstance(data[0], pd.DataFrame):
            meta_df, ary_intensity, ary_energy = data[0], data[1], data[2]
        else:
            ary_intensity, ary_energy, meta_df = data[0], data[1], data[2]
    elif isinstance(data, dict):
        meta_df = data.get("meta_df", data.get("metadata"))
        ary_intensity = data.get("ary_intensity", data.get("intensity"))
        ary_energy = data.get("ary_energy", data.get("energy"))
    else:
        meta_df = kwargs.get("meta_df")
        ary_intensity = kwargs.get("ary_intensity")
        ary_energy = kwargs.get("ary_energy")

    if meta_df is None or ary_intensity is None or ary_energy is None:
        raise ValueError("run_model_pipeline requires meta_df, ary_intensity, and ary_energy to be provided.")

    h = dict(hooks or {})
    bo_fn = h.get("bo_fn", kwargs.get("bo_fn"))
    model_record_fn = h.get("model_record_fn", kwargs.get("model_record_fn"))
    dataset_prep_fn = h.get("dataset_prep_fn", kwargs.get("dataset_prep_fn"))
    cfg = config or {}
    source_tool: str = cfg.get("source_tool", "J4")
    target_tool: str = cfg.get("target_tool", "H1")
    train_ratio: float = float(cfg.get("train_ratio", 0.5))
    val_ratio: float = float(cfg.get("val_ratio", 0.2))
    test_ratio: float = float(cfg.get("test_ratio", 0.3))
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

    # Unified global session partitioning across all spectral regions
    session_splits: dict[str, set[str]] | None = cfg.get("session_splits")
    if session_splits is None:
        partition_cfg = {
            "source_tool": source_tool,
            "target_tool": target_tool,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
        }
        session_splits = partition_measurement_sessions(meta_df, config=partition_cfg)

    if verbose:
        n_tr = len(session_splits.get("train", set()))
        n_va = len(session_splits.get("val", set()))
        n_te = len(session_splits.get("test", set()))
        print(f"Unified session partition: {n_tr} train, {n_va} val, {n_te} test sessions.")

    models: dict[str, Any] = {}
    eval_results: dict[str, float] = {}
    test_eval_results: dict[str, float] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    bo_results: dict[str, dict[str, Any]] = {}

    for region in regions:
        if verbose:
            print(f"\n--- Processing Region: {region} ({source_tool} -> {target_tool}) ---")

        split_cfg = {
            "region": region,
            "source_tool": source_tool,
            "target_tool": target_tool,
            "session_splits": session_splits,
            "return_test": True,
        }
        train_ds, val_ds, test_ds = split_session_datasets(
            meta_df, ary_intensity, ary_energy, config=split_cfg
        )

        region_train_cfg = dict(train_cfg)

        if dataset_prep_fn is not None:
            train_ds, val_ds, region_train_cfg = dataset_prep_fn(
                train_ds, val_ds, region_train_cfg
            )

        if use_bo and bo_fn is not None:
            if verbose:
                print(f"Running Ax Bayesian Optimization for {region}...")
            bo_res = bo_fn(train_ds, val_ds, config=bo_cfg)
            bo_results[region] = bo_res
            best_params = bo_res.get("best_parameters", {})
            if best_params:
                region_train_cfg.update(best_params)
            if verbose:
                print(f"Ax optimal params for {region}: {best_params}")

        # Train model for this region
        train_result = train_region_fn(train_ds, val_ds, config=region_train_cfg)
        if model_record_fn is not None:
            models[region] = model_record_fn(train_result, region_train_cfg)
        else:
            models[region] = train_result["model"]

        eval_results[region] = float(train_result["best_val_loss"])
        histories[region] = train_result["history"]

        # Evaluate model generalization on held-out test dataset
        if len(test_ds) > 0:
            trained_model_entry = models[region]
            if isinstance(trained_model_entry, (tuple, list)):
                from models.sliding_window import evaluate_sliding_window

                sw_model, sw_w, sw_s = trained_model_entry[0], trained_model_entry[1], trained_model_entry[2]
                t_loss = evaluate_sliding_window(
                    sw_model, test_ds, criterion=NormalizedMSELoss(), window_size=sw_w, stride=sw_s
                )
            else:
                raw_m = trained_model_entry if isinstance(trained_model_entry, torch.nn.Module) else train_result["model"]
                _, test_loader = create_dataloaders(
                    test_ds, test_ds, config={"batch_size": region_train_cfg.get("batch_size", 16), "shuffle_train": False}
                )
                t_loss = evaluate(raw_m, test_loader, NormalizedMSELoss())
            test_eval_results[region] = float(t_loss)
            if verbose:
                print(f"Region {region} test loss: {t_loss:.6f}")

        if verbose:
            print(f"Region {region} training complete. Best Val Loss: {train_result['best_val_loss']:.6f}")

    # Generate predictions if requested
    predictions = None
    if predict_source:
        pred_cfg = {
            "source_tool": source_tool,
            "target_tool": target_tool,
            "session_splits": session_splits,
        }
        predictions = predict_fn(
            models=models,
            data=(ary_intensity, ary_energy, meta_df),
            config=pred_cfg,
        )

    return {
        "models": models,
        "evaluation": eval_results,
        "test_evaluation": test_eval_results,
        "session_splits": session_splits,
        "histories": histories,
        "bayesian_opt_results": bo_results,
        "predictions": predictions,
    }


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
    """
    from models.bayesian_opt import optimize_baseline_hyperparameters
    from models.inference import predict_spectra
    from models.trainer import train_baseline_region

    return run_model_pipeline(
        data=(ary_intensity, ary_energy, meta_df),
        train_region_fn=train_baseline_region,
        predict_fn=predict_spectra,
        config=config,
        hooks={"bo_fn": optimize_baseline_hyperparameters},
    )
