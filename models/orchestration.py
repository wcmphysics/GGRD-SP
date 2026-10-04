"""Main orchestration module executing the unified 8-step spectral transformation pipeline.

Workflow:
1. Data splitting (session-level train-val-test split applied across all regions).
2. Sliding window setup (if enabled, compute patch window and stride).
3. Dataset and data loader setup (source-referenced normalization).
4. Cost function definition (Normalized MSE with regional weights and L2 penalty).
5. Model definition (ResNet1D, ResidualUNet1D, ConventionalUNet1D).
6. Bayesian optimization setup (Ax hyperparameter search).
7. Model training and validation (with log10 progress reporting).
8. Test data prediction (sequence-to-sequence prediction, physical counts, zero clamping).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.bayesian_opt import optimize_model_hyperparameters
from models.cost import NormalizedMSELoss, format_loss_log10
from models.dataset import (
    SpectrumPairDataset,
    SpectrumPatchDataset,
    create_dataloaders,
    partition_measurement_sessions,
    split_session_datasets,
)
from models.inference import predict_spectra
from models.resnet import ResNet1D
from models.trainer import evaluate, evaluate_sliding_window, train_model_region
from models.unet import ConventionalUNet1D, ResidualUNet1D, UNet1D
from utility.patching import calculate_window_points


def instantiate_model(
    model_type: str,
    seq_len: int,
    config: dict[str, Any] | None = None,
) -> nn.Module:
    """Instantiate a neural network model by architecture name and sequence length.

    Parameters
    ----------
    model_type : str
        Architecture type: 'resnet', 'residual_unet', or 'unet'.
    seq_len : int
        Input sequence length in points (patch window or full spectrum).
    config : dict[str, Any] | None, optional
        Architecture-specific hyperparameters.

    Returns
    -------
    nn.Module
        Instantiated PyTorch neural network module.
    """
    m_type = model_type.lower()
    cfg = config or {}
    if m_type in ("resnet", "baseline"):
        return ResNet1D(n_points=seq_len, config=cfg)
    elif m_type in ("residual_unet", "unet_residual"):
        return ResidualUNet1D(n_points=seq_len, config=cfg)
    elif m_type == "unet":
        return ConventionalUNet1D(n_points=seq_len, config=cfg)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose 'resnet', 'residual_unet', or 'unet'."
        )


def run_spectral_pipeline(
    data: tuple[np.ndarray, np.ndarray, pd.DataFrame] | dict[str, Any] | pd.DataFrame,
    model_type: str = "resnet",
    use_sliding_window: bool = False,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute the unified 8-step end-to-end spectral transformation pipeline.

    Parameters
    ----------
    data : tuple[np.ndarray, np.ndarray, pd.DataFrame] | dict[str, Any] | pd.DataFrame
        Measured data containers:
        - tuple: (ary_intensity, ary_energy, meta_df),
        - dict: {'ary_intensity': ..., 'ary_energy': ..., 'meta_df': ...}, or
        - DataFrame: meta_df with ary_intensity and ary_energy passed in kwargs.
    model_type : str, optional
        Architecture: 'resnet' (default), 'residual_unet', or 'unet'.
    use_sliding_window : bool, optional
        Whether to train using sliding window patch data augmentation (default False).
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'source_tool' (str): Source tool identifier (default 'J4').
        - 'target_tool' (str): Target tool identifier (default 'H1').
        - 'train_ratio' (float): Training session fraction (default 0.5).
        - 'val_ratio' (float): Validation session fraction (default 0.2).
        - 'test_ratio' (float): Held-out test session fraction (default 0.3).
        - 'seed' (int | None): Global random seed (default 42).
        - 'regions' (list[str] | None): Specific spectral regions to train (default auto-detected).
        - 'use_bayesian_opt' (bool): Whether to run Ax Bayesian optimization (default False).
        - 'bayesian_opt_config' (dict[str, Any]): Ax hyperparameter search configuration.
        - 'train_config' (dict[str, Any]): Regional training configuration (epochs, lr, batch_size, etc.).
        - 'window_size_ev' (float): Patch window size in eV if sliding window enabled (default 2.0).
        - 'sliding_stride_ev' (float): Patch stride in eV if sliding window enabled (default 1.0).
        - 'window_size' | 'window_size_points' (int): Explicit patch points override.
        - 'stride' | 'sliding_stride_points' (int): Explicit stride points override.
        - 'predict_source' (bool): Whether to predict target spectra for source sessions (default True).
        - 'verbose' (bool): Whether to log detailed progress (default False).
    **kwargs : Any
        Backward-compatibility keyword arguments.

    Returns
    -------
    dict[str, Any]
        Pipeline results dictionary containing:
        - 'models': Dict mapping region -> trained model (or (model, w_size, stride) tuple).
        - 'evaluation': Dict mapping region -> best validation loss (Normalized MSE).
        - 'test_evaluation': Dict mapping region -> test loss on held-out sessions.
        - 'session_splits': Dict with sets of 'train', 'val', and 'test' measurement IDs.
        - 'histories': Dict mapping region -> epoch loss histories.
        - 'bayesian_opt_results': Dict mapping region -> Ax optimization outputs.
        - 'predictions': Tuple (ary_intensity_pred, ary_energy_pred, meta_df_pred).
    """
    # Unpack data container
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
        raise ValueError(
            "run_spectral_pipeline requires meta_df, ary_intensity, and ary_energy."
        )

    cfg = dict(config or {})
    m_type: str = str(cfg.get("model_type", model_type)).lower()
    use_sw: bool = bool(cfg.get("use_sliding_window", use_sliding_window))

    source_tool: str = str(cfg.get("source_tool", "J4"))
    target_tool: str = str(cfg.get("target_tool", "H1"))
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

    # Auto-detect regions if not explicitly provided
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

    # =========================================================================
    # STEP 1: DATA SPLITTING (Session-level split applied across all regions)
    # =========================================================================
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
        print(f"Step 1: Session partition -> {n_tr} train, {n_va} val, {n_te} test sessions.")

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
        train_full_ds, val_full_ds, test_full_ds = split_session_datasets(
            meta_df, ary_intensity, ary_energy, config=split_cfg
        )

        sample_item = train_full_ds[0]
        full_len = len(sample_item["x"])

        # =====================================================================
        # STEP 2: SLIDING WINDOW SETUP (if enabled)
        # =====================================================================
        if use_sw:
            sample_energy = sample_item.get("energy")
            if sample_energy is not None:
                e_grid = sample_energy.cpu().numpy()
            else:
                e_grid = np.linspace(0.0, 10.0, full_len)

            sw_calc_cfg = {
                "window_size_ev": cfg.get("window_size_ev", 2.0),
                "sliding_stride_ev": cfg.get("sliding_stride_ev", 1.0),
                "window_size_points": cfg.get("window_size_points", cfg.get("window_size")),
                "sliding_stride_points": cfg.get("sliding_stride_points", cfg.get("stride")),
            }
            w_size, s_step = calculate_window_points(e_grid, config=sw_calc_cfg)
            w_size = min(full_len, max(3, w_size))
            s_step = max(1, min(w_size, s_step))
            seq_len = w_size
        else:
            w_size = full_len
            s_step = full_len
            seq_len = full_len

        # =====================================================================
        # STEP 3: DATASET AND DATA LOADER SETUP
        # =====================================================================
        # Note: train_full_ds and val_full_ds already implement source-referenced
        # normalization (dividing by max(|x|)). For patch training, SpectrumPatchDataset
        # extracts patches and inherits normalized scale.
        if use_sw:
            train_ds: Dataset = SpectrumPatchDataset(
                train_full_ds, window_size=w_size, stride=s_step
            )
            val_ds: Dataset = val_full_ds
        else:
            train_ds = train_full_ds
            val_ds = val_full_ds

        # =====================================================================
        # STEP 4: COST FUNCTION DEFINITION
        # =====================================================================
        l2_weight = float(train_cfg.get("l2_weight", 1e-4))
        region_weights = cfg.get("region_weights")
        criterion = NormalizedMSELoss(l2_weight=l2_weight, region_weights=region_weights)

        # =====================================================================
        # STEP 5: MODEL DEFINITION
        # =====================================================================
        region_train_cfg = dict(train_cfg)
        region_train_cfg.update(
            {
                "model_type": m_type,
                "use_sliding_window": use_sw,
                "window_size": w_size,
                "stride": s_step,
                "n_points": seq_len,
            }
        )
        model = instantiate_model(m_type, seq_len=seq_len, config=region_train_cfg)
        region_train_cfg["model"] = model

        # =====================================================================
        # STEP 6: BAYESIAN OPTIMIZATION SETUP
        # =====================================================================
        if use_bo:
            if verbose:
                print(f"Running Ax Bayesian Optimization for {region}...")
            region_bo_cfg = dict(bo_cfg)
            region_bo_cfg.update(
                {
                    "model_type": m_type,
                    "use_sliding_window": use_sw,
                    "window_size": w_size,
                    "stride": s_step,
                }
            )
            # When sliding window is active, pass train_full_ds to Ax so trials can slice
            # candidate window_sizes without nesting or clamping
            bo_res = optimize_model_hyperparameters(
                train_dataset=train_full_ds if use_sw else train_ds,
                val_dataset=val_ds,
                config=region_bo_cfg,
            )
            bo_results[region] = bo_res
            best_params = bo_res.get("best_parameters", {})
            if best_params:
                region_train_cfg.update(best_params)
                if use_sw and "window_size" in best_params:
                    w_size = int(best_params["window_size"])
                    s_step = max(1, w_size // 2)
                    seq_len = w_size
                    train_ds = SpectrumPatchDataset(train_full_ds, window_size=w_size, stride=s_step)
                    region_train_cfg["window_size"] = w_size
                    region_train_cfg["stride"] = s_step

                # Re-instantiate model with optimal parameters if architecture parameters changed
                if any(k in best_params for k in ("kernel_size", "hidden_channels", "base_channels", "depth", "window_size")):
                    model = instantiate_model(m_type, seq_len=seq_len, config=region_train_cfg)
                    region_train_cfg["model"] = model
            if verbose:
                print(f"Ax optimal params for {region}: {best_params}")

        # =====================================================================
        # STEP 7: MODEL TRAINING AND VALIDATION
        # =====================================================================
        if verbose:
            region_train_cfg["verbose"] = True

        train_result = train_model_region(train_ds, val_ds, config=region_train_cfg)
        trained_model = train_result["model"]

        if use_sw:
            models[region] = (trained_model, w_size, s_step)
        else:
            models[region] = trained_model

        best_val = float(train_result["best_val_loss"])
        eval_results[region] = best_val
        histories[region] = train_result["history"]

        # Evaluate model generalization on held-out test dataset
        if len(test_full_ds) > 0:
            if use_sw:
                t_loss = evaluate_sliding_window(
                    trained_model,
                    test_full_ds,
                    criterion=criterion,
                    config={"window_size": w_size, "stride": s_step},
                )
            else:
                _, test_loader = create_dataloaders(
                    test_full_ds,
                    test_full_ds,
                    config={"batch_size": region_train_cfg.get("batch_size", 16), "shuffle_train": False},
                )
                t_loss = evaluate(trained_model, test_loader, criterion)
            test_eval_results[region] = float(t_loss)
            if verbose:
                t_str = format_loss_log10(t_loss)
                print(f"Region {region} test loss: {t_str}")

        if verbose:
            b_str = format_loss_log10(best_val)
            print(f"Region {region} training complete. Best Val Loss: {b_str}")

    # =========================================================================
    # STEP 8: TEST DATA PREDICTION
    # =========================================================================
    predictions = None
    if predict_source:
        pred_cfg = {
            "source_tool": source_tool,
            "target_tool": target_tool,
            "session_splits": session_splits,
            "use_sliding_window": use_sw,
            "normalize_by_source": True,
            "clamp_non_negative": True,
        }
        predictions = predict_spectra(
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


def run_resnet_pipeline(
    meta_df: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute the end-to-end ResNet modeling pipeline."""
    cfg = dict(config or {})
    cfg.setdefault("model_type", "resnet")
    return run_spectral_pipeline(
        data=(ary_intensity, ary_energy, meta_df),
        model_type="resnet",
        use_sliding_window=bool(cfg.get("use_sliding_window", False)),
        config=cfg,
        **kwargs,
    )


def run_residual_unet_pipeline(
    meta_df: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute the end-to-end Residual U-Net modeling pipeline."""
    cfg = dict(config or {})
    cfg.setdefault("model_type", "residual_unet")
    return run_spectral_pipeline(
        data=(ary_intensity, ary_energy, meta_df),
        model_type="residual_unet",
        use_sliding_window=bool(cfg.get("use_sliding_window", False)),
        config=cfg,
        **kwargs,
    )


def run_unet_pipeline(
    meta_df: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute the end-to-end Conventional U-Net modeling pipeline."""
    cfg = dict(config or {})
    cfg.setdefault("model_type", "unet")
    return run_spectral_pipeline(
        data=(ary_intensity, ary_energy, meta_df),
        model_type="unet",
        use_sliding_window=bool(cfg.get("use_sliding_window", False)),
        config=cfg,
        **kwargs,
    )


def run_sliding_window_pipeline(
    meta_df: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute the sliding window modeling pipeline (ResNet with patch training)."""
    cfg = dict(config or {})
    cfg["use_sliding_window"] = True
    cfg.setdefault("model_type", "resnet")
    return run_spectral_pipeline(
        data=(ary_intensity, ary_energy, meta_df),
        model_type="resnet",
        use_sliding_window=True,
        config=cfg,
        **kwargs,
    )


# Backward compatibility wrappers
run_baseline_pipeline = run_resnet_pipeline


def run_model_pipeline(
    data: tuple[np.ndarray, np.ndarray, pd.DataFrame] | dict[str, Any] | pd.DataFrame,
    train_region_fn: Callable[..., Any] | None = None,
    predict_fn: Callable[..., Any] | None = None,
    config: dict[str, Any] | None = None,
    hooks: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Backward compatibility wrapper redirecting to run_spectral_pipeline."""
    from models.root import run_model_pipeline as _legacy_run

    return _legacy_run(
        data=data,
        train_region_fn=train_region_fn,
        predict_fn=predict_fn,
        config=config,
        hooks=hooks,
        **kwargs,
    )
