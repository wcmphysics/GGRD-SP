"""Main orchestration module executing the unified 8-step spectral
transformation pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch.nn as nn

from models.bayesian_opt import optimize_model_hyperparameters
from models.cost import CompositeSpectralLoss, NormalizedMSELoss
from models.dataset import (
    SpectrumPatchDataset,
    create_dataloaders,
    partition_measurement_sessions,
    split_session_datasets,
)
from models.deeplabv3 import DeepLabV3
from models.inference import predict_spectra
from models.resnet import ResNet1D
from models.trainer import evaluate, evaluate_sliding_window, train_model_region
from models.unet import ConventionalUNet1D, ResidualUNet1D
from utility.patching import calculate_window_points


def instantiate_model(model_type: str, seq_len: int, config: dict[str, Any] | None = None) -> nn.Module:
    """Instantiate a neural network model by architecture name and
    sequence length.

    Parameters
    ----------
    model_type : str
        Architecture type: 'resnet', 'residual_unet', 'unet', or
        'deeplabv3'.
    seq_len : int
        Input sequence length in points.
    config : dict[str, Any] | None, optional
        Architecture hyperparameters.

    Returns
    -------
    nn.Module
        Instantiated neural network module.
    """
    m_map = dict(resnet=ResNet1D, residual_unet=ResidualUNet1D, unet=ConventionalUNet1D, deeplabv3=DeepLabV3)
    m_cls = m_map.get(model_type.lower())
    if m_cls is None:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose 'resnet', 'residual_unet', 'unet', or 'deeplabv3'."
        )
    return m_cls(n_points=seq_len, config=config or {})


def _normalize_config_aliases(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize hyperparameter keys and aliases within a configuration
    dictionary.
    """
    res = dict(config_dict)
    alias_groups = {
        "epochs": ("epoch", "num_epochs", "n_epochs", "max_epochs"),
        "learning_rate": ("lr",), "batch_size": ("batch", "batchsize"),
        "early_stopping_patience": ("patience", "early_stop_patience"),
        "l2_weight": ("l2", "l2_reg", "l2_penalty", "weight_decay_l2"),
        "model_type": ("model", "architecture"),
        "use_sliding_window": ("sliding_window", "use_sw"),
        "use_bayesian_opt": ("bayesian_opt", "use_bo"), "use_film": ("film",),
        "sliding_stride_ev": ("stride_ev",), "sliding_stride_points": ("stride_points", "stride"),
        "window_size_points": ("window_points", "window_size"), "cost_type": ("loss_type",),
        "use_lr_scheduler": ("use_reduce_lr_on_plateau", "reduce_lr_on_plateau", "use_scheduler", "lr_scheduler"),
        "lr_reduce_factor": ("lr_factor", "factor", "lr_scheduler_factor", "reduction_factor"),
        "lr_scheduler_patience": ("lr_patience", "scheduler_patience"),
        "min_lr": ("lr_min", "lr_scheduler_min_lr"),
    }
    for canon, aliases in alias_groups.items():
        keys = (canon,) + aliases
        hit = next((k for k in keys if k in res), None)
        if hit is not None:
            val = res[hit]
            res.update({k: val for k in keys})
    return res


def _resolve_region_config(
    global_cfg: dict[str, Any], region: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve regional sliding window, training, and Bayesian
    optimization configurations.
    """
    r_cfgs = global_cfg.get("region_configs") or global_cfg.get("region_config") or {}
    reg = dict(r_cfgs.get(region, {})) if isinstance(r_cfgs, dict) and region in r_cfgs else {}

    def _pick(k1: str, k2: str, def_v: Any = None) -> Any:
        return reg.get(k1, reg.get(k2, global_cfg.get(k1, global_cfg.get(k2, def_v))))

    sw_calc_cfg = {
        "window_size_ev": reg.get("window_size_ev", global_cfg.get("window_size_ev", 2.0)),
        "sliding_stride_ev": _pick("sliding_stride_ev", "stride_ev", 1.0),
        "window_size_points": _pick("window_size_points", "window_size"),
        "sliding_stride_points": _pick("sliding_stride_points", "stride"),
    }

    g_train = global_cfg.get("train_config") or global_cfg.get("training_config") or global_cfg.get("train_cfg") or {}
    base_train = _normalize_config_aliases(dict(g_train))
    n_train = reg.get("train_config") or reg.get("training_config") or reg.get("train_cfg") or {}
    if isinstance(n_train, dict):
        base_train.update(_normalize_config_aliases(n_train))

    excl = {
        "train_config", "training_config", "train_cfg", "bayesian_opt_config", "bayesian_opt_configs",
        "bo_config", "bo_cfg", "window_size_ev", "sliding_stride_ev", "stride_ev",
    }
    flat_ovr = {k: v for k, v in reg.items() if k not in excl}
    if flat_ovr:
        base_train.update(_normalize_config_aliases(flat_ovr))
    base_train = _normalize_config_aliases(base_train)

    g_bo = global_cfg.get("bayesian_opt_config") or global_cfg.get("bo_config") or global_cfg.get("bo_cfg") or {}
    base_bo = _normalize_config_aliases(dict(g_bo))
    n_bo = reg.get("bayesian_opt_config") or reg.get("bo_config") or reg.get("bo_cfg") or {}
    if isinstance(n_bo, dict):
        base_bo.update(_normalize_config_aliases(n_bo))

    bo_keys = (
        "batch_sizes", "kernel_sizes", "window_size_choices", "lr_bounds", "l2_bounds",
        "num_trials", "epochs_per_trial", "seed", "verbose",
    )
    for k in bo_keys:
        if k in reg:
            base_bo[k] = reg[k]
    for arch in ("hidden_channels", "base_channels", "depths", "backbone_channels", "aspp_channels"):
        if arch in reg and isinstance(reg[arch], (list, tuple)):
            base_bo[arch] = list(reg[arch])

    return sw_calc_cfg, base_train, _normalize_config_aliases(base_bo)


def run_spectral_pipeline(
    data: tuple[np.ndarray, np.ndarray, pd.DataFrame] | dict[str, Any] | pd.DataFrame,
    model_type: str = "resnet",
    use_sliding_window: bool = False,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute the unified 8-step end-to-end spectral transformation
    pipeline.

    Parameters
    ----------
    data : tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Measured data canonical tuple
        (ary_intensity, ary_energy, meta_df).
    model_type : str, optional
        Architecture: 'resnet' (default), 'residual_unet', 'unet', or
        'deeplabv3'.
    use_sliding_window : bool, optional
        Whether to train using sliding window patch data augmentation.
    config : dict[str, Any] | None, optional
        Pipeline configuration dictionary.

    Returns
    -------
    dict[str, Any]
        Pipeline results dictionary containing trained models,
        evaluations, histories, and predictions.
    """
    if not isinstance(data, (tuple, list)) or len(data) != 3:
        raise ValueError(
            "run_spectral_pipeline requires data as canonical tuple: (ary_intensity, ary_energy, meta_df)."
        )
    ary_intensity, ary_energy, meta_df = data[0], data[1], data[2]
    if (not isinstance(ary_intensity, np.ndarray) or not isinstance(ary_energy, np.ndarray)
            or not isinstance(meta_df, pd.DataFrame)):
        raise ValueError(
            "run_spectral_pipeline expects data tuple elements of type: (np.ndarray, np.ndarray, pd.DataFrame)."
        )

    cfg = dict(config or {})
    m_type = str(cfg.get("model_type", model_type)).lower()
    use_sw = bool(cfg.get("use_sliding_window", use_sliding_window))
    source_tool, target_tool = str(cfg.get("source_tool", "J4")), str(cfg.get("target_tool", "H1"))
    ratios = (float(cfg.get("train_ratio", 0.5)), float(cfg.get("val_ratio", 0.2)), float(cfg.get("test_ratio", 0.3)))
    seed, verbose = cfg.get("seed", 42), bool(cfg.get("verbose", False))
    use_bo, bo_cfg = bool(cfg.get("use_bayesian_opt", False)), dict(cfg.get("bayesian_opt_config", {}))
    if "seed" not in bo_cfg and seed is not None:
        bo_cfg["seed"] = seed

    regions = cfg.get("regions")
    if regions is None:
        mask = meta_df["measurement_id_target"].notna()
        if source_tool:
            mask = mask & (meta_df["tool"] == source_tool)
        regions = sorted(meta_df[mask]["region"].unique().tolist())
    if not regions:
        raise ValueError(f"No valid spectral regions found to train for {source_tool=}, {target_tool=}")

    session_splits = cfg.get("session_splits")
    if session_splits is None:
        p_cfg = {
            "source_tool": source_tool, "target_tool": target_tool,
            "train_ratio": ratios[0], "val_ratio": ratios[1], "test_ratio": ratios[2], "seed": seed,
        }
        session_splits = partition_measurement_sessions(meta_df, config=p_cfg)

    models, eval_results, test_eval_results, histories, bo_results = {}, {}, {}, {}, {}

    for region in regions:
        sw_calc_cfg, r_train_cfg, r_bo_cfg = _resolve_region_config(cfg, region)
        reg_use_sw = bool(r_train_cfg.get("use_sliding_window", use_sw))
        reg_m_type = str(r_train_cfg.get("model_type", m_type)).lower()
        reg_use_film = bool(r_train_cfg.get("use_film", cfg.get("use_film", False)))
        reg_use_bo = bool(r_bo_cfg.get("use_bayesian_opt", r_train_cfg.get("use_bayesian_opt", use_bo)))

        split_cfg = {
            "region": region, "source_tool": source_tool, "target_tool": target_tool,
            "session_splits": session_splits, "return_test": True,
            "normalization_mode": cfg.get("normalization_mode", "source_referenced"),
        }
        train_full_ds, val_full_ds, test_full_ds = split_session_datasets(
            meta_df, ary_intensity, ary_energy, config=split_cfg,
        )
        full_len = len(train_full_ds[0]["x"])

        if reg_use_sw:
            e_samp = train_full_ds[0].get("energy")
            e_grid = e_samp.cpu().numpy() if e_samp is not None else np.linspace(0.0, 10.0, full_len)
            w_size, s_step = calculate_window_points(e_grid, config=sw_calc_cfg)
            w_size, s_step = min(full_len, max(3, w_size)), max(1, min(w_size, s_step))
            seq_len, train_ds = w_size, SpectrumPatchDataset(train_full_ds, window_size=w_size, stride=s_step)
        else:
            w_size = s_step = seq_len = full_len
            train_ds = train_full_ds

        l2_w = float(r_train_cfg.get("l2_weight", 1e-4))
        c_type = str(r_train_cfg.get("cost_type", cfg.get("cost_type", "normalized_mse"))).lower()
        cost_kw = dict(l2_weight=l2_w, region_weights=cfg.get("region_weights"))
        loss_cls = CompositeSpectralLoss if "composite" in c_type else NormalizedMSELoss
        def_mse, def_shape = (0.5, 0.5) if "composite" in c_type else (1.0, 0.0)
        criterion = loss_cls(
            w_mse=float(r_train_cfg.get("w_mse", cfg.get("w_mse", def_mse))),
            w_shape=float(r_train_cfg.get("w_shape", cfg.get("w_shape", def_shape))),
            w_scale=float(r_train_cfg.get("w_scale", cfg.get("w_scale", 0.0))), **cost_kw,
        )

        r_train_cfg.update({
            "model_type": reg_m_type, "use_sliding_window": reg_use_sw,
            "window_size": w_size, "stride": s_step, "n_points": seq_len, "use_film": reg_use_film,
        })
        model = instantiate_model(reg_m_type, seq_len=seq_len, config=r_train_cfg)
        r_train_cfg.update(dict(model=model, criterion=criterion))

        if reg_use_bo:
            r_bo_cfg.update({
                "model_type": reg_m_type, "use_sliding_window": reg_use_sw,
                "window_size": w_size, "stride": s_step, "criterion": criterion,
            })
            bo_res = optimize_model_hyperparameters(
                train_full_ds if reg_use_sw else train_ds, val_full_ds, config=r_bo_cfg,
            )
            bo_results[region] = bo_res
            best_p = bo_res.get("best_parameters", {})
            if best_p:
                r_train_cfg.update(best_p)
                if reg_use_sw and "window_size" in best_p:
                    w_size = int(best_p["window_size"])
                    s_step = max(1, w_size // 2)
                    seq_len = w_size
                    train_ds = SpectrumPatchDataset(train_full_ds, window_size=w_size, stride=s_step)
                    r_train_cfg.update({"window_size": w_size, "stride": s_step})
                arch_keys = (
                    "kernel_size", "hidden_channels", "base_channels", "depth", "backbone_channels",
                    "aspp_channels", "window_size",
                )
                if any(k in best_p for k in arch_keys):
                    model = instantiate_model(reg_m_type, seq_len=seq_len, config=r_train_cfg)
                    r_train_cfg["model"] = model

        if verbose:
            r_train_cfg["verbose"] = True
        train_res = train_model_region(train_ds, val_full_ds, config=r_train_cfg)
        trained_m = train_res["model"]
        models[region] = (trained_m, w_size, s_step) if reg_use_sw else trained_m
        eval_results[region] = float(train_res["best_val_loss"])
        histories[region] = train_res["history"]

        if len(test_full_ds) > 0:
            if reg_use_sw:
                t_loss = evaluate_sliding_window(
                    trained_m, test_full_ds, criterion=criterion, config={"window_size": w_size, "stride": s_step},
                )
            else:
                _, t_loader = create_dataloaders(
                    test_full_ds, test_full_ds,
                    config={"batch_size": r_train_cfg.get("batch_size", 16), "shuffle_train": False},
                )
                t_loss = evaluate(trained_m, t_loader, criterion)
            test_eval_results[region] = float(t_loss)

    predictions = None
    if cfg.get("predict_source", True):
        p_cfg = {
            "source_tool": source_tool, "target_tool": target_tool, "session_splits": session_splits,
            "normalization_mode": cfg.get("normalization_mode", "source_referenced"), "clamp_non_negative": True,
        }
        predictions = predict_spectra(models=models, data=(ary_intensity, ary_energy, meta_df), config=p_cfg)

    return {
        "models": models, "evaluation": eval_results, "test_evaluation": test_eval_results,
        "session_splits": session_splits, "histories": histories, "bayesian_opt_results": bo_results,
        "predictions": predictions,
    }


def run_resnet_pipeline(
    meta_df: pd.DataFrame, ary_intensity: np.ndarray, ary_energy: np.ndarray,
    config: dict[str, Any] | None = None, **kwargs: Any,
) -> dict[str, Any]:
    """Execute the end-to-end ResNet modeling pipeline."""
    cfg = {**(config or {}), "model_type": "resnet"}
    return run_spectral_pipeline((ary_intensity, ary_energy, meta_df), "resnet", config=cfg, **kwargs)


def run_residual_unet_pipeline(
    meta_df: pd.DataFrame, ary_intensity: np.ndarray, ary_energy: np.ndarray,
    config: dict[str, Any] | None = None, **kwargs: Any,
) -> dict[str, Any]:
    """Execute the end-to-end Residual U-Net modeling pipeline."""
    cfg = {**(config or {}), "model_type": "residual_unet"}
    return run_spectral_pipeline((ary_intensity, ary_energy, meta_df), "residual_unet", config=cfg, **kwargs)


def run_unet_pipeline(
    meta_df: pd.DataFrame, ary_intensity: np.ndarray, ary_energy: np.ndarray,
    config: dict[str, Any] | None = None, **kwargs: Any,
) -> dict[str, Any]:
    """Execute the end-to-end Conventional U-Net modeling pipeline."""
    cfg = {**(config or {}), "model_type": "unet"}
    return run_spectral_pipeline((ary_intensity, ary_energy, meta_df), "unet", config=cfg, **kwargs)


def run_deeplabv3_pipeline(
    meta_df: pd.DataFrame, ary_intensity: np.ndarray, ary_energy: np.ndarray,
    config: dict[str, Any] | None = None, **kwargs: Any,
) -> dict[str, Any]:
    """Execute the end-to-end DeepLabV3 modeling pipeline."""
    cfg = {**(config or {}), "model_type": "deeplabv3"}
    return run_spectral_pipeline((ary_intensity, ary_energy, meta_df), "deeplabv3", config=cfg, **kwargs)
