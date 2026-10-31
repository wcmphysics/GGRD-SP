"""Model training and Bayesian optimization orchestration module for GGRD-SP."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.bayesian_opt import optimize_model_hyperparameters
from models.dataset import SpectrumPairDataset, SpectrumPatchDataset
from models.trainer import instantiate_model, train_model_region

_CONTAINER_KEYS = {
    "train_config",
    "bayesian_opt_config",
    "film_config",
    "region_configs",
    "window_size_ev",
    "sliding_stride_ev",
}


def _resolve_region_config(
    global_config: dict[str, Any],
    region: str,
    bayesian_opt_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve per-region sliding window, training, and Bayesian optimization
    configurations.

    Parameters
    ----------
    global_config : dict[str, Any]
        Top-level configuration dictionary containing global defaults and
        optional 'region_configs' overrides.
    region : str
        Name of the XPS spectral region (e.g., 'Al2p', 'Ti2p').
    bayesian_opt_config : dict[str, Any] | None, optional
        Dedicated Bayesian optimization configuration dictionary.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
        3-tuple of (sliding_window_config, resolved_train_config,
        resolved_bo_config).
    """
    region_configs = global_config.get("region_configs") or {}
    region_override = (
        dict(region_configs.get(region, {}))
        if isinstance(region_configs, dict) and region in region_configs
        else {}
    )

    window_size_points = region_override.get(
        "window_size", global_config.get("window_size")
    )
    stride_points = region_override.get("stride", global_config.get("stride"))
    sliding_window_config = {
        "window_size_ev": region_override.get(
            "window_size_ev", global_config.get("window_size_ev", 6.0)
        ),
        "sliding_stride_ev": region_override.get(
            "sliding_stride_ev", global_config.get("sliding_stride_ev", 2.0)
        ),
        "window_size": window_size_points,
        "stride": stride_points,
    }

    top_level_train_defaults = {
        k: v for k, v in global_config.items() if k not in _CONTAINER_KEYS
    }
    flat_region_overrides = {
        k: v for k, v in region_override.items() if k not in _CONTAINER_KEYS
    }
    resolved_train_config = {
        **top_level_train_defaults,
        **dict(global_config.get("train_config") or {}),
        **dict(region_override.get("train_config") or {}),
        **flat_region_overrides,
    }

    base_bo_cfg = (
        bayesian_opt_config
        if bayesian_opt_config is not None
        else (global_config.get("bayesian_opt_config") or {})
    )
    resolved_bo_config = {
        **dict(base_bo_cfg),
        **dict(region_override.get("bayesian_opt_config") or {}),
    }

    return sliding_window_config, resolved_train_config, resolved_bo_config


_SHARED_WINDOW_HINT = (
    "A unified model needs one input length: remove per-region window "
    "overrides in 'dataset.region_configs', or set 'dataset.window_size' and "
    "'dataset.stride' in points so every region resolves to the same values."
)


def _require_shared_region_value(
    values_by_region: dict[str, int], description: str, hint: str
) -> int:
    """Return the value shared by all regions or raise a ValueError."""
    if not values_by_region:
        raise ValueError(
            f"Cannot determine shared {description} from an empty set of regions."
        )
    if len(set(values_by_region.values())) > 1:
        raise ValueError(
            f"unified_model=True requires every region to share the same "
            f"{description}, got {values_by_region}. {hint}"
        )
    return next(iter(values_by_region.values()))


def _merge_regional_train_patches(
    regional_datasets: dict[str, dict[str, Any]],
    train_full: SpectrumPairDataset,
    seq_len: int,
) -> tuple[SpectrumPairDataset | SpectrumPatchDataset, int, int]:
    """Merge per-region training patches that share one window and stride.

    Regions without training samples are ignored. Returns the merged train
    dataset with its window size and stride in points.
    """
    trained = {
        region: bundle
        for region, bundle in regional_datasets.items()
        if len(bundle.get("dataset_train_full", bundle["dataset_train"])) > 0
    }
    patched = [
        region
        for region, bundle in trained.items()
        if isinstance(bundle["dataset_train"], SpectrumPatchDataset)
    ]
    if not patched:
        return train_full, seq_len, seq_len

    unpatched = sorted(set(trained) - set(patched))
    if unpatched:
        raise ValueError(
            "unified_model=True requires sliding window to be either on or off "
            f"for all regions, but it is on for {patched} and off for "
            f"{unpatched}. Align 'use_sliding_window' in 'dataset.region_configs'."
        )
    w_pts = _require_shared_region_value(
        {r: int(trained[r]["window_size_points"]) for r in patched},
        "sliding window size in points",
        _SHARED_WINDOW_HINT,
    )
    s_pts = _require_shared_region_value(
        {r: int(trained[r]["sliding_stride_points"]) for r in patched},
        "sliding stride in points",
        _SHARED_WINDOW_HINT,
    )
    merged = SpectrumPatchDataset.concatenate(
        [trained[r]["dataset_train"] for r in patched]
    )
    return merged, w_pts, s_pts


def _combine_regional_bundles(
    regional_datasets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Merge multiple regional dataset bundles into a single unified bundle."""
    if not regional_datasets:
        raise ValueError("regional_datasets dictionary cannot be empty.")
    if "all" in regional_datasets:
        return regional_datasets["all"]

    train_full_items = [
        b.get("dataset_train_full", b["dataset_train"])
        for b in regional_datasets.values()
    ]
    val_items = [b["dataset_val"] for b in regional_datasets.values()]
    test_items = [
        b.get("dataset_test")
        for b in regional_datasets.values()
        if b.get("dataset_test") is not None
    ]
    seq_len = _require_shared_region_value(
        {r: int(d.x.shape[-1]) for r, d in zip(regional_datasets, train_full_items)},
        "spectrum length (N_points)",
        "Resample all regions onto a common number of points.",
    )

    def _merge_pair_datasets(
        ds_list: list[SpectrumPairDataset],
    ) -> SpectrumPairDataset:
        merged_x = torch.cat([d.x for d in ds_list], dim=0)
        merged_y = torch.cat([d.y for d in ds_list], dim=0)
        has_all_energy = all(d.energy is not None for d in ds_list)
        merged_energy = (
            torch.cat([d.energy for d in ds_list], dim=0)
            if has_all_energy
            else None
        )
        merged_meta: list[dict[str, Any]] = []
        for d in ds_list:
            merged_meta.extend(d.metadata)
        return SpectrumPairDataset(
            merged_x, merged_y, energy=merged_energy, metadata=merged_meta
        )

    train_full = _merge_pair_datasets(train_full_items)
    val_full = _merge_pair_datasets(val_items)
    test_full = _merge_pair_datasets(test_items) if test_items else None

    train_ds, w_pts, s_pts = _merge_regional_train_patches(
        regional_datasets, train_full, seq_len
    )

    return {
        "dataset_train": train_ds,
        "dataset_val": val_full,
        "dataset_test": test_full,
        "dataset_train_full": train_full,
        "window_size_points": w_pts,
        "sliding_stride_points": s_pts,
        "full_spectrum_length": seq_len,
    }


def perform_model_training_and_bayesian_optimization(
    regional_datasets: dict[str, dict[str, Any]],
    criterion: nn.Module,
    config: dict[str, Any] | None = None,
    bayesian_opt_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train regional spectral neural networks and optimize hyperparameters
    via Bayesian search.

    Parameters
    ----------
    regional_datasets : dict[str, dict[str, Any]]
        Dictionary mapping region names to their dataset bundles:
        - 'dataset_train': training dataset (SpectrumPatchDataset if sliding
          window on).
        - 'dataset_val': validation full-spectrum dataset.
        - 'dataset_train_full': unpatched full-spectrum training dataset.
        - 'window_size_points': window size in data points.
        - 'sliding_stride_points': sliding stride in data points.
        - 'full_spectrum_length': full spectrum sequence length.
    criterion : nn.Module
        Spectral loss criterion for model training.
    config : dict[str, Any] | None, optional
        Training configuration dictionary (singular hyperparameter keys):
        - 'model_type': architecture name ('resnet', 'resnet_scaled_short',
          'residual_unet', 'unet', 'deeplabv3').
        - 'use_film': whether to enable FiLM tool metadata conditioning
          (default False).
        - 'train_config': training parameters (epochs, learning_rate,
          batch_size, etc.).
        - 'region_configs': optional per-region overrides.
        - 'verbose': verbosity toggle (default False).
    bayesian_opt_config : dict[str, Any] | None, optional
        Dedicated Bayesian optimization configuration dictionary (plural keys):
        - 'use_bayesian_opt': whether to execute Ax Bayesian optimization
          (default False).
        - 'num_trials', 'epochs_per_trial', 'batch_sizes',
          'window_size_choices', 'kernel_sizes', 'depths', 'resnet_channels',
          'residual_unet_channels', 'unet_channels',
          'deeplabv3_backbone_channels', 'deeplabv3_aspp_channels',
          'learning_rate_bounds', 'l2_bounds'.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'models': dict mapping each region to its trained model (or tuple of
          (model, window_size, stride)).
        - 'val_evaluation': dict mapping each region to its best validation
          loss.
        - 'histories': dict mapping each region to its epoch training history.
        - 'bayesian_opt_results': dict mapping each region to Ax optimization
          results.
    """
    pipeline_config = dict(config or {})
    bo_cfg_input = (
        dict(bayesian_opt_config) if bayesian_opt_config is not None else None
    )
    default_model_type = str(
        pipeline_config.get("model_type", "resnet")
    ).lower()
    default_use_bayesian_opt = bool(
        (bo_cfg_input or {}).get(
            "use_bayesian_opt",
            pipeline_config.get("use_bayesian_opt", False),
        )
    )
    verbose = bool(pipeline_config.get("verbose", False))
    film_cfg = dict(pipeline_config.get("film_config") or {})

    trained_models: dict[str, Any] = {}
    val_evaluation_results: dict[str, float] = {}
    training_histories: dict[str, dict[str, list[float]]] = {}
    bayesian_opt_results: dict[str, dict[str, Any]] = {}

    is_unified = bool(pipeline_config.get("unified_model", False))
    if is_unified:
        film_cfg = dict(film_cfg)
        film_cfg.setdefault("encode_region", True)
        if "all" in regional_datasets:
            actual_num = len(regional_datasets["all"].get("regions", []))
        else:
            actual_num = len(regional_datasets)
        target_num_regions = max(
            16, actual_num, int(film_cfg.get("num_regions", 16))
        )
        film_cfg["num_regions"] = target_num_regions
        pipeline_config["film_config"] = film_cfg
        target_datasets = {"all": _combine_regional_bundles(regional_datasets)}
    else:
        target_datasets = regional_datasets

    for region, bundle in target_datasets.items():
        dataset_train = bundle["dataset_train"]
        dataset_val = bundle["dataset_val"]
        dataset_train_full = bundle.get("dataset_train_full", dataset_train)
        window_size_points = int(
            bundle.get(
                "window_size_points", bundle.get("full_spectrum_length", 100)
            )
        )
        stride_points = int(
            bundle.get("sliding_stride_points", window_size_points)
        )
        full_spectrum_length = int(
            bundle.get("full_spectrum_length", window_size_points)
        )

        _, region_train_config, region_bo_config = _resolve_region_config(
            pipeline_config, region, bayesian_opt_config=bo_cfg_input
        )
        region_model_type = str(
            region_train_config.get("model_type", default_model_type)
        ).lower()
        region_use_film = bool(
            region_train_config.get(
                "use_film",
                film_cfg.get("use_film", pipeline_config.get("use_film", False)),
            )
        )
        region_use_bayesian_opt = bool(
            region_bo_config.get(
                "use_bayesian_opt",
                region_train_config.get(
                    "use_bayesian_opt", default_use_bayesian_opt
                ),
            )
        )

        is_sw = isinstance(dataset_train, SpectrumPatchDataset)
        input_sequence_length = (
            window_size_points if is_sw else full_spectrum_length
        )

        for k, v in film_cfg.items():
            region_train_config.setdefault(k, v)
        region_train_config.update(
            {
                "model_type": region_model_type,
                "use_sliding_window": is_sw,
                "n_points": input_sequence_length,
                "use_film": region_use_film,
            }
        )
        region_train_config.setdefault("window_size", window_size_points)
        region_train_config.setdefault("stride", stride_points)

        if region_use_bayesian_opt:
            train_fixed_cfg = dict(region_train_config)
            bo_run_cfg = {
                **train_fixed_cfg,
                **film_cfg,
                **region_bo_config,
                "model_type": region_model_type,
                "use_sliding_window": is_sw,
                "stride": stride_points,
                "use_film": region_use_film,
                "criterion": criterion,
            }
            if is_sw and not (
                {"window_size_choices", "window_size", "window_size_choice"}
                & bo_run_cfg.keys()
            ):
                bo_run_cfg["window_size_choices"] = [window_size_points]
            bo_result = optimize_model_hyperparameters(
                dataset_train_full if is_sw else dataset_train,
                dataset_val,
                config=bo_run_cfg,
            )
            bayesian_opt_results[region] = bo_result
            best_parameters = bo_result.get("best_parameters", {})
            if best_parameters:
                region_train_config.update(best_parameters)
                if is_sw and "window_size" in best_parameters:
                    window_size_points = int(best_parameters["window_size"])
                    stride_points = max(1, window_size_points // 2)
                    input_sequence_length = window_size_points
                    dataset_train = SpectrumPatchDataset(
                        dataset_train_full,
                        window_size=window_size_points,
                        stride=stride_points,
                    )
                    region_train_config.update(
                        {
                            "window_size": window_size_points,
                            "stride": stride_points,
                            "n_points": input_sequence_length,
                        }
                    )

        model = instantiate_model(
            region_model_type,
            seq_len=input_sequence_length,
            config=region_train_config,
        )
        region_train_config.update({"model": model, "criterion": criterion})
        if verbose:
            region_train_config["verbose"] = True

        training_result = train_model_region(
            dataset_train, dataset_val, config=region_train_config
        )
        trained_model = training_result["model"]
        trained_models[region] = (
            (trained_model, window_size_points, stride_points)
            if is_sw
            else trained_model
        )
        val_evaluation_results[region] = float(training_result["best_val_loss"])
        training_histories[region] = training_result["history"]

    if is_unified:
        single_entry = trained_models.get("all")
        single_loss = val_evaluation_results.get("all", 0.0)
        single_hist = training_histories.get("all", {})
        single_bo = bayesian_opt_results.get("all", {})
        for region in regional_datasets.keys():
            if region != "all":
                trained_models[region] = single_entry
                val_evaluation_results[region] = single_loss
                training_histories[region] = single_hist
                if single_bo:
                    bayesian_opt_results[region] = single_bo

    return {
        "models": trained_models,
        "val_evaluation": val_evaluation_results,
        "histories": training_histories,
        "bayesian_opt_results": bayesian_opt_results,
    }
