"""Model training and Bayesian optimization orchestration module for GGRD-SP."""

from typing import Any

import torch
import torch.nn as nn

from models.bayesian_opt import optimize_model_hyperparameters
from models.dataset import SpectrumPairDataset, SpectrumPatchDataset
from models.factory import NO_SLIDING_WINDOW_MODEL_TYPES
from models.trainer import instantiate_model, train_model_region

_CONTAINER_KEYS = {
    "train_config",
    "bayesian_opt_config",
    "film_config",
    "region_configs",
    "window_size_ev",
    "sliding_stride_ev",
}


def _extract_cfg_value(
    primary: dict[str, Any],
    fallback: dict[str, Any],
    keys: tuple[str, ...],
    default: float,
) -> float:
    """Extract first matching float value from primary or fallback dictionary."""
    for k in keys:
        val = primary.get(k)
        if val is not None:
            return float(val)
        val = fallback.get(k)
        if val is not None:
            return float(val)
    return default


def _resolve_sliding_window_config(
    global_config: dict[str, Any],
    region_override: dict[str, Any],
    region: str,
) -> dict[str, Any]:
    """Resolve sliding window configuration for a single region or unified layout."""
    dataset_cfg = (
        global_config.get("dataset")
        if isinstance(global_config.get("dataset"), dict)
        else {}
    )
    if region == "all":
        w_keys = ("unified_window_size_ev", "window_size_ev")
        s_keys = ("unified_window_stride_ev", "sliding_stride_ev")
    else:
        w_keys = ("window_size_ev",)
        s_keys = ("sliding_stride_ev",)
    default_window_size_ev = _extract_cfg_value(global_config, dataset_cfg, w_keys, 6.0)
    default_sliding_stride_ev = _extract_cfg_value(global_config, dataset_cfg, s_keys, 2.0)

    window_size_points = region_override.get(
        "window_size",
        global_config.get("window_size") or dataset_cfg.get("window_size"),
    )
    stride_points = region_override.get(
        "stride",
        global_config.get("stride") or dataset_cfg.get("stride"),
    )
    return {
        "window_size_ev": region_override.get(
            "window_size_ev", default_window_size_ev
        ),
        "sliding_stride_ev": region_override.get(
            "sliding_stride_ev", default_sliding_stride_ev
        ),
        "window_size": window_size_points,
        "stride": stride_points,
    }


def _resolve_train_and_bo_configs(
    global_config: dict[str, Any],
    region_override: dict[str, Any],
    bayesian_opt_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve training and Bayesian optimization configurations for a region."""
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
    return resolved_train_config, resolved_bo_config


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
    dataset_cfg = (
        global_config.get("dataset")
        if isinstance(global_config.get("dataset"), dict)
        else {}
    )
    reg_cfgs = global_config.get("region_configs")
    if not isinstance(reg_cfgs, dict):
        reg_cfgs = dataset_cfg.get("region_configs", {})
    region_override = (
        dict(reg_cfgs.get(region, {}))
        if isinstance(reg_cfgs, dict) and region in reg_cfgs
        else {}
    )

    sliding_window_config = _resolve_sliding_window_config(
        global_config, region_override, region
    )
    resolved_train_config, resolved_bo_config = _resolve_train_and_bo_configs(
        global_config, region_override, bayesian_opt_config
    )
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


def _run_regional_bo(
    dataset_train: Any,
    dataset_train_full: Any,
    dataset_val: Any,
    region_train_config: dict[str, Any],
    region_bo_config: dict[str, Any],
    film_cfg: dict[str, Any],
    region_model_type: str,
    is_sw: bool,
    window_size_points: int,
    stride_points: int,
    input_sequence_length: int,
    region_use_film: bool,
    criterion: nn.Module,
) -> tuple[dict[str, Any], Any, int, int, int]:
    """Execute Bayesian optimization for a region and update config and dataset."""
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
    best_parameters = bo_result.get("best_parameters", {})
    if best_parameters:
        region_train_config.update(best_parameters)
        if is_sw and "window_size" in best_parameters:
            window_size_points = int(best_parameters["window_size"])
            stride_points = max(1, min(window_size_points, stride_points))
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
    return bo_result, dataset_train, window_size_points, stride_points, input_sequence_length


def _train_single_region_pipeline(
    region: str,
    bundle: dict[str, Any],
    pipeline_config: dict[str, Any],
    bo_cfg_input: dict[str, Any] | None,
    default_model_type: str,
    default_use_bayesian_opt: bool,
    verbose: bool,
    film_cfg: dict[str, Any],
    criterion: nn.Module,
) -> tuple[Any, float, dict[str, list[float]], dict[str, Any] | None]:
    """Train single region model optionally running Bayesian optimization."""
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
    if region_model_type in NO_SLIDING_WINDOW_MODEL_TYPES and (
        is_sw or region_train_config.get("use_sliding_window", False)
    ):
        raise ValueError(
            f"Sliding window mode (use_sliding_window=True) is not supported "
            f"for model_type={region_model_type!r}. Multi-field-of-view "
            "architectures process full regional spectra directly. "
            "Please set 'use_sliding_window: false'."
        )
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

    bo_result: dict[str, Any] | None = None
    if region_use_bayesian_opt:
        (
            bo_result,
            dataset_train,
            window_size_points,
            stride_points,
            input_sequence_length,
        ) = _run_regional_bo(
            dataset_train=dataset_train,
            dataset_train_full=dataset_train_full,
            dataset_val=dataset_val,
            region_train_config=region_train_config,
            region_bo_config=region_bo_config,
            film_cfg=film_cfg,
            region_model_type=region_model_type,
            is_sw=is_sw,
            window_size_points=window_size_points,
            stride_points=stride_points,
            input_sequence_length=input_sequence_length,
            region_use_film=region_use_film,
            criterion=criterion,
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
    model_entry = (
        (trained_model, window_size_points, stride_points)
        if is_sw
        else trained_model
    )
    val_loss = float(training_result["best_val_loss"])
    history = training_result["history"]
    return model_entry, val_loss, history, bo_result


def _setup_unified_training(
    pipeline_config: dict[str, Any],
    film_cfg: dict[str, Any],
    regional_datasets: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Configure FiLM regions and combine bundles for unified training."""
    cfg_copy = dict(film_cfg)
    cfg_copy.setdefault("encode_region", True)
    if "all" in regional_datasets:
        actual_num = len(regional_datasets["all"].get("regions", []))
    else:
        actual_num = len(regional_datasets)
    target_num_regions = int(cfg_copy.get("num_regions") or actual_num)
    cfg_copy["num_regions"] = target_num_regions
    pipeline_config["film_config"] = cfg_copy
    return cfg_copy, {"all": _combine_regional_bundles(regional_datasets)}


def _fan_out_unified_results(
    regional_datasets: dict[str, dict[str, Any]],
    trained_models: dict[str, Any],
    val_evaluation_results: dict[str, float],
    training_histories: dict[str, dict[str, list[float]]],
    bayesian_opt_results: dict[str, dict[str, Any]],
) -> None:
    """Propagate unified model results to all individual regional entries."""
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
        film_cfg, target_datasets = _setup_unified_training(
            pipeline_config, film_cfg, regional_datasets
        )
    else:
        target_datasets = regional_datasets

    for region, bundle in target_datasets.items():
        (
            model_entry,
            val_loss,
            history,
            bo_result,
        ) = _train_single_region_pipeline(
            region=region,
            bundle=bundle,
            pipeline_config=pipeline_config,
            bo_cfg_input=bo_cfg_input,
            default_model_type=default_model_type,
            default_use_bayesian_opt=default_use_bayesian_opt,
            verbose=verbose,
            film_cfg=film_cfg,
            criterion=criterion,
        )
        trained_models[region] = model_entry
        val_evaluation_results[region] = val_loss
        training_histories[region] = history
        if bo_result is not None:
            bayesian_opt_results[region] = bo_result

    if is_unified:
        _fan_out_unified_results(
            regional_datasets,
            trained_models,
            val_evaluation_results,
            training_histories,
            bayesian_opt_results,
        )

    return {
        "models": trained_models,
        "val_evaluation": val_evaluation_results,
        "histories": training_histories,
        "bayesian_opt_results": bayesian_opt_results,
    }
