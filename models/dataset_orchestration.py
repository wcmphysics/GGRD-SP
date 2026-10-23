"""Dataset orchestration module for GGRD-SP.

Provides orchestration functions to assemble and partition regional and
unified spectral datasets for model training, validation, and evaluation.
"""

from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from models.dataset import (
    SpectrumPairDataset,
    SpectrumPatchDataset,
)
from utility.patching import calculate_window_points


def create_regional_datasets(
    df_meta: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Create regional datasets partitioned by train, val, and test splits.

    Builds PyTorch Dataset objects for each spectral region. When sliding
    window data augmentation is enabled, training spectra are sliced into
    unpadded SpectrumPatchDataset instances with right-edge anchoring, while
    validation and test datasets remain full-spectrum datasets.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata DataFrame containing 'split', 'region', 'spectrum_index',
        and 'spectrum_index_target' columns.
    ary_intensity : np.ndarray
        2D array of measured intensities of shape (N_total, N_points).
    ary_energy : np.ndarray
        2D array of binding energies of shape (N_total, N_points).
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'regions': list of spectral region names (default: all unique in
          paired df_meta).
        - 'source_tool': optional source tool filter string.
        - 'target_tool': optional target tool filter string.
        - 'use_sliding_window': bool (default False).
        - 'window_size_ev': window size in eV (default 6.0).
        - 'sliding_stride_ev': sliding stride in eV (default 2.0).
        - 'region_configs': optional dict mapping region names to specific
          overrides.

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary mapping each region name to a dictionary containing:
        - 'dataset_train': training Dataset (SpectrumPatchDataset if sliding
          window on, else SpectrumPairDataset).
        - 'dataset_val': validation full-spectrum Dataset.
        - 'dataset_test': held-out test full-spectrum Dataset.
        - 'dataset_train_full': unpatched training full-spectrum Dataset.
        - 'window_size_points': window size in data points.
        - 'sliding_stride_points': sliding stride in data points.
        - 'full_spectrum_length': full spectrum sequence length.
    """
    cfg = config or {}
    source_tool = cfg.get("source_tool")
    target_tool = cfg.get("target_tool")
    default_use_sw = bool(cfg.get("use_sliding_window", False))
    region_configs = cfg.get("region_configs") or {}

    regions = cfg.get("regions")
    if regions is None:
        paired_mask = df_meta["measurement_id_target"].notna()
        if source_tool is not None:
            paired_mask = paired_mask & (df_meta["tool"] == str(source_tool))
        regions = sorted(
            df_meta[paired_mask]["region"].dropna().unique().tolist()
        )

    if not regions:
        raise ValueError(
            f"No valid spectral regions found in df_meta for "
            f"{source_tool=}, {target_tool=}."
        )

    if bool(cfg.get("unified", False)) or bool(
        cfg.get("unified_dataset", False)
    ):
        return {
            "all": create_unified_dataset(
                df_meta, ary_intensity, ary_energy, config=cfg
            )
        }

    if "region_index" not in df_meta.columns:
        unique_regs = sorted(df_meta["region"].dropna().astype(str).unique())
        reg_map = {r: i for i, r in enumerate(unique_regs)}
        df_meta = df_meta.copy()
        df_meta["region_index"] = df_meta["region"].map(reg_map)

    def _slice_rows(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
        if len(indices) > 0:
            return arr[indices]
        return np.empty((0, arr.shape[1]), dtype=np.float32)

    regional_datasets: dict[str, dict[str, Any]] = {}

    for region in regions:
        region_override = (
            dict(region_configs.get(region, {}))
            if isinstance(region_configs, dict)
            else {}
        )
        use_sw = bool(
            region_override.get("use_sliding_window", default_use_sw)
        )

        cond = (
            (df_meta["region"] == region)
            & df_meta["measurement_id_target"].notna()
        )
        if source_tool is not None:
            cond = cond & (df_meta["tool"] == str(source_tool))
        if target_tool is not None:
            cond = cond & (df_meta["tool_target"] == str(target_tool))

        df_region = df_meta[cond]
        if df_region.empty:
            continue

        def _make_pair_ds(split_name: str) -> SpectrumPairDataset:
            df_split = df_region[df_region["split"] == split_name]
            x_idx = df_split["spectrum_index"].astype(int).to_numpy()
            y_idx = df_split["spectrum_index_target"].astype(int).to_numpy()
            return SpectrumPairDataset(
                _slice_rows(ary_intensity, x_idx),
                _slice_rows(ary_intensity, y_idx),
                energy=_slice_rows(ary_energy, x_idx),
                metadata=df_split.to_dict(orient="records"),
            )

        dataset_train_full = _make_pair_ds("train")
        dataset_val_full = _make_pair_ds("val")
        dataset_test_full = _make_pair_ds("test")

        full_spectrum_length = (
            int(dataset_train_full[0]["x"].shape[-1])
            if len(dataset_train_full) > 0
            else ary_intensity.shape[1]
        )

        if use_sw and len(dataset_train_full) > 0:
            sample_energy_tensor = dataset_train_full[0].get("energy")
            energy_grid = (
                sample_energy_tensor.cpu().numpy().ravel()
                if sample_energy_tensor is not None
                else np.linspace(0.0, 10.0, full_spectrum_length)
            )
            sw_cfg = {
                "window_size_ev": region_override.get(
                    "window_size_ev", cfg.get("window_size_ev", 6.0)
                ),
                "sliding_stride_ev": region_override.get(
                    "sliding_stride_ev", cfg.get("sliding_stride_ev", 2.0)
                ),
                "window_size": region_override.get(
                    "window_size", cfg.get("window_size")
                ),
                "stride": region_override.get("stride", cfg.get("stride")),
            }
            w_pts, s_pts = calculate_window_points(energy_grid, config=sw_cfg)
            window_size_points = min(full_spectrum_length, max(3, w_pts))
            sliding_stride_points = max(1, min(window_size_points, s_pts))
            dataset_train: Dataset = SpectrumPatchDataset(
                dataset_train_full,
                window_size=window_size_points,
                stride=sliding_stride_points,
            )
        else:
            window_size_points = sliding_stride_points = full_spectrum_length
            dataset_train = dataset_train_full

        regional_datasets[region] = {
            "dataset_train": dataset_train,
            "dataset_val": dataset_val_full,
            "dataset_test": dataset_test_full,
            "dataset_train_full": dataset_train_full,
            "window_size_points": window_size_points,
            "sliding_stride_points": sliding_stride_points,
            "full_spectrum_length": full_spectrum_length,
        }

    return regional_datasets


def create_unified_dataset(
    df_meta: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a unified multi-region dataset partitioned by train, val,
    and test splits.

    Combines paired regional spectra across all regions into single unified
    PyTorch datasets. Each sample preserves 'region' and 'region_index' metadata
    for FiLM conditioning and regional error weighting.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata DataFrame containing 'split', 'region', 'spectrum_index',
        and 'spectrum_index_target' columns.
    ary_intensity : np.ndarray
        2D array of measured intensities of shape (N_total, N_points).
    ary_energy : np.ndarray
        2D array of binding energies of shape (N_total, N_points).
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'regions': list of spectral region names (default: all unique in
          paired df_meta).
        - 'source_tool': optional source tool filter string.
        - 'target_tool': optional target tool filter string.
        - 'use_sliding_window': bool (default False).
        - 'window_size_ev': window size in eV (default 6.0).
        - 'sliding_stride_ev': sliding stride in eV (default 2.0).
        - 'window_size': window size in data points.
        - 'stride': sliding stride in data points.
        - 'region_index_mapping': optional mapping from region to integer ID.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'dataset_train': training Dataset (SpectrumPatchDataset if sliding
          window on).
        - 'dataset_val': validation full-spectrum Dataset.
        - 'dataset_test': held-out test full-spectrum Dataset.
        - 'dataset_train_full': unpatched training full-spectrum Dataset.
        - 'window_size_points': window size in data points.
        - 'sliding_stride_points': sliding stride in data points.
        - 'full_spectrum_length': full spectrum sequence length.
        - 'regions': list of region names included.
        - 'region_index_mapping': mapping of region name to integer index.
    """
    cfg = config or {}
    source_tool = cfg.get("source_tool")
    target_tool = cfg.get("target_tool")
    use_sw = bool(cfg.get("use_sliding_window", False))

    paired_mask = df_meta["measurement_id_target"].notna()
    if source_tool is not None:
        paired_mask = paired_mask & (df_meta["tool"] == str(source_tool))
    if target_tool is not None:
        paired_mask = (
            paired_mask & (df_meta["tool_target"] == str(target_tool))
        )

    regions = cfg.get("regions")
    if regions is None:
        regions = sorted(
            df_meta[paired_mask]["region"].dropna().unique().tolist()
        )
    if not regions:
        raise ValueError(
            f"No valid spectral regions found in df_meta for "
            f"{source_tool=}, {target_tool=}."
        )

    region_index_mapping = cfg.get("region_index_mapping") or {
        r: i for i, r in enumerate(regions)
    }
    df_paired = df_meta[paired_mask & df_meta["region"].isin(regions)].copy()
    if "region_index" not in df_paired.columns:
        df_paired["region_index"] = df_paired["region"].map(
            region_index_mapping
        )
    else:
        df_paired["region_index"] = df_paired["region_index"].fillna(
            df_paired["region"].map(region_index_mapping)
        ).astype(int)

    def _slice_rows(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
        if len(indices) > 0:
            return arr[indices]
        return np.empty((0, arr.shape[1]), dtype=np.float32)

    def _make_pair_ds(split_name: str) -> SpectrumPairDataset:
        df_split = df_paired[df_paired["split"] == split_name]
        x_idx = df_split["spectrum_index"].astype(int).to_numpy()
        y_idx = df_split["spectrum_index_target"].astype(int).to_numpy()
        return SpectrumPairDataset(
            _slice_rows(ary_intensity, x_idx),
            _slice_rows(ary_intensity, y_idx),
            energy=_slice_rows(ary_energy, x_idx),
            metadata=df_split.to_dict(orient="records"),
        )

    dataset_train_full = _make_pair_ds("train")
    dataset_val_full = _make_pair_ds("val")
    dataset_test_full = _make_pair_ds("test")

    full_spectrum_length = (
        int(dataset_train_full[0]["x"].shape[-1])
        if len(dataset_train_full) > 0
        else ary_intensity.shape[1]
    )

    if use_sw and len(dataset_train_full) > 0:
        sample_energy_tensor = dataset_train_full[0].get("energy")
        energy_grid = (
            sample_energy_tensor.cpu().numpy().ravel()
            if sample_energy_tensor is not None
            else np.linspace(0.0, 10.0, full_spectrum_length)
        )
        sw_cfg = {
            "window_size_ev": cfg.get("window_size_ev", 6.0),
            "sliding_stride_ev": cfg.get("sliding_stride_ev", 2.0),
            "window_size": cfg.get("window_size"),
            "stride": cfg.get("stride"),
        }
        w_pts, s_pts = calculate_window_points(energy_grid, config=sw_cfg)
        window_size_points = min(full_spectrum_length, max(3, w_pts))
        sliding_stride_points = max(1, min(window_size_points, s_pts))
        dataset_train: Dataset = SpectrumPatchDataset(
            dataset_train_full,
            window_size=window_size_points,
            stride=sliding_stride_points,
        )
    else:
        window_size_points = sliding_stride_points = full_spectrum_length
        dataset_train = dataset_train_full

    return {
        "dataset_train": dataset_train,
        "dataset_val": dataset_val_full,
        "dataset_test": dataset_test_full,
        "dataset_train_full": dataset_train_full,
        "window_size_points": window_size_points,
        "sliding_stride_points": sliding_stride_points,
        "full_spectrum_length": full_spectrum_length,
        "regions": regions,
        "region_index_mapping": region_index_mapping,
    }
