"""Dataset classes and regional dataset builders for spectral transfer."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utility.patching import calculate_window_points, extract_sliding_windows


def _as_3d_tensor(arr: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert a 2D or 3D array/tensor to a 3D float32 tensor (N, 1, L)."""
    t = torch.as_tensor(arr, dtype=torch.float32)
    return t.unsqueeze(1) if t.dim() == 2 else t


class SpectrumPairDataset(Dataset):
    """PyTorch Dataset yielding paired source and target spectral vectors.

    Parameters
    ----------
    x_data : np.ndarray | torch.Tensor
        Array of source intensities of shape (N_samples, N_points).
    y_data : np.ndarray | torch.Tensor
        Array of target intensities of shape (N_samples, N_points).
    config : dict[str, Any] | None, optional
        Optional configuration dictionary containing 'energy' or 'metadata'.
    energy : np.ndarray | torch.Tensor | None, optional
        Array of binding energies of shape (N_samples, N_points).
    metadata : list[dict[str, Any]] | None, optional
        List of metadata dictionaries per sample.
    """

    def __init__(
        self,
        x_data: np.ndarray | torch.Tensor,
        y_data: np.ndarray | torch.Tensor,
        config: dict[str, Any] | None = None,
        *,
        energy: np.ndarray | torch.Tensor | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        if len(x_data) != len(y_data):
            raise ValueError(f"Lengths differ: {len(x_data)} vs {len(y_data)}")

        self.x = _as_3d_tensor(x_data)
        self.y = _as_3d_tensor(y_data)
        cfg = config or {}

        energy = energy if energy is not None else cfg.get("energy")
        metadata = metadata if metadata is not None else cfg.get("metadata")

        self.energy = _as_3d_tensor(energy) if energy is not None else None
        self.metadata = metadata or []

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "x": self.x[idx],
            "y": self.y[idx],
        }
        if self.energy is not None:
            item["energy"] = self.energy[idx]
            e_1d = self.energy[idx].squeeze(0)
            bounds = torch.tensor(
                [e_1d[0].item(), e_1d[-1].item()], dtype=torch.float32
            )
            item["energy_bounds"] = bounds
        if self.metadata and idx < len(self.metadata):
            item["meta"] = self.metadata[idx]
        return item


class SpectrumPatchDataset(Dataset):
    """PyTorch Dataset yielding paired sliding window spectral patches.

    Stores lightweight index tuples (spectrum_index, start_point, end_point)
    and slices patches on-the-fly during __getitem__ from the underlying
    SpectrumPairDataset without duplicating full arrays.

    Parameters
    ----------
    full_dataset : SpectrumPairDataset
        The full-spectrum paired dataset.
    window_size : int
        Window size in points.
    stride : int
        Stride in points.
    """

    def __init__(
        self, full_dataset: SpectrumPairDataset, window_size: int, stride: int
    ) -> None:
        self.full_dataset = full_dataset
        self.full_datasets: list[SpectrumPairDataset] = [full_dataset]
        self.cum_dataset_sizes: list[int] = [len(full_dataset)]
        self.window_size, self.stride = window_size, stride
        patch_indices: list[tuple[int, int, int]] = []
        max_vals_list: list[float] = []
        meta_list: list[dict[str, Any]] = []
        energy_bounds_list: list[list[float]] = []

        for idx in range(len(full_dataset)):
            item = full_dataset[idx]
            x_np = item["x"].cpu().numpy()
            y_np = item["y"].cpu().numpy()
            _, start_indices = extract_sliding_windows(
                x_np, window_size, stride
            )
            n_win = len(start_indices)

            for s in start_indices:
                patch_indices.append((idx, s, s + window_size))

            scale_ref = float(np.max(np.abs(y_np)))
            eps = max(1e-12, 1e-7 * float(scale_ref))
            if scale_ref <= eps:
                warnings.warn(
                    f"Denominator scale <= eps ({eps:.4e}) encountered in "
                    "patch dataset; scale is tiny/near-zero and dynamic "
                    "eps was used.",
                    UserWarning,
                    stacklevel=2,
                )
            max_val = max(scale_ref, eps)
            max_vals_list.extend([max_val] * n_win)

            if "meta" in item:
                meta_list.extend([item["meta"]] * n_win)

            e_tensor = item.get("energy")
            if e_tensor is not None:
                e_arr = (
                    e_tensor.cpu().numpy().ravel()
                    if isinstance(e_tensor, torch.Tensor)
                    else np.asarray(e_tensor).ravel()
                )
                for s in start_indices:
                    end_idx = min(s + window_size - 1, len(e_arr) - 1)
                    energy_bounds_list.append(
                        [float(e_arr[s]), float(e_arr[end_idx])]
                    )
            else:
                energy_bounds_list.extend([[0.0, 0.0]] * n_win)

        if not patch_indices:
            raise ValueError("No patches could be extracted from the dataset.")

        self.patch_indices = patch_indices
        self.max_vals = torch.tensor(
            max_vals_list, dtype=torch.float32
        ).view(-1, 1, 1)
        self.meta_list = meta_list
        self.energy_bounds = torch.tensor(
            energy_bounds_list, dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.patch_indices)

    def _get_full_spectrum_item(self, global_spec_idx: int) -> dict[str, Any]:
        """Retrieve full spectrum item across single or concatenated datasets."""
        if len(self.full_datasets) == 1:
            return self.full_datasets[0][global_spec_idx]
        import bisect

        ds_idx = bisect.bisect_right(self.cum_dataset_sizes, global_spec_idx)
        prev_cum = self.cum_dataset_sizes[ds_idx - 1] if ds_idx > 0 else 0
        local_idx = global_spec_idx - prev_cum
        return self.full_datasets[ds_idx][local_idx]

    @property
    def x(self) -> torch.Tensor:
        """Dynamic stacked view of patches for backwards compatibility."""
        return torch.stack([self[i]["x"] for i in range(len(self))], dim=0)

    @property
    def y(self) -> torch.Tensor:
        """Dynamic stacked view of target patches for backwards compatibility."""
        return torch.stack([self[i]["y"] for i in range(len(self))], dim=0)

    @classmethod
    def concatenate(
        cls, datasets: list[SpectrumPatchDataset]
    ) -> SpectrumPatchDataset:
        """Merge patch datasets that share one window size and stride."""
        if not datasets:
            raise ValueError("Cannot concatenate an empty list of patch datasets.")
        geometries = {(d.window_size, d.stride) for d in datasets}
        if len(geometries) > 1:
            raise ValueError(
                "Patch datasets must share (window_size, stride) to be "
                f"concatenated, got {sorted(geometries)}."
            )
        has_meta = [len(d.meta_list) == len(d) for d in datasets]
        if any(has_meta) and not all(has_meta):
            raise ValueError(
                "Cannot concatenate patch datasets where only some carry "
                f"per-patch metadata (has_meta per dataset: {has_meta})."
            )
        merged = cls.__new__(cls)
        merged.window_size, merged.stride = geometries.pop()

        merged.full_datasets = [
            fd for d in datasets for fd in d.full_datasets
        ]
        cum_sizes: list[int] = []
        total = 0
        for fd in merged.full_datasets:
            total += len(fd)
            cum_sizes.append(total)
        merged.cum_dataset_sizes = cum_sizes

        # Provide full_dataset attribute when sequence lengths match
        lengths = {int(fd.x.shape[-1]) for fd in merged.full_datasets}
        if len(lengths) == 1:
            merged_x = torch.cat([fd.x for fd in merged.full_datasets], dim=0)
            merged_y = torch.cat([fd.y for fd in merged.full_datasets], dim=0)
            has_all_e = all(fd.energy is not None for fd in merged.full_datasets)
            merged_e = (
                torch.cat([fd.energy for fd in merged.full_datasets], dim=0)
                if has_all_e
                else None
            )
            merged_m = [m for fd in merged.full_datasets for m in fd.metadata]
            merged.full_dataset = SpectrumPairDataset(
                merged_x, merged_y, energy=merged_e, metadata=merged_m
            )
        else:
            merged.full_dataset = merged.full_datasets[0]

        merged_indices: list[tuple[int, int, int]] = []
        offset = 0
        for d in datasets:
            for spec_idx, s, e in d.patch_indices:
                merged_indices.append((spec_idx + offset, s, e))
            offset += sum(len(fd) for fd in d.full_datasets)

        merged.patch_indices = merged_indices
        merged.max_vals = torch.cat([d.max_vals for d in datasets], dim=0)
        merged.energy_bounds = torch.cat(
            [d.energy_bounds for d in datasets], dim=0
        )
        merged.meta_list = [m for d in datasets for m in d.meta_list]
        return merged

    def __getitem__(self, idx: int) -> dict[str, Any]:
        spec_idx, s, e = self.patch_indices[idx]
        item = self._get_full_spectrum_item(spec_idx)
        ret: dict[str, Any] = {
            "x": item["x"][..., s:e],
            "y": item["y"][..., s:e],
            "max_val": self.max_vals[idx],
            "energy_bounds": self.energy_bounds[idx],
        }
        if self.meta_list and idx < len(self.meta_list):
            ret["meta"] = self.meta_list[idx]
        return ret


def default_spectrum_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function stacking tensors and safely keeping metadata
    lists.
    """
    if not batch:
        return {}

    def _collate_key(val: Any, key: str) -> Any:
        if isinstance(val, torch.Tensor):
            return torch.stack([b[key] for b in batch])
        if isinstance(val, np.ndarray):
            return torch.from_numpy(np.stack([b[key] for b in batch]))
        return [b.get(key) for b in batch]

    return {k: _collate_key(v, k) for k, v in batch[0].items()}


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: dict[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training and validation datasets.

    Parameters
    ----------
    train_dataset : Dataset
        Training dataset.
    val_dataset : Dataset
        Validation dataset.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'batch_size' (int): Batch size (default 16).
        - 'shuffle_train' (bool): Shuffle train set (default True).
        - 'num_workers' (int): Worker processes (default 0).
        - 'collate_fn' (Callable | None): Custom collate function.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        (train_loader, val_loader)
    """
    cfg = config or {}
    loader_kwargs = {
        "batch_size": int(cfg.get("batch_size", 16)),
        "num_workers": int(cfg.get("num_workers", 0)),
        "collate_fn": cfg.get("collate_fn", default_spectrum_collate),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=bool(cfg.get("shuffle_train", True)),
        **loader_kwargs,
    )
    return train_loader, DataLoader(val_dataset, shuffle=False, **loader_kwargs)


def _build_split_dataset_bundle(
    df_subset: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    use_sw: bool,
    sw_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Slice train/val/test rows and construct paired or patch datasets."""

    def _slice_rows(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
        if len(indices) > 0:
            return arr[indices]
        return np.empty((0, arr.shape[1]), dtype=np.float32)

    def _make_pair_ds(split_name: str) -> SpectrumPairDataset:
        df_split = df_subset[df_subset["split"] == split_name]
        x_idx = df_split["spectrum_index"].astype(int).to_numpy()
        y_idx = df_split["target_spectrum_index"].astype(int).to_numpy()
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
    }


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
        and 'target_spectrum_index' columns.
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
        paired_mask = df_meta["target_measurement_id"].notna()
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
            & df_meta["target_measurement_id"].notna()
        )
        if source_tool is not None:
            cond = cond & (df_meta["tool"] == str(source_tool))
        if target_tool is not None:
            cond = cond & (df_meta["target_tool"] == str(target_tool))

        df_region = df_meta[cond]
        if df_region.empty:
            continue

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
        regional_datasets[region] = _build_split_dataset_bundle(
            df_region, ary_intensity, ary_energy, use_sw, sw_cfg
        )

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
        and 'target_spectrum_index' columns.
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

    paired_mask = df_meta["target_measurement_id"].notna()
    if source_tool is not None:
        paired_mask = paired_mask & (df_meta["tool"] == str(source_tool))
    if target_tool is not None:
        paired_mask = (
            paired_mask & (df_meta["target_tool"] == str(target_tool))
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

    sw_cfg = {
        "window_size_ev": cfg.get("window_size_ev", 6.0),
        "sliding_stride_ev": cfg.get("sliding_stride_ev", 2.0),
        "window_size": cfg.get("window_size"),
        "stride": cfg.get("stride"),
    }
    bundle = _build_split_dataset_bundle(
        df_paired, ary_intensity, ary_energy, use_sw, sw_cfg
    )

    # Warn about region geometry variations under fixed point window
    if use_sw and len(df_paired) > 0:
        first_region = str(df_paired["region"].iloc[0])
        w_pts = bundle["window_size_points"]
        spans: dict[str, str] = {}
        for reg in regions:
            reg_rows = df_paired[df_paired["region"] == reg]
            if not reg_rows.empty:
                spec_idx = int(reg_rows["spectrum_index"].iloc[0])
                reg_e = ary_energy[spec_idx]
                end_idx = min(w_pts - 1, len(reg_e) - 1)
                span_ev = abs(float(reg_e[end_idx] - reg_e[0]))
                spans[reg] = f"{span_ev:.2f} eV"
        warnings.warn(
            f"Sliding window in unified dataset: point conversion derived from "
            f"first region '{first_region}'. Physical eV span under {w_pts} points "
            f"across regions: {spans}.",
            UserWarning,
            stacklevel=2,
        )

    return {
        **bundle,
        "regions": regions,
        "region_index_mapping": region_index_mapping,
    }


__all__ = [
    "SpectrumPairDataset",
    "SpectrumPatchDataset",
    "default_spectrum_collate",
    "create_dataloaders",
    "create_regional_datasets",
    "create_unified_dataset",
]
