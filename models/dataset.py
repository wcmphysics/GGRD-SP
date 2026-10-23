"""Dataset classes and regional dataset builders for spectral transfer."""

from __future__ import annotations

from typing import Any

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
        self.window_size, self.stride = window_size, stride
        x_patches_list, y_patches_list, max_vals_list = [], [], []
        meta_list: list[dict[str, Any]] = []
        energy_bounds_list: list[list[float]] = []

        for idx in range(len(full_dataset)):
            item = full_dataset[idx]
            x_np = item["x"].cpu().numpy()
            y_np = item["y"].cpu().numpy()
            x_win, start_indices = extract_sliding_windows(
                x_np, window_size, stride
            )
            y_win, _ = extract_sliding_windows(y_np, window_size, stride)
            n_win = len(x_win)

            x_patches_list.append(x_win)
            y_patches_list.append(y_win)
            max_vals_list.extend(
                [max(float(np.max(np.abs(y_np))), 1e-4)] * n_win
            )
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

        if not x_patches_list:
            raise ValueError("No patches could be extracted from the dataset.")

        self.x = torch.from_numpy(
            np.vstack(x_patches_list).astype(np.float32)
        ).unsqueeze(1)
        self.y = torch.from_numpy(
            np.vstack(y_patches_list).astype(np.float32)
        ).unsqueeze(1)
        self.max_vals = torch.tensor(
            max_vals_list, dtype=torch.float32
        ).view(-1, 1, 1)
        self.meta_list = meta_list
        self.energy_bounds = torch.tensor(
            energy_bounds_list, dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ret: dict[str, Any] = {
            "x": self.x[idx],
            "y": self.y[idx],
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


# Re-exported from models.dataset_orchestration for backward compatibility
from models.dataset_orchestration import (
    create_regional_datasets,
    create_unified_dataset,
)

__all__ = [
    "SpectrumPairDataset",
    "SpectrumPatchDataset",
    "default_spectrum_collate",
    "create_dataloaders",
    "create_regional_datasets",
    "create_unified_dataset",
]
