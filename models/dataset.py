"""Dataset classes and session-level splitting utilities for spectral transfer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SpectrumPairDataset(Dataset):
    """PyTorch Dataset yielding paired source and target spectral vectors.

    Parameters
    ----------
    x_data : np.ndarray | torch.Tensor
        Array of source intensities of shape (N_samples, N_points).
    y_data : np.ndarray | torch.Tensor
        Array of target intensities of shape (N_samples, N_points).
    energy : np.ndarray | torch.Tensor | None, optional
        Array of binding energies of shape (N_samples, N_points) or (N_points,).
    metadata : list[dict[str, Any]] | None, optional
        List of metadata dictionaries per sample.

    Raises
    ------
    ValueError
        If lengths of x_data and y_data do not match.
    """

    def __init__(
        self,
        x_data: np.ndarray | torch.Tensor,
        y_data: np.ndarray | torch.Tensor,
        energy: np.ndarray | torch.Tensor | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        if len(x_data) != len(y_data):
            raise ValueError(
                f"x_data length ({len(x_data)}) must match y_data length ({len(y_data)})"
            )

        if isinstance(x_data, np.ndarray):
            self.x = torch.from_numpy(x_data.astype(np.float32))
        else:
            self.x = x_data.float()

        if isinstance(y_data, np.ndarray):
            self.y = torch.from_numpy(y_data.astype(np.float32))
        else:
            self.y = y_data.float()

        if energy is not None:
            if isinstance(energy, np.ndarray):
                self.energy = torch.from_numpy(energy.astype(np.float32))
            else:
                self.energy = energy.float()
        else:
            self.energy = None

        self.metadata = metadata or []

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "x": self.x[idx],
            "y": self.y[idx],
        }
        if self.energy is not None:
            item["energy"] = self.energy[idx] if self.energy.dim() > 1 else self.energy
        if self.metadata and idx < len(self.metadata):
            item["meta"] = self.metadata[idx]
        return item


def split_session_datasets(
    meta_df: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[SpectrumPairDataset, SpectrumPairDataset]:
    """Split paired spectra into train and validation datasets grouped strictly by measurement session.

    This ensures that all dies belonging to the same measurement session stay in either
    train or validation, preventing inter-die data leakage across splits.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Metadata DataFrame containing pairing columns ('measurement_id_target', etc.).
    ary_intensity : np.ndarray
        2D array of measured intensities of shape (N_total, N_points).
    ary_energy : np.ndarray
        2D array of binding energies of shape (N_total, N_points).
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'region' (str): Spectral region to extract (e.g., 'Al2p').
        - 'source_tool' (str | None): Filter for source tool (default auto-detect).
        - 'target_tool' (str | None): Filter for target tool (default auto-detect).
        - 'val_ratio' (float): Fraction of measurement sessions for validation (default 0.2).
        - 'seed' (int | None): Random seed for session partitioning (default 42).

    Returns
    -------
    tuple[SpectrumPairDataset, SpectrumPairDataset]
        (train_dataset, val_dataset)
    """
    cfg = config or {}
    region: str = cfg.get("region", "Al2p")
    source_tool: str | None = cfg.get("source_tool")
    target_tool: str | None = cfg.get("target_tool")
    val_ratio: float = float(cfg.get("val_ratio", 0.2))
    seed: int | None = cfg.get("seed", 42)

    # Filter paired rows
    condition = meta_df["measurement_id_target"].notna() & (meta_df["region"] == region)
    if source_tool:
        condition = condition & (meta_df["tool"] == source_tool)
    if target_tool:
        condition = condition & (meta_df["tool_target"] == target_tool)

    paired_df = meta_df[condition].copy()

    if paired_df.empty:
        raise ValueError(
            f"No paired records found for region='{region}', source_tool='{source_tool}', target_tool='{target_tool}'"
        )

    # Group sessions by unique source measurement_id
    sessions = np.array(sorted(paired_df["measurement_id"].unique()))
    n_sessions = len(sessions)

    if n_sessions < 2:
        raise ValueError(
            f"Need at least 2 distinct measurement sessions to split into train and val, but found only {n_sessions}"
        )

    # Shuffle sessions
    rng = np.random.default_rng(seed)
    shuffled_sessions = rng.permutation(sessions)

    # Clamp n_val_sessions to [1, n_sessions - 1] so neither train nor val is ever empty
    n_val_sessions = max(1, min(n_sessions - 1, int(round(n_sessions * val_ratio))))
    val_sessions = set(shuffled_sessions[:n_val_sessions])
    train_sessions = set(shuffled_sessions[n_val_sessions:])

    # Split rows
    train_df = paired_df[paired_df["measurement_id"].isin(train_sessions)]
    val_df = paired_df[paired_df["measurement_id"].isin(val_sessions)]

    # Assert no target session leakage across splits
    target_leakage = set(train_df["measurement_id_target"].dropna()).intersection(
        set(val_df["measurement_id_target"].dropna())
    )
    if target_leakage:
        raise ValueError(
            f"Target measurement sessions leak across train and val splits: {target_leakage}"
        )

    def _build_dataset(df_subset: pd.DataFrame) -> SpectrumPairDataset:
        x_indices = df_subset["spectrum_index"].astype(int).to_numpy()
        y_indices = df_subset["spectrum_index_target"].astype(int).to_numpy()

        x_arr = ary_intensity[x_indices]
        y_arr = ary_intensity[y_indices]
        energy_arr = ary_energy[x_indices]

        meta_list = df_subset.to_dict(orient="records")
        return SpectrumPairDataset(x_arr, y_arr, energy=energy_arr, metadata=meta_list)

    train_ds = _build_dataset(train_df)
    val_ds = _build_dataset(val_df)

    return train_ds, val_ds


def default_spectrum_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function that stacks tensors and safely keeps non-tensor metadata as lists."""
    if not batch:
        return {}
    first = batch[0]
    result: dict[str, Any] = {}
    for key, val in first.items():
        if val is None:
            result[key] = [b.get(key) for b in batch]
        elif isinstance(val, torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])
        elif isinstance(val, np.ndarray):
            result[key] = torch.from_numpy(np.stack([b[key] for b in batch]))
        else:
            result[key] = [b.get(key) for b in batch]
    return result


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
        - 'shuffle_train' (bool): Whether to shuffle train set (default True).
        - 'num_workers' (int): Number of worker processes (default 0).
        - 'collate_fn' (Callable | None): Custom collate function (default default_spectrum_collate).

    Returns
    -------
    tuple[DataLoader, DataLoader]
        (train_loader, val_loader)
    """
    cfg = config or {}
    batch_size: int = int(cfg.get("batch_size", 16))
    shuffle_train: bool = bool(cfg.get("shuffle_train", True))
    num_workers: int = int(cfg.get("num_workers", 0))
    collate_fn = cfg.get("collate_fn", default_spectrum_collate)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
