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


def partition_measurement_sessions(
    meta_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, set[str]]:
    """Partition unique measurement sessions globally across all spectral regions.

    Guarantees that a physical measurement session is assigned to one and only one
    split ('train', 'val', 'test') across all regions, completely eliminating
    cross-region data leakage.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Metadata DataFrame containing source-to-target pairing columns.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'train_ratio' (float): Fraction of sessions for training (default 0.5).
        - 'val_ratio' (float): Fraction of sessions for validation (default 0.2).
        - 'test_ratio' (float): Fraction of sessions for test (default 0.3).
        - 'source_tool' (str | None): Source tool filter (default auto-detect).
        - 'target_tool' (str | None): Target tool filter (default auto-detect).
        - 'seed' (int | None): Random seed for permutation (default 42).

    Returns
    -------
    dict[str, set[str]]
        Dictionary with keys 'train', 'val', 'test', each mapping to a set of measurement IDs.

    Raises
    ------
    ValueError
        If no paired sessions exist, if insufficient sessions exist, or if target leakage occurs.
    """
    cfg = config or {}
    source_tool: str | None = cfg.get("source_tool")
    target_tool: str | None = cfg.get("target_tool")
    seed: int | None = cfg.get("seed", 42)

    # Filter paired rows
    condition = meta_df["measurement_id_target"].notna()
    if source_tool:
        condition = condition & (meta_df["tool"] == source_tool)
    if target_tool:
        condition = condition & (meta_df["tool_target"] == target_tool)

    paired_df = meta_df[condition].copy()
    if paired_df.empty:
        raise ValueError(
            f"No paired records found for source_tool='{source_tool}', target_tool='{target_tool}'"
        )

    sessions = np.array(sorted(paired_df["measurement_id"].unique()))
    n_sessions = len(sessions)

    train_ratio = float(cfg.get("train_ratio", 0.5))
    val_ratio = float(cfg.get("val_ratio", 0.2))
    test_ratio = float(cfg.get("test_ratio", 0.3))

    if train_ratio < 0.0 or val_ratio < 0.0 or test_ratio < 0.0:
        raise ValueError(
            f"Split ratios must be non-negative: train_ratio={train_ratio}, "
            f"val_ratio={val_ratio}, test_ratio={test_ratio}"
        )

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0.0:
        raise ValueError("Sum of train_ratio, val_ratio, and test_ratio must be positive.")

    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    rng = np.random.default_rng(seed)
    shuffled_sessions = rng.permutation(sessions)

    if test_ratio > 0.0 and val_ratio > 0.0:
        if n_sessions < 3:
            raise ValueError(
                f"Need at least 3 distinct measurement sessions to split into train, val, and test, but found only {n_sessions}"
            )
        n_train = max(1, int(round(n_sessions * train_ratio)))
        n_val = max(1, int(round(n_sessions * val_ratio)))
        n_test = n_sessions - n_train - n_val

        while n_test < 1:
            if n_train > 1 and (n_train >= n_val or n_val == 1):
                n_train -= 1
                n_test += 1
            elif n_val > 1:
                n_val -= 1
                n_test += 1
            else:
                break
        while n_val < 1:
            if n_train > 1:
                n_train -= 1
                n_val += 1
            elif n_test > 1:
                n_test -= 1
                n_val += 1
            else:
                break
        while n_train < 1:
            if n_val > 1:
                n_val -= 1
                n_train += 1
            elif n_test > 1:
                n_test -= 1
                n_train += 1
            else:
                break

        train_sessions = {str(s) for s in shuffled_sessions[:n_train]}
        val_sessions = {str(s) for s in shuffled_sessions[n_train : n_train + n_val]}
        test_sessions = {str(s) for s in shuffled_sessions[n_train + n_val :]}
    else:
        if n_sessions < 2:
            raise ValueError(
                f"Need at least 2 distinct measurement sessions to split, but found only {n_sessions}"
            )
        eval_ratio = val_ratio if val_ratio > 0.0 else test_ratio
        n_eval = max(1, min(n_sessions - 1, int(round(n_sessions * eval_ratio))))
        n_train = n_sessions - n_eval

        train_sessions = {str(s) for s in shuffled_sessions[:n_train]}
        if val_ratio > 0.0:
            val_sessions = {str(s) for s in shuffled_sessions[n_train:]}
            test_sessions = set()
        else:
            val_sessions = set()
            test_sessions = {str(s) for s in shuffled_sessions[n_train:]}

    splits_dict = {"train": train_sessions, "val": val_sessions, "test": test_sessions}
    target_sessions = {
        k: set(paired_df[paired_df["measurement_id"].isin(s)]["measurement_id_target"].dropna())
        for k, s in splits_dict.items()
    }
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        leakage = target_sessions[s1].intersection(target_sessions[s2])
        if leakage:
            raise ValueError(
                f"Target measurement sessions leak across {s1} and {s2} splits: {leakage}"
            )

    return splits_dict


def split_session_datasets(
    meta_df: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[SpectrumPairDataset, ...]:
    """Split paired spectra into train, validation, and optional test datasets grouped by session.

    Guarantees that all dies belonging to the same measurement session stay in the same split,
    preventing inter-die and inter-region data leakage.

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
        - 'session_splits' (dict[str, set[str]] | None): Pre-partitioned session sets.
        - 'return_test' (bool): Whether to return test_dataset as 3rd tuple element (default False).
        - 'train_ratio' (float): Fraction of sessions for training (default 0.5).
        - 'val_ratio' (float): Fraction of sessions for validation (default 0.2).
        - 'test_ratio' (float): Fraction of sessions for test (default 0.3).
        - 'source_tool' (str | None): Filter for source tool.
        - 'target_tool' (str | None): Filter for target tool.
        - 'seed' (int | None): Random seed for partitioning (default 42).

    Returns
    -------
    tuple[SpectrumPairDataset, ...]
        (train_dataset, val_dataset) if return_test is False,
        or (train_dataset, val_dataset, test_dataset) if return_test is True.
    """
    cfg = config or {}
    region: str = cfg.get("region", "Al2p")
    source_tool: str | None = cfg.get("source_tool")
    target_tool: str | None = cfg.get("target_tool")
    return_test: bool = bool(cfg.get("return_test", False))

    if len(ary_intensity) != len(ary_energy):
        raise ValueError(
            f"ary_intensity and ary_energy must have the same number of rows, "
            f"got {len(ary_intensity)} vs {len(ary_energy)}"
        )

    # Filter paired rows for this region
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

    session_splits = cfg.get("session_splits")
    if session_splits is None:
        if "test_ratio" in cfg or return_test:
            session_splits = partition_measurement_sessions(meta_df, config=cfg)
        else:
            p_cfg = dict(cfg)
            p_cfg["val_ratio"] = float(cfg.get("val_ratio", 0.2))
            p_cfg["test_ratio"] = 0.0
            session_splits = partition_measurement_sessions(meta_df, config=p_cfg)

    train_sessions = session_splits.get("train", set())
    val_sessions = session_splits.get("val", set())
    test_sessions = session_splits.get("test", set())

    train_df = paired_df[paired_df["measurement_id"].isin(train_sessions)]
    val_df = paired_df[paired_df["measurement_id"].isin(val_sessions)]
    test_df = paired_df[paired_df["measurement_id"].isin(test_sessions)]

    n_spectra = len(ary_intensity)

    def _build_dataset(df_subset: pd.DataFrame) -> SpectrumPairDataset:
        x_indices = df_subset["spectrum_index"].astype(int).to_numpy()
        y_indices = df_subset["spectrum_index_target"].astype(int).to_numpy()

        if len(x_indices) > 0:
            if np.any((x_indices < 0) | (x_indices >= n_spectra)):
                raise IndexError(
                    f"spectrum_index values out of bounds [0, {n_spectra - 1}]: "
                    f"min={x_indices.min()}, max={x_indices.max()}"
                )
            if np.any((y_indices < 0) | (y_indices >= n_spectra)):
                raise IndexError(
                    f"spectrum_index_target values out of bounds [0, {n_spectra - 1}]: "
                    f"min={y_indices.min()}, max={y_indices.max()}"
                )

        x_arr = ary_intensity[x_indices]
        y_arr = ary_intensity[y_indices]
        energy_arr = ary_energy[x_indices]

        meta_list = df_subset.to_dict(orient="records")
        return SpectrumPairDataset(x_arr, y_arr, energy=energy_arr, metadata=meta_list)

    train_ds = _build_dataset(train_df)
    val_ds = _build_dataset(val_df)

    if return_test:
        test_ds = _build_dataset(test_df)
        return train_ds, val_ds, test_ds

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
