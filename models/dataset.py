"""Dataset classes and session-level splitting utilities for
spectral transfer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utility.patching import extract_sliding_windows
from utility.quantification import compute_die_total_flux


def _to_scale_tensor(val: Any, n: int, eps: float) -> torch.Tensor:
    """Ensure normalization scale factor is formatted as a 2D float
    tensor.
    """
    if val is None: return torch.ones((n, 1), dtype=torch.float32)
    t = val.clone().detach() if isinstance(val, torch.Tensor) else torch.as_tensor(val, dtype=torch.float32)
    return torch.clamp(t.repeat(n, 1) if t.dim() == 0 else (t.unsqueeze(-1) if t.dim() == 1 else t), min=eps)


class SpectrumPairDataset(Dataset):
    """PyTorch Dataset yielding paired source and target spectral
    vectors.

    Parameters
    ----------
    x_data : np.ndarray | torch.Tensor
        Array of source intensities of shape (N_samples, N_points).
    y_data : np.ndarray | torch.Tensor
        Array of target intensities of shape (N_samples, N_points).
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'normalization_mode' (str): Normalization strategy:
          'source_referenced' (default), 'die_total_flux', or 'none'.
        - 'eps' (float): Epsilon floor to prevent division by zero
          (default 1e-4).
    energy : np.ndarray | torch.Tensor | None, optional
        Array of binding energies of shape (N_samples, N_points).
    metadata : list[dict[str, Any]] | None, optional
        List of metadata dictionaries per sample.
    scale_x : np.ndarray | torch.Tensor | None, optional
        Precomputed source scale factors of shape (N_samples, 1).
    scale_y : np.ndarray | torch.Tensor | None, optional
        Precomputed target scale factors of shape (N_samples, 1).
    **kwargs : Any
        Optional keyword arguments for backward compatibility.
    """

    def __init__(
        self,
        x_data: np.ndarray | torch.Tensor,
        y_data: np.ndarray | torch.Tensor,
        config: dict[str, Any] | None = None,
        *,
        energy: np.ndarray | torch.Tensor | None = None,
        metadata: list[dict[str, Any]] | None = None,
        scale_x: np.ndarray | torch.Tensor | None = None,
        scale_y: np.ndarray | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        if len(x_data) != len(y_data): raise ValueError(f"Lengths differ: {len(x_data)} vs {len(y_data)}")

        self.x, self.y = torch.as_tensor(x_data, dtype=torch.float32), torch.as_tensor(y_data, dtype=torch.float32)
        cfg = {**(config or {}), **kwargs}

        if energy is None: energy = cfg.get("energy")
        if metadata is None: metadata = cfg.get("metadata")
        if scale_x is None: scale_x = cfg.get("scale_x")
        if scale_y is None: scale_y = cfg.get("scale_y")

        self.normalization_mode = str(cfg.get("normalization_mode", "source_referenced")).lower()
        eps: float = float(cfg.get("eps", 1e-4))

        if self.normalization_mode == "die_total_flux":
            if scale_x is None:
                raise ValueError(f"normalization_mode='{self.normalization_mode}' requires 'scale_x' to be provided.")
            self.scale_x = _to_scale_tensor(scale_x, len(self.x), eps)
            self.scale_y = _to_scale_tensor(scale_y if scale_y is not None else scale_x, len(self.y), eps)
            self.x, self.y = self.x / self.scale_x, self.y / self.scale_y
        elif self.normalization_mode == "source_referenced" and len(self.x) > 0:
            max_abs = self.x.abs().amax(dim=-1, keepdim=True) if self.x.dim() > 1 else self.x.abs().amax()
            self.scale_x = self.scale_y = torch.clamp(max_abs, min=eps)
            self.x, self.y = self.x / self.scale_x, self.y / self.scale_y
        else:
            self.scale_x = self.scale_y = torch.ones((len(self.x), 1)) if self.x.dim() > 1 else torch.tensor(1.0)

        self.energy = torch.as_tensor(energy, dtype=torch.float32) if energy is not None else None
        self.metadata = metadata or []

    def __len__(self) -> int: return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sx = self.scale_x[idx] if self.scale_x.numel() > 1 else self.scale_x
        sy = self.scale_y[idx] if self.scale_y.numel() > 1 else sx
        item: dict[str, Any] = {"x": self.x[idx], "y": self.y[idx], "scale_x": sx, "scale_y": sy}
        if self.energy is not None: item["energy"] = self.energy[idx] if self.energy.dim() > 1 else self.energy
        if self.metadata and idx < len(self.metadata): item["meta"] = self.metadata[idx]
        return item

    def get_scale(self, idx: int) -> float:
        """Return source normalization scalar for a given sample."""
        return float(self.scale_x[idx].item() if self.scale_x.numel() > 1 else self.scale_x.item())

    def get_target_scale(self, idx: int) -> float:
        """Return target normalization scalar for a given sample."""
        return float(self.scale_y[idx].item() if self.scale_y.numel() > 1 else self.scale_y.item())

    def unnormalize_y(self, y_norm: torch.Tensor | np.ndarray, idx: int) -> torch.Tensor | np.ndarray:
        """Scale normalized target prediction to physical counts."""
        return y_norm * self.get_scale(idx)


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

    def __init__(self, full_dataset: SpectrumPairDataset, window_size: int, stride: int) -> None:
        self.window_size, self.stride = window_size, stride
        x_patches_list, y_patches_list, max_vals_list = [], [], []
        meta_list, scale_x_list, scale_y_list = [], [], []

        for idx in range(len(full_dataset)):
            item = full_dataset[idx]
            x_win, _ = extract_sliding_windows(item["x"].cpu().numpy(), window_size, stride)
            y_win, _ = extract_sliding_windows(item["y"].cpu().numpy(), window_size, stride)
            x_patches_list.append(x_win)
            y_patches_list.append(y_win)
            max_vals_list.extend([max(float(np.max(np.abs(item["y"].cpu().numpy()))), 1e-4)] * len(x_win))
            for k, lst in (("meta", meta_list), ("scale_x", scale_x_list), ("scale_y", scale_y_list)):
                if k in item: lst.extend([item[k]] * len(x_win))

        if not x_patches_list: raise ValueError("No patches could be extracted from the dataset.")

        self.x = torch.from_numpy(np.vstack(x_patches_list).astype(np.float32))
        self.y = torch.from_numpy(np.vstack(y_patches_list).astype(np.float32))
        self.max_vals = torch.tensor(max_vals_list, dtype=torch.float32).unsqueeze(1)
        self.meta_list, self.scale_x_list, self.scale_y_list = meta_list, scale_x_list, scale_y_list

    def __len__(self) -> int: return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ret: dict[str, Any] = {"x": self.x[idx], "y": self.y[idx], "max_val": self.max_vals[idx]}
        for key, lst in (("meta", self.meta_list), ("scale_x", self.scale_x_list), ("scale_y", self.scale_y_list)):
            if lst and idx < len(lst): ret[key] = lst[idx]
        return ret


def partition_measurement_sessions(
    meta_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, set[str]]:
    """Partition unique measurement sessions globally across regions.

    Guarantees physical sessions are assigned to one and only one split
    ('train', 'val', 'test') across all regions without data leakage.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Metadata DataFrame containing source-to-target pairing columns.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'train_ratio' (float): Fraction for training (default 0.5).
        - 'val_ratio' (float): Fraction for validation (default 0.2).
        - 'test_ratio' (float): Fraction for test (default 0.3).
        - 'source_tool' (str | None): Source tool filter.
        - 'target_tool' (str | None): Target tool filter.
        - 'seed' (int | None): Random seed for permutation (default 42).

    Returns
    -------
    dict[str, set[str]]
        Dictionary with keys 'train', 'val', 'test' mapping to session
        IDs.
    """
    cfg = config or {}
    source_tool, target_tool = cfg.get("source_tool"), cfg.get("target_tool")
    seed: int | None = cfg.get("seed", 42)

    condition = meta_df["measurement_id_target"].notna()
    if source_tool: condition = condition & (meta_df["tool"] == source_tool)
    if target_tool: condition = condition & (meta_df["tool_target"] == target_tool)

    paired_df = meta_df[condition].copy()
    if paired_df.empty:
        raise ValueError(f"No paired records found for source_tool='{source_tool}', target_tool='{target_tool}'")

    sessions = np.array(sorted(paired_df["measurement_id"].unique()))
    n_sessions = len(sessions)
    ratios = (float(cfg.get("train_ratio", 0.5)), float(cfg.get("val_ratio", 0.2)), float(cfg.get("test_ratio", 0.3)))
    if any(r < 0.0 for r in ratios):
        raise ValueError(f"Split ratios must be non-negative: {ratios}")
    if sum(ratios) <= 0.0:
        raise ValueError("Sum of train_ratio, val_ratio, and test_ratio must be positive.")

    train_ratio, val_ratio, test_ratio = [r / sum(ratios) for r in ratios]
    sess = [str(s) for s in np.random.default_rng(seed).permutation(sessions)]

    if test_ratio > 0.0 and val_ratio > 0.0:
        if n_sessions < 3:
            raise ValueError(f"Need >= 3 distinct sessions for train/val/test, found {n_sessions}")
        n_tr = max(1, min(n_sessions - 2, int(round(n_sessions * train_ratio))))
        n_va = max(1, min(n_sessions - n_tr - 1, int(round(n_sessions * val_ratio))))
        tr_s, va_s, te_s = set(sess[:n_tr]), set(sess[n_tr : n_tr + n_va]), set(sess[n_tr + n_va :])
    else:
        if n_sessions < 2:
            raise ValueError(f"Need at least 2 distinct measurement sessions to split, found {n_sessions}")
        n_eval = max(1, min(n_sessions - 1, int(round(n_sessions * (val_ratio if val_ratio > 0.0 else test_ratio)))))
        tr_s, ev_s = set(sess[: n_sessions - n_eval]), set(sess[n_sessions - n_eval :])
        va_s, te_s = (ev_s, set()) if val_ratio > 0.0 else (set(), ev_s)

    splits_dict = {"train": tr_s, "val": va_s, "test": te_s}
    target_sessions = {
        k: set(paired_df[paired_df["measurement_id"].isin(s)]["measurement_id_target"].dropna())
        for k, s in splits_dict.items()
    }
    for s1, s2 in (("train", "val"), ("train", "test"), ("val", "test")):
        leak = target_sessions[s1] & target_sessions[s2]
        if leak:
            raise ValueError(f"Target measurement sessions leak across {s1} and {s2} splits: {leak}")

    return splits_dict


def split_session_datasets(
    meta_df: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[SpectrumPairDataset, ...]:
    """Split paired spectra into train, val, and test datasets by
    session.

    Guarantees dies belonging to the same measurement session stay in
    the same split, preventing cross-die and cross-region data leakage.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Metadata DataFrame containing pairing columns.
    ary_intensity : np.ndarray
        2D array of measured intensities of shape (N_total, N_points).
    ary_energy : np.ndarray
        2D array of binding energies of shape (N_total, N_points).
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'region' (str): Spectral region to extract (e.g., 'Al2p').
        - 'session_splits' (dict[str, set[str]] | None): Pre-partitioned
          sets.
        - 'return_test' (bool): Whether to return test_dataset
          (default False).
        - 'normalization_mode' (str): Normalization mode
          (default 'source_referenced').
        - Split ratios and tool filters.

    Returns
    -------
    tuple[SpectrumPairDataset, ...]
        (train_dataset, val_dataset) or (train_ds, val_ds, test_ds).
    """
    cfg = config or {}
    region = cfg.get("region", "Al2p")
    source_tool, target_tool = cfg.get("source_tool"), cfg.get("target_tool")
    return_test = bool(cfg.get("return_test", False))
    norm_mode = str(cfg.get("normalization_mode", "source_referenced")).lower()
    eps_val: float = float(cfg.get("eps", 1e-4))

    if len(ary_intensity) != len(ary_energy):
        raise ValueError(
            f"ary_intensity and ary_energy must have the same rows, got {len(ary_intensity)} vs {len(ary_energy)}"
        )

    condition = meta_df["measurement_id_target"].notna() & (meta_df["region"] == region)
    if source_tool: condition = condition & (meta_df["tool"] == source_tool)
    if target_tool: condition = condition & (meta_df["tool_target"] == target_tool)

    paired_df = meta_df[condition].copy()
    if paired_df.empty:
        raise ValueError(f"No paired records for {region=}, {source_tool=}, {target_tool=}")

    split_cfg = cfg if ("test_ratio" in cfg or return_test) else {**cfg, "test_ratio": 0.0}
    session_splits = cfg.get("session_splits") or partition_measurement_sessions(meta_df, split_cfg)

    n_spectra = len(ary_intensity)
    use_flux = norm_mode in ("die_total_flux", "die_flux")
    flux_map = compute_die_total_flux(meta_df, ary_intensity, ary_energy) if use_flux else {}

    def _build_dataset(df_subset: pd.DataFrame) -> SpectrumPairDataset:
        x_idx = df_subset["spectrum_index"].astype(int).to_numpy()
        y_idx = df_subset["spectrum_index_target"].astype(int).to_numpy()
        if len(x_idx) > 0 and (min(x_idx.min(), y_idx.min()) < 0 or max(x_idx.max(), y_idx.max()) >= n_spectra):
            raise IndexError(f"spectrum indices out of bounds [0, {n_spectra - 1}]")

        scale_x_arr = scale_y_arr = None
        if use_flux and flux_map and len(df_subset) > 0:
            def _lookup(m_col: str, idx_col: str) -> np.ndarray:
                def _val(r: pd.Series) -> float:
                    fallback = max(float(np.abs(ary_intensity[int(r[idx_col])]).max()), eps_val)
                    return float(flux_map.get((str(r[m_col]), int(r["die"])), fallback))
                return np.array([_val(r) for _, r in df_subset.iterrows()], dtype=np.float32)[:, None]
            scale_x_arr = _lookup("measurement_id", "spectrum_index")
            scale_y_arr = _lookup("measurement_id_target", "spectrum_index_target")

        return SpectrumPairDataset(
            ary_intensity[x_idx], ary_intensity[y_idx], energy=ary_energy[x_idx],
            metadata=df_subset.to_dict(orient="records"),
            config={"normalization_mode": norm_mode, "eps": eps_val},
            scale_x=scale_x_arr, scale_y=scale_y_arr,
        )

    datasets = tuple(
        _build_dataset(paired_df[paired_df["measurement_id"].isin(session_splits.get(k, set()))])
        for k in (("train", "val", "test") if return_test else ("train", "val"))
    )
    return datasets


def default_spectrum_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function stacking tensors and safely keeping metadata
    lists.
    """
    if not batch: return {}
    def _c(val: Any, k: str) -> Any:
        if isinstance(val, torch.Tensor): return torch.stack([b[k] for b in batch])
        if isinstance(val, np.ndarray): return torch.from_numpy(np.stack([b[k] for b in batch]))
        return [b.get(k) for b in batch]
    return {k: _c(v, k) for k, v in batch[0].items()}


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
    kw = dict(
        batch_size=int(cfg.get("batch_size", 16)), num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=cfg.get("collate_fn", default_spectrum_collate),
    )
    train_loader = DataLoader(train_dataset, shuffle=bool(cfg.get("shuffle_train", True)), **kw)
    return train_loader, DataLoader(val_dataset, shuffle=False, **kw)
