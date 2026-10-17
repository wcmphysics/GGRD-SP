"""Spectral patching and sliding window geometric transformations.

Provides sequence slicing, window-to-point calculations, and overlap
reconstruction utilities independent of neural network models.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch


def _resolve_step_points(
    cfg: dict[str, Any],
    delta_e: float,
    pts_keys: tuple[str, str],
    ev_spec: tuple[str, float],
) -> int:
    """Resolve a step or window size in data points from points or eV."""
    primary_pts_key, alias_pts_key = pts_keys
    pts_val = cfg.get(primary_pts_key, cfg.get(alias_pts_key))
    if pts_val is not None:
        points = int(pts_val)
        if points <= 0:
            raise ValueError(
                f"{primary_pts_key} must be strictly positive, got {points}."
            )
        return points

    ev_key, default_ev = ev_spec
    ev_val = float(cfg.get(ev_key, default_ev))
    if ev_val <= 0.0:
        raise ValueError(f"{ev_key} must be strictly positive, got {ev_val}.")
    return int(round(ev_val / delta_e))


def calculate_window_points(
    energy: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Calculate window size and stride in data points from binding energy grid.

    Parameters
    ----------
    energy : np.ndarray
        1D array of binding energies for the spectral region.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'window_size_ev' (float): Window width in eV (default 2.0).
        - 'sliding_stride_ev' (float): Sliding stride in eV (default 1.0).
        - 'window_size_points' | 'window_size' (int | None): Explicit window
          size in points.
        - 'sliding_stride_points' | 'stride' (int | None): Explicit stride in
          points.
        - 'force_odd_window' (bool): Ensure window size is odd (default True).

    Returns
    -------
    tuple[int, int]
        (window_size_points, sliding_stride_points)

    Raises
    ------
    ValueError
        If energy length < 3 or spacing delta_e <= 0.
    """
    cfg = config or {}
    n_total = len(energy)
    if n_total < 3:
        raise ValueError(f"Energy array length must be >= 3, got {n_total}")

    delta_e = float(np.mean(np.abs(np.diff(energy))))
    if delta_e <= 0.0:
        raise ValueError("Energy spacing delta_e must be strictly positive.")

    window_size = _resolve_step_points(
        cfg,
        delta_e,
        ("window_size_points", "window_size"),
        ("window_size_ev", 2.0),
    )
    force_odd = bool(cfg.get("force_odd_window", True))
    window_size = min(n_total, window_size)
    if force_odd and window_size % 2 == 0:
        window_size = (
            window_size - 1 if window_size == n_total else window_size + 1
        )
    window_size = max(3, window_size)

    stride = _resolve_step_points(
        cfg,
        delta_e,
        ("sliding_stride_points", "stride"),
        ("sliding_stride_ev", 1.0),
    )
    stride = max(1, min(window_size, stride))

    return window_size, stride



def extract_sliding_windows(
    spectrum: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, list[int]]:
    """Slice a 1D regional spectrum into overlapping window patches.

    Extracts valid sub-sequences [start : start + window_size] directly from
    the unpadded spectrum. Anchors the final window to the right edge
    (N - window_size) if not aligned by stride, guaranteeing complete 100%
    spectral coverage using only real physical measurements without artificial
    boundary padding.

    Parameters
    ----------
    spectrum : np.ndarray
        1D array of spectral intensities of length N (squeezed/raveled if 2D).
    window_size : int
        Number of points in each window (W).
    stride : int
        Step size in points between consecutive window starts (S).

    Returns
    -------
    tuple[np.ndarray, list[int]]
        - windows: 2D array of shape (N_windows, window_size).
        - start_indices: List of starting indices in the spectrum.

    Raises
    ------
    ValueError
        If window_size < 1, window_size > len(spectrum), stride < 1, or
        stride > window_size.
    """
    spectrum_1d = np.asarray(spectrum).ravel()
    n_points = len(spectrum_1d)
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if window_size > n_points:
        raise ValueError(
            f"window_size ({window_size}) cannot exceed spectrum length ({n_points})"
        )
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if stride > window_size:
        raise ValueError(
            f"stride ({stride}) cannot exceed window_size ({window_size}) "
            "as it creates reconstruction gaps."
        )

    start_indices = list(range(0, n_points - window_size + 1, stride))

    # Ensure the rightmost edge of the spectrum is fully covered
    last_possible = n_points - window_size
    if start_indices and start_indices[-1] < last_possible:
        start_indices.append(last_possible)

    windows_list = [spectrum_1d[j : j + window_size] for j in start_indices]
    windows = np.vstack(windows_list).astype(np.float32)

    return windows, start_indices


def reconstruct_from_patches(
    patches: np.ndarray | torch.Tensor,
    start_indices: list[int],
    original_length: int,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Reconstruct a 1D spectrum by averaging overlapping predicted patches.

    Bundles optional parameters into config dictionary adhering to project
    interface rules.

    Parameters
    ----------
    patches : np.ndarray | torch.Tensor
        2D array of predicted window patches of shape (N_windows, window_size).
    start_indices : list[int]
        Starting index of each window in the spectrum sequence.
    original_length : int
        Length of the target spectrum (N).
    config : dict[str, Any] | None, optional
        Configuration dictionary containing:
        - 'window_size' (int | None): Window width in points.
        - 'clamp_non_negative' (bool): Whether to clamp final reconstructed
          spectrum to non-negative (default False).

    Returns
    -------
    np.ndarray
        Reconstructed 1D spectrum of length original_length.
    """
    cfg = dict(config or {})

    if isinstance(patches, torch.Tensor):
        patches_np = patches.detach().cpu().numpy()
    else:
        patches_np = np.asarray(patches)
    patches_np = np.atleast_2d(patches_np)

    w_size = int(cfg.get("window_size", patches_np.shape[1]))
    clamp_non_negative = bool(cfg.get("clamp_non_negative", False))

    accum = np.zeros(original_length, dtype=np.float32)
    counts = np.zeros(original_length, dtype=np.float32)

    for patch, j in zip(patches_np, start_indices):
        accum[j : j + w_size] += patch
        counts[j : j + w_size] += 1.0

    if original_length > 0 and np.any(counts == 0.0):
        missing = np.where(counts == 0.0)[0]
        warnings.warn(
            f"Patch reconstruction left {len(missing)} uncovered points in "
            f"spectrum (e.g. index {missing[0]})."
        )

    reconstructed = accum / np.maximum(counts, 1.0)
    if clamp_non_negative:
        reconstructed = np.clip(reconstructed, 0.0, None)

    return reconstructed

