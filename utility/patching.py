"""Spectral patching and sliding window geometric transformations.

Provides sequence slicing, window-to-point calculations, and overlap reconstruction
utilities independent of neural network models.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch


def calculate_window_points(
    energy: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Calculate window size and sliding stride in data points from binding energy grid.

    Parameters
    ----------
    energy : np.ndarray
        1D array of binding energies for the spectral region.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'window_size_ev' (float): Desired window width in eV (default 2.0).
        - 'sliding_stride_ev' (float): Desired sliding stride in eV (default 1.0).
        - 'window_size_points' | 'window_size' (int | None): Explicit window size in points.
        - 'sliding_stride_points' | 'stride' (int | None): Explicit stride in points.
        - 'force_odd_window' (bool): Ensure window size is an odd integer (default True).

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
    w_pts = cfg.get("window_size_points", cfg.get("window_size"))
    s_pts = cfg.get("sliding_stride_points", cfg.get("stride"))

    n_total = len(energy)
    if n_total < 3:
        raise ValueError(f"Energy array length must be >= 3, got {n_total}")

    delta_e = float(np.mean(np.abs(np.diff(energy))))
    if delta_e <= 0.0:
        raise ValueError("Energy spacing delta_e must be strictly positive.")

    # Determine window size in points
    if w_pts is not None:
        window_size = int(w_pts)
    else:
        window_size_ev = float(cfg.get("window_size_ev", 2.0))
        window_size = int(round(window_size_ev / delta_e))

    force_odd = bool(cfg.get("force_odd_window", True))
    window_size = min(n_total, window_size)
    if force_odd and window_size % 2 == 0:
        window_size = window_size - 1 if window_size == n_total else window_size + 1

    window_size = max(3, window_size)

    # Determine sliding stride in points
    if s_pts is not None:
        stride = int(s_pts)
    else:
        stride_ev = float(cfg.get("sliding_stride_ev", 1.0))
        stride = int(round(stride_ev / delta_e))

    stride = max(1, min(window_size, stride))

    return window_size, stride


def extract_sliding_windows(
    spectrum: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, list[int]]:
    """Slice a 1D regional spectrum into overlapping window patches without padding.

    Extracts valid sub-sequences [start : start + window_size] directly from the
    unpadded spectrum. Anchors the final window to the right edge (N - window_size)
    if not aligned by stride, guaranteeing complete 100% spectral coverage using
    only real physical measurements without artificial boundary padding.

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
        If window_size < 1, window_size > len(spectrum), stride < 1, or stride > window_size.
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

    start_indices: list[int] = []
    current_idx = 0
    while current_idx + window_size <= n_points:
        start_indices.append(current_idx)
        current_idx += stride

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
    **kwargs: Any,
) -> np.ndarray:
    """Reconstruct a full 1D spectrum by accumulating and averaging overlapping predicted patches.

    Bundles optional parameters into config dictionary adhering to project interface rules.

    Parameters
    ----------
    patches : np.ndarray | torch.Tensor
        2D array of predicted window patches of shape (N_windows, window_size).
    start_indices : list[int]
        Starting index of each window in the spectrum sequence.
    original_length : int
        Length of the target spectrum (N).
    config : dict[str, Any] | None, optional
        Configuration dictionary containing optional 'window_size'.
    **kwargs : Any
        Optional keyword arguments (e.g. window_size) for backward compatibility.

    Returns
    -------
    np.ndarray
        Reconstructed 1D spectrum of length original_length.
    """
    cfg = dict(config or {})
    cfg.update(kwargs)

    if isinstance(patches, torch.Tensor):
        patches_np = patches.detach().cpu().numpy()
    else:
        patches_np = np.asarray(patches)
    patches_np = np.atleast_2d(patches_np)

    w_size = int(cfg.get("window_size", patches_np.shape[1]))

    accum = np.zeros(original_length, dtype=np.float32)
    counts = np.zeros(original_length, dtype=np.float32)

    for patch, j in zip(patches_np, start_indices):
        accum[j : j + w_size] += patch
        counts[j : j + w_size] += 1.0

    if original_length > 0 and np.any(counts == 0.0):
        missing = np.where(counts == 0.0)[0]
        warnings.warn(
            f"Patch reconstruction left {len(missing)} uncovered points in spectrum (e.g. index {missing[0]})."
        )

    return accum / np.maximum(counts, 1.0)
