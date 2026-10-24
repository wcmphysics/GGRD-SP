"""Training loss progression and validation history visualization."""

from __future__ import annotations

from typing import Any
import matplotlib.pyplot as plt
import numpy as np

from visualization.common import (
    _finalize_plot,
    _finish_fig,
    _setup_subplot_grid,
)

__all__ = [
    "plot_training_history",
]


def plot_training_history(
    histories: dict[str, dict[str, list[float]]],
    plot_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, np.ndarray | plt.Axes]:
    """Plot training and validation loss curves over epochs per region.

    Parameters
    ----------
    histories : dict[str, dict[str, list[float]]]
        Dictionary mapping region to history dict with 'train_loss', 'val_loss'.
    plot_config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'log_scale' (bool): Logarithmic loss scale (default True).
        - 'title' (str | None): Overall figure supertitle.
        - 'show' (bool): Whether to call plt.show() (default False).

    Returns
    -------
    tuple[plt.Figure, np.ndarray | plt.Axes]
        Figure and axes array.
    """
    cfg, regs = plot_config or {}, list(histories.keys())
    if not regs:
        raise ValueError("Histories dictionary is empty, no training curves to plot.")
    log_scale = bool(cfg.get("log_scale", True))
    fig, axes, axes_flat = _setup_subplot_grid(
        len(regs), max_cols=3, col_w=5.0, row_h=3.8
    )
    for idx, reg in enumerate(regs):
        ax, h = axes_flat[idx], histories[reg]
        ax.plot(
            range(1, len(h.get("train_loss", [])) + 1),
            h.get("train_loss", []),
            label="Train Loss",
            color="royalblue",
            lw=1.8,
        )
        if h.get("val_loss"):
            ax.plot(
                range(1, len(h["val_loss"]) + 1),
                h["val_loss"],
                label="Val Loss",
                color="darkorange",
                ls="--",
                lw=1.8,
            )
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize="small")
        _finalize_plot(
            ax,
            title=f"Region: {reg}",
            xlabel="Epoch",
            ylabel="Loss (log scale)"
            if log_scale
            else "Loss (Normalized MSE)",
            invert_x=False,
            grid_alpha=0.6,
        )
    _finish_fig(
        fig,
        cfg,
        cfg.get("title", "Baseline Neural Network Training History"),
    )
    return fig, axes
