"""Common plotting infrastructure, canvas management, and style utilities."""

from __future__ import annotations

import colorsys
from typing import Any
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "_prepare_canvas",
    "_setup_subplot_grid",
    "_finalize_plot",
    "_finish_fig",
    "_check_meta",
    "_derive_tool_die_shades",
]


def _prepare_canvas(
    ax: Any = None,
    figsize: tuple[float, float] = (8, 5),
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: Any,
) -> tuple[plt.Figure, Any]:
    """Return figure and axes, creating new ones if ax is None."""
    if ax is not None:
        return (
            ax.get_figure() if hasattr(ax, "get_figure") else ax[0].get_figure(),
            ax,
        )
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


def _setup_subplot_grid(
    n_items: int,
    max_cols: int = 3,
    col_w: float = 5.0,
    row_h: float = 3.8,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    """Create a standard grid of subplots for multi-region visualization."""
    nc = min(max_cols, max(1, n_items))
    nr = (n_items + nc - 1) // nc
    fig, axes = plt.subplots(
        nr, nc, figsize=figsize or (col_w * nc, row_h * nr), squeeze=False
    )
    for ax in axes.flatten()[n_items:]:
        ax.set_visible(False)
    return fig, axes, axes.flatten()


def _finalize_plot(
    ax: plt.Axes,
    config: dict[str, Any] | None = None,
    *,
    fig: plt.Figure | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    invert_x: bool = True,
    grid: bool = True,
    grid_alpha: float = 0.6,
    show: bool = False,
    **kwargs: Any,
) -> None:
    """Apply standard plot styling, labels, inverted x-axis, and display."""
    cfg = {**kwargs, **(config or {})}
    for val, setter in [
        (title or cfg.get("title"), ax.set_title),
        (xlabel or cfg.get("xlabel"), ax.set_xlabel),
        (ylabel or cfg.get("ylabel"), ax.set_ylabel),
    ]:
        if val:
            setter(val)
    if cfg.get("invert_x", invert_x) and not ax.xaxis_inverted():
        ax.invert_xaxis()
    if cfg.get("grid", grid):
        ax.grid(
            True, linestyle="--", alpha=float(cfg.get("grid_alpha", grid_alpha))
        )
    if fig is not None:
        fig.tight_layout()
    if cfg.get("show", show):
        plt.show()


def _finish_fig(
    fig: plt.Figure,
    cfg: dict[str, Any],
    title: str | None = None,
) -> None:
    """Apply figure suptitle, tight layout, and optional show."""
    if t := (title if title is not None else cfg.get("title")):
        fig.suptitle(t, fontsize=12, fontweight="bold")
    fig.tight_layout()
    if cfg.get("show"):
        plt.show()


def _check_meta(
    df: pd.DataFrame,
    req_cols: set[str],
    name: str = "df_meta",
) -> None:
    """Validate DataFrame presence and required column existence."""
    if df.empty:
        raise ValueError(f"{name} cannot be empty.")
    if missing := (req_cols - set(df.columns)):
        raise KeyError(f"{name} missing columns: {sorted(missing)}")


def _derive_tool_die_shades(base_color: Any) -> dict[str, str]:
    """Derive 3-tier colors (center, donut, edge) from base color."""
    if (norm := str(base_color).lower().strip()) in (
        "black",
        "#000000",
        "#000",
        "k",
    ):
        return {"center": "#000000", "donut": "#555555", "edge": "#999999"}
    if norm in ("red", "#ff0000", "#f00", "tab:red", "#d62728"):
        return {"center": "#8b0000", "donut": "#d62728", "edge": "#f4a582"}
    try:
        h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
        if s < 0.08:
            return {
                "center": mcolors.to_hex((max(0.0, l * 0.4),) * 3),
                "donut": mcolors.to_hex((l,) * 3),
                "edge": mcolors.to_hex((min(0.88, l + (1.0 - l) * 0.55),) * 3),
            }
        return {
            "center": mcolors.to_hex(
                colorsys.hls_to_rgb(h, max(0.12, l * 0.55), min(1.0, s * 1.15))
            ),
            "donut": mcolors.to_hex(colorsys.hls_to_rgb(h, l, s)),
            "edge": mcolors.to_hex(
                colorsys.hls_to_rgb(
                    h, min(0.85, l + (1.0 - l) * 0.55), max(0.2, s * 0.75)
                )
            ),
        }
    except Exception:
        return {"center": "#333333", "donut": "#777777", "edge": "#bbbbbb"}
