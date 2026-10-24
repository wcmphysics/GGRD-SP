"""Spectral plotting module for regional spectra, tool overlays, and patches."""

from __future__ import annotations

from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utility.patching import calculate_window_points, extract_sliding_windows
from visualization.common import (
    _check_meta,
    _finalize_plot,
    _finish_fig,
    _prepare_canvas,
)

__all__ = [
    "plot_regional_spectra",
    "plot_tool_comparison",
    "plot_sliding_window_slices",
]


def plot_regional_spectra(
    ary_energy: np.ndarray,
    ary_intensity: np.ndarray,
    df_meta: pd.DataFrame,
    plot_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot regional spectra filtered by metadata attributes.

    Parameters
    ----------
    ary_energy : np.ndarray
        2D array of binding energies, shape (N_total, N_points).
    ary_intensity : np.ndarray
        2D array of intensities, shape (N_total, N_points).
    df_meta : pd.DataFrame
        Metadata DataFrame containing spectrum details and 'spectrum_index'.
    plot_config : dict[str, Any] | None, optional
        Dictionary bundling plotting options:
        - 'filters' (dict[str, Any] | None): Key-value pairs to filter metadata
          (e.g., {'tool': 'J4', 'region': 'Al2p'}). If None, filters to the
          first available measurement session.
        - 'max_spectra' (int): Maximum spectra to display simultaneously
          (default 10).
        - 'invert_x' (bool): Invert x-axis per standard XPS convention
          (default True).
        - 'title' (str | None): Custom plot title.
        - 'ax' (plt.Axes | None): Matplotlib Axes to draw on.
        - 'show' (bool): Whether to invoke plt.show() (default False).

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The matplotlib Figure and Axes objects.

    Raises
    ------
    KeyError
        If filter key is not found in df_meta columns.
    ValueError
        If df_meta is empty or no spectra match specified filters.
    """
    if df_meta.empty:
        raise ValueError("df_meta cannot be empty")
    cfg, df = plot_config or {}, df_meta.copy()
    for k, v in (cfg.get("filters") or {}).items():
        if k not in df.columns:
            raise KeyError(
                f"Filter key '{k}' not found in df_meta columns: {list(df.columns)}"
            )
        df = df[df[k] == v]
    if not cfg.get("filters"):
        df = df[df["measurement_id"] == df["measurement_id"].iloc[0]]
    if df.empty:
        raise ValueError(
            f"No spectra match specified filters: {cfg.get('filters')}"
        )
    fig, ax = _prepare_canvas(cfg.get("ax"), figsize=(8, 5))
    df = df.iloc[: cfg.get("max_spectra", 10)]
    for _, r in df.iterrows():
        lbl = (
            f"{r.get('tool', 'Tool')} | {r.get('region', 'Region')} | "
            f"Die {r.get('die', '-')} | {r.get('measurement_id', '')}"
        )
        ax.plot(
            ary_energy[int(r["spectrum_index"])],
            ary_intensity[int(r["spectrum_index"])],
            label=lbl,
        )
    if len(df) <= 15:
        ax.legend(loc="best", fontsize="small")
    t_str = (
        f"XPS Spectra - Region: {', '.join(map(str, df['region'].unique()))} | "
        f"Tool: {', '.join(map(str, df['tool'].unique()))}"
    )
    _finalize_plot(
        ax,
        cfg,
        fig=fig,
        title=cfg.get("title") or t_str,
        xlabel="Binding Energy (eV)",
        ylabel="Intensity (counts / a.u.)",
        invert_x=cfg.get("invert_x", True),
    )
    return fig, ax


def plot_tool_comparison(
    ary_energy: np.ndarray,
    ary_intensity: np.ndarray,
    df_meta: pd.DataFrame,
    compare_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot an overlay comparison of spectra between two tools.

    Parameters
    ----------
    ary_energy : np.ndarray
        2D array of binding energies.
    ary_intensity : np.ndarray
        2D array of intensities.
    df_meta : pd.DataFrame
        Metadata DataFrame.
    compare_config : dict[str, Any] | None, optional
        Dictionary bundling comparison options:
        - 'region' (str): Target spectral region (default 'Al2p').
        - 'die' (int): Target die index (default 0).
        - 'tools' (list[str] | tuple[str, ...]): Tools to compare
          (default ('J4', 'J5')).
        - 'invert_x' (bool): Whether to invert x-axis per XPS convention
          (default True).
        - 'ax' (plt.Axes | None): Existing matplotlib Axes.
        - 'show' (bool): Whether to invoke plt.show() (default False).

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The matplotlib Figure and Axes objects.

    Raises
    ------
    KeyError
        If required metadata columns are missing.
    ValueError
        If df_meta is empty, fewer than 2 tools given, or no spectra match.
    """
    _check_meta(df_meta, {"region", "die", "tool", "spectrum_index"})
    cfg = compare_config or {}
    region = cfg.get("region", "Al2p")
    die = int(cfg.get("die", 0))
    tools = list(cfg.get("tools", ("J4", "J5")))
    if len(tools) < 2:
        raise ValueError(
            f"At least 2 tools required for comparison, got {len(tools)}: {tools}"
        )
    sub = df_meta[(df_meta["region"] == region) & (df_meta["die"] == die)]
    fig, ax = _prepare_canvas(cfg.get("ax"), figsize=(8, 5))
    plotted = 0
    for t in tools:
        if not (t_sub := sub[sub["tool"] == t]).empty:
            idx = int(t_sub.iloc[0]["spectrum_index"])
            m_id = t_sub.iloc[0].get("measurement_id", "")
            ax.plot(
                ary_energy[idx],
                ary_intensity[idx],
                label=f"Tool: {t} ({m_id})",
            )
            plotted += 1
    if plotted == 0:
        raise ValueError(
            f"No spectra found for tools {tools} at region='{region}', die={die}"
        )
    ax.legend(loc="best", fontsize="small")
    _finalize_plot(
        ax,
        cfg,
        fig=fig,
        title=cfg.get("title") or f"Tool Comparison - Region: {region} | Die: {die}",
        xlabel="Binding Energy (eV)",
        ylabel="Intensity (counts / a.u.)",
        invert_x=cfg.get("invert_x", True),
    )
    return fig, ax


def plot_sliding_window_slices(
    spectrum: np.ndarray,
    energy: np.ndarray | None = None,
    plot_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Visualize 1D spectrum alongside extracted sliding window slices.

    Parameters
    ----------
    spectrum : np.ndarray
        1D array of spectral intensity values, shape (N_points,).
    energy : np.ndarray | None, optional
        1D array of binding energies in eV matching spectrum length.
    plot_config : dict[str, Any] | None, optional
        Dictionary bundling options:
        - 'window_size' (int | float): Window width in pts or eV (default 2.0 eV
          or 15 pts).
        - 'stride' (int | float): Stride step in pts or eV (default 1.0 eV
          or 7 pts).
        - 'region' (str | None): Spectral region name.
        - 'offset_patches' (bool): Stagger slices vertically (default False).
        - 'invert_x' (bool): Invert x-axis per standard XPS convention.
        - 'title' (str | None): Custom figure supertitle.
        - 'axes' (tuple[plt.Axes, plt.Axes] | None): Pair of Axes.
        - 'show' (bool): Whether to invoke plt.show() (default False).

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Figure and tuple of (ax_spectrum, ax_patches).

    Raises
    ------
    ValueError
        If spectrum is empty, or if energy length does not match spectrum.
    """
    cfg, s1d = plot_config or {}, np.asarray(spectrum).ravel()
    if len(s1d) == 0:
        raise ValueError("Cannot plot sliding window slices for an empty spectrum.")
    if energy is not None:
        if len(e1d := np.asarray(energy).ravel()) != len(s1d):
            raise ValueError(
                f"Length mismatch: spectrum has {len(s1d)} points but energy has {len(e1d)} points."
            )
        w_size, s_step = calculate_window_points(e1d, config=cfg)
        x_vals, x_lbl = e1d, "Binding Energy (eV)"
    else:
        e1d, x_vals, x_lbl = None, np.arange(len(s1d)), "Data Point Index"
        w_size = int(cfg.get("window_size", 15))
        s_step = int(cfg.get("stride", max(1, 15 // 2)))

    windows, starts = extract_sliding_windows(
        s1d, window_size=w_size, stride=s_step
    )
    n_win = len(windows)
    has_anch = (
        ((len(s1d) - w_size) % s_step != 0)
        and n_win > 1
        and starts[-1] == len(s1d) - w_size
    )
    fig, (ax_top, ax_bot) = _prepare_canvas(
        cfg.get("axes"),
        (10, 7),
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2]},
    )
    cmap = plt.get_cmap("tab10" if n_win <= 10 else "plasma")
    colors = [
        cmap(i % 10 if n_win <= 10 else (i / max(1, n_win - 1)))
        for i in range(n_win)
    ]
    ax_top.plot(
        x_vals, s1d, color="#1a252f", lw=2.2, label="Original Spectrum", zorder=3
    )
    offset = (
        (float(np.ptp(s1d) or 1.0) * 0.15)
        if cfg.get("offset_patches", False)
        else 0.0
    )
    for i, (win, s_idx) in enumerate(zip(windows, starts)):
        x_w = x_vals[s_idx : s_idx + w_size]
        is_a = has_anch and (i == n_win - 1)
        ax_top.axvspan(
            min(x_w[0], x_w[-1]),
            max(x_w[0], x_w[-1]),
            facecolor=colors[i],
            alpha=0.25 if is_a else 0.15,
            ls="--" if is_a else "-",
            edgecolor=colors[i] if is_a else "none",
        )
        lbl = (
            f"Patch {i+1} ({'anchored' if is_a else f'pts {s_idx}:{s_idx+w_size}'})"
            if (n_win <= 8 or is_a)
            else None
        )
        ax_bot.plot(
            x_w,
            win + (i * offset),
            color=colors[i],
            lw=2.0 if is_a else 1.8,
            ls="--" if is_a else "-",
            marker=".",
            markersize=3,
            alpha=0.9 if is_a else 0.85,
            label=lbl,
        )
    reg = cfg.get("region")
    de_str = (
        rf", $\Delta E$={abs(e1d[-1] - e1d[0]):.2f} eV"
        if e1d is not None
        else ""
    )
    _finalize_plot(
        ax_top,
        title=f"Original Spectrum {f'({reg}) ' if reg else ''}[N={len(s1d)} pts{de_str}]",
        ylabel="Intensity (counts / a.u.)",
        invert_x=False,
        grid_alpha=0.5,
    )
    ax_top.legend(loc="upper right", framealpha=0.9)
    t_bot = (
        f"Extracted Sliding Windows (W={w_size} pts, S={s_step} pts, "
        f"Total Patches={n_win}{' [Right-Edge Anchored]' if has_anch else ''})"
    )
    _finalize_plot(
        ax_bot,
        title=t_bot,
        xlabel=x_lbl,
        ylabel="Patch Intensity"
        + (" (Waterfall Offset)" if cfg.get("offset_patches", False) else ""),
        invert_x=False,
        grid_alpha=0.5,
    )
    if n_win <= 8 or has_anch:
        ax_bot.legend(loc="upper right", fontsize=8, framealpha=0.9)
    if cfg.get("invert_x", energy is not None):
        for ax in (ax_top, ax_bot):
            if not ax.xaxis_inverted():
                ax.invert_xaxis()
    _finish_fig(
        fig,
        cfg,
        cfg.get(
            "title",
            f"Sliding Window Spectral Decomposition {f'({reg})' if reg else ''}",
        ),
    )
    return fig, (ax_top, ax_bot)
