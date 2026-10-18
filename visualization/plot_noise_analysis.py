"""Plots for the Gaussian-process spectral noise analysis results."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

AVAILABLE_NOISE_PLOTS: tuple[str, ...] = (
    "noise_vs_time",
    "snr_by_region",
    "noise_heatmap",
    "example_spectrum",
)


def _analysed_rows(df_meta: pd.DataFrame) -> pd.DataFrame:
    """Return the rows that carry noise analysis results."""
    required = {
        "noise_analysis_index", "gp_noise_sigma", "gp_snr", "tool", "time",
        "die", "region", "measurement_id", "spectrum_index",
    }
    missing = required - set(df_meta.columns)
    if missing:
        raise ValueError(
            f"df_meta is missing noise analysis columns {sorted(missing)}; run "
            "perform_spectral_noise_analysis first."
        )
    df = df_meta[df_meta["noise_analysis_index"].notna()]
    if df.empty:
        raise ValueError("df_meta contains no spectra with noise analysis results.")
    return df


def _plot_noise_vs_time(df: pd.DataFrame, metric: str) -> Figure:
    """Scatter the GP noise (or SNR) against measurement time per tool."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for tool, df_tool in df.groupby("tool"):
        ax.scatter(pd.to_datetime(df_tool["time"]), df_tool[metric], s=14, alpha=0.6, label=str(tool))
    ax.set_xlabel("Measurement time")
    ax.set_ylabel("GP noise sigma" if metric == "gp_noise_sigma" else "GP SNR")
    ax.set_title(f"{ax.get_ylabel()} vs. time")
    ax.legend(title="Tool")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _plot_snr_by_region(df: pd.DataFrame) -> Figure:
    """Draw SNR box plots per region, grouped by tool."""
    regions = sorted(df["region"].unique())
    tools = sorted(df["tool"].unique())
    width = 0.8 / len(tools)
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(regions)), 5))
    for tool_pos, tool in enumerate(tools):
        data = [df.loc[(df["region"] == r) & (df["tool"] == tool), "gp_snr"].to_numpy() for r in regions]
        positions = np.arange(len(regions)) + (tool_pos - (len(tools) - 1) / 2) * width
        box = ax.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True)
        color = f"C{tool_pos}"
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.plot([], [], color=color, linewidth=8, alpha=0.6, label=str(tool))
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels(regions)
    ax.set_ylabel("GP SNR")
    ax.set_title("SNR by region and tool")
    ax.legend(title="Tool")
    fig.tight_layout()
    return fig


def _plot_noise_heatmap(df: pd.DataFrame) -> Figure:
    """Draw die x region heatmaps of the mean GP noise, one panel per tool."""
    tools = sorted(df["tool"].unique())
    fig, axes = plt.subplots(1, len(tools), figsize=(5 * len(tools), 4.5), squeeze=False)
    vmax = float(df["gp_noise_sigma"].max())
    image = None
    for ax, tool in zip(axes[0], tools):
        pivot = df[df["tool"] == tool].pivot_table(
            index="die", columns="region", values="gp_noise_sigma", aggfunc="mean"
        )
        image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0.0, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Region")
        ax.set_ylabel("Die")
        ax.set_title(f"Tool {tool}: mean GP noise sigma")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="GP noise sigma")
    return fig


def _plot_example_spectrum(data: tuple[Any, ...], example: dict[str, Any]) -> Figure:
    """Plot one spectrum with its GP mean and confidence band."""
    ary_intensity, ary_energy, df, ary_signal, ary_noise = data
    row = df
    for key in ("measurement_id", "die", "region"):
        if example.get(key) is not None:
            row = row[row[key] == example[key]]
    if row.empty:
        raise ValueError(f"No analysed spectrum matches example selection {example}.")
    row = row.iloc[0]
    pos, idx = int(row["noise_analysis_index"]), int(row["spectrum_index"])
    energy, mean, half = ary_energy[idx], ary_signal[pos], ary_noise[pos]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(energy, ary_intensity[idx], ".", color="gray", markersize=4, label="Measured")
    ax.plot(energy, mean, color="C3", label="GP mean")
    ax.fill_between(energy, mean - half, mean + half, color="C3", alpha=0.25, label="GP confidence band")
    ax.set_xlabel("Binding energy (eV)")
    ax.set_ylabel("Intensity")
    ax.set_title(
        f"{row['measurement_id']} die {row['die']} {row['region']}: "
        f"noise={row['gp_noise_sigma']:.3g}, SNR={row['gp_snr']:.3g}"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def plot_spectral_noise_analysis(
    data: tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray],
    config: dict[str, Any] | None = None,
) -> dict[str, Figure]:
    """Plot how GP noise and SNR vary over time, tool, die and region.

    Parameters
    ----------
    data : tuple
        ``(ary_intensity, ary_energy, df_meta, ary_spec_analysis_signal,
        ary_spec_analysis_noise)`` where ``df_meta`` carries the columns added
        by ``perform_spectral_noise_analysis``.
    config : dict[str, Any] | None
        ``plots``: names to draw, subset of ``AVAILABLE_NOISE_PLOTS``
        (default all four); ``time_metric``: ``gp_noise_sigma`` (default) or
        ``gp_snr`` for the time plot; ``example``: dict with optional
        ``measurement_id``, ``die``, ``region`` for the example spectrum
        (default first analysed spectrum); ``save_dir``: directory where
        ``<plot>.png`` files are written (default None, not saved).

    Returns
    -------
    dict[str, Figure]
        The created figures keyed by plot name.
    """
    cfg = {"plots": list(AVAILABLE_NOISE_PLOTS), "time_metric": "gp_noise_sigma",
           "example": {}, "save_dir": None, **(config or {})}
    unknown = [p for p in cfg["plots"] if p not in AVAILABLE_NOISE_PLOTS]
    if unknown:
        raise ValueError(f"Unknown plots {unknown}; available: {AVAILABLE_NOISE_PLOTS}.")
    if cfg["time_metric"] not in ("gp_noise_sigma", "gp_snr"):
        raise ValueError(f"time_metric must be gp_noise_sigma or gp_snr, got {cfg['time_metric']!r}.")
    df = _analysed_rows(data[2])
    data = (data[0], data[1], df, data[3], data[4])
    builders = {
        "noise_vs_time": lambda: _plot_noise_vs_time(df, cfg["time_metric"]),
        "snr_by_region": lambda: _plot_snr_by_region(df),
        "noise_heatmap": lambda: _plot_noise_heatmap(df),
        "example_spectrum": lambda: _plot_example_spectrum(data, cfg["example"]),
    }
    figures = {name: builders[name]() for name in cfg["plots"]}
    if cfg["save_dir"] is not None:
        os.makedirs(cfg["save_dir"], exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(os.path.join(cfg["save_dir"], f"{name}.png"), dpi=150)
    return figures
