"""Plots for spectral noise analysis and denoising comparisons."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from denoise import trace_gp_hyperparameter_convergence
from visualization.common import (
    _finalize_plot,
    _finish_fig,
    _prepare_canvas,
    _setup_subplot_grid,
)

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


def plot_gp_hyperparameter_convergence(
    data: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    plot_config: dict[str, Any] | None = None,
) -> Figure:
    """Plot the optimization convergence of GP hyperparameters over iterations.

    Generates a 4-panel figure showing:
      (a) Negative Marginal Log-Likelihood (-MLL)
      (b) Learned Noise Sigma (physical intensity units)
      (c) Learned Kernel Lengthscale (eV)
      (d) Relative Error of Noise Sigma vs. Final Iteration (%)

    Parameters
    ----------
    data : tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Tuple of (ary_intensity, ary_energy, df_meta).
    plot_config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'measurement_id' (str | None): Measurement session to sample
          (default: first available session).
        - 'die' (int): Die index to sample (default: 0).
        - 'tool' (str | None): Optional tool filter.
        - 'regions' (list[str] | None): Regions to plot (default: all
          available in the session).
        - 'num_iterations' (int): Number of Adam iterations to track
          (default: 100).
        - 'learning_rate' (float): Adam learning rate (default: 0.1).
        - 'kernel' (dict[str, Any] | None): GP kernel configuration.
        - 'noise_model' (str): Noise model (default: 'homoscedastic').
        - 'save_path' (str | None): Path to save the PNG file (default:
          'gp_hyperparameter_convergence.png').
        - 'show' (bool): Whether to display the plot with plt.show()
          (default: False).

    Returns
    -------
    Figure
        The created 4-panel matplotlib Figure.
    """
    if not (isinstance(data, tuple) and len(data) == 3):
        raise ValueError(
            "data must be a 3-tuple (ary_intensity, ary_energy, df_meta)."
        )

    ary_intensity, ary_energy, df_meta = data
    if df_meta.empty:
        raise ValueError("df_meta is empty.")

    cfg = plot_config or {}
    die = int(cfg.get("die", 0))
    meas_id = cfg.get("measurement_id")
    tool = cfg.get("tool")
    num_iterations = int(cfg.get("num_iterations", 100))
    learning_rate = float(cfg.get("learning_rate", 0.1))
    noise_model = str(cfg.get("noise_model", "homoscedastic"))
    save_path = cfg.get("save_path", "gp_hyperparameter_convergence.png")
    show = bool(cfg.get("show", False))

    cond = df_meta["die"] == die
    if meas_id is not None:
        cond = cond & (df_meta["measurement_id"] == str(meas_id))
    if tool is not None:
        cond = cond & (df_meta["tool"] == str(tool))

    matching_df = df_meta[cond]
    if matching_df.empty:
        raise ValueError(
            f"No matching spectra found for measurement_id={meas_id!r}, "
            f"die={die}, tool={tool!r}."
        )

    if meas_id is None:
        meas_id = str(matching_df["measurement_id"].iloc[0])
        matching_df = matching_df[matching_df["measurement_id"] == meas_id]

    available_regions = list(matching_df["region"].unique())
    requested_regions = cfg.get("regions")
    if requested_regions is not None:
        missing = set(requested_regions) - set(available_regions)
        if missing:
            raise ValueError(
                f"Requested regions {sorted(missing)} not found in session "
                f"'{meas_id}' die {die}; available: {sorted(available_regions)}."
            )
        regions = list(requested_regions)
    else:
        regions = available_regions

    trace_cfg: dict[str, Any] = {
        "noise_model": noise_model,
        "training": {
            "num_iterations": num_iterations,
            "learning_rate": learning_rate,
        },
        "device": "cpu",
    }
    if "kernel" in cfg:
        trace_cfg["kernel"] = cfg["kernel"]

    history_data: dict[str, dict[str, list[float]]] = {}
    for reg in regions:
        row = matching_df[matching_df["region"] == reg].iloc[0]
        idx = int(row["spectrum_index"])
        h = trace_gp_hyperparameter_convergence(
            energy=ary_energy[idx],
            intensity=ary_intensity[idx],
            config=trace_cfg,
            meta=row.to_dict(),
        )
        history_data[reg] = h

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    iters = np.arange(1, num_iterations + 1)
    cmap = plt.get_cmap("tab10")
    colors = {reg: cmap(i % 10) for i, reg in enumerate(regions)}

    # Subplot (a): Marginal Log-Likelihood Loss
    ax1 = axes[0, 0]
    for reg in regions:
        ax1.plot(
            iters,
            history_data[reg]["loss"],
            label=reg,
            color=colors[reg],
            lw=2,
        )
    ax1.set_title(
        "(a) Negative Marginal Log-Likelihood (-MLL)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Adam Iterations")
    ax1.set_ylabel("Loss (-MLL)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Subplot (b): Noise Sigma
    ax2 = axes[0, 1]
    for reg in regions:
        ax2.plot(
            iters,
            history_data[reg]["noise_sigma"],
            label=reg,
            color=colors[reg],
            lw=2,
        )
    ax2.set_title(
        "(b) Learned Noise Sigma (Physical Intensity Units)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Adam Iterations")
    ax2.set_ylabel("Noise Sigma (counts)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # Subplot (c): Kernel Lengthscale (eV)
    ax3 = axes[1, 0]
    for reg in regions:
        ax3.plot(
            iters,
            history_data[reg]["lengthscale"],
            label=reg,
            color=colors[reg],
            lw=2,
        )
    ax3.set_title(
        "(c) Learned Kernel Lengthscale (eV)",
        fontsize=12,
        fontweight="bold",
    )
    ax3.set_xlabel("Adam Iterations")
    ax3.set_ylabel("Lengthscale (eV)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")

    # Subplot (d): Relative Error vs Final (%)
    ax4 = axes[1, 1]
    for reg in regions:
        final_s = history_data[reg]["noise_sigma"][-1]
        denom = final_s if abs(final_s) > 1e-12 else 1.0
        rel_err = np.array(
            [
                abs(s - final_s) / denom * 100.0
                for s in history_data[reg]["noise_sigma"]
            ]
        )
        ax4.plot(iters, rel_err, label=reg, color=colors[reg], lw=2)
    ax4.axhline(5.0, color="gray", linestyle="--", label="5% Error Threshold")
    ax4.axhline(1.0, color="black", linestyle=":", label="1% Error Threshold")
    ax4.set_title(
        f"(d) Relative Error of Noise Sigma vs Iteration {num_iterations} (%)",
        fontsize=12,
        fontweight="bold",
    )
    ax4.set_xlabel("Adam Iterations")
    ax4.set_ylabel("Relative Error (%)")
    ax4.set_ylim(-1, 50)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right")

    fig.tight_layout()

    if save_path is not None:
        dir_name = os.path.dirname(os.path.abspath(save_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    if show:
        plt.show()

    return fig


def plot_denoise_spectrum_comparison(
    ary_energy: np.ndarray,
    intensity_pair: tuple[np.ndarray, np.ndarray],
    df_meta: pd.DataFrame,
    plot_config: dict[str, Any] | None = None,
) -> tuple[Figure, Any]:
    """Plot non-denoised vs. denoised spectra across all regions for a session.

    Renders non-denoised spectra in solid black and denoised spectra in dashed
    dark pink across all regions of a selected measurement session (defaulting
    to the first available training measurement).

    Parameters
    ----------
    ary_energy : np.ndarray
        2D array of binding energies in eV of shape (N_total, N_points).
    intensity_pair : tuple[np.ndarray, np.ndarray]
        Tuple of (ary_intensity_non_denoised, ary_intensity_denoised).
    df_meta : pd.DataFrame
        Metadata DataFrame containing 'spectrum_index', 'measurement_id',
        'die', and 'region'.
    plot_config : dict[str, Any] | None, optional
        Dictionary of plotting options:
        - 'measurement_id' (str | None): Session to plot (defaults to first
          available 'train' session).
        - 'die' (int): Die index to plot (default 0).
        - 'invert_x' (bool): Invert energy axis per XPS convention (default
          True).
        - 'save_path' (str | None): Optional file path to save the figure.
        - 'show' (bool): Whether to invoke plt.show() (default False).

    Returns
    -------
    tuple[Figure, Any]
        Tuple of (Figure, Axes array).
    """
    if not (isinstance(intensity_pair, tuple) and len(intensity_pair) == 2):
        raise ValueError(
            "intensity_pair must be a 2-tuple of "
            "(ary_intensity_non_denoised, ary_intensity_denoised)."
        )
    ary_non_denoised, ary_denoised = intensity_pair
    if df_meta.empty:
        raise ValueError("df_meta is empty.")

    cfg = plot_config or {}
    die = int(cfg.get("die", 0))
    subset_die = df_meta[df_meta["die"] == die]
    if subset_die.empty:
        raise ValueError(f"No spectra found in df_meta for die={die}.")

    meas_id = cfg.get("measurement_id")
    if meas_id is None:
        if "split" in subset_die.columns:
            train_rows = subset_die[subset_die["split"] == "train"]
            if not train_rows.empty:
                meas_id = str(train_rows["measurement_id"].iloc[0])
        if meas_id is None:
            meas_id = str(subset_die["measurement_id"].iloc[0])

    session_rows = subset_die[subset_die["measurement_id"] == str(meas_id)]
    if session_rows.empty:
        raise ValueError(
            f"No spectra found for measurement_id={meas_id!r}, die={die}."
        )

    regions = list(session_rows["region"].unique())
    fig, axes, axes_flat = _setup_subplot_grid(
        len(regions), max_cols=3, col_w=5.0, row_h=3.8
    )

    for idx, region_name in enumerate(regions):
        ax = axes_flat[idx]
        reg_row = session_rows[session_rows["region"] == region_name].iloc[0]
        spec_idx = int(reg_row["spectrum_index"])
        energy_grid = ary_energy[spec_idx]
        raw_curve = ary_non_denoised[spec_idx]
        den_curve = ary_denoised[spec_idx]

        ax.plot(
            energy_grid,
            raw_curve,
            color="black",
            linestyle="-",
            linewidth=1.5,
            label="Non-denoised",
        )
        ax.plot(
            energy_grid,
            den_curve,
            color="#C71585",
            linestyle="--",
            linewidth=1.7,
            label="Denoised",
        )
        ax.legend(loc="best", fontsize="x-small")
        _finalize_plot(
            ax,
            title=f"Region: {region_name}",
            xlabel="Binding Energy (eV)",
            ylabel="Intensity (counts / a.u.)",
            invert_x=bool(cfg.get("invert_x", True)),
            grid_alpha=0.5,
        )

    fig.suptitle(
        f"Non-Denoised vs. Denoised Spectra | {meas_id} (Die {die})",
        fontsize=13,
        fontweight="bold",
    )
    if cfg.get("save_path"):
        save_path = str(cfg["save_path"])
        dir_name = os.path.dirname(os.path.abspath(save_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    _finish_fig(fig, cfg)
    return fig, axes


def plot_denoise_atomic_percentage_mae(
    df_denoise_atomic_samples: pd.DataFrame,
    plot_config: dict[str, Any] | None = None,
) -> tuple[Figure, plt.Axes]:
    """Plot box plot of atomic percentage MAE (Denoised vs. Non-Denoised).

    Draws regional box plots of ``abs_diff_atomic_percentage`` across all
    evaluated dies and sessions, with mean and standard deviation annotated for
    each region.

    Parameters
    ----------
    df_denoise_atomic_samples : pd.DataFrame
        Sample-level DataFrame returned by
        ``compare_denoised_atomic_percentages`` containing 'region' and
        'abs_diff_atomic_percentage'.
    plot_config : dict[str, Any] | None, optional
        Dictionary of plotting options:
        - 'figsize' (tuple[float, float]): Figure size (default (9, 5.5)).
        - 'title' (str | None): Custom plot title.
        - 'save_path' (str | None): Optional file path to save the figure.
        - 'show' (bool): Whether to invoke plt.show() (default False).

    Returns
    -------
    tuple[Figure, plt.Axes]
        Tuple of (Figure, Axes).
    """
    if df_denoise_atomic_samples.empty:
        raise ValueError("df_denoise_atomic_samples is empty.")
    required_cols = {"region", "abs_diff_atomic_percentage"}
    missing = required_cols - set(df_denoise_atomic_samples.columns)
    if missing:
        raise KeyError(
            f"df_denoise_atomic_samples missing required columns: "
            f"{sorted(missing)}."
        )

    cfg = plot_config or {}
    df = df_denoise_atomic_samples.copy()
    regions = list(df["region"].unique())
    fig, ax = _prepare_canvas(
        cfg.get("ax"), cfg.get("figsize", (9.0, 5.5)), 1, 1
    )

    sns.boxplot(
        data=df,
        x="region",
        y="abs_diff_atomic_percentage",
        order=regions,
        ax=ax,
        color="#8ecae6",
        width=0.5,
        showfliers=False,
        boxprops={"alpha": 0.75, "edgecolor": "#023047"},
    )
    sns.stripplot(
        data=df,
        x="region",
        y="abs_diff_atomic_percentage",
        order=regions,
        ax=ax,
        color="#023047",
        alpha=0.55,
        size=4,
        jitter=0.18,
    )

    max_y = float(df["abs_diff_atomic_percentage"].max())
    for idx, region_name in enumerate(regions):
        vals = df.loc[
            df["region"] == region_name, "abs_diff_atomic_percentage"
        ].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        top_pos = max(float(np.max(vals)), mean_val + std_val)
        max_y = max(max_y, top_pos)

        ax.errorbar(
            idx,
            mean_val,
            yerr=[[min(mean_val, std_val)], [std_val]],
            fmt="o",
            color="#d81b60",
            ecolor="#d81b60",
            elinewidth=1.8,
            capsize=4.5,
            markersize=6,
            zorder=5,
            label="Mean ± Std (MAE)" if idx == 0 else None,
        )
        ax.text(
            idx,
            top_pos + max(0.02 * max_y, 0.005),
            f"{mean_val:.3f}%\n±{std_val:.3f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#023047",
            zorder=6,
        )

    ax.set_ylim(0.0, max_y * 1.28 + 0.02)
    ax.legend(loc="upper right", fontsize=9)
    plot_title = cfg.get(
        "title",
        "Atomic % MAE by Region (Denoised vs. Non-Denoised Spectra)",
    )
    _finalize_plot(
        ax,
        title=plot_title,
        xlabel="Region",
        ylabel="Absolute Error |Denoised - Non-Denoised| (at. %)",
        invert_x=False,
        grid_alpha=0.5,
    )
    if cfg.get("save_path"):
        save_path = str(cfg["save_path"])
        dir_name = os.path.dirname(os.path.abspath(save_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    _finish_fig(fig, cfg)
    return fig, ax
