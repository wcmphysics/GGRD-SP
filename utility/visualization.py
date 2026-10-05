"""Visualization module for XPS spectra in GGRD-SP.

Provides utility functions to plot spectra, pairing relationships, and Shirley
background subtraction directly from ary_energy, ary_intensity, and meta_df.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utility.evaluation import calculate_prediction_metrics
from utility.patching import calculate_window_points, extract_sliding_windows
from utility.quantification import calculate_shirley_background


def _prepare_canvas(
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Return figure and axes, creating a new figure and axes if ax is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def _finalize_plot(
    ax: plt.Axes,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    invert_x: bool = True,
    grid: bool = True,
    grid_alpha: float = 0.6,
    show: bool = False,
) -> None:
    """Apply standard plot styling, labels, inverted x-axis, and display."""
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if invert_x and not ax.xaxis_inverted():
        ax.invert_xaxis()
    if grid:
        ax.grid(True, linestyle="--", alpha=grid_alpha)
    if show:
        plt.show()


def plot_regional_spectra(
    ary_energy: np.ndarray,
    ary_intensity: np.ndarray,
    meta_df: pd.DataFrame,
    plot_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot regional spectra filtered by metadata attributes.

    Parameters
    ----------
    ary_energy : np.ndarray
        2D array of binding energies, shape (N_total, N_points).
    ary_intensity : np.ndarray
        2D array of intensities, shape (N_total, N_points).
    meta_df : pd.DataFrame
        Metadata DataFrame containing spectrum details and 'spectrum_index'.
    plot_config : dict[str, Any] | None, optional
        Dictionary bundling plotting options:
        - 'filters' (dict[str, Any] | None): Key-value pairs to filter metadata
          (e.g., {'tool': 'J4', 'region': 'Al2p'}). If None, filters to the
          first available measurement session.
        - 'max_spectra' (int): Maximum spectra to display simultaneously (default 10).
        - 'invert_x' (bool): Invert x-axis per standard XPS convention (default True).
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
        If filter key is not found in meta_df columns.
    ValueError
        If no spectra match filters or empty DataFrame provided.
    """
    cfg = plot_config or {}
    filters = cfg.get("filters")
    invert_x: bool = cfg.get("invert_x", True)
    title: str | None = cfg.get("title")
    ax: plt.Axes | None = cfg.get("ax")
    show: bool = cfg.get("show", False)
    max_spectra: int = cfg.get("max_spectra", 10)

    if meta_df.empty:
        raise ValueError("meta_df cannot be empty")

    filtered_df = meta_df.copy()
    if filters:
        for key, value in filters.items():
            if key not in filtered_df.columns:
                raise KeyError(
                    f"Filter key '{key}' not found in meta_df columns: "
                    f"{list(filtered_df.columns)}"
                )
            filtered_df = filtered_df[filtered_df[key] == value]
    else:
        # Default to first available measurement to prevent overplotting
        first_meas = filtered_df["measurement_id"].iloc[0]
        filtered_df = filtered_df[filtered_df["measurement_id"] == first_meas]

    if filtered_df.empty:
        raise ValueError(f"No spectra match the specified filters: {filters}")

    if len(filtered_df) > max_spectra:
        filtered_df = filtered_df.iloc[:max_spectra]

    fig, ax = _prepare_canvas(ax, figsize=(8, 5))

    for _, row in filtered_df.iterrows():
        idx = int(row["spectrum_index"])
        label = (
            f"{row.get('tool', 'Tool')} | {row.get('region', 'Region')} | "
            f"Die {row.get('die', '-')} | {row.get('measurement_id', '')}"
        )
        ax.plot(ary_energy[idx], ary_intensity[idx], label=label)

    if not title:
        regions = filtered_df["region"].unique()
        tools = filtered_df["tool"].unique()
        title = (
            f"XPS Spectra - Region: {', '.join(map(str, regions))} | "
            f"Tool: {', '.join(map(str, tools))}"
        )

    if len(filtered_df) <= 15:
        ax.legend(loc="best", fontsize="small")

    _finalize_plot(
        ax,
        title=title,
        xlabel="Binding Energy (eV)",
        ylabel="Intensity (counts / a.u.)",
        invert_x=invert_x,
        show=show,
    )
    return fig, ax


def plot_tool_comparison(
    ary_energy: np.ndarray,
    ary_intensity: np.ndarray,
    meta_df: pd.DataFrame,
    compare_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot an overlay comparison of spectra between two tools for a given region and die.

    Parameters
    ----------
    ary_energy : np.ndarray
        2D array of binding energies.
    ary_intensity : np.ndarray
        2D array of intensities.
    meta_df : pd.DataFrame
        Metadata DataFrame.
    compare_config : dict[str, Any] | None, optional
        Dictionary bundling comparison options:
        - 'region' (str): Target spectral region (default 'Al2p').
        - 'die' (int): Target die index (default 0).
        - 'tools' (list[str] | tuple[str, ...]): Tools to compare (default ('J4', 'J5')).
        - 'invert_x' (bool): Whether to invert x-axis per XPS convention (default True).
        - 'ax' (plt.Axes | None): Existing matplotlib Axes.
        - 'show' (bool): Whether to invoke plt.show() (default False).

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The matplotlib Figure and Axes objects.

    Raises
    ------
    ValueError
        If fewer than 2 tools are provided or no matching spectra are found.
    """
    cfg = compare_config or {}
    region: str = cfg.get("region", "Al2p")
    die: int = cfg.get("die", 0)
    tools = list(cfg.get("tools", ("J4", "J5")))
    invert_x: bool = cfg.get("invert_x", True)
    ax: plt.Axes | None = cfg.get("ax")
    show: bool = cfg.get("show", False)

    if len(tools) < 2:
        raise ValueError(f"At least 2 tools are required for comparison, got {tools}")

    fig, ax = _prepare_canvas(ax, figsize=(8, 5))
    palette = ["black", "red", "tab:blue", "tab:green", "tab:purple"]
    color_map = {tool: palette[i % len(palette)] for i, tool in enumerate(tools)}

    plotted_count = 0
    for tool in tools:
        subset = meta_df[
            (meta_df["tool"] == tool)
            & (meta_df["region"] == region)
            & (meta_df["die"] == die)
        ]
        if subset.empty:
            continue

        first_row = subset.iloc[0]
        idx = int(first_row["spectrum_index"])
        ax.plot(
            ary_energy[idx],
            ary_intensity[idx],
            label=f"{tool} ({first_row['measurement_id']})",
            color=color_map.get(tool),
            linewidth=1.8,
        )
        plotted_count += 1

    if plotted_count == 0:
        raise ValueError(
            f"No matching spectra found for tools={tools}, region='{region}', die={die}"
        )

    ax.legend(loc="best")
    _finalize_plot(
        ax,
        title=f"Tool Comparison ({' vs '.join(tools)}) - Region: {region} (Die {die})",
        xlabel="Binding Energy (eV)",
        ylabel="Intensity (counts / a.u.)",
        invert_x=invert_x,
        show=show,
    )
    return fig, ax


def plot_pairing_timeline(
    meta_df: pd.DataFrame,
    plot_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot timeline of measurement sessions and connect 1-to-1 paired sessions.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Metadata DataFrame with pairing columns ('tool_target', 'measurement_id_target',
        'time_target', 'time_diff_hours').
    plot_config : dict[str, Any] | None, optional
        Dictionary bundling plotting options:
        - 'source_tool' (str): Name of the source tool (default 'J4').
        - 'target_tool' (str): Name of the target tool (default 'J5').
        - 'title' (str | None): Custom title for the plot.
        - 'ax' (plt.Axes | None): Existing matplotlib Axes.
        - 'show' (bool): Whether to invoke plt.show() (default False).

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The matplotlib Figure and Axes objects.

    Raises
    ------
    KeyError
        If required columns are missing from meta_df.
    ValueError
        If no measurement sessions exist for source or target tool.
    """
    cfg = plot_config or {}
    source_tool: str = cfg.get("source_tool", "J4")
    target_tool: str = cfg.get("target_tool", "J5")
    title: str | None = cfg.get("title")
    ax: plt.Axes | None = cfg.get("ax")
    show: bool = cfg.get("show", False)

    required_cols = {"tool", "measurement_id", "time"}
    missing = required_cols - set(meta_df.columns)
    if missing:
        raise KeyError(f"meta_df is missing required columns: {sorted(missing)}")

    # Extract distinct sessions per tool
    df = meta_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        try:
            df["time"] = pd.to_datetime(df["time"], format="mixed")
        except ValueError:
            df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)

    src_df = (
        df[df["tool"] == source_tool]
        .drop_duplicates(subset=["measurement_id"])
        .sort_values("time")
    )
    tgt_df = (
        df[df["tool"] == target_tool]
        .drop_duplicates(subset=["measurement_id"])
        .sort_values("time")
    )

    if src_df.empty and tgt_df.empty:
        raise ValueError(
            f"No measurements found for tools '{source_tool}' or '{target_tool}'."
        )

    fig, ax = _prepare_canvas(ax, figsize=(10, 4.5))

    y_src = 1.0
    y_tgt = 0.0

    # Draw horizontal baseline tracks
    all_times = pd.concat([src_df["time"], tgt_df["time"]]).dropna()
    if not all_times.empty:
        t_min, t_max = all_times.min(), all_times.max()
        ax.hlines(
            y=[y_src, y_tgt],
            xmin=t_min,
            xmax=t_max,
            colors="gray",
            linestyles=":",
            alpha=0.5,
        )

    # Track paired vs unpaired sessions
    has_pairing_info = (
        "measurement_id_target" in src_df.columns and "time_target" in src_df.columns
    )
    paired_src_ids: set[str] = set()
    paired_tgt_ids: set[str] = set()

    if has_pairing_info:
        for _, row in src_df.iterrows():
            tgt_meas = row.get("measurement_id_target")
            tgt_time = row.get("time_target")
            diff_h = row.get("time_diff_hours")

            if pd.notna(tgt_meas) and pd.notna(tgt_time):
                src_time = row["time"]
                paired_src_ids.add(str(row["measurement_id"]))
                paired_tgt_ids.add(str(tgt_meas))

                # Draw connection line
                ax.plot(
                    [src_time, tgt_time],
                    [y_src, y_tgt],
                    color="tab:purple",
                    linestyle="-",
                    alpha=0.7,
                    linewidth=1.5,
                )

                # Annotate mid-point with time difference
                mid_time = src_time + (tgt_time - src_time) / 2
                mid_y = (y_src + y_tgt) / 2
                diff_str = f"{diff_h:.1f}h" if pd.notna(diff_h) else ""
                ax.annotate(
                    diff_str,
                    (mid_time, mid_y),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=8,
                    color="darkmagenta",
                )

    # Plot source nodes
    paired_src = src_df[src_df["measurement_id"].isin(paired_src_ids)]
    unpaired_src = src_df[~src_df["measurement_id"].isin(paired_src_ids)]

    if not paired_src.empty:
        ax.scatter(
            paired_src["time"],
            [y_src] * len(paired_src),
            color="tab:blue",
            s=70,
            zorder=3,
            label=f"{source_tool} (Paired)",
        )
    if not unpaired_src.empty:
        ax.scatter(
            unpaired_src["time"],
            [y_src] * len(unpaired_src),
            facecolors="none",
            edgecolors="tab:blue",
            s=70,
            linewidth=1.8,
            zorder=3,
            label=f"{source_tool} (Unpaired)",
        )

    # Plot target nodes
    paired_tgt = tgt_df[tgt_df["measurement_id"].isin(paired_tgt_ids)]
    unpaired_tgt = tgt_df[~tgt_df["measurement_id"].isin(paired_tgt_ids)]

    if not paired_tgt.empty:
        ax.scatter(
            paired_tgt["time"],
            [y_tgt] * len(paired_tgt),
            color="tab:green",
            s=70,
            zorder=3,
            label=f"{target_tool} (Paired)",
        )
    if not unpaired_tgt.empty:
        ax.scatter(
            unpaired_tgt["time"],
            [y_tgt] * len(unpaired_tgt),
            facecolors="none",
            edgecolors="tab:green",
            s=70,
            linewidth=1.8,
            zorder=3,
            label=f"{target_tool} (Unpaired)",
        )

    ax.set_yticks([y_tgt, y_src])
    ax.set_yticklabels([f"Target ({target_tool})", f"Source ({source_tool})"])
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel("Measurement Time")

    timeline_title = title or (
        f"1-to-1 Measurement Pairing Timeline ({source_tool} -> {target_tool}) "
        f"| Paired: {len(paired_src_ids)}/{len(src_df)}"
    )
    ax.set_title(timeline_title)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax.legend(loc="upper right", fontsize="small", framealpha=0.9)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, ax


def plot_max_intensity_vs_time(
    ary_intensity: np.ndarray,
    meta_df: pd.DataFrame,
    plot_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot maximum spectral intensity vs measurement time using a Seaborn scatter plot.

    Parameters
    ----------
    ary_intensity : np.ndarray
        2D array of measured spectral intensities, shape (N_spectra, N_points).
    meta_df : pd.DataFrame
        Metadata DataFrame containing 'spectrum_index', 'time', and metadata columns
        such as 'tool', 'die', and 'region'.
    plot_config : dict[str, Any] | None, optional
        Dictionary bundling plotting options:
        - 'hue' (str): Column to group/color by ('tool', 'die', or 'region'; default 'tool').
        - 'filters' (dict[str, Any] | None): Optional key-value filter mapping on metadata
          (e.g., {'region': 'Al2p'} or {'die': 0}).
        - 'style' (str | None): Optional column to vary marker style by.
        - 'palette' (str | list | dict | None): Seaborn color palette.
        - 'alpha' (float): Marker transparency (default 0.75).
        - 's' (float): Marker size in points**2 (default 45.0).
        - 'title' (str | None): Custom figure title.
        - 'figsize' (tuple[float, float]): Figure dimensions (default (10, 5.5)).
        - 'ax' (plt.Axes | None): Existing matplotlib Axes.
        - 'show' (bool): Whether to invoke plt.show() (default False).

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The matplotlib Figure and Axes objects.

    Raises
    ------
    ValueError
        If meta_df is empty or no records match the specified filters.
    KeyError
        If required columns ('time', 'spectrum_index', or specified 'hue') are missing.
    """
    cfg = plot_config or {}
    hue: str = str(cfg.get("hue", "tool"))
    filters = cfg.get("filters")
    style: str | None = cfg.get("style")
    palette = cfg.get("palette")
    alpha: float = float(cfg.get("alpha", 0.75))
    s: float = float(cfg.get("s", 45.0))
    title: str | None = cfg.get("title")
    figsize: tuple[float, float] = tuple(cfg.get("figsize", (10, 5.5)))
    custom_ax: plt.Axes | None = cfg.get("ax")
    show: bool = bool(cfg.get("show", False))

    if meta_df.empty:
        raise ValueError("meta_df cannot be empty")

    plot_df = meta_df.copy()
    if filters:
        for k, v in filters.items():
            if k not in plot_df.columns:
                raise KeyError(f"Filter key '{k}' not found in meta_df columns: {list(plot_df.columns)}")
            plot_df = plot_df[plot_df[k] == v]

    if plot_df.empty:
        raise ValueError(f"No spectra match specified filters: {filters}")

    if "time" not in plot_df.columns:
        raise KeyError("meta_df must contain a 'time' column for time-series plotting.")
    if "spectrum_index" not in plot_df.columns:
        raise KeyError("meta_df must contain a 'spectrum_index' column.")
    if hue not in plot_df.columns:
        raise KeyError(f"Hue column '{hue}' not found in meta_df columns: {list(plot_df.columns)}")

    indices = plot_df["spectrum_index"].astype(int).values
    plot_df["max_intensity"] = np.max(ary_intensity[indices], axis=1)

    try:
        plot_df["time"] = pd.to_datetime(plot_df["time"], format="mixed")
    except (ValueError, TypeError):
        plot_df["time"] = pd.to_datetime(plot_df["time"])

    # Discrete formatting for hue
    plot_hue = hue
    hue_order = None
    if hue == "die":
        dies = sorted(plot_df["die"].dropna().unique())
        plot_df["die_formatted"] = plot_df["die"].apply(lambda d: f"Die {int(d)}" if pd.notna(d) else "N/A")
        plot_hue = "die_formatted"
        hue_order = [f"Die {int(d)}" for d in dies]
    elif hue == "tool":
        hue_order = sorted(plot_df["tool"].dropna().unique())
    elif hue == "region":
        hue_order = sorted(plot_df["region"].dropna().unique())

    fig, ax = _prepare_canvas(custom_ax, figsize=figsize)

    sns.scatterplot(
        data=plot_df,
        x="time",
        y="max_intensity",
        hue=plot_hue,
        hue_order=hue_order,
        style=style,
        palette=palette,
        alpha=alpha,
        s=s,
        ax=ax,
    )

    if not title:
        filt_str = f" | Filtered: {filters}" if filters else ""
        title = f"Maximum Intensity vs. Time (hue={hue}){filt_str}"

    ax.set_title(title)
    ax.set_xlabel("Measurement Time")
    ax.set_ylabel("Maximum Intensity (counts)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, ax


def plot_normalized_max_intensity_vs_time(
    ary_intensity: np.ndarray,
    meta_df: pd.DataFrame,
    ary_energy: np.ndarray | None = None,
    plot_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot maximum intensity normalized by total integrated area vs time using Seaborn.

    Normalizing by total integrated area cancels common power and illumination drifts
    across time, tools, and wafer dies, exposing the true normalized peak behavior.

    Parameters
    ----------
    ary_intensity : np.ndarray
        2D array of measured spectral intensities, shape (N_spectra, N_points).
    meta_df : pd.DataFrame
        Metadata DataFrame containing 'spectrum_index', 'time', 'tool', 'die', 'region',
        and 'measurement_id'.
    ary_energy : np.ndarray | None, optional
        2D array of binding energies. If provided, trapezoidal numerical integration in eV
        is performed. If None, sum across intensity points is used.
    plot_config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'hue' (str): Column to group/color by ('tool', 'die', or 'region'; default 'tool').
        - 'normalization_mode' (str): Normalization strategy:
          * 'die_total_flux' (default): Normalizes by the sum of integrated areas across
            all measured regions on that die (Tier 1 physical reference).
          * 'spectrum_area': Normalizes by each individual regional spectrum's integrated area.
        - 'filters' (dict[str, Any] | None): Optional key-value filter mapping on metadata.
        - 'style' (str | None): Optional column for scatter marker style.
        - 'palette' (str | list | dict | None): Seaborn color palette.
        - 'alpha' (float): Marker transparency (default 0.75).
        - 's' (float): Marker size in points**2 (default 45.0).
        - 'title' (str | None): Custom figure title.
        - 'ylabel' (str | None): Custom y-axis label.
        - 'figsize' (tuple[float, float]): Figure dimensions (default (10, 5.5)).
        - 'ax' (plt.Axes | None): Existing matplotlib Axes.
        - 'show' (bool): Whether to invoke plt.show() (default False).

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The matplotlib Figure and Axes objects.

    Raises
    ------
    ValueError
        If meta_df is empty or no records match the specified filters.
    KeyError
        If required columns are missing from meta_df.
    """
    cfg = plot_config or {}
    hue: str = str(cfg.get("hue", "tool"))
    norm_mode: str = str(cfg.get("normalization_mode", "die_total_flux")).lower()
    filters = cfg.get("filters")
    style: str | None = cfg.get("style")
    palette = cfg.get("palette")
    alpha: float = float(cfg.get("alpha", 0.75))
    s: float = float(cfg.get("s", 45.0))
    title: str | None = cfg.get("title")
    ylabel: str | None = cfg.get("ylabel")
    figsize: tuple[float, float] = tuple(cfg.get("figsize", (10, 5.5)))
    custom_ax: plt.Axes | None = cfg.get("ax")
    show: bool = bool(cfg.get("show", False))

    if meta_df.empty:
        raise ValueError("meta_df cannot be empty")

    # Upfront column validation
    if "time" not in meta_df.columns:
        raise KeyError("meta_df must contain a 'time' column for time-series plotting.")
    if "spectrum_index" not in meta_df.columns:
        raise KeyError("meta_df must contain a 'spectrum_index' column.")
    if hue not in meta_df.columns:
        raise KeyError(f"Hue column '{hue}' not found in meta_df columns: {list(meta_df.columns)}")

    valid_die_flux_modes = {"die_total_flux", "die_flux", "die", "canonical"}
    valid_spec_area_modes = {"spectrum_area", "spectrum", "regional"}

    if norm_mode not in (valid_die_flux_modes | valid_spec_area_modes):
        raise ValueError(
            f"Unsupported normalization_mode '{norm_mode}'. "
            f"Must be 'die_total_flux' or 'spectrum_area'."
        )

    if norm_mode in valid_die_flux_modes:
        for req_col in ("measurement_id", "die"):
            if req_col not in meta_df.columns:
                raise KeyError(
                    f"Column '{req_col}' is required in meta_df for 'die_total_flux' normalization."
                )

        from models.dataset import compute_die_total_flux

        # Compute die total flux across the full meta_df before filtering
        flux_map = compute_die_total_flux(meta_df, ary_intensity, ary_energy)

        plot_df = meta_df.copy()
        if filters:
            for k, v in filters.items():
                if k not in plot_df.columns:
                    raise KeyError(f"Filter key '{k}' not found in meta_df columns: {list(plot_df.columns)}")
                plot_df = plot_df[plot_df[k] == v]

        if plot_df.empty:
            raise ValueError(f"No spectra match specified filters: {filters}")

        indices = plot_df["spectrum_index"].astype(int).values
        max_vals = np.max(ary_intensity[indices], axis=1)

        norm_vals: list[float] = []
        for max_val, (_, row) in zip(max_vals, plot_df.iterrows()):
            key = (str(row["measurement_id"]), int(row["die"]))
            f_die = flux_map.get(key, 1.0)
            norm_vals.append(float(max_val) / max(float(f_die), 1e-4))
        plot_df["normalized_max"] = norm_vals
        default_ylabel = "Max Intensity / Die Total Integrated Area (1/eV)"

    else:  # 'spectrum_area'
        plot_df = meta_df.copy()
        if filters:
            for k, v in filters.items():
                if k not in plot_df.columns:
                    raise KeyError(f"Filter key '{k}' not found in meta_df columns: {list(plot_df.columns)}")
                plot_df = plot_df[plot_df[k] == v]

        if plot_df.empty:
            raise ValueError(f"No spectra match specified filters: {filters}")

        indices = plot_df["spectrum_index"].astype(int).values
        max_vals = np.max(ary_intensity[indices], axis=1)

        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        areas: list[float] = []
        for idx in indices:
            i_vec = ary_intensity[idx]
            if ary_energy is not None:
                e_vec = ary_energy[idx]
                a = float(abs(trapz_fn(i_vec, x=e_vec)))
            else:
                a = float(np.sum(i_vec))
            areas.append(max(a, 1e-4))

        plot_df["normalized_max"] = max_vals / np.array(areas)
        default_ylabel = "Max Intensity / Spectrum Integrated Area (1/eV)"

    if "time" not in plot_df.columns:
        raise KeyError("meta_df must contain a 'time' column for time-series plotting.")
    if hue not in plot_df.columns:
        raise KeyError(f"Hue column '{hue}' not found in meta_df columns: {list(plot_df.columns)}")

    try:
        plot_df["time"] = pd.to_datetime(plot_df["time"], format="mixed")
    except (ValueError, TypeError):
        plot_df["time"] = pd.to_datetime(plot_df["time"])

    plot_hue = hue
    hue_order = None
    if hue == "die":
        dies = sorted(plot_df["die"].dropna().unique())
        plot_df["die_formatted"] = plot_df["die"].apply(lambda d: f"Die {int(d)}" if pd.notna(d) else "N/A")
        plot_hue = "die_formatted"
        hue_order = [f"Die {int(d)}" for d in dies]
    elif hue == "tool":
        hue_order = sorted(plot_df["tool"].dropna().unique())
    elif hue == "region":
        hue_order = sorted(plot_df["region"].dropna().unique())

    fig, ax = _prepare_canvas(custom_ax, figsize=figsize)

    sns.scatterplot(
        data=plot_df,
        x="time",
        y="normalized_max",
        hue=plot_hue,
        hue_order=hue_order,
        style=style,
        palette=palette,
        alpha=alpha,
        s=s,
        ax=ax,
    )

    if not title:
        filt_str = f" | Filtered: {filters}" if filters else ""
        title = f"Normalized Maximum Intensity vs. Time (hue={hue}){filt_str}"

    ax.set_title(title)
    ax.set_xlabel("Measurement Time")
    ax.set_ylabel(ylabel or default_ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, ax


def plot_shirley_background(
    ary_energy: np.ndarray,
    ary_intensity: np.ndarray,
    meta_df: pd.DataFrame,
    plot_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, Any]:
    """Plot regional spectrum with Shirley background overlay and shaded net peak area.

    Parameters
    ----------
    ary_energy : np.ndarray
        2D array of binding energies.
    ary_intensity : np.ndarray
        2D array of measured intensities.
    meta_df : pd.DataFrame
        Metadata DataFrame.
    plot_config : dict[str, Any] | None, optional
        Dictionary bundling plotting options:
        - 'measurement_id' (str | None): Measurement session to display.
        - 'die' (int): Die index (default 0).
        - 'region' (str | None): Specific region to plot (e.g. 'Ti2p').
          If None, all regions for the measurement session and die are plotted in subplots.
        - 'strategy' (str): Default endpoint strategy ('minima', 'edge', or 'direct').
        - 'average_width_ev' (float | None): Boundary averaging window in eV (default 1.0).
        - 'region_endpoint_config' (dict | None): Per-region endpoint strategy overrides.
        - 'invert_x' (bool): Invert x-axis per standard XPS convention (default True).
        - 'ax' (plt.Axes | None): Matplotlib Axes (only if single region specified).
        - 'show' (bool): Whether to invoke plt.show() (default False).

    Returns
    -------
    tuple[plt.Figure, Any]
        The Figure and Axes (or array of Axes for multi-panel).

    Raises
    ------
    ValueError
        If no matching spectra are found.
    """
    cfg = plot_config or {}
    meas_id: str | None = cfg.get("measurement_id")
    die: int = int(cfg.get("die", 0))
    target_region: str | None = cfg.get("region")
    invert_x: bool = cfg.get("invert_x", True)
    custom_ax: plt.Axes | None = cfg.get("ax")
    show: bool = cfg.get("show", False)

    subset = meta_df[meta_df["die"] == die]
    if meas_id is not None:
        subset = subset[subset["measurement_id"] == meas_id]
    elif not subset.empty:
        meas_id = str(subset["measurement_id"].iloc[0])
        subset = subset[subset["measurement_id"] == meas_id]

    if target_region is not None:
        subset = subset[subset["region"] == target_region]

    if subset.empty:
        raise ValueError(
            f"No matching spectra found for measurement_id='{meas_id}', die={die}, "
            f"region='{target_region}'"
        )

    # Single-region plot
    if target_region is not None or len(subset) == 1:
        row = subset.iloc[0]
        region_name = str(row["region"])
        spec_idx = int(row["spectrum_index"])
        e_arr, i_arr = ary_energy[spec_idx], ary_intensity[spec_idx]

        shirley_cfg = dict(cfg, region=region_name)
        b_arr, net_area = calculate_shirley_background(e_arr, i_arr, shirley_cfg)

        fig, ax = _prepare_canvas(custom_ax, figsize=(7, 4.5))
        ax.plot(e_arr, i_arr, label="Raw Spectrum", color="navy", linewidth=1.8)
        ax.plot(
            e_arr, b_arr, label="Shirley Background", color="crimson", linestyle="--", linewidth=1.6
        )
        ax.fill_between(
            e_arr,
            b_arr,
            i_arr,
            where=(i_arr > b_arr),
            color="skyblue",
            alpha=0.35,
            label=f"Net Area: {net_area:.1f}",
        )
        ax.legend(loc="best")

        _finalize_plot(
            ax,
            title=f"{meas_id} | Region: {region_name} (Die {die})",
            xlabel="Binding Energy (eV)",
            ylabel="Intensity (counts / a.u.)",
            invert_x=invert_x,
            grid_alpha=0.5,
            show=show,
        )
        return fig, ax

    # Multi-region subplot grid
    regions = list(subset["region"].unique())
    n_regs = len(regions)
    n_cols = min(n_regs, 3)
    n_rows = (n_regs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, reg in enumerate(regions):
        ax = axes_flat[idx]
        reg_row = subset[subset["region"] == reg].iloc[0]
        spec_idx = int(reg_row["spectrum_index"])
        e_arr, i_arr = ary_energy[spec_idx], ary_intensity[spec_idx]

        shirley_cfg = dict(cfg, region=reg)
        b_arr, net_area = calculate_shirley_background(e_arr, i_arr, shirley_cfg)

        ax.plot(e_arr, i_arr, label="Raw", color="navy", linewidth=1.5)
        ax.plot(e_arr, b_arr, label="Shirley", color="crimson", linestyle="--", linewidth=1.4)
        ax.fill_between(
            e_arr,
            b_arr,
            i_arr,
            where=(i_arr > b_arr),
            color="skyblue",
            alpha=0.35,
            label=f"Area: {net_area:.1f}",
        )
        ax.legend(loc="best", fontsize="x-small")

        _finalize_plot(
            ax,
            title=f"Region: {reg}",
            xlabel="Binding Energy (eV)",
            ylabel="Intensity",
            invert_x=invert_x,
            grid_alpha=0.5,
            show=False,
        )

    # Hide unused subplot panels
    for empty_idx in range(n_regs, len(axes_flat)):
        axes_flat[empty_idx].set_visible(False)

    fig.suptitle(f"Shirley Background Subtraction | {meas_id} (Die {die})", fontsize=13)
    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes


def plot_training_history(
    histories: dict[str, dict[str, list[float]]],
    plot_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, np.ndarray | plt.Axes]:
    """Plot training and validation loss curves over epochs for each trained spectral region.

    Parameters
    ----------
    histories : dict[str, dict[str, list[float]]]
        Dictionary mapping region name to its history dict containing 'train_loss' and 'val_loss'.
    plot_config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'log_scale' (bool): Whether to plot loss on a logarithmic scale (default True).
        - 'title' (str | None): Overall figure supertitle.
        - 'show' (bool): Whether to call plt.show() (default False).

    Returns
    -------
    tuple[plt.Figure, np.ndarray | plt.Axes]
        Figure and axes array.
    """
    cfg = plot_config or {}
    log_scale: bool = bool(cfg.get("log_scale", True))
    title: str | None = cfg.get("title", "Baseline Neural Network Training History")
    show: bool = bool(cfg.get("show", False))

    regions = list(histories.keys())
    n_regs = len(regions)
    if n_regs == 0:
        raise ValueError("Histories dictionary is empty; no training curves to plot.")

    n_cols = min(n_regs, 3)
    n_rows = (n_regs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, reg in enumerate(regions):
        ax = axes_flat[idx]
        h = histories[reg]
        train_loss = h.get("train_loss", [])
        val_loss = h.get("val_loss", [])
        epochs = range(1, len(train_loss) + 1)

        ax.plot(epochs, train_loss, label="Train Loss", color="royalblue", linewidth=1.8)
        if val_loss:
            ax.plot(epochs, val_loss, label="Val Loss", color="darkorange", linestyle="--", linewidth=1.8)

        if log_scale:
            ax.set_yscale("log")

        ax.legend(loc="upper right", fontsize="small")
        _finalize_plot(
            ax,
            title=f"Region: {reg}",
            xlabel="Epoch",
            ylabel="Loss (Normalized MSE)" if not log_scale else "Loss (log scale)",
            invert_x=False,
            grid_alpha=0.6,
            show=False,
        )

    # Hide unused panels
    for empty_idx in range(n_regs, len(axes_flat)):
        axes_flat[empty_idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


def plot_prediction_comparison(
    data_original: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    data_predicted: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot an overlay comparison of Source, True Target, and Model Predicted Target spectra.

    Parameters
    ----------
    data_original : tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Tuple of (ary_intensity, ary_energy, meta_df) for measured data.
    data_predicted : tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Tuple of (ary_intensity_predicted, ary_energy_predicted, meta_df_predicted).
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'measurement_id' (str | None): Source measurement session ID (default first paired session).
        - 'region' (str | None): Target region (default 'Al2p' or first available).
        - 'die' (int): Die index (default 0).
        - 'show_residual' (bool): Whether to include a residual subplot below (default True).
        - 'invert_x' (bool): Whether to invert binding energy axis (default True).
        - 'show' (bool): Whether to invoke plt.show() (default False).

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Figure and axes array.
    """
    ary_intensity_orig, ary_energy_orig, meta_df_orig = data_original
    ary_intensity_pred, ary_energy_pred, meta_df_pred = data_predicted

    cfg = config or {}
    meas_id: str | None = cfg.get("measurement_id")
    region: str | None = cfg.get("region")
    die: int = int(cfg.get("die", 0))
    show_residual: bool = bool(cfg.get("show_residual", True))
    invert_x: bool = bool(cfg.get("invert_x", True))
    show: bool = bool(cfg.get("show", False))

    # Auto-detect paired source measurement if not provided
    if meas_id is None:
        paired_sources = meta_df_orig[meta_df_orig["measurement_id_target"].notna()]
        if paired_sources.empty:
            raise ValueError("No paired source measurements found in meta_df_orig.")
        meas_id = str(paired_sources["measurement_id"].iloc[0])

    # Filter source row
    src_rows = meta_df_orig[
        (meta_df_orig["measurement_id"] == meas_id)
        & (meta_df_orig["die"] == die)
    ]
    if region is not None:
        src_rows = src_rows[src_rows["region"] == region]

    if src_rows.empty:
        raise ValueError(f"No source spectrum matching meas_id='{meas_id}', die={die}, region='{region}'.")

    src_row = src_rows.iloc[0]
    actual_region = str(src_row["region"])
    if pd.isna(src_row.get("measurement_id_target")):
        raise ValueError(f"Source measurement '{meas_id}' is not paired with a target measurement.")
    tgt_meas_id = str(src_row.get("measurement_id_target"))
    src_tool = str(src_row.get("tool"))
    tgt_tool = str(src_row.get("tool_target"))

    # Filter target row
    tgt_rows = meta_df_orig[
        (meta_df_orig["measurement_id"] == tgt_meas_id)
        & (meta_df_orig["die"] == die)
        & (meta_df_orig["region"] == actual_region)
    ]
    if tgt_rows.empty:
        raise ValueError(f"Target spectrum matching meas_id='{tgt_meas_id}' not found.")
    tgt_row = tgt_rows.iloc[0]

    # Filter predicted row
    pred_rows = meta_df_pred[
        (meta_df_pred["source_measurement_id"] == meas_id)
        & (meta_df_pred["die"] == die)
        & (meta_df_pred["region"] == actual_region)
    ]
    if pred_rows.empty:
        raise ValueError(f"Predicted spectrum for source meas_id='{meas_id}', die={die}, region='{actual_region}' not found.")
    pred_row = pred_rows.iloc[0]

    # Extract vectors
    energy = ary_energy_orig[int(src_row["spectrum_index"])]
    i_src = ary_intensity_orig[int(src_row["spectrum_index"])]
    i_tgt = ary_intensity_orig[int(tgt_row["spectrum_index"])]
    i_pred = ary_intensity_pred[int(pred_row["spectrum_index"])]
    residual = i_tgt - i_pred

    if show_residual:
        fig, axes = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax_main, ax_res = axes[0], axes[1]
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=(8, 4.5))
        axes = np.array([ax_main])
        ax_res = None

    # Main spectral plot
    ax_main.plot(energy, i_src, label=f"Source Measured ({src_tool})", color="slategray", linestyle="--", linewidth=1.5)
    ax_main.plot(energy, i_tgt, label=f"True Target ({tgt_tool})", color="forestgreen", linewidth=1.8)
    ax_main.plot(energy, i_pred, label=f"Predicted Target ({src_tool} -> {tgt_tool})", color="crimson", linewidth=1.8)
    ax_main.legend(loc="best", fontsize="small")
    _finalize_plot(
        ax_main,
        title=(
            f"Spectral Transfer Comparison | Region: {actual_region} | Die {die}\n"
            f"Source: {meas_id} ({src_tool}) -> Target: {tgt_meas_id} ({tgt_tool})"
        ),
        ylabel="Intensity (counts)",
        invert_x=invert_x,
        grid_alpha=0.6,
        show=False,
    )

    # Residual plot
    if ax_res is not None:
        ax_res.plot(energy, residual, color="purple", linewidth=1.4, label="Target - Predicted")
        ax_res.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
        ax_res.legend(loc="best", fontsize="x-small")
        _finalize_plot(
            ax_res,
            xlabel="Binding Energy (eV)",
            ylabel="Residual",
            invert_x=invert_x,
            grid_alpha=0.6,
            show=False,
        )
    else:
        ax_main.set_xlabel("Binding Energy (eV)")

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes


def plot_sliding_window_slices(
    spectrum: np.ndarray,
    energy: np.ndarray | None = None,
    plot_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot an original regional spectrum alongside its extracted sliding window patches.

    Visualizes how a continuous 1D regional spectrum is decomposed into overlapping
    sequence windows (patches) for localized sequence-to-sequence neural network training.
    The top panel displays the full continuous spectrum with transparent shaded bands
    marking the location and span of each window slice. The bottom panel displays each
    extracted patch plotted across the exact same binding energy scale, illustrating
    the local spatial extent, overlap depth, and edge anchoring.

    Parameters
    ----------
    spectrum : np.ndarray
        1D array of spectral intensities of length N.
    energy : np.ndarray | None, optional
        1D array of binding energies of length N. If None, data point indices are used.
    plot_config : dict[str, Any] | None, optional
        Configuration dictionary containing:
        - 'window_size_ev' (float): Window size in eV (default 2.0).
        - 'sliding_stride_ev' (float): Sliding stride in eV (default 1.0).
        - 'window_size' (int | None): Explicit window size in points (overrides eV).
        - 'stride' (int | None): Explicit stride in points (overrides eV).
        - 'region' (str | None): Spectral region name (e.g. 'Ti2p').
        - 'title' (str | None): Custom suptitle for the figure.
        - 'invert_x' (bool): Invert x-axis per standard XPS convention (default True if energy is given).
        - 'offset_patches' (bool): If True, offset patches vertically in waterfall style (default False).
        - 'axes' (tuple[plt.Axes, plt.Axes] | None): Existing matplotlib axes.
        - 'show' (bool): Whether to invoke plt.show() (default False).

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Matplotlib figure and a tuple containing (ax_top, ax_bottom).

    Raises
    ------
    ValueError
        If spectrum is empty, or if energy is provided but does not match spectrum length.
    """
    cfg = plot_config or {}
    region = cfg.get("region")
    custom_title = cfg.get("title")
    invert_x: bool = cfg.get("invert_x", energy is not None)
    offset_patches: bool = bool(cfg.get("offset_patches", False))
    show: bool = bool(cfg.get("show", False))
    axes = cfg.get("axes")

    spectrum_1d = np.asarray(spectrum).ravel()
    n_points = len(spectrum_1d)
    if n_points == 0:
        raise ValueError("Cannot plot sliding window slices for an empty spectrum.")

    if energy is not None:
        energy_1d = np.asarray(energy).ravel()
        if len(energy_1d) != n_points:
            raise ValueError(
                f"Length mismatch: spectrum has {n_points} points but energy has {len(energy_1d)} points."
            )
        w_size, s_step = calculate_window_points(energy_1d, config=cfg)
        x_vals = energy_1d
        x_label = "Binding Energy (eV)"
    else:
        energy_1d = None
        x_vals = np.arange(n_points)
        x_label = "Data Point Index"
        w_size = int(cfg.get("window_size", cfg.get("window_size_points", 15)))
        s_step = int(cfg.get("stride", cfg.get("sliding_stride_points", max(1, w_size // 2))))

    windows, start_indices = extract_sliding_windows(spectrum_1d, window_size=w_size, stride=s_step)
    n_windows = len(windows)

    if axes is None:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]}
        )
    else:
        ax_top, ax_bottom = axes
        fig = ax_top.get_figure()

    # Generate distinct colors for windows
    cmap = plt.get_cmap("tab10" if n_windows <= 10 else "plasma")
    colors = [
        cmap(i % 10 if n_windows <= 10 else (i / max(1, n_windows - 1)))
        for i in range(n_windows)
    ]

    has_anchored = (
        (n_points - w_size) % s_step != 0
        and n_windows > 1
        and start_indices[-1] == n_points - w_size
    )

    # Top plot: Original spectrum with window spans
    ax_top.plot(x_vals, spectrum_1d, color="#1a252f", linewidth=2.2, label="Original Spectrum", zorder=3)

    for i, start_idx in enumerate(start_indices):
        x_win = x_vals[start_idx : start_idx + w_size]
        x_min, x_max = min(x_win[0], x_win[-1]), max(x_win[0], x_win[-1])
        is_last_anchored = has_anchored and (i == n_windows - 1)
        ax_top.axvspan(
            x_min,
            x_max,
            facecolor=colors[i],
            alpha=0.25 if is_last_anchored else 0.15,
            linestyle="--" if is_last_anchored else "-",
            edgecolor=colors[i] if is_last_anchored else "none",
            zorder=1,
        )

    delta_e_str = f", $\\Delta E$={abs(energy_1d[-1] - energy_1d[0]):.2f} eV" if energy_1d is not None else ""
    top_title = f"Original Spectrum {f'({region}) ' if region else ''}[N={n_points} pts{delta_e_str}]"
    ax_top.set_title(top_title, fontsize=11, fontweight="bold")
    ax_top.set_ylabel("Intensity (counts / a.u.)")
    ax_top.grid(True, linestyle="--", alpha=0.5)
    ax_top.legend(loc="upper right", framealpha=0.9)

    # Bottom plot: Extracted individual window patches
    spec_range = float(np.ptp(spectrum_1d)) or 1.0
    offset_step = (spec_range * 0.15) if offset_patches else 0.0

    for i, (win, start_idx) in enumerate(zip(windows, start_indices)):
        x_win = x_vals[start_idx : start_idx + w_size]
        y_win = win + (i * offset_step)
        is_anchored = has_anchored and (i == n_windows - 1)
        tag = "anchored" if is_anchored else f"pts {start_idx}:{start_idx+w_size}"
        lbl = f"Patch {i+1} ({tag})" if (n_windows <= 8 or is_anchored) else None
        ls = "--" if is_anchored else "-"
        ax_bottom.plot(
            x_win,
            y_win,
            color=colors[i],
            linewidth=2.0 if is_anchored else 1.8,
            linestyle=ls,
            marker=".",
            markersize=3,
            alpha=0.9 if is_anchored else 0.85,
            label=lbl,
        )

    anchor_badge = " [Right-Edge Anchored]" if has_anchored else ""
    bottom_title = (
        f"Extracted Sliding Windows (W={w_size} pts, S={s_step} pts, "
        f"Total Patches={n_windows}{anchor_badge})"
    )
    ax_bottom.set_title(bottom_title, fontsize=11, fontweight="bold")
    ax_bottom.set_xlabel(x_label)
    ax_bottom.set_ylabel("Patch Intensity" + (" (Waterfall Offset)" if offset_patches else ""))
    ax_bottom.grid(True, linestyle="--", alpha=0.5)

    if n_windows <= 8 or has_anchored:
        ax_bottom.legend(loc="upper right", fontsize=8, framealpha=0.9)

    if invert_x:
        if not ax_bottom.xaxis_inverted():
            ax_bottom.invert_xaxis()
        if not ax_top.xaxis_inverted():
            ax_top.invert_xaxis()

    suptitle = custom_title or f"Sliding Window Spectral Decomposition {f'({region})' if region else ''}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, (ax_top, ax_bottom)


__all__ = [
    "plot_regional_spectra",
    "plot_tool_comparison",
    "plot_pairing_timeline",
    "plot_max_intensity_vs_time",
    "plot_normalized_max_intensity_vs_time",
    "plot_shirley_background",
    "plot_training_history",
    "plot_prediction_comparison",
    "calculate_prediction_metrics",
    "plot_sliding_window_slices",
]
