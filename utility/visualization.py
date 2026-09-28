"""Visualization module for XPS spectra in GGRD-SP.

Provides utility functions to plot spectra and pairing relationships directly
from ary_energy, ary_intensity, and meta_df according to standard XPS conventions.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    for _, row in filtered_df.iterrows():
        idx = int(row["spectrum_index"])
        energy = ary_energy[idx]
        intensity = ary_intensity[idx]

        label = (
            f"{row.get('tool', 'Tool')} | {row.get('region', 'Region')} | "
            f"Die {row.get('die', '-')} | {row.get('measurement_id', '')}"
        )
        ax.plot(energy, intensity, label=label)

    ax.set_xlabel("Binding Energy (eV)")
    ax.set_ylabel("Intensity (counts / a.u.)")

    if invert_x and not ax.xaxis_inverted():
        ax.invert_xaxis()

    if title:
        ax.set_title(title)
    else:
        regions = filtered_df["region"].unique()
        tools = filtered_df["tool"].unique()
        ax.set_title(
            f"XPS Spectra - Region: {', '.join(map(str, regions))} | "
            f"Tool: {', '.join(map(str, tools))}"
        )

    ax.grid(True, linestyle="--", alpha=0.6)
    if len(filtered_df) <= 15:
        ax.legend(loc="best", fontsize="small")

    if show:
        plt.show()

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

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    palette = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
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
        energy = ary_energy[idx]
        intensity = ary_intensity[idx]

        ax.plot(
            energy,
            intensity,
            label=f"{tool} ({first_row['measurement_id']})",
            color=color_map.get(tool),
            linewidth=1.8,
        )
        plotted_count += 1

    if plotted_count == 0:
        raise ValueError(
            f"No matching spectra found for tools={tools}, region='{region}', die={die}"
        )

    ax.set_xlabel("Binding Energy (eV)")
    ax.set_ylabel("Intensity (counts / a.u.)")
    ax.set_title(f"Tool Comparison ({' vs '.join(tools)}) - Region: {region} (Die {die})")

    if invert_x and not ax.xaxis_inverted():
        ax.invert_xaxis()

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")

    if show:
        plt.show()

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

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
    else:
        fig = ax.get_figure()

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

    if title:
        ax.set_title(title)
    else:
        n_pairs = len(paired_src_ids)
        ax.set_title(
            f"1-to-1 Measurement Pairing Timeline ({source_tool} -> {target_tool}) "
            f"| Paired: {n_pairs}/{len(src_df)}"
        )

    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax.legend(loc="upper right", fontsize="small", framealpha=0.9)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, ax
