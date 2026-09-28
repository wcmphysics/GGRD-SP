"""Visualization module for XPS spectra in GGRD-SP.

Provides utility functions to plot spectra directly from ary_energy, ary_intensity,
and meta_df according to standard XPS conventions.
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
