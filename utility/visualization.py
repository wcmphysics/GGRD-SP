"""Visualization module for XPS spectra in GGRD-SP.

Provides utility functions to plot spectra, pairing relationships, and Shirley
background subtraction directly from ary_energy, ary_intensity, and meta_df.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utility.quantification import (
    DEFAULT_SCOFIELD_RSF,
    calculate_shirley_background,
)


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
        - 'ti2p_auto_endpoints' (bool): Auto-endpoints for Ti2p (default True).
        - 'ti2p_smooth_endpoints_search' (bool): Smooth search for Ti2p minima (default False).
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
    ti2p_auto: bool = cfg.get("ti2p_auto_endpoints", True)
    ti2p_smooth: bool = cfg.get("ti2p_smooth_endpoints_search", False)
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
        e_arr = ary_energy[spec_idx]
        i_arr = ary_intensity[spec_idx]

        shirley_cfg = {
            "region": region_name,
            "ti2p_auto_endpoints": ti2p_auto,
            "ti2p_smooth_endpoints_search": ti2p_smooth,
        }
        b_arr, net_area = calculate_shirley_background(e_arr, i_arr, shirley_cfg)

        if custom_ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.5))
        else:
            fig = custom_ax.get_figure()
            ax = custom_ax

        ax.plot(e_arr, i_arr, label="Raw Spectrum", color="navy", linewidth=1.8)
        ax.plot(e_arr, b_arr, label="Shirley Background", color="crimson", linestyle="--", linewidth=1.6)
        ax.fill_between(e_arr, b_arr, i_arr, where=(i_arr > b_arr), color="skyblue", alpha=0.35, label=f"Net Area: {net_area:.1f}")

        ax.set_xlabel("Binding Energy (eV)")
        ax.set_ylabel("Intensity (counts / a.u.)")
        ax.set_title(f"{meas_id} | Region: {region_name} (Die {die})")
        if invert_x and not ax.xaxis_inverted():
            ax.invert_xaxis()

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")
        if show:
            plt.show()
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
        e_arr = ary_energy[spec_idx]
        i_arr = ary_intensity[spec_idx]

        shirley_cfg = {
            "region": reg,
            "ti2p_auto_endpoints": ti2p_auto,
            "ti2p_smooth_endpoints_search": ti2p_smooth,
        }
        b_arr, net_area = calculate_shirley_background(e_arr, i_arr, shirley_cfg)

        ax.plot(e_arr, i_arr, label="Raw", color="navy", linewidth=1.5)
        ax.plot(e_arr, b_arr, label="Shirley", color="crimson", linestyle="--", linewidth=1.4)
        ax.fill_between(e_arr, b_arr, i_arr, where=(i_arr > b_arr), color="skyblue", alpha=0.35, label=f"Area: {net_area:.1f}")

        ax.set_title(f"Region: {reg}")
        ax.set_xlabel("Binding Energy (eV)")
        ax.set_ylabel("Intensity")
        if invert_x and not ax.xaxis_inverted():
            ax.invert_xaxis()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize="x-small")

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

        ax.set_title(f"Region: {reg}", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (Normalized MSE)" if not log_scale else "Loss (log scale)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", fontsize="small")

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

    ax_main.set_ylabel("Intensity (counts)")
    ax_main.set_title(
        f"Spectral Transfer Comparison | Region: {actual_region} | Die {die}\n"
        f"Source: {meas_id} ({src_tool}) -> Target: {tgt_meas_id} ({tgt_tool})",
        fontsize=11,
    )
    ax_main.legend(loc="best", fontsize="small")
    ax_main.grid(True, linestyle="--", alpha=0.6)

    # Residual plot
    if ax_res is not None:
        ax_res.plot(energy, residual, color="purple", linewidth=1.4, label="Target - Predicted")
        ax_res.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
        ax_res.set_xlabel("Binding Energy (eV)")
        ax_res.set_ylabel("Residual")
        ax_res.grid(True, linestyle="--", alpha=0.6)
        ax_res.legend(loc="best", fontsize="x-small")
    else:
        ax_main.set_xlabel("Binding Energy (eV)")

    if invert_x and not ax_main.xaxis_inverted():
        ax_main.invert_xaxis()

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes


def calculate_prediction_metrics(
    data_original: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    data_predicted: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate quantitative evaluation metrics comparing predicted spectra with ground truth target spectra.

    Computes Normalized MSE, Raw RMSE, and Peak Relative Error % per paired sample.

    Parameters
    ----------
    data_original : tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Original measured data (ary_intensity, ary_energy, meta_df).
    data_predicted : tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Predicted data (ary_intensity_predicted, ary_energy_predicted, meta_df_predicted).
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'regions' (list[str] | None): Specific regions to evaluate (default all).
        - 'eps' (float): Epsilon floor for maximum normalization (default 1e-4).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_per_sample, df_summary_by_region)
    """
    ary_intensity_orig, _, meta_df_orig = data_original
    ary_intensity_pred, _, meta_df_pred = data_predicted

    cfg = config or {}
    regions_filter = cfg.get("regions")
    eps: float = float(cfg.get("eps", 1e-4))

    records: list[dict[str, Any]] = []

    # Iterate over predicted spectra records
    for _, pred_row in meta_df_pred.iterrows():
        region = str(pred_row["region"])
        if regions_filter is not None and region not in regions_filter:
            continue

        src_meas_id = str(pred_row["source_measurement_id"])
        die = int(pred_row["die"])
        pred_idx = int(pred_row["spectrum_index"])

        # Find matching source row in meta_df_orig to obtain paired target measurement ID
        src_match = meta_df_orig[
            (meta_df_orig["measurement_id"] == src_meas_id)
            & (meta_df_orig["region"] == region)
            & (meta_df_orig["die"] == die)
        ]
        if src_match.empty or pd.isna(src_match["measurement_id_target"].iloc[0]):
            continue

        tgt_meas_id = str(src_match["measurement_id_target"].iloc[0])
        tgt_match = meta_df_orig[
            (meta_df_orig["measurement_id"] == tgt_meas_id)
            & (meta_df_orig["region"] == region)
            & (meta_df_orig["die"] == die)
        ]
        if tgt_match.empty:
            continue

        tgt_idx = int(tgt_match["spectrum_index"].iloc[0])

        y_true = ary_intensity_orig[tgt_idx]
        y_pred = ary_intensity_pred[pred_idx]

        max_true = max(float(np.max(np.abs(y_true))), eps)
        max_pred = float(np.max(np.abs(y_pred)))

        norm_mse = float(np.mean(((y_pred - y_true) / max_true) ** 2))
        raw_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        peak_err_pct = float(abs(max_pred - max_true) / max_true * 100.0)

        records.append({
            "source_measurement_id": src_meas_id,
            "target_measurement_id": tgt_meas_id,
            "region": region,
            "die": die,
            "normalized_mse": norm_mse,
            "rmse": raw_rmse,
            "peak_err_pct": peak_err_pct,
        })

    df_per_sample = pd.DataFrame(records)
    if df_per_sample.empty:
        df_summary = pd.DataFrame(
            columns=[
                "region",
                "normalized_mse_mean",
                "normalized_mse_std",
                "rmse_mean",
                "rmse_std",
                "peak_err_pct_mean",
                "peak_err_pct_std",
            ]
        )
    else:
        df_summary = (
            df_per_sample.groupby("region")[["normalized_mse", "rmse", "peak_err_pct"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        # Flatten column multi-index
        df_summary.columns = [
            "_".join(col).strip("_") for col in df_summary.columns.values
        ]

    return df_per_sample, df_summary
