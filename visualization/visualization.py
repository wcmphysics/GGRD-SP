"""Visualization utilities for regional spectra, timelines, and model
comparisons."""

from __future__ import annotations

import colorsys
from typing import Any
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utility.patching import calculate_window_points, extract_sliding_windows
from utility.quantification import calculate_shirley_background, compute_die_total_flux

__all__ = [
    "plot_regional_spectra",
    "plot_tool_comparison",
    "plot_pairing_timeline",
    "plot_max_intensity_vs_time",
    "plot_normalized_max_intensity_vs_time",
    "plot_shirley_background",
    "plot_training_history",
    "plot_prediction_comparison",
    "plot_sliding_window_slices",
    "plot_atomic_percentage_distributions",
    "plot_atomic_percentage_mae",
    "plot_flux_and_averaged_intensity_vs_time",
]


def _prepare_canvas(ax: Any = None, figsize: tuple[float, float] = (8, 5), nrows: int = 1, ncols: int = 1,
                    **kwargs: Any) -> tuple[plt.Figure, Any]:
    """Return figure and axes, creating new ones if ax is None."""
    if ax is not None:
        return (ax.get_figure() if hasattr(ax, "get_figure") else ax[0].get_figure(), ax)
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


def _setup_subplot_grid(n_items: int, max_cols: int = 3, col_w: float = 5.0, row_h: float = 3.8,
                        figsize: tuple[float, float] | None = None) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    """Create a standard grid of subplots for multi-region visualization."""
    nc = min(max_cols, max(1, n_items))
    nr = (n_items + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=figsize or (col_w * nc, row_h * nr), squeeze=False)
    for ax in axes.flatten()[n_items:]:
        ax.set_visible(False)
    return fig, axes, axes.flatten()


def _finalize_plot(ax: plt.Axes, config: dict[str, Any] | None = None, *, fig: plt.Figure | None = None,
                   title: str | None = None, xlabel: str | None = None, ylabel: str | None = None,
                   invert_x: bool = True, grid: bool = True, grid_alpha: float = 0.6, show: bool = False,
                   **kwargs: Any) -> None:
    """Apply standard plot styling, labels, inverted x-axis, and display."""
    cfg = {**kwargs, **(config or {})}
    for val, setter in [(title or cfg.get("title"), ax.set_title),
                        (xlabel or cfg.get("xlabel"), ax.set_xlabel),
                        (ylabel or cfg.get("ylabel"), ax.set_ylabel)]:
        if val:
            setter(val)
    if cfg.get("invert_x", invert_x) and not ax.xaxis_inverted():
        ax.invert_xaxis()
    if cfg.get("grid", grid):
        ax.grid(True, linestyle="--", alpha=float(cfg.get("grid_alpha", grid_alpha)))
    if fig is not None:
        fig.tight_layout()
    if cfg.get("show", show):
        plt.show()


def _finish_fig(fig: plt.Figure, cfg: dict[str, Any], title: str | None = None) -> None:
    """Apply figure suptitle, tight layout, and optional show."""
    if t := (title if title is not None else cfg.get("title")):
        fig.suptitle(t, fontsize=12, fontweight="bold")
    if fig is not None:
        fig.tight_layout()
    if cfg.get("show"):
        plt.show()


def _check_meta(df: pd.DataFrame, req_cols: set[str], name: str = "df_meta") -> None:
    """Validate DataFrame presence and required column existence."""
    if df.empty:
        raise ValueError(f"{name} cannot be empty.")
    if missing := (req_cols - set(df.columns)):
        raise KeyError(f"{name} missing columns: {sorted(missing)}")


def plot_regional_spectra(ary_energy: np.ndarray, ary_intensity: np.ndarray, df_meta: pd.DataFrame,
                          plot_config: dict[str, Any] | None = None) -> tuple[plt.Figure, plt.Axes]:
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
            raise KeyError(f"Filter key '{k}' not found in df_meta columns: {list(df.columns)}")
        df = df[df[k] == v]
    if not cfg.get("filters"):
        df = df[df["measurement_id"] == df["measurement_id"].iloc[0]]
    if df.empty:
        raise ValueError(f"No spectra match specified filters: {cfg.get('filters')}")
    fig, ax = _prepare_canvas(cfg.get("ax"), figsize=(8, 5))
    df = df.iloc[: cfg.get("max_spectra", 10)]
    for _, r in df.iterrows():
        lbl = (f"{r.get('tool', 'Tool')} | {r.get('region', 'Region')} | "
               f"Die {r.get('die', '-')} | {r.get('measurement_id', '')}")
        ax.plot(ary_energy[int(r["spectrum_index"])], ary_intensity[int(r["spectrum_index"])], label=lbl)
    if len(df) <= 15:
        ax.legend(loc="best", fontsize="small")
    t_str = (f"XPS Spectra - Region: {', '.join(map(str, df['region'].unique()))} | "
             f"Tool: {', '.join(map(str, df['tool'].unique()))}")
    _finalize_plot(ax, cfg, fig=fig, title=cfg.get("title") or t_str, xlabel="Binding Energy (eV)",
                   ylabel="Intensity (counts / a.u.)", invert_x=cfg.get("invert_x", True))
    return fig, ax


def plot_tool_comparison(ary_energy: np.ndarray, ary_intensity: np.ndarray, df_meta: pd.DataFrame,
                          compare_config: dict[str, Any] | None = None) -> tuple[plt.Figure, plt.Axes]:
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
    region, die, tools = cfg.get("region", "Al2p"), int(cfg.get("die", 0)), list(cfg.get("tools", ("J4", "J5")))
    if len(tools) < 2:
        raise ValueError(f"At least 2 tools required for comparison, got {len(tools)}: {tools}")
    sub = df_meta[(df_meta["region"] == region) & (df_meta["die"] == die)]
    fig, ax = _prepare_canvas(cfg.get("ax"), figsize=(8, 5))
    plotted = 0
    for t in tools:
        if not (t_sub := sub[sub["tool"] == t]).empty:
            idx = int(t_sub.iloc[0]["spectrum_index"])
            ax.plot(ary_energy[idx], ary_intensity[idx], label=f"Tool: {t} ({t_sub.iloc[0].get('measurement_id', '')})")
            plotted += 1
    if plotted == 0:
        raise ValueError(f"No spectra found for tools {tools} at region='{region}', die={die}")
    ax.legend(loc="best", fontsize="small")
    _finalize_plot(ax, cfg, fig=fig, title=cfg.get("title") or f"Tool Comparison - Region: {region} | Die: {die}",
                   xlabel="Binding Energy (eV)", ylabel="Intensity (counts / a.u.)", invert_x=cfg.get("invert_x", True))
    return fig, ax


def _calculate_staggered_levels(times: list[pd.Timestamp], max_levels: int = 3,
                                 proximity_ratio: float = 0.08) -> list[int]:
    """Calculate staggered vertical offset levels to prevent collision.

    Parameters
    ----------
    times : list[pd.Timestamp]
        Chronologically sorted list of timestamps.
    max_levels : int
        Maximum number of staggered vertical offset levels (default 3).
    proximity_ratio : float
        Fraction of total timeline span considered close (default 0.08).

    Returns
    -------
    list[int]
        Assigned level indices (0 to max_levels - 1) for each timestamp.
    """
    if not times or len(times) == 1:
        return [0] if times else []
    if (max_lvl := max(1, int(max_levels))) <= 1 or (t_span := (max(times) - min(times)).total_seconds()) <= 0:
        return [i % max_lvl for i in range(len(times))]
    thresh, assigned, last_t = t_span * proximity_ratio, [], {lvl: -1e18 for lvl in range(max_lvl)}
    for t in times:
        lvl = next((l for l in range(max_lvl) if (t.timestamp() - last_t[l]) >= thresh),
                   min(range(max_lvl), key=lambda l: last_t[l]))
        assigned.append(lvl)
        last_t[lvl] = t.timestamp()
    return assigned


def plot_pairing_timeline(df_meta: pd.DataFrame, plot_config: dict[str, Any] | None = None
                          ) -> tuple[plt.Figure, plt.Axes]:
    """Plot timeline of measurement sessions and connect 1-to-1 paired sessions.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata DataFrame with pairing columns ('tool_target',
        'measurement_id_target', 'time_target', 'time_diff_hours').
    plot_config : dict[str, Any] | None, optional
        Dictionary bundling plotting options:
        - 'source_tool' (str): Name of the source tool (default 'J4').
        - 'target_tool' (str): Name of the target tool (default 'J5').
        - 'annotate_measurements' (bool): Label points with measurement_id
          (default True).
        - 'label_fontsize' (int): Font size for labels (default 8).
        - 'stagger_levels' (int): Vertical levels for collision avoidance
          (default 3).
        - 'leader_lines' (bool): Draw leader lines for offset labels
          (default True).
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
        If required columns are missing from df_meta.
    ValueError
        If no measurement sessions exist for source or target tool.
    """
    _check_meta(df_meta, {"tool", "measurement_id", "time"})
    cfg, df = plot_config or {}, df_meta.copy()
    src_tool, tgt_tool = cfg.get("source_tool", "J4"), cfg.get("target_tool", "J5")
    for col in ("time", "time_target"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    src_df = df[df["tool"] == src_tool].drop_duplicates("measurement_id").sort_values("time")
    tgt_df = df[df["tool"] == tgt_tool].drop_duplicates("measurement_id").sort_values("time")
    if src_df.empty and tgt_df.empty:
        raise ValueError(f"No measurements found for tools '{src_tool}' or '{tgt_tool}'.")
    fig, ax = _prepare_canvas(cfg.get("ax"), figsize=(10, 4.5))
    if not (all_times := pd.concat([src_df["time"], tgt_df["time"]]).dropna()).empty:
        ax.hlines([1.0, 0.0], all_times.min(), all_times.max(), colors="gray", linestyles=":", alpha=0.5)
    s_tgt = (src_df.dropna(subset=["measurement_id_target", "time_target"])
             if {"measurement_id_target", "time_target"}.issubset(src_df.columns) else pd.DataFrame())
    paired_src, paired_tgt = (set(s_tgt.get("measurement_id", []).astype(str)),
                              set(s_tgt.get("measurement_id_target", []).astype(str)))
    for _, r in s_tgt.iterrows():
        ax.plot([r["time"], r["time_target"]], [1.0, 0.0], color="tab:purple", alpha=0.7, lw=1.5)
        diff_h = r.get("time_diff_hours")
        ax.annotate(f"{diff_h:.1f}h" if pd.notna(diff_h) else "",
                    (r["time"] + (r["time_target"] - r["time"]) / 2, 0.5),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, color="darkmagenta")
    for sub, y_pos, t_name, p_ids, col in [(src_df, 1.0, src_tool, paired_src, "tab:blue"),
                                           (tgt_df, 0.0, tgt_tool, paired_tgt, "tab:green")]:
        p, up = sub[sub["measurement_id"].isin(p_ids)], sub[~sub["measurement_id"].isin(p_ids)]
        if not p.empty:
            ax.scatter(p["time"], [y_pos] * len(p), color=col, s=70, zorder=3, label=f"{t_name} (Paired)")
        if not up.empty:
            ax.scatter(up["time"], [y_pos] * len(up), facecolors="none", edgecolors=col, s=70, lw=1.8,
                       zorder=3, label=f"{t_name} (Unpaired)")
    if bool(cfg.get("annotate_measurements", cfg.get("show_labels", True))):
        stagger = max(1, int(cfg.get("stagger_levels", 3)))
        lead, sz = bool(cfg.get("leader_lines", True)), int(cfg.get("label_fontsize", 8))
        for s_df, y_pos, is_src, col in [(src_df, 1.0, True, "tab:blue"), (tgt_df, 0.0, False, "tab:green")]:
            if not s_df.empty:
                times = [pd.Timestamp(t) for t in s_df["time"]]
                for t, m_id, lvl in zip(times, s_df["measurement_id"].astype(str),
                                        _calculate_staggered_levels(times, max_levels=stagger)):
                    off_y = (10.0 + lvl * 14.0) if is_src else (-12.0 - lvl * 14.0)
                    arr = ({"arrowstyle": "-", "color": "gray", "alpha": 0.5, "linewidth": 0.8}
                           if (lead and lvl > 0) else None)
                    ax.annotate(m_id, xy=(t, y_pos), xytext=(0, off_y), textcoords="offset points", ha="center",
                                va="bottom" if is_src else "top", fontsize=sz, color=col, arrowprops=arr, clip_on=False)
        ax.set_ylim(-0.25 - (stagger - 1) * 0.10, 1.25 + (stagger - 1) * 0.10)
    else:
        ax.set_ylim(-0.3, 1.3)
    ax.set_yticks([0.0, 1.0], [f"Target ({tgt_tool})", f"Source ({src_tool})"])
    ax.legend(loc="upper right", fontsize="small", framealpha=0.9)
    d_title = (f"1-to-1 Measurement Pairing Timeline ({src_tool} -> {tgt_tool}) | "
               f"Paired: {len(paired_src)}/{len(src_df)}")
    _finalize_plot(ax, cfg, fig=fig, title=cfg.get("title") or d_title,
                   xlabel="Measurement Time", invert_x=False, grid_alpha=0.5)
    return fig, ax


def _render_intensity_time_series(df_meta: pd.DataFrame, y_calc: Any, cfg: dict[str, Any],
                                  region: str | None, die: int | None, d_title: str, d_ylabel: str
                                  ) -> tuple[plt.Figure, plt.Axes]:
    """Resolve criteria, filter df_meta, compute y series, and render plot."""
    if df_meta.empty:
        raise ValueError("df_meta cannot be empty")
    filts = dict(cfg.get("filters") or {})
    t_reg = str(r) if (r := region or cfg.get("region") or filts.get("region")) is not None else None
    d = die if die is not None else (cfg.get("die") if "die" in cfg else filts.get("die"))
    t_die = int(d) if d is not None and str(d).isdigit() else d
    comb = {**filts, **({"region": t_reg} if t_reg is not None else {}),
            **({"die": t_die} if t_die is not None else {})}
    df = df_meta.copy()
    for k, v in comb.items():
        if k not in df.columns:
            raise KeyError(f"Filter key '{k}' not found in df_meta columns: {list(df.columns)}")
        df = df[((df["die"] == v) | (df["die"] == str(v))) if (k == "die" and str(v).isdigit()) else (df[k] == v)]
    if df.empty:
        raise ValueError(f"No spectra match filters (region={t_reg!r}, die={t_die!r}, filters={comb}).")
    df["time"], df["_y"], hue = pd.to_datetime(df["time"]), y_calc(df), str(cfg.get("hue", "tool"))
    if hue == "die":
        df["_hue"] = df["die"].apply(lambda x: f"Die {int(x)}" if str(x).isdigit() else f"Die {x}")
        p_hue, h_order = "_hue", [f"Die {int(x)}" if str(x).isdigit() else f"Die {x}"
                                  for x in sorted(df["die"].dropna().unique())]
    else:
        p_hue, h_order = hue, (sorted(df[hue].dropna().unique()) if hue in ("tool", "region") else None)
    fig, ax = _prepare_canvas(cfg.get("ax"), figsize=tuple(cfg.get("figsize", (10, 5.5))))
    sns.scatterplot(data=df, x="time", y="_y", hue=p_hue, hue_order=h_order, style=cfg.get("style"),
                    palette=cfg.get("palette"), alpha=float(cfg.get("alpha", 0.75)), s=float(cfg.get("s", 45.0)), ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    tags = ([f"Region: {t_reg}"] if t_reg else []) + ([f"Die: {t_die}"] if t_die is not None else []) + [
        f"{str(k).capitalize()}: {v}" for k, v in filts.items() if k not in ("region", "die")
    ]
    tag_suffix = f" ({', '.join(tags)})" if tags else ""
    title = cfg.get("title") or f"{d_title}{tag_suffix}"    
    _finalize_plot(ax, cfg, fig=fig, title=title,
                   xlabel="Measurement Time", ylabel=cfg.get("ylabel") or d_ylabel, invert_x=False, grid_alpha=0.5)
    return fig, ax


def plot_max_intensity_vs_time(ary_intensity: np.ndarray, df_meta: pd.DataFrame,
                               plot_config: dict[str, Any] | None = None, *,
                               region: str | None = None, die: int | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Plot maximum intensity over measurement time.

    Parameters
    ----------
    ary_intensity : np.ndarray
        2D array of measured spectral intensities, shape (N_spectra, N_points).
    df_meta : pd.DataFrame
        Metadata DataFrame with 'time', 'spectrum_index', and hue column.
    plot_config : dict[str, Any] | None, optional
        Dictionary bundling plotting options:
        - 'filters' (dict[str, Any] | None): Criteria dictionary.
        - 'region' (str | None): Region filter override.
        - 'die' (int | None): Die filter override.
        - 'hue' (str): Column to group/color by (default 'tool').
        - 'style' (str | None): Categorical column for marker shapes.
        - 'palette' (str | dict | list | None): Seaborn color palette.
        - 'alpha' (float): Marker transparency (default 0.75).
        - 's' (float): Marker size in points**2 (default 45.0).
        - 'title' (str | None): Custom figure title.
        - 'ylabel' (str | None): Custom y-axis label.
        - 'figsize' (tuple[float, float]): Dimensions (default (10, 5.5)).
        - 'ax' (plt.Axes | None): Existing matplotlib Axes.
        - 'show' (bool): Whether to invoke plt.show() (default False).
    region : str | None, optional
        Target spectral region. Precedence over plot_config['region'].
    die : int | None, optional
        Target wafer die index. Precedence over plot_config['die'].

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The matplotlib Figure and Axes objects.

    Raises
    ------
    ValueError
        If df_meta is empty or no spectra match filters.
    KeyError
        If required columns are missing from df_meta.
    """
    cfg, hue = plot_config or {}, str((plot_config or {}).get("hue", "tool"))
    _check_meta(df_meta, {"time", "spectrum_index", hue})
    return _render_intensity_time_series(
        df_meta, lambda df: np.max(ary_intensity[df["spectrum_index"].astype(int).values], axis=1),
        cfg, region, die, f"Maximum Intensity vs. Time (hue={hue})", "Maximum Intensity (counts)"
    )


def plot_normalized_max_intensity_vs_time(ary_intensity: np.ndarray, df_meta: pd.DataFrame,
                                          ary_energy: np.ndarray | None = None,
                                          plot_config: dict[str, Any] | None = None, *,
                                          region: str | None = None, die: int | None = None
                                          ) -> tuple[plt.Figure, plt.Axes]:
    """Plot normalized maximum intensity over measurement time.

    Parameters
    ----------
    ary_intensity : np.ndarray
        2D array of measured spectral intensities, shape (N_spectra, N_points).
    df_meta : pd.DataFrame
        Metadata DataFrame containing spectrum details and 'spectrum_index'.
    ary_energy : np.ndarray | None, optional
        2D array of binding energies, shape (N_spectra, N_points).
    plot_config : dict[str, Any] | None, optional
        Dictionary bundling plotting options:
        - 'normalization_mode' (str): Either 'die_total_flux' (default)
          or 'spectrum_area'.
        - 'filters' (dict[str, Any] | None): Criteria dictionary.
        - 'region' (str | None): Region filter override.
        - 'die' (int | None): Die filter override.
        - 'hue' (str): Column to group/color by (default 'tool').
        - 'style' (str | None): Categorical column for marker shapes.
        - 'palette' (str | dict | list | None): Seaborn color palette.
        - 'alpha' (float): Marker transparency (default 0.75).
        - 's' (float): Marker size in points**2 (default 45.0).
        - 'title' (str | None): Custom figure title.
        - 'ylabel' (str | None): Custom y-axis label.
        - 'figsize' (tuple[float, float]): Dimensions (default (10, 5.5)).
        - 'ax' (plt.Axes | None): Existing matplotlib Axes.
        - 'show' (bool): Whether to invoke plt.show() (default False).
    region : str | None, optional
        Target spectral region. Precedence over plot_config['region'].
    die : int | None, optional
        Target wafer die index. Precedence over plot_config['die'].

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The matplotlib Figure and Axes objects.

    Raises
    ------
    ValueError
        If df_meta is empty, no records match filters, or unsupported mode.
    KeyError
        If required columns are missing from df_meta.
    """
    cfg, hue = plot_config or {}, str((plot_config or {}).get("hue", "tool"))
    norm_mode = str(cfg.get("normalization_mode", "die_total_flux")).lower()
    _check_meta(df_meta, {"time", "spectrum_index", hue})
    if norm_mode not in ("die_total_flux", "spectrum_area"):
        raise ValueError(f"Unsupported normalization_mode '{norm_mode}'. Must be 'die_total_flux' or 'spectrum_area'.")
    if norm_mode == "die_total_flux":
        _check_meta(df_meta, {"measurement_id", "die"})
        f_map = compute_die_total_flux(df_meta, ary_intensity, ary_energy)
        y_calc = lambda df: [float(m) / max(float(f_map.get((str(r["measurement_id"]), int(r["die"])), 1.0)), 1e-4)
                             for m, (_, r) in zip(np.max(ary_intensity[df["spectrum_index"].astype(int).values],
                                                         axis=1), df.iterrows())]
        d_lbl = "Max Intensity / Die Total Integrated Area (1/eV)"
    else:
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        def y_calc(df: pd.DataFrame) -> np.ndarray:
            idx = df["spectrum_index"].astype(int).values
            areas = [max(float(abs(trapz_fn(ary_intensity[i], x=ary_energy[i]))) if ary_energy is not None
                         else float(np.sum(ary_intensity[i])), 1e-4) for i in idx]
            return np.max(ary_intensity[idx], axis=1) / np.array(areas)
        d_lbl = "Max Intensity / Spectrum Integrated Area (1/eV)"
    return _render_intensity_time_series(df_meta, y_calc, cfg, region, die,
                                         f"Normalized Maximum Intensity vs. Time (hue={hue})", d_lbl)


def plot_shirley_background(ary_energy: np.ndarray, ary_intensity: np.ndarray, df_meta: pd.DataFrame,
                            plot_config: dict[str, Any] | None = None) -> tuple[plt.Figure, Any]:
    """Plot regional spectrum with Shirley background and net peak area.

    Parameters
    ----------
    ary_energy : np.ndarray
        2D array of binding energies.
    ary_intensity : np.ndarray
        2D array of measured intensities.
    df_meta : pd.DataFrame
        Metadata DataFrame.
    plot_config : dict[str, Any] | None, optional
        Dictionary bundling plotting options:
        - 'measurement_id' (str | None): Measurement session to display.
        - 'die' (int): Die index (default 0).
        - 'region' (str | None): Specific region to plot (e.g. 'Ti2p').
          If None, all regions are plotted in subplots.
        - 'strategy' (str): Endpoint strategy ('minima', 'edge', 'direct').
        - 'average_width_ev' (float | None): Window in eV (default 1.0).
        - 'region_endpoint_config' (dict | None): Regional overrides.
        - 'invert_x' (bool): Invert x-axis per standard XPS convention
          (default True).
        - 'ax' (plt.Axes | None): Matplotlib Axes (single region only).
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
    die, target_region = int(cfg.get("die", 0)), cfg.get("region")
    subset = df_meta[df_meta["die"] == die]
    meas_id = cfg.get("measurement_id") or (str(subset["measurement_id"].iloc[0]) if not subset.empty else None)
    if meas_id is not None:
        subset = subset[subset["measurement_id"] == meas_id]
    if target_region is not None:
        subset = subset[subset["region"] == target_region]
    if subset.empty:
        raise ValueError(f"No matching spectra for meas_id='{meas_id}', die={die}, region='{target_region}'")
    is_single = target_region is not None or len(subset) == 1
    regions = [str(subset["region"].iloc[0])] if is_single else list(subset["region"].unique())
    if is_single:
        fig, ax = _prepare_canvas(cfg.get("ax"), (7, 4.5))
        axes, axes_flat = ax, [ax]
    else:
        fig, axes, axes_flat = _setup_subplot_grid(len(regions), max_cols=3, col_w=5.0, row_h=3.8)
    for idx, reg in enumerate(regions):
        ax = axes_flat[idx]
        spec_idx = int(subset[subset["region"] == reg].iloc[0]["spectrum_index"])
        e_arr, i_arr = ary_energy[spec_idx], ary_intensity[spec_idx]
        b_arr, net_area = calculate_shirley_background(e_arr, i_arr, {**cfg, "region": reg})
        ax.plot(e_arr, i_arr, label="Raw Spectrum" if is_single else "Raw", color="navy", lw=1.6)
        ax.plot(e_arr, b_arr, label="Shirley Background" if is_single else "Shirley", color="crimson", ls="--", lw=1.5)
        ax.fill_between(e_arr, b_arr, i_arr, where=(i_arr > b_arr), color="skyblue", alpha=0.35,
                        label=f"{'Net Area' if is_single else 'Area'}: {net_area:.1f}")
        ax.legend(loc="best", fontsize="small" if is_single else "x-small")
        _finalize_plot(ax, title=f"{meas_id} | Region: {reg} (Die {die})" if is_single else f"Region: {reg}",
                       xlabel="Binding Energy (eV)", ylabel="Intensity (counts / a.u.)" if is_single else "Intensity",
                       invert_x=cfg.get("invert_x", True), grid_alpha=0.5)
    if not is_single:
        fig.suptitle(f"Shirley Background Subtraction | {meas_id} (Die {die})", fontsize=13)
    _finish_fig(fig, cfg)
    return fig, axes


def plot_training_history(histories: dict[str, dict[str, list[float]]],
                          plot_config: dict[str, Any] | None = None
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
    fig, axes, axes_flat = _setup_subplot_grid(len(regs), max_cols=3, col_w=5.0, row_h=3.8)
    for idx, reg in enumerate(regs):
        ax, h = axes_flat[idx], histories[reg]
        ax.plot(range(1, len(h.get("train_loss", [])) + 1), h.get("train_loss", []), label="Train Loss",
                color="royalblue", lw=1.8)
        if h.get("val_loss"):
            ax.plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], label="Val Loss", color="darkorange",
                    ls="--", lw=1.8)
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize="small")
        _finalize_plot(ax, title=f"Region: {reg}", xlabel="Epoch",
                       ylabel="Loss (log scale)" if log_scale else "Loss (Normalized MSE)",
                       invert_x=False, grid_alpha=0.6)
    _finish_fig(fig, cfg, cfg.get("title", "Baseline Neural Network Training History"))
    return fig, axes


def plot_prediction_comparison(data_original: tuple[np.ndarray, np.ndarray, pd.DataFrame],
                               data_predicted: tuple[np.ndarray, np.ndarray, pd.DataFrame],
                               config: dict[str, Any] | None = None) -> tuple[plt.Figure, Any]:
    """Plot overlay comparison of Source, True Target, and Transformed spectra.

    Color scheme:
    - Source tool spectrum: solid black line
    - Target tool spectrum: solid red line
    - Transformed source spectrum: blue dotted line

    When 'region' is None (default), plots all available spectral regions for
    the measurement in a multi-panel subplot grid. Defaults to a test split
    measurement session if available.

    Parameters
    ----------
    data_original : tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Tuple of (ary_intensity_orig, ary_energy_orig, df_meta_orig).
    data_predicted : tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Tuple of (ary_intensity_pred, ary_energy_pred, df_meta_pred).
    config : dict[str, Any] | None, optional
        Dictionary bundling options:
        - 'measurement_id' (str | None): Source measurement session ID.
        - 'region' (str | None): Specific spectral region. If None, plots all
          available regions in a subplot grid.
        - 'die' (int): Target wafer die index (default 0).
        - 'session_splits' (dict[str, list[str]] | None): Splits dict.
        - 'show_residual' (bool): Show difference subplot (default True).
        - 'invert_x' (bool): Invert binding energy axis (default True).
        - 'figsize' (tuple[float, float]): Figure size.
        - 'title' (str | None): Custom plot title.
        - 'show' (bool): Whether to invoke plt.show() (default False).

    Returns
    -------
    tuple[plt.Figure, Any]
        The Figure and Axes (or array of Axes for multi-panel).

    Raises
    ------
    ValueError
        If no paired sessions exist, die/region not found, or spectra missing.
    """
    (ary_i_orig, ary_e_orig, meta_orig), (ary_i_pred, ary_e_pred, meta_pred) = data_original, data_predicted
    cfg, die = config or {}, int((config or {}).get("die", 0))
    meas_id, target_region = cfg.get("measurement_id"), cfg.get("region")
    if meas_id is None:
        p_ids = set(meta_pred.get("source_measurement_id", []).dropna().astype(str))
        cands = ([str(c) for c in (cfg.get("session_splits") or {}).get("test", [])] or
                 (list(meta_orig[meta_orig["split"].astype(str).str.lower() == "test"]["measurement_id"].astype(str))
                  if "split" in meta_orig else []) or
                 (list(meta_orig.dropna(subset=["measurement_id_target"])["measurement_id"].astype(str))
                  if "measurement_id_target" in meta_orig else []))
        if not (meas_id := next((c for c in (cands or list(p_ids)) if c in p_ids), None)):
            raise ValueError("No paired source measurements found in df_meta_orig.")
    src_rows = meta_orig[(meta_orig["measurement_id"] == meas_id) & (meta_orig["die"] == die)]
    if src_rows.empty:
        raise ValueError(f"No source spectrum matching meas_id='{meas_id}', die={die}.")
    if is_single := (target_region is not None):
        if target_region not in src_rows["region"].values:
            raise ValueError(f"No source spectrum matching meas_id='{meas_id}', die={die}, region='{target_region}'.")
        regions, show_res = [target_region], bool(cfg.get("show_residual", True))
        fig, axes = plt.subplots(2 if show_res else 1, 1, figsize=cfg.get("figsize", (8, 6 if show_res else 4.5)),
                                 sharex=show_res, gridspec_kw={"height_ratios": [3, 1]} if show_res else None)
        axes_to_use, ax_res = [axes[0] if show_res else axes], (axes[1] if show_res else None)
        if not show_res:
            axes = np.array([axes])
    else:
        if not (regions := list(src_rows["region"].unique())):
            raise ValueError(f"No regions found for source measurement '{meas_id}', die={die}.")
        fig, axes, axes_to_use = _setup_subplot_grid(len(regions), max_cols=3, figsize=cfg.get("figsize"))
        ax_res = None
    tgt_m = str(src_rows.iloc[0].get("measurement_id_target", ""))
    t_sub = meta_orig[(meta_orig["measurement_id"] == tgt_m) & (meta_orig["die"] == die)]
    p_sub = meta_pred[(meta_pred["source_measurement_id"] == meas_id) & (meta_pred["die"] == die)]
    plotted = 0
    for idx, reg in enumerate(regions):
        ax, reg_src = axes_to_use[idx], src_rows[src_rows["region"] == reg].iloc[0]
        tgt_m_df, pred_m_df = t_sub[t_sub["region"] == reg], p_sub[p_sub["region"] == reg]
        if tgt_m_df.empty or pred_m_df.empty:
            if is_single:
                raise ValueError(f"Target or predicted spectrum for {reg} not found.")
            ax.set_visible(False)
            continue
        tgt_r, pred_r = tgt_m_df.iloc[0], pred_m_df.iloc[0]
        i_s, i_t, i_p = int(reg_src["spectrum_index"]), int(tgt_r["spectrum_index"]), int(pred_r["spectrum_index"])
        src_t, tgt_t = str(reg_src.get("tool", "Tool")), str(reg_src.get("tool_target", "TargetTool"))
        for e, i, lbl, c, ls, lw in [
            (ary_e_orig[i_s], ary_i_orig[i_s], f"Source ({src_t})" if not is_single else f"Source Measured ({src_t})",
             "black", "-", 1.5),
            (ary_e_orig[i_t], ary_i_orig[i_t], f"Target ({tgt_t})" if not is_single else f"True Target ({tgt_t})",
             "red", "-", 1.8),
            (ary_e_pred[i_p], ary_i_pred[i_p], f"Transformed ({src_t}->{tgt_t})", "blue", ":", 2.0),
        ]:
            ax.plot(e, i, label=lbl, color=c, ls=ls, lw=lw)
        ax.legend(loc="best", fontsize="small" if is_single else "x-small")
        title_str = (f"Spectral Transfer Comparison | Region: {reg} | Die {die}\n"
                     f"Source: {meas_id} ({src_t}) -> Target: {tgt_m} ({tgt_t})"
                     if is_single else f"Region: {reg}")
        _finalize_plot(ax, title=title_str, xlabel="Binding Energy (eV)" if (not is_single or ax_res is None) else None,
                       ylabel="Intensity (counts)", invert_x=cfg.get("invert_x", True), grid_alpha=0.5)
        if is_single and ax_res is not None:
            ax_res.plot(ary_e_orig[i_t], ary_i_orig[i_t] - ary_i_pred[i_p], color="purple", lw=1.4,
                        label="Target - Transformed")
            ax_res.axhline(0, color="black", ls=":", lw=1.0, alpha=0.7)
            ax_res.legend(loc="best", fontsize="x-small")
            _finalize_plot(ax_res, xlabel="Binding Energy (eV)", ylabel="Residual",
                           invert_x=cfg.get("invert_x", True), grid_alpha=0.6)
        plotted += 1
    if plotted == 0:
        raise ValueError(f"No matching target or predicted spectra for source session '{meas_id}', die={die}.")
    if not is_single:
        fig.suptitle(cfg.get("title", f"Spectral Transfer Comparison | Session: {meas_id} (Die {die})"), fontsize=13)
    _finish_fig(fig, cfg)
    return fig, axes


def plot_sliding_window_slices(spectrum: np.ndarray, energy: np.ndarray | None = None,
                               plot_config: dict[str, Any] | None = None
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
            raise ValueError(f"Length mismatch: spectrum has {len(s1d)} points but energy has {len(e1d)} points.")
        w_size, s_step, x_vals, x_lbl = *calculate_window_points(e1d, config=cfg), e1d, "Binding Energy (eV)"
    else:
        e1d, x_vals, x_lbl = None, np.arange(len(s1d)), "Data Point Index"
        w_size, s_step = int(cfg.get("window_size", 15)), int(cfg.get("stride", max(1, 15 // 2)))
    windows, starts = extract_sliding_windows(s1d, window_size=w_size, stride=s_step)
    n_win = len(windows)
    has_anch = ((len(s1d) - w_size) % s_step != 0) and n_win > 1 and starts[-1] == len(s1d) - w_size
    fig, (ax_top, ax_bot) = _prepare_canvas(cfg.get("axes"), (10, 7), 2, 1, sharex=True,
                                            gridspec_kw={"height_ratios": [1, 1.2]})
    cmap = plt.get_cmap("tab10" if n_win <= 10 else "plasma")
    colors = [cmap(i % 10 if n_win <= 10 else (i / max(1, n_win - 1))) for i in range(n_win)]
    ax_top.plot(x_vals, s1d, color="#1a252f", lw=2.2, label="Original Spectrum", zorder=3)
    offset = (float(np.ptp(s1d) or 1.0) * 0.15) if cfg.get("offset_patches", False) else 0.0
    for i, (win, s_idx) in enumerate(zip(windows, starts)):
        x_w, is_a = x_vals[s_idx : s_idx + w_size], has_anch and (i == n_win - 1)
        ax_top.axvspan(min(x_w[0], x_w[-1]), max(x_w[0], x_w[-1]), facecolor=colors[i],
                       alpha=0.25 if is_a else 0.15, ls="--" if is_a else "-", edgecolor=colors[i] if is_a else "none")
        lbl = f"Patch {i+1} ({'anchored' if is_a else f'pts {s_idx}:{s_idx+w_size}'})" if (n_win <= 8 or is_a) else None
        ax_bot.plot(x_w, win + (i * offset), color=colors[i], lw=2.0 if is_a else 1.8,
                    ls="--" if is_a else "-", marker=".", markersize=3, alpha=0.9 if is_a else 0.85, label=lbl)
    reg, de_str = cfg.get("region"), (rf", $\Delta E$={abs(e1d[-1] - e1d[0]):.2f} eV" if e1d is not None else "")
    _finalize_plot(ax_top, title=f"Original Spectrum {f'({reg}) ' if reg else ''}[N={len(s1d)} pts{de_str}]",
                   ylabel="Intensity (counts / a.u.)", invert_x=False, grid_alpha=0.5)
    ax_top.legend(loc="upper right", framealpha=0.9)
    t_bot = (f"Extracted Sliding Windows (W={w_size} pts, S={s_step} pts, "
             f"Total Patches={n_win}{' [Right-Edge Anchored]' if has_anch else ''})")
    _finalize_plot(ax_bot, title=t_bot, xlabel=x_lbl,
                   ylabel="Patch Intensity" + (" (Waterfall Offset)" if cfg.get("offset_patches", False) else ""),
                   invert_x=False, grid_alpha=0.5)
    if n_win <= 8 or has_anch:
        ax_bot.legend(loc="upper right", fontsize=8, framealpha=0.9)
    if cfg.get("invert_x", energy is not None):
        for ax in (ax_top, ax_bot):
            if not ax.xaxis_inverted():
                ax.invert_xaxis()
    _finish_fig(fig, cfg, cfg.get("title", f"Sliding Window Spectral Decomposition {f'({reg})' if reg else ''}"))
    return fig, (ax_top, ax_bot)


def plot_atomic_percentage_distributions(df_at_samples: pd.DataFrame, plot_config: dict[str, Any] | None = None
                                         ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot atomic percentage distributions comparing Target and Predicted.

    Parameters
    ----------
    df_at_samples : pd.DataFrame
        DataFrame with 'split', 'element', 'target_at%', 'pred_at%', and
        'diff_at%'.
    plot_config : dict[str, Any] | None, optional
        Dictionary bundling plotting options:
        - 'splits' (list[str] | None): Splits to filter.
        - 'elements' (list[str] | None): Elements to filter.
        - 'title' (str | None): Custom supertitle for the figure.
        - 'figsize' (tuple[float, float]): Dimensions (default (14, 5.5)).
        - 'ax' (tuple[plt.Axes, plt.Axes] | None): Existing pair of Axes.
        - 'show' (bool): Whether to invoke plt.show() (default False).

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Tuple of (fig, (ax_absolute, ax_difference)).

    Raises
    ------
    KeyError
        If required columns are missing from df_at_samples.
    ValueError
        If df_at_samples is empty or filtered dataset has zero records.
    """
    _check_meta(df_at_samples, {"split", "element", "target_at%", "pred_at%", "diff_at%"}, "df_at_samples")
    cfg, df = plot_config or {}, df_at_samples.copy()
    if cfg.get("splits") is not None:
        df = df[df["split"].str.lower().isin([str(s).lower() for s in cfg["splits"]])]
    if cfg.get("elements") is not None:
        df = df[df["element"].isin(cfg["elements"])]
    if df.empty:
        raise ValueError("No records in df_at_samples match requested splits or elements filter.")
    fig, (ax_abs, ax_diff) = _prepare_canvas(cfg.get("ax"), cfg.get("figsize", (14, 5.5)), 1, 2)
    df_abs = pd.melt(df, id_vars=["split", "element"], value_vars=["target_at%", "pred_at%"],
                     var_name="spectrum_type", value_name="atomic_percent")
    df_abs["group"] = (df_abs["split"].str.capitalize() + " (" +
                       df_abs["spectrum_type"].map({"target_at%": "Target", "pred_at%": "Predicted"}) + ")")
    pal_abs = {"Train (Target)": "#1f77b4", "Train (Predicted)": "#aec7e8", "Test (Target)": "#2ca02c",
               "Test (Predicted)": "#98df8a", "Val (Target)": "#ff7f0e", "Val (Predicted)": "#ffbb78"}
    groups, splits = list(df_abs["group"].unique()), list(df["split"].unique())
    pal_diff = {"train": "#1f77b4", "test": "#2ca02c", "val": "#ff7f0e"}
    for ax, y_col, hue_col, grps, pal, t_str, y_lbl, is_diff in [
        (ax_abs, "atomic_percent", "group", groups, {g: pal_abs.get(g, "#7f7f7f") for g in groups},
         "Atomic % Distribution: Target vs. Predicted", "Atomic Percentage (%)", False),
        (ax_diff, "diff_at%", "split", splits, {s: pal_diff.get(str(s).lower(), "#7f7f7f") for s in splits},
         "Atomic % Difference Distribution (Pred - Target)", "Difference: Pred - Target (at. %)", True),
    ]:
        data_src = df if is_diff else df_abs
        sns.boxplot(data=data_src, x="element", y=y_col, hue=hue_col, hue_order=grps,
                    ax=ax, palette=pal, showfliers=False, boxprops=dict(alpha=0.75))
        sns.stripplot(data=data_src, x="element", y=y_col, hue=hue_col, hue_order=grps,
                      dodge=True, jitter=0.2, size=5 if is_diff else 4, alpha=0.65 if is_diff else 0.6,
                      ax=ax, palette=pal, legend=False)
        if is_diff:
            ax.axhline(0.0, color="crimson", ls="--", lw=1.2, alpha=0.85, zorder=2)
        h, l = ax.get_legend_handles_labels()
        lbls = [lab.capitalize() for lab in l[:len(grps)]] if is_diff else l[:len(grps)]
        ax.legend(h[:len(grps)], lbls, title="Split" if is_diff else "Split (Spectrum)", fontsize=8, loc="upper right")
        _finalize_plot(ax, title=t_str, xlabel="Element", ylabel=y_lbl, invert_x=False, grid_alpha=0.5)
    _finish_fig(fig, cfg)
    return fig, (ax_abs, ax_diff)


def plot_atomic_percentage_mae(df_at_data: pd.DataFrame, plot_config: dict[str, Any] | None = None
                               ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot atomic percentage Mean Absolute Error via Strip and ECDF plots.

    Parameters
    ----------
    df_at_data : pd.DataFrame
        DataFrame containing per-sample records with 'split', 'element',
        and 'abs_diff_at%' (or 'diff_at%').
    plot_config : dict[str, Any] | None, optional
        Dictionary bundling plotting options:
        - 'splits' (list[str] | None): Splits to display in Panel 1.
        - 'ecdf_splits' (list[str] | None): Splits for ECDF.
        - 'elements' (list[str] | None): Elements to display.
        - 'include_overall' (bool): Include 'Overall' category (default True).
        - 'reference_percentile' (float): Reference line (default 90.0).
        - 'palette' (dict[str, str] | None): Custom split color mapping.
        - 'element_palette' (dict[str, str] | None): Custom element palette.
        - 'show_values' (bool): Annotate numeric values (default True).
        - 'value_fontsize' (int): Font size for labels (default 8).
        - 'multiline_labels' (bool | None): Multiline text labels.
        - 'bar_half_width' (float): Width of horizontal MAE bar (default 0.09).
        - 'title' (str | None): Overall figure suptitle.
        - 'figsize' (tuple[float, float]): Dimensions (default (14, 5.5)).
        - 'ax' (tuple[plt.Axes, plt.Axes] | None): Existing pair of Axes.
        - 'show' (bool): Whether to invoke plt.show() (default False).

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Matplotlib Figure and the pair of Axes objects (ax_strip, ax_ecdf).

    Raises
    ------
    ValueError
        If df_at_data is empty or contains no records matching filter criteria.
    KeyError
        If required columns are missing.
    """
    _check_meta(df_at_data, {"split", "element"}, "df_at_data")
    df = df_at_data.copy()
    if "abs_diff_at%" not in df.columns:
        if "diff_at%" in df.columns:
            df["abs_diff_at%"] = df["diff_at%"].abs()
        else:
            raise KeyError("df_at_data must contain sample-level 'abs_diff_at%' (or 'diff_at%') column.")
    cfg = plot_config or {}
    if cfg.get("splits") is not None:
        df = df[df["split"].str.lower().isin([str(s).lower() for s in cfg["splits"]])]
    if cfg.get("elements") is not None:
        df = df[df["element"].isin(cfg["elements"])]
    if df.empty:
        raise ValueError("No records in df_at_data match the requested splits or elements filter.")
    inc_ovr, sh_val = bool(cfg.get("include_overall", True)), bool(cfg.get("show_values", True))
    bar_w, ref_pct = float(cfg.get("bar_half_width", 0.09)), float(cfg.get("reference_percentile", 90.0))
    fig, (ax_strip, ax_ecdf) = _prepare_canvas(cfg.get("ax"), cfg.get("figsize", (14, 5.5)), 1, 2)
    present_splits = ([s for s in cfg.get("splits", []) if s.lower() in df["split"].str.lower().unique()]
                      if cfg.get("splits") else list(df["split"].unique()))
    elements_list = sorted(df["element"].unique())
    split_pal = {"train": "#1f77b4", "test": "#2ca02c", "val": "#ff7f0e",
                 **{str(k).lower(): v for k, v in (cfg.get("palette") or {}).items()}}
    categories = list(elements_list) + (["Overall"] if inc_ovr else [])
    elem_pal = {e: (cfg.get("element_palette") or {}).get(e, plt.cm.tab10(i % 10))
                for i, e in enumerate(elements_list)}
    n_splits, dodge = len(present_splits), (0.32 if len(present_splits) > 1 else 0.0)
    rng, max_y = np.random.default_rng(42), 0.0
    multiline = (len(present_splits) > 1) if cfg.get("multiline_labels") is None else bool(cfg.get("multiline_labels"))
    for i, cat in enumerate(categories):
        c_df = df if cat == "Overall" else df[df["element"] == cat]
        for j, split in enumerate(present_splits):
            vals = c_df[c_df["split"].str.lower() == str(split).lower()]["abs_diff_at%"].values
            if len(vals) == 0:
                continue
            x_c, col = i + (j - (n_splits - 1) / 2.0) * dodge, split_pal.get(str(split).lower(), "#7f7f7f")
            m_v, s_v = float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0) or 0.0
            top_pt = max(float(np.max(vals)), m_v + s_v)
            max_y = max(max_y, top_pt)
            ax_strip.scatter(x_c + rng.uniform(-0.06, 0.06, len(vals)), vals, color=col, alpha=0.55, s=24, zorder=3)
            ax_strip.errorbar(x_c, m_v, yerr=[[min(m_v, s_v)], [s_v]], fmt="none", ecolor="black",
                              elinewidth=1.6, capsize=4.0, zorder=4)
            ax_strip.plot([x_c - bar_w, x_c + bar_w], [m_v, m_v], color=col, lw=3.2, zorder=5)
            if sh_val:
                lbl = f"{m_v:.2f}%\n±{s_v:.2f}%" if multiline else f"{m_v:.2f}% ± {s_v:.2f}%"
                ax_strip.text(x_c, top_pt + 0.06, lbl, ha="center", va="bottom",
                              fontsize=int(cfg.get("value_fontsize", 8)), fontweight="bold", zorder=6)
    if inc_ovr and len(categories) > 1:
        ax_strip.axvline(len(elements_list) - 0.5, color="gray", ls="--", alpha=0.6, lw=1.2)
    ax_strip.set_xticks(range(len(categories)), labels=categories, fontweight="bold")
    ax_strip.set_ylim(0.0, max_y * 1.25 + 0.20)
    sh_h = [plt.Line2D([0], [0], color=split_pal.get(str(s).lower(), "#7f7f7f"), lw=3.0,
                       label=f"{str(s).capitalize()} (Mean ± Std)") for s in present_splits]
    ax_strip.legend(handles=sh_h, title="Split", fontsize=8.5, loc="upper right")
    _finalize_plot(ax_strip, title="Sample Absolute Errors & Mean (MAE)", xlabel="Element",
                   ylabel="Absolute Error |Pred - Target| (at. %)", invert_x=False, grid_alpha=0.5)
    splits_lower = [str(s).lower() for s in df["split"].unique()]
    if cfg.get("ecdf_splits") is not None:
        if not (active_ecdf := [str(s).lower() for s in cfg["ecdf_splits"] if str(s).lower() in splits_lower]):
            raise ValueError(f"None of the requested ecdf_splits {cfg['ecdf_splits']} were found in the data.")
    else:
        active_ecdf = [s for s in ["train", "test"] if s in splits_lower] or list(dict.fromkeys(splits_lower))
    split_ls, max_x = {"train": "-", "test": "--", "val": ":"}, 0.0
    for elem in elements_list:
        for split in active_ecdf:
            sub = df[(df["element"] == elem) & (df["split"].str.lower() == split)]
            if (vals := np.sort(sub["abs_diff_at%"].values)).size > 0:
                max_x = max(max_x, float(vals[-1]))
                ax_ecdf.step(np.r_[0.0, vals], np.arange(len(vals) + 1) / len(vals) * 100.0, where="post",
                             color=elem_pal[elem], ls=split_ls.get(split, "-"), lw=1.7, alpha=0.9,
                             label=f"{elem} ({split.capitalize()})")
    ax_ecdf.axhline(ref_pct, color="black", linestyle=":", lw=1.3, alpha=0.75)
    ax_ecdf.text(0.02 * max(max_x, 1.0), ref_pct + 1.2, f"{int(ref_pct)}% Coverage Reference",
                 color="black", fontsize=8, alpha=0.85, fontweight="bold")
    ax_ecdf.set(ylim=(0, 103), xlim=(0.0, max_x * 1.05 + 0.05))
    ax_ecdf.legend(fontsize=8, loc="lower right", ncol=max(1, len(active_ecdf)))
    _finalize_plot(ax_ecdf, title="Empirical Cumulative Error (ECDF)", xlabel="Absolute Error |Pred - Target| (at. %)",
                   ylabel="Cumulative Percentage of Samples (%)", invert_x=False, grid_alpha=0.5)
    _finish_fig(fig, cfg)
    return fig, (ax_strip, ax_ecdf)


def _derive_tool_die_shades(base_color: Any) -> dict[str, str]:
    """Derive 3-tier colors (center, donut, edge) from base color."""
    if (norm := str(base_color).lower().strip()) in ("black", "#000000", "#000", "k"):
        return {"center": "#000000", "donut": "#555555", "edge": "#999999"}
    if norm in ("red", "#ff0000", "#f00", "tab:red", "#d62728"):
        return {"center": "#8b0000", "donut": "#d62728", "edge": "#f4a582"}
    try:
        h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
        if s < 0.08:
            return {"center": mcolors.to_hex((max(0.0, l * 0.4),) * 3), "donut": mcolors.to_hex((l,) * 3),
                    "edge": mcolors.to_hex((min(0.88, l + (1.0 - l) * 0.55),) * 3)}
        return {"center": mcolors.to_hex(colorsys.hls_to_rgb(h, max(0.12, l * 0.55), min(1.0, s * 1.15))),
                "donut": mcolors.to_hex(colorsys.hls_to_rgb(h, l, s)),
                "edge": mcolors.to_hex(colorsys.hls_to_rgb(h, min(0.85, l + (1.0 - l) * 0.55), max(0.2, s * 0.75)))}
    except Exception:
        return {"center": "#333333", "donut": "#777777", "edge": "#bbbbbb"}


def plot_flux_and_averaged_intensity_vs_time(ary_intensity: np.ndarray, ary_energy: np.ndarray,
                                             df_meta: pd.DataFrame, plot_config: dict[str, Any] | None = None, *,
                                             region: str | None = None
                                             ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot total flux and averaged intensity vs. measurement time.

    Upper panel shows total flux and lower panel shows averaged intensity.
    Displays all wafer dies on the timeline with 3-tier shading (dark for
    center, medium for donut, light for edge) per tool, setting the source
    tool base color to black and target tool to red by default.

    Parameters
    ----------
    ary_intensity : np.ndarray
        2D array of measured spectral intensities, shape (N_spectra, N_points).
    ary_energy : np.ndarray
        2D array of binding energies, shape (N_spectra, N_points). Mandatory
        for trapezoidal numerical integration.
    df_meta : pd.DataFrame
        Metadata DataFrame containing 'spectrum_index', 'measurement_id',
        'die', 'tool', and 'time'.
    plot_config : dict[str, Any] | None, optional
        Dictionary bundling options:
        - 'source_tool' (str | None): Source tool name (base color: black).
        - 'target_tool' (str | None): Target tool name (base color: red).
        - 'connect_lines' / 'show_lines' (bool): Connect chronological points
          per tool and die group (default False).
        - 'die_groups' (dict[str, list[int]] | None): Mapping of 'center',
          'donut', 'edge' to die indices (default: center: [0], donut: [5-8],
          edge: [1-4]).
        - 'tool_base_colors' (dict[str, str] | None): Custom base colors.
        - 'title' (str | None): Custom supertitle for the figure.
        - 'title_flux' (str | None): Custom title for upper panel.
        - 'title_intensity' (str | None): Custom title for lower panel.
        - 'figsize' (tuple[float, float]): Figure dimensions (default (10, 7)).
        - 'axes' (tuple[plt.Axes, plt.Axes] | None): Existing pair of Axes.
        - 'show' (bool): Whether to invoke plt.show() (default False).
    region : str | None, optional
        Target spectral region to filter. If None (default), aggregates across
        all measured regions for the complete measurement. Precedence over
        plot_config['region'].

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        The matplotlib Figure and tuple of (ax_flux, ax_intensity).

    Raises
    ------
    ValueError
        If ary_energy is None or shapes mismatch, or if df_meta is empty or
        filters match no data.
    KeyError
        If required metadata columns are missing.
    """
    if ary_energy is None or ary_intensity.ndim != 2 or ary_energy.ndim != 2 or ary_intensity.shape != ary_energy.shape:
        raise ValueError("ary_intensity and ary_energy must both be 2D arrays with identical matching shapes.")
    _check_meta(df_meta, {"spectrum_index", "measurement_id", "die", "tool", "time"})

    cfg = plot_config or {}
    t_reg, work_df = (region if region is not None else cfg.get("region")), df_meta.copy()
    if t_reg is not None:
        if "region" not in df_meta.columns:
            raise KeyError("df_meta must contain 'region' column when region is specified.")
        if (work_df := work_df[work_df["region"] == t_reg]).empty:
            raise ValueError(f"No spectra match region='{t_reg}' in df_meta.")

    die_groups = cfg.get("die_groups", {"center": [0], "donut": [5, 6, 7, 8], "edge": [1, 2, 3, 4]})
    die_to_group = {k: g for g, d_list in die_groups.items() for d in d_list
                    for k in (d, int(d) if str(d).isdigit() else d, str(d))}

    trapz_fn, idx = getattr(np, "trapezoid", getattr(np, "trapz", None)), work_df["spectrum_index"].astype(int).values
    work_df["time"] = pd.to_datetime(work_df["time"])
    work_df["total_flux"] = np.abs(trapz_fn(ary_intensity[idx], x=ary_energy[idx], axis=1))
    work_df["_int_sum"] = np.sum(ary_intensity[idx], axis=1)

    agg = work_df.groupby(["measurement_id", "die", "tool", "time"], as_index=False).agg(
        {"total_flux": "sum", "_int_sum": "sum", "spectrum_index": "count"}
    )
    agg["avg_intensity"] = agg["_int_sum"] / (agg["spectrum_index"] * ary_intensity.shape[1])
    agg["die_group"] = agg["die"].map(lambda d: die_to_group.get(d, "donut"))

    u_tools = list(pd.Series(agg["tool"].dropna().unique()))
    p_s = df_meta.dropna(subset=["tool_target"]) if "tool_target" in df_meta else pd.DataFrame()
    s_tool = cfg.get("source_tool") or (p_s["tool"].iloc[0] if not p_s.empty else (u_tools[0] if u_tools else None))
    t_tool = (cfg.get("target_tool") or
              (p_s["tool_target"].iloc[0] if not p_s.empty else (u_tools[1] if len(u_tools) > 1 else None)))
    pal = {**dict(zip(u_tools, ["tab:blue", "tab:green", "tab:purple", "tab:orange"])),
           **(cfg.get("tool_base_colors") or cfg.get("palette") or {})}
    pal.update({t: c for t, c in [(s_tool, "black"), (t_tool, "red")]
                if t and t not in (cfg.get("tool_base_colors") or {})})
    tool_shades = {t: _derive_tool_die_shades(pal[t]) for t in u_tools}

    fig, (ax_flux, ax_int) = _prepare_canvas(cfg.get("axes"), cfg.get("figsize", (10, 7)), 2, 1, sharex=True)
    panels, conn = [(ax_flux, "total_flux"), (ax_int, "avg_intensity")], bool(cfg.get("connect_lines",
                                                                                       cfg.get("show_lines", False)))
    styles = {"center": (50.0, 0.95, 3, "-", 1.4), "donut": (38.0, 0.85, 2, "--", 1.0),
              "edge": (32.0, 0.75, 2, ":", 1.0)}

    for t in u_tools:
        t_df, t_sh = agg[agg["tool"] == t], tool_shades[t]
        for g in ("edge", "donut", "center"):
            if not (sub := t_df[t_df["die_group"] == g]).empty:
                s, a, zo, _, _ = styles[g]
                for ax, col in panels:
                    ax.scatter(sub["time"], sub[col], color=t_sh[g], s=s, alpha=a, zorder=zo)
        if conn:
            for _, d_sub in t_df.groupby("die"):
                if len(d_sub := d_sub.sort_values("time")) >= 2:
                    g = d_sub["die_group"].iloc[0]
                    _, _, _, ls, lw = styles.get(g, (0, 0, 0, "-", 1.0))
                    for ax, col in panels:
                        ax.plot(d_sub["time"], d_sub[col], color=t_sh.get(g, "#777777"), ls=ls, lw=lw,
                                alpha=0.65, zorder=1)

    labels = {"center": "Center (Die 0)", "donut": "Donut (Dies 5-8)", "edge": "Edge (Dies 1-4)"}
    hnds = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=tool_shades[t][g],
                       markeredgecolor=tool_shades[t][g], markersize=7 if g == "center" else 5.5,
                       label=f"{t} - {labels.get(g, g.capitalize())}")
            for t in u_tools for g in ("center", "donut", "edge")]
    r_tag = f" [{t_reg}]" if t_reg is not None else " [All Regions]"
    for ax, lbl, k in [(ax_flux, "Total Flux", "title_flux"), (ax_int, "Averaged Intensity", "title_intensity")]:
        _finalize_plot(ax, title=cfg.get(k, f"{lbl} vs. Time{r_tag}"),
                       ylabel=f"{lbl}{r_tag} ({'counts · eV' if 'Flux' in lbl else 'counts'})",
                       invert_x=False, grid_alpha=0.5)

    ax_flux.legend(handles=hnds, bbox_to_anchor=(1.02, 1), loc="upper left",
                   borderaxespad=0.0, fontsize=8.5, framealpha=0.9)
    ax_int.set_xlabel("Measurement Time")
    plt.setp(ax_int.get_xticklabels(), rotation=25, ha="right")
    _finish_fig(fig, cfg, cfg.get("title", f"Measurement Total Flux and Averaged Intensity Timeline{r_tag}"))
    return fig, (ax_flux, ax_int)

