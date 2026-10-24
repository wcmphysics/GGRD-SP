"""Timeline plotting module for measurement sessions, intensity drift,
and GP flux."""

from __future__ import annotations

from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utility.flux_and_intensity import (
    compute_die_total_flux,
    integrate_spectral_flux,
)
from visualization.common import (
    _check_meta,
    _derive_tool_die_shades,
    _finalize_plot,
    _finish_fig,
    _prepare_canvas,
)

__all__ = [
    "_calculate_staggered_levels",
    "_render_intensity_time_series",
    "plot_pairing_timeline",
    "plot_max_intensity_vs_time",
    "plot_normalized_max_intensity_vs_time",
    "plot_flux_and_averaged_intensity_vs_time",
    "plot_gp_die_flux_timeline",
]


def _calculate_staggered_levels(
    times: list[pd.Timestamp],
    max_levels: int = 3,
    proximity_ratio: float = 0.08,
) -> list[int]:
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
    max_lvl = max(1, int(max_levels))
    t_span = (max(times) - min(times)).total_seconds()
    if max_lvl <= 1 or t_span <= 0:
        return [i % max_lvl for i in range(len(times))]
    thresh = t_span * proximity_ratio
    assigned = []
    last_t = {lvl: -1e18 for lvl in range(max_lvl)}
    for t in times:
        lvl = next(
            (
                l
                for l in range(max_lvl)
                if (t.timestamp() - last_t[l]) >= thresh
            ),
            min(range(max_lvl), key=lambda l: last_t[l]),
        )
        assigned.append(lvl)
        last_t[lvl] = t.timestamp()
    return assigned


def plot_pairing_timeline(
    df_meta: pd.DataFrame,
    plot_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot timeline of measurement sessions and connect 1-to-1 paired sessions.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata DataFrame with pairing columns ('target_tool',
        'target_measurement_id', 'target_time', 'time_diff_hours').
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
    src_tool = cfg.get("source_tool", "J4")
    tgt_tool = cfg.get("target_tool", "J5")
    for col in ("time", "target_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    src_df = (
        df[df["tool"] == src_tool]
        .drop_duplicates("measurement_id")
        .sort_values("time")
    )
    tgt_df = (
        df[df["tool"] == tgt_tool]
        .drop_duplicates("measurement_id")
        .sort_values("time")
    )
    if src_df.empty and tgt_df.empty:
        raise ValueError(
            f"No measurements found for tools '{src_tool}' or '{tgt_tool}'."
        )
    fig, ax = _prepare_canvas(cfg.get("ax"), figsize=(10, 4.5))
    all_times = pd.concat([src_df["time"], tgt_df["time"]]).dropna()
    if not all_times.empty:
        ax.hlines(
            [1.0, 0.0],
            all_times.min(),
            all_times.max(),
            colors="gray",
            linestyles=":",
            alpha=0.5,
        )
    s_tgt = (
        src_df.dropna(subset=["target_measurement_id", "target_time"])
        if {"target_measurement_id", "target_time"}.issubset(src_df.columns)
        else pd.DataFrame()
    )
    paired_src = set(s_tgt.get("measurement_id", []).astype(str))
    paired_tgt = set(s_tgt.get("target_measurement_id", []).astype(str))
    for _, r in s_tgt.iterrows():
        ax.plot(
            [r["time"], r["target_time"]],
            [1.0, 0.0],
            color="tab:purple",
            alpha=0.7,
            lw=1.5,
        )
        diff_h = r.get("time_diff_hours")
        ax.annotate(
            f"{diff_h:.1f}h" if pd.notna(diff_h) else "",
            (r["time"] + (r["target_time"] - r["time"]) / 2, 0.5),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
            color="darkmagenta",
        )
    for sub, y_pos, t_name, p_ids, col in [
        (src_df, 1.0, src_tool, paired_src, "tab:blue"),
        (tgt_df, 0.0, tgt_tool, paired_tgt, "tab:green"),
    ]:
        p = sub[sub["measurement_id"].isin(p_ids)]
        up = sub[~sub["measurement_id"].isin(p_ids)]
        if not p.empty:
            ax.scatter(
                p["time"],
                [y_pos] * len(p),
                color=col,
                s=70,
                zorder=3,
                label=f"{t_name} (Paired)",
            )
        if not up.empty:
            ax.scatter(
                up["time"],
                [y_pos] * len(up),
                facecolors="none",
                edgecolors=col,
                s=70,
                lw=1.8,
                zorder=3,
                label=f"{t_name} (Unpaired)",
            )
    if bool(cfg.get("annotate_measurements", True)):
        stagger = max(1, int(cfg.get("stagger_levels", 3)))
        lead = bool(cfg.get("leader_lines", True))
        sz = int(cfg.get("label_fontsize", 8))
        for s_df, y_pos, is_src, col in [
            (src_df, 1.0, True, "tab:blue"),
            (tgt_df, 0.0, False, "tab:green"),
        ]:
            if not s_df.empty:
                times = [pd.Timestamp(t) for t in s_df["time"]]
                staggered = _calculate_staggered_levels(
                    times, max_levels=stagger
                )
                for t, m_id, lvl in zip(
                    times, s_df["measurement_id"].astype(str), staggered
                ):
                    off_y = (
                        (10.0 + lvl * 14.0) if is_src else (-12.0 - lvl * 14.0)
                    )
                    arr = (
                        {
                            "arrowstyle": "-",
                            "color": "gray",
                            "alpha": 0.5,
                            "linewidth": 0.8,
                        }
                        if (lead and lvl > 0)
                        else None
                    )
                    ax.annotate(
                        m_id,
                        xy=(t, y_pos),
                        xytext=(0, off_y),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if is_src else "top",
                        fontsize=sz,
                        color=col,
                        arrowprops=arr,
                        clip_on=False,
                    )
        ax.set_ylim(-0.25 - (stagger - 1) * 0.10, 1.25 + (stagger - 1) * 0.10)
    else:
        ax.set_ylim(-0.3, 1.3)
    ax.set_yticks([0.0, 1.0], [f"Target ({tgt_tool})", f"Source ({src_tool})"])
    ax.legend(loc="upper right", fontsize="small", framealpha=0.9)
    d_title = (
        f"1-to-1 Measurement Pairing Timeline ({src_tool} -> {tgt_tool}) | "
        f"Paired: {len(paired_src)}/{len(src_df)}"
    )
    _finalize_plot(
        ax,
        cfg,
        fig=fig,
        title=cfg.get("title") or d_title,
        xlabel="Measurement Time",
        invert_x=False,
        grid_alpha=0.5,
    )
    return fig, ax


def _render_intensity_time_series(
    df_meta: pd.DataFrame,
    y_calc: Any,
    cfg: dict[str, Any],
    region: str | None,
    die: int | None,
    d_title: str,
    d_ylabel: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Resolve criteria, filter df_meta, compute y series, and render plot."""
    if df_meta.empty:
        raise ValueError("df_meta cannot be empty")
    filts = dict(cfg.get("filters") or {})
    r = region or cfg.get("region") or filts.get("region")
    t_reg = str(r) if r is not None else None
    d = (
        die
        if die is not None
        else (cfg.get("die") if "die" in cfg else filts.get("die"))
    )
    t_die = int(d) if d is not None and str(d).isdigit() else d
    comb = {
        **filts,
        **({"region": t_reg} if t_reg is not None else {}),
        **({"die": t_die} if t_die is not None else {}),
    }
    df = df_meta.copy()
    for k, v in comb.items():
        if k not in df.columns:
            raise KeyError(
                f"Filter key '{k}' not found in df_meta columns: {list(df.columns)}"
            )
        if k == "die" and str(v).isdigit():
            df = df[(df["die"] == v) | (df["die"] == str(v))]
        else:
            df = df[df[k] == v]
    if df.empty:
        raise ValueError(
            f"No spectra match filters (region={t_reg!r}, die={t_die!r}, filters={comb})."
        )
    df["time"] = pd.to_datetime(df["time"])
    df["_y"] = y_calc(df)
    hue = str(cfg.get("hue", "tool"))
    if hue == "die":
        df["_hue"] = df["die"].apply(
            lambda x: f"Die {int(x)}" if str(x).isdigit() else f"Die {x}"
        )
        p_hue = "_hue"
        h_order = [
            f"Die {int(x)}" if str(x).isdigit() else f"Die {x}"
            for x in sorted(df["die"].dropna().unique())
        ]
    else:
        p_hue = hue
        h_order = (
            sorted(df[hue].dropna().unique())
            if hue in ("tool", "region")
            else None
        )
    fig, ax = _prepare_canvas(
        cfg.get("ax"), figsize=tuple(cfg.get("figsize", (10, 5.5)))
    )
    sns.scatterplot(
        data=df,
        x="time",
        y="_y",
        hue=p_hue,
        hue_order=h_order,
        style=cfg.get("style"),
        palette=cfg.get("palette"),
        alpha=float(cfg.get("alpha", 0.75)),
        s=float(cfg.get("s", 45.0)),
        ax=ax,
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    tags = (
        ([f"Region: {t_reg}"] if t_reg else [])
        + ([f"Die: {t_die}"] if t_die is not None else [])
        + [
            f"{str(k).capitalize()}: {v}"
            for k, v in filts.items()
            if k not in ("region", "die")
        ]
    )
    tag_suffix = f" ({', '.join(tags)})" if tags else ""
    title = cfg.get("title") or f"{d_title}{tag_suffix}"
    _finalize_plot(
        ax,
        cfg,
        fig=fig,
        title=title,
        xlabel="Measurement Time",
        ylabel=cfg.get("ylabel") or d_ylabel,
        invert_x=False,
        grid_alpha=0.5,
    )
    return fig, ax


def plot_max_intensity_vs_time(
    ary_intensity: np.ndarray,
    df_meta: pd.DataFrame,
    plot_config: dict[str, Any] | None = None,
    *,
    region: str | None = None,
    die: int | None = None,
) -> tuple[plt.Figure, plt.Axes]:
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
    cfg = plot_config or {}
    hue = str(cfg.get("hue", "tool"))
    _check_meta(df_meta, {"time", "spectrum_index", hue})
    return _render_intensity_time_series(
        df_meta,
        lambda df: np.max(
            ary_intensity[df["spectrum_index"].astype(int).values], axis=1
        ),
        cfg,
        region,
        die,
        f"Maximum Intensity vs. Time (hue={hue})",
        "Maximum Intensity (counts)",
    )


def plot_normalized_max_intensity_vs_time(
    ary_intensity: np.ndarray,
    df_meta: pd.DataFrame,
    ary_energy: np.ndarray | None = None,
    plot_config: dict[str, Any] | None = None,
    *,
    region: str | None = None,
    die: int | None = None,
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
    cfg = plot_config or {}
    hue = str(cfg.get("hue", "tool"))
    norm_mode = str(cfg.get("normalization_mode", "die_total_flux")).lower()
    _check_meta(df_meta, {"time", "spectrum_index", hue})
    if norm_mode not in ("die_total_flux", "spectrum_area"):
        raise ValueError(
            f"Unsupported normalization_mode '{norm_mode}'. "
            "Must be 'die_total_flux' or 'spectrum_area'."
        )
    if norm_mode == "die_total_flux":
        _check_meta(df_meta, {"measurement_id", "die"})
        f_map = compute_die_total_flux(df_meta, ary_intensity, ary_energy)

        def y_calc(df: pd.DataFrame) -> list[float]:
            maxes = np.max(
                ary_intensity[df["spectrum_index"].astype(int).values], axis=1
            )
            return [
                float(m)
                / max(
                    float(
                        f_map.get(
                            (str(r["measurement_id"]), int(r["die"])), 1.0
                        )
                    ),
                    1e-4,
                )
                for m, (_, r) in zip(maxes, df.iterrows())
            ]

        d_lbl = "Max Intensity / Die Total Integrated Area (1/eV)"
    else:

        def y_calc(df: pd.DataFrame) -> np.ndarray:
            idx = df["spectrum_index"].astype(int).values
            areas = [
                max(
                    float(
                        integrate_spectral_flux(ary_intensity[i], ary_energy[i])
                    ),
                    1e-4,
                )
                if ary_energy is not None
                else max(float(np.sum(ary_intensity[i])), 1e-4)
                for i in idx
            ]
            return np.max(ary_intensity[idx], axis=1) / np.array(areas)

        d_lbl = "Max Intensity / Spectrum Integrated Area (1/eV)"

    return _render_intensity_time_series(
        df_meta,
        y_calc,
        cfg,
        region,
        die,
        f"Normalized Maximum Intensity vs. Time (hue={hue})",
        d_lbl,
    )


def plot_flux_and_averaged_intensity_vs_time(
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    df_meta: pd.DataFrame,
    plot_config: dict[str, Any] | None = None,
    *,
    region: str | None = None,
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
        - 'connect_lines' (bool): Connect chronological points
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
    if (
        ary_energy is None
        or ary_intensity.ndim != 2
        or ary_energy.ndim != 2
        or ary_intensity.shape != ary_energy.shape
    ):
        raise ValueError(
            "ary_intensity and ary_energy must both be 2D arrays with identical matching shapes."
        )
    _check_meta(
        df_meta, {"spectrum_index", "measurement_id", "die", "tool", "time"}
    )

    cfg = plot_config or {}
    t_reg = region if region is not None else cfg.get("region")
    work_df = df_meta.copy()
    if t_reg is not None:
        if "region" not in df_meta.columns:
            raise KeyError(
                "df_meta must contain 'region' column when region is specified."
            )
        work_df = work_df[work_df["region"] == t_reg]
        if work_df.empty:
            raise ValueError(f"No spectra match region='{t_reg}' in df_meta.")

    die_groups = cfg.get(
        "die_groups",
        {"center": [0], "donut": [5, 6, 7, 8], "edge": [1, 2, 3, 4]},
    )
    die_to_group = {
        k: g
        for g, d_list in die_groups.items()
        for d in d_list
        for k in (d, int(d) if str(d).isdigit() else d, str(d))
    }

    idx = work_df["spectrum_index"].astype(int).values
    work_df["time"] = pd.to_datetime(work_df["time"])
    work_df["total_flux"] = integrate_spectral_flux(
        ary_intensity[idx], ary_energy[idx], axis=-1
    )
    work_df["_int_sum"] = np.sum(ary_intensity[idx], axis=1)

    agg = work_df.groupby(
        ["measurement_id", "die", "tool", "time"], as_index=False
    ).agg(
        {"total_flux": "sum", "_int_sum": "sum", "spectrum_index": "count"}
    )
    agg["avg_intensity"] = agg["_int_sum"] / (
        agg["spectrum_index"] * ary_intensity.shape[1]
    )
    agg["die_group"] = agg["die"].map(lambda d: die_to_group.get(d, "donut"))

    u_tools = list(pd.Series(agg["tool"].dropna().unique()))
    p_s = (
        df_meta.dropna(subset=["target_tool"])
        if "target_tool" in df_meta
        else pd.DataFrame()
    )
    s_tool = cfg.get("source_tool") or (
        p_s["tool"].iloc[0]
        if not p_s.empty
        else (u_tools[0] if u_tools else None)
    )
    t_tool = cfg.get("target_tool") or (
        p_s["target_tool"].iloc[0]
        if not p_s.empty
        else (u_tools[1] if len(u_tools) > 1 else None)
    )
    pal = {
        **dict(
            zip(u_tools, ["tab:blue", "tab:green", "tab:purple", "tab:orange"])
        ),
        **(cfg.get("tool_base_colors") or {}),
    }
    pal.update(
        {
            t: c
            for t, c in [(s_tool, "black"), (t_tool, "red")]
            if t and t not in (cfg.get("tool_base_colors") or {})
        }
    )
    tool_shades = {t: _derive_tool_die_shades(pal[t]) for t in u_tools}

    fig, (ax_flux, ax_int) = _prepare_canvas(
        cfg.get("axes"), cfg.get("figsize", (10, 7)), 2, 1, sharex=True
    )
    panels = [(ax_flux, "total_flux"), (ax_int, "avg_intensity")]
    conn = bool(cfg.get("connect_lines", False))
    styles = {
        "center": (50.0, 0.95, 3, "-", 1.4),
        "donut": (38.0, 0.85, 2, "--", 1.0),
        "edge": (32.0, 0.75, 2, ":", 1.0),
    }

    for t in u_tools:
        t_df = agg[agg["tool"] == t]
        t_sh = tool_shades[t]
        for g in ("edge", "donut", "center"):
            sub = t_df[t_df["die_group"] == g]
            if not sub.empty:
                s, a, zo, _, _ = styles[g]
                for ax, col in panels:
                    ax.scatter(
                        sub["time"],
                        sub[col],
                        color=t_sh[g],
                        s=s,
                        alpha=a,
                        zorder=zo,
                    )
        if conn:
            for _, d_sub in t_df.groupby("die"):
                d_sub = d_sub.sort_values("time")
                if len(d_sub) >= 2:
                    g = d_sub["die_group"].iloc[0]
                    _, _, _, ls, lw = styles.get(g, (0, 0, 0, "-", 1.0))
                    for ax, col in panels:
                        ax.plot(
                            d_sub["time"],
                            d_sub[col],
                            color=t_sh.get(g, "#777777"),
                            ls=ls,
                            lw=lw,
                            alpha=0.65,
                            zorder=1,
                        )

    labels = {
        "center": "Center (Die 0)",
        "donut": "Donut (Dies 5-8)",
        "edge": "Edge (Dies 1-4)",
    }
    hnds = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=tool_shades[t][g],
            markeredgecolor=tool_shades[t][g],
            markersize=7 if g == "center" else 5.5,
            label=f"{t} - {labels.get(g, g.capitalize())}",
        )
        for t in u_tools
        for g in ("center", "donut", "edge")
    ]
    r_tag = f" [{t_reg}]" if t_reg is not None else " [All Regions]"
    for ax, lbl, k in [
        (ax_flux, "Total Flux", "title_flux"),
        (ax_int, "Averaged Intensity", "title_intensity"),
    ]:
        _finalize_plot(
            ax,
            title=cfg.get(k, f"{lbl} vs. Time{r_tag}"),
            ylabel=f"{lbl}{r_tag} ({'counts · eV' if 'Flux' in lbl else 'counts'})",
            invert_x=False,
            grid_alpha=0.5,
        )

    ax_flux.legend(
        handles=hnds,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8.5,
        framealpha=0.9,
    )
    ax_int.set_xlabel("Measurement Time")
    plt.setp(ax_int.get_xticklabels(), rotation=25, ha="right")
    _finish_fig(
        fig,
        cfg,
        cfg.get(
            "title",
            f"Measurement Total Flux and Averaged Intensity Timeline{r_tag}",
        ),
    )
    return fig, (ax_flux, ax_int)


def plot_gp_die_flux_timeline(
    gp_bundle: dict[str, Any],
    plot_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot two-panel GP die flux prediction timeline for a selected die.

    Displays source and target tool total flux timelines for the chosen die.
    Points are colored by split (train: blue, val: orange, test: bright green)
    and annotated with percentage prediction error relative to ground truth.

    Parameters
    ----------
    gp_bundle : dict[str, Any]
        Dictionary from models.run_gp_flux_pipeline containing 'models',
        'datasets', and 'config'.
    plot_config : dict[str, Any] | None, optional
        Plot configuration options including 'plot_die', 'figsize', 'show'.

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Matplotlib figure and tuple of two axes (ax_source, ax_target).

    Raises
    ------
    ValueError
        If gp_bundle is missing required keys or datasets are empty.
    """
    import torch
    from models.gp_flux import predict_gp_die_flux

    if "models" not in gp_bundle or "datasets" not in gp_bundle:
        raise ValueError("gp_bundle must contain 'models' and 'datasets' keys.")

    cfg = {**(gp_bundle.get("config", {})), **(plot_config or {})}
    plot_die = int(cfg.get("plot", {}).get("plot_die", cfg.get("plot_die", 0)))
    ref_date = cfg.get("reference_date", "2026-01-01")

    datasets = gp_bundle["datasets"]
    models_dict = gp_bundle["models"]
    tools = list(datasets.keys())
    if len(tools) < 2:
        raise ValueError(
            f"Expected at least 2 tools in gp_bundle, got {len(tools)}."
        )

    source_tool, target_tool = tools[0], tools[1]
    fig, (ax_s, ax_t) = plt.subplots(
        1, 2, figsize=cfg.get("figsize", (14, 5.5))
    )
    tool_axes = [(source_tool, ax_s), (target_tool, ax_t)]

    split_colors = {
        "train": "tab:blue",
        "val": "tab:orange",
        "test": "limegreen",
        **(cfg.get("split_colors") or {}),
    }

    for tool_name, ax in tool_axes:
        ds = datasets[tool_name]
        model, likelihood = models_dict[tool_name]
        die_df = ds["die_df"]
        scaler = ds["scaler"]
        target_col = ds["target_column"]

        d_sub = die_df[die_df["die_idx"] == plot_die].sort_values("time_days")
        if d_sub.empty:
            ax.set_title(f"{tool_name} (Die {plot_die} - No Data)")
            continue

        # Continuous prediction grid
        t_min = float(d_sub["time_days"].min())
        t_max = float(d_sub["time_days"].max())
        span = max(t_max - t_min, 1.0)
        grid_t = np.linspace(t_min - 0.05 * span, t_max + 0.05 * span, 200)
        grid_x = torch.tensor(
            np.column_stack([grid_t, np.full_like(grid_t, plot_die)]),
            dtype=torch.float32,
        )

        pred_mean, pred_lower, pred_upper = predict_gp_die_flux(
            model, likelihood, grid_x, scaler
        )

        ax.plot(
            grid_t,
            pred_mean,
            color="black",
            linestyle="-",
            linewidth=1.6,
            label="GP Mean Prediction",
            zorder=2,
        )
        ax.fill_between(
            grid_t,
            pred_lower,
            pred_upper,
            color="gray",
            alpha=0.25,
            label="95% Confidence (±2σ)",
            zorder=1,
        )

        # Predict discrete points for percentage error annotations
        pts_x = torch.tensor(
            d_sub[["time_days", "die_idx"]].to_numpy(dtype=np.float32)
        )
        pts_pred, _, _ = predict_gp_die_flux(model, likelihood, pts_x, scaler)

        # Plot scatter markers colored by split
        for split_name, color in split_colors.items():
            mask = (d_sub["split"] == split_name).to_numpy()
            if np.any(mask):
                sub_t = d_sub["time_days"].to_numpy()[mask]
                sub_y = d_sub[target_col].to_numpy()[mask]
                ax.scatter(
                    sub_t,
                    sub_y,
                    color=color,
                    s=55,
                    edgecolors="black",
                    linewidth=0.8,
                    label=f"{split_name.capitalize()} Data",
                    zorder=4,
                )

        # Annotate percentage error near each point
        t_arr = d_sub["time_days"].to_numpy()
        y_arr = d_sub[target_col].to_numpy()
        for t_val, y_val, y_pred in zip(t_arr, y_arr, pts_pred):
            pct_err = ((y_pred - y_val) / max(y_val, 1e-8)) * 100.0
            ax.annotate(
                f"{pct_err:+.1f}%",
                (t_val, y_val),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7.5,
                fontweight="bold",
                color="#222222",
                zorder=5,
            )

        ax.set_title(
            f"Tool {tool_name} (Die {plot_die})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel(
            f"Elapsed Time from {ref_date} (days)",
            fontsize=9.5,
        )
        ax.set_ylabel("Total Die Flux (counts · eV)", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=8.5, framealpha=0.9)

    fig_title = cfg.get(
        "title",
        f"Gaussian Process Die {plot_die} Total Flux Prediction Timeline",
    )
    _finish_fig(fig, cfg, fig_title)
    return fig, (ax_s, ax_t)
