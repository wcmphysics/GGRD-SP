"""Quantification visualization for Shirley backgrounds
and atomic percentages."""

from __future__ import annotations

from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utility.shirley import calculate_shirley_background
from visualization.common import (
    _check_meta,
    _finalize_plot,
    _finish_fig,
    _prepare_canvas,
    _setup_subplot_grid,
)

__all__ = [
    "plot_shirley_background",
    "_filter_atomic_df",
    "plot_atomic_percentage_distributions",
    "plot_atomic_percentage_mae",
]


def plot_shirley_background(
    ary_energy: np.ndarray,
    ary_intensity: np.ndarray,
    df_meta: pd.DataFrame,
    plot_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, Any]:
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
    die = int(cfg.get("die", 0))
    target_region = cfg.get("region")
    subset = df_meta[df_meta["die"] == die]
    meas_id = cfg.get("measurement_id") or (
        str(subset["measurement_id"].iloc[0]) if not subset.empty else None
    )
    if meas_id is not None:
        subset = subset[subset["measurement_id"] == meas_id]
    if target_region is not None:
        subset = subset[subset["region"] == target_region]
    if subset.empty:
        raise ValueError(
            f"No matching spectra for meas_id='{meas_id}', "
            f"die={die}, region='{target_region}'"
        )
    is_single = target_region is not None or len(subset) == 1
    regions = (
        [str(subset["region"].iloc[0])]
        if is_single
        else list(subset["region"].unique())
    )
    if is_single:
        fig, ax = _prepare_canvas(cfg.get("ax"), (7, 4.5))
        axes, axes_flat = ax, [ax]
    else:
        fig, axes, axes_flat = _setup_subplot_grid(
            len(regions), max_cols=3, col_w=5.0, row_h=3.8
        )
    for idx, reg in enumerate(regions):
        ax = axes_flat[idx]
        spec_idx = int(
            subset[subset["region"] == reg].iloc[0]["spectrum_index"]
        )
        e_arr, i_arr = ary_energy[spec_idx], ary_intensity[spec_idx]
        b_arr, net_area = calculate_shirley_background(
            e_arr, i_arr, {**cfg, "region": reg}
        )
        ax.plot(
            e_arr,
            i_arr,
            label="Raw Spectrum" if is_single else "Raw",
            color="navy",
            lw=1.6,
        )
        ax.plot(
            e_arr,
            b_arr,
            label="Shirley Background" if is_single else "Shirley",
            color="crimson",
            ls="--",
            lw=1.5,
        )
        ax.fill_between(
            e_arr,
            b_arr,
            i_arr,
            where=(i_arr > b_arr),
            color="skyblue",
            alpha=0.35,
            label=f"{'Net Area' if is_single else 'Area'}: {net_area:.1f}",
        )
        ax.legend(loc="best", fontsize="small" if is_single else "x-small")
        _finalize_plot(
            ax,
            title=f"{meas_id} | Region: {reg} (Die {die})"
            if is_single
            else f"Region: {reg}",
            xlabel="Binding Energy (eV)",
            ylabel="Intensity (counts / a.u.)" if is_single else "Intensity",
            invert_x=cfg.get("invert_x", True),
            grid_alpha=0.5,
        )
    if not is_single:
        fig.suptitle(
            f"Shirley Background Subtraction | {meas_id} (Die {die})",
            fontsize=13,
        )
    _finish_fig(fig, cfg)
    return fig, axes


def _filter_atomic_df(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    name: str,
) -> pd.DataFrame:
    """Filter atomic percentage DataFrame by configured splits and elements."""
    if cfg.get("splits") is not None:
        valid_splits = [str(s).lower() for s in cfg["splits"]]
        df = df[df["split"].str.lower().isin(valid_splits)]
    if cfg.get("elements") is not None:
        df = df[df["element"].isin(cfg["elements"])]
    if df.empty:
        raise ValueError(
            f"No records in {name} match requested splits or elements filter."
        )
    return df


def plot_atomic_percentage_distributions(
    df_at_samples: pd.DataFrame,
    plot_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot atomic percentage distributions comparing Target and Predicted.

    Parameters
    ----------
    df_at_samples : pd.DataFrame
        DataFrame with 'split', 'element', 'target_atomic_percentage',
        'predicted_atomic_percentage', and 'diff_atomic_percentage'.
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
    _check_meta(
        df_at_samples,
        {
            "split",
            "element",
            "target_atomic_percentage",
            "predicted_atomic_percentage",
            "diff_atomic_percentage",
        },
        "df_at_samples",
    )
    cfg = plot_config or {}
    df = _filter_atomic_df(df_at_samples.copy(), cfg, "df_at_samples")
    fig, (ax_abs, ax_diff) = _prepare_canvas(
        cfg.get("ax"), cfg.get("figsize", (14, 5.5)), 1, 2
    )
    df_abs = pd.melt(
        df,
        id_vars=["split", "element"],
        value_vars=[
            "target_atomic_percentage",
            "predicted_atomic_percentage",
        ],
        var_name="spectrum_type",
        value_name="atomic_percent",
    )
    df_abs["group"] = (
        df_abs["split"].str.capitalize()
        + " ("
        + df_abs["spectrum_type"].map(
            {
                "target_atomic_percentage": "Target",
                "predicted_atomic_percentage": "Predicted",
            }
        )
        + ")"
    )
    pal_abs = {
        "Train (Target)": "#1f77b4",
        "Train (Predicted)": "#aec7e8",
        "Test (Target)": "#2ca02c",
        "Test (Predicted)": "#98df8a",
        "Val (Target)": "#ff7f0e",
        "Val (Predicted)": "#ffbb78",
    }
    groups = list(df_abs["group"].unique())
    splits = list(df["split"].unique())
    pal_diff = {"train": "#1f77b4", "test": "#2ca02c", "val": "#ff7f0e"}
    for ax, y_col, hue_col, grps, pal, t_str, y_lbl, is_diff in [
        (
            ax_abs,
            "atomic_percent",
            "group",
            groups,
            {g: pal_abs.get(g, "#7f7f7f") for g in groups},
            "Atomic % Distribution: Target vs. Predicted",
            "Atomic Percentage (%)",
            False,
        ),
        (
            ax_diff,
            "diff_atomic_percentage",
            "split",
            splits,
            {s: pal_diff.get(str(s).lower(), "#7f7f7f") for s in splits},
            "Atomic % Difference Distribution (Pred - Target)",
            "Difference: Pred - Target (at. %)",
            True,
        ),
    ]:
        data_src = df if is_diff else df_abs
        sns.boxplot(
            data=data_src,
            x="element",
            y=y_col,
            hue=hue_col,
            hue_order=grps,
            ax=ax,
            palette=pal,
            showfliers=False,
            boxprops=dict(alpha=0.75),
        )
        sns.stripplot(
            data=data_src,
            x="element",
            y=y_col,
            hue=hue_col,
            hue_order=grps,
            dodge=True,
            jitter=0.2,
            size=5 if is_diff else 4,
            alpha=0.65 if is_diff else 0.6,
            ax=ax,
            palette=pal,
            legend=False,
        )
        if is_diff:
            ax.axhline(
                0.0, color="crimson", ls="--", lw=1.2, alpha=0.85, zorder=2
            )
        h, l = ax.get_legend_handles_labels()
        lbls = (
            [lab.capitalize() for lab in l[: len(grps)]]
            if is_diff
            else l[: len(grps)]
        )
        ax.legend(
            h[: len(grps)],
            lbls,
            title="Split" if is_diff else "Split (Spectrum)",
            fontsize=8,
            loc="upper right",
        )
        _finalize_plot(
            ax,
            title=t_str,
            xlabel="Element",
            ylabel=y_lbl,
            invert_x=False,
            grid_alpha=0.5,
        )
    _finish_fig(fig, cfg)
    return fig, (ax_abs, ax_diff)


def _render_strip_category_split(
    ax: plt.Axes,
    vals: np.ndarray,
    x_c: float,
    col: str,
    bar_w: float,
    sh_val: bool,
    multiline: bool,
    font_size: int,
    rng: np.random.Generator,
) -> float:
    """Plot scatter points, error bar, and MAE bar for a category-split group."""
    m_v = float(np.mean(vals))
    s_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    top_pt = max(float(np.max(vals)), m_v + s_v)

    # Plot sample scatter points with horizontal jitter
    ax.scatter(
        x_c + rng.uniform(-0.06, 0.06, len(vals)),
        vals,
        color=col,
        alpha=0.55,
        s=24,
        zorder=3,
    )

    # Plot mean bar and uncertainty error bar
    ax.errorbar(
        x_c,
        m_v,
        yerr=[[min(m_v, s_v)], [s_v]],
        fmt="none",
        ecolor="black",
        elinewidth=1.6,
        capsize=4.0,
        zorder=4,
    )
    ax.plot([x_c - bar_w, x_c + bar_w], [m_v, m_v], color=col, lw=3.2, zorder=5)

    # Annotate numeric mean and std value on top of category
    if sh_val:
        lbl = (
            f"{m_v:.2f}%\n±{s_v:.2f}%"
            if multiline
            else f"{m_v:.2f}% ± {s_v:.2f}%"
        )
        ax.text(
            x_c,
            top_pt + 0.06,
            lbl,
            ha="center",
            va="bottom",
            fontsize=font_size,
            fontweight="bold",
            zorder=6,
        )
    return top_pt


def _render_atomic_percentage_strip_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    elements_list: list[str],
    present_splits: list[str],
    split_pal: dict[str, str],
    cfg: dict[str, Any],
) -> None:
    """Render strip plot of sample errors and MAE summary bars on ax."""
    inc_ovr = bool(cfg.get("include_overall", True))
    sh_val = bool(cfg.get("show_values", True))
    bar_w = float(cfg.get("bar_half_width", 0.09))
    font_size = int(cfg.get("value_fontsize", 8))
    categories = list(elements_list) + (["Overall"] if inc_ovr else [])
    n_splits = len(present_splits)
    dodge = 0.32 if len(present_splits) > 1 else 0.0
    rng = np.random.default_rng(42)
    max_y = 0.0
    multiline = (
        (len(present_splits) > 1)
        if cfg.get("multiline_labels") is None
        else bool(cfg.get("multiline_labels"))
    )

    # Plot sample points and summary bars across categories and splits
    for i, cat in enumerate(categories):
        c_df = df if cat == "Overall" else df[df["element"] == cat]
        for j, split in enumerate(present_splits):
            vals = c_df[
                c_df["split"].str.lower() == str(split).lower()
            ]["abs_diff_atomic_percentage"].values
            if len(vals) == 0:
                continue
            x_c = i + (j - (n_splits - 1) / 2.0) * dodge
            col = split_pal.get(str(split).lower(), "#7f7f7f")
            top_pt = _render_strip_category_split(
                ax, vals, x_c, col, bar_w, sh_val, multiline, font_size, rng
            )
            max_y = max(max_y, top_pt)

    # Add visual separation and formatting for Overall category
    if inc_ovr and len(categories) > 1:
        ax.axvline(
            len(elements_list) - 0.5,
            color="gray",
            ls="--",
            alpha=0.6,
            lw=1.2,
        )
    ax.set_xticks(range(len(categories)), labels=categories, fontweight="bold")
    ax.set_ylim(0.0, max_y * 1.25 + 0.20)
    sh_h = [
        plt.Line2D(
            [0],
            [0],
            color=split_pal.get(str(s).lower(), "#7f7f7f"),
            lw=3.0,
            label=f"{str(s).capitalize()} (Mean ± Std)",
        )
        for s in present_splits
    ]
    ax.legend(handles=sh_h, title="Split", fontsize=8.5, loc="upper right")
    _finalize_plot(
        ax,
        title="Sample Absolute Errors & Mean (MAE)",
        xlabel="Element",
        ylabel="Absolute Error |Pred - Target| (at. %)",
        invert_x=False,
        grid_alpha=0.5,
    )


def _render_atomic_percentage_ecdf_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    elements_list: list[str],
    elem_pal: dict[str, Any],
    cfg: dict[str, Any],
) -> None:
    """Render empirical cumulative distribution (ECDF) panel on ax."""
    # Determine active splits for ECDF curves
    splits_lower = [str(s).lower() for s in df["split"].unique()]
    if cfg.get("ecdf_splits") is not None:
        active_ecdf = [
            str(s).lower()
            for s in cfg["ecdf_splits"]
            if str(s).lower() in splits_lower
        ]
        if not active_ecdf:
            raise ValueError(
                f"None of the requested ecdf_splits {cfg['ecdf_splits']} were found in the data."
            )
    else:
        active_ecdf = [
            s for s in ["train", "test"] if s in splits_lower
        ] or list(dict.fromkeys(splits_lower))

    # Plot step functions for each element and split
    split_ls = {"train": "-", "test": "--", "val": ":"}
    max_x = 0.0
    for elem in elements_list:
        for split in active_ecdf:
            sub = df[
                (df["element"] == elem)
                & (df["split"].str.lower() == split)
            ]
            vals = np.sort(sub["abs_diff_atomic_percentage"].values)
            if vals.size > 0:
                max_x = max(max_x, float(vals[-1]))
                ax.step(
                    np.r_[0.0, vals],
                    np.arange(len(vals) + 1) / len(vals) * 100.0,
                    where="post",
                    color=elem_pal[elem],
                    ls=split_ls.get(split, "-"),
                    lw=1.7,
                    alpha=0.9,
                    label=f"{elem} ({split.capitalize()})",
                )

    # Add reference coverage threshold and finalize panel
    ref_pct = float(cfg.get("reference_percentile", 90.0))
    ax.axhline(ref_pct, color="black", linestyle=":", lw=1.3, alpha=0.75)
    ax.text(
        0.02 * max(max_x, 1.0),
        ref_pct + 1.2,
        f"{int(ref_pct)}% Coverage Reference",
        color="black",
        fontsize=8,
        alpha=0.85,
        fontweight="bold",
    )
    ax.set(ylim=(0, 103), xlim=(0.0, max_x * 1.05 + 0.05))
    ax.legend(fontsize=8, loc="lower right", ncol=max(1, len(active_ecdf)))
    _finalize_plot(
        ax,
        title="Empirical Cumulative Error (ECDF)",
        xlabel="Absolute Error |Pred - Target| (at. %)",
        ylabel="Cumulative Percentage of Samples (%)",
        invert_x=False,
        grid_alpha=0.5,
    )


def _prepare_atomic_diff_dataframe(
    df_at_data: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Validate metadata and prepare absolute atomic percentage difference."""
    _check_meta(df_at_data, {"split", "element"}, "df_at_data")
    df = df_at_data.copy()
    if "abs_diff_atomic_percentage" not in df.columns:
        if "diff_atomic_percentage" not in df.columns:
            raise KeyError(
                "df_at_data must contain sample-level "
                "'abs_diff_atomic_percentage' (or 'diff_atomic_percentage') column."
            )
        df["abs_diff_atomic_percentage"] = df["diff_atomic_percentage"].abs()
    return _filter_atomic_df(df, cfg, "df_at_data")


def _resolve_atomic_splits(
    df: pd.DataFrame, cfg: dict[str, Any]
) -> list[str]:
    """Identify present splits ordered by configuration preference."""
    if cfg.get("splits"):
        return [
            s
            for s in cfg["splits"]
            if s.lower() in df["split"].str.lower().unique()
        ]
    return list(df["split"].unique())


def _resolve_atomic_palettes(
    cfg: dict[str, Any], elements: list[str]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Resolve split and element color palettes for atomic percentage plots."""
    default_splits = {"train": "#1f77b4", "test": "#2ca02c", "val": "#ff7f0e"}
    custom_splits = {
        str(k).lower(): v for k, v in (cfg.get("palette") or {}).items()
    }
    split_pal = {**default_splits, **custom_splits}
    elem_overrides = cfg.get("element_palette") or {}
    elem_pal = {
        e: elem_overrides.get(e, plt.cm.tab10(i % 10))
        for i, e in enumerate(elements)
    }
    return split_pal, elem_pal


def plot_atomic_percentage_mae(
    df_at_data: pd.DataFrame,
    plot_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot atomic percentage Mean Absolute Error via Strip and ECDF plots.

    Parameters
    ----------
    df_at_data : pd.DataFrame
        DataFrame containing per-sample records with 'split', 'element',
        and 'abs_diff_atomic_percentage' (or 'diff_atomic_percentage').
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
    cfg = plot_config or {}
    df = _prepare_atomic_diff_dataframe(df_at_data, cfg)
    fig, (ax_strip, ax_ecdf) = _prepare_canvas(
        cfg.get("ax"), cfg.get("figsize", (14, 5.5)), 1, 2
    )

    present_splits = _resolve_atomic_splits(df, cfg)
    elements_list = sorted(df["element"].unique())
    split_pal, elem_pal = _resolve_atomic_palettes(cfg, elements_list)

    _render_atomic_percentage_strip_panel(
        ax_strip, df, elements_list, present_splits, split_pal, cfg
    )
    _render_atomic_percentage_ecdf_panel(
        ax_ecdf, df, elements_list, elem_pal, cfg
    )

    _finish_fig(fig, cfg)
    return fig, (ax_strip, ax_ecdf)
