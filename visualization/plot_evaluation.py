"""Model prediction evaluation, spectral transfer comparison,
and distribution plots."""

from __future__ import annotations

from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utility.evaluation import match_prediction_source_target_indices
from visualization.common import (
    _finalize_plot,
    _finish_fig,
    _setup_subplot_grid,
)

__all__ = [
    "plot_prediction_comparison",
    "plot_regional_spectrum_distribution",
]


def _resolve_target_session_id(
    meta_orig: pd.DataFrame,
    meta_pred: pd.DataFrame,
    session_splits: dict[str, list[str]] | None = None,
    requested_id: str | None = None,
) -> str:
    """Resolve the source measurement session ID for comparison plot."""
    if requested_id is not None:
        return str(requested_id)
    p_ids = set(
        meta_pred.get("source_measurement_id", []).dropna().astype(str)
    )
    cands = (
        [str(c) for c in (session_splits or {}).get("test", [])]
        or (
            list(
                meta_orig[
                    meta_orig["split"].astype(str).str.lower() == "test"
                ]["measurement_id"].astype(str)
            )
            if "split" in meta_orig
            else []
        )
        or (
            list(
                meta_orig.dropna(subset=["target_measurement_id"])[
                    "measurement_id"
                ].astype(str)
            )
            if "target_measurement_id" in meta_orig
            else []
        )
    )
    meas_id = next((c for c in (cands or list(p_ids)) if c in p_ids), None)
    if not meas_id:
        raise ValueError("No paired source measurements found in df_meta_orig.")
    return str(meas_id)


def _setup_comparison_canvas(
    cfg: dict[str, Any], regions: list[str], is_single: bool
) -> tuple[plt.Figure, Any, list[plt.Axes], plt.Axes | None]:
    """Initialize matplotlib figure and axes for comparison panels."""
    if is_single:
        show_res = bool(cfg.get("show_residual", True))
        fig, axes = plt.subplots(
            2 if show_res else 1,
            1,
            figsize=cfg.get("figsize", (8, 6 if show_res else 4.5)),
            sharex=show_res,
            gridspec_kw={"height_ratios": [3, 1]} if show_res else None,
        )
        axes_to_use = [axes[0] if show_res else axes]
        ax_res = axes[1] if show_res else None
        if not show_res:
            axes = np.array([axes])
        return fig, axes, axes_to_use, ax_res
    fig, axes, axes_to_use = _setup_subplot_grid(
        len(regions), max_cols=3, figsize=cfg.get("figsize")
    )
    return fig, axes, axes_to_use, None


def _render_comparison_curves(
    ax: plt.Axes,
    reg: str,
    triplet: tuple[int, int, int],
    data_orig: tuple[np.ndarray, np.ndarray],
    data_pred: tuple[np.ndarray, np.ndarray],
    tools: tuple[str, str],
    info: dict[str, Any],
    has_res: bool,
    conf: dict[str, Any],
) -> None:
    """Plot source, target, and transformed spectral curves on the main axes."""
    i_s, i_t, i_p = triplet
    ary_i_orig, ary_e_orig = data_orig
    ary_i_pred, ary_e_pred = data_pred
    src_t, tgt_t = tools
    is_single = bool(info.get("is_single", False))

    # Build spectral curves with distinct styles for source, target, and prediction
    src_label = (
        f"Source Measured ({src_t})" if is_single else f"Source ({src_t})"
    )
    tgt_label = (
        f"True Target ({tgt_t})" if is_single else f"Target ({tgt_t})"
    )
    curves = [
        (ary_e_orig[i_s], ary_i_orig[i_s], src_label, "black", "-", 1.5),
        (ary_e_orig[i_t], ary_i_orig[i_t], tgt_label, "red", "-", 1.8),
        (
            ary_e_pred[i_p],
            ary_i_pred[i_p],
            f"Transformed ({src_t}->{tgt_t})",
            "blue",
            ":",
            2.0,
        ),
    ]
    for e, i, lbl, c, ls, lw in curves:
        ax.plot(e, i, label=lbl, color=c, ls=ls, lw=lw)

    # Format panel legend, descriptive title, and axis labels
    ax.legend(loc="best", fontsize="small" if is_single else "x-small")
    meas_id = str(info.get("meas_id", ""))
    die = int(info.get("die", 0))
    tgt_m = str(info.get("tgt_m", ""))
    title_str = (
        f"Spectral Transfer Comparison | Region: {reg} | Die {die}\n"
        f"Source: {meas_id} ({src_t}) -> Target: {tgt_m} ({tgt_t})"
        if is_single
        else f"Region: {reg}"
    )
    _finalize_plot(
        ax,
        title=title_str,
        xlabel="Binding Energy (eV)"
        if (not is_single or not has_res)
        else None,
        ylabel="Intensity (counts)",
        invert_x=conf.get("invert_x", True),
        grid_alpha=0.5,
    )


def _render_comparison_residual(
    ax_res: plt.Axes,
    triplet: tuple[int, int, int],
    data_orig: tuple[np.ndarray, np.ndarray],
    data_pred: tuple[np.ndarray, np.ndarray],
    conf: dict[str, Any],
) -> None:
    """Plot difference residual curve between target and predicted spectrum."""
    _, i_t, i_p = triplet
    ary_i_orig, ary_e_orig = data_orig
    ary_i_pred, _ = data_pred
    ax_res.plot(
        ary_e_orig[i_t],
        ary_i_orig[i_t] - ary_i_pred[i_p],
        color="purple",
        lw=1.4,
        label="Target - Transformed",
    )
    ax_res.axhline(0, color="black", ls=":", lw=1.0, alpha=0.7)
    ax_res.legend(loc="best", fontsize="x-small")
    _finalize_plot(
        ax_res,
        xlabel="Binding Energy (eV)",
        ylabel="Residual",
        invert_x=conf.get("invert_x", True),
        grid_alpha=0.6,
    )


def _render_prediction_comparison_panel(
    ax: plt.Axes,
    reg: str,
    triplets: list[tuple[int, int, int]],
    data_orig: tuple[np.ndarray, np.ndarray],
    data_pred: tuple[np.ndarray, np.ndarray],
    orig_by_idx: dict[int, Any],
    info: dict[str, Any],
    ax_res: plt.Axes | None = None,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Render a single spectral region panel and optional residual."""
    is_single = bool(info.get("is_single", False))

    # Verify presence of matched spectrum triplets for this region
    if not triplets:
        if is_single:
            raise ValueError(
                f"Target or predicted spectrum for {reg} not found."
            )
        ax.set_visible(False)
        return False

    # Extract tool labels and render spectral curves and residual
    i_s = triplets[0][0]
    reg_src = orig_by_idx.get(i_s)
    src_t = str(getattr(reg_src, "tool", "Tool")) if reg_src else "Tool"
    tgt_t = (
        str(getattr(reg_src, "target_tool", "TargetTool"))
        if reg_src
        else "TargetTool"
    )

    conf = cfg or {}
    has_res = is_single and ax_res is not None
    _render_comparison_curves(
        ax,
        reg,
        triplets[0],
        data_orig,
        data_pred,
        (src_t, tgt_t),
        info,
        has_res,
        conf,
    )
    if has_res:
        _render_comparison_residual(
            ax_res, triplets[0], data_orig, data_pred, conf
        )
    return True


def _resolve_comparison_regions(
    src_rows: pd.DataFrame, target_region: str | None, meas_id: str, die: int
) -> tuple[list[str], bool]:
    """Resolve region list and single-panel flag for comparison plot."""
    is_single = target_region is not None
    if is_single:
        if target_region not in src_rows["region"].values:
            raise ValueError(
                f"No source spectrum matching meas_id='{meas_id}', "
                f"die={die}, region='{target_region}'."
            )
        return [target_region], True
    regions = list(src_rows["region"].unique())
    if not regions:
        raise ValueError(
            f"No regions found for source measurement '{meas_id}', die={die}."
        )
    return regions, False


def _resolve_comparison_session_data(
    meta_orig: pd.DataFrame,
    meta_pred: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[str, int, pd.DataFrame, str]:
    """Resolve session ID, die index, source rows, and target session ID."""
    die = int(cfg.get("die", 0))
    meas_id = _resolve_target_session_id(
        meta_orig,
        meta_pred,
        cfg.get("session_splits"),
        cfg.get("measurement_id"),
    )
    src_rows = meta_orig[
        (meta_orig["measurement_id"] == meas_id) & (meta_orig["die"] == die)
    ]
    if src_rows.empty:
        raise ValueError(
            f"No source spectrum matching meas_id='{meas_id}', die={die}."
        )
    tgt_m = str(src_rows.iloc[0].get("target_measurement_id", ""))
    return meas_id, die, src_rows, tgt_m


def _render_all_comparison_panels(
    axes_to_use: list[plt.Axes],
    regions: list[str],
    matched: dict[str, list[tuple[int, int, int]]],
    data_orig: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    data_pred: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    info: dict[str, Any],
    ax_res: plt.Axes | None,
    cfg: dict[str, Any],
) -> None:
    """Render all regional panels and assert at least one spectrum matched."""
    ary_i_orig, ary_e_orig, meta_orig = data_orig
    ary_i_pred, ary_e_pred, _ = data_pred
    orig_by_idx = {int(r.spectrum_index): r for r in meta_orig.itertuples()}
    plotted = sum(
        1
        for idx, reg in enumerate(regions)
        if _render_prediction_comparison_panel(
            axes_to_use[idx],
            reg,
            matched.get(reg, []),
            (ary_i_orig, ary_e_orig),
            (ary_i_pred, ary_e_pred),
            orig_by_idx,
            info,
            ax_res,
            cfg,
        )
    )
    if plotted == 0:
        meas_id, die = info["meas_id"], info["die"]
        raise ValueError(
            f"No matching target or predicted spectra for source session "
            f"'{meas_id}', die={die}."
        )


def _match_comparison_indices(
    meta_orig: pd.DataFrame,
    meta_pred: pd.DataFrame,
    meas_id: str,
    die: int,
    regions: list[str],
) -> dict[str, list[tuple[int, int, int]]]:
    """Find matching source, target, and prediction spectrum triplets."""
    pred_subset = meta_pred[
        (meta_pred["source_measurement_id"].astype(str) == str(meas_id))
        & (meta_pred["die"].astype(str) == str(die))
    ]
    return match_prediction_source_target_indices(
        meta_orig, pred_subset, regions_filter=regions, die_filter=[str(die)]
    )


def _finish_comparison_figure(
    fig: plt.Figure,
    is_single: bool,
    info: dict[str, Any],
    cfg: dict[str, Any],
) -> None:
    """Apply supertitle and finish figure configuration."""
    if not is_single:
        meas_id, die = info["meas_id"], info["die"]
        default_title = (
            f"Spectral Transfer Comparison | Session: {meas_id} (Die {die})"
        )
        fig.suptitle(cfg.get("title", default_title), fontsize=13)
    _finish_fig(fig, cfg)


def plot_prediction_comparison(
    data_original: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    data_predicted: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, Any]:
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
    cfg = config or {}
    meta_orig, meta_pred = data_original[2], data_predicted[2]
    meas_id, die, src_rows, tgt_m = _resolve_comparison_session_data(
        meta_orig, meta_pred, cfg
    )
    regions, is_single = _resolve_comparison_regions(
        src_rows, cfg.get("region"), meas_id, die
    )
    fig, axes, axes_to_use, ax_res = _setup_comparison_canvas(
        cfg, regions, is_single
    )

    matched = _match_comparison_indices(
        meta_orig, meta_pred, meas_id, die, regions
    )
    info = {
        "meas_id": meas_id, "die": die, "tgt_m": tgt_m, "is_single": is_single
    }
    _render_all_comparison_panels(
        axes_to_use, regions, matched, data_original, data_predicted, info, ax_res, cfg
    )
    _finish_comparison_figure(fig, is_single, info, cfg)
    return fig, axes


def _filter_distribution_metadata(
    meta_pred: pd.DataFrame,
    split: str,
    die_filter: int | None,
) -> pd.DataFrame:
    """Filter prediction metadata by dataset split and die index."""
    p_df = meta_pred.copy()
    if split and str(split).lower() != "all" and "split" in p_df.columns:
        p_df = p_df[
            p_df["split"].astype(str).str.lower() == str(split).lower()
        ]
    if die_filter is not None:
        p_df = p_df[p_df["die"].astype(str) == str(die_filter)]
    if p_df.empty:
        raise ValueError(
            f"No predicted spectra found for split='{split}'"
            + (f", die={die_filter}" if die_filter is not None else "")
            + "."
        )
    return p_df


def _resolve_distribution_regions(
    matched_by_region: dict[str, list[tuple[int, int, int]]],
    specified_regions: list[str] | None,
    split: str,
    die_filter: int | None,
) -> list[str]:
    """Determine ordered list of regions to plot in distribution rows."""
    if specified_regions is not None:
        regions_to_plot = [
            r for r in specified_regions if r in matched_by_region
        ]
    else:
        regions_to_plot = sorted(matched_by_region.keys())
    if not regions_to_plot:
        raise ValueError(
            f"No paired source, target, and predicted spectra found for split='{split}'"
            + (f", die={die_filter}" if die_filter is not None else "")
            + "."
        )
    return regions_to_plot


def _compute_regional_distribution_statistics(
    triplets: list[tuple[int, int, int]],
    ary_i_orig: np.ndarray,
    ary_e_orig: np.ndarray,
    ary_i_pred: np.ndarray,
) -> dict[str, Any]:
    """Compute normalized intensity statistics and absolute errors for a region."""
    # Extract raw spectra stacks for source, target, and prediction
    src_raw = np.array(
        [ary_i_orig[s] for s, _, _ in triplets], dtype=np.float64
    )
    tgt_raw = np.array(
        [ary_i_orig[t] for _, t, _ in triplets], dtype=np.float64
    )
    pred_raw = np.array(
        [ary_i_pred[p] for _, _, p in triplets], dtype=np.float64
    )

    # Normalize by target maximum intensity expressed as percentage
    tgt_max = np.max(tgt_raw, axis=1, keepdims=True)
    tgt_max = np.where(
        (tgt_max <= 0.0) | (~np.isfinite(tgt_max)), 1e-12, tgt_max
    )
    src_stack = (src_raw / tgt_max) * 100.0
    tgt_stack = (tgt_raw / tgt_max) * 100.0
    pred_stack = (pred_raw / tgt_max) * 100.0
    err_stack = np.abs(pred_stack - tgt_stack)

    # Calculate means, standard deviations, and defensive non-negative bounds
    src_mean, tgt_mean = np.mean(src_stack, axis=0), np.mean(tgt_stack, axis=0)
    pred_mean, err_mean = np.mean(pred_stack, axis=0), np.mean(err_stack, axis=0)
    src_std, tgt_std = np.std(src_stack, axis=0), np.std(tgt_stack, axis=0)
    pred_std, err_std = np.std(pred_stack, axis=0), np.std(err_stack, axis=0)

    src_low = np.maximum(src_mean - src_std, 0.0)
    tgt_low = np.maximum(tgt_mean - tgt_std, 0.0)
    pred_low = np.maximum(pred_mean - pred_std, 0.0)
    err_low = np.maximum(err_mean - err_std, 0.0)

    return {
        "energy_arr": ary_e_orig[triplets[0][1]],
        "n_samples": len(src_stack),
        "src_mean": src_mean,
        "src_low": src_low,
        "src_high": np.maximum(src_mean + src_std, src_low),
        "tgt_mean": tgt_mean,
        "tgt_low": tgt_low,
        "tgt_high": np.maximum(tgt_mean + tgt_std, tgt_low),
        "pred_mean": pred_mean,
        "pred_low": pred_low,
        "pred_high": np.maximum(pred_mean + pred_std, pred_low),
        "err_mean": err_mean,
        "err_low": err_low,
        "err_high": np.maximum(err_mean + err_std, err_low),
    }


def _render_regional_distribution_row(
    ax: plt.Axes,
    reg: str,
    stats: dict[str, Any],
    die_filter: int | None,
    error_color: str,
    invert_x: bool,
) -> None:
    """Render a single region's distribution curves and error profile."""
    energy_arr = stats["energy_arr"]

    # Plot mean and uncertainty ribbons for source, target, and predicted
    ax.plot(
        energy_arr, stats["src_mean"], color="black", ls="-", lw=1.6, label="Source"
    )
    ax.fill_between(
        energy_arr, stats["src_low"], stats["src_high"], color="black", alpha=0.15
    )
    ax.plot(
        energy_arr, stats["tgt_mean"], color="red", ls="-", lw=1.6, label="Target"
    )
    ax.fill_between(
        energy_arr, stats["tgt_low"], stats["tgt_high"], color="red", alpha=0.15
    )
    ax.plot(
        energy_arr, stats["pred_mean"], color="blue", ls="--", lw=1.8, label="Predicted"
    )
    ax.fill_between(
        energy_arr, stats["pred_low"], stats["pred_high"], color="blue", alpha=0.15
    )

    ax.set_xlabel("Binding Energy (eV)", fontsize=8)
    ax.set_ylabel("Intensity (% of Target Max)", color="black", fontsize=8)
    if invert_x and not ax.xaxis_inverted():
        ax.invert_xaxis()
    ax.grid(True, linestyle="--", alpha=0.5)

    # Plot secondary y-axis for absolute error profile
    ax_err = ax.twinx()
    ax_err.plot(
        energy_arr, stats["err_mean"], color=error_color, ls="-", lw=1.6, label="Abs Error"
    )
    ax_err.fill_between(
        energy_arr, stats["err_low"], stats["err_high"], color=error_color, alpha=0.20
    )
    ax_err.set_ylabel("Absolute Error (%)", color="black", fontsize=8)
    ax_err.tick_params(axis="y", labelcolor="black", labelsize=7.5)
    ax_err.set_ylim(bottom=0.0)
    ax.tick_params(axis="both", labelsize=7.5)

    die_desc = f"Die {die_filter}" if die_filter is not None else "All Dies"
    ax.set_title(
        f"Region: {reg} (N={stats['n_samples']}) | {die_desc}", fontsize=9
    )


def _finish_distribution_figure(
    fig: plt.Figure,
    n_rows: int,
    split: str,
    error_color: str,
    cfg: dict[str, Any],
) -> None:
    """Configure legend, super title, and subplot spacing for distribution plot."""
    # Build common legend handles for all panels
    legend_handles = [
        plt.Line2D(
            [], [], color="black", ls="-", lw=1.6, label="Source (Mean ± 1σ)"
        ),
        plt.Line2D(
            [], [], color="red", ls="-", lw=1.6, label="Target (Mean ± 1σ)"
        ),
        plt.Line2D(
            [],
            [],
            color="blue",
            ls="--",
            lw=1.8,
            label="Predicted (Mean ± 1σ)",
        ),
        plt.Line2D(
            [],
            [],
            color=error_color,
            ls="-",
            lw=1.6,
            label="Abs Error (Mean ± 1σ)",
        ),
    ]

    # Position super title and figure legend above subplots
    fig_h = float(fig.get_size_inches()[1])
    title_in, legend_in, gap_in = 0.30, 0.38, 0.25
    header_in = title_in + legend_in + gap_in
    split_title = f"Split: {split}" if split else "All Splits"
    fig_title = cfg.get(
        "title",
        f"Regional Spectrum & Absolute Error Distributions ({split_title})",
    )
    if not fig_title:
        header_in -= title_in
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0 - (header_in - gap_in - legend_in) / fig_h),
        ncol=4,
        fontsize=7.5,
        frameon=True,
    )
    if fig_title:
        fig.suptitle(
            fig_title, fontsize=10, fontweight="bold", y=1.0 - 0.05 / fig_h
        )

    # Adjust subplot grid geometry and save or display figure
    fig_w = float(fig.get_size_inches()[0])
    bottom_in, side_in, row_gap_in = 0.55, 0.85, 0.80
    top_frac = 1.0 - header_in / fig_h
    bottom_frac = bottom_in / fig_h
    axes_h_in = (
        fig_h - header_in - bottom_in - row_gap_in * (n_rows - 1)
    ) / n_rows
    fig.subplots_adjust(
        left=side_in / fig_w,
        right=1.0 - side_in / fig_w,
        top=top_frac,
        bottom=bottom_frac,
        hspace=row_gap_in / max(axes_h_in, 1e-6),
    )
    dpi = int(cfg.get("dpi", 300))
    if dpi <= 0:
        raise ValueError(f"dpi must be a positive integer, got {dpi}")
    if cfg.get("save_path"):
        fig.savefig(str(cfg["save_path"]), dpi=dpi)
    if cfg.get("show"):
        plt.show()


def _create_distribution_canvas(
    regions: list[str], cfg: dict[str, Any]
) -> tuple[plt.Figure, np.ndarray, int]:
    """Create figure canvas and axes array for distribution rows."""
    n_rows = len(regions)
    figsize = cfg.get(
        "figsize", (10.0, 1.5 + 2.4 * n_rows + 0.8 * (n_rows - 1))
    )
    dpi = int(cfg.get("dpi", 300))
    if dpi <= 0:
        raise ValueError(f"dpi must be a positive integer, got {dpi}")
    fig, axes = plt.subplots(
        n_rows, 1, figsize=figsize, dpi=dpi, squeeze=False
    )
    return fig, axes[:, 0], n_rows


def plot_regional_spectrum_distribution(
    data_original: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    data_predicted: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, Any]:
    """Plot distribution of Source, Target, Predicted, and Absolute Error
    spectra across spectral regions.

    Displays a single-column by N-row subplot layout, where each row corresponds
    to one spectral region. For each die within each measurement session,
    the regional spectra are normalized by the maximum intensity of the target
    regional spectrum (multiplied by 100), expressing intensity as a
    percentage of the target maximum. Mean, variation (standard deviation),
    and absolute error are calculated based on these normalized spectra.

    Parameters
    ----------
    data_original : tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Tuple of (ary_intensity_orig, ary_energy_orig, df_meta_orig).
    data_predicted : tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Tuple of (ary_intensity_pred, ary_energy_pred, df_meta_pred).
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'split' (str): Dataset split to visualize (default 'test').
        - 'die' (int | None): Specific die to filter by (default None).
        - 'regions' (list[str] | None): Specific regions to plot.
        - 'invert_x' (bool): Invert binding energy axis (default True).
        - 'error_color' (str): Color for absolute error (default '#EFC934').
        - 'figsize' (tuple[float, float] | None): Figure size.
        - 'dpi' (int): Figure dots per inch resolution (default 300).
        - 'save_path' (str | None): Save path for image file.
        - 'title' (str | None): Figure super title.
        - 'show' (bool): Whether to invoke plt.show() (default False).

    Returns
    -------
    tuple[plt.Figure, Any]
        The Matplotlib Figure and array of Axes (one per region).

    Raises
    ------
    ValueError
        If no paired samples match the split/die filters or spectra are missing.
    """
    (ary_i_orig, ary_e_orig, meta_orig), (ary_i_pred, _, meta_pred) = (
        data_original,
        data_predicted,
    )
    cfg = config or {}
    split = cfg.get("split", "test")
    die_filter = cfg.get("die")

    p_df = _filter_distribution_metadata(meta_pred, split, die_filter)
    matched_by_region = match_prediction_source_target_indices(
        meta_orig,
        p_df,
        regions_filter=cfg.get("regions"),
        die_filter=[str(die_filter)] if die_filter is not None else None,
    )
    regions = _resolve_distribution_regions(
        matched_by_region, cfg.get("regions"), split, die_filter
    )
    fig, axes_col, n_rows = _create_distribution_canvas(regions, cfg)
    err_col = str(cfg.get("error_color", "#EFC934"))
    inv_x = bool(cfg.get("invert_x", True))

    for row_idx, reg in enumerate(regions):
        stats = _compute_regional_distribution_statistics(
            matched_by_region[reg], ary_i_orig, ary_e_orig, ary_i_pred
        )
        _render_regional_distribution_row(
            axes_col[row_idx], reg, stats, die_filter, err_col, inv_x
        )

    _finish_distribution_figure(fig, n_rows, split, err_col, cfg)
    return fig, axes_col
