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
    (ary_i_orig, ary_e_orig, meta_orig), (
        ary_i_pred,
        ary_e_pred,
        meta_pred,
    ) = (data_original, data_predicted)
    cfg = config or {}
    die = int(cfg.get("die", 0))
    meas_id = cfg.get("measurement_id")
    target_region = cfg.get("region")
    if meas_id is None:
        p_ids = set(
            meta_pred.get("source_measurement_id", []).dropna().astype(str)
        )
        cands = (
            [
                str(c)
                for c in (cfg.get("session_splits") or {}).get("test", [])
            ]
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
            raise ValueError(
                "No paired source measurements found in df_meta_orig."
            )
    src_rows = meta_orig[
        (meta_orig["measurement_id"] == meas_id) & (meta_orig["die"] == die)
    ]
    if src_rows.empty:
        raise ValueError(
            f"No source spectrum matching meas_id='{meas_id}', die={die}."
        )
    is_single = target_region is not None
    if is_single:
        if target_region not in src_rows["region"].values:
            raise ValueError(
                f"No source spectrum matching meas_id='{meas_id}', "
                f"die={die}, region='{target_region}'."
            )
        regions = [target_region]
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
    else:
        regions = list(src_rows["region"].unique())
        if not regions:
            raise ValueError(
                f"No regions found for source measurement '{meas_id}', die={die}."
            )
        fig, axes, axes_to_use = _setup_subplot_grid(
            len(regions), max_cols=3, figsize=cfg.get("figsize")
        )
        ax_res = None
    tgt_m = str(src_rows.iloc[0].get("target_measurement_id", ""))
    pred_subset = meta_pred[
        (meta_pred["source_measurement_id"] == meas_id)
        & (meta_pred["die"] == die)
    ]
    matched_by_region = match_prediction_source_target_indices(
        meta_orig,
        pred_subset,
        regions_filter=regions,
        die_filter=[die],
    )
    orig_by_idx = {int(r.spectrum_index): r for r in meta_orig.itertuples()}
    plotted = 0
    for idx, reg in enumerate(regions):
        ax = axes_to_use[idx]
        triplets = matched_by_region.get(reg, [])
        if not triplets:
            if is_single:
                raise ValueError(
                    f"Target or predicted spectrum for {reg} not found."
                )
            ax.set_visible(False)
            continue
        i_s, i_t, i_p = triplets[0]
        reg_src = orig_by_idx.get(i_s)
        src_t = str(getattr(reg_src, "tool", "Tool")) if reg_src else "Tool"
        tgt_t = (
            str(getattr(reg_src, "target_tool", "TargetTool"))
            if reg_src
            else "TargetTool"
        )
        for e, i, lbl, c, ls, lw in [
            (
                ary_e_orig[i_s],
                ary_i_orig[i_s],
                f"Source ({src_t})"
                if not is_single
                else f"Source Measured ({src_t})",
                "black",
                "-",
                1.5,
            ),
            (
                ary_e_orig[i_t],
                ary_i_orig[i_t],
                f"Target ({tgt_t})"
                if not is_single
                else f"True Target ({tgt_t})",
                "red",
                "-",
                1.8,
            ),
            (
                ary_e_pred[i_p],
                ary_i_pred[i_p],
                f"Transformed ({src_t}->{tgt_t})",
                "blue",
                ":",
                2.0,
            ),
        ]:
            ax.plot(e, i, label=lbl, color=c, ls=ls, lw=lw)
        ax.legend(loc="best", fontsize="small" if is_single else "x-small")
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
            if (not is_single or ax_res is None)
            else None,
            ylabel="Intensity (counts)",
            invert_x=cfg.get("invert_x", True),
            grid_alpha=0.5,
        )
        if is_single and ax_res is not None:
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
                invert_x=cfg.get("invert_x", True),
                grid_alpha=0.6,
            )
        plotted += 1
    if plotted == 0:
        raise ValueError(
            f"No matching target or predicted spectra for source session '{meas_id}', die={die}."
        )
    if not is_single:
        fig.suptitle(
            cfg.get(
                "title",
                f"Spectral Transfer Comparison | Session: {meas_id} (Die {die})",
            ),
            fontsize=13,
        )
    _finish_fig(fig, cfg)
    return fig, axes


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

    p_df = meta_pred.copy()
    if split and str(split).lower() != "all" and "split" in p_df.columns:
        p_df = p_df[
            p_df["split"].astype(str).str.lower() == str(split).lower()
        ]
    if die_filter is not None:
        p_df = p_df[p_df["die"] == int(die_filter)]
    if p_df.empty:
        raise ValueError(
            f"No predicted spectra found for split='{split}'"
            + (f", die={die_filter}" if die_filter is not None else "")
            + "."
        )

    matched_by_region = match_prediction_source_target_indices(
        meta_orig,
        p_df,
        regions_filter=cfg.get("regions"),
        die_filter=[int(die_filter)] if die_filter is not None else None,
    )

    specified_regions = cfg.get("regions")
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

    n_rows = len(regions_to_plot)
    figsize = cfg.get(
        "figsize", (10.0, 1.5 + 2.4 * n_rows + 0.8 * (n_rows - 1))
    )
    dpi = int(cfg.get("dpi", 300))
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, dpi=dpi, squeeze=False)
    axes_col = axes[:, 0]
    error_color = str(cfg.get("error_color", "#EFC934"))
    invert_x = bool(cfg.get("invert_x", True))

    for row_idx, reg in enumerate(regions_to_plot):
        ax = axes_col[row_idx]
        triplets = matched_by_region[reg]

        src_raw = np.array(
            [ary_i_orig[s] for s, _, _ in triplets], dtype=np.float64
        )
        tgt_raw = np.array(
            [ary_i_orig[t] for _, t, _ in triplets], dtype=np.float64
        )
        pred_raw = np.array(
            [ary_i_pred[p] for _, _, p in triplets], dtype=np.float64
        )

        tgt_max = np.max(tgt_raw, axis=1, keepdims=True)
        tgt_max = np.where(
            (tgt_max <= 0.0) | (~np.isfinite(tgt_max)), 1e-12, tgt_max
        )

        src_stack = (src_raw / tgt_max) * 100.0
        tgt_stack = (tgt_raw / tgt_max) * 100.0
        pred_stack = (pred_raw / tgt_max) * 100.0
        err_stack = np.abs(pred_stack - tgt_stack)
        energy_arr = ary_e_orig[triplets[0][1]]

        src_mean = np.mean(src_stack, axis=0)
        tgt_mean = np.mean(tgt_stack, axis=0)
        pred_mean = np.mean(pred_stack, axis=0)
        err_mean = np.mean(err_stack, axis=0)

        src_std = np.std(src_stack, axis=0)
        tgt_std = np.std(tgt_stack, axis=0)
        pred_std = np.std(pred_stack, axis=0)
        err_std = np.std(err_stack, axis=0)

        src_low = np.maximum(src_mean - src_std, 0.0)
        src_high = np.maximum(src_mean + src_std, src_low)

        tgt_low = np.maximum(tgt_mean - tgt_std, 0.0)
        tgt_high = np.maximum(tgt_mean + tgt_std, tgt_low)

        pred_low = np.maximum(pred_mean - pred_std, 0.0)
        pred_high = np.maximum(pred_mean + pred_std, pred_low)

        err_low = np.maximum(err_mean - err_std, 0.0)
        err_high = np.maximum(err_mean + err_std, err_low)

        ax.plot(
            energy_arr, src_mean, color="black", ls="-", lw=1.6, label="Source"
        )
        ax.fill_between(
            energy_arr, src_low, src_high, color="black", alpha=0.15
        )

        ax.plot(
            energy_arr, tgt_mean, color="red", ls="-", lw=1.6, label="Target"
        )
        ax.fill_between(energy_arr, tgt_low, tgt_high, color="red", alpha=0.15)

        ax.plot(
            energy_arr,
            pred_mean,
            color="blue",
            ls="--",
            lw=1.8,
            label="Predicted",
        )
        ax.fill_between(
            energy_arr, pred_low, pred_high, color="blue", alpha=0.15
        )

        ax.set_xlabel("Binding Energy (eV)", fontsize=8)
        ax.set_ylabel("Intensity (% of Target Max)", color="black", fontsize=8)
        if invert_x and not ax.xaxis_inverted():
            ax.invert_xaxis()
        ax.grid(True, linestyle="--", alpha=0.5)

        ax_err = ax.twinx()
        ax_err.plot(
            energy_arr,
            err_mean,
            color=error_color,
            ls="-",
            lw=1.6,
            label="Abs Error",
        )
        ax_err.fill_between(
            energy_arr, err_low, err_high, color=error_color, alpha=0.20
        )
        ax_err.set_ylabel("Absolute Error (%)", color="black", fontsize=8)
        ax_err.tick_params(axis="y", labelcolor="black", labelsize=7.5)
        ax_err.set_ylim(bottom=0.0)
        ax.tick_params(axis="both", labelsize=7.5)

        die_desc = (
            f"Die {die_filter}" if die_filter is not None else "All Dies"
        )
        ax.set_title(
            f"Region: {reg} (N={len(src_stack)}) | {die_desc}", fontsize=9
        )

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
    if cfg.get("save_path"):
        fig.savefig(str(cfg["save_path"]), dpi=dpi)
    if cfg.get("show"):
        plt.show()
    return fig, axes_col
