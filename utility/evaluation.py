"""Quantitative model evaluation and prediction metrics for GGRD-SP.

Provides numerical metrics comparing predicted spectra against ground truth target spectra,
fulfilling Part 4 of GEMINI.md independently from plotting libraries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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
        max_err_pct = float(np.max(np.abs(y_pred - y_true)) / max_true * 100.0)

        split_val = str(pred_row.get("split", "all"))

        records.append({
            "source_measurement_id": src_meas_id,
            "target_measurement_id": tgt_meas_id,
            "region": region,
            "die": die,
            "split": split_val,
            "normalized_mse": norm_mse,
            "rmse": raw_rmse,
            "peak_err_pct": peak_err_pct,
            "max_err_pct": max_err_pct,
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
                "max_err_pct_mean",
                "max_err_pct_std",
            ]
        )
    else:
        group_cols = list(cfg.get("group_by", ["region"]))
        metrics_to_agg = ["normalized_mse", "rmse", "peak_err_pct", "max_err_pct"]
        df_summary = (
            df_per_sample.groupby(group_cols)[metrics_to_agg]
            .agg(["mean", "std"])
            .reset_index()
        )
        # Flatten column multi-index
        df_summary.columns = [
            "_".join(col).strip("_") for col in df_summary.columns.values
        ]

    return df_per_sample, df_summary


METRIC_SPECS: dict[str, tuple[str, str, Any]] = {
    "norm_mse": ("normalized_mse", "norm_mse", lambda m, s: f"{m:.4f} ± {s:.4f}"),
    "normalized_mse": ("normalized_mse", "norm_mse", lambda m, s: f"{m:.4f} ± {s:.4f}"),
    "rmse": ("rmse", "rmse", lambda m, s: f"{m:.2f} ± {s:.2f}"),
    "rms": ("rmse", "rmse", lambda m, s: f"{m:.2f} ± {s:.2f}"),
    "peak_err%": ("peak_err_pct", "peak_err%", lambda m, s: f"{m:.2f}% ± {s:.2f}%"),
    "peak_err_pct": ("peak_err_pct", "peak_err%", lambda m, s: f"{m:.2f}% ± {s:.2f}%"),
    "max%err": ("max_err_pct", "max%err", lambda m, s: f"{m:.2f}% ± {s:.2f}%"),
    "max_err%": ("max_err_pct", "max%err", lambda m, s: f"{m:.2f}% ± {s:.2f}%"),
    "max_err_pct": ("max_err_pct", "max%err", lambda m, s: f"{m:.2f}% ± {s:.2f}%"),
}


def format_side_by_side_metrics(
    df_per_sample: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Format sample prediction metrics into a side-by-side comparison across splits.

    Produces a clean comparison table comparing model performance across configured
    splits ('train', 'test', etc.) and metrics ('norm_mse', 'peak_err%', 'max%err', etc.)
    for each spectral region, with columns grouped by split.

    Parameters
    ----------
    df_per_sample : pd.DataFrame
        DataFrame of per-sample evaluation metrics containing 'region', 'split',
        'normalized_mse', 'rmse', 'peak_err_pct', and 'max_err_pct'.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'splits' (list[str]): Data splits to display (default ['train', 'test']).
        - 'metrics' (list[str]): Metrics to display (default ['norm_mse', 'peak_err%', 'max%err']).
          Supported metrics: 'norm_mse' (or 'normalized_mse'), 'rmse' (or 'rms'),
          'peak_err%' (or 'peak_err_pct'), 'max%err' (or 'max_err%').
        - 'format_str' (bool): Format values as 'mean ± std' string (default True).

    Returns
    -------
    pd.DataFrame
        DataFrame with regions as rows and side-by-side metric columns grouped by split.
    """
    if df_per_sample.empty or "split" not in df_per_sample.columns:
        return pd.DataFrame()

    cfg = config or {}
    format_str = bool(cfg.get("format_str", True))
    preferred_splits = list(cfg.get("splits", ["train", "test"]))
    chosen_metrics = list(cfg.get("metrics", ["norm_mse", "peak_err%", "max%err"]))

    # Resolve metric specifications
    resolved_metrics: list[tuple[str, str, Any]] = []
    for m in chosen_metrics:
        m_lower = str(m).lower()
        if m_lower not in METRIC_SPECS:
            raise ValueError(
                f"Unknown metric '{m}'. Choose from: {list(METRIC_SPECS.keys())}"
            )
        resolved_metrics.append(METRIC_SPECS[m_lower])

    # Filter splits to those requested and actually present in the data
    present_splits = [s for s in preferred_splits if s in df_per_sample["split"].unique()]
    if not present_splits:
        return pd.DataFrame()

    regions = sorted(df_per_sample["region"].unique())
    rows: list[dict[str, Any]] = []

    for reg in regions:
        reg_df = df_per_sample[df_per_sample["region"] == reg]
        row_dict: dict[str, Any] = {"region": reg}

        # Group by split: for each split, output configured metrics in order
        for split in present_splits:
            s_df = reg_df[reg_df["split"] == split]
            if s_df.empty:
                for col_name, label, _ in resolved_metrics:
                    if format_str:
                        row_dict[f"{label}_{split}"] = "N/A"
                    else:
                        row_dict[f"{label}_{split}_mean"] = np.nan
                        row_dict[f"{label}_{split}_std"] = np.nan
                continue

            for col_name, label, fmt_fn in resolved_metrics:
                if col_name not in s_df.columns:
                    val_mean = np.nan
                    val_std = np.nan
                else:
                    val_mean = float(s_df[col_name].mean())
                    val_std = float(s_df[col_name].std()) if len(s_df) > 1 else 0.0

                if format_str:
                    if np.isnan(val_mean):
                        row_dict[f"{label}_{split}"] = "N/A"
                    else:
                        row_dict[f"{label}_{split}"] = fmt_fn(val_mean, val_std)
                else:
                    row_dict[f"{label}_{split}_mean"] = val_mean
                    row_dict[f"{label}_{split}_std"] = val_std

        rows.append(row_dict)

    return pd.DataFrame(rows)
