"""Quantitative model evaluation and prediction metrics for GGRD-SP.

Provides numerical metrics comparing predicted spectra against ground truth
target spectra, fulfilling Part 4 of GEMINI.md independently from plotting
libraries.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd


def calculate_prediction_metrics(
    data_original: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    data_predicted: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate quantitative metrics comparing predicted and target spectra.

    Computes Normalized MSE, Raw RMSE, and Peak Relative Error % per paired
    sample.

    Parameters
    ----------
    data_original : tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Original measured data (ary_intensity, ary_energy, df_meta).
    data_predicted : tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Predicted data (ary_intensity_pred, ary_energy_pred, df_meta_pred).
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'regions' (list[str] | None): Specific regions to evaluate.
        - 'eps' (float): Epsilon floor for maximum normalization (default 1e-4).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_per_sample, df_summary_by_region)
    """
    ary_intensity_orig, _, df_meta_orig = data_original
    ary_intensity_pred, _, df_meta_pred = data_predicted

    cfg = config or {}
    regions_filter = cfg.get("regions")
    eps = max(float(cfg.get("eps", 1e-4)), 1e-12)
    norm_by_tgt_max = bool(cfg.get("normalized_by_target_max", True))
    mse_col = "MSE_normalized_by_target" if norm_by_tgt_max else "MSE"

    records: list[dict[str, Any]] = []

    # Iterate over predicted spectra records
    for _, pred_row in df_meta_pred.iterrows():
        region = str(pred_row["region"])
        if regions_filter is not None and region not in regions_filter:
            continue

        src_meas_id = str(pred_row["source_measurement_id"])
        die = int(pred_row["die"])
        pred_idx = int(pred_row["spectrum_index"])

        # Find matching source row in df_meta_orig
        src_match = df_meta_orig[
            (df_meta_orig["measurement_id"] == src_meas_id)
            & (df_meta_orig["region"] == region)
            & (df_meta_orig["die"] == die)
        ]
        if src_match.empty or pd.isna(src_match["target_measurement_id"].iloc[0]):
            continue

        src_row = src_match.iloc[0]
        tgt_meas_id = str(src_row["target_measurement_id"])

        # Use target_spectrum_index directly from df_meta if available
        if (
            "target_spectrum_index" in src_match.columns
            and pd.notna(src_row["target_spectrum_index"])
        ):
            tgt_idx = int(src_row["target_spectrum_index"])
        else:
            tgt_match = df_meta_orig[
                (df_meta_orig["measurement_id"] == tgt_meas_id)
                & (df_meta_orig["region"] == region)
                & (df_meta_orig["die"] == die)
            ]
            if tgt_match.empty:
                continue
            tgt_idx = int(tgt_match["spectrum_index"].iloc[0])

        y_true = ary_intensity_orig[tgt_idx]
        y_pred = ary_intensity_pred[pred_idx]
        diff = y_pred - y_true

        max_true = max(float(np.max(np.abs(y_true))), eps)
        max_pred = float(np.max(np.abs(y_pred)))

        mse_val = float(
            np.mean((diff / max_true) ** 2) if norm_by_tgt_max else np.mean(diff**2)
        )
        raw_rmse = float(np.sqrt(np.mean(diff**2)))
        peak_err_pct = float(abs(max_pred - max_true) / max_true * 100.0)
        max_err_pct = float(np.max(np.abs(diff)) / max_true * 100.0)

        records.append({
            "source_measurement_id": src_meas_id,
            "target_measurement_id": tgt_meas_id,
            "region": region,
            "die": die,
            "split": str(pred_row.get("split", "all")),
            mse_col: mse_val,
            "rmse": raw_rmse,
            "peak_error_percentage": peak_err_pct,
            "max_error_percentage": max_err_pct,
        })

    df_per_sample = pd.DataFrame(records)
    if df_per_sample.empty:
        if not df_meta_pred.empty:
            warnings.warn(
                "calculate_prediction_metrics found zero matching target records for "
                f"{len(df_meta_pred)} prediction rows in df_meta_orig."
            )
        df_summary = pd.DataFrame(
            columns=[
                "region",
                f"{mse_col}_mean",
                f"{mse_col}_std",
                "rmse_mean",
                "rmse_std",
                "peak_error_percentage_mean",
                "peak_error_percentage_std",
                "max_error_percentage_mean",
                "max_error_percentage_std",
            ]
        )
        return df_per_sample, df_summary

    group_cols = list(cfg.get("group_by", ["region"]))
    metrics_to_agg = [mse_col, "rmse", "peak_error_percentage", "max_error_percentage"]
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


_MSE_SPEC = ("MSE", "MSE", lambda m, s: f"{m:.6f} ± {s:.6f}")
_NORM_MSE_SPEC = (
    "MSE_normalized_by_target",
    "norm_mse",
    lambda m, s: f"{m:.4f} ± {s:.4f}",
)
_RMSE_SPEC = ("rmse", "rmse", lambda m, s: f"{m:.2f} ± {s:.2f}")
_PEAK_ERR_SPEC = (
    "peak_error_percentage",
    "peak_error_percentage",
    lambda m, s: f"{m:.2f}% ± {s:.2f}%",
)
_MAX_ERR_SPEC = (
    "max_error_percentage",
    "max_error_percentage",
    lambda m, s: f"{m:.2f}% ± {s:.2f}%",
)

METRIC_SPECS: dict[str, tuple[str, str, Any]] = {
    "mse": _MSE_SPEC,
    "mse_normalized_by_target": _NORM_MSE_SPEC,
    "norm_mse": _NORM_MSE_SPEC,
    "normalized_mse": _NORM_MSE_SPEC,
    "rmse": _RMSE_SPEC,
    "rms": _RMSE_SPEC,
    "peak_error_percentage": _PEAK_ERR_SPEC,
    "max_error_percentage": _MAX_ERR_SPEC,
}


def _resolve_metric_column(
    columns: pd.Index,
    col_name: str,
    label: str,
    is_normalized: bool,
) -> tuple[str, str]:
    """Resolve effective metric column name and display label."""
    actual_col = col_name
    actual_label = "MSE" if (not is_normalized and label == "norm_mse") else label
    if actual_col not in columns and col_name in (
        "MSE",
        "MSE_normalized_by_target",
        "normalized_mse",
    ):
        for alt in ("MSE_normalized_by_target", "MSE", "normalized_mse"):
            if alt in columns:
                actual_col = alt
                actual_label = "MSE" if alt == "MSE" else "norm_mse"
                break
    return actual_col, actual_label


def format_side_by_side_metrics(
    df_per_sample: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Format sample prediction metrics into side-by-side split comparisons.

    Produces a clean comparison table comparing model performance across
    configured splits ('train', 'test', etc.) and metrics ('norm_mse',
    'peak_error_percentage', 'max_error_percentage', etc.) for each spectral
    region, with columns grouped by split.

    Parameters
    ----------
    df_per_sample : pd.DataFrame
        DataFrame of per-sample evaluation metrics containing 'region',
        'split', 'normalized_mse', 'rmse', 'peak_error_percentage', and
        'max_error_percentage'.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'splits' (list[str]): Splits to display (default ['train', 'test']).
        - 'metrics' (list[str]): Metrics to display (default ['norm_mse',
          'peak_error_percentage', 'max_error_percentage']).
        - 'format_str' (bool): Format as 'mean ± std' string (default True).

    Returns
    -------
    pd.DataFrame
        DataFrame with regions as rows and metric columns grouped by split.
    """
    if df_per_sample.empty or "split" not in df_per_sample.columns:
        return pd.DataFrame()

    cfg = config or {}
    format_str = bool(cfg.get("format_str", True))
    preferred_splits = list(cfg.get("splits", ["train", "test"]))
    is_normalized = cfg.get("normalized_by_target_max")
    if is_normalized is None:
        is_normalized = (
            "MSE_normalized_by_target" in df_per_sample.columns
            or "normalized_mse" in df_per_sample.columns
        )
    default_mse_key = "norm_mse" if is_normalized else "mse"
    chosen_metrics = list(
        cfg.get("metrics", [default_mse_key, "peak_error_percentage", "max_error_percentage"])
    )

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
    present_splits = [
        s for s in preferred_splits if s in df_per_sample["split"].unique()
    ]
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
            for col_name, label, fmt_fn in resolved_metrics:
                actual_col, actual_label = _resolve_metric_column(
                    s_df.columns, col_name, label, bool(is_normalized)
                )
                if s_df.empty or actual_col not in s_df.columns:
                    val_mean, val_std = np.nan, np.nan
                else:
                    val_mean = float(s_df[actual_col].mean())
                    val_std = (
                        float(s_df[actual_col].std()) if len(s_df) > 1 else 0.0
                    )

                if format_str:
                    row_dict[f"{actual_label}_{split}"] = (
                        "N/A"
                        if np.isnan(val_mean)
                        else fmt_fn(val_mean, val_std)
                    )
                else:
                    row_dict[f"{actual_label}_{split}_mean"] = val_mean
                    row_dict[f"{actual_label}_{split}_std"] = val_std

        rows.append(row_dict)

    return pd.DataFrame(rows)

