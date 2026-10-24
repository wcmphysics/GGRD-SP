"""Quantitative model evaluation and prediction metrics for GGRD-SP.

Provides numerical metrics comparing predicted spectra against ground truth
target spectra, fulfilling Part 4 of AGENTS.md independently from plotting
libraries.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd

from utility.reporting import format_split_comparison_table


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

    # Build fast O(1) lookup map from df_meta_orig:
    # Key: (measurement_id, region, die) -> (target_id, target_idx, spec_idx)
    meas_reg_die_to_spec_idx: dict[tuple[str, str, int], int] = {}
    src_to_tgt_info: dict[tuple[str, str, int], tuple[str | None, int | None]] = {}

    for row in df_meta_orig.itertuples():
        m_id = str(row.measurement_id)
        reg = str(row.region)
        die_val = int(row.die)
        key = (m_id, reg, die_val)
        meas_reg_die_to_spec_idx[key] = int(row.spectrum_index)

        tgt_m = getattr(row, "target_measurement_id", None)
        tgt_s = getattr(row, "target_spectrum_index", None)
        tgt_m_str = str(tgt_m) if pd.notna(tgt_m) else None
        tgt_s_int = int(tgt_s) if pd.notna(tgt_s) else None
        src_to_tgt_info[key] = (tgt_m_str, tgt_s_int)

    # Iterate over predicted spectra records using fast itertuples
    for pred_row in df_meta_pred.itertuples():
        region = str(pred_row.region)
        if regions_filter is not None and region not in regions_filter:
            continue

        src_meas_id = str(pred_row.source_measurement_id)
        die = int(pred_row.die)
        pred_idx = int(pred_row.spectrum_index)

        lookup_key = (src_meas_id, region, die)
        if lookup_key not in src_to_tgt_info:
            continue

        tgt_meas_id, tgt_idx = src_to_tgt_info[lookup_key]
        if tgt_meas_id is None:
            continue

        if tgt_idx is None:
            tgt_key = (tgt_meas_id, region, die)
            if tgt_key not in meas_reg_die_to_spec_idx:
                continue
            tgt_idx = meas_reg_die_to_spec_idx[tgt_key]

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
            "split": str(getattr(pred_row, "split", "all")),
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
    metric_specs: list[dict[str, Any]] = []
    for m in chosen_metrics:
        m_lower = str(m).lower()
        if m_lower not in METRIC_SPECS:
            raise ValueError(
                f"Unknown metric '{m}'. Choose from: {list(METRIC_SPECS.keys())}"
            )
        col_name, label, fmt_fn = METRIC_SPECS[m_lower]
        actual_col, actual_label = _resolve_metric_column(
            df_per_sample.columns, col_name, label, bool(is_normalized)
        )
        metric_specs.append({
            "col": actual_col,
            "label": actual_label,
            "formatter": fmt_fn,
        })

    return format_split_comparison_table(
        df_per_sample,
        group_col="region",
        metric_specs=metric_specs,
        config=config,
    )

