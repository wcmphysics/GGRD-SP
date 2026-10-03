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
