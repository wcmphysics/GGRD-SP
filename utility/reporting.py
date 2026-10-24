"""Reporting and side-by-side split comparison table utilities.

Provides reusable formatting of per-sample and summary metrics grouped
by splits (e.g., train vs. test) across regions or chemical elements.
"""

from __future__ import annotations

from typing import Any, Callable
import numpy as np
import pandas as pd


def format_split_comparison_table(
    df_data: pd.DataFrame,
    group_col: str,
    metric_specs: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Format sample or summary records into side-by-side split comparisons.

    Parameters
    ----------
    df_data : pd.DataFrame
        Source DataFrame containing at least group_col, 'split', and metric
        columns.
    group_col : str
        Column name to group rows by (e.g., 'region' or 'element').
    metric_specs : list[dict[str, Any]]
        List of metric specification dictionaries, each containing:
        - 'col': Column name in df_data.
        - 'label': Column prefix in the output table.
        - 'formatter': Optional Callable(mean, std) -> str.
        - 'std_col': Optional explicit column name for std if df_data is
          already aggregated.
        - 'num_mean_col': Optional pattern for numeric mean column.
        - 'num_std_col': Optional pattern for numeric std column.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'splits' (list[str]): Preferred split order (default
          ['train', 'test']).
        - 'format_str' (bool): Format as 'mean ± std' string (default True).
        - 'include_overall' (bool): Include overall row (default False).
        - 'overall_label' (str): Label for overall row
          (default 'Overall (MAE)').
        - 'overall_metrics' (list[str]): Columns to average for overall row.

    Returns
    -------
    pd.DataFrame
        Side-by-side comparison DataFrame with items as rows and columns
        grouped by split.
    """
    if df_data.empty or "split" not in df_data.columns or group_col not in df_data.columns:
        return pd.DataFrame()

    cfg = config or {}
    preferred_splits = list(cfg.get("splits", ["train", "test"]))
    format_str = bool(cfg.get("format_str", True))
    include_overall = bool(cfg.get("include_overall", False))
    overall_label = str(cfg.get("overall_label", "Overall (MAE)"))
    overall_metrics = list(cfg.get("overall_metrics", []))

    present_splits = [s for s in preferred_splits if s in df_data["split"].unique()]
    if not present_splits:
        return pd.DataFrame()

    unique_groups = sorted(df_data[group_col].dropna().unique())
    rows: list[dict[str, Any]] = []

    def _resolve_keys(spec: dict[str, Any], split: str, label: str) -> tuple[str, str]:
        mean_p = spec.get("num_mean_col", f"{label}_{{split}}_mean")
        std_p = spec.get("num_std_col", f"{label}_{{split}}_std")
        return mean_p.format(split=split), std_p.format(split=split)

    for group_val in unique_groups:
        row_dict: dict[str, Any] = {group_col: group_val}
        group_df = df_data[df_data[group_col] == group_val]

        for split in present_splits:
            split_df = group_df[group_df["split"] == split]

            for spec in metric_specs:
                col_name = str(spec["col"])
                label = str(spec.get("label", col_name))
                fmt_fn: Callable[[float, float], str] | None = spec.get("formatter")
                std_col_name = spec.get("std_col")

                if split_df.empty or col_name not in split_df.columns:
                    val_mean, val_std = np.nan, np.nan
                elif std_col_name and std_col_name in split_df.columns:
                    val_mean = float(split_df[col_name].iloc[0])
                    val_std = float(split_df[std_col_name].iloc[0])
                else:
                    series = split_df[col_name].dropna()
                    val_mean = float(series.mean()) if not series.empty else np.nan
                    val_std = float(series.std()) if len(series) > 1 else 0.0

                if format_str:
                    col_key = f"{label}_{split}"
                    if np.isnan(val_mean):
                        row_dict[col_key] = "N/A"
                    elif fmt_fn is not None:
                        row_dict[col_key] = fmt_fn(val_mean, val_std)
                    else:
                        row_dict[col_key] = f"{val_mean:.2f} ± {val_std:.2f}"
                else:
                    m_key, s_key = _resolve_keys(spec, split, label)
                    row_dict[m_key] = val_mean
                    row_dict[s_key] = val_std

        rows.append(row_dict)

    if include_overall and rows and overall_metrics:
        overall_row: dict[str, Any] = {group_col: overall_label}
        for split in present_splits:
            split_df = df_data[df_data["split"] == split]
            for spec in metric_specs:
                col_name = str(spec["col"])
                label = str(spec.get("label", col_name))
                fmt_fn = spec.get("formatter")
                m_key, s_key = _resolve_keys(spec, split, label)

                if col_name in overall_metrics and not split_df.empty and col_name in split_df.columns:
                    series = split_df[col_name].dropna()
                    m_val = float(series.mean()) if not series.empty else np.nan
                    s_val = float(series.std()) if len(series) > 1 else 0.0
                    if format_str:
                        overall_row[f"{label}_{split}"] = (
                            fmt_fn(m_val, s_val) if fmt_fn is not None
                            else f"{m_val:.2f} ± {s_val:.2f}"
                        )
                    else:
                        overall_row[m_key] = m_val
                        overall_row[s_key] = s_val
                else:
                    if format_str:
                        overall_row[f"{label}_{split}"] = "-"
                    else:
                        overall_row[m_key] = np.nan
                        overall_row[s_key] = np.nan
        rows.append(overall_row)

    return pd.DataFrame(rows)
