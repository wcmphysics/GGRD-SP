"""Reporting and side-by-side split comparison table utilities.

Provides reusable formatting of per-sample and summary metrics grouped
by splits (e.g., train vs. test) across regions or chemical elements.
"""

from __future__ import annotations

from typing import Any, Callable
import numpy as np
import pandas as pd


def _resolve_metric_keys(
    spec: dict[str, Any], split: str, label: str
) -> tuple[str, str]:
    """Resolve column names for mean and std numeric outputs."""
    mean_p = spec.get("num_mean_col", f"{label}_{{split}}_mean")
    std_p = spec.get("num_std_col", f"{label}_{{split}}_std")
    return mean_p.format(split=split), std_p.format(split=split)


def _extract_metric_mean_std(
    split_df: pd.DataFrame, col_name: str, std_col_name: str | None
) -> tuple[float, float]:
    """Extract or calculate mean and std for a metric in a split subset."""
    if split_df.empty or col_name not in split_df.columns:
        return np.nan, np.nan
    if std_col_name and std_col_name in split_df.columns:
        return float(split_df[col_name].iloc[0]), float(
            split_df[std_col_name].iloc[0]
        )
    series = split_df[col_name].dropna()
    if series.empty:
        return np.nan, np.nan
    val_std = float(series.std()) if len(series) > 1 else 0.0
    return float(series.mean()), val_std


def _format_metric_cell(
    val_mean: float,
    val_std: float,
    fmt_fn: Callable[[float, float], str] | None,
) -> str:
    """Format numeric mean and standard deviation into a display string."""
    if np.isnan(val_mean):
        return "N/A"
    if fmt_fn is not None:
        return fmt_fn(val_mean, val_std)
    return f"{val_mean:.2f} ± {val_std:.2f}"


def _format_group_row(
    group_df: pd.DataFrame,
    group_col: str,
    group_val: Any,
    present_splits: list[str],
    metric_specs: list[dict[str, Any]],
    format_str: bool,
) -> dict[str, Any]:
    """Format metric values for a single group across present splits."""
    row_dict: dict[str, Any] = {group_col: group_val}
    for split in present_splits:
        split_df = group_df[group_df["split"] == split]
        for spec in metric_specs:
            col_name = str(spec["col"])
            label = str(spec.get("label", col_name))
            val_mean, val_std = _extract_metric_mean_std(
                split_df, col_name, spec.get("std_col")
            )
            if format_str:
                row_dict[f"{label}_{split}"] = _format_metric_cell(
                    val_mean, val_std, spec.get("formatter")
                )
            else:
                m_key, s_key = _resolve_metric_keys(spec, split, label)
                row_dict[m_key] = val_mean
                row_dict[s_key] = val_std
    return row_dict


def _calculate_overall_summary_row(
    df_data: pd.DataFrame,
    group_col: str,
    overall_label: str,
    present_splits: list[str],
    metric_specs: list[dict[str, Any]],
    overall_metrics: list[str],
    format_str: bool,
) -> dict[str, Any]:
    """Calculate the overall summary row aggregated across all groups."""
    overall_row: dict[str, Any] = {group_col: overall_label}
    for split in present_splits:
        split_df = df_data[df_data["split"] == split]
        for spec in metric_specs:
            col_name = str(spec["col"])
            label = str(spec.get("label", col_name))
            m_key, s_key = _resolve_metric_keys(spec, split, label)

            if col_name not in overall_metrics:
                overall_row[f"{label}_{split}" if format_str else m_key] = (
                    "-" if format_str else np.nan
                )
                if not format_str:
                    overall_row[s_key] = np.nan
                continue

            val_mean, val_std = _extract_metric_mean_std(
                split_df, col_name, None
            )
            if format_str:
                fmt_fn = spec.get("formatter")
                overall_row[f"{label}_{split}"] = (
                    fmt_fn(val_mean, val_std)
                    if fmt_fn is not None and not np.isnan(val_mean)
                    else f"{val_mean:.2f} ± {val_std:.2f}"
                )
            else:
                overall_row[m_key] = val_mean
                overall_row[s_key] = val_std
    return overall_row


def _is_valid_reporting_dataframe(
    df_data: pd.DataFrame, group_col: str
) -> bool:
    """Check if the source DataFrame has required columns and rows."""
    if df_data.empty or "split" not in df_data.columns:
        return False
    return group_col in df_data.columns


def _should_append_overall(cfg: dict[str, Any], has_rows: bool) -> bool:
    """Determine whether overall summary row should be included."""
    if not (bool(cfg.get("include_overall", False)) and has_rows):
        return False
    return bool(cfg.get("overall_metrics"))


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
    if not _is_valid_reporting_dataframe(df_data, group_col):
        return pd.DataFrame()

    cfg = config or {}
    preferred_splits = list(cfg.get("splits", ["train", "test"]))
    present_splits = [
        s for s in preferred_splits if s in df_data["split"].unique()
    ]
    if not present_splits:
        return pd.DataFrame()

    format_str = bool(cfg.get("format_str", True))
    unique_groups = sorted(df_data[group_col].dropna().unique())
    rows = [
        _format_group_row(
            df_data[df_data[group_col] == g],
            group_col,
            g,
            present_splits,
            metric_specs,
            format_str,
        )
        for g in unique_groups
    ]

    if _should_append_overall(cfg, bool(rows)):
        rows.append(
            _calculate_overall_summary_row(
                df_data,
                group_col,
                str(cfg.get("overall_label", "Overall (MAE)")),
                present_splits,
                metric_specs,
                list(cfg.get("overall_metrics", [])),
                format_str,
            )
        )

    return pd.DataFrame(rows)
