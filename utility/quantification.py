"""Quantification and Shirley background subtraction module for GGRD-SP.

Provides Shirley background subtraction, peak area integration, and atomic
percentage calculation fulfilling Stage 12 of AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml
import warnings

import numpy as np
import pandas as pd

from utility.reporting import format_split_comparison_table
from utility.shirley import (
    _calculate_endpoint_intensity,
    _find_minima_endpoints,
    _resolve_direct_endpoints,
    calculate_shirley_background,
    determine_shirley_endpoints,
)


def load_rsf_table(
    config_or_path: dict[str, Any] | str | Path | None = None,
) -> dict[str, float]:
    """Load Scofield relative sensitivity factor (RSF) table from configuration.

    Parameters
    ----------
    config_or_path : dict[str, Any] | str | Path | None, optional
        Configuration dictionary containing 'quantification.rsf_table' or
        'rsf_table', file path to YAML, or None (defaults to
        'config/config.yaml').

    Returns
    -------
    dict[str, float]
        Dictionary mapping spectral region name to its RSF value.

    Raises
    ------
    FileNotFoundError
        If configuration file does not exist.
    KeyError
        If 'rsf_table' is missing.
    ValueError
        If 'rsf_table' content is invalid.
    """
    if config_or_path is None:
        config_or_path = Path("config/config.yaml")

    if isinstance(config_or_path, (str, Path)):
        p = Path(config_or_path)
        if not p.is_file():
            raise FileNotFoundError(f"Configuration file not found: '{p}'")
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"YAML content in '{p}' must be a dictionary.")
        quant = raw.get("quantification", {})
        tbl = (
            quant.get("rsf_table")
            if isinstance(quant, dict)
            else raw.get("rsf_table")
        )
        if tbl and isinstance(tbl, dict):
            return {str(k): float(v) for k, v in tbl.items()}
        raise KeyError(
            f"Configuration file '{p}' does not contain 'quantification.rsf_table'."
        )

    if isinstance(config_or_path, dict):
        if "rsf_table" in config_or_path and isinstance(
            config_or_path["rsf_table"], dict
        ):
            return {
                str(k): float(v) for k, v in config_or_path["rsf_table"].items()
            }
        if "quantification" in config_or_path and isinstance(
            config_or_path["quantification"], dict
        ):
            q_tbl = config_or_path["quantification"].get("rsf_table")
            if isinstance(q_tbl, dict):
                return {str(k): float(v) for k, v in q_tbl.items()}

    raise KeyError(
        "Configuration missing required 'rsf_table' (or 'quantification.rsf_table')."
    )


class _LazyScofieldRSF(dict):
    """Lazy dictionary wrapper deferring load_rsf_table() until first access."""

    def __init__(self, config_or_path: Any = None) -> None:
        super().__init__()
        self._config_or_path = config_or_path
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            loaded_dict = load_rsf_table(self._config_or_path)
            super().update(loaded_dict)
            self._loaded = True

    def __getitem__(self, key: Any) -> Any:
        self._ensure_loaded()
        return super().__getitem__(key)

    def __iter__(self) -> Any:
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self) -> int:
        self._ensure_loaded()
        return super().__len__()

    def __contains__(self, key: Any) -> bool:
        self._ensure_loaded()
        return super().__contains__(key)

    def keys(self) -> Any:
        self._ensure_loaded()
        return super().keys()

    def values(self) -> Any:
        self._ensure_loaded()
        return super().values()

    def items(self) -> Any:
        self._ensure_loaded()
        return super().items()

    def get(self, key: Any, default: Any = None) -> Any:
        self._ensure_loaded()
        return super().get(key, default)

    def copy(self) -> dict:
        self._ensure_loaded()
        return super().copy()

    def __repr__(self) -> str:
        self._ensure_loaded()
        return super().__repr__()

    def __str__(self) -> str:
        self._ensure_loaded()
        return super().__str__()

    def __bool__(self) -> bool:
        self._ensure_loaded()
        return len(self) > 0

    def __eq__(self, other: Any) -> bool:
        self._ensure_loaded()
        return super().__eq__(other)


# Standard Scofield relative sensitivity factors (RSF) for Al-Ka excitation (lazily loaded)
DEFAULT_SCOFIELD_RSF: dict[str, float] = _LazyScofieldRSF()


def _mean_std(
    series: pd.Series,
    decimals: int | None = None,
) -> tuple[float, float]:
    """Return sample mean and sample standard deviation (0.0 if len <= 1)."""
    mean_val = float(series.mean())
    std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    if decimals is not None:
        return round(mean_val, decimals), round(std_val, decimals)
    return mean_val, std_val


def get_default_quantification_config() -> dict[str, Any]:
    """Return default configuration for atomic percentage quantification.

    Returns
    -------
    dict[str, Any]
        Dictionary packaging quantification parameters:
        - 'material': 'NMG'
        - 'measurement_id': None (if None, calculates for all sessions)
        - 'rsf_table': dict mapping region name to relative sensitivity factor
        - 'default_endpoint_strategy': 'minima' ('minima', 'edge', or 'direct')
        - 'default_endpoint_average_ev': 1.0 (average over 1.0 eV window)
        - 'region_endpoint_config': None (optional per-region overrides)
        - 'smooth_search': False
        - 'max_iter': 50
        - 'tol': 1e-5
    """
    return {
        "material": "NMG",
        "measurement_id": None,
        "rsf_table": dict(DEFAULT_SCOFIELD_RSF),
        "default_endpoint_strategy": "minima",
        "default_endpoint_average_ev": 1.0,
        "region_endpoint_config": None,
        "smooth_search": False,
        "max_iter": 50,
        "tol": 1e-5,
    }


def calculate_atomic_percentages(
    ary_energy: np.ndarray,
    ary_intensity: np.ndarray,
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate atomic percentages in wide format (per-die and summary).

    Fulfills Stage 12 of AGENTS.md:
    - Calculates net peak area via Shirley background subtraction per region.
    - Supports systematic Shirley endpoint determination ('minima', 'edge', or
      'direct') across all regions.
    - Normalizes net areas by Relative Sensitivity Factors (RSF).
    - Returns per-die atomic percentages and summary statistics across dies.

    Parameters
    ----------
    ary_energy : np.ndarray
        2D array of binding energies, shape (N_total, N_points).
    ary_intensity : np.ndarray
        2D array of intensities, shape (N_total, N_points).
    df_meta : pd.DataFrame
        Metadata DataFrame describing spectra rows.
    config : dict[str, Any] | None, optional
        Dictionary bundling options:
        - 'measurement_id' (str | None): Filter to specific measurement session.
        - 'material' (str | None): Target material (default 'NMG').
        - 'rsf_dict' (dict[str, float]): Element RSF mapping.
        - 'default_endpoint_strategy' (str): 'minima', 'edge', or 'direct'.
        - 'default_endpoint_average_ev' (float | None): Default averaging width.
        - 'region_endpoint_config' (dict | None): Per-region overrides.
        - 'max_iter' (int): Shirley max iterations (default 50).
        - 'tol' (float): Shirley tolerance (default 1e-5).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - df_per_die: Wide DataFrame with per-die atomic percentages.
        - df_summary: DataFrame with mean and std across dies per measurement.

    Raises
    ------
    ValueError
        If no matching data is found, required columns are missing, or RSF <= 0.
    """
    cfg = get_default_quantification_config()
    if config is not None:
        cfg.update(config)

    target_meas: str | None = cfg.get("measurement_id")
    material: str | None = cfg.get("material")

    rsf_table = cfg.get("rsf_table")
    if not isinstance(rsf_table, dict):
        raise KeyError("Configuration missing required 'rsf_table' dictionary.")

    required_cols = {"spectrum_index", "measurement_id", "die", "region"}
    missing = required_cols - set(df_meta.columns)
    if missing:
        raise ValueError(f"df_meta is missing required columns: {sorted(missing)}")

    filtered_df = df_meta.copy()
    if target_meas is not None:
        filtered_df = filtered_df[filtered_df["measurement_id"] == target_meas]
    if material is not None and "material" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["material"] == material]

    if filtered_df.empty:
        raise ValueError(
            f"No spectra found matching criteria: "
            f"measurement_id={target_meas}, material={material}"
        )

    norm_mode = cfg.get("normalization_mode")
    if norm_mode is None:
        norm_mode = df_meta.attrs.get("normalization_mode")
    if norm_mode is None and "normalization_mode" in df_meta.columns:
        norm_series = df_meta["normalization_mode"].dropna()
        if not norm_series.empty:
            norm_mode = str(norm_series.iloc[0])
    if norm_mode in (
        "by_regional_max_intensity_of_source",
        "by_regional_max_intensity",
    ):
        warnings.warn(
            f"Atomic percentage quantification: normalization_mode '{norm_mode}' "
            "normalizes each regional spectrum independently, which distorts "
            "relative atomic percentages across elements.",
            UserWarning,
            stacklevel=2,
        )

    rows_data: list[dict[str, Any]] = []
    for row in filtered_df.itertuples():
        region = str(row.region)
        spec_idx = int(row.spectrum_index)
        _, net_area = calculate_shirley_background(
            ary_energy[spec_idx],
            ary_intensity[spec_idx],
            dict(cfg, region=region),
        )

        if region not in rsf_table:
            raise KeyError(
                f"Spectral region/element '{region}' is absent from "
                f"quantification.rsf_table. Available RSF regions: "
                f"{sorted(rsf_table.keys())}."
            )
        rsf = float(rsf_table[region])
        if rsf <= 0.0:
            raise ValueError(
                f"RSF for region '{region}' must be positive, got {rsf}"
            )

        rows_data.append({
            "measurement_id": str(row.measurement_id),
            "material": str(getattr(row, "material", "NMG")),
            "die": int(row.die),
            "region": region,
            "scaled_area": max(0.0, net_area) / rsf,
        })

    df_calc = pd.DataFrame(rows_data).drop_duplicates(
        subset=["measurement_id", "material", "die", "region"], keep="last"
    )

    # Normalize scaled areas row-wise to 100% per measurement session and die
    totals = df_calc.groupby(["measurement_id", "material", "die"])[
        "scaled_area"
    ].transform("sum")
    df_calc["at_pct"] = np.where(
        totals > 0.0, (df_calc["scaled_area"] / totals) * 100.0, 0.0
    ).round(4)
    df_calc["col_name"] = df_calc["region"] + "_atomic_percentage"

    # Pivot regions to wide format preserving regional discovery order
    df_per_die = df_calc.pivot(
        index=["measurement_id", "material", "die"],
        columns="col_name",
        values="at_pct",
    ).reset_index()
    df_per_die.columns.name = None
    region_cols = [
        f"{r}_atomic_percentage"
        for r in filtered_df["region"].unique()
        if f"{r}_atomic_percentage" in df_per_die.columns
    ]
    df_per_die = df_per_die[["measurement_id", "material", "die"] + region_cols]

    # Summary statistics (mean and std across dies) per measurement session
    summary_records: list[dict[str, Any]] = []
    for meas_id, m_df in df_per_die.groupby("measurement_id", sort=False):
        mat_val = m_df["material"].iloc[0]
        mean_row: dict[str, Any] = {
            "measurement_id": meas_id,
            "material": mat_val,
            "metric": "mean",
        }
        std_row: dict[str, Any] = {
            "measurement_id": meas_id,
            "material": mat_val,
            "metric": "std",
        }
        for col in region_cols:
            mean_row[col], std_row[col] = _mean_std(m_df[col], decimals=4)
        summary_records.extend([mean_row, std_row])

    return df_per_die, pd.DataFrame(summary_records)


def calculate_atomic_percentage_split_statistics(
    data_original: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    data_predicted: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate atomic percentage statistics across data splits.

    Fulfills chemical quantification comparison across splits:
    1. Computes Shirley net peak areas and atomic percentages for Source, True
       Target, and Predicted Target spectra across each die of paired sessions.
    2. Groups results by data split ('train', 'test', etc.) and element.
    3. Provides detailed element-wise statistics: Source mean/std, Target
       mean/std, Predicted mean/std, error (Pred - Target), MAE, and base MAE.

    Parameters
    ----------
    data_original : tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Original measured data (ary_intensity, ary_energy, df_meta).
    data_predicted : tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Predicted data (ary_intensity_pred, ary_energy_pred, df_meta_pred).
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'splits' (list[str]): Splits to evaluate (default ['train', 'test']).
        - 'material' (str | None): Target material (default 'NMG').
        - 'rsf_dict' (dict[str, float]): RSF mapping.
        - 'quantification_config' (dict | None): Extra quantification options.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - df_samples: Detailed DataFrame per (split, measurement, die, element).
        - df_summary: Summary DataFrame aggregated by (split, element).
    """
    ary_intensity_orig, ary_energy_orig, df_meta_orig = data_original
    ary_intensity_pred, ary_energy_pred, df_meta_pred = data_predicted

    cfg = config or {}
    splits_filter = list(cfg.get("splits", ["train", "test"]))
    quant_cfg = dict(cfg.get("quantification_config", {}))
    quant_cfg.setdefault("material", cfg.get("material", "NMG"))
    if "rsf_table" in cfg:
        quant_cfg["rsf_table"] = cfg["rsf_table"]
    elif "rsf_table" not in quant_cfg:
        quant_cfg["rsf_table"] = dict(DEFAULT_SCOFIELD_RSF)

    df_per_die_orig, _ = calculate_atomic_percentages(
        ary_energy_orig, ary_intensity_orig, df_meta_orig, config=quant_cfg
    )
    df_per_die_pred, _ = calculate_atomic_percentages(
        ary_energy_pred, ary_intensity_pred, df_meta_pred, config=quant_cfg
    )

    pred_meta_subset = (
        df_meta_pred[["measurement_id", "source_measurement_id", "split"]]
        .dropna()
        .drop_duplicates()
    )
    src_target_map = (
        df_meta_orig[["measurement_id", "target_measurement_id"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"measurement_id": "source_measurement_id"})
    )
    session_mapping = pred_meta_subset.merge(
        src_target_map, on="source_measurement_id", how="inner"
    )
    if splits_filter is not None:
        session_mapping = session_mapping[
            session_mapping["split"].isin(splits_filter)
        ]

    if session_mapping.empty:
        warnings.warn(
            "No paired sessions found matching requested splits for atomic "
            "percentage statistics."
        )
        return pd.DataFrame(), pd.DataFrame()

    norm_mode = cfg.get("normalization_mode")
    if norm_mode is None:
        norm_mode = df_meta_orig.attrs.get("normalization_mode")
    if norm_mode is None and "normalization_mode" in df_meta_orig.columns:
        norm_series = df_meta_orig["normalization_mode"].dropna()
        if not norm_series.empty:
            norm_mode = str(norm_series.iloc[0])
    if norm_mode in (
        "by_regional_max_intensity_of_source",
        "by_regional_max_intensity",
    ):
        warnings.warn(
            f"Atomic percentage quantification: normalization_mode '{norm_mode}' "
            "normalizes each regional spectrum independently, which distorts "
            "relative atomic percentages across elements.",
            UserWarning,
            stacklevel=2,
        )

    element_cols = [
        c for c in df_per_die_orig.columns if c.endswith("_atomic_percentage")
    ]

    records: list[dict[str, Any]] = []
    for map_row in session_mapping.itertuples():
        p_id = str(map_row.measurement_id)
        s_id = str(map_row.source_measurement_id)
        t_id = str(map_row.target_measurement_id)
        split_val = str(map_row.split)

        sub_p = df_per_die_pred[
            df_per_die_pred["measurement_id"] == p_id
        ].set_index("die")
        sub_s = df_per_die_orig[
            df_per_die_orig["measurement_id"] == s_id
        ].set_index("die")
        sub_t = df_per_die_orig[
            df_per_die_orig["measurement_id"] == t_id
        ].set_index("die")

        common_dies = sorted(
            set(sub_p.index).intersection(sub_s.index).intersection(sub_t.index)
        )
        for die in common_dies:
            for c in element_cols:
                val_s = float(sub_s.loc[die, c])
                val_t = float(sub_t.loc[die, c])
                val_p = float(sub_p.loc[die, c])
                diff_val = val_p - val_t

                records.append({
                    "split": split_val,
                    "element": c[:-len("_atomic_percentage")],
                    "source_measurement_id": s_id,
                    "target_measurement_id": t_id,
                    "predicted_measurement_id": p_id,
                    "die": int(die),
                    "source_atomic_percentage": val_s,
                    "target_atomic_percentage": val_t,
                    "predicted_atomic_percentage": val_p,
                    "diff_atomic_percentage": diff_val,
                    "abs_diff_atomic_percentage": abs(diff_val),
                    "base_abs_diff_atomic_percentage": abs(val_s - val_t),
                })

    df_samples = pd.DataFrame(records)
    if df_samples.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_records: list[dict[str, Any]] = []
    ordered_splits = [
        s for s in splits_filter if s in df_samples["split"].unique()
    ]
    ordered_elements = [
        c[:-len("_atomic_percentage")]
        for c in element_cols
        if c[:-len("_atomic_percentage")] in df_samples["element"].unique()
    ]

    for split in ordered_splits:
        sp_df = df_samples[df_samples["split"] == split]
        for elem in ordered_elements:
            el_df = sp_df[sp_df["element"] == elem]
            if el_df.empty:
                continue

            src_mean, src_std = _mean_std(el_df["source_atomic_percentage"], decimals=3)
            tgt_mean, tgt_std = _mean_std(el_df["target_atomic_percentage"], decimals=3)
            pred_mean, pred_std = _mean_std(el_df["predicted_atomic_percentage"], decimals=3)
            diff_mean, diff_std = _mean_std(el_df["diff_atomic_percentage"], decimals=3)
            mae, mae_std = _mean_std(el_df["abs_diff_atomic_percentage"], decimals=3)
            base_mae = round(float(el_df["base_abs_diff_atomic_percentage"].mean()), 3)

            summary_records.append({
                "split": split,
                "element": elem,
                "n_samples": len(el_df),
                "source_mean": src_mean,
                "source_std": src_std,
                "target_mean": tgt_mean,
                "target_std": tgt_std,
                "predicted_mean": pred_mean,
                "predicted_std": pred_std,
                "diff_mean": diff_mean,
                "diff_std": diff_std,
                "mae": mae,
                "mae_std": mae_std,
                "base_mae": base_mae,
            })

    return df_samples, pd.DataFrame(summary_records)


def format_side_by_side_atomic_percentages(
    df_split_summary: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Format split atomic percentage summary into a side-by-side table.

    Produces a clean comparison table with elements as rows and metrics grouped
    side-by-side across splits (e.g. Train vs. Test).

    Parameters
    ----------
    df_split_summary : pd.DataFrame
        Summary DataFrame from calculate_atomic_percentage_split_statistics.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'splits' (list[str]): Splits to display (default ['train', 'test']).
        - 'format_str' (bool): Format as 'mean ± std%' string (default True).
        - 'include_overall' (bool): Include 'Overall (MAE)' row (default True).

    Returns
    -------
    pd.DataFrame
        Side-by-side comparison table.
    """
    if df_split_summary.empty or "split" not in df_split_summary.columns:
        return pd.DataFrame()

    cfg = dict(config or {})
    cfg.setdefault("include_overall", True)
    cfg.setdefault("overall_metrics", ["mae"])
    cfg.setdefault("overall_label", "Overall (MAE)")

    metric_specs = [
        {
            "col": "target_mean",
            "std_col": "target_std",
            "label": "target",
            "formatter": lambda m, s: f"{m:.2f}% ± {s:.2f}%",
            "num_mean_col": "target_mean_{split}",
            "num_std_col": "target_std_{split}",
        },
        {
            "col": "predicted_mean",
            "std_col": "predicted_std",
            "label": "predicted",
            "formatter": lambda m, s: f"{m:.2f}% ± {s:.2f}%",
            "num_mean_col": "predicted_mean_{split}",
            "num_std_col": "predicted_std_{split}",
        },
        {
            "col": "diff_mean",
            "std_col": "diff_std",
            "label": "diff",
            "formatter": lambda m, s: f"{m:+.2f}% ± {s:.2f}%",
            "num_mean_col": "diff_mean_{split}",
            "num_std_col": "diff_std_{split}",
        },
        {
            "col": "mae",
            "std_col": "mae_std",
            "label": "mae",
            "formatter": lambda m, s: f"{m:.2f}% ± {s:.2f}%",
            "num_mean_col": "mae_{split}",
            "num_std_col": "mae_std_{split}",
        },
    ]

    return format_split_comparison_table(
        df_split_summary,
        group_col="element",
        metric_specs=metric_specs,
        config=cfg,
    )
