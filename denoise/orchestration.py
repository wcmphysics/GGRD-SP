"""Orchestration for spectral noise analysis, denoising, and quantification."""

from __future__ import annotations

import copy
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from denoise.common import (
    _SNR_DEFINITIONS,
    _VALID_APPLY_TO,
    _VALID_METHODS,
    _select_denoise_rows,
    _select_spectra,
)
from denoise.fourier import (
    _resolve_fourier_config,
    denoise_batch_fourier,
    denoise_spectrum_fourier,
    get_default_fourier_config,
)
from denoise.gp import (
    _resolve_gp_config,
    denoise_batch_gp,
    denoise_spectrum_gp,
    get_default_gp_denoise_config,
)
from denoise.savgol import (
    _resolve_savgol_config,
    denoise_batch_savgol,
    denoise_spectrum_savgol,
    get_default_savgol_config,
)
from denoise.wavelet import (
    _resolve_wavelet_config,
    denoise_batch_wavelet,
    denoise_spectrum_wavelet,
    get_default_wavelet_config,
)
from utility.quantification import (
    DEFAULT_SCOFIELD_RSF,
    calculate_atomic_percentages,
)

_NEW_META_COLUMNS: tuple[str, ...] = (
    "noise_analysis_index",
    "gp_noise_sigma",
    "gp_snr",
    "gp_lengthscale",
)


def get_default_noise_analysis_config() -> dict[str, Any]:
    """Return default configuration for ``perform_spectral_noise_analysis``.

    Returns
    -------
    dict[str, Any]
        Default noise analysis configuration dictionary supporting GP and
        deterministic filtering methods ('gp', 'savgol', 'wavelet', 'fourier').
    """
    gp_defaults = get_default_gp_denoise_config()
    return {
        "method": "gp",
        "splits": None,
        "dies": None,
        "regions": None,
        "unpaired_split": None,
        **gp_defaults,
        "savgol": get_default_savgol_config(),
        "wavelet": get_default_wavelet_config(),
        "fourier": get_default_fourier_config(),
    }


def _resolve_noise_analysis_config(
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate and merge user noise analysis config over defaults."""
    if config is not None and config.get("_resolved"):
        return config
    cfg = get_default_noise_analysis_config()
    user_cfg = copy.deepcopy(config or {})
    for key in user_cfg:
        if key not in cfg:
            raise ValueError(
                f"Unknown noise analysis config key '{key}'. "
                f"Valid keys: {sorted(cfg)}"
            )
    for key in ("splits", "dies", "regions"):
        val = user_cfg.get(key)
        if val is not None and (isinstance(val, str) or len(val) == 0):
            raise ValueError(
                f"'{key}' must be None or a non-empty list, got {val!r}."
            )
    method = str(user_cfg.get("method", cfg["method"])).lower()
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Invalid noise analysis method={method!r}; "
            f"choose one of {_VALID_METHODS}."
        )
    for sub_method, resolver in (
        ("savgol", _resolve_savgol_config),
        ("wavelet", _resolve_wavelet_config),
        ("fourier", _resolve_fourier_config),
    ):
        if sub_method in user_cfg:
            sub_dict = user_cfg.pop(sub_method)
            if not isinstance(sub_dict, dict):
                raise ValueError(
                    f"'{sub_method}' config must be a dict, got {sub_dict!r}."
                )
            cfg[sub_method] = resolver(sub_dict)

    gp_resolved = _resolve_gp_config(user_cfg)
    cfg.update(gp_resolved)
    cfg["method"] = method
    cfg["_resolved"] = True
    return cfg


def _init_worker() -> None:
    """Limit torch to one thread in worker processes."""
    torch.set_num_threads(1)


def _analyse_chunk_by_method(
    energies: np.ndarray,
    intensities: np.ndarray,
    metas: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch a chunk of spectra to the configured noise analysis method."""
    method = cfg["method"]
    if method == "gp":
        return denoise_batch_gp(energies, intensities, metas, cfg)
    sub_cfg = dict(cfg.get(method, {}))
    sub_cfg.pop("_resolved", None)
    sub_cfg["snr_definition"] = cfg["snr_definition"]
    sub_cfg["confidence_z"] = cfg["confidence_z"]
    if method == "savgol":
        return denoise_batch_savgol(energies, intensities, metas, sub_cfg)
    if method == "wavelet":
        return denoise_batch_wavelet(energies, intensities, metas, sub_cfg)
    return denoise_batch_fourier(energies, intensities, metas, sub_cfg)


def perform_spectral_noise_analysis(
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Estimate noise and SNR of regional spectra using configured technique.

    Parameters
    ----------
    ary_intensity, ary_energy : np.ndarray
        Canonical 2D arrays, one regional spectrum per row.
    df_meta : pd.DataFrame
        Canonical meta data (needs ``spectrum_index``, ``die``, ``region`` and
        normally ``split``). It is not modified.
    config : dict[str, Any] | None
        See ``get_default_noise_analysis_config``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame]
        ``ary_spec_analysis_signal``, ``ary_spec_analysis_noise``, both of
        shape (n_analysed, N_points), and a copy of ``df_meta`` with columns
        ``noise_analysis_index``, ``gp_noise_sigma``, ``gp_snr`` and
        ``gp_lengthscale``.
    """
    cfg = _resolve_noise_analysis_config(config)
    if ary_intensity.shape != ary_energy.shape or ary_intensity.ndim != 2:
        raise ValueError(
            f"ary_intensity {ary_intensity.shape} and ary_energy "
            f"{ary_energy.shape} must be 2D arrays of the same shape."
        )
    selected = _select_spectra(df_meta, cfg)
    spectrum_indices = selected["spectrum_index"].to_numpy()
    if (
        spectrum_indices.min() < 0
        or spectrum_indices.max() >= ary_intensity.shape[0]
    ):
        raise ValueError(
            f"spectrum_index range [{spectrum_indices.min()}, "
            f"{spectrum_indices.max()}] is outside the arrays "
            f"({ary_intensity.shape[0]} rows)."
        )

    n_rows, n_points = len(selected), ary_intensity.shape[1]
    ary_signal = np.empty((n_rows, n_points), dtype=np.float64)
    ary_noise = np.empty((n_rows, n_points), dtype=np.float64)
    summary = np.empty((n_rows, 3), dtype=np.float64)
    metas = selected.to_dict("records")
    workers = cfg["num_workers"]
    if cfg["batch_size"] > 1:
        chunk_size = cfg["batch_size"]
    elif workers > 1:
        chunk_size = math.ceil(n_rows / (workers * 4))
    else:
        chunk_size = 1
    chunks = [
        (start, min(start + chunk_size, n_rows))
        for start in range(0, n_rows, chunk_size)
    ]

    def chunk_args(start: int, stop: int) -> tuple[Any, ...]:
        rows = spectrum_indices[start:stop]
        return ary_energy[rows], ary_intensity[rows], metas[start:stop], cfg

    desc = f"{cfg['method'].upper()} noise analysis"
    with tqdm(
        total=n_rows, desc=desc, disable=not cfg["show_progress"]
    ) as bar:
        if workers > 1:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_init_worker
            ) as pool:
                futures = {
                    pool.submit(_analyse_chunk_by_method, *chunk_args(*c)): c
                    for c in chunks
                }
                try:
                    for future in as_completed(futures):
                        start, stop = futures[future]
                        (
                            ary_signal[start:stop],
                            ary_noise[start:stop],
                            summary[start:stop],
                        ) = future.result()
                        bar.update(stop - start)
                except BaseException:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
        else:
            for start, stop in chunks:
                result = _analyse_chunk_by_method(*chunk_args(start, stop))
                (
                    ary_signal[start:stop],
                    ary_noise[start:stop],
                    summary[start:stop],
                ) = result
                bar.update(stop - start)

    df_out = df_meta.copy()
    df_out["noise_analysis_index"] = pd.array(
        [pd.NA] * len(df_out), dtype="Int64"
    )
    for column in _NEW_META_COLUMNS[1:]:
        df_out[column] = np.nan
    df_out.loc[selected.index, "noise_analysis_index"] = np.arange(n_rows)
    for column, values in zip(_NEW_META_COLUMNS[1:], summary.T):
        df_out.loc[selected.index, column] = values
    return ary_signal, ary_noise, df_out


def get_default_denoise_config() -> dict[str, Any]:
    """Return default configuration for ``denoise_regional_spectra``.

    Returns
    -------
    dict[str, Any]
        Default denoising configuration dictionary:
        - 'apply_denoise': Whether downstream pipeline uses denoised data
          (default False).
        - 'method': Default denoising technique ('savgol', 'wavelet',
          'fourier', or 'gp'; default 'savgol').
        - 'regions': Regions to denoise (default ['Cl2p', 'Ti2p']).
        - 'apply_to': Tool scope ('both', 'source', or 'target';
          default 'both').
        - 'source_tool': Source tool name (default 'J4').
        - 'target_tool': Target tool name (default 'J5').
        - 'material': Material name for quantification (default 'NMG').
        - 'rsf_table': Relative sensitivity factors dictionary.
        - 'snr_definition': 'range' or 'std' (default 'range').
        - 'confidence_z': Confidence band multiplier (default 1.96).
        - 'savgol': Default Savitzky-Golay parameters.
        - 'wavelet': Default Wavelet shrinkage parameters.
        - 'fourier': Default Fourier/Butterworth low-pass parameters.
        - 'gp': Default Gaussian Process parameters.
        - 'region_configs': Optional per-region override dictionary.
    """
    return {
        "apply_denoise": False,
        "method": "savgol",
        "regions": ["Cl2p", "Ti2p"],
        "apply_to": "both",
        "source_tool": "J4",
        "target_tool": "J5",
        "material": "NMG",
        "rsf_table": dict(DEFAULT_SCOFIELD_RSF),
        "snr_definition": "range",
        "confidence_z": 1.96,
        "savgol": get_default_savgol_config(),
        "wavelet": get_default_wavelet_config(),
        "fourier": get_default_fourier_config(),
        "gp": get_default_gp_denoise_config(),
        "region_configs": {},
    }


def _resolve_denoise_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and merge user denoising configuration over defaults."""
    if config is not None and config.get("_resolved"):
        return config
    cfg = get_default_denoise_config()
    user_cfg = copy.deepcopy(config or {})
    for key in user_cfg:
        if key not in cfg:
            raise ValueError(
                f"Unknown denoise config key '{key}'. "
                f"Valid keys: {sorted(cfg)}"
            )
    if not isinstance(user_cfg.get("apply_denoise", False), bool):
        raise ValueError(
            f"apply_denoise must be a boolean, "
            f"got {user_cfg.get('apply_denoise')!r}."
        )
    method = str(user_cfg.get("method", cfg["method"])).lower()
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Invalid denoise method={method!r}; "
            f"choose one of {_VALID_METHODS}."
        )
    apply_to = str(user_cfg.get("apply_to", cfg["apply_to"])).lower()
    if apply_to not in _VALID_APPLY_TO:
        raise ValueError(
            f"Invalid apply_to={apply_to!r}; choose one of {_VALID_APPLY_TO}."
        )
    snr_def = str(user_cfg.get("snr_definition", cfg["snr_definition"]))
    if snr_def not in _SNR_DEFINITIONS:
        raise ValueError(
            f"Invalid snr_definition={snr_def!r}; "
            f"choose one of {_SNR_DEFINITIONS}."
        )
    for sub_name, resolver in (
        ("savgol", _resolve_savgol_config),
        ("wavelet", _resolve_wavelet_config),
        ("fourier", _resolve_fourier_config),
        ("gp", _resolve_gp_config),
    ):
        if sub_name in user_cfg:
            sub_val = user_cfg.pop(sub_name)
            if not isinstance(sub_val, dict):
                raise ValueError(
                    f"'{sub_name}' config must be a dict, got {sub_val!r}."
                )
            cfg[sub_name] = resolver(sub_val)

    region_configs = user_cfg.pop("region_configs", None)
    if region_configs is not None:
        if not isinstance(region_configs, dict):
            raise ValueError(
                f"'region_configs' must be a dict, got {region_configs!r}."
            )
        cfg["region_configs"] = region_configs

    cfg.update(user_cfg)
    cfg["method"] = method
    cfg["apply_to"] = apply_to
    cfg["_resolved"] = True
    return cfg


def _resolve_region_denoise_spec(
    region: str,
    cfg: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Resolve the active method and method config for a specific region."""
    reg_override = (cfg.get("region_configs") or {}).get(region) or {}
    if not isinstance(reg_override, dict):
        raise ValueError(
            f"region_configs[{region!r}] must be a dict, got {reg_override!r}."
        )
    allowed_reg_keys = {"method", "savgol", "wavelet", "fourier", "gp"}
    extra_reg_keys = set(reg_override) - allowed_reg_keys
    if extra_reg_keys:
        raise ValueError(
            f"Unknown keys {sorted(extra_reg_keys)} in "
            f"region_configs[{region!r}]. Allowed: {sorted(allowed_reg_keys)}."
        )
    method = str(reg_override.get("method", cfg["method"])).lower()
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Invalid method={method!r} for region {region!r}; "
            f"choose one of {_VALID_METHODS}."
        )
    base_method_cfg = copy.deepcopy(cfg[method])
    base_method_cfg.pop("_resolved", None)
    if method in reg_override:
        if not isinstance(reg_override[method], dict):
            raise ValueError(
                f"region_configs[{region!r}][{method!r}] must be a dict."
            )
        if method == "gp":
            for nested in ("kernel", "training"):
                if nested in reg_override[method]:
                    base_method_cfg[nested] = {
                        **base_method_cfg.get(nested, {}),
                        **reg_override[method][nested],
                    }
            other_gp = {
                k: v
                for k, v in reg_override[method].items()
                if k not in ("kernel", "training")
            }
            base_method_cfg.update(other_gp)
        else:
            base_method_cfg.update(reg_override[method])

    base_method_cfg["snr_definition"] = cfg["snr_definition"]
    base_method_cfg["confidence_z"] = cfg["confidence_z"]
    resolvers = {
        "savgol": _resolve_savgol_config,
        "wavelet": _resolve_wavelet_config,
        "fourier": _resolve_fourier_config,
        "gp": _resolve_gp_config,
    }
    return method, resolvers[method](base_method_cfg)


def denoise_single_spectrum(
    energy: np.ndarray,
    intensity: np.ndarray,
    config: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Denoise one 1D regional spectrum using the configured method.

    Parameters
    ----------
    energy : np.ndarray
        1D array of binding energies in eV.
    intensity : np.ndarray
        1D array of measured intensities.
    config : dict[str, Any] | None, optional
        Denoising configuration dictionary.
    meta : dict[str, Any] | None, optional
        Optional metadata dictionary for error reporting.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, tuple[float, float, float]]
        Tuple of (denoised_signal, confidence_curve, summary_tuple).
    """
    cfg = _resolve_denoise_config(config)
    region = str((meta or {}).get("region", ""))
    method, method_cfg = _resolve_region_denoise_spec(region, cfg)
    dispatch = {
        "savgol": denoise_spectrum_savgol,
        "wavelet": denoise_spectrum_wavelet,
        "fourier": denoise_spectrum_fourier,
        "gp": denoise_spectrum_gp,
    }
    return dispatch[method](energy, intensity, method_cfg, meta=meta)


def denoise_regional_spectra(
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Denoise configured regional spectra and return active and raw copies.

    Deep-copies ``ary_intensity`` before updating so the original non-denoised
    spectra remain intact for downstream comparison and visualization.

    Parameters
    ----------
    ary_intensity : np.ndarray
        2D array of measured regional intensities of shape (N_total, N_points).
    ary_energy : np.ndarray
        2D array of binding energies in eV of shape (N_total, N_points).
    df_meta : pd.DataFrame
        Metadata DataFrame containing 'spectrum_index', 'tool', 'die', and
        'region'.
    config : dict[str, Any] | None, optional
        Denoising configuration dictionary (see get_default_denoise_config).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of:
        - ary_intensity_active: Updated denoised array if apply_denoise is True,
          otherwise a copy of the non-denoised array.
        - ary_intensity_non_denoised: Deep copy of original non-denoised array.
        - ary_intensity_denoised: Array with configured regions denoised and
          all other spectra unchanged.
    """
    cfg = _resolve_denoise_config(config)
    if ary_intensity.shape != ary_energy.shape or ary_intensity.ndim != 2:
        raise ValueError(
            f"ary_intensity {ary_intensity.shape} and ary_energy "
            f"{ary_energy.shape} must be 2D arrays of the same shape."
        )

    ary_intensity_non_denoised = copy.deepcopy(ary_intensity)
    ary_intensity_denoised = copy.deepcopy(ary_intensity)

    selected = _select_denoise_rows(df_meta, cfg)
    if selected.empty:
        return (
            copy.deepcopy(ary_intensity_non_denoised),
            ary_intensity_non_denoised,
            ary_intensity_denoised,
        )

    spectrum_indices = selected["spectrum_index"].to_numpy()
    if (
        spectrum_indices.min() < 0
        or spectrum_indices.max() >= ary_intensity.shape[0]
    ):
        raise ValueError(
            f"spectrum_index range [{spectrum_indices.min()}, "
            f"{spectrum_indices.max()}] is outside the arrays "
            f"({ary_intensity.shape[0]} rows)."
        )

    batch_dispatch = {
        "savgol": denoise_batch_savgol,
        "wavelet": denoise_batch_wavelet,
        "fourier": denoise_batch_fourier,
        "gp": denoise_batch_gp,
    }

    for region_name, df_region in selected.groupby("region", sort=False):
        method, method_cfg = _resolve_region_denoise_spec(
            str(region_name), cfg
        )
        reg_indices = df_region["spectrum_index"].to_numpy(dtype=int)
        reg_metas = df_region.to_dict("records")
        denoised_signals, _, _ = batch_dispatch[method](
            ary_energy[reg_indices],
            ary_intensity_non_denoised[reg_indices],
            reg_metas,
            method_cfg,
        )
        ary_intensity_denoised[reg_indices] = denoised_signals.astype(
            ary_intensity_denoised.dtype
        )

    if cfg["apply_denoise"]:
        ary_intensity_active = copy.deepcopy(ary_intensity_denoised)
    else:
        ary_intensity_active = copy.deepcopy(ary_intensity_non_denoised)

    return (
        ary_intensity_active,
        ary_intensity_non_denoised,
        ary_intensity_denoised,
    )


def compare_denoised_atomic_percentages(
    ary_energy: np.ndarray,
    intensity_pair: tuple[np.ndarray, np.ndarray],
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare atomic percentages between non-denoised and denoised spectra.

    Calculates Shirley background-subtracted atomic percentages on both raw
    non-denoised and denoised intensity arrays and evaluates sample-level
    absolute errors and regional Mean Absolute Error (MAE).

    Parameters
    ----------
    ary_energy : np.ndarray
        2D array of binding energies in eV of shape (N_total, N_points).
    intensity_pair : tuple[np.ndarray, np.ndarray]
        Tuple of (ary_intensity_non_denoised, ary_intensity_denoised).
    df_meta : pd.DataFrame
        Metadata DataFrame describing all spectra.
    config : dict[str, Any] | None, optional
        Denoising / quantification configuration dictionary.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - df_samples: Per-(measurement_id, die, region) comparison records with
          'non_denoised_atomic_percentage', 'denoised_atomic_percentage',
          'diff_atomic_percentage', and 'abs_diff_atomic_percentage'.
        - df_summary: Per-region summary table with mean and std of MAE.
    """
    if not (isinstance(intensity_pair, tuple) and len(intensity_pair) == 2):
        raise ValueError(
            "intensity_pair must be a 2-tuple of "
            "(ary_intensity_non_denoised, ary_intensity_denoised)."
        )
    ary_non_denoised, ary_denoised = intensity_pair
    cfg = dict(config or {})
    apply_to = str(cfg.get("apply_to", "both")).lower()
    source_tool = str(cfg.get("source_tool", "J4"))
    target_tool = str(cfg.get("target_tool", "J5"))
    if apply_to == "source":
        allowed_tools = {source_tool}
    elif apply_to == "target":
        allowed_tools = {target_tool}
    else:
        allowed_tools = {source_tool, target_tool}

    df_filtered = df_meta[
        df_meta["tool"].astype(str).isin(allowed_tools)
    ].copy()
    if df_filtered.empty:
        raise ValueError(
            f"No spectra found in df_meta for tools {sorted(allowed_tools)}."
        )

    quant_cfg: dict[str, Any] = {
        "material": cfg.get("material", "NMG"),
        "rsf_table": cfg.get("rsf_table") or dict(DEFAULT_SCOFIELD_RSF),
    }
    if isinstance(cfg.get("quantification"), dict):
        quant_cfg.update(cfg["quantification"])

    df_raw_die, _ = calculate_atomic_percentages(
        ary_energy, ary_non_denoised, df_filtered, config=quant_cfg
    )
    df_den_die, _ = calculate_atomic_percentages(
        ary_energy, ary_denoised, df_filtered, config=quant_cfg
    )

    session_info = (
        df_filtered[["measurement_id", "tool"] + (["split"] if "split" in df_filtered.columns else [])]
        .drop_duplicates(subset=["measurement_id"])
        .set_index("measurement_id")
    )

    element_cols = [
        col
        for col in df_raw_die.columns
        if col.endswith("_atomic_percentage")
    ]
    merged = df_raw_die.merge(
        df_den_die,
        on=["measurement_id", "material", "die"],
        suffixes=("_non_denoised", "_denoised"),
    )

    sample_records: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        meas_id = str(row["measurement_id"])
        tool_val = str(session_info.loc[meas_id, "tool"])
        split_val = (
            str(session_info.loc[meas_id, "split"])
            if "split" in session_info.columns
            and pd.notna(session_info.loc[meas_id, "split"])
            else "unpaired"
        )
        for col in element_cols:
            region_name = col[: -len("_atomic_percentage")]
            val_raw = float(row[f"{col}_non_denoised"])
            val_den = float(row[f"{col}_denoised"])
            diff_val = val_den - val_raw
            sample_records.append(
                {
                    "measurement_id": meas_id,
                    "material": str(row["material"]),
                    "tool": tool_val,
                    "split": split_val,
                    "die": int(row["die"]),
                    "region": region_name,
                    "element": region_name,
                    "non_denoised_atomic_percentage": val_raw,
                    "denoised_atomic_percentage": val_den,
                    "diff_atomic_percentage": diff_val,
                    "abs_diff_atomic_percentage": abs(diff_val),
                }
            )

    df_samples = pd.DataFrame(sample_records)
    if df_samples.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_records: list[dict[str, Any]] = []
    for region_name, grp in df_samples.groupby("region", sort=False):
        abs_err = grp["abs_diff_atomic_percentage"]
        diff_err = grp["diff_atomic_percentage"]
        raw_vals = grp["non_denoised_atomic_percentage"]
        den_vals = grp["denoised_atomic_percentage"]
        ddof = 1 if len(grp) > 1 else 0
        summary_records.append(
            {
                "region": str(region_name),
                "n_samples": len(grp),
                "non_denoised_mean": round(float(raw_vals.mean()), 4),
                "non_denoised_std": round(float(raw_vals.std(ddof=ddof)), 4),
                "denoised_mean": round(float(den_vals.mean()), 4),
                "denoised_std": round(float(den_vals.std(ddof=ddof)), 4),
                "diff_mean": round(float(diff_err.mean()), 4),
                "diff_std": round(float(diff_err.std(ddof=ddof)), 4),
                "mae": round(float(abs_err.mean()), 4),
                "mae_std": round(float(abs_err.std(ddof=ddof)), 4),
            }
        )

    return df_samples, pd.DataFrame(summary_records)
