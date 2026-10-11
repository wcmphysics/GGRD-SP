"""Inference and standardized prediction utilities for spectral
transfer.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.film import prepare_film_condition_tensor
from utility.patching import extract_sliding_windows, reconstruct_from_patches
from utility.quantification import compute_die_total_flux


def _repeat_cond(c_dict: dict[str, torch.Tensor] | None, n: int) -> dict[str, torch.Tensor] | None:
    if not c_dict: return None
    return {k: v.repeat(n, *([1] * (v.dim() - 1))) if isinstance(v, torch.Tensor) else v for k, v in c_dict.items()}


def predict_sliding_window_spectrum(
    model: nn.Module,
    spectrum: np.ndarray,
    config: dict[str, Any] | None = None,
    cond: dict[str, torch.Tensor] | None = None,
) -> np.ndarray:
    """Transform a full 1D spectrum using a trained sliding window
    patch model.
    """
    cfg = config or {}
    w_size = int(cfg.get("window_size", 15))
    s_step = int(cfg.get("stride", max(1, w_size // 2)))
    norm_mode = str(cfg.get("normalization_mode", "source_referenced")).lower()
    clamp_non_negative, eps = bool(cfg.get("clamp_non_negative", True)), float(cfg.get("eps", 1e-4))
    device = torch.device(cfg.get("device") or next((p.device for p in model.parameters()), "cpu"))
    bs = int(cfg.get("batch_size", 64))

    spectrum_1d = np.asarray(spectrum).ravel()
    s_val = cfg.get("scale_factor", np.abs(spectrum_1d).max() if norm_mode == "source_referenced" else 1.0)
    scale_x = max(float(s_val), eps) if (norm_mode != "none" or "scale_factor" in cfg) else 1.0

    windows, start_indices = extract_sliding_windows(spectrum_1d / scale_x, window_size=w_size, stride=s_step)
    model.eval()
    with torch.no_grad():
        patches = [
            (model(b, cond=_repeat_cond(cond, len(b))) if cond is not None else model(b)).cpu().numpy()
            for i in range(0, len(windows), bs)
            for b in [torch.as_tensor(windows[i : i + bs], dtype=torch.float32, device=device)]
        ]

    recon = reconstruct_from_patches(np.vstack(patches), start_indices, len(spectrum_1d), {"window_size": w_size})
    return np.clip(recon * scale_x, 0.0, None) if clamp_non_negative else recon * scale_x


def format_predicted_measurement_id(source_measurement_id: str, source_tool: str, target_tool: str) -> str:
    """Format a predicted measurement session ID uniquely and
    deterministically.
    """
    p, tag = f"{source_tool}{target_tool}", f"{source_tool}_"
    mat, _, rest = source_measurement_id.rpartition("_M_")
    return f"{mat}_P_{p}_{rest.removeprefix(tag)}" if mat else f"P_{p}_{rest.removeprefix('M_').removeprefix(tag)}"


def assemble_prediction_metadata(
    source_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build standardized prediction metadata DataFrame for transferred
    spectra.
    """
    cfg = config or {}
    source_tool, target_tool = str(cfg.get("source_tool", "")), str(cfg.get("target_tool", ""))
    n_points, session_splits = int(cfg.get("n_points", 0)), cfg.get("session_splits")

    def _split(mid: str, cur: Any) -> str:
        if not session_splits: return str(cur) if pd.notna(cur) else "unassigned"
        return next((s for s in ("train", "val", "test") if mid in session_splits.get(s, set())), "unpaired")

    return pd.DataFrame([
        {
            "spectrum_index": idx, "material": getattr(r, "material", "NMG"), "tool": target_tool,
            "measurement_id": format_predicted_measurement_id(str(r.measurement_id), source_tool, target_tool),
            "die": getattr(r, "die", 0), "region": getattr(r, "region", None), "n_points": n_points,
            "time": getattr(r, "time", None), "source_tool": source_tool,
            "source_measurement_id": str(r.measurement_id), "source_spectrum_index": int(r.spectrum_index),
            "split": _split(str(r.measurement_id), getattr(r, "split", None)), "is_predicted": True,
            **({"t7_code": r.t7_code} if hasattr(r, "t7_code") else {}),
        }
        for idx, r in enumerate(source_df.itertuples())
    ])


def predict_spectra(
    models: dict[str, nn.Module],
    data: tuple[np.ndarray, np.ndarray, pd.DataFrame] | dict[str, Any],
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Apply trained models to source spectra to generate predicted
    target spectra.
    """
    if not isinstance(data, (tuple, list, dict)): raise TypeError(f"Expected tuple/dict, got {type(data)}")
    d_vals = (data["ary_intensity"], data["ary_energy"], data["meta_df"]) if isinstance(data, dict) else data[:3]
    ary_intensity, ary_energy, meta_df = d_vals

    cfg = config or {}
    source_tool, target_tool = cfg.get("source_tool", "J4"), cfg.get("target_tool", "H1")
    measurement_ids = cfg.get("measurement_ids")
    batch_size = int(cfg.get("batch_size", 64))
    norm_mode = str(cfg.get("normalization_mode", "source_referenced")).lower()
    clamp_non_negative, eps = bool(cfg.get("clamp_non_negative", True)), float(cfg.get("eps", 1e-4))
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _film_cond(m: nn.Module, sub_df: pd.DataFrame, scale: np.ndarray, dev: torch.device) -> dict[str, Any] | None:
        if not getattr(m, "use_film", False): return None
        meta = [{**r, "tool_target": target_tool} for r in sub_df.to_dict("records")]
        return prepare_film_condition_tensor(metadata=meta, scale_x=scale, device=dev, num_samples=len(sub_df))

    mask = meta_df["tool"] == source_tool
    if measurement_ids: mask = mask & meta_df["measurement_id"].isin(measurement_ids)
    source_df = meta_df[mask].copy()
    if source_df.empty: raise ValueError(f"No source spectra found for {source_tool=}")

    for m in models.values():
        mod = m[0] if isinstance(m, (tuple, list)) else m
        if isinstance(mod, nn.Module): mod.eval().to(device)

    predicted_intensities = np.zeros((len(source_df), ary_intensity.shape[1]), dtype=np.float32)
    predicted_energies = np.array(ary_energy[source_df["spectrum_index"].astype(int).to_numpy()], dtype=np.float32)
    flux_map = compute_die_total_flux(source_df, ary_intensity, ary_energy) if norm_mode == "die_total_flux" else {}

    for region, group in source_df.groupby("region"):
        group_new_indices = np.flatnonzero((source_df["region"] == region).to_numpy())
        orig_indices = group["spectrum_index"].astype(int).to_numpy()

        if region not in models:
            warnings.warn(f"Region '{region}' not in models dictionary; falling back to identity pass-through.")
            predicted_intensities[group_new_indices] = ary_intensity[orig_indices]
            continue

        model_entry, region_x = models[str(region)], ary_intensity[orig_indices]

        if norm_mode == "die_total_flux" and flux_map:
            def _fx(r: Any) -> float:
                fb = max(float(np.abs(ary_intensity[int(r.spectrum_index)]).max()), eps)
                return float(flux_map.get((str(r.measurement_id), int(r.die)), fb))
            scales = np.array([_fx(r) for r in group.itertuples()], dtype=np.float32)[:, None]
        elif norm_mode == "source_referenced":
            scales = np.maximum(np.max(np.abs(region_x), axis=-1, keepdims=True), eps)
        else:
            scales = np.ones((len(region_x), 1), dtype=np.float32)
        region_x_in = region_x / scales

        is_sw = bool(cfg.get("use_sliding_window", False)) or isinstance(model_entry, (tuple, list))

        if is_sw:
            is_seq = isinstance(model_entry, (tuple, list))
            m_obj = model_entry[0] if is_seq else model_entry
            w_size = int(model_entry[1]) if is_seq else int(cfg.get("window_size", 15))
            s_step = int(model_entry[2]) if is_seq else int(cfg.get("stride", max(1, w_size // 2)))
            m_dev = next(m_obj.parameters()).device if list(m_obj.parameters()) else device
            sw_cfg = dict(window_size=w_size, stride=s_step, device=m_dev, normalization_mode=norm_mode, eps=eps)
            predicted_intensities[group_new_indices] = np.vstack([
                predict_sliding_window_spectrum(
                    m_obj, x_s, config={**sw_cfg, "scale_factor": float(scales[i, 0])},
                    cond=_film_cond(m_obj, group.iloc[i : i + 1], scales[i : i + 1], m_dev),
                )
                for i, x_s in enumerate(region_x)
            ])
        else:
            with torch.no_grad():
                batches = [
                    (torch.as_tensor(region_x_in[i : i + batch_size], dtype=torch.float32, device=device),
                     _film_cond(model_entry, group.iloc[i : i + batch_size], scales[i : i + batch_size], device))
                    for i in range(0, len(region_x_in), batch_size)
                ]
                preds = [
                    (model_entry(bx, cond=c) if c is not None else model_entry(bx)).cpu().numpy()
                    for bx, c in batches
                ]
            predicted_intensities[group_new_indices] = np.vstack(preds) * scales

    if clamp_non_negative:
        predicted_intensities = np.clip(predicted_intensities, 0.0, None)
    meta_cfg = dict(source_tool=source_tool, target_tool=target_tool, session_splits=cfg.get("session_splits"))
    meta_df_predicted = assemble_prediction_metadata(source_df, {**meta_cfg, "n_points": ary_intensity.shape[1]})
    return predicted_intensities, predicted_energies, meta_df_predicted
