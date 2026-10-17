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


def _repeat_cond(
    c_dict: dict[str, torch.Tensor] | None, n: int
) -> dict[str, torch.Tensor] | None:
    if not c_dict:
        return None
    return {
        k: (
            v.repeat(n, *([1] * (v.dim() - 1)))
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in c_dict.items()
    }


def predict_sliding_window_spectrum(
    model: nn.Module,
    spectrum: np.ndarray,
    config: dict[str, Any] | None = None,
    cond: dict[str, torch.Tensor] | None = None,
    energy: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    """Transform a full 1D spectrum using a trained sliding window
    patch model.
    """
    cfg = config or {}
    w_size = int(cfg.get("window_size", 15))
    s_step = int(cfg.get("stride", max(1, w_size // 2)))
    clamp_non_negative = bool(cfg.get("clamp_non_negative", True))
    device = torch.device(
        cfg.get("device") or next((p.device for p in model.parameters()), "cpu")
    )
    bs = int(cfg.get("batch_size", 64))

    spectrum_1d = np.asarray(spectrum).ravel()
    windows, start_indices = extract_sliding_windows(
        spectrum_1d, window_size=w_size, stride=s_step
    )
    film_gen = getattr(model, "film_gen", None)
    encode_energy = bool(getattr(film_gen, "encode_window_energy", False))
    energy_scale_ev = float(getattr(model, "energy_scale_ev", 1000.0))

    energy_np = None
    if energy is not None:
        energy_np = (
            energy.cpu().numpy().ravel()
            if isinstance(energy, torch.Tensor)
            else np.asarray(energy).ravel()
        )

    model.eval()
    with torch.no_grad():
        patches = []
        for i in range(0, len(windows), bs):
            b_wins = torch.as_tensor(
                windows[i : i + bs], dtype=torch.float32, device=device
            ).unsqueeze(1)
            b_cond = _repeat_cond(cond, len(b_wins)) if cond is not None else None
            if b_cond is not None and encode_energy:
                if energy_np is not None:
                    bounds = [
                        [
                            float(energy_np[s]),
                            float(
                                energy_np[min(s + w_size - 1, len(energy_np) - 1)]
                            ),
                        ]
                        for s in start_indices[i : i + bs]
                    ]
                else:
                    bounds = [[0.0, 0.0]] * len(b_wins)
                scale_floor = max(energy_scale_ev, 1e-6)
                b_cond["window_energy"] = (
                    torch.tensor(bounds, dtype=torch.float32, device=device)
                    / scale_floor
                )
            out = (
                model(b_wins, cond=b_cond)
                if b_cond is not None
                else model(b_wins)
            )
            patches.append(out.squeeze(1).cpu().numpy())

    recon = reconstruct_from_patches(
        np.vstack(patches),
        start_indices,
        len(spectrum_1d),
        {"window_size": w_size},
    )
    return np.clip(recon, 0.0, None) if clamp_non_negative else recon


def format_predicted_measurement_id(
    source_measurement_id: str, source_tool: str, target_tool: str
) -> str:
    """Format a predicted measurement session ID uniquely and
    deterministically.
    """
    pair_tag, src_prefix = f"{source_tool}{target_tool}", f"{source_tool}_"
    mat, _, rest = source_measurement_id.rpartition("_M_")
    if mat:
        return f"{mat}_P_{pair_tag}_{rest.removeprefix(src_prefix)}"
    return f"P_{pair_tag}_{rest.removeprefix('M_').removeprefix(src_prefix)}"


def assemble_prediction_metadata(
    source_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build standardized prediction metadata DataFrame for transferred
    spectra.
    """
    cfg = config or {}
    source_tool = str(cfg.get("source_tool", ""))
    target_tool = str(cfg.get("target_tool", ""))
    n_points = int(cfg.get("n_points", 0))
    session_splits = cfg.get("session_splits")

    def _resolve_split(mid: str, cur: Any) -> str:
        if not session_splits:
            return str(cur) if pd.notna(cur) else "unassigned"
        return next(
            (
                s
                for s in ("train", "val", "test")
                if mid in session_splits.get(s, set())
            ),
            "unpaired",
        )

    return pd.DataFrame(
        [
            {
                "spectrum_index": idx,
                "material": getattr(r, "material", "NMG"),
                "tool": target_tool,
                "measurement_id": format_predicted_measurement_id(
                    str(r.measurement_id), source_tool, target_tool
                ),
                "die": getattr(r, "die", 0),
                "region": getattr(r, "region", None),
                "n_points": n_points,
                "time": getattr(r, "time", None),
                "source_tool": source_tool,
                "source_measurement_id": str(r.measurement_id),
                "source_spectrum_index": int(r.spectrum_index),
                "split": _resolve_split(
                    str(r.measurement_id), getattr(r, "split", None)
                ),
                "is_predicted": True,
                **({"t7_code": r.t7_code} if hasattr(r, "t7_code") else {}),
                **(
                    {"die_flux_normalized": r.die_flux_normalized}
                    if hasattr(r, "die_flux_normalized")
                    else {}
                ),
                **(
                    {"time_normalized": r.time_normalized}
                    if hasattr(r, "time_normalized")
                    else {}
                ),
            }
            for idx, r in enumerate(source_df.itertuples())
        ]
    )


def predict_spectra(
    models: dict[str, nn.Module],
    data: tuple[np.ndarray, np.ndarray, pd.DataFrame] | dict[str, Any],
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Apply trained models to source spectra to generate predicted
    target spectra.
    """
    if not isinstance(data, (tuple, list, dict)):
        raise TypeError(f"Expected tuple/dict, got {type(data)}")
    ary_intensity, ary_energy, df_meta = (
        (data["ary_intensity"], data["ary_energy"], data["df_meta"])
        if isinstance(data, dict)
        else data[:3]
    )

    cfg = config or {}
    source_tool = cfg.get("source_tool", "J4")
    target_tool = cfg.get("target_tool", "H1")
    measurement_ids = cfg.get("measurement_ids")
    batch_size = int(cfg.get("batch_size", 64))
    clamp_non_negative = bool(cfg.get("clamp_non_negative", True))
    device = torch.device(
        cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    def _film_cond(
        m: nn.Module,
        sub_df: pd.DataFrame,
        dev: torch.device,
    ) -> dict[str, Any] | None:
        if not getattr(m, "use_film", False):
            return None
        film_gen = getattr(m, "film_gen", None)
        meta = [
            {**r, "tool_target": target_tool}
            for r in sub_df.to_dict("records")
        ]
        return prepare_film_condition_tensor(
            metadata=meta,
            config={
                "num_samples": len(sub_df),
                "energy_scale_ev": float(getattr(m, "energy_scale_ev", 1000.0)),
                "encode_flux": bool(getattr(film_gen, "encode_flux", True)),
                "encode_time": bool(getattr(film_gen, "encode_time", True)),
            },
            device=dev,
        )

    mask = df_meta["tool"] == source_tool
    if "split" in df_meta.columns:
        mask = mask & df_meta["split"].isin(["train", "val", "test"])
    if measurement_ids:
        mask = mask & df_meta["measurement_id"].isin(measurement_ids)
    source_df = df_meta[mask].copy()
    if source_df.empty:
        raise ValueError(
            f"No source spectra found for {source_tool=} in train/val/test splits"
        )

    for m in models.values():
        mod = m[0] if isinstance(m, (tuple, list)) else m
        if isinstance(mod, nn.Module):
            mod.eval().to(device)

    predicted_intensities = np.zeros(
        (len(source_df), ary_intensity.shape[1]), dtype=np.float32
    )
    predicted_energies = np.array(
        ary_energy[source_df["spectrum_index"].astype(int).to_numpy()],
        dtype=np.float32,
    )

    for region, group in source_df.groupby("region"):
        group_new_indices = np.flatnonzero(
            (source_df["region"] == region).to_numpy()
        )
        orig_indices = group["spectrum_index"].astype(int).to_numpy()

        if region not in models:
            warnings.warn(
                f"Region '{region}' not in models dictionary; "
                "falling back to identity pass-through."
            )
            predicted_intensities[group_new_indices] = ary_intensity[
                orig_indices
            ]
            continue

        model_entry = models[str(region)]
        region_x = ary_intensity[orig_indices]

        is_seq = isinstance(model_entry, (tuple, list))
        is_sw = bool(cfg.get("use_sliding_window", False)) or is_seq

        if is_sw:
            m_obj = model_entry[0] if is_seq else model_entry
            w_size = (
                int(model_entry[1])
                if is_seq
                else int(cfg.get("window_size", 15))
            )
            s_step = (
                int(model_entry[2])
                if is_seq
                else int(cfg.get("stride", max(1, w_size // 2)))
            )
            m_dev = next((p.device for p in m_obj.parameters()), device)
            sw_cfg = {
                "window_size": w_size,
                "stride": s_step,
                "device": m_dev,
            }
            predicted_intensities[group_new_indices] = np.vstack(
                [
                    predict_sliding_window_spectrum(
                        m_obj,
                        x_s,
                        config=sw_cfg,
                        cond=_film_cond(
                            m_obj,
                            group.iloc[i : i + 1],
                            m_dev,
                        ),
                        energy=ary_energy[orig_indices[i]],
                    )
                    for i, x_s in enumerate(region_x)
                ]
            )
            continue

        with torch.no_grad():
            preds = []
            for i in range(0, len(region_x), batch_size):
                bx = torch.as_tensor(
                    region_x[i : i + batch_size],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(1)
                cond = _film_cond(
                    model_entry,
                    group.iloc[i : i + batch_size],
                    device,
                )
                out = (
                    model_entry(bx, cond=cond)
                    if cond is not None
                    else model_entry(bx)
                )
                preds.append(out.squeeze(1).cpu().numpy())
        predicted_intensities[group_new_indices] = np.vstack(preds)

    if clamp_non_negative:
        predicted_intensities = np.clip(predicted_intensities, 0.0, None)
    df_meta_predicted = assemble_prediction_metadata(
        source_df,
        {
            "source_tool": source_tool,
            "target_tool": target_tool,
            "session_splits": cfg.get("session_splits"),
            "n_points": ary_intensity.shape[1],
        },
    )
    return predicted_intensities, predicted_energies, df_meta_predicted
