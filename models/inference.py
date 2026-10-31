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
    encode_energy = bool(getattr(film_gen, "encode_energy", False))
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
                scaled_b = (
                    torch.tensor(bounds, dtype=torch.float32, device=device)
                    / scale_floor
                )
                b_cond["energy_bounds"] = scaled_b
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


def _accumulate_overlapping_patches(
    preds: torch.Tensor,
    start_indices: list[int],
    seq_len: int,
    window_size: int,
    device: torch.device,
    clamp_non_negative: bool,
) -> torch.Tensor:
    """Accumulate overlapping window predictions and normalize by overlap
    count.
    """
    n_samples = preds.shape[0]
    accum = torch.zeros(
        n_samples, 1, seq_len, device=device, dtype=preds.dtype
    )
    counts = torch.zeros(1, 1, seq_len, device=device, dtype=preds.dtype)

    # Accumulate overlapping patch predictions and track patch counts.
    for k, s in enumerate(start_indices):
        accum[:, :, s : s + window_size] += preds[:, k, :, :]
        counts[:, :, s : s + window_size] += 1.0

    recon = accum / counts.clamp(min=1.0)
    if clamp_non_negative:
        recon = torch.clamp(recon, min=0.0)
    return recon


def predict_sliding_window_tensor(
    model: nn.Module,
    x: torch.Tensor,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Predict and reconstruct full spectra using vectorized window batches
    in PyTorch.

    Parameters
    ----------
    model : nn.Module
        Trained patch model.
    x : torch.Tensor
        Input spectral tensor of shape (N, 1, L) or (N, L).
    config : dict[str, Any] | None, optional
        Configuration dictionary containing 'window_size', 'stride',
        'clamp_non_negative', 'batch_size', 'device'.

    Returns
    -------
    torch.Tensor
        Reconstructed 3D tensor of shape (N, 1, L).
    """
    cfg = config or {}
    w_size = int(cfg.get("window_size", 15))
    s_step = int(cfg.get("stride", max(1, w_size // 2)))
    clamp_non_negative = bool(cfg.get("clamp_non_negative", True))
    batch_size = int(cfg.get("batch_size", 64))

    if w_size < 1:
        raise ValueError(f"window_size must be >= 1, got {w_size}")
    if s_step < 1:
        raise ValueError(f"stride must be >= 1, got {s_step}")
    if s_step > w_size:
        raise ValueError(
            f"stride ({s_step}) cannot exceed window_size ({w_size}) "
            "as it creates reconstruction gaps."
        )

    # Standardize input to 3D tensor (N, 1, L).
    x_tensor = x.unsqueeze(1) if x.dim() == 2 else x
    if x_tensor.dim() != 3 or x_tensor.shape[1] != 1:
        raise ValueError(
            f"Expected input tensor of shape (N, 1, L) or (N, L), "
            f"got {tuple(x.shape)}"
        )

    n_samples, _, seq_len = x_tensor.shape
    if w_size > seq_len:
        raise ValueError(
            f"window_size ({w_size}) cannot exceed spectrum length ({seq_len})"
        )

    model_dev = next(
        (p.device for p in model.parameters()), torch.device("cpu")
    )
    device = torch.device(cfg.get("device") or model_dev)
    model_dtype = next((p.dtype for p in model.parameters()), torch.float32)
    x_dev = x_tensor.to(device=device, dtype=model_dtype)

    # Build window start indices with right-edge anchoring.
    start_indices = list(range(0, seq_len - w_size + 1, s_step))
    last_possible = seq_len - w_size
    if start_indices and start_indices[-1] < last_possible:
        start_indices.append(last_possible)

    patches = torch.stack(
        [x_dev[:, :, s : s + w_size] for s in start_indices], dim=1
    )
    flat_patches = patches.view(-1, 1, w_size)

    # Evaluate patches through model in mini-batches.
    model.eval()
    with torch.no_grad():
        if flat_patches.shape[0] <= batch_size:
            flat_preds = model(flat_patches)
        else:
            flat_preds = torch.cat(
                [
                    model(flat_patches[i : i + batch_size])
                    for i in range(0, flat_patches.shape[0], batch_size)
                ],
                dim=0,
            )

    preds = flat_preds.view(n_samples, len(start_indices), 1, w_size)
    return _accumulate_overlapping_patches(
        preds, start_indices, seq_len, w_size, device, clamp_non_negative
    )


def predict_sliding_window_spectra(
    model: nn.Module,
    spectra: np.ndarray | torch.Tensor,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Predict and reconstruct multiple full spectra using vectorized
    PyTorch batches.

    Parameters
    ----------
    model : nn.Module
        Trained patch model.
    spectra : np.ndarray | torch.Tensor
        Array or tensor of spectra of shape (N, L) or (N, 1, L).
    config : dict[str, Any] | None, optional
        Configuration dictionary containing 'window_size', 'stride',
        'clamp_non_negative', 'batch_size', 'device'.

    Returns
    -------
    np.ndarray
        Reconstructed 2D NumPy array of shape (N, L).
    """
    arr = torch.as_tensor(spectra, dtype=torch.float32)
    if arr.dim() == 1:
        arr = arr.unsqueeze(0).unsqueeze(0)
    elif arr.dim() == 2:
        arr = arr.unsqueeze(1)
    recon = predict_sliding_window_tensor(model, arr, config=config)
    return recon.squeeze(1).detach().cpu().numpy()


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

    extra_metric_cols = [
        "t7_code",
        "relative_strength",
        "mean_intensity_single_region",
        "mean_intensity_single_die",
        "relative_intensity_single_region",
        "total_flux_single_region",
        "mean_flux_across_regions_single_die",
        "total_flux_single_die",
        "mean_flux_across_dies_single_measurement",
        "total_flux_single_measurement",
        "total_flux_single_die_standardized",
        "time_standardized",
    ]

    records: list[dict[str, Any]] = []
    for idx, r in enumerate(source_df.itertuples()):
        row_dict: dict[str, Any] = {
            "spectrum_index": idx,
            "material": getattr(r, "material", "NMG"),
            "tool": target_tool,
            "target_tool": target_tool,
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
        }
        for col in extra_metric_cols:
            if hasattr(r, col):
                row_dict[col] = getattr(r, col)
        records.append(row_dict)

    return pd.DataFrame(records)


def _extract_film_condition(
    m: nn.Module,
    sub_df: pd.DataFrame,
    target_tool: str,
    dev: torch.device,
    cfg: dict[str, Any],
    energy_bounds: torch.Tensor | np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Extract FiLM condition dictionary for a sub-dataframe slice."""
    if not getattr(m, "use_film", False):
        return None
    film_gen = getattr(m, "film_gen", None) or getattr(
        m, "shortcut_film_gen", None
    )
    meta = [
        {**r, "target_tool": target_tool}
        for r in sub_df.to_dict("records")
    ]
    return film_gen.prepare_condition_tensor(
        metadata=meta,
        device=dev,
        energy_bounds=energy_bounds,
        energy_scale_ev=float(getattr(m, "energy_scale_ev", 1000.0)),
        extra_config={
            "num_samples": len(sub_df),
            "region_index_mapping": cfg.get("region_index_mapping"),
        },
    )


def _filter_source_metadata(
    df_meta: pd.DataFrame,
    source_tool: str,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Filter df_meta for active source tool records in evaluation splits."""
    measurement_ids = cfg.get("measurement_ids")
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

    if "region_index" not in source_df.columns and "region" in source_df.columns:
        unique_regs = sorted(source_df["region"].dropna().astype(str).unique())
        reg_map = cfg.get("region_index_mapping") or {
            r: i for i, r in enumerate(unique_regs)
        }
        source_df["region_index"] = source_df["region"].map(reg_map)

    return source_df


def _predict_sliding_window_region(
    model_entry: Any,
    region_x: np.ndarray,
    orig_indices: np.ndarray,
    group: pd.DataFrame,
    device: torch.device,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Predict target spectra for a region using sliding-window patch models."""
    # Resolve window size, stride step, and target device from model entry.
    is_seq = isinstance(model_entry, (tuple, list))
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
    sw_cfg = {"window_size": w_size, "stride": s_step, "device": m_dev}
    energy_arr = cfg.get("ary_energy")
    target_tool = str(cfg.get("target_tool", "H1"))

    if len(region_x) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    sw_cfg["clamp_non_negative"] = bool(cfg.get("clamp_non_negative", True))
    sw_cfg["batch_size"] = int(cfg.get("batch_size", 64))

    # Fast path: vectorized batch evaluation when FiLM conditioning is off.
    if not getattr(m_obj, "use_film", False):
        return predict_sliding_window_spectra(m_obj, region_x, config=sw_cfg)

    # Predict sliding window spectra for each sample and stack horizontally.
    return np.vstack(
        [
            predict_sliding_window_spectrum(
                m_obj,
                x_s,
                config=sw_cfg,
                cond=_extract_film_condition(
                    m_obj, group.iloc[i : i + 1], target_tool, m_dev, cfg
                ),
                energy=(
                    energy_arr[orig_indices[i]]
                    if energy_arr is not None
                    else None
                ),
            )
            for i, x_s in enumerate(region_x)
        ]
    )


def _predict_full_spectrum_region(
    model_entry: nn.Module,
    region_x: np.ndarray,
    orig_indices: np.ndarray,
    group: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    ary_energy: np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> np.ndarray:
    """Predict target spectra for a region without sliding window."""
    # Resolve target tool identity and batch processing parameters.
    config_dict = cfg or {}
    target_tool = str(
        config_dict.get(
            "target_tool",
            group["target_tool"].iloc[0]
            if "target_tool" in group.columns and not group.empty
            else "H1",
        )
    )

    # Execute batched forward inferences with optional FiLM conditioning.
    with torch.no_grad():
        preds = []
        for i in range(0, len(region_x), batch_size):
            bx = torch.as_tensor(
                region_x[i : i + batch_size],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1)
            batch_orig = orig_indices[i : i + batch_size]
            batch_e = ary_energy[batch_orig]
            bounds = np.column_stack([batch_e[:, 0], batch_e[:, -1]])
            cond = _extract_film_condition(
                model_entry,
                group.iloc[i : i + batch_size],
                target_tool,
                device,
                config_dict,
                energy_bounds=bounds,
            )
            out = (
                model_entry(bx, cond=cond)
                if cond is not None
                else model_entry(bx)
            )
            preds.append(out.squeeze(1).cpu().numpy())
    return np.vstack(preds)


def _predict_all_regions(
    source_df: pd.DataFrame,
    models_dict: dict[str, Any],
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    device: torch.device,
    batch_size: int,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Predict transformed intensities across all spectral regions."""
    predicted_intensities = np.zeros(
        (len(source_df), ary_intensity.shape[1]), dtype=np.float32
    )

    # Iterate over unique spectral regions and apply corresponding models.
    for region, group in source_df.groupby("region"):
        group_new_indices = np.flatnonzero(
            (source_df["region"] == region).to_numpy()
        )
        orig_indices = group["spectrum_index"].astype(int).to_numpy()

        if region not in models_dict and "all" not in models_dict:
            warnings.warn(
                f"Region '{region}' not in models dictionary; "
                "falling back to identity pass-through."
            )
            predicted_intensities[group_new_indices] = ary_intensity[orig_indices]
            continue
        model_entry = models_dict.get(str(region), models_dict.get("all"))
        region_x = ary_intensity[orig_indices]

        is_sw = bool(cfg.get("use_sliding_window", False)) or isinstance(
            model_entry, (tuple, list)
        )
        if is_sw:
            sw_cfg = {**cfg, "ary_energy": ary_energy}
            predicted_intensities[group_new_indices] = (
                _predict_sliding_window_region(
                    model_entry, region_x, orig_indices, group, device, sw_cfg
                )
            )
        else:
            predicted_intensities[group_new_indices] = (
                _predict_full_spectrum_region(
                    model_entry,
                    region_x,
                    orig_indices,
                    group,
                    device,
                    batch_size,
                    ary_energy,
                    cfg=cfg,
                )
            )
    return predicted_intensities


def _extract_prediction_inputs(
    data: tuple[np.ndarray, np.ndarray, pd.DataFrame] | dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract intensity, energy arrays and metadata dataframe from input
    container.
    """
    if not isinstance(data, (tuple, list, dict)):
        raise TypeError(f"Expected tuple/dict, got {type(data)}")
    if isinstance(data, dict):
        return data["ary_intensity"], data["ary_energy"], data["df_meta"]
    return data[0], data[1], data[2]


def _normalize_prediction_models(
    models: dict[str, nn.Module] | nn.Module | tuple[Any, ...] | list[Any],
    device: torch.device,
) -> dict[str, Any]:
    """Normalize model container into a dictionary and transfer to target
    device.
    """
    models_dict = (
        {"all": models}
        if isinstance(models, (nn.Module, tuple, list))
        else dict(models)
    )
    for m in models_dict.values():
        mod = m[0] if isinstance(m, (tuple, list)) else m
        if isinstance(mod, nn.Module):
            mod.eval().to(device)
    return models_dict


def _assemble_prediction_outputs(
    source_df: pd.DataFrame,
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    source_tool: str,
    target_tool: str,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Assemble prediction energies array and metadata dataframe."""
    pred_energies = np.array(
        ary_energy[source_df["spectrum_index"].astype(int).to_numpy()],
        dtype=np.float32,
    )
    df_meta_pred = assemble_prediction_metadata(
        source_df,
        {
            "source_tool": source_tool,
            "target_tool": target_tool,
            "session_splits": cfg.get("session_splits"),
            "n_points": ary_intensity.shape[1],
        },
    )
    return pred_energies, df_meta_pred


def predict_spectra(
    models: dict[str, nn.Module],
    data: tuple[np.ndarray, np.ndarray, pd.DataFrame] | dict[str, Any],
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Apply trained models to source spectra to generate predicted target
    spectra.
    """
    ary_intensity, ary_energy, df_meta = _extract_prediction_inputs(data)
    cfg = dict(config or {})
    source_tool = str(cfg.get("source_tool", "J4"))
    target_tool = str(cfg.get("target_tool", "H1"))
    cfg.setdefault("target_tool", target_tool)
    batch_size = int(cfg.get("batch_size", 64))
    device = torch.device(
        cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    source_df = _filter_source_metadata(df_meta, source_tool, cfg)
    models_dict = _normalize_prediction_models(models, device)
    pred_intensities = _predict_all_regions(
        source_df, models_dict, ary_intensity, ary_energy, device, batch_size, cfg
    )
    if bool(cfg.get("clamp_non_negative", True)):
        pred_intensities = np.clip(pred_intensities, 0.0, None)

    pred_energies, df_meta_pred = _assemble_prediction_outputs(
        source_df, ary_intensity, ary_energy, source_tool, target_tool, cfg
    )
    return pred_intensities, pred_energies, df_meta_pred
