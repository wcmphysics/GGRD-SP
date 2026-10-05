"""Inference and standardized prediction utilities for spectral transfer."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utility.patching import extract_sliding_windows, reconstruct_from_patches
from models.dataset import compute_die_total_flux
from models.film import prepare_film_condition_tensor


def predict_sliding_window_spectrum(
    model: nn.Module,
    spectrum: np.ndarray,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Transform a full 1D spectrum using a trained sliding window patch model.

    Applies unbiased linear overlap-averaging across patches, scales back to physical
    intensity using source scale, and enforces physical non-negativity strictly at the
    final reconstructed full spectrum level.

    Parameters
    ----------
    model : nn.Module
        Trained patch model expecting input length window_size.
    spectrum : np.ndarray
        1D source spectrum array of length N.
    config : dict[str, Any] | None, optional
        Configuration dictionary containing:
        - 'window_size' (int): Window points (default 15).
        - 'stride' (int): Stride points (default window_size // 2).
        - 'device' (str | torch.device | None): Computation device.
        - 'batch_size' (int): Batch size for patch forward pass (default 64).
        - 'normalize_by_source' (bool): Whether to apply source-referenced normalization (default True).
        - 'clamp_non_negative' (bool): Whether to clamp final reconstructed spectrum to non-negative (default True).
        - 'eps' (float): Epsilon floor for source scale (default 1e-4).

    Returns
    -------
    np.ndarray
        Reconstructed target spectrum of length N.
    """
    cfg = config or {}
    w_size = int(cfg.get("window_size", 15))
    s_step = int(cfg.get("stride", max(1, w_size // 2)))
    normalize_by_source = bool(cfg.get("normalize_by_source", True))
    clamp_non_negative = bool(cfg.get("clamp_non_negative", True))
    eps = float(cfg.get("eps", 1e-4))

    device = cfg.get("device")
    if device is None:
        params = list(model.parameters())
        device = params[0].device if params else torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    batch_size = int(cfg.get("batch_size", 64))

    spectrum_1d = np.asarray(spectrum).ravel()
    scale_factor = cfg.get("scale_factor")
    if scale_factor is not None:
        scale_x = max(float(scale_factor), eps)
        spectrum_in = spectrum_1d / scale_x
    elif normalize_by_source:
        scale_x = max(float(np.max(np.abs(spectrum_1d))), eps)
        spectrum_in = spectrum_1d / scale_x
    else:
        scale_x = 1.0
        spectrum_in = spectrum_1d

    windows, start_indices = extract_sliding_windows(
        spectrum_in, window_size=w_size, stride=s_step
    )

    model.eval()
    pred_patches_list: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i : i + batch_size]).to(device)
            out = model(batch).cpu().numpy()
            pred_patches_list.append(out)

    pred_patches = np.vstack(pred_patches_list)
    reconstructed_norm = reconstruct_from_patches(
        patches=pred_patches,
        start_indices=start_indices,
        original_length=len(spectrum_1d),
        config={"window_size": w_size, "clamp_non_negative": False},
    )

    # Rescale to physical counts
    reconstructed_phys = reconstructed_norm * scale_x if normalize_by_source else reconstructed_norm

    # Zero clamping at the final full spectrum reconstruction level
    if clamp_non_negative:
        reconstructed_final = np.clip(reconstructed_phys, 0.0, None)
    else:
        reconstructed_final = reconstructed_phys

    return reconstructed_final


def format_predicted_measurement_id(source_measurement_id: str, source_tool: str, target_tool: str) -> str:
    """Format a predicted measurement session ID uniquely and deterministically.

    Examples:
        'NMG_M_J4_00001', source_tool='J4', target_tool='H1' -> 'NMG_P_J4H1_00001'
        'M_J4_001', source_tool='J4', target_tool='H1' -> 'P_J4H1_001'
        'BATCH_A_001', source_tool='J4', target_tool='H1' -> 'P_J4H1_BATCH_A_001'

    Parameters
    ----------
    source_measurement_id : str
        Original source measurement ID.
    source_tool : str
        Source tool identifier (e.g. 'J4').
    target_tool : str
        Target tool identifier (e.g. 'H1').

    Returns
    -------
    str
        Formatted prediction measurement ID.
    """
    tool_marker = f"_M_{source_tool}_"
    if tool_marker in source_measurement_id:
        mat_prefix, rest = source_measurement_id.rsplit(tool_marker, 1)
        return f"{mat_prefix}_P_{source_tool}{target_tool}_{rest}"

    if "_M_" in source_measurement_id:
        mat_prefix, rest = source_measurement_id.rsplit("_M_", 1)
        prefix_to_strip = f"{source_tool}_"
        if rest.startswith(prefix_to_strip):
            rest = rest[len(prefix_to_strip):]
        return f"{mat_prefix}_P_{source_tool}{target_tool}_{rest}"

    if source_measurement_id.startswith("M_"):
        clean_suffix = source_measurement_id[2:]
        prefix_to_strip = f"{source_tool}_"
        if clean_suffix.startswith(prefix_to_strip):
            clean_suffix = clean_suffix[len(prefix_to_strip):]
        return f"P_{source_tool}{target_tool}_{clean_suffix}"
    return f"P_{source_tool}{target_tool}_{source_measurement_id}"


def assemble_prediction_metadata(
    source_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build standardized prediction metadata DataFrame for transferred spectra.

    Parameters
    ----------
    source_df : pd.DataFrame
        DataFrame of source rows being predicted.
    config : dict[str, Any] | None, optional
        Configuration dictionary containing:
        - 'source_tool' (str): Source tool identifier.
        - 'target_tool' (str): Target tool identifier.
        - 'n_points' (int): Number of spectral data points per region.
        - 'session_splits' (dict[str, set[str]] | None): Pre-partitioned session sets.

    Returns
    -------
    pd.DataFrame
        DataFrame containing standardized prediction metadata.
    """
    cfg = config or {}

    source_tool = str(cfg.get("source_tool", ""))
    target_tool = str(cfg.get("target_tool", ""))
    n_points = int(cfg.get("n_points", 0))
    session_splits = cfg.get("session_splits")

    records: list[dict[str, Any]] = []
    for new_idx, (_, row) in enumerate(source_df.iterrows()):
        orig_meas_id = str(row["measurement_id"])
        pred_meas_id = format_predicted_measurement_id(orig_meas_id, source_tool, target_tool)

        split_label = "unassigned"
        if session_splits:
            if orig_meas_id in session_splits.get("train", set()):
                split_label = "train"
            elif orig_meas_id in session_splits.get("val", set()):
                split_label = "val"
            elif orig_meas_id in session_splits.get("test", set()):
                split_label = "test"
            else:
                split_label = "unpaired"
        elif "split" in row and pd.notna(row["split"]):
            split_label = str(row["split"])

        meta_rec: dict[str, Any] = {
            "spectrum_index": new_idx,
            "material": row.get("material", "NMG"),
            "tool": target_tool,
            "measurement_id": pred_meas_id,
            "die": row.get("die", 0),
            "region": row.get("region"),
            "n_points": n_points,
            "time": row.get("time"),
            "source_tool": source_tool,
            "source_measurement_id": orig_meas_id,
            "source_spectrum_index": int(row["spectrum_index"]),
            "split": split_label,
            "is_predicted": True,
        }
        if "t7_code" in row:
            meta_rec["t7_code"] = row["t7_code"]
        records.append(meta_rec)

    return pd.DataFrame(records)


def predict_spectra(
    models: dict[str, nn.Module],
    data: tuple[np.ndarray, np.ndarray, pd.DataFrame] | dict[str, Any],
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Apply trained models to source spectra to generate predicted target spectra.

    Bundles input data into a single parameter following project clean interface rules.

    Outputs standardized data containers:
        (ary_intensity_predicted, ary_energy_predicted, meta_df_predicted)

    Parameters
    ----------
    models : dict[str, nn.Module]
        Dictionary mapping spectral region name (e.g., 'Al2p') to its trained PyTorch model.
    data : tuple[np.ndarray, np.ndarray, pd.DataFrame] | dict[str, Any]
        Original measured data bundled as:
        - tuple: (ary_intensity, ary_energy, meta_df), OR
        - dict: {'ary_intensity': ..., 'ary_energy': ..., 'meta_df': ...}
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'source_tool' (str): Identifier of the source tool (default 'J4').
        - 'target_tool' (str): Identifier of the target tool (default 'H1').
        - 'measurement_ids' (list[str] | None): Specific source measurement IDs to predict
          (default None = all sessions for source_tool).
        - 'batch_size' (int): Batch size for inference (default 64).
        - 'device' (str | torch.device | None): Computation device (default CPU/CUDA auto).
        - 'normalize_by_source' (bool): Whether to apply source-referenced normalization (default True).
        - 'clamp_non_negative' (bool): Whether to clamp final predicted intensities to non-negative (default True).
        - 'eps' (float): Epsilon floor for source normalization scale (default 1e-4).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame]
        - ary_intensity_predicted: 2D numpy array of shape (N_predicted, N_points).
        - ary_energy_predicted: 2D numpy array of shape (N_predicted, N_points).
        - meta_df_predicted: DataFrame containing standardized prediction metadata.
    """
    if isinstance(data, (tuple, list)):
        ary_intensity, ary_energy, meta_df = data[0], data[1], data[2]
    elif isinstance(data, dict):
        ary_intensity = data["ary_intensity"]
        ary_energy = data["ary_energy"]
        meta_df = data["meta_df"]
    else:
        raise TypeError(
            f"Expected data to be a tuple (ary_intensity, ary_energy, meta_df) or dict, got {type(data)}"
        )

    cfg = config or {}
    source_tool: str = cfg.get("source_tool", "J4")
    target_tool: str = cfg.get("target_tool", "H1")
    measurement_ids: list[str] | None = cfg.get("measurement_ids")
    batch_size: int = int(cfg.get("batch_size", 64))
    normalize_by_source: bool = bool(cfg.get("normalize_by_source", True))
    clamp_non_negative: bool = bool(cfg.get("clamp_non_negative", True))
    eps: float = float(cfg.get("eps", 1e-4))
    device_str = cfg.get("device")

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device_str, str):
        device = torch.device(device_str)
    else:
        device = device_str

    # Filter source rows
    condition = meta_df["tool"] == source_tool
    if measurement_ids is not None:
        condition = condition & meta_df["measurement_id"].isin(measurement_ids)

    source_df = meta_df[condition].copy()

    if source_df.empty:
        raise ValueError(f"No source spectra found for tool='{source_tool}' with criteria: {cfg}")

    # Set all models to eval mode on target device
    for m in models.values():
        if isinstance(m, nn.Module):
            m.eval()
            m.to(device)

    n_samples = len(source_df)
    n_points = ary_intensity.shape[1]

    predicted_intensities = np.zeros((n_samples, n_points), dtype=np.float32)
    predicted_energies = np.zeros((n_samples, n_points), dtype=np.float32)

    # Map original rows to new index
    source_indices = source_df["spectrum_index"].astype(int).to_numpy()
    predicted_energies[:] = ary_energy[source_indices]

    raw_mode = cfg.get("normalization_mode")
    if raw_mode is not None:
        norm_mode = str(raw_mode).lower()
    else:
        norm_mode = "source_referenced" if normalize_by_source else "none"

    source_flux_map: dict[tuple[str, int], float] = {}
    if norm_mode in ("die_total_flux", "die_flux"):
        source_flux_map = compute_die_total_flux(source_df, ary_intensity, ary_energy)

    # Process region-by-region in batches for vectorization speedup
    for region, group in source_df.groupby("region"):
        group_new_indices = np.arange(len(source_df))[(source_df["region"] == region).to_numpy()]
        orig_indices = group["spectrum_index"].astype(int).to_numpy()

        if region in models:
            model_entry = models[str(region)]
            region_x = ary_intensity[orig_indices]

            if norm_mode in ("die_total_flux", "die_flux") and source_flux_map:
                scales = np.array([
                    source_flux_map.get(
                        (str(r_row["measurement_id"]), int(r_row["die"])),
                        max(float(np.max(np.abs(ary_intensity[int(r_row["spectrum_index"])]))), eps),
                    )
                    for _, r_row in group.iterrows()
                ], dtype=np.float32)[:, None]
                region_x_in = region_x / scales
            elif normalize_by_source and norm_mode != "none":
                scales = np.maximum(np.max(np.abs(region_x), axis=-1, keepdims=True), eps)
                region_x_in = region_x / scales
            else:
                scales = np.ones((len(region_x), 1), dtype=np.float32)
                region_x_in = region_x

            is_sw = bool(cfg.get("use_sliding_window", False)) or isinstance(model_entry, (tuple, list))

            if is_sw:
                if isinstance(model_entry, (tuple, list)):
                    m_obj = model_entry[0]
                    w_size = int(model_entry[1])
                    s_step = int(model_entry[2])
                else:
                    m_obj = model_entry
                    w_size = int(cfg.get("window_size", 15))
                    s_step = int(cfg.get("stride", max(1, w_size // 2)))

                m_params = list(m_obj.parameters())
                m_dev = m_params[0].device if m_params else device

                sw_preds = []
                for idx_in_region, x_single in enumerate(region_x):
                    recon = predict_sliding_window_spectrum(
                        model=m_obj,
                        spectrum=x_single,
                        config={
                            "window_size": w_size,
                            "stride": s_step,
                            "device": m_dev,
                            "scale_factor": float(scales[idx_in_region, 0]),
                            "normalize_by_source": normalize_by_source,
                            "clamp_non_negative": clamp_non_negative,
                            "eps": eps,
                        },
                    )
                    sw_preds.append(recon)
                pred_arr = np.vstack(sw_preds)
            else:
                model = model_entry
                # Batch forward pass
                preds = []
                for i in range(0, len(region_x_in), batch_size):
                    batch_x = region_x_in[i : i + batch_size]
                    x_tensor = torch.from_numpy(batch_x.astype(np.float32)).to(device)
                    cond = None
                    if getattr(model, "use_film", False):
                        batch_meta = group.iloc[i : i + batch_size].to_dict(orient="records")
                        batch_scale = scales[i : i + batch_size]
                        cond = prepare_film_condition_tensor(
                            metadata=batch_meta,
                            scale_x=batch_scale,
                            device=device,
                            num_samples=len(batch_x),
                        )
                    with torch.no_grad():
                        y_pred = model(x_tensor, cond=cond).cpu().numpy() if cond is not None else model(x_tensor).cpu().numpy()
                    preds.append(y_pred)

                pred_arr = np.vstack(preds)
                pred_arr = pred_arr * scales

                # Zero clamping at final full spectrum reconstruction level
                if clamp_non_negative:
                    pred_arr = np.clip(pred_arr, 0.0, None)

            predicted_intensities[group_new_indices] = pred_arr
        else:
            warnings.warn(
                f"Region '{region}' not in models dictionary; falling back to identity pass-through."
            )
            fallback = ary_intensity[orig_indices]
            if clamp_non_negative:
                fallback = np.clip(fallback, 0.0, None)
            predicted_intensities[group_new_indices] = fallback

    meta_df_predicted = assemble_prediction_metadata(
        source_df,
        config={
            "source_tool": source_tool,
            "target_tool": target_tool,
            "n_points": n_points,
            "session_splits": cfg.get("session_splits"),
        },
    )

    return predicted_intensities, predicted_energies, meta_df_predicted
