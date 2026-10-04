"""Inference and standardized prediction utilities for spectral transfer."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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
    *args: Any,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
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
    *args : Any
        Positional parameters for backward compatibility: (source_tool, target_tool, n_points).
    **kwargs : Any
        Keyword arguments for backward compatibility (e.g. session_splits).

    Returns
    -------
    pd.DataFrame
        DataFrame containing standardized prediction metadata.
    """
    cfg = dict(config or {})
    cfg.update(kwargs)
    if len(args) >= 1:
        cfg["source_tool"] = args[0]
    if len(args) >= 2:
        cfg["target_tool"] = args[1]
    if len(args) >= 3:
        cfg["n_points"] = args[2]

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

    # Process region-by-region in batches for vectorization speedup
    for region, group in source_df.groupby("region"):
        group_new_indices = np.arange(len(source_df))[(source_df["region"] == region).to_numpy()]
        orig_indices = group["spectrum_index"].astype(int).to_numpy()

        if region in models:
            model = models[str(region)]
            region_x = ary_intensity[orig_indices]

            if normalize_by_source:
                scales = np.maximum(np.max(np.abs(region_x), axis=-1, keepdims=True), eps)
                region_x_in = region_x / scales
            else:
                scales = 1.0
                region_x_in = region_x

            # Batch forward pass
            preds = []
            for i in range(0, len(region_x_in), batch_size):
                batch_x = region_x_in[i : i + batch_size]
                x_tensor = torch.from_numpy(batch_x.astype(np.float32)).to(device)
                with torch.no_grad():
                    y_pred = model(x_tensor).cpu().numpy()
                preds.append(y_pred)

            pred_arr = np.vstack(preds)
            if normalize_by_source:
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
