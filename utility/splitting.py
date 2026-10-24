"""Measurement session splitting utilities for GGRD-SP."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def split_measurement_sessions(
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, set[str]]]:
    """Partition paired measurement sessions into train, val, and test splits.

    Assigns all dies and spectral regions belonging to a physical measurement
    session into the same split group, populating the 'split' column in df_meta.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata DataFrame with 'measurement_id' and 'target_measurement_id'.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'method': 'random' (default) or 'direct'.
        - 'train_ratio': fraction for training split (default 0.5).
        - 'val_ratio': fraction for validation split (default 0.2).
        - 'test_ratio': fraction for held-out test split (default 0.3).
        - 'seed': random seed for permutation (default None).
        - 'source_tool': optional source tool filter string.
        - 'target_tool': optional target tool filter string.
        - 'direct_assignment': dict of 'train', 'val', 'test' session ID lists.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str], dict[str, set[str]]]
        Tuple of:
        - df_meta with 'split' column ('train', 'val', 'test', or 'none').
        - measurement_mapping: dict mapping source measurement ID to target ID.
        - session_splits: dict mapping 'train', 'val', 'test' to source ID sets.
    """
    cfg = config or {}
    df_meta = df_meta.copy()

    source_tool = cfg.get("source_tool")
    target_tool = cfg.get("target_tool")
    method = str(cfg.get("method", "random")).lower()
    seed: int | None = cfg.get("seed", None)

    # Filter paired rows
    condition = df_meta["target_measurement_id"].notna()
    if source_tool is not None:
        condition = condition & (df_meta["tool"] == str(source_tool))
    if target_tool is not None:
        condition = condition & (df_meta["target_tool"] == str(target_tool))

    paired_df = df_meta[condition]
    if paired_df.empty:
        raise ValueError(
            f"No paired measurements found in df_meta for {source_tool=}, {target_tool=}."
        )

    # Build 1-to-1 measurement mapping: source_meas_id -> target_meas_id
    meas_pairs = (
        paired_df[["measurement_id", "target_measurement_id"]]
        .drop_duplicates()
    )
    measurement_mapping: dict[str, str] = {
        str(row["measurement_id"]): str(row["target_measurement_id"])
        for _, row in meas_pairs.iterrows()
    }
    source_sessions = np.array(sorted(measurement_mapping.keys()))
    n_sessions = len(source_sessions)

    set_train: set[str] = set()
    set_val: set[str] = set()
    set_test: set[str] = set()

    if method in ("direct", "direct_assignment"):
        direct = cfg.get("direct_assignment")
        if not isinstance(direct, dict):
            raise ValueError(
                "method='direct' requires 'direct_assignment' dict with 'train', 'val', 'test' keys."
            )
        set_train = {str(s) for s in direct.get("train", [])}
        set_val = {str(s) for s in direct.get("val", [])}
        set_test = {str(s) for s in direct.get("test", [])}

        # Verification 1: Each measurement appears in only one group
        overlap_tv = set_train & set_val
        overlap_tt = set_train & set_test
        overlap_vt = set_val & set_test
        if overlap_tv or overlap_tt or overlap_vt:
            raise ValueError(
                f"Direct assignment contains overlapping measurement sessions: "
                f"train&val={overlap_tv}, train&test={overlap_tt}, val&test={overlap_vt}"
            )

        # Verification 2: The 3 groups take all paired measurements
        all_assigned = set_train | set_val | set_test
        all_paired = set(source_sessions)
        if all_assigned != all_paired:
            missing = all_paired - all_assigned
            extra = all_assigned - all_paired
            raise ValueError(
                f"Direct assignment must cover all paired measurements. Missing: {missing}, Extra: {extra}"
            )

    elif method == "random":
        ratios = (
            float(cfg.get("train_ratio", 0.5)),
            float(cfg.get("val_ratio", 0.2)),
            float(cfg.get("test_ratio", 0.3)),
        )
        if any(r < 0.0 for r in ratios):
            raise ValueError(f"Split ratios must be non-negative: {ratios}")
        total_ratio = sum(ratios)
        if total_ratio <= 0.0:
            raise ValueError("Sum of train_ratio, val_ratio, and test_ratio must be positive.")

        train_ratio, val_ratio, test_ratio = [r / total_ratio for r in ratios]
        rng = np.random.default_rng(seed)
        shuffled = [str(s) for s in rng.permutation(source_sessions)]

        if test_ratio > 0.0 and val_ratio > 0.0:
            if n_sessions < 3:
                raise ValueError(
                    f"Need >= 3 distinct sessions for train/val/test splits, found {n_sessions}."
                )
            n_tr = max(1, min(n_sessions - 2, int(round(n_sessions * train_ratio))))
            n_va = max(1, min(n_sessions - n_tr - 1, int(round(n_sessions * val_ratio))))
            set_train = set(shuffled[:n_tr])
            set_val = set(shuffled[n_tr : n_tr + n_va])
            set_test = set(shuffled[n_tr + n_va :])
        else:
            if n_sessions < 2:
                raise ValueError(
                    f"Need >= 2 distinct sessions to split, found {n_sessions}."
                )
            eval_ratio = val_ratio if val_ratio > 0.0 else test_ratio
            n_eval = max(1, min(n_sessions - 1, int(round(n_sessions * eval_ratio))))
            set_train = set(shuffled[: n_sessions - n_eval])
            eval_set = set(shuffled[n_sessions - n_eval :])
            if val_ratio > 0.0:
                set_val, set_test = eval_set, set()
            else:
                set_val, set_test = set(), eval_set
    else:
        raise ValueError(
            f"Unknown splitting method '{method}'. Choose 'random' or 'direct'."
        )

    # Populate 'split' column in df_meta
    df_meta["split"] = "none"
    session_splits = {
        "train": set_train,
        "val": set_val,
        "test": set_test,
    }

    for split_name, src_set in session_splits.items():
        tgt_set = {
            measurement_mapping[s] for s in src_set if s in measurement_mapping
        }
        df_meta.loc[
            df_meta["measurement_id"].isin(src_set | tgt_set), "split"
        ] = split_name

    return df_meta, measurement_mapping, session_splits
