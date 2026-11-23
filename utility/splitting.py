"""Measurement session splitting utilities for GGRD-SP."""

from typing import Any
import numpy as np
import pandas as pd


_VALID_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


def _extract_code_name(measurement_id: str) -> str:
    """Extract leading underscore-delimited code name from measurement_id."""
    code = str(measurement_id).split("_", 1)[0].strip()
    if not code:
        raise ValueError(
            f"Cannot extract code name from measurement_id {measurement_id!r}."
        )
    return code


def _validate_split_code_list(raw_codes: Any, split_name: str) -> list[str]:
    """Validate and normalize one split's list of code names."""
    if raw_codes is None:
        raw_codes = []
    if not isinstance(raw_codes, (list, tuple, set)):
        raise ValueError(
            f"code_name_assignment['{split_name}'] must be a list of strings, "
            f"got {type(raw_codes).__name__}."
        )
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw_code in raw_codes:
        code = str(raw_code).strip()
        if not code:
            raise ValueError(
                f"code_name_assignment['{split_name}'] contains an empty code name."
            )
        if code in seen:
            raise ValueError(
                f"Duplicate code name '{code}' in code_name_assignment['{split_name}']."
            )
        seen.add(code)
        cleaned.append(code)
    return cleaned


def _invert_code_name_assignment(code_cfg: Any) -> dict[str, list[str]]:
    """Validate code_name_assignment and map codes to eligible splits."""
    if not isinstance(code_cfg, dict):
        raise ValueError(
            "method='code_name' requires 'code_name_assignment' dict with "
            "'train', 'val', and/or 'test' keys."
        )
    unknown_keys = set(code_cfg.keys()) - set(_VALID_SPLIT_NAMES)
    if unknown_keys:
        raise ValueError(
            f"Unknown split keys in code_name_assignment: {sorted(unknown_keys)}. "
            f"Expected subset of {_VALID_SPLIT_NAMES}."
        )

    code_to_splits: dict[str, list[str]] = {}
    has_eval = False
    for split_name in _VALID_SPLIT_NAMES:
        codes = _validate_split_code_list(code_cfg.get(split_name, []), split_name)
        if split_name in ("val", "test") and codes:
            has_eval = True
        for code in codes:
            code_to_splits.setdefault(code, []).append(split_name)

    if not code_to_splits:
        raise ValueError(
            "method='code_name' requires at least one code name in 'code_name_assignment'."
        )
    if not code_cfg.get("train"):
        raise ValueError(
            "method='code_name' requires at least one code name in 'train'."
        )
    if not has_eval:
        raise ValueError(
            "method='code_name' requires at least one code name in 'val' or 'test'."
        )
    return code_to_splits


def _partition_code_group(
    code_name: str,
    sessions: list[str],
    eligible_splits: list[str],
    ratio_map: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, set[str]]:
    """Partition sessions of one code name across its eligible splits."""
    if len(eligible_splits) == 1:
        return {eligible_splits[0]: set(sessions)}

    active_ratios = [ratio_map[s] for s in eligible_splits]
    if any(r <= 0.0 for r in active_ratios):
        raise ValueError(
            f"Code name '{code_name}' is assigned to splits {eligible_splits}, "
            f"so each must have a positive ratio; got {ratio_map}."
        )
    n_sessions = len(sessions)
    if n_sessions < len(eligible_splits):
        raise ValueError(
            f"Code name '{code_name}' has {n_sessions} paired session(s), "
            f"need >= {len(eligible_splits)} for splits {eligible_splits}."
        )

    shuffled = [str(s) for s in rng.permutation(sessions)]
    total = sum(active_ratios)
    if len(eligible_splits) == 2:
        s0, s1 = eligible_splits
        n0 = max(1, min(n_sessions - 1, int(round(n_sessions * (ratio_map[s0] / total)))))
        return {s0: set(shuffled[:n0]), s1: set(shuffled[n0:])}

    s0, s1, s2 = eligible_splits
    n0 = max(1, min(n_sessions - 2, int(round(n_sessions * (ratio_map[s0] / total)))))
    n1 = max(1, min(n_sessions - n0 - 1, int(round(n_sessions * (ratio_map[s1] / total)))))
    return {
        s0: set(shuffled[:n0]),
        s1: set(shuffled[n0 : n0 + n1]),
        s2: set(shuffled[n0 + n1 :]),
    }


def _execute_code_name_split(
    source_sessions: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[set[str], set[str], set[str]]:
    """Partition paired source sessions by measurement_id code name."""
    code_to_splits = _invert_code_name_assignment(cfg.get("code_name_assignment"))
    ratio_map = {
        "train": float(cfg.get("train_ratio", 0.5)),
        "val": float(cfg.get("val_ratio", 0.2)),
        "test": float(cfg.get("test_ratio", 0.3)),
    }
    if any(r < 0.0 for r in ratio_map.values()):
        raise ValueError(f"Split ratios must be non-negative: {ratio_map}")

    grouped: dict[str, list[str]] = {}
    for session_id in sorted(str(s) for s in source_sessions):
        code = _extract_code_name(session_id)
        grouped.setdefault(code, []).append(session_id)

    configured_codes = set(code_to_splits.keys())
    data_codes = set(grouped.keys())
    unknown_codes = sorted(configured_codes - data_codes)
    if unknown_codes:
        raise ValueError(
            f"code_name_assignment references unknown code name(s) {unknown_codes} "
            f"not found in paired measurements. Available: {sorted(data_codes)}."
        )
    unassigned_codes = sorted(data_codes - configured_codes)
    if unassigned_codes:
        raise ValueError(
            f"Paired measurements contain unassigned code name(s): {unassigned_codes}. "
            "All paired code names must be assigned in code_name_assignment."
        )

    rng = np.random.default_rng(cfg.get("seed"))
    split_sets: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for code_name in sorted(grouped.keys()):
        partitioned = _partition_code_group(
            code_name,
            grouped[code_name],
            code_to_splits[code_name],
            ratio_map,
            rng,
        )
        for split_name, sess_set in partitioned.items():
            split_sets[split_name].update(sess_set)

    return split_sets["train"], split_sets["val"], split_sets["test"]


def _execute_direct_split(
    source_sessions: np.ndarray,
    direct_cfg: Any,
) -> tuple[set[str], set[str], set[str]]:
    """Validate and return session split sets for direct assignment.

    Parameters
    ----------
    source_sessions : np.ndarray
        Array of all available paired source session IDs.
    direct_cfg : Any
        Configuration dict with 'train', 'val', and 'test' session ID lists.

    Returns
    -------
    tuple[set[str], set[str], set[str]]
        Sets of train, val, and test session IDs.
    """
    if not isinstance(direct_cfg, dict):
        raise ValueError(
            "method='direct' requires 'direct_assignment' dict with 'train', 'val', 'test' keys."
        )
    set_train = {str(s) for s in direct_cfg.get("train", [])}
    set_val = {str(s) for s in direct_cfg.get("val", [])}
    set_test = {str(s) for s in direct_cfg.get("test", [])}

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

    return set_train, set_val, set_test


def _execute_random_split(
    source_sessions: np.ndarray,
    ratios: tuple[float, float, float],
    seed: int | None = None,
) -> tuple[set[str], set[str], set[str]]:
    """Partition session IDs randomly according to train/val/test ratios.

    Parameters
    ----------
    source_sessions : np.ndarray
        Array of all available paired source session IDs.
    ratios : tuple[float, float, float]
        Ratios for train, val, and test splits.
    seed : int | None, optional
        Random seed for permutation reproducibility.

    Returns
    -------
    tuple[set[str], set[str], set[str]]
        Sets of train, val, and test session IDs.
    """
    # Validate non-negative split ratios and normalize to sum to 1.0.
    if any(r < 0.0 for r in ratios):
        raise ValueError(f"Split ratios must be non-negative: {ratios}")
    total_ratio = sum(ratios)
    if total_ratio <= 0.0:
        raise ValueError("Sum of train_ratio, val_ratio, and test_ratio must be positive.")

    # Permute session IDs randomly according to supplied random seed.
    n_sessions = len(source_sessions)
    train_ratio, val_ratio, test_ratio = [r / total_ratio for r in ratios]
    rng = np.random.default_rng(seed)
    shuffled = [str(s) for s in rng.permutation(source_sessions)]

    # Partition shuffled sessions into train, validation, and test subsets.
    if test_ratio > 0.0 and val_ratio > 0.0:
        if n_sessions < 3:
            raise ValueError(
                f"Need >= 3 distinct sessions for train/val/test splits, found {n_sessions}."
            )
        n_tr = max(1, min(n_sessions - 2, int(round(n_sessions * train_ratio))))
        n_va = max(1, min(n_sessions - n_tr - 1, int(round(n_sessions * val_ratio))))
        return set(shuffled[:n_tr]), set(shuffled[n_tr : n_tr + n_va]), set(shuffled[n_tr + n_va :])

    if n_sessions < 2:
        raise ValueError(
            f"Need >= 2 distinct sessions to split, found {n_sessions}."
        )
    eval_ratio = val_ratio if val_ratio > 0.0 else test_ratio
    n_eval = max(1, min(n_sessions - 1, int(round(n_sessions * eval_ratio))))
    set_train = set(shuffled[: n_sessions - n_eval])
    eval_set = set(shuffled[n_sessions - n_eval :])
    if val_ratio > 0.0:
        return set_train, eval_set, set()
    return set_train, set(), eval_set


def _build_session_mapping(
    df_meta: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[dict[str, str], np.ndarray]:
    """Filter paired measurements and return mapping and sorted source IDs."""
    source_tool = cfg.get("source_tool")
    target_tool = cfg.get("target_tool")
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

    meas_pairs = (
        paired_df[["measurement_id", "target_measurement_id"]]
        .drop_duplicates()
    )
    mapping: dict[str, str] = {
        str(row["measurement_id"]): str(row["target_measurement_id"])
        for _, row in meas_pairs.iterrows()
    }
    return mapping, np.array(sorted(mapping.keys()))


def _populate_split_column(
    df_meta: pd.DataFrame,
    session_splits: dict[str, set[str]],
    mapping: dict[str, str],
) -> None:
    """Populate the 'split' column in df_meta for each split group."""
    df_meta["split"] = "none"
    for split_name, src_set in session_splits.items():
        tgt_set = {mapping[s] for s in src_set if s in mapping}
        df_meta.loc[
            df_meta["measurement_id"].isin(src_set | tgt_set), "split"
        ] = split_name


def _resolve_session_split_sets(
    source_sessions: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, set[str]]:
    """Determine train, val, and test session sets from config."""
    method = str(cfg.get("method", "random")).lower()
    if method in ("direct", "direct_assignment"):
        train_s, val_s, test_s = _execute_direct_split(
            source_sessions, cfg.get("direct_assignment")
        )
    elif method == "code_name":
        train_s, val_s, test_s = _execute_code_name_split(
            source_sessions, cfg
        )
    elif method == "random":
        ratios = (
            float(cfg.get("train_ratio", 0.5)),
            float(cfg.get("val_ratio", 0.2)),
            float(cfg.get("test_ratio", 0.3)),
        )
        train_s, val_s, test_s = _execute_random_split(
            source_sessions, ratios, cfg.get("seed")
        )
    else:
        raise ValueError(
            f"Unknown splitting method '{method}'. Choose 'random', 'direct', or 'code_name'."
        )
    return {"train": train_s, "val": val_s, "test": test_s}


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
        - 'method': 'random' (default), 'direct', or 'code_name'.
        - 'train_ratio': fraction for training split (default 0.5).
        - 'val_ratio': fraction for validation split (default 0.2).
        - 'test_ratio': fraction for held-out test split (default 0.3).
        - 'seed': random seed for permutation (default None).
        - 'source_tool': optional source tool filter string.
        - 'target_tool': optional target tool filter string.
        - 'direct_assignment': dict of 'train', 'val', 'test' session ID lists.
        - 'code_name_assignment': dict of 'train', 'val', 'test' code lists.

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
    mapping, source_sessions = _build_session_mapping(df_meta, cfg)
    session_splits = _resolve_session_split_sets(source_sessions, cfg)
    _populate_split_column(df_meta, session_splits, mapping)
    return df_meta, mapping, session_splits
