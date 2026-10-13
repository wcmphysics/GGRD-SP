"""Feature-wise Linear Modulation (FiLM) module.

Enables dynamic feature conditioning using metadata (die ID,
tool indices, measurement time, and Tier 2 source integrated
area/flux) without target data leakage.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _parse_time_diff(t1: Any, t2: Any) -> float:
    """Compute time difference in days between two timestamps."""
    if t1 is None or t2 is None or pd.isna(t1) or pd.isna(t2):
        return 0.0
    try:
        return abs((pd.Timestamp(t2) - pd.Timestamp(t1)).total_seconds()) / 86400.0
    except (ValueError, TypeError) as exc:
        warnings.warn(f"Failed to parse timestamps '{t1}' and '{t2}': {exc}. Defaulting time diff to 0.0.")
        return 0.0


def _parse_die(value: Any) -> int:
    """Extract die index bounded within [0, 8]."""
    if value is None or pd.isna(value):
        return 0
    try:
        raw_val = value.item() if hasattr(value, "item") else value
        die_idx = int(raw_val)
        if not (0 <= die_idx <= 8):
            warnings.warn(f"Die index {die_idx} outside expected [0, 8] range; clamping to [0, 8].")
        return min(max(die_idx, 0), 8)
    except (ValueError, TypeError) as exc:
        warnings.warn(f"Invalid die value '{value}': {exc}. Defaulting die index to 0.")
        return 0


def _parse_tool_index(val: Any, field_name: str) -> int:
    """Extract integer tool index or raise informative ValueError."""
    if val is None or pd.isna(val):
        raise ValueError(
            f"Metadata field '{field_name}' is missing or null. "
            "Please ensure 'tool_index' and 'tool_target_index' are "
            "populated in df_meta."
        )
    try:
        idx = int(val.item() if hasattr(val, "item") else val)
        if idx < 0:
            raise ValueError(f"Tool index in '{field_name}' cannot be negative, got {idx}.")
        return idx
    except (ValueError, TypeError) as err:
        raise ValueError(f"Invalid tool index '{val}' in '{field_name}'. Expected non-negative int.") from err


def prepare_film_condition_tensor(
    metadata: list[dict[str, Any]] | dict[str, Any] | None = None,
    scale_x: torch.Tensor | np.ndarray | None = None,
    device: torch.device | None = None,
    num_samples: int = 1,
    window_energy: torch.Tensor | np.ndarray | None = None,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Extract and pack conditioning features from metadata.

    Guarantees zero target leakage: only source measurements,
    die indices, and explicit tool indices are used. Supports
    list-of-dicts, collated dict-of-lists, and DataFrame.

    Parameters
    ----------
    metadata : list[dict[str, Any]] | dict[str, Any] | None, optional
        Sample metadata with 'die', 'tool_index',
        'tool_target_index', 'time'.
    scale_x : torch.Tensor | np.ndarray | None, optional
        Source scale factors (e.g. Total Integrated Flux F_src).
    device : torch.device | None, optional
        PyTorch computation device.
    num_samples : int, optional
        Batch size fallback when metadata is empty (default 1).
    window_energy : torch.Tensor | np.ndarray | None, optional
        Starting and ending binding energies of the window patch,
        shape (B, 2).
    config : dict[str, Any] | None, optional
        Optional configuration containing 'energy_scale_ev' (default
        1000.0).
    **kwargs : Any
        Keyword arguments for backward compatibility.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with conditioning tensor features:
        - 'die': LongTensor (B,)
        - 'tool_src': LongTensor (B,)
        - 'tool_tgt': LongTensor (B,)
        - 'log_flux_src': FloatTensor (B, 1)
        - 'delta_time_days': FloatTensor (B, 1)
        - 'window_energy': FloatTensor (B, 2) (if provided)
    """
    cfg = {**(config or {}), **kwargs}
    energy_scale_ev = float(cfg.get("energy_scale_ev", 1000.0))

    meta = metadata.to_dict(orient="records") if isinstance(metadata, pd.DataFrame) else metadata
    lens = [
        len(v) for v in meta.values() if isinstance(v, (list, tuple, np.ndarray, torch.Tensor))
    ] if isinstance(meta, dict) else []
    n = lens[0] if lens else (len(meta) if isinstance(meta, list) and meta else num_samples)

    def _col(k: str) -> list[Any]:
        if isinstance(meta, dict):
            val = meta.get(k)
            is_seq = isinstance(val, (list, tuple, np.ndarray, torch.Tensor)) and not isinstance(val, (str, bytes))
            return list(val) if is_seq else [val] * n
        return [
            (m.get(k) if isinstance(m, dict) else None)
            for m in meta
        ] if isinstance(meta, list) and meta else [None] * n

    sx = torch.zeros((n, 1), dtype=torch.float32) if scale_x is None else (
        scale_x.float() if isinstance(scale_x, torch.Tensor) else torch.tensor(scale_x, dtype=torch.float32)
    )
    sx_2d = sx.view(1, 1).expand(n, 1) if sx.numel() == 1 and n > 1 else sx.view(n, 1)
    log_f = torch.log10(torch.clamp(sx_2d, min=1e-4)) if scale_x is not None else sx
    t_src = [_parse_tool_index(x, "tool_index") for x in _col("tool_index")]
    t_tgt = [_parse_tool_index(x, "tool_target_index") for x in _col("tool_target_index")]
    dt = [_parse_time_diff(t1, t2) for t1, t2 in zip(_col("time"), _col("time_target"))]

    ret: dict[str, torch.Tensor] = {
        "die": torch.tensor([_parse_die(x) for x in _col("die")], dtype=torch.long, device=device),
        "tool_src": torch.tensor(t_src, dtype=torch.long, device=device),
        "tool_tgt": torch.tensor(t_tgt, dtype=torch.long, device=device),
        "log_flux_src": log_f.to(device) if device is not None else log_f,
        "delta_time_days": torch.tensor(dt, dtype=torch.float32, device=device).unsqueeze(-1),
    }

    # Extract window energy bounds if provided explicitly or in metadata
    we_data = window_energy
    if we_data is None:
        we_from_meta = _col("window_energy")
        if any(x is not None for x in we_from_meta):
            we_data = we_from_meta

    if we_data is not None:
        if isinstance(we_data, torch.Tensor):
            we_t = we_data.clone().detach().float()
        else:
            we_t = torch.as_tensor(np.asarray(we_data), dtype=torch.float32)
        we_2d = we_t.view(n, 2) if we_t.numel() == 2 * n else we_t.view(-1, 2)
        scale_floor = max(energy_scale_ev, 1e-6)
        ret["window_energy"] = (we_2d / scale_floor).to(device) if device is not None else (we_2d / scale_floor)

    return ret


class FiLMGenerator(nn.Module):
    """Generates feature-wise linear modulation parameters.

    Initializes output projections to zero so that initial
    gamma=1.0 and beta=0.0 for strict identity modulation.

    Parameters
    ----------
    channels_list : list[int]
        Target convolutional channel dimensions to modulate.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'num_dies' (int): Total unique dies (default 9).
        - 'die_embed_dim' (int): Die embedding dim (default 4).
        - 'num_tools' (int): Maximum tools count (default 8).
        - 'tool_embed_dim' (int): Tool embedding dim (default 4).
        - 'hidden_dim' (int): FiLM MLP hidden dim (default 32).
        - 'use_sliding_window' (bool): Whether sliding window is used.
        - 'encode_window_energy' (bool): Whether to encode window energy.
    """

    def __init__(self, channels_list: list[int], config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        num_dies, die_dim = int(cfg.get("num_dies", 9)), int(cfg.get("die_embed_dim", 4))
        num_tools, tool_dim = int(cfg.get("num_tools", 8)), int(cfg.get("tool_embed_dim", 4))
        hidden_dim = int(cfg.get("hidden_dim", 32))

        # Activate window energy encoding when both sliding window
        # and encoding flag are active
        use_sw = bool(cfg.get("use_sliding_window", False))
        encode_energy = bool(cfg.get("encode_window_energy", False))
        self.encode_window_energy = use_sw and encode_energy
        energy_dim = 2 if self.encode_window_energy else 0

        in_dim = die_dim + (2 * tool_dim) + 2 + energy_dim
        self.channels_list = list(channels_list)
        self.die_embedding = nn.Embedding(num_dies, die_dim)
        self.tool_embedding = nn.Embedding(num_tools, tool_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
        )
        self.gamma_projs = nn.ModuleList([nn.Linear(hidden_dim, c) for c in self.channels_list])
        self.beta_projs = nn.ModuleList([nn.Linear(hidden_dim, c) for c in self.channels_list])
        for p in list(self.gamma_projs) + list(self.beta_projs):
            nn.init.zeros_(p.weight)
            nn.init.zeros_(p.bias)

    def forward(self, cond: dict[str, torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute affine (gamma, beta) modulation parameters per layer.

        Parameters
        ----------
        cond : dict[str, torch.Tensor]
            Conditioning features with 'die', 'tool_src',
            'tool_tgt', 'log_flux_src', 'delta_time_days', and
            optionally 'window_energy'.

        Returns
        -------
        list[tuple[torch.Tensor, torch.Tensor]]
            List of (gamma, beta) modulation pairs per target layer:
            - gamma: Multiplicative feature scale factor (B, channels),
              centered at 1.0 for identity modulation.
            - beta: Additive feature shift factor (B, channels),
              centered at 0.0 for identity modulation.
            Applied to feature map z as: FiLM(z) = gamma * z + beta.
        """
        target_device = self.die_embedding.weight.device
        cond_dev = {
            k: v.to(target_device) if isinstance(v, torch.Tensor) and v.device != target_device else v
            for k, v in cond.items()
        }
        feat_list = [
            self.die_embedding(cond_dev["die"]),
            self.tool_embedding(cond_dev["tool_src"]),
            self.tool_embedding(cond_dev["tool_tgt"]),
            cond_dev["log_flux_src"],
            cond_dev["delta_time_days"],
        ]
        if self.encode_window_energy:
            we = cond_dev.get("window_energy")
            if we is None:
                raise KeyError(
                    "FiLMGenerator expects 'window_energy' in conditioning dictionary "
                    "when encode_window_energy=True and use_sliding_window=True."
                )
            feat_list.append(we)

        raw_feat = torch.cat(feat_list, dim=-1)
        h = self.mlp(raw_feat)
        modulations = []
        for gamma_proj, beta_proj in zip(self.gamma_projs, self.beta_projs):
            gamma = 1.0 + gamma_proj(h)
            beta = beta_proj(h)
            modulations.append((gamma, beta))
        return modulations
