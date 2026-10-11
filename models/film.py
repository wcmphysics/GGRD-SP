"""Feature-wise Linear Modulation (FiLM) module.

Enables dynamic feature conditioning using metadata (die ID,
tool indices, measurement time, and Tier 2 source integrated
area/flux) without target data leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _parse_time_diff(t1: Any, t2: Any) -> float:
    """Compute time difference in days between two timestamps."""
    try:
        if t1 is not None and t2 is not None and not (pd.isna(t1) or pd.isna(t2)):
            return abs((pd.Timestamp(t2) - pd.Timestamp(t1)).total_seconds()) / 86400.0
    except Exception:
        pass
    return 0.0


def _parse_die(value: Any) -> int:
    """Extract die index bounded within [0, 8]."""
    try:
        raw_val = value.item() if hasattr(value, "item") else value
        return min(max(int(raw_val), 0), 8) if value is not None and not pd.isna(value) else 0
    except Exception:
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
    device: torch.device | None = None, num_samples: int = 1,
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

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with conditioning tensor features:
        - 'die': LongTensor (B,)
        - 'tool_src': LongTensor (B,)
        - 'tool_tgt': LongTensor (B,)
        - 'log_flux_src': FloatTensor (B, 1)
        - 'delta_time_days': FloatTensor (B, 1)
    """
    meta = metadata.to_dict(orient="records") if isinstance(metadata, pd.DataFrame) else metadata
    lens = [
        len(v) for v in meta.values() if isinstance(v, (list, tuple, np.ndarray, torch.Tensor))
    ] if isinstance(meta, dict) else []
    n = lens[0] if lens else (len(meta) if isinstance(meta, list) and meta else num_samples)

    def _col(k: str, alt: str | None = None) -> list[Any]:
        if isinstance(meta, dict):
            val = meta.get(k, meta.get(alt) if alt else None)
            is_seq = isinstance(val, (list, tuple, np.ndarray, torch.Tensor)) and not isinstance(val, (str, bytes))
            return list(val) if is_seq else [val] * n
        return [
            (m.get(k, m.get(alt) if alt else None) if isinstance(m, dict) else None)
            for m in meta
        ] if isinstance(meta, list) and meta else [None] * n

    sx = torch.zeros((n, 1), dtype=torch.float32) if scale_x is None else (
        scale_x.float() if isinstance(scale_x, torch.Tensor) else torch.tensor(scale_x, dtype=torch.float32)
    )
    sx_2d = sx.view(1, 1).expand(n, 1) if sx.numel() == 1 and n > 1 else sx.view(n, 1)
    log_f = torch.log10(torch.clamp(sx_2d, min=1e-4)) if scale_x is not None else sx
    t_src = [_parse_tool_index(x, "tool_index") for x in _col("tool_index", "tool_src_index")]
    t_tgt = [_parse_tool_index(x, "tool_target_index") for x in _col("tool_target_index", "target_tool_index")]
    dt = [_parse_time_diff(t1, t2) for t1, t2 in zip(_col("time"), _col("time_target"))]
    return {
        "die": torch.tensor([_parse_die(x) for x in _col("die")], dtype=torch.long, device=device),
        "tool_src": torch.tensor(t_src, dtype=torch.long, device=device),
        "tool_tgt": torch.tensor(t_tgt, dtype=torch.long, device=device),
        "log_flux_src": log_f.to(device) if device is not None else log_f,
        "delta_time_days": torch.tensor(dt, dtype=torch.float32, device=device).unsqueeze(-1),
    }


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
    """

    def __init__(self, channels_list: list[int], config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        num_dies, die_dim = int(cfg.get("num_dies", 9)), int(cfg.get("die_embed_dim", 4))
        num_tools, tool_dim = int(cfg.get("num_tools", 8)), int(cfg.get("tool_embed_dim", 4))
        hidden_dim, in_dim = int(cfg.get("hidden_dim", 32)), die_dim + (2 * tool_dim) + 2
        self.channels_list = list(channels_list)
        self.die_embedding, self.tool_embedding = nn.Embedding(num_dies, die_dim), nn.Embedding(num_tools, tool_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
        )
        self.gamma_projs = nn.ModuleList([nn.Linear(hidden_dim, c) for c in self.channels_list])
        self.beta_projs = nn.ModuleList([nn.Linear(hidden_dim, c) for c in self.channels_list])
        for p in list(self.gamma_projs) + list(self.beta_projs):
            nn.init.zeros_(p.weight)
            nn.init.zeros_(p.bias)

    def forward(self, cond: dict[str, torch.Tensor] | torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute (gamma, beta) pairs for each modulated layer.

        Parameters
        ----------
        cond : dict[str, torch.Tensor] | torch.Tensor
            Conditioning dictionary or pre-concatenated tensor.

        Returns
        -------
        list[tuple[torch.Tensor, torch.Tensor]]
            List of (gamma, beta) tuples of shape (B, channels).
        """
        raw_feat = cond if isinstance(cond, torch.Tensor) else torch.cat(
            [
                self.die_embedding(cond["die"]), self.tool_embedding(cond["tool_src"]),
                self.tool_embedding(cond["tool_tgt"]), cond["log_flux_src"], cond["delta_time_days"],
            ],
            dim=-1,
        )
        h = self.mlp(raw_feat)
        return [(1.0 + g(h), b(h)) for g, b in zip(self.gamma_projs, self.beta_projs)]
