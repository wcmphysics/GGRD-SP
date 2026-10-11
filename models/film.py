"""Feature-wise Linear Modulation (FiLM) module.

Enables dynamic feature conditioning using metadata (die ID,
tools, measurement time, and Tier 2 source integrated
area/flux) without target data leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Default mapping for known tool identifiers
TOOL_INDEX_MAP: dict[str, int] = {"J4": 0, "J5": 1, "H1": 2, "H2": 3, "UNKNOWN": 4}


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


def _parse_tool(tool_id: Any, default: str) -> int:
    """Map tool identifier string to category index with fallback."""
    name = default if tool_id is None or pd.isna(tool_id) else str(tool_id)
    return TOOL_INDEX_MAP.get(name.upper(), TOOL_INDEX_MAP.get(name, TOOL_INDEX_MAP["UNKNOWN"]))


def prepare_film_condition_tensor(
    metadata: list[dict[str, Any]] | dict[str, Any] | None = None,
    scale_x: torch.Tensor | np.ndarray | None = None,
    device: torch.device | None = None,
    num_samples: int = 1,
) -> dict[str, torch.Tensor]:
    """Extract and pack conditioning features from metadata.

    Guarantees zero target leakage: only source measurements,
    die indices, and known tool identifiers are used. Supports
    list-of-dicts, collated dict-of-lists, and DataFrame.

    Parameters
    ----------
    metadata : list[dict[str, Any]] | dict[str, Any] | None, optional
        Sample metadata with 'die', 'tool', 'tool_target', 'time'.
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
    if isinstance(metadata, pd.DataFrame):
        metadata = metadata.to_dict(orient="records")

    # Determine batch size n
    if isinstance(metadata, dict):
        seqs = [
            len(v) for v in metadata.values()
            if isinstance(v, (list, tuple, np.ndarray, torch.Tensor)) and not isinstance(v, (str, bytes))
        ]
        n = seqs[0] if seqs else num_samples
    else:
        n = len(metadata) if isinstance(metadata, list) and len(metadata) > 0 else num_samples

    # Helper to extract column sequences uniformly
    def _col(k: str, alt: str | None = None) -> list[Any]:
        if isinstance(metadata, dict):
            val = metadata.get(k, metadata.get(alt) if alt else None)
            is_seq = isinstance(val, (list, tuple, np.ndarray, torch.Tensor)) and not isinstance(val, (str, bytes))
            return list(val) if is_seq else [val] * n
        if isinstance(metadata, list) and len(metadata) > 0:
            return [(m.get(k, m.get(alt) if alt else None) if isinstance(m, dict) else None) for m in metadata]
        return [None] * n

    # Continuous log-flux feature extraction
    if scale_x is not None:
        tensor_x = scale_x.float() if isinstance(scale_x, torch.Tensor) else torch.tensor(scale_x, dtype=torch.float32)
        tensor_x = tensor_x.view(1, 1).expand(n, 1) if tensor_x.numel() == 1 and n > 1 else tensor_x.view(n, 1)
        log_flux = torch.log10(torch.clamp(tensor_x, min=1e-4))
    else:
        log_flux = torch.zeros((n, 1), dtype=torch.float32)

    return {
        "die": torch.tensor([_parse_die(x) for x in _col("die")], dtype=torch.long, device=device),
        "tool_src": torch.tensor([_parse_tool(x, "J4") for x in _col("tool")], dtype=torch.long, device=device),
        "tool_tgt": torch.tensor(
            [_parse_tool(x, "J5") for x in _col("tool_target", "target_tool")], dtype=torch.long, device=device
        ),
        "log_flux_src": log_flux.to(device) if device is not None else log_flux,
        "delta_time_days": torch.tensor(
            [_parse_time_diff(t1, t2) for t1, t2 in zip(_col("time"), _col("time_target"))],
            dtype=torch.float32, device=device,
        ).unsqueeze(-1),
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
        hidden_dim = int(cfg.get("hidden_dim", 32))

        self.channels_list = list(channels_list)
        self.die_embedding = nn.Embedding(num_dies, die_dim)
        self.tool_embedding = nn.Embedding(num_tools, tool_dim)

        input_dim = die_dim + (2 * tool_dim) + 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
        )

        self.gamma_projs = nn.ModuleList([nn.Linear(hidden_dim, c) for c in self.channels_list])
        self.beta_projs = nn.ModuleList([nn.Linear(hidden_dim, c) for c in self.channels_list])

        for proj in list(self.gamma_projs) + list(self.beta_projs):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

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
        if isinstance(cond, torch.Tensor):
            feat = cond
        else:
            feat = torch.cat(
                [
                    self.die_embedding(cond["die"]), self.tool_embedding(cond["tool_src"]),
                    self.tool_embedding(cond["tool_tgt"]), cond["log_flux_src"], cond["delta_time_days"],
                ],
                dim=-1,
            )

        h = self.mlp(feat)
        return [(1.0 + g(h), b(h)) for g, b in zip(self.gamma_projs, self.beta_projs)]
