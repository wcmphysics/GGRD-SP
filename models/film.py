"""Feature-wise Linear Modulation (FiLM) module for spectral transformation models.

Enables dynamic feature conditioning using metadata (die ID, tools, measurement time,
and Tier 2 source integrated area/flux) without target data leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TOOL_INDEX_MAP: dict[str, int] = {"J4": 0, "J5": 1, "H1": 2, "H2": 3, "UNKNOWN": 4}


def _parse_time_diff(t1: Any, t2: Any) -> float:
    try:
        return abs((pd.Timestamp(t2) - pd.Timestamp(t1)).total_seconds()) / 86400.0 if t1 is not None and t2 is not None and not (pd.isna(t1) or pd.isna(t2)) else 0.0
    except Exception:
        return 0.0


def _parse_die(value: Any) -> int:
    try:
        return min(max(int(value.item() if hasattr(value, "item") else value), 0), 8) if value is not None and not pd.isna(value) else 0
    except Exception:
        return 0


def _parse_tool(tool_id: Any, default: str) -> int:
    name = default if tool_id is None or pd.isna(tool_id) else str(tool_id)
    return TOOL_INDEX_MAP.get(name.upper(), TOOL_INDEX_MAP.get(name, TOOL_INDEX_MAP["UNKNOWN"]))


def prepare_film_condition_tensor(
    metadata: list[dict[str, Any]] | dict[str, Any] | None = None,
    scale_x: torch.Tensor | np.ndarray | None = None,
    device: torch.device | None = None,
    num_samples: int = 1,
) -> dict[str, torch.Tensor]:
    """Extract and pack conditioning features from batch metadata and source scales."""
    meta = metadata.to_dict(orient="records") if isinstance(metadata, pd.DataFrame) else metadata
    lens = [len(v) for v in meta.values() if isinstance(v, (list, tuple, np.ndarray, torch.Tensor)) and not isinstance(v, (str, bytes))] if isinstance(meta, dict) else []
    n = lens[0] if lens else (len(meta) if isinstance(meta, list) and meta else num_samples)

    def _col(k: str, alt: str | None = None) -> list[Any]:
        if isinstance(meta, dict):
            val = meta.get(k, meta.get(alt) if alt else None)
            return list(val) if isinstance(val, (list, tuple, np.ndarray, torch.Tensor)) and not isinstance(val, (str, bytes)) else [val] * n
        return [(item.get(k, item.get(alt) if alt else None) if isinstance(item, dict) else None) for item in meta] if isinstance(meta, list) and meta else [None] * n

    sx = torch.zeros((n, 1), dtype=torch.float32) if scale_x is None else (scale_x.float() if isinstance(scale_x, torch.Tensor) else torch.from_numpy(np.asarray(scale_x, dtype=np.float32)))
    log_flux = sx if scale_x is None else torch.log10(torch.clamp(sx.view(1, 1).expand(n, 1) if sx.numel() == 1 and n > 1 else sx.view(n, 1), min=1e-4))
    return {
        "die": torch.tensor([_parse_die(x) for x in _col("die")], dtype=torch.long, device=device),
        "tool_src": torch.tensor([_parse_tool(x, "J4") for x in _col("tool")], dtype=torch.long, device=device),
        "tool_tgt": torch.tensor([_parse_tool(x, "J5") for x in _col("tool_target", "target_tool")], dtype=torch.long, device=device),
        "log_flux_src": log_flux.to(device) if device is not None else log_flux,
        "delta_time_days": torch.tensor([_parse_time_diff(t1, t2) for t1, t2 in zip(_col("time"), _col("time_target"))], dtype=torch.float32, device=device).unsqueeze(-1),
    }


class FiLMGenerator(nn.Module):
    """Generates feature-wise linear modulation parameters (gamma, beta)."""

    def __init__(self, channels_list: list[int], config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        num_dies, die_dim, num_tools, tool_dim, hidden_dim = int(cfg.get("num_dies", 9)), int(cfg.get("die_embed_dim", 4)), int(cfg.get("num_tools", 8)), int(cfg.get("tool_embed_dim", 4)), int(cfg.get("hidden_dim", 32))
        self.channels_list, self.die_embedding, self.tool_embedding = list(channels_list), nn.Embedding(num_dies, die_dim), nn.Embedding(num_tools, tool_dim)
        self.mlp = nn.Sequential(nn.Linear(die_dim + 2 * tool_dim + 2, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True))
        self.gamma_projs = nn.ModuleList([nn.Linear(hidden_dim, c) for c in self.channels_list])
        self.beta_projs = nn.ModuleList([nn.Linear(hidden_dim, c) for c in self.channels_list])
        for proj in list(self.gamma_projs) + list(self.beta_projs):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, cond: dict[str, torch.Tensor] | torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        feat = cond if isinstance(cond, torch.Tensor) else torch.cat([self.die_embedding(cond["die"]), self.tool_embedding(cond["tool_src"]), self.tool_embedding(cond["tool_tgt"]), cond["log_flux_src"], cond["delta_time_days"]], dim=-1)
        h = self.mlp(feat)
        return [(1.0 + g(h), b(h)) for g, b in zip(self.gamma_projs, self.beta_projs)]
