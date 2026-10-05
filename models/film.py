"""Feature-wise Linear Modulation (FiLM) module for spectral transformation models.

Enables dynamic feature conditioning using metadata (die ID, tools, measurement time,
and Tier 2 source integrated area/flux) without target data leakage.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# Default mapping for known tool identifiers
TOOL_INDEX_MAP: dict[str, int] = {
    "J4": 0,
    "J5": 1,
    "H1": 2,
    "H2": 3,
    "UNKNOWN": 4,
}


def prepare_film_condition_tensor(
    metadata: list[dict[str, Any]] | dict[str, Any] | None = None,
    scale_x: torch.Tensor | np.ndarray | None = None,
    device: torch.device | None = None,
    num_samples: int = 1,
) -> dict[str, torch.Tensor]:
    """Extract and pack conditioning features from batch metadata and source scales.

    Guarantees zero target leakage: only source measurements, die indices, and
    known tool identifiers are used. Supports both list-of-dicts and collated
    dict-of-lists metadata structures.

    Parameters
    ----------
    metadata : list[dict[str, Any]] | dict[str, Any] | None, optional
        Sample metadata containing 'die', 'tool', 'tool_target', 'time'.
    scale_x : torch.Tensor | np.ndarray | None, optional
        Source scale factors (e.g. Die Total Integrated Flux F_src or max(x)).
    device : torch.device | None, optional
        PyTorch computation device.
    num_samples : int, optional
        Batch size fallback when metadata is empty (default 1).

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with tensor features:
        - 'die': LongTensor (B,)
        - 'tool_src': LongTensor (B,)
        - 'tool_tgt': LongTensor (B,)
        - 'log_flux_src': FloatTensor (B, 1)
        - 'delta_time_days': FloatTensor (B, 1)
    """
    if isinstance(metadata, pd.DataFrame):
        metadata = metadata.to_dict(orient="records")

    # Determine batch size n_items
    if metadata is not None and isinstance(metadata, list):
        n_items = len(metadata) if len(metadata) > 0 else num_samples
    elif metadata is not None and isinstance(metadata, dict):
        seq_lens = [
            len(v)
            for v in metadata.values()
            if isinstance(v, (list, tuple, np.ndarray, torch.Tensor)) and not isinstance(v, (str, bytes))
        ]
        n_items = seq_lens[0] if seq_lens else num_samples
    else:
        n_items = num_samples

    die_list: list[int] = []
    tool_src_list: list[int] = []
    tool_tgt_list: list[int] = []
    delta_time_list: list[float] = []

    def _safe_die(d: Any) -> int:
        if d is None or pd.isna(d):
            return 0
        try:
            return min(max(int(d), 0), 8)
        except Exception:
            return 0

    def _safe_tool(t: Any, default: str) -> int:
        if t is None or pd.isna(t):
            t_str = default
        else:
            t_str = str(t).upper()
        return TOOL_INDEX_MAP.get(t_str, TOOL_INDEX_MAP.get(str(t), TOOL_INDEX_MAP["UNKNOWN"]))

    def _safe_delta_time(t1: Any, t2: Any) -> float:
        if t1 is None or t2 is None or pd.isna(t1) or pd.isna(t2):
            return 0.0
        try:
            ts1 = pd.Timestamp(t1)
            ts2 = pd.Timestamp(t2)
            return abs((ts2 - ts1).total_seconds()) / 86400.0
        except Exception:
            return 0.0

    if metadata is not None and isinstance(metadata, dict):
        # 1. Die
        raw_dies = metadata.get("die")
        if raw_dies is not None:
            if isinstance(raw_dies, (list, tuple, np.ndarray, torch.Tensor)) and not isinstance(raw_dies, (str, bytes)):
                die_list = [_safe_die(d.item() if hasattr(d, "item") else d) for d in raw_dies]
            else:
                die_list = [_safe_die(raw_dies)] * n_items
        else:
            die_list = [0] * n_items

        # 2. Source Tool
        raw_src = metadata.get("tool")
        if raw_src is not None:
            if isinstance(raw_src, (list, tuple, np.ndarray, torch.Tensor)) and not isinstance(raw_src, (str, bytes)):
                tool_src_list = [_safe_tool(t, "J4") for t in raw_src]
            else:
                tool_src_list = [_safe_tool(raw_src, "J4")] * n_items
        else:
            tool_src_list = [_safe_tool("J4", "J4")] * n_items

        # 3. Target Tool
        raw_tgt = metadata.get("tool_target", metadata.get("target_tool"))
        if raw_tgt is not None:
            if isinstance(raw_tgt, (list, tuple, np.ndarray, torch.Tensor)) and not isinstance(raw_tgt, (str, bytes)):
                tool_tgt_list = [_safe_tool(t, "J5") for t in raw_tgt]
            else:
                tool_tgt_list = [_safe_tool(raw_tgt, "J5")] * n_items
        else:
            tool_tgt_list = [_safe_tool("J5", "J5")] * n_items

        # 4. Delta Time
        raw_t1 = metadata.get("time")
        raw_t2 = metadata.get("time_target")
        is_seq_t1 = isinstance(raw_t1, (list, tuple, np.ndarray, torch.Tensor)) and not isinstance(raw_t1, (str, bytes))
        is_seq_t2 = isinstance(raw_t2, (list, tuple, np.ndarray, torch.Tensor)) and not isinstance(raw_t2, (str, bytes))

        if is_seq_t1 and is_seq_t2:
            delta_time_list = [_safe_delta_time(t1, t2) for t1, t2 in zip(raw_t1, raw_t2)]
        elif raw_t1 is not None and raw_t2 is not None:
            dt_val = _safe_delta_time(raw_t1, raw_t2)
            delta_time_list = [dt_val] * n_items
        else:
            delta_time_list = [0.0] * n_items

    elif metadata is not None and isinstance(metadata, list) and len(metadata) > 0:
        for m in metadata:
            if isinstance(m, dict):
                die_list.append(_safe_die(m.get("die")))
                tool_src_list.append(_safe_tool(m.get("tool"), "J4"))
                tool_tgt_list.append(_safe_tool(m.get("tool_target", m.get("target_tool")), "J5"))
                delta_time_list.append(_safe_delta_time(m.get("time"), m.get("time_target")))
            else:
                die_list.append(0)
                tool_src_list.append(_safe_tool("J4", "J4"))
                tool_tgt_list.append(_safe_tool("J5", "J5"))
                delta_time_list.append(0.0)
    else:
        die_list = [0] * n_items
        tool_src_list = [_safe_tool("J4", "J4")] * n_items
        tool_tgt_list = [_safe_tool("J5", "J5")] * n_items
        delta_time_list = [0.0] * n_items

    # Parse source integrated flux
    if scale_x is not None:
        if isinstance(scale_x, torch.Tensor):
            s_tensor = scale_x.float()
        else:
            s_tensor = torch.from_numpy(np.asarray(scale_x, dtype=np.float32))
        if s_tensor.numel() == 1 and n_items > 1:
            s_tensor = s_tensor.view(1, 1).expand(n_items, 1)
        else:
            s_tensor = s_tensor.view(n_items, 1)
        log_flux = torch.log10(torch.clamp(s_tensor, min=1e-4))
    else:
        log_flux = torch.zeros((n_items, 1), dtype=torch.float32)

    die_t = torch.tensor(die_list, dtype=torch.long)
    tool_src_t = torch.tensor(tool_src_list, dtype=torch.long)
    tool_tgt_t = torch.tensor(tool_tgt_list, dtype=torch.long)
    dt_t = torch.tensor(delta_time_list, dtype=torch.float32).unsqueeze(-1)

    if device is not None:
        die_t = die_t.to(device)
        tool_src_t = tool_src_t.to(device)
        tool_tgt_t = tool_tgt_t.to(device)
        log_flux = log_flux.to(device)
        dt_t = dt_t.to(device)

    return {
        "die": die_t,
        "tool_src": tool_src_t,
        "tool_tgt": tool_tgt_t,
        "log_flux_src": log_flux,
        "delta_time_days": dt_t,
    }


class FiLMGenerator(nn.Module):
    """Generates feature-wise linear modulation parameters (gamma, beta) from conditioning features.

    Initializes output projections to zero so that initial gamma=1.0 and beta=0.0,
    ensuring strict identity modulation at training initialization.

    Parameters
    ----------
    channels_list : list[int]
        Target convolutional channel dimensions for each layer to modulate.
    config : dict[str, Any] | None, optional
        Configuration dictionary:
        - 'num_dies' (int): Total number of unique dies (default 9).
        - 'die_embed_dim' (int): Die embedding dimension (default 4).
        - 'num_tools' (int): Maximum number of tools (default 8).
        - 'tool_embed_dim' (int): Tool embedding dimension (default 4).
        - 'hidden_dim' (int): FiLM MLP hidden dimension (default 32).
    """

    def __init__(
        self,
        channels_list: list[int],
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg = config or {}
        num_dies = int(cfg.get("num_dies", 9))
        die_dim = int(cfg.get("die_embed_dim", 4))
        num_tools = int(cfg.get("num_tools", 8))
        tool_dim = int(cfg.get("tool_embed_dim", 4))
        hidden_dim = int(cfg.get("hidden_dim", 32))

        self.channels_list = list(channels_list)

        # Categorical embeddings
        self.die_embedding = nn.Embedding(num_dies, die_dim)
        self.tool_embedding = nn.Embedding(num_tools, tool_dim)

        # Total input dimension: die_dim + 2 * tool_dim + 2 continuous features (log_flux, delta_time)
        input_dim = die_dim + (2 * tool_dim) + 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Layer-wise (gamma, beta) projections initialized to zero for identity start
        self.gamma_projs = nn.ModuleList()
        self.beta_projs = nn.ModuleList()

        for ch in self.channels_list:
            g_proj = nn.Linear(hidden_dim, ch)
            b_proj = nn.Linear(hidden_dim, ch)

            nn.init.zeros_(g_proj.weight)
            nn.init.zeros_(g_proj.bias)
            nn.init.zeros_(b_proj.weight)
            nn.init.zeros_(b_proj.bias)

            self.gamma_projs.append(g_proj)
            self.beta_projs.append(b_proj)

    def forward(
        self,
        cond: dict[str, torch.Tensor] | torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute (gamma, beta) pairs for each target convolutional layer.

        Parameters
        ----------
        cond : dict[str, torch.Tensor] | torch.Tensor
            Dictionary containing 'die', 'tool_src', 'tool_tgt', 'log_flux_src', 'delta_time_days',
            or already concatenated FloatTensor of shape (B, input_dim).

        Returns
        -------
        list[tuple[torch.Tensor, torch.Tensor]]
            List of (gamma, beta) tuples of shape (B, channels) for each modulated layer.
        """
        if isinstance(cond, dict):
            die_emb = self.die_embedding(cond["die"])
            src_tool_emb = self.tool_embedding(cond["tool_src"])
            tgt_tool_emb = self.tool_embedding(cond["tool_tgt"])
            log_flux = cond["log_flux_src"]
            dt = cond["delta_time_days"]
            feat = torch.cat([die_emb, src_tool_emb, tgt_tool_emb, log_flux, dt], dim=-1)
        else:
            feat = cond

        h = self.mlp(feat)

        modulations: list[tuple[torch.Tensor, torch.Tensor]] = []
        for g_proj, b_proj in zip(self.gamma_projs, self.beta_projs):
            gamma = 1.0 + g_proj(h)
            beta = b_proj(h)
            modulations.append((gamma, beta))

        return modulations
