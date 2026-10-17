"""Feature-wise Linear Modulation (FiLM) module.

Enables dynamic feature conditioning using metadata (die ID,
tool indices, standardized measurement time, and standardized
source die flux) without target data leakage.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _parse_die(value: Any) -> int:
    """Extract die index bounded within [0, 8]."""
    if value is None or pd.isna(value):
        return 0
    try:
        raw_val = value.item() if hasattr(value, "item") else value
        die_idx = int(raw_val)
        if not (0 <= die_idx <= 8):
            warnings.warn(
                f"Die index {die_idx} outside expected [0, 8] range; clamping to [0, 8]."
            )
        return min(max(die_idx, 0), 8)
    except (ValueError, TypeError) as exc:
        warnings.warn(
            f"Invalid die value '{value}': {exc}. Defaulting die index to 0."
        )
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
            raise ValueError(
                f"Tool index in '{field_name}' cannot be negative, got {idx}."
            )
        return idx
    except (ValueError, TypeError) as err:
        raise ValueError(
            f"Invalid tool index '{val}' in '{field_name}'. Expected non-negative int."
        ) from err


def _parse_required_float(val: Any, field_name: str) -> float:
    """Extract finite float metadata value or raise informative ValueError."""
    if val is None or pd.isna(val):
        raise ValueError(
            f"Metadata field '{field_name}' is missing or null. "
            f"Please ensure '{field_name}' is populated in df_meta."
        )
    try:
        raw_val = val.item() if hasattr(val, "item") else val
        res = float(raw_val)
        if not np.isfinite(res):
            raise ValueError(
                f"Metadata field '{field_name}' must be finite, got {res}."
            )
        return res
    except (ValueError, TypeError) as err:
        raise ValueError(
            f"Invalid numeric value '{val}' in '{field_name}'."
        ) from err


def _extract_meta_column(meta: Any, key: str, n: int) -> list[Any]:
    """Extract a list of length n for a metadata key from dict or list."""
    if isinstance(meta, dict):
        val = meta.get(key)
        if isinstance(val, (list, tuple, np.ndarray, torch.Tensor)) and not isinstance(
            val, (str, bytes)
        ):
            return list(val)
        return [val] * n
    if isinstance(meta, list) and meta:
        return [m.get(key) if isinstance(m, dict) else None for m in meta]
    return [None] * n


def prepare_film_condition_tensor(
    metadata: list[dict[str, Any]] | dict[str, Any] | pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    *,
    device: torch.device | None = None,
    window_energy: torch.Tensor | np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """Extract and pack conditioning features from metadata.

    Guarantees zero target leakage: only source measurements,
    die indices, and explicit tool indices are used. Supports
    list-of-dicts, collated dict-of-lists, and DataFrame.

    Parameters
    ----------
    metadata : list[dict[str, Any]] | dict[str, Any] | pd.DataFrame | None
        Sample metadata with 'die', 'tool_index', 'tool_target_index',
        and (when enabled) 'die_flux_normalized' and 'time_normalized'.
    config : dict[str, Any] | None, optional
        Optional configuration containing:
        - 'encode_flux' (bool): Include 'die_flux_normalized' (default True).
        - 'encode_time' (bool): Include 'time_normalized' (default True).
        - 'energy_scale_ev' (float): Window energy scale factor
          (default 1000.0).
        - 'num_samples' (int): Batch size fallback when metadata is empty.
    device : torch.device | None, optional
        PyTorch computation device.
    window_energy : torch.Tensor | np.ndarray | None, optional
        Starting and ending binding energies of the window patch,
        shape (B, 2).

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with conditioning tensor features:
        - 'die': LongTensor (B,)
        - 'tool_src': LongTensor (B,)
        - 'tool_tgt': LongTensor (B,)
        - 'die_flux_normalized': FloatTensor (B, 1) (if encode_flux=True)
        - 'time_normalized': FloatTensor (B, 1) (if encode_time=True)
        - 'window_energy': FloatTensor (B, 2) (if provided)
    """
    cfg = config or {}
    num_samples = int(cfg.get("num_samples", 1))
    energy_scale_ev = float(cfg.get("energy_scale_ev", 1000.0))
    encode_flux = bool(cfg.get("encode_flux", cfg.get("use_flux", True)))
    encode_time = bool(cfg.get("encode_time", cfg.get("use_time", True)))

    meta = (
        metadata.to_dict(orient="records")
        if isinstance(metadata, pd.DataFrame)
        else metadata
    )
    if isinstance(meta, dict):
        seq_lens = [
            len(v)
            for v in meta.values()
            if isinstance(v, (list, tuple, np.ndarray, torch.Tensor))
        ]
        n = seq_lens[0] if seq_lens else num_samples
    elif isinstance(meta, list) and meta:
        n = len(meta)
    else:
        n = num_samples

    t_src = [
        _parse_tool_index(x, "tool_index")
        for x in _extract_meta_column(meta, "tool_index", n)
    ]
    t_tgt = [
        _parse_tool_index(x, "tool_target_index")
        for x in _extract_meta_column(meta, "tool_target_index", n)
    ]

    ret: dict[str, torch.Tensor] = {
        "die": torch.tensor(
            [_parse_die(x) for x in _extract_meta_column(meta, "die", n)],
            dtype=torch.long,
            device=device,
        ),
        "tool_src": torch.tensor(t_src, dtype=torch.long, device=device),
        "tool_tgt": torch.tensor(t_tgt, dtype=torch.long, device=device),
    }

    if encode_flux:
        flux_vals = [
            _parse_required_float(x, "die_flux_normalized")
            for x in _extract_meta_column(meta, "die_flux_normalized", n)
        ]
        ret["die_flux_normalized"] = torch.tensor(
            flux_vals, dtype=torch.float32, device=device
        ).unsqueeze(-1)

    if encode_time:
        time_vals = [
            _parse_required_float(x, "time_normalized")
            for x in _extract_meta_column(meta, "time_normalized", n)
        ]
        ret["time_normalized"] = torch.tensor(
            time_vals, dtype=torch.float32, device=device
        ).unsqueeze(-1)

    we_data = window_energy
    if we_data is None:
        we_from_meta = _extract_meta_column(meta, "window_energy", n)
        if any(x is not None for x in we_from_meta):
            we_data = we_from_meta

    if we_data is not None:
        we_t = (
            we_data.clone().detach().float()
            if isinstance(we_data, torch.Tensor)
            else torch.as_tensor(np.asarray(we_data), dtype=torch.float32)
        )
        we_2d = we_t.view(n, 2) if we_t.numel() == 2 * n else we_t.view(-1, 2)
        scaled_we = we_2d / max(energy_scale_ev, 1e-6)
        ret["window_energy"] = (
            scaled_we.to(device) if device is not None else scaled_we
        )

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
        - 'encode_flux' (bool): Encode 'die_flux_normalized' (default True).
        - 'encode_time' (bool): Encode 'time_normalized' (default True).
        - 'use_sliding_window' (bool): Whether sliding window is used.
        - 'encode_window_energy' (bool): Whether to encode window energy.
    """

    def __init__(
        self, channels_list: list[int], config: dict[str, Any] | None = None
    ) -> None:
        super().__init__()
        cfg = config or {}
        num_dies = int(cfg.get("num_dies", 9))
        die_dim = int(cfg.get("die_embed_dim", 4))
        num_tools = int(cfg.get("num_tools", 8))
        tool_dim = int(cfg.get("tool_embed_dim", 4))
        hidden_dim = int(cfg.get("hidden_dim", 32))

        self.encode_flux = bool(
            cfg.get("encode_flux", cfg.get("use_flux", True))
        )
        self.encode_time = bool(
            cfg.get("encode_time", cfg.get("use_time", True))
        )
        self.encode_window_energy = bool(
            cfg.get("use_sliding_window", False)
        ) and bool(cfg.get("encode_window_energy", False))

        flux_dim = 1 if self.encode_flux else 0
        time_dim = 1 if self.encode_time else 0
        energy_dim = 2 if self.encode_window_energy else 0

        in_dim = die_dim + (2 * tool_dim) + flux_dim + time_dim + energy_dim
        self.channels_list = list(channels_list)
        self.die_embedding = nn.Embedding(num_dies, die_dim)
        self.tool_embedding = nn.Embedding(num_tools, tool_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gamma_projs = nn.ModuleList(
            [nn.Linear(hidden_dim, c) for c in self.channels_list]
        )
        self.beta_projs = nn.ModuleList(
            [nn.Linear(hidden_dim, c) for c in self.channels_list]
        )
        for proj in (*self.gamma_projs, *self.beta_projs):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self, cond: dict[str, torch.Tensor]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute affine (gamma, beta) modulation parameters per layer.

        Parameters
        ----------
        cond : dict[str, torch.Tensor]
            Conditioning features with 'die', 'tool_src', 'tool_tgt',
            and optional 'die_flux_normalized', 'time_normalized', and
            'window_energy'.

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
            k: (
                v.to(target_device)
                if isinstance(v, torch.Tensor) and v.device != target_device
                else v
            )
            for k, v in cond.items()
        }
        feat_list = [
            self.die_embedding(cond_dev["die"]),
            self.tool_embedding(cond_dev["tool_src"]),
            self.tool_embedding(cond_dev["tool_tgt"]),
        ]
        if self.encode_flux:
            flux_t = cond_dev.get("die_flux_normalized")
            if flux_t is None:
                raise KeyError(
                    "FiLMGenerator expects 'die_flux_normalized' in conditioning "
                    "dictionary when encode_flux=True."
                )
            feat_list.append(flux_t)

        if self.encode_time:
            time_t = cond_dev.get("time_normalized")
            if time_t is None:
                raise KeyError(
                    "FiLMGenerator expects 'time_normalized' in conditioning "
                    "dictionary when encode_time=True."
                )
            feat_list.append(time_t)

        if self.encode_window_energy:
            we = cond_dev.get("window_energy")
            if we is None:
                raise KeyError(
                    "FiLMGenerator expects 'window_energy' in conditioning dictionary "
                    "when encode_window_energy=True and use_sliding_window=True."
                )
            feat_list.append(we)

        h = self.mlp(torch.cat(feat_list, dim=-1))
        return [
            (1.0 + gamma_proj(h), beta_proj(h))
            for gamma_proj, beta_proj in zip(self.gamma_projs, self.beta_projs)
        ]
