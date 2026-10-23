"""Feature-wise Linear Modulation (FiLM) module.

Enables dynamic feature conditioning using metadata (die ID,
tool indices, standardized measurement time, standardized source
die flux, relative regional intensity, and element multi-hot)
without target data leakage.
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


def _parse_region_index(
    val: Any,
    field_name: str = "region_index",
    region_mapping: dict[str, int] | None = None,
) -> int:
    """Extract integer region index or default gracefully to 0."""
    if val is None or pd.isna(val):
        return 0
    if isinstance(val, (int, np.integer)):
        if int(val) < 0:
            warnings.warn(
                f"Negative region index {int(val)} in '{field_name}'; clamping to 0."
            )
            return 0
        return int(val)
    if isinstance(val, str):
        if region_mapping and val in region_mapping:
            return max(0, int(region_mapping[val]))
        try:
            parsed = int(val)
            if parsed < 0:
                warnings.warn(
                    f"Negative region index {parsed} in '{field_name}'; clamping to 0."
                )
                return 0
            return parsed
        except (ValueError, TypeError):
            warnings.warn(
                f"Unmapped region string '{val}' in '{field_name}'; defaulting to index 0."
            )
            return 0
    try:
        raw_val = val.item() if hasattr(val, "item") else val
        idx = int(raw_val)
        if idx < 0:
            warnings.warn(
                f"Negative region index {idx} in '{field_name}'; clamping to 0."
            )
            return 0
        return idx
    except (ValueError, TypeError):
        warnings.warn(
            f"Invalid region value '{val}' in '{field_name}'; defaulting to 0."
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
    energy_bounds: torch.Tensor | np.ndarray | None = None,
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
        and (when enabled) flux, time, relative intensity, and material.
    config : dict[str, Any] | None, optional
        Optional configuration containing:
        - 'encode_die' (bool): Include die index (default True).
        - 'encode_tool' (bool): Include tool indices (default True).
        - 'encode_flux' (bool): Include standardized die flux.
        - 'encode_time' (bool): Include 'time_normalized'.
        - 'encode_energy' (bool): Include bounding energy (default False).
        - 'encode_relative_intensity' (bool): Include relative regional
          intensity.
        - 'encode_elements' (bool): Include material element multi-hot vector.
        - 'element_vocab' (list[str] | None): Dynamic element vocabulary.
        - 'material_elements' (dict[str, list[str]] | None): Material map.
        - 'energy_scale_ev' (float): Energy scale factor (1000.0).
        - 'num_samples' (int): Batch size fallback when metadata is empty.
    device : torch.device | None, optional
        PyTorch computation device.
    energy_bounds : torch.Tensor | np.ndarray | None, optional
        Starting and ending binding energies, shape (B, 2).
    window_energy : torch.Tensor | np.ndarray | None, optional
        Alias for energy_bounds for backward compatibility.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with conditioning tensor features:
        - 'die': LongTensor (B,) (if encode_die=True)
        - 'tool_src': LongTensor (B,) (if encode_tool=True)
        - 'tool_tgt': LongTensor (B,) (if encode_tool=True)
        - 'region': LongTensor (B,) (if encode_region=True)
        - 'die_flux_normalized': FloatTensor (B, 1) (if encode_flux=True)
        - 'time_normalized': FloatTensor (B, 1) (if encode_time=True)
        - 'relative_intensity_single_region': FloatTensor (B, 1)
        - 'element_multi_hot': FloatTensor (B, vocab_size)
        - 'energy_bounds': FloatTensor (B, 2) (if provided)
        - 'window_energy': FloatTensor (B, 2) (if provided)
    """
    cfg = config or {}
    num_samples = int(cfg.get("num_samples", 1))
    energy_scale_ev = float(cfg.get("energy_scale_ev", 1000.0))
    encode_die = bool(cfg.get("encode_die", True))
    encode_tool = bool(cfg.get("encode_tool", True))
    encode_flux = bool(cfg.get("encode_flux", cfg.get("use_flux", True)))
    encode_time = bool(cfg.get("encode_time", cfg.get("use_time", True)))
    encode_region = bool(cfg.get("encode_region", cfg.get("use_region", False)))
    encode_rel_int = bool(
        cfg.get("encode_relative_intensity", cfg.get("use_relative_intensity", False))
    )
    encode_elements = bool(
        cfg.get("encode_elements", cfg.get("use_elements", False))
    )
    encode_energy = bool(
        cfg.get("encode_energy", cfg.get("encode_window_energy", False))
    )

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

    ret: dict[str, torch.Tensor] = {}

    if encode_die:
        ret["die"] = torch.tensor(
            [_parse_die(x) for x in _extract_meta_column(meta, "die", n)],
            dtype=torch.long,
            device=device,
        )

    if encode_tool:
        t_src = [
            _parse_tool_index(x, "tool_index")
            for x in _extract_meta_column(meta, "tool_index", n)
        ]
        t_tgt = [
            _parse_tool_index(x, "tool_target_index")
            for x in _extract_meta_column(meta, "tool_target_index", n)
        ]
        ret["tool_src"] = torch.tensor(t_src, dtype=torch.long, device=device)
        ret["tool_tgt"] = torch.tensor(t_tgt, dtype=torch.long, device=device)

    if encode_flux:
        flux_vals = _extract_meta_column(
            meta, "total_flux_single_die_normalized", n
        )
        flux_col = "total_flux_single_die_normalized"
        if all(x is None for x in flux_vals):
            flux_vals = _extract_meta_column(meta, "die_flux_normalized", n)
            flux_col = (
                "total_flux_single_die_normalized (or die_flux_normalized)"
            )
        ret["die_flux_normalized"] = torch.tensor(
            [_parse_required_float(x, flux_col) for x in flux_vals],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)
        ret["total_flux_single_die_normalized"] = ret["die_flux_normalized"]

    if encode_time:
        time_vals = [
            _parse_required_float(x, "time_normalized")
            for x in _extract_meta_column(meta, "time_normalized", n)
        ]
        ret["time_normalized"] = torch.tensor(
            time_vals, dtype=torch.float32, device=device
        ).unsqueeze(-1)

    if encode_rel_int:
        rel_vals = _extract_meta_column(meta, "relative_intensity_single_region", n)
        if all(x is None for x in rel_vals):
            reg_means = _extract_meta_column(meta, "mean_intensity_single_region", n)
            die_means = _extract_meta_column(meta, "mean_intensity_single_die", n)
            if any(x is not None for x in reg_means) and any(x is not None for x in die_means):
                rel_vals = [
                    float(r) / max(float(d), 1e-12)
                    for r, d in zip(reg_means, die_means)
                ]
            else:
                rel_vals = [1.0] * n
        ret["relative_intensity_single_region"] = torch.tensor(
            [_parse_required_float(x, "relative_intensity_single_region") for x in rel_vals],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)

    if encode_elements:
        raw_mat = cfg.get("material_elements")
        if raw_mat is None:
            raw_mat = cfg.get("materials")
        if raw_mat is None:
            raise ValueError(
                "prepare_film_condition_tensor requires 'material_elements' "
                "(or 'materials') in config when encode_elements=True."
            )

        # Normalize raw_mat to dict[str, list[str]]
        mat_elements: dict[str, list[str]] = {}
        for m_k, m_v in raw_mat.items():
            if isinstance(m_v, dict) and "elements" in m_v:
                mat_elements[str(m_k)] = [str(x).strip() for x in m_v["elements"]]
            elif isinstance(m_v, (list, tuple, set)):
                mat_elements[str(m_k)] = [str(x).strip() for x in m_v]
            else:
                mat_elements[str(m_k)] = [str(m_v).strip()]

        elem_vocab = cfg.get("element_vocab")
        if elem_vocab is None or len(elem_vocab) == 0:
            from utility.config_loader import extract_element_vocabulary
            elem_vocab = extract_element_vocabulary(mat_elements)

        mat_names = _extract_meta_column(meta, "material", n)
        multi_hot_list: list[list[float]] = []
        for m in mat_names:
            if m is None or pd.isna(m):
                raise ValueError("Metadata field 'material' is missing or null.")
            m_str = str(m)
            if m_str not in mat_elements:
                raise KeyError(
                    f"Material '{m_str}' not found in material definitions. "
                    f"Available materials: {sorted(mat_elements.keys())}."
                )
            active_set = set(mat_elements[m_str])
            multi_hot_list.append(
                [1.0 if elem in active_set else 0.0 for elem in elem_vocab]
            )
        ret["element_multi_hot"] = torch.tensor(
            multi_hot_list, dtype=torch.float32, device=device
        )

    reg_mapping = cfg.get("region_index_mapping") or cfg.get("region_mapping")
    reg_vals = _extract_meta_column(meta, "region_index", n)
    if all(x is None for x in reg_vals):
        reg_vals = _extract_meta_column(meta, "region", n)
    if encode_region or any(x is not None for x in reg_vals):
        ret["region"] = torch.tensor(
            [
                _parse_region_index(
                    x, "region_index", region_mapping=reg_mapping
                )
                for x in reg_vals
            ],
            dtype=torch.long,
            device=device,
        )

    we_data = energy_bounds if energy_bounds is not None else window_energy
    if we_data is None:
        we_from_meta = _extract_meta_column(meta, "energy_bounds", n)
        if all(x is None for x in we_from_meta):
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
        scaled_we_dev = (
            scaled_we.to(device) if device is not None else scaled_we
        )
        ret["energy_bounds"] = scaled_we_dev
        ret["window_energy"] = scaled_we_dev
    elif encode_energy:
        warnings.warn(
            "encode_energy=True but no energy bounds were provided to "
            "prepare_film_condition_tensor; defaulting to zeros."
        )
        scaled_we = torch.zeros((n, 2), dtype=torch.float32, device=device)
        ret["energy_bounds"] = scaled_we
        ret["window_energy"] = scaled_we

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
        - 'encode_die' (bool): Encode die identity (default True).
        - 'num_dies' (int): Total unique dies (default 9).
        - 'die_embed_dim' (int): Die embedding dim (default 4).
        - 'encode_tool' (bool): Encode tool identities (default True).
        - 'num_tools' (int): Maximum tools count (default 8).
        - 'tool_embed_dim' (int): Tool embedding dim (default 4).
        - 'num_regions' (int): Maximum regions count (default 16).
        - 'region_embed_dim' (int): Region embedding dim (default 4).
        - 'encode_region' (bool): Encode region identity (default False).
        - 'hidden_dim' (int): FiLM MLP hidden dim (default 32).
        - 'encode_flux' (bool): Encode standardized die flux (default True).
        - 'encode_time' (bool): Encode 'time_normalized' (default True).
        - 'encode_relative_intensity' (bool): Encode relative regional
          intensity.
        - 'encode_elements' (bool): Encode material element multi-hot vector.
        - 'element_vocab' (list[str] | None): Dynamic element vocabulary.
        - 'material_elements' (dict[str, list[str]] | None): Material map.
        - 'use_sliding_window' (bool): Whether sliding window is used.
        - 'encode_energy' (bool): Whether to encode bounding energy.
    """

    def __init__(
        self, channels_list: list[int], config: dict[str, Any] | None = None
    ) -> None:
        super().__init__()
        cfg = config or {}
        self.encode_die = bool(cfg.get("encode_die", True))
        self.encode_tool = bool(cfg.get("encode_tool", True))
        self.encode_flux = bool(
            cfg.get("encode_flux", cfg.get("use_flux", True))
        )
        self.encode_time = bool(
            cfg.get("encode_time", cfg.get("use_time", True))
        )
        self.encode_region = bool(
            cfg.get("encode_region", cfg.get("use_region", False))
        )
        self.encode_relative_intensity = bool(
            cfg.get("encode_relative_intensity", False)
        )
        self.encode_elements = bool(
            cfg.get("encode_elements", False)
        )
        self.encode_energy = bool(
            cfg.get("encode_energy", cfg.get("encode_window_energy", False))
        )
        self.encode_window_energy = self.encode_energy

        all_encodings = [
            self.encode_die,
            self.encode_tool,
            self.encode_flux,
            self.encode_time,
            self.encode_region,
            self.encode_relative_intensity,
            self.encode_elements,
            self.encode_energy,
        ]
        if not any(all_encodings):
            raise ValueError(
                "FiLM is enabled, but all feature encodings are disabled "
                "(encode_die, encode_tool, encode_flux, encode_time, encode_region, "
                "encode_relative_intensity, encode_elements, encode_energy are all False). "
                "At least one feature encoding must be enabled when FiLM is active."
            )

        num_dies = int(cfg.get("num_dies", 9))
        die_dim = int(cfg.get("die_embed_dim", 4)) if self.encode_die else 0
        num_tools = int(cfg.get("num_tools", 8))
        tool_dim = int(cfg.get("tool_embed_dim", 4)) if self.encode_tool else 0
        num_regions = int(cfg.get("num_regions", 16))
        region_dim = (
            int(cfg.get("region_embed_dim", 4)) if self.encode_region else 0
        )
        hidden_dim = int(cfg.get("hidden_dim", 32))

        self.element_vocab = cfg.get("element_vocab")
        self.material_elements = cfg.get("material_elements")
        if self.encode_elements:
            if not self.material_elements and "materials" in cfg:
                self.material_elements = cfg.get("materials")
            has_vocab_size = bool(cfg.get("element_vocab_size"))
            if (
                not self.material_elements
                and not self.element_vocab
                and not has_vocab_size
            ):
                raise ValueError(
                    "FiLMGenerator requires 'material_elements' in config "
                    "when encode_elements=True."
                )
            if self.material_elements and (
                not self.element_vocab or len(self.element_vocab) == 0
            ):
                from utility.config_loader import extract_element_vocabulary

                self.element_vocab = extract_element_vocabulary(
                    self.material_elements
                )

        flux_dim = 1 if self.encode_flux else 0
        time_dim = 1 if self.encode_time else 0
        energy_dim = 2 if self.encode_energy else 0
        rel_int_dim = 1 if self.encode_relative_intensity else 0
        elem_dim = (
            (
                len(self.element_vocab)
                if (self.encode_elements and self.element_vocab)
                else int(cfg.get("element_vocab_size", 0))
            )
            if self.encode_elements
            else 0
        )

        in_dim = (
            die_dim
            + (2 * tool_dim)
            + region_dim
            + flux_dim
            + time_dim
            + energy_dim
            + rel_int_dim
            + elem_dim
        )
        self.channels_list = list(channels_list)
        self.die_embedding = (
            nn.Embedding(num_dies, die_dim) if self.encode_die else None
        )
        self.tool_embedding = (
            nn.Embedding(num_tools, tool_dim) if self.encode_tool else None
        )
        if self.encode_region:
            self.region_embedding = nn.Embedding(num_regions, region_dim)
        else:
            self.region_embedding = None

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
            Conditioning features dictionary.

        Returns
        -------
        list[tuple[torch.Tensor, torch.Tensor]]
            List of (gamma, beta) modulation pairs per target layer.
        """
        target_device = self.mlp[0].weight.device
        cond_dev = {
            k: (
                v.to(target_device)
                if isinstance(v, torch.Tensor) and v.device != target_device
                else v
            )
            for k, v in cond.items()
        }
        feat_list: list[torch.Tensor] = []
        if self.encode_die:
            if "die" not in cond_dev:
                raise KeyError(
                    "FiLMGenerator expects 'die' in conditioning dictionary "
                    "when encode_die=True."
                )
            feat_list.append(self.die_embedding(cond_dev["die"]))

        if self.encode_tool:
            if "tool_src" not in cond_dev or "tool_tgt" not in cond_dev:
                raise KeyError(
                    "FiLMGenerator expects 'tool_src' and 'tool_tgt' in "
                    "conditioning dictionary when encode_tool=True."
                )
            feat_list.append(self.tool_embedding(cond_dev["tool_src"]))
            feat_list.append(self.tool_embedding(cond_dev["tool_tgt"]))

        if self.encode_region:
            reg_t = cond_dev.get("region")
            if reg_t is None:
                b_size = next(
                    (
                        v.shape[0]
                        for v in cond_dev.values()
                        if isinstance(v, torch.Tensor)
                    ),
                    1,
                )
                reg_t = torch.zeros(
                    b_size, dtype=torch.long, device=target_device
                )
            else:
                max_allowed = self.region_embedding.num_embeddings - 1
                if (reg_t > max_allowed).any() or (reg_t < 0).any():
                    warnings.warn(
                        f"Region index outside expected [0, {max_allowed}]; "
                        f"clamping indices to [0, {max_allowed}]."
                    )
                    reg_t = torch.clamp(reg_t, 0, max_allowed)
            feat_list.append(self.region_embedding(reg_t))
        if self.encode_flux:
            flux_t = cond_dev.get("total_flux_single_die_normalized")
            if flux_t is None:
                flux_t = cond_dev.get("die_flux_normalized")
            if flux_t is None:
                raise KeyError(
                    "FiLMGenerator expects 'total_flux_single_die_normalized' in "
                    "conditioning dictionary when encode_flux=True."
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

        if self.encode_relative_intensity:
            rel_t = cond_dev.get("relative_intensity_single_region")
            if rel_t is None:
                raise KeyError(
                    "FiLMGenerator expects 'relative_intensity_single_region' in "
                    "conditioning dictionary when encode_relative_intensity=True."
                )
            feat_list.append(rel_t)

        if self.encode_elements:
            elem_t = cond_dev.get("element_multi_hot")
            if elem_t is None:
                raise KeyError(
                    "FiLMGenerator expects 'element_multi_hot' in conditioning "
                    "dictionary when encode_elements=True."
                )
            feat_list.append(elem_t)

        if self.encode_energy:
            energy_t = cond_dev.get("energy_bounds")
            if energy_t is None:
                energy_t = cond_dev.get("window_energy")
            if energy_t is None:
                raise KeyError(
                    "FiLMGenerator expects 'energy_bounds' (or 'window_energy') "
                    "in conditioning dictionary when encode_energy=True."
                )
            feat_list.append(energy_t)

        h = self.mlp(torch.cat(feat_list, dim=-1))
        return [
            (1.0 + gamma_proj(h), beta_proj(h))
            for gamma_proj, beta_proj in zip(self.gamma_projs, self.beta_projs)
        ]
