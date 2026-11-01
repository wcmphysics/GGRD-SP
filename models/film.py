"""Feature-wise Linear Modulation (FiLM) module.

Enables dynamic feature conditioning using metadata (die one-hot,
tool one-hot, standardized measurement time, standardized source
die flux, relative regional intensity, region one-hot, element multi-hot,
and energy bounds) without target data leakage.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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
        if n == 1:
            return [val]
        if isinstance(val, (list, tuple, np.ndarray, torch.Tensor)) and not isinstance(
            val, (str, bytes)
        ):
            return list(val)
        return [val] * n
    if isinstance(meta, list) and meta:
        return [m.get(key) if isinstance(m, dict) else None for m in meta]
    return [None] * n


def _extract_vector_column(
    meta: Any, key: str, n: int, device: torch.device | None
) -> torch.Tensor:
    """Extract 2D float tensor of shape (N, D) for pre-computed one-hot or multi-hot."""
    raw_vals = _extract_meta_column(meta, key, n)
    if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in raw_vals):
        raise ValueError(
            f"Metadata field '{key}' contains missing, null, or unpaired entries. "
            f"Ensure '{key}' is properly populated in df_meta before FiLM conditioning."
        )
    try:
        arr = np.asarray(raw_vals, dtype=np.float32)
        if arr.ndim == 1 and n == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(
                f"Expected 2D array for '{key}', got shape {arr.shape}."
            )
        t = torch.from_numpy(arr)
        return t.to(device) if device is not None else t
    except Exception as exc:
        raise ValueError(f"Failed to convert '{key}' into float tensor: {exc}") from exc


def _extract_categorical_one_hot_tensors(
    meta: Any,
    n: int,
    device: torch.device | None,
    encode_die: bool,
    encode_tool: bool,
    encode_region: bool,
) -> dict[str, torch.Tensor]:
    """Extract pre-computed one-hot tensors for die, tools, and region."""
    ret: dict[str, torch.Tensor] = {}
    if encode_die:
        ret["die_one_hot"] = _extract_vector_column(meta, "die_one_hot", n, device)
    if encode_tool:
        ret["source_tool_one_hot"] = _extract_vector_column(
            meta, "source_tool_one_hot", n, device
        )
        ret["target_tool_one_hot"] = _extract_vector_column(
            meta, "target_tool_one_hot", n, device
        )
    if encode_region:
        ret["region_one_hot"] = _extract_vector_column(
            meta, "region_one_hot", n, device
        )
    return ret


def _extract_continuous_condition_tensors(
    meta: Any,
    n: int,
    device: torch.device | None,
    encode_flux: bool,
    encode_time: bool,
    encode_rel_int: bool,
) -> dict[str, torch.Tensor]:
    """Extract continuous numerical condition tensors (flux, time, rel_int)."""
    ret: dict[str, torch.Tensor] = {}

    if encode_flux:
        flux_vals = _extract_meta_column(
            meta, "total_flux_single_die_standardized", n
        )
        ret["total_flux_single_die_standardized"] = torch.tensor(
            [
                _parse_required_float(x, "total_flux_single_die_standardized")
                for x in flux_vals
            ],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)

    if encode_time:
        time_vals = [
            _parse_required_float(x, "time_standardized")
            for x in _extract_meta_column(meta, "time_standardized", n)
        ]
        ret["time_standardized"] = torch.tensor(
            time_vals, dtype=torch.float32, device=device
        ).unsqueeze(-1)

    if encode_rel_int:
        rel_vals = _extract_meta_column(
            meta, "relative_intensity_single_region", n
        )
        ret["relative_intensity_single_region"] = torch.tensor(
            [
                _parse_required_float(x, "relative_intensity_single_region")
                for x in rel_vals
            ],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)

    return ret


def _extract_energy_bounds_tensor(
    meta: Any,
    n: int,
    device: torch.device | None,
    energy_bounds: torch.Tensor | np.ndarray | None,
    energy_scale_ev: float,
    encode_energy: bool,
) -> torch.Tensor | None:
    """Extract and scale bounding binding energy interval."""
    we_data = energy_bounds
    if we_data is None:
        we_from_meta = _extract_meta_column(meta, "energy_bounds", n)
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
        return scaled_we.to(device) if device is not None else scaled_we
    elif encode_energy:
        raise ValueError(
            "encode_energy=True but no energy bounds were provided to "
            "prepare_film_condition_tensor."
        )
    return None


def _is_single_sample_record(d: dict[str, Any], default_num_samples: int) -> bool:
    """Check if dictionary represents a single sample record rather than column batches."""
    if default_num_samples > 1:
        scalar_keys = (
            "time_standardized",
            "total_flux_single_die_standardized",
            "relative_intensity_single_region",
        )
        for k in scalar_keys:
            if k in d:
                val = d[k]
                if isinstance(val, (list, tuple, np.ndarray, torch.Tensor)):
                    if len(val) == default_num_samples:
                        return False
        vector_keys = (
            "die_one_hot",
            "source_tool_one_hot",
            "target_tool_one_hot",
            "region_one_hot",
            "element_multi_hot",
        )
        for k in vector_keys:
            if k in d:
                val = d[k]
                if isinstance(val, (list, tuple)) and len(val) == default_num_samples:
                    if len(val) > 0 and isinstance(val[0], (list, tuple, np.ndarray, torch.Tensor)):
                        return False
                if isinstance(val, (np.ndarray, torch.Tensor)) and val.ndim >= 2:
                    return False
        return False

    for v in d.values():
        if v is None:
            return True
        if np.isscalar(v) and (pd.isna(v) or isinstance(v, (int, float, str, np.number))):
            return True
        if isinstance(v, (list, tuple)) and len(v) > 0 and not isinstance(
            v[0], (list, tuple, np.ndarray, torch.Tensor)
        ):
            return True
        if isinstance(v, (np.ndarray, torch.Tensor)) and v.ndim == 1:
            return True
    return False


def _resolve_sample_count(
    metadata: list[dict[str, Any]] | dict[str, Any] | pd.DataFrame | None,
    default_num_samples: int,
) -> tuple[Any, int]:
    """Normalize metadata container and determine batch sample count."""
    meta = (
        metadata.to_dict(orient="records")
        if isinstance(metadata, pd.DataFrame)
        else metadata
    )
    if isinstance(meta, dict):
        if _is_single_sample_record(meta, default_num_samples):
            return [meta], 1
        seq_lens = [
            len(v)
            for v in meta.values()
            if isinstance(v, (list, tuple, np.ndarray, torch.Tensor))
        ]
        return meta, (seq_lens[0] if seq_lens else default_num_samples)
    if isinstance(meta, list) and meta:
        return meta, len(meta)
    return meta, default_num_samples


REQUIRED_FILM_ENCODE_KEYS = (
    "encode_die",
    "encode_tool",
    "encode_flux",
    "encode_time",
    "encode_energy",
    "encode_region",
    "encode_relative_intensity",
    "encode_elements",
)


def _validate_film_config_keys(cfg: dict[str, Any]) -> None:
    """Strictly validate that required encode_* flags are present without fallbacks."""
    missing = [k for k in REQUIRED_FILM_ENCODE_KEYS if k not in cfg]
    if missing:
        raise KeyError(
            f"FiLM configuration missing required keys: {missing}. "
            "Please ensure all encode_* keys are explicitly configured."
        )


def prepare_film_condition_tensor(
    metadata: list[dict[str, Any]] | dict[str, Any] | pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    *,
    device: torch.device | None = None,
    energy_bounds: torch.Tensor | np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """Extract and pack conditioning features from metadata directly without fallbacks.

    Guarantees zero target leakage: only source measurements, die one-hot,
    and explicit tool one-hot are used.
    """
    if config is None:
        raise KeyError("prepare_film_condition_tensor requires an explicit 'config' dict.")
    _validate_film_config_keys(config)

    dev_cfg = config.get("device")
    dev = torch.device(dev_cfg) if dev_cfg is not None else device
    meta, n = _resolve_sample_count(metadata, int(config.get("num_samples", 1)))

    cat_tensors = _extract_categorical_one_hot_tensors(
        meta,
        n,
        dev,
        encode_die=bool(config["encode_die"]),
        encode_tool=bool(config["encode_tool"]),
        encode_region=bool(config["encode_region"]),
    )
    cont_tensors = _extract_continuous_condition_tensors(
        meta,
        n,
        dev,
        encode_flux=bool(config["encode_flux"]),
        encode_time=bool(config["encode_time"]),
        encode_rel_int=bool(config["encode_relative_intensity"]),
    )

    elem_energy_tensors: dict[str, torch.Tensor] = {}
    if bool(config["encode_elements"]):
        elem_energy_tensors["element_multi_hot"] = _extract_vector_column(
            meta, "element_multi_hot", n, dev
        )

    eb = _extract_energy_bounds_tensor(
        meta,
        n,
        dev,
        energy_bounds,
        float(config["energy_scale_ev"]) if "energy_scale_ev" in config else 1000.0,
        bool(config["encode_energy"]),
    )
    if eb is not None:
        elem_energy_tensors["energy_bounds"] = eb

    return {**cat_tensors, **cont_tensors, **elem_energy_tensors}


class FiLMGenerator(nn.Module):
    """Generates feature-wise linear modulation parameters via dense MLP.

    Initializes output projections to zero so that initial
    gamma=1.0 and beta=0.0 for strict identity modulation.
    """

    def __init__(
        self, channels_list: list[int], config: dict[str, Any] | None = None
    ) -> None:
        super().__init__()
        if config is None:
            raise KeyError("FiLMGenerator requires an explicit 'config' dict.")
        _validate_film_config_keys(config)

        self.encode_die = bool(config["encode_die"])
        self.encode_tool = bool(config["encode_tool"])
        self.encode_flux = bool(config["encode_flux"])
        self.encode_time = bool(config["encode_time"])
        self.encode_region = bool(config["encode_region"])
        self.encode_relative_intensity = bool(config["encode_relative_intensity"])
        self.encode_elements = bool(config["encode_elements"])
        self.encode_energy = bool(config["encode_energy"])

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

        if "hidden_dim" not in config:
            raise KeyError("FiLMGenerator requires 'hidden_dim' in config.")
        hidden_dim = int(config["hidden_dim"])

        # Validate and accumulate dynamic dimensions for one-hot and
        # continuous features.
        die_dim = int(config["num_dies"]) if self.encode_die else 0
        tool_dim = (2 * int(config["num_tools"])) if self.encode_tool else 0
        region_dim = int(config["num_regions"]) if self.encode_region else 0
        elem_dim = int(config["num_elements"]) if self.encode_elements else 0
        flux_dim = 1 if self.encode_flux else 0
        time_dim = 1 if self.encode_time else 0
        rel_int_dim = 1 if self.encode_relative_intensity else 0
        energy_dim = 2 if self.encode_energy else 0

        in_dim = (
            die_dim
            + tool_dim
            + region_dim
            + elem_dim
            + flux_dim
            + time_dim
            + rel_int_dim
            + energy_dim
        )
        self.in_dim = in_dim
        self.channels_list = list(channels_list)
        self.config = dict(config)

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
        """Compute affine (gamma, beta) modulation parameters per layer."""
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
            if "die_one_hot" not in cond_dev:
                raise KeyError(
                    "FiLMGenerator expects 'die_one_hot' in conditioning dictionary "
                    "when encode_die=True."
                )
            feat_list.append(cond_dev["die_one_hot"])

        if self.encode_tool:
            if (
                "source_tool_one_hot" not in cond_dev
                or "target_tool_one_hot" not in cond_dev
            ):
                raise KeyError(
                    "FiLMGenerator expects 'source_tool_one_hot' and 'target_tool_one_hot' in "
                    "conditioning dictionary when encode_tool=True."
                )
            feat_list.append(cond_dev["source_tool_one_hot"])
            feat_list.append(cond_dev["target_tool_one_hot"])

        if self.encode_region:
            if "region_one_hot" not in cond_dev:
                raise KeyError(
                    "FiLMGenerator expects 'region_one_hot' in conditioning dictionary "
                    "when encode_region=True."
                )
            feat_list.append(cond_dev["region_one_hot"])

        if self.encode_elements:
            if "element_multi_hot" not in cond_dev:
                raise KeyError(
                    "FiLMGenerator expects 'element_multi_hot' in conditioning dictionary "
                    "when encode_elements=True."
                )
            feat_list.append(cond_dev["element_multi_hot"])

        if self.encode_flux:
            flux_t = cond_dev.get("total_flux_single_die_standardized")
            if flux_t is None:
                raise KeyError(
                    "FiLMGenerator expects 'total_flux_single_die_standardized' in "
                    "conditioning dictionary when encode_flux=True."
                )
            feat_list.append(flux_t)

        if self.encode_time:
            time_t = cond_dev.get("time_standardized")
            if time_t is None:
                raise KeyError(
                    "FiLMGenerator expects 'time_standardized' in conditioning "
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

        if self.encode_energy:
            energy_t = cond_dev.get("energy_bounds")
            if energy_t is None:
                raise KeyError(
                    "FiLMGenerator expects 'energy_bounds' in conditioning "
                    "dictionary when encode_energy=True."
                )
            feat_list.append(energy_t)

        h = self.mlp(torch.cat(feat_list, dim=-1))
        return [
            (1.0 + gamma_proj(h), beta_proj(h))
            for gamma_proj, beta_proj in zip(self.gamma_projs, self.beta_projs)
        ]

    def get_condition_config(
        self,
        num_samples: int = 1,
        energy_scale_ev: float = 1000.0,
        extra_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Export canonical conditioning configuration matching this generator."""
        base: dict[str, Any] = {
            **self.config,
            "num_samples": num_samples,
            "energy_scale_ev": energy_scale_ev,
            "encode_die": self.encode_die,
            "encode_tool": self.encode_tool,
            "encode_flux": self.encode_flux,
            "encode_time": self.encode_time,
            "encode_region": self.encode_region,
            "encode_relative_intensity": self.encode_relative_intensity,
            "encode_elements": self.encode_elements,
            "encode_energy": self.encode_energy,
        }
        if extra_config:
            base.update(extra_config)
        return base

    def prepare_condition_tensor(
        self,
        metadata: list[dict[str, Any]] | dict[str, Any] | pd.DataFrame | None,
        *,
        device: torch.device | None = None,
        energy_bounds: torch.Tensor | np.ndarray | None = None,
        energy_scale_ev: float = 1000.0,
        extra_config: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Pack conditioning tensors using this generator's configuration."""
        n_samples = (
            len(metadata)
            if isinstance(metadata, (list, tuple, pd.DataFrame)) and len(metadata) > 0
            else 1
        )
        cfg = self.get_condition_config(
            num_samples=n_samples,
            energy_scale_ev=energy_scale_ev,
            extra_config=extra_config,
        )
        return prepare_film_condition_tensor(
            metadata,
            config=cfg,
            device=device,
            energy_bounds=energy_bounds,
        )
