"""Pseudo-measurement generation module for GGRD-SP.

Generates synthetic XPS regional spectra and associated metadata according to
project specifications in AGENTS.md.
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from utility.material_profiles import (
    compute_intrinsic_profile_max,
    resolve_material_region_profile,
    resolve_material_region_strength,
)


# Default die-to-die spatial baseline factors for NMG (center > mid > edge)
DEFAULT_DIE_FACTORS_NMG: dict[int, float] = {
    0: 1.20,  # Center (Die 0)
    1: 0.82,  # Edge (Die 1)
    2: 0.78,  # Edge (Die 2)
    3: 0.81,  # Edge (Die 3)
    4: 0.77,  # Edge (Die 4)
    5: 1.02,  # Mid (Die 5)
    6: 0.99,  # Mid (Die 6)
    7: 1.01,  # Mid (Die 7)
    8: 0.98,  # Mid (Die 8)
}


def _parse_start_time(start_time: str | datetime) -> datetime:
    """Parse start_time string or return datetime directly."""
    return (
        datetime.fromisoformat(start_time)
        if isinstance(start_time, str)
        else start_time
    )


def pseudo_voigt(
    energy: np.ndarray,
    peak_params: dict[str, Any],
) -> np.ndarray:
    """Calculate a pseudo-Voigt profile (Gaussian + Lorentzian combination).

    Parameters
    ----------
    energy : np.ndarray
        1D array of binding energy points.
    peak_params : dict[str, Any]
        Dictionary packaging peak configuration:
        - 'center' (float): Peak center position in eV.
        - 'amplitude' (float): Peak height/amplitude.
        - 'fwhm' (float): Full width at half maximum in eV (must be > 0).
        - 'eta' (float, optional): Lorentzian fraction in [0.0, 1.0] (def 0.3).

    Returns
    -------
    np.ndarray
        Evaluated peak intensity profile.

    Raises
    ------
    ValueError
        If fwhm is not strictly positive or required keys are missing.
    """
    for req_key in ("center", "amplitude", "fwhm"):
        if req_key not in peak_params:
            raise ValueError(f"Missing required key '{req_key}' in peak_params")

    center = float(peak_params["center"])
    amplitude = float(peak_params["amplitude"])
    fwhm = float(peak_params["fwhm"])
    eta = float(peak_params.get("eta", 0.3))

    if fwhm <= 0.0:
        raise ValueError(f"fwhm must be strictly positive, got {fwhm}")

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0

    gaussian = np.exp(-0.5 * ((energy - center) / sigma) ** 2)
    lorentzian = 1.0 / (1.0 + ((energy - center) / gamma) ** 2)

    return amplitude * ((1.0 - eta) * gaussian + eta * lorentzian)


def calculate_time_decay_factor(
    meas_time: datetime,
    base_time: datetime,
    config: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
    cycle_cache: dict[int, float] | None = None,
) -> float:
    """Calculate the time decay scale factor with maintenance jump resets.

    Models linear decay over operating days with periodic maintenance resets,
    capturing renewed intensity (jump up) after optics cleaning or filament
    replacement.

    Parameters
    ----------
    meas_time : datetime
        Timestamp of the measurement.
    base_time : datetime
        Reference start timestamp.
    config : dict[str, Any] | None, optional
        Configuration dictionary with decay parameters (can package 'rng',
        'cycle_cache').
    rng : np.random.Generator | None, optional
        Random number generator.
    cycle_cache : dict[int, float] | None, optional
        Dictionary caching post-maintenance renewed intensity levels per cycle.

    Returns
    -------
    float
        Multiplicative time-dependent intensity scale factor.
    """
    cfg = dict(config or {})
    rng = rng or cfg.get("rng") or np.random.default_rng()
    cycle_cache = (
        cycle_cache if cycle_cache is not None else cfg.get("cycle_cache")
    )

    decay_rate = float(cfg.get("time_decay_rate", 0.005))
    if decay_rate <= 0.0:
        return 1.0

    interval_days = float(cfg.get("maintenance_interval_days", 30.0))
    if interval_days <= 0.0:
        interval_days = 30.0
    jump_std = max(0.0, float(cfg.get("maintenance_jump_noise_std", 0.03)))
    decay_floor = float(cfg.get("decay_floor", 0.2))

    delta_days = max(0.0, (meas_time - base_time).total_seconds() / 86400.0)
    cycle_idx = int(delta_days // interval_days)
    days_in_cycle = delta_days - (cycle_idx * interval_days)

    if cycle_cache is not None:
        if cycle_idx not in cycle_cache:
            reset_level = (
                1.0
                if (cycle_idx == 0 or jump_std == 0.0)
                else float(rng.normal(1.0, jump_std))
            )
            cycle_cache[cycle_idx] = float(np.clip(reset_level, 0.8, 1.3))
        base_level = cycle_cache[cycle_idx]
    else:
        base_level = 1.0

    scale = base_level - (decay_rate * days_in_cycle)
    return float(np.clip(scale, decay_floor, 2.0))


def get_default_measurement_config() -> dict[str, Any]:
    """Return the default configuration dictionary for pseudo-measurements.

    Returns
    -------
    dict[str, Any]
        Configuration options bundling multi-material and recipe parameters.
    """
    return {
        "materials": {
            "TiAlC": {
                "regions": ["Al2p", "Ti2p", "C1s", "O1s", "Si2p", "Cl2p"],
            },
            "TiN": {
                "regions": ["Ti2p", "N1s", "O1s", "C1s", "Si2p"],
            },
            "HfO2": {
                "regions": ["Hf4f", "O1s", "C1s", "Si2p"],
            },
        },
        "recipes": {
            "RCP_NMG_A": {
                "Al2p": (70.0, 78.0),
                "Ti2p": (452.0, 470.0),
                "C1s": (280.0, 292.0),
                "O1s": (526.0, 536.0),
                "Cl2p": (196.0, 206.0),
            },
            "RCP_NMG_B": {
                "Ti2p": (450.0, 468.0),
                "O1s": (527.0, 537.0),
                "Si2p": (97.0, 107.0),
            },
            "RCP_PMG_A": {
                "Ti2p": (452.0, 470.0),
                "N1s": (394.0, 404.0),
                "O1s": (526.0, 536.0),
                "C1s": (280.0, 292.0),
            },
            "RCP_PMG_B": {
                "Ti2p": (451.0, 469.0),
                "N1s": (395.0, 405.0),
                "O1s": (527.0, 537.0),
                "Al2p": (70.0, 78.0),
                "Si2p": (97.0, 107.0),
            },
            "RCP_HK_A": {
                "Hf4f": (13.0, 23.0),
                "O1s": (526.0, 536.0),
                "Si2p": (97.0, 107.0),
                "Ti2p": (452.0, 470.0),
            },
        },
        "material_recipes": {
            "TiAlC": {
                "RCP_NMG_A": {"J4": 6, "J5": 8},
                "RCP_NMG_B": {"J4": 5, "J5": 6},
                "RCP_PMG_A": {"J4": 5, "J5": 5},
            },
            "TiN": {
                "RCP_PMG_A": {"J4": 6, "J5": 8},
                "RCP_PMG_B": {"J4": 5, "J5": 6},
            },
            "HfO2": {
                "RCP_HK_A": {"J4": 6, "J5": 8},
            },
        },
        "n_die": 9,
        "n_points": 100,
        "measurements_per_t7_code": 3,
        "start_time": "2026-01-01 00:00:00",
        "interval_hours_range": (4.0, 24.0),
        "tool_offsets": {
            "J4": {"shift_ev": 0.0, "scale": 1.0},
            "J5": {"shift_ev": 0.2, "scale": 1.1},
        },
        "die_variation_std": 0.015,
        "die_base_factors": DEFAULT_DIE_FACTORS_NMG,
        "die_repeatability_std": 0.015,
        "time_decay_rate": 0.005,
        "maintenance_interval_days": 30.0,
        "maintenance_jump_noise_std": 0.03,
        "decay_floor": 0.2,
        "noise_relative_std": 0.015,
        "background_relative_strength": 0.5,
        "shirley_background": {"enabled": True, "step_ratio": 0.10},
        "seed": 42,
    }


def _validate_identifier(
    name: Any, kind: str, *, forbid_underscore: bool = False
) -> None:
    """Validate identifier string against measurement_id delimiter rules."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"{kind} identifier must be a non-empty string, got {name!r}."
        )
    if "-" in name:
        raise ValueError(
            f"{kind} identifier {name!r} must not contain '-'; "
            "'-' is reserved as the measurement_id field delimiter."
        )
    if forbid_underscore and "_" in name:
        raise ValueError(
            f"{kind} identifier {name!r} must not contain '_'; "
            "'_' is reserved for predicted tool pairs '{source}_{target}'."
        )


def _as_dict(val: Any) -> dict[str, Any]:
    """Convert a dict or Pydantic model to a plain dictionary."""
    if hasattr(val, "model_dump"):
        return dict(val.model_dump())
    if isinstance(val, dict):
        return val
    return {}


def _validate_materials_config(materials: Any) -> None:
    """Validate materials dictionary and per-material region lists."""
    if not isinstance(materials, dict) or not materials:
        raise ValueError(
            "'materials' must be a non-empty dictionary mapping material names "
            "to material specifications."
        )
    for mat_name, raw_spec in materials.items():
        _validate_identifier(mat_name, "Material")
        spec = _as_dict(raw_spec)
        regions = spec.get("regions")
        if not isinstance(regions, (list, tuple)) or not regions:
            raise ValueError(
                f"Material '{mat_name}' must specify a non-empty 'regions' "
                f"list, got {regions!r}."
            )
        _validate_regional_shirley(spec.get("region_profiles"))


def _validate_single_recipe_regions(
    recipe_name: str, region_map: Any
) -> None:
    """Validate region mapping and [e_min, e_max] bounds for one recipe."""
    if not isinstance(region_map, dict) or not region_map:
        raise ValueError(
            f"Recipe '{recipe_name}' must probe at least one spectral region."
        )
    for region_name, bounds in region_map.items():
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(
                f"Recipe '{recipe_name}' region '{region_name}' energy range "
                f"must be a 2-element [e_min, e_max], got {bounds!r}."
            )
        e_min, e_max = float(bounds[0]), float(bounds[1])
        if not (np.isfinite(e_min) and np.isfinite(e_max)) or e_min >= e_max:
            raise ValueError(
                f"Recipe '{recipe_name}' region '{region_name}' energy range "
                f"must satisfy finite e_min < e_max, got ({e_min}, {e_max})."
            )


def _validate_recipes_config(recipes: Any) -> None:
    """Validate recipes dictionary and per-region [e_min, e_max] bounds."""
    if not isinstance(recipes, dict) or not recipes:
        raise ValueError(
            "'recipes' must be a non-empty dictionary mapping recipe names "
            "to regional energy ranges."
        )
    for recipe_name, region_map in recipes.items():
        _validate_identifier(recipe_name, "Recipe")
        _validate_single_recipe_regions(recipe_name, region_map)


def _validate_group_tool_counts(
    mat_name: str, rcp_name: str, tool_counts: Any
) -> int:
    """Validate {tool: count} dictionary for one (material, recipe) group."""
    if not isinstance(tool_counts, dict) or not tool_counts:
        raise ValueError(
            f"material_recipes['{mat_name}']['{rcp_name}'] must be a "
            "non-empty dict of {tool: count}."
        )
    subtotal = 0
    for tool_name, count in tool_counts.items():
        _validate_identifier(tool_name, "Tool", forbid_underscore=True)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"Measurement count for ({mat_name!r}, {rcp_name!r}, "
                f"{tool_name!r}) must be a non-negative integer, got {count!r}."
            )
        subtotal += count
    return subtotal


def _validate_material_recipes_config(
    material_recipes: Any,
    materials: dict[str, Any],
    recipes: dict[str, Any],
) -> None:
    """Validate material -> recipe -> {tool: count} schedule."""
    if not isinstance(material_recipes, dict) or not material_recipes:
        raise ValueError(
            "'material_recipes' must be a non-empty dictionary mapping "
            "material -> recipe -> {tool: count}."
        )
    total_count = 0
    for mat_name, rcp_map in material_recipes.items():
        if mat_name not in materials:
            raise ValueError(
                f"material_recipes references unknown material '{mat_name}'. "
                f"Defined materials: {sorted(materials.keys())}."
            )
        if not isinstance(rcp_map, dict) or not rcp_map:
            raise ValueError(
                f"material_recipes['{mat_name}'] must map at least one recipe "
                "to tool measurement counts."
            )
        for rcp_name, tool_counts in rcp_map.items():
            if rcp_name not in recipes:
                raise ValueError(
                    f"material_recipes['{mat_name}'] references unknown recipe "
                    f"'{rcp_name}'. Defined recipes: {sorted(recipes.keys())}."
                )
            total_count += _validate_group_tool_counts(
                mat_name, rcp_name, tool_counts
            )
    if total_count <= 0:
        raise ValueError(
            "material_recipes must specify at least one (material, recipe, "
            "tool) group with measurement count > 0."
        )


def _validate_positive_int(val: Any, name: str, min_val: int = 1) -> None:
    """Validate that val is an integer >= min_val (rejecting booleans)."""
    if isinstance(val, bool) or not isinstance(val, int) or val < min_val:
        raise ValueError(f"{name} must be an integer >= {min_val}, got {val}")


def _validate_numeric_config(config: dict[str, Any]) -> None:
    """Validate scalar counts, background strength, and time interval range."""
    _validate_positive_int(config.get("n_points"), "n_points", min_val=2)
    _validate_positive_int(config.get("n_die"), "n_die", min_val=1)
    _validate_positive_int(
        config.get("measurements_per_t7_code", 10), "measurements_per_t7_code"
    )

    bg_strength = config.get("background_relative_strength", 0.5)
    if (
        isinstance(bg_strength, bool)
        or not isinstance(bg_strength, (int, float))
        or not np.isfinite(bg_strength)
        or float(bg_strength) <= 0.0
    ):
        raise ValueError(
            "background_relative_strength must be a positive finite number, "
            f"got {bg_strength!r}."
        )

    interval_range = config.get("interval_hours_range", (4.0, 24.0))
    if not (isinstance(interval_range, (tuple, list)) and len(interval_range) == 2):
        raise ValueError(
            f"interval_hours_range must be a tuple/list of 2 floats, "
            f"got {interval_range}"
        )
    if interval_range[0] < 0 or interval_range[1] <= interval_range[0]:
        raise ValueError(
            "interval_hours_range must satisfy min >= 0 and max > min, "
            f"got {interval_range}"
        )


def _validate_config(config: dict[str, Any]) -> None:
    """Validate configuration dictionary parameters."""
    _validate_numeric_config(config)
    _validate_materials_config(config.get("materials"))
    _validate_recipes_config(config.get("recipes"))
    _validate_material_recipes_config(
        config.get("material_recipes"),
        config["materials"],
        config["recipes"],
    )
    for tool_name in (config.get("tool_offsets") or {}):
        _validate_identifier(tool_name, "Tool", forbid_underscore=True)
    _validate_shirley_config(config)


def _validate_step_ratio(value: Any, name: str) -> None:
    """Validate that a step ratio is a finite float in [0.0, 1.0]."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value < 0.0
        or value > 1.0
    ):
        raise ValueError(f"{name} must be a float in [0.0, 1.0], got {value}")


def _validate_regional_shirley(region_profiles: Any) -> None:
    """Validate Shirley settings in region profiles."""
    if not isinstance(region_profiles, dict):
        return
    for reg_name, profile in region_profiles.items():
        if not isinstance(profile, dict):
            continue
        if "shirley_enabled" in profile and not isinstance(
            profile["shirley_enabled"], bool
        ):
            raise ValueError(
                f"Region '{reg_name}' shirley_enabled must be a boolean, "
                f"got {profile['shirley_enabled']}"
            )
        if "shirley_step_ratio" in profile:
            _validate_step_ratio(
                profile["shirley_step_ratio"],
                f"Region '{reg_name}' shirley_step_ratio",
            )


def _validate_shirley_config(config: dict[str, Any]) -> None:
    """Validate Shirley background configuration and regional overrides."""
    if "shirley_background" in config and config["shirley_background"] is not None:
        shirley_cfg = config["shirley_background"]
        if hasattr(shirley_cfg, "model_dump"):
            shirley_cfg = shirley_cfg.model_dump()
        if not isinstance(shirley_cfg, dict):
            raise ValueError(
                f"'shirley_background' must be a dictionary, got {type(shirley_cfg).__name__}"
            )
        if "enabled" in shirley_cfg and not isinstance(
            shirley_cfg["enabled"], bool
        ):
            raise ValueError(
                f"'shirley_background.enabled' must be a boolean, got {shirley_cfg['enabled']}"
            )
        if "step_ratio" in shirley_cfg:
            _validate_step_ratio(
                shirley_cfg["step_ratio"], "'shirley_background.step_ratio'"
            )


def generate_tool_timeline(
    tools: list[str],
    timeline_config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, list[datetime]]:
    """Generate independent measurement timestamps per tool.

    Parameters
    ----------
    tools : list[str]
        List of tool identifiers (e.g. ['J4', 'J5']).
    timeline_config : dict[str, Any]
        Configuration dictionary containing:
        - 'measurements_per_tool' (dict[str, int])
        - 'start_time' (str | datetime)
        - 'interval_hours_range' (tuple[float, float])
    rng : np.random.Generator
        NumPy random number generator.

    Returns
    -------
    dict[str, list[datetime]]
        Mapping of tool name to chronological list of measurement datetimes.
    """
    measurements_per_tool: dict[str, int] = timeline_config.get(
        "measurements_per_tool", {}
    )
    base_dt = _parse_start_time(
        timeline_config.get("start_time", "2026-01-01 00:00:00")
    )
    interval_range: tuple[float, float] = timeline_config.get(
        "interval_hours_range", (4.0, 24.0)
    )

    timeline: dict[str, list[datetime]] = {}
    for tool in tools:
        n_meas = measurements_per_tool.get(tool, 0)
        current_dt = base_dt + timedelta(hours=float(rng.uniform(0.0, 4.0)))
        tool_times: list[datetime] = []
        for _ in range(n_meas):
            tool_times.append(current_dt)
            delta_hours = float(
                rng.uniform(interval_range[0], interval_range[1])
            )
            current_dt += timedelta(hours=delta_hours)
        timeline[tool] = tool_times

    return timeline


def _calculate_shirley_step(
    e_grid: np.ndarray,
    peak_intensity: np.ndarray,
    step_ratio: float,
) -> np.ndarray:
    """Calculate the Shirley background step along ascending binding energy.

    Parameters
    ----------
    e_grid : np.ndarray
        1D array of binding energies.
    peak_intensity : np.ndarray
        Pure photoelectron peak intensity before baseline or noise.
    step_ratio : float
        Inelastic step height relative to regional peak maximum.

    Returns
    -------
    np.ndarray
        Shirley inelastic scattering background array of same shape as e_grid.
    """
    if step_ratio <= 0.0 or len(e_grid) < 2:
        return np.zeros_like(peak_intensity)

    peak_max = float(np.max(peak_intensity))
    if peak_max <= 0.0:
        return np.zeros_like(peak_intensity)

    # Ensure cumulative integral runs along ascending binding energy order
    is_descending = bool(e_grid[0] > e_grid[-1])
    e_work = e_grid[::-1] if is_descending else e_grid
    i_work = peak_intensity[::-1] if is_descending else peak_intensity

    dx = np.diff(e_work)
    trap_steps = 0.5 * (i_work[:-1] + i_work[1:]) * dx
    cum_area = np.concatenate(([0.0], np.cumsum(trap_steps)))
    total_area = float(cum_area[-1])

    if total_area <= 0.0:
        return np.zeros_like(peak_intensity)

    step_work = step_ratio * peak_max * (cum_area / total_area)
    return step_work[::-1] if is_descending else step_work


def _synthesize_regional_spectrum(
    e_grid: np.ndarray,
    profile: dict[str, Any],
    synth_params: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize a regional XPS spectrum with peaks, baseline, and noise."""
    tool_shift = float(synth_params["tool_shift"])
    total_scale = float(synth_params["total_scale"])
    noise_relative_std = float(synth_params["noise_relative_std"])
    intensity = np.zeros_like(e_grid)

    # 1. Synthesize pure pseudo-Voigt peaks
    for peak in profile["peaks"]:
        peak_shifted = {
            "center": peak["center"] + tool_shift,
            "amplitude": peak["amplitude"] * total_scale,
            "fwhm": peak["fwhm"],
            "eta": peak.get("eta", 0.3),
        }
        intensity += pseudo_voigt(energy=e_grid, peak_params=peak_shifted)

    # 2. Add Shirley inelastic scattering background step if enabled
    shirley_cfg = synth_params.get("shirley_background") or {}
    if hasattr(shirley_cfg, "model_dump"):
        shirley_cfg = shirley_cfg.model_dump()
    elif not isinstance(shirley_cfg, dict):
        shirley_cfg = {}

    is_enabled = bool(profile.get("shirley_enabled", shirley_cfg.get("enabled", True)))
    step_ratio_raw = profile.get(
        "shirley_step_ratio", shirley_cfg.get("step_ratio", 0.10)
    )
    step_ratio = float(step_ratio_raw) if step_ratio_raw is not None else 0.0

    if is_enabled:
        intensity += _calculate_shirley_step(e_grid, intensity, step_ratio)

    # 3. Add flat baseline and Gaussian noise
    intensity += profile.get("baseline", 10.0) * total_scale
    noise_sigma = float(np.max(intensity)) * noise_relative_std
    intensity += rng.normal(0.0, noise_sigma, size=len(e_grid))
    return np.clip(intensity, 0.0, None)


def _synthesize_background_spectrum(
    e_grid: np.ndarray,
    synth_params: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize flat XPS continuum baseline and noise for absent regions."""
    baseline_level = float(synth_params["total_scale"])
    noise_sigma = baseline_level * float(synth_params["noise_relative_std"])
    intensity = baseline_level + rng.normal(0.0, noise_sigma, size=len(e_grid))
    return np.clip(intensity, 0.0, None)


def _build_recipe_energy_grids(
    recipes: dict[str, dict[str, Any]], n_points: int
) -> dict[tuple[str, str], np.ndarray]:
    """Precompute ascending 1D binding energy grids per (recipe, region)."""
    grids: dict[tuple[str, str], np.ndarray] = {}
    for recipe_name, region_map in recipes.items():
        for region_name, bounds in region_map.items():
            e_min, e_max = float(bounds[0]), float(bounds[1])
            grids[(recipe_name, region_name)] = np.linspace(
                e_min, e_max, n_points, dtype=np.float64
            )
    return grids


def _build_material_resources(
    materials: dict[str, Any],
) -> tuple[
    dict[str, set[str]],
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
]:
    """Precompute active regions, peak profiles, strengths, and norm factors."""
    active_regions: dict[str, set[str]] = {}
    profiles: dict[tuple[str, str], dict[str, Any]] = {}
    strengths: dict[tuple[str, str], float] = {}
    norm_factors: dict[tuple[str, str], float] = {}

    for mat_name, raw_spec in materials.items():
        spec = _as_dict(raw_spec)
        reg_list = [str(r) for r in spec["regions"]]
        active_regions[mat_name] = set(reg_list)
        custom_profiles = spec.get("region_profiles")
        custom_strengths = spec.get("relative_strengths")
        for reg_name in reg_list:
            profile = resolve_material_region_profile(
                mat_name, reg_name, custom_profiles=custom_profiles
            )
            strength = resolve_material_region_strength(
                mat_name, reg_name, custom_strengths=custom_strengths
            )
            base_max = compute_intrinsic_profile_max(profile)
            profiles[(mat_name, reg_name)] = profile
            strengths[(mat_name, reg_name)] = strength
            norm_factors[(mat_name, reg_name)] = strength / base_max

    return active_regions, profiles, strengths, norm_factors


def _count_total_spectra(
    material_recipes: dict[str, dict[str, dict[str, int]]],
    recipes: dict[str, dict[str, Any]],
    n_die: int,
) -> int:
    """Count total regional spectra across all (material, recipe, tool)."""
    total = 0
    for rcp_map in material_recipes.values():
        for rcp_name, tool_counts in rcp_map.items():
            n_regions = len(recipes[rcp_name])
            total += sum(tool_counts.values()) * n_die * n_regions
    return total


def _resolve_die_scales_config(
    merged_config: dict[str, Any], config: dict[str, Any] | None
) -> tuple[dict[int, float], float]:
    """Resolve per-die base factors and repeatability standard deviation."""
    n_die = int(merged_config["n_die"])
    die_base_factors_cfg = merged_config.get("die_base_factors")
    if die_base_factors_cfg is None:
        die_base_factors = (
            DEFAULT_DIE_FACTORS_NMG
            if n_die == 9
            else {i: 1.0 for i in range(n_die)}
        )
    else:
        die_base_factors = {
            int(k): float(v) for k, v in die_base_factors_cfg.items()
        }

    cfg_in = config or {}
    if "die_repeatability_std" in cfg_in:
        die_rep_std = float(cfg_in["die_repeatability_std"])
    elif "die_variation_std" in cfg_in:
        die_rep_std = float(cfg_in["die_variation_std"])
    else:
        die_rep_std = float(
            merged_config.get(
                "die_repeatability_std", merged_config["die_variation_std"]
            )
        )
    return die_base_factors, die_rep_std


def _sample_die_scale(
    die_idx: int,
    die_base_factors: dict[int, float],
    die_rep_std: float,
    rng: np.random.Generator,
) -> float:
    """Sample multiplicative illumination scale for a single die."""
    base_die = float(die_base_factors.get(die_idx, 1.0))
    if die_rep_std <= 0.0:
        return base_die
    die_noise = float(rng.normal(0.0, die_rep_std))
    return float(np.clip(base_die * (1.0 + die_noise), 0.1, 3.0))


def _synthesize_single_region_row(
    mat_name: str,
    reg_name: str,
    e_grid: np.ndarray,
    is_present: bool,
    ctx: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Synthesize one regional spectrum (peaks or continuum background)."""
    total_scale = float(ctx["total_scale"])
    noise_rel_std = float(ctx["noise_rel_std"])
    if is_present:
        reg_factor = ctx["mat_norm_factors"][(mat_name, reg_name)]
        rel_strength = ctx["mat_strengths"][(mat_name, reg_name)]
        spec = _synthesize_regional_spectrum(
            e_grid=e_grid,
            profile=ctx["mat_profiles"][(mat_name, reg_name)],
            synth_params={
                "tool_shift": ctx["tool_shift"],
                "total_scale": total_scale * reg_factor,
                "noise_relative_std": noise_rel_std,
                "shirley_background": ctx["shirley_cfg"],
            },
            rng=rng,
        )
        return spec, float(rel_strength)

    bg_strength = float(ctx["bg_rel_strength"])
    spec = _synthesize_background_spectrum(
        e_grid=e_grid,
        synth_params={
            "total_scale": total_scale * bg_strength,
            "noise_relative_std": noise_rel_std,
        },
        rng=rng,
    )
    return spec, bg_strength


def generate_pseudo_measurements(
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate multi-material, recipe-driven synthetic XPS spectra.

    Fulfills Stage 1 of the program structure defined in AGENTS.md.
    Binding energy grids are defined per (recipe, region) in ascending order.
    Regions absent from a material's active region list are synthesized as
    scaled XPS continuum background plus Gaussian noise.

    Parameters
    ----------
    config : dict[str, Any] | None, optional
        Configuration dictionary. If None, default settings from
        `get_default_measurement_config()` are used.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame]
        - ary_intensity: 2D array of shape (N_total, N_points)
        - ary_energy: 2D array of shape (N_total, N_points)
        - df_meta: DataFrame of length N_total describing each spectrum
    """
    # Merge user overrides onto defaults and validate configuration structure.
    merged_config = get_default_measurement_config()
    if config is not None:
        merged_config.update(config)
    _validate_config(merged_config)

    materials = merged_config["materials"]
    recipes = merged_config["recipes"]
    material_recipes = merged_config["material_recipes"]
    n_die = int(merged_config["n_die"])
    n_points = int(merged_config["n_points"])
    meas_per_t7 = int(merged_config.get("measurements_per_t7_code", 10))
    tool_offsets = merged_config.get("tool_offsets", {})

    rng = np.random.default_rng(merged_config.get("seed"))
    base_time = _parse_start_time(
        merged_config.get("start_time", "2026-01-01 00:00:00")
    )
    die_base_factors, die_rep_std = _resolve_die_scales_config(
        merged_config, config
    )

    # Precompute recipe energy grids and material-specific spectral resources.
    recipe_grids = _build_recipe_energy_grids(recipes, n_points)
    active_regions, mat_profiles, mat_strengths, mat_norm_factors = (
        _build_material_resources(materials)
    )
    synth_ctx = {
        "mat_profiles": mat_profiles,
        "mat_strengths": mat_strengths,
        "mat_norm_factors": mat_norm_factors,
        "noise_rel_std": float(merged_config["noise_relative_std"]),
        "bg_rel_strength": float(merged_config["background_relative_strength"]),
        "shirley_cfg": _as_dict(merged_config.get("shirley_background")),
    }

    n_total = _count_total_spectra(material_recipes, recipes, n_die)
    ary_intensity = np.empty((n_total, n_points), dtype=np.float64)
    ary_energy = np.empty((n_total, n_points), dtype=np.float64)
    meta_records: list[dict[str, Any]] = []
    tool_maint_caches: dict[str, dict[int, float]] = {}

    # Synthesize spectra per (material, recipe, tool) measurement group.
    spectrum_idx = 0
    pair_idx = 0
    for mat_name, rcp_map in material_recipes.items():
        mat_active = active_regions[mat_name]
        for rcp_name, tool_counts in rcp_map.items():
            active_tools = [t for t, c in tool_counts.items() if c > 0]
            group_start = base_time + timedelta(minutes=15 * pair_idx)
            timelines = generate_tool_timeline(
                tools=active_tools,
                timeline_config={
                    "measurements_per_tool": tool_counts,
                    "start_time": group_start,
                    "interval_hours_range": merged_config["interval_hours_range"],
                },
                rng=rng,
            )
            pair_idx += 1
            rcp_regions = list(recipes[rcp_name].keys())

            for tool in active_tools:
                t_off = _as_dict(tool_offsets.get(tool, {}))
                synth_ctx["tool_shift"] = float(t_off.get("shift_ev", 0.0))
                tool_scale = float(t_off.get("scale", 1.0))
                maint_cache = tool_maint_caches.setdefault(tool, {})

                for meas_num, meas_time in enumerate(timelines[tool]):
                    meas_id = f"{mat_name}-{rcp_name}-M-{tool}-{meas_num:05d}"
                    t7_code = f"T7_{(meas_num // meas_per_t7) + 1:03d}"
                    time_scale = calculate_time_decay_factor(
                        meas_time=meas_time,
                        base_time=base_time,
                        config=merged_config,
                        rng=rng,
                        cycle_cache=maint_cache,
                    )

                    for die_idx in range(n_die):
                        die_scale = _sample_die_scale(
                            die_idx, die_base_factors, die_rep_std, rng
                        )
                        total_scale = float(tool_scale * time_scale * die_scale)
                        synth_ctx["total_scale"] = total_scale

                        for reg_name in rcp_regions:
                            e_grid = recipe_grids[(rcp_name, reg_name)]
                            spec, rel_strength = _synthesize_single_region_row(
                                mat_name,
                                reg_name,
                                e_grid,
                                reg_name in mat_active,
                                synth_ctx,
                                rng,
                            )
                            ary_intensity[spectrum_idx] = spec
                            ary_energy[spectrum_idx] = e_grid
                            meta_records.append(
                                {
                                    "spectrum_index": spectrum_idx,
                                    "material": mat_name,
                                    "recipe": rcp_name,
                                    "tool": tool,
                                    "measurement_id": meas_id,
                                    "t7_code": t7_code,
                                    "time": pd.Timestamp(meas_time),
                                    "die": die_idx,
                                    "region": reg_name,
                                    "n_points": n_points,
                                    "relative_strength": rel_strength,
                                    "scale_tool": tool_scale,
                                    "scale_time": time_scale,
                                    "scale_die": die_scale,
                                    "scale_total": total_scale,
                                }
                            )
                            spectrum_idx += 1

    return ary_intensity, ary_energy, pd.DataFrame(meta_records)


