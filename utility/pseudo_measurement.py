"""Pseudo-measurement generation module for GGRD-SP.

Generates synthetic XPS regional spectra and associated metadata according to
project specifications in GEMINI.md.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


# Default regional energy boundaries and characteristic peak parameters for TiAlC (NMG)
DEFAULT_REGION_PROFILES: dict[str, dict[str, Any]] = {
    "Al2p": {
        "energy_range": (70.0, 78.0),
        "baseline": 15.0,
        "peaks": [
            # Al-C / Al-Ti metallic-covalent bond
            {"center": 72.8, "amplitude": 120.0, "fwhm": 1.3, "eta": 0.3},
            # Surface Al oxide
            {"center": 74.8, "amplitude": 85.0, "fwhm": 1.4, "eta": 0.3},
        ],
    },
    "Ti2p": {
        "energy_range": (452.0, 467.0),
        "baseline": 25.0,
        "peaks": [
            # Ti-C 2p3/2
            {"center": 454.8, "amplitude": 260.0, "fwhm": 1.2, "eta": 0.35},
            # Ti-C 2p1/2 (spin-orbit split ~5.7 eV, 2:1 branching ratio)
            {"center": 460.5, "amplitude": 130.0, "fwhm": 1.8, "eta": 0.35},
            # Surface TiO2 2p3/2
            {"center": 458.5, "amplitude": 70.0, "fwhm": 1.4, "eta": 0.35},
            # Surface TiO2 2p1/2
            {"center": 464.2, "amplitude": 35.0, "fwhm": 2.0, "eta": 0.35},
        ],
    },
    "O1s": {
        "energy_range": (526.0, 536.0),
        "baseline": 20.0,
        "peaks": [
            # Metal-oxygen lattice (Ti-O / Al-O)
            {"center": 530.2, "amplitude": 180.0, "fwhm": 1.5, "eta": 0.3},
            # Surface hydroxyl / adsorbed C-O species
            {"center": 532.0, "amplitude": 95.0, "fwhm": 1.6, "eta": 0.3},
        ],
    },
    "C1s": {
        "energy_range": (280.0, 292.0),
        "baseline": 18.0,
        "peaks": [
            # Ti-C carbide
            {"center": 281.8, "amplitude": 150.0, "fwhm": 1.2, "eta": 0.3},
            # Adventitious carbon C-C
            {"center": 284.8, "amplitude": 220.0, "fwhm": 1.4, "eta": 0.3},
            # C-O / C-O-C
            {"center": 286.5, "amplitude": 45.0, "fwhm": 1.5, "eta": 0.3},
        ],
    },
    "Cl2p": {
        "energy_range": (196.0, 206.0),
        "baseline": 12.0,
        "peaks": [
            # Cl 2p3/2 residual
            {"center": 198.5, "amplitude": 60.0, "fwhm": 1.3, "eta": 0.3},
            # Cl 2p1/2 residual (spin-orbit split ~1.6 eV, 2:1 ratio)
            {"center": 200.1, "amplitude": 30.0, "fwhm": 1.3, "eta": 0.3},
        ],
    },
}


def pseudo_voigt(
    energy: np.ndarray,
    peak_params: dict[str, Any],
) -> np.ndarray:
    """Calculate a pseudo-Voigt profile as linear combination of Gaussian and Lorentzian.

    Parameters
    ----------
    energy : np.ndarray
        1D array of binding energy points.
    peak_params : dict[str, Any]
        Dictionary packaging peak configuration:
        - 'center' (float): Peak center position in eV.
        - 'amplitude' (float): Peak height/amplitude.
        - 'fwhm' (float): Full width at half maximum in eV (must be > 0).
        - 'eta' (float, optional): Lorentzian fraction (0.0 to 1.0, default 0.3).

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


def get_default_measurement_config() -> dict[str, Any]:
    """Return the default configuration dictionary for pseudo-measurement generation.

    Returns
    -------
    dict[str, Any]
        Configuration options bundling all generation parameters.
    """
    return {
        "material": "NMG",
        "regions": ["Al2p", "Ti2p", "O1s", "C1s", "Cl2p"],
        "n_die": 9,
        "n_points": 100,
        "measurements_per_tool": {"J4": 10, "J5": 10},
        "start_time": "2026-01-01 00:00:00",
        "interval_hours_range": (4.0, 24.0),
        "tool_offsets": {
            "J4": {"shift_ev": 0.0, "scale": 1.0},
            "J5": {"shift_ev": 0.2, "scale": 1.1},
        },
        "die_variation_std": 0.03,
        "noise_relative_std": 0.015,
        "region_profiles": DEFAULT_REGION_PROFILES,
        "seed": 42,
    }


def _validate_config(config: dict[str, Any]) -> None:
    """Validate configuration dictionary parameters.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary to validate.

    Raises
    ------
    ValueError
        If required parameters are missing, invalid, or out of range.
    """
    n_points = config.get("n_points")
    if not isinstance(n_points, int) or n_points < 2:
        raise ValueError(f"n_points must be an integer >= 2, got {n_points}")

    n_die = config.get("n_die")
    if not isinstance(n_die, int) or n_die <= 0:
        raise ValueError(f"n_die must be a positive integer, got {n_die}")

    if not config.get("regions"):
        raise ValueError("regions list cannot be empty")

    measurements_per_tool = config.get("measurements_per_tool", {})
    if not measurements_per_tool or all(v <= 0 for v in measurements_per_tool.values()):
        raise ValueError("measurements_per_tool must specify at least one tool with count > 0")

    if any(v < 0 for v in measurements_per_tool.values()):
        raise ValueError(f"measurements_per_tool counts cannot be negative: {measurements_per_tool}")

    interval_range = config.get("interval_hours_range", (4.0, 24.0))
    if not (isinstance(interval_range, (tuple, list)) and len(interval_range) == 2):
        raise ValueError(
            f"interval_hours_range must be a tuple/list of 2 floats, got {interval_range}"
        )
    if interval_range[0] < 0 or interval_range[1] <= interval_range[0]:
        raise ValueError(
            "interval_hours_range must satisfy min >= 0 and max > min, "
            f"got {interval_range}"
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
    measurements_per_tool: dict[str, int] = timeline_config.get("measurements_per_tool", {})
    start_time: str | datetime = timeline_config.get("start_time", "2026-01-01 00:00:00")
    interval_range: tuple[float, float] = timeline_config.get("interval_hours_range", (4.0, 24.0))

    if isinstance(start_time, str):
        base_dt = datetime.fromisoformat(start_time)
    else:
        base_dt = start_time

    timeline: dict[str, list[datetime]] = {}
    for tool in tools:
        n_meas = measurements_per_tool.get(tool, 0)
        current_dt = base_dt + timedelta(hours=float(rng.uniform(0.0, 4.0)))
        tool_times: list[datetime] = []
        for _ in range(n_meas):
            tool_times.append(current_dt)
            delta_hours = float(rng.uniform(interval_range[0], interval_range[1]))
            current_dt += timedelta(hours=delta_hours)
        timeline[tool] = tool_times

    return timeline


def generate_pseudo_measurements(
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate synthetic XPS regional spectra and corresponding metadata.

    Fulfills Part 1 of the program structure defined in GEMINI.md.
    Binding energy arrays for regions with the same name are identical and sorted
    in ascending order. Tool discrepancies (energy shifts and intensity scales) and
    die-to-die variations are captured in the intensity arrays.

    Parameters
    ----------
    config : dict[str, Any] | None, optional
        Configuration dictionary. If None, default settings from
        `get_default_measurement_config()` are used. Custom keys override defaults.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame]
        - ary_intensity: 2D array of shape (N_total, N_points)
        - ary_energy: 2D array of shape (N_total, N_points)
        - meta_df: DataFrame of length N_total describing each spectrum
    """
    merged_config = get_default_measurement_config()
    if config is not None:
        merged_config.update(config)

    _validate_config(merged_config)

    material: str = merged_config["material"]
    regions: list[str] = merged_config["regions"]
    n_die: int = merged_config["n_die"]
    n_points: int = merged_config["n_points"]
    meas_per_tool: dict[str, int] = merged_config["measurements_per_tool"]
    tool_offsets: dict[str, dict[str, float]] = merged_config["tool_offsets"]
    die_variation_std: float = merged_config["die_variation_std"]
    noise_relative_std: float = merged_config["noise_relative_std"]
    region_profiles: dict[str, dict[str, Any]] = merged_config["region_profiles"]

    seed: int | None = merged_config.get("seed")
    rng = np.random.default_rng(seed)

    # Precompute fixed 1D binding energy arrays for each region (ascending order)
    region_energy_grids: dict[str, np.ndarray] = {}
    for region_name in regions:
        profile = region_profiles.get(region_name)
        if profile is None:
            raise ValueError(f"No profile definition found for region: '{region_name}'")
        e_min, e_max = profile["energy_range"]
        if e_min >= e_max:
            raise ValueError(
                f"energy_range for region '{region_name}' must have min < max, "
                f"got ({e_min}, {e_max})"
            )
        region_energy_grids[region_name] = np.linspace(e_min, e_max, n_points, dtype=np.float64)

    # Generate timeline per tool
    tools = list(meas_per_tool.keys())
    timelines = generate_tool_timeline(
        tools=tools,
        timeline_config={
            "measurements_per_tool": meas_per_tool,
            "start_time": merged_config["start_time"],
            "interval_hours_range": merged_config["interval_hours_range"],
        },
        rng=rng,
    )

    # Calculate total spectra count
    total_measurements = sum(meas_per_tool.values())
    n_total = total_measurements * n_die * len(regions)

    ary_intensity = np.empty((n_total, n_points), dtype=np.float64)
    ary_energy = np.empty((n_total, n_points), dtype=np.float64)
    meta_records: list[dict[str, Any]] = []

    spectrum_idx = 0

    for tool in tools:
        t_offset = tool_offsets.get(tool, {"shift_ev": 0.0, "scale": 1.0})
        tool_shift = t_offset.get("shift_ev", 0.0)
        tool_scale = t_offset.get("scale", 1.0)
        meas_times = timelines.get(tool, [])

        for meas_num, meas_time in enumerate(meas_times):
            meas_id = f"M_{tool}_{meas_num:03d}"

            # Die-to-die spatial variation across the wafer
            die_factors = rng.normal(1.0, die_variation_std, size=n_die)
            die_factors = np.clip(die_factors, 0.5, 1.5)

            for die_idx in range(n_die):
                die_scale = float(die_factors[die_idx])

                for region_name in regions:
                    e_grid = region_energy_grids[region_name]
                    profile = region_profiles[region_name]

                    # Base profile shifted by tool shift (observed peak position = center + shift)
                    intensity = np.zeros_like(e_grid)
                    for peak in profile["peaks"]:
                        peak_shifted = {
                            "center": peak["center"] + tool_shift,
                            "amplitude": peak["amplitude"] * tool_scale * die_scale,
                            "fwhm": peak["fwhm"],
                            "eta": peak.get("eta", 0.3),
                        }
                        intensity += pseudo_voigt(energy=e_grid, peak_params=peak_shifted)

                    # Add baseline with die factor
                    baseline = profile.get("baseline", 10.0) * tool_scale * die_scale
                    intensity += baseline

                    # Add measurement noise (Poisson-like scaling + Gaussian detector floor)
                    peak_max = float(np.max(intensity))
                    noise_sigma = peak_max * noise_relative_std
                    noise = rng.normal(0.0, noise_sigma, size=n_points)
                    intensity += noise

                    # XPS intensity cannot be negative
                    intensity = np.clip(intensity, 0.0, None)

                    # Assign into output arrays
                    ary_intensity[spectrum_idx] = intensity
                    # Fixed binding energy grid for this region
                    ary_energy[spectrum_idx] = e_grid

                    meta_records.append(
                        {
                            "spectrum_index": spectrum_idx,
                            "material": material,
                            "tool": tool,
                            "measurement_id": meas_id,
                            "time": meas_time.isoformat(),
                            "die": die_idx,
                            "region": region_name,
                            "n_points": n_points,
                        }
                    )

                    spectrum_idx += 1

    meta_df = pd.DataFrame(meta_records)

    return ary_intensity, ary_energy, meta_df
