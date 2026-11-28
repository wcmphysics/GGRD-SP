"""Built-in material-specific XPS spectral physics profiles for GGRD-SP.

Defines chemical-state peak parameters and default regional relative strengths
for synthetic thin-film materials deposited on silicon substrates and exposed
to air (native/interfacial SiO2, surface oxidation, and adventitious carbon).
"""

import copy
from typing import Any

import numpy as np


# Built-in (material -> region -> peak/baseline profile) physics library.
BUILTIN_MATERIAL_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "TiAlC": {
        "Al2p": {
            "baseline": 15.0,
            "peaks": [
                # Al-Ti / Al-C metallic-carbidic bond
                {"center": 72.6, "amplitude": 125.0, "fwhm": 1.3, "eta": 0.3},
                # Air-exposed surface Al2O3 native oxide
                {"center": 74.7, "amplitude": 85.0, "fwhm": 1.4, "eta": 0.3},
            ],
        },
        "Ti2p": {
            "baseline": 25.0,
            "peaks": [
                # Ti-Al metallic/intermetallic 2p3/2 and 2p1/2 (2:1 ratio)
                {"center": 453.8, "amplitude": 150.0, "fwhm": 1.1, "eta": 0.35},
                {"center": 459.9, "amplitude": 75.0, "fwhm": 1.6, "eta": 0.35},
                # Ti-C carbide 2p3/2 and 2p1/2 (~5.7 eV split, 2:1 ratio)
                {"center": 454.8, "amplitude": 220.0, "fwhm": 1.2, "eta": 0.35},
                {"center": 460.5, "amplitude": 110.0, "fwhm": 1.8, "eta": 0.35},
                # Air-exposed surface TiO2 2p3/2 and 2p1/2
                {"center": 458.5, "amplitude": 70.0, "fwhm": 1.4, "eta": 0.35},
                {"center": 464.2, "amplitude": 35.0, "fwhm": 2.0, "eta": 0.35},
            ],
        },
        "C1s": {
            "baseline": 18.0,
            "peaks": [
                # Ti-C carbide
                {"center": 281.8, "amplitude": 150.0, "fwhm": 1.2, "eta": 0.3},
                # Adventitious carbon C-C / C-H
                {"center": 284.8, "amplitude": 220.0, "fwhm": 1.4, "eta": 0.3},
                # Surface C-O / C-O-C
                {"center": 286.5, "amplitude": 45.0, "fwhm": 1.5, "eta": 0.3},
            ],
        },
        "O1s": {
            "baseline": 20.0,
            "peaks": [
                # Ti-O surface oxide
                {"center": 530.3, "amplitude": 140.0, "fwhm": 1.4, "eta": 0.3},
                # Al-O native oxide
                {"center": 531.5, "amplitude": 110.0, "fwhm": 1.5, "eta": 0.3},
                # Surface hydroxyl -OH / C-O / interfacial Si-O
                {"center": 532.4, "amplitude": 80.0, "fwhm": 1.6, "eta": 0.3},
            ],
        },
        "Si2p": {
            "baseline": 14.0,
            "peaks": [
                # Si substrate bulk Si0 2p3/2 and 2p1/2 (~0.6 eV split, 2:1)
                {"center": 99.3, "amplitude": 110.0, "fwhm": 0.9, "eta": 0.25},
                {"center": 99.9, "amplitude": 55.0, "fwhm": 0.9, "eta": 0.25},
                # Interfacial SiO2
                {"center": 103.3, "amplitude": 45.0, "fwhm": 1.5, "eta": 0.3},
            ],
        },
        "Cl2p": {
            "baseline": 12.0,
            "peaks": [
                # Residual precursor Cl 2p3/2 and 2p1/2 (~1.6 eV split, 2:1)
                {"center": 198.5, "amplitude": 60.0, "fwhm": 1.3, "eta": 0.3},
                {"center": 200.1, "amplitude": 30.0, "fwhm": 1.3, "eta": 0.3},
            ],
        },
    },
    "TiN": {
        "Ti2p": {
            "baseline": 24.0,
            "peaks": [
                # Ti-N nitride 2p3/2 and 2p1/2 (no Ti-Al or Ti-C peak)
                {"center": 455.2, "amplitude": 260.0, "fwhm": 1.2, "eta": 0.35},
                {"center": 461.1, "amplitude": 130.0, "fwhm": 1.7, "eta": 0.35},
                # Ti-O-N oxynitride / shake-up satellite 2p3/2 and 2p1/2
                {"center": 456.9, "amplitude": 110.0, "fwhm": 1.5, "eta": 0.4},
                {"center": 462.6, "amplitude": 55.0, "fwhm": 1.9, "eta": 0.4},
                # Air-exposed surface TiO2 2p3/2 and 2p1/2
                {"center": 458.5, "amplitude": 95.0, "fwhm": 1.4, "eta": 0.35},
                {"center": 464.2, "amplitude": 47.5, "fwhm": 2.0, "eta": 0.35},
            ],
        },
        "N1s": {
            "baseline": 18.0,
            "peaks": [
                # Oxynitride N-Ti-O shoulder
                {"center": 396.2, "amplitude": 75.0, "fwhm": 1.2, "eta": 0.3},
                # Stoichiometric Ti-N nitride
                {"center": 397.2, "amplitude": 240.0, "fwhm": 1.1, "eta": 0.3},
                # Adsorbed / oxidized surface nitrogen
                {"center": 399.5, "amplitude": 35.0, "fwhm": 1.5, "eta": 0.3},
            ],
        },
        "O1s": {
            "baseline": 20.0,
            "peaks": [
                # Native TiO2 lattice oxide
                {"center": 530.1, "amplitude": 175.0, "fwhm": 1.3, "eta": 0.3},
                # Ti-O-N oxynitride
                {"center": 531.4, "amplitude": 95.0, "fwhm": 1.5, "eta": 0.3},
                # Adsorbed -OH / H2O
                {"center": 532.6, "amplitude": 70.0, "fwhm": 1.6, "eta": 0.3},
            ],
        },
        "C1s": {
            "baseline": 16.0,
            "peaks": [
                # Adventitious C-C / C-H only (no Ti-C carbide peak)
                {"center": 284.8, "amplitude": 200.0, "fwhm": 1.4, "eta": 0.3},
                # C-O
                {"center": 286.4, "amplitude": 50.0, "fwhm": 1.5, "eta": 0.3},
                # O-C=O
                {"center": 288.5, "amplitude": 25.0, "fwhm": 1.5, "eta": 0.3},
            ],
        },
        "Si2p": {
            "baseline": 14.0,
            "peaks": [
                # Si substrate bulk Si0 2p3/2 and 2p1/2
                {"center": 99.3, "amplitude": 100.0, "fwhm": 0.9, "eta": 0.25},
                {"center": 99.9, "amplitude": 50.0, "fwhm": 0.9, "eta": 0.25},
                # Interfacial SiOxNy / SiO2
                {"center": 102.8, "amplitude": 50.0, "fwhm": 1.5, "eta": 0.3},
            ],
        },
        "Cl2p": {
            "baseline": 12.0,
            "peaks": [
                # Residual TiCl4 precursor Cl 2p3/2 and 2p1/2
                {"center": 198.4, "amplitude": 55.0, "fwhm": 1.3, "eta": 0.3},
                {"center": 200.0, "amplitude": 27.5, "fwhm": 1.3, "eta": 0.3},
            ],
        },
    },
    "HfO2": {
        "Hf4f": {
            "baseline": 22.0,
            "peaks": [
                # Bulk HfO2 4f7/2 and 4f5/2 (~1.7 eV split, 4:3 ratio)
                {"center": 17.0, "amplitude": 280.0, "fwhm": 1.1, "eta": 0.3},
                {"center": 18.7, "amplitude": 210.0, "fwhm": 1.1, "eta": 0.3},
                # Interfacial Hf-O-Si silicate shoulder (4:3 ratio)
                {"center": 17.8, "amplitude": 64.0, "fwhm": 1.3, "eta": 0.3},
                {"center": 19.5, "amplitude": 48.0, "fwhm": 1.3, "eta": 0.3},
            ],
        },
        "O1s": {
            "baseline": 22.0,
            "peaks": [
                # Dominant Hf-O lattice oxide
                {"center": 530.2, "amplitude": 260.0, "fwhm": 1.3, "eta": 0.3},
                # Interfacial Hf-O-Si / SiO2
                {"center": 531.9, "amplitude": 85.0, "fwhm": 1.5, "eta": 0.3},
                # Air-exposed surface -OH
                {"center": 532.8, "amplitude": 55.0, "fwhm": 1.5, "eta": 0.3},
            ],
        },
        "Si2p": {
            "baseline": 15.0,
            "peaks": [
                # Si substrate bulk Si0 2p3/2 and 2p1/2
                {"center": 99.3, "amplitude": 130.0, "fwhm": 0.9, "eta": 0.25},
                {"center": 99.9, "amplitude": 65.0, "fwhm": 0.9, "eta": 0.25},
                # Interfacial HfSiOx / SiO2
                {"center": 102.9, "amplitude": 75.0, "fwhm": 1.5, "eta": 0.3},
            ],
        },
        "C1s": {
            "baseline": 16.0,
            "peaks": [
                # Adventitious C-C / C-H only
                {"center": 284.8, "amplitude": 190.0, "fwhm": 1.4, "eta": 0.3},
                # Surface C-O
                {"center": 286.5, "amplitude": 45.0, "fwhm": 1.5, "eta": 0.3},
            ],
        },
        "Cl2p": {
            "baseline": 12.0,
            "peaks": [
                # Residual HfCl4 precursor Cl 2p3/2 and 2p1/2
                {"center": 198.6, "amplitude": 50.0, "fwhm": 1.3, "eta": 0.3},
                {"center": 200.2, "amplitude": 25.0, "fwhm": 1.3, "eta": 0.3},
            ],
        },
    },
}


# Default per-material regional relative strengths.
BUILTIN_MATERIAL_STRENGTHS: dict[str, dict[str, float]] = {
    "TiAlC": {
        "Al2p": 5.0,
        "Ti2p": 7.5,
        "C1s": 6.5,
        "O1s": 6.0,
        "Si2p": 2.0,
        "Cl2p": 1.0,
    },
    "TiN": {
        "Ti2p": 8.5,
        "N1s": 6.5,
        "O1s": 5.5,
        "C1s": 3.5,
        "Si2p": 2.0,
        "Cl2p": 0.8,
    },
    "HfO2": {
        "Hf4f": 9.0,
        "O1s": 10.0,
        "Si2p": 3.0,
        "C1s": 3.0,
        "Cl2p": 0.7,
    },
}


def resolve_material_region_profile(
    material: str,
    region: str,
    custom_profiles: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve peak and baseline profile for a (material, region) pair.

    Parameters
    ----------
    material : str
        Material name (e.g., 'TiAlC', 'TiN', 'HfO2').
    region : str
        Spectral region name (e.g., 'Ti2p', 'Hf4f', 'O1s').
    custom_profiles : dict[str, Any] | None, optional
        Optional per-region profile overrides for this material.

    Returns
    -------
    dict[str, Any]
        Deep-copied profile dictionary containing 'peaks' and 'baseline'.

    Raises
    ------
    ValueError
        If neither a custom override nor a built-in profile exists.
    """
    base_profile = (
        BUILTIN_MATERIAL_PROFILES.get(material, {}).get(region)
    )
    override = (
        custom_profiles.get(region)
        if isinstance(custom_profiles, dict)
        else None
    )

    if base_profile is None and override is None:
        supported_regions = sorted(
            BUILTIN_MATERIAL_PROFILES.get(material, {}).keys()
        )
        raise ValueError(
            f"No spectral profile defined for material='{material}', "
            f"region='{region}'. Built-in regions for '{material}': "
            f"{supported_regions}; built-in materials: "
            f"{sorted(BUILTIN_MATERIAL_PROFILES.keys())}. Provide a custom "
            "entry under materials.<material>.region_profiles.<region>."
        )

    merged = copy.deepcopy(base_profile) if base_profile is not None else {}
    if isinstance(override, dict):
        merged.update(copy.deepcopy(override))

    if "peaks" not in merged or not isinstance(merged["peaks"], list):
        raise ValueError(
            f"Resolved profile for material='{material}', region='{region}' "
            "must define a 'peaks' list."
        )
    return merged


def resolve_material_region_strength(
    material: str,
    region: str,
    custom_strengths: dict[str, float] | None = None,
) -> float:
    """Resolve positive relative strength for a (material, region) pair.

    Parameters
    ----------
    material : str
        Material name.
    region : str
        Spectral region name.
    custom_strengths : dict[str, float] | None, optional
        Optional per-region strength overrides for this material.

    Returns
    -------
    float
        Positive finite relative strength scalar.

    Raises
    ------
    ValueError
        If no strength is defined or the value is non-positive/non-finite.
    """
    if isinstance(custom_strengths, dict) and region in custom_strengths:
        raw_val: Any = custom_strengths[region]
    else:
        raw_val = BUILTIN_MATERIAL_STRENGTHS.get(material, {}).get(region)

    if raw_val is None:
        raise ValueError(
            f"No relative strength defined for material='{material}', "
            f"region='{region}'. Specify it under "
            f"materials.{material}.relative_strengths.{region}."
        )
    if (
        isinstance(raw_val, bool)
        or not isinstance(raw_val, (int, float))
        or not np.isfinite(raw_val)
        or float(raw_val) <= 0.0
    ):
        raise ValueError(
            f"Relative strength for material='{material}', region='{region}' "
            f"must be a positive finite number, got {raw_val!r}."
        )
    return float(raw_val)


def compute_intrinsic_profile_max(profile: dict[str, Any]) -> float:
    """Compute the intrinsic peak + baseline maximum of a region profile.

    Evaluates the unshifted pseudo-Voigt peak sum over a dense energy span
    covering all peaks so that changing a recipe's [e_min, e_max] scan window
    does not artificially rescale physical peak amplitudes.
    """
    peaks = profile.get("peaks", [])
    baseline = float(profile.get("baseline", 10.0))
    if not peaks:
        return baseline if baseline > 0.0 else 1.0

    centers = [float(p["center"]) for p in peaks]
    max_fwhm = max(float(p["fwhm"]) for p in peaks)
    e_min = min(centers) - 4.0 * max_fwhm
    e_max = max(centers) + 4.0 * max_fwhm
    e_dense = np.unique(
        np.concatenate(
            [np.linspace(e_min, e_max, 500, dtype=np.float64), np.array(centers)]
        )
    )

    intensity = np.full_like(e_dense, baseline)
    for peak in peaks:
        fwhm = float(peak["fwhm"])
        center = float(peak["center"])
        amplitude = float(peak["amplitude"])
        eta = float(peak.get("eta", 0.3))
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        gamma = fwhm / 2.0
        gaussian = np.exp(-0.5 * ((e_dense - center) / sigma) ** 2)
        lorentzian = 1.0 / (1.0 + ((e_dense - center) / gamma) ** 2)
        intensity += amplitude * ((1.0 - eta) * gaussian + eta * lorentzian)

    peak_max = float(np.max(intensity))
    return peak_max if peak_max > 0.0 else 1.0


__all__ = [
    "BUILTIN_MATERIAL_PROFILES",
    "BUILTIN_MATERIAL_STRENGTHS",
    "compute_intrinsic_profile_max",
    "resolve_material_region_profile",
    "resolve_material_region_strength",
]
