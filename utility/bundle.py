"""Spectral data bundle container and validation."""

from __future__ import annotations

from typing import NamedTuple
import numpy as np
import pandas as pd


class SpectralDataBundle(NamedTuple):
    """Container for matched spectral intensity, energy grid, and metadata.

    Fields
    ------
    ary_intensity : np.ndarray
        2D intensity array of shape (N_rows, N_points).
    ary_energy : np.ndarray
        2D energy array of shape (N_rows, N_points) in ascending eV.
    df_meta : pd.DataFrame
        Metadata dataframe containing at least 'measurement_id', 'die',
        'region', and 'spectrum_index'.
    """

    ary_intensity: np.ndarray
    ary_energy: np.ndarray
    df_meta: pd.DataFrame

    def validate(self) -> SpectralDataBundle:
        """Validate tensor shapes and index alignment."""
        if self.ary_intensity.ndim != 2:
            raise ValueError(
                f"ary_intensity must be 2D, got shape {self.ary_intensity.shape}"
            )
        if self.ary_energy.ndim != 2:
            raise ValueError(
                f"ary_energy must be 2D, got shape {self.ary_energy.shape}"
            )
        if self.ary_intensity.shape != self.ary_energy.shape:
            raise ValueError(
                f"Shape mismatch: ary_intensity {self.ary_intensity.shape} "
                f"vs ary_energy {self.ary_energy.shape}"
            )
        if len(self.df_meta) > self.ary_intensity.shape[0]:
            raise ValueError(
                f"Metadata rows {len(self.df_meta)} exceed intensity rows "
                f"{self.ary_intensity.shape[0]}"
            )
        return self
