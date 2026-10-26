"""Unit tests for denoise package and noise visualization routines."""

from __future__ import annotations

import os
import tempfile
import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend for testing

from denoise import (
    compare_denoised_atomic_percentages,
    create_mean_and_confidence_curves,
    denoise_regional_spectra,
    denoise_spectrum_fourier,
    denoise_spectrum_savgol,
    denoise_spectrum_wavelet,
    fit_regional_spectrum_by_gp,
    perform_spectral_noise_analysis,
    trace_gp_hyperparameter_convergence,
)
from visualization.plot_noise_analysis import (
    plot_denoise_atomic_percentage_mae,
    plot_denoise_spectrum_comparison,
    plot_gp_hyperparameter_convergence,
    plot_spectral_noise_analysis,
)

_FAST = {"training": {"num_iterations": 40, "learning_rate": 0.1}, "show_progress": False}


def _make_data(
    noise_levels: dict[str, float], n_points: int = 100, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build canonical 2-region x 2-die test dataset."""
    rng = np.random.default_rng(seed)
    rows, intensity, energy = [], [], []
    for m_idx, (split, tool) in enumerate([("train", "J4"), ("test", "J5"), (None, "J4")]):
        for die in range(2):
            for region, center in (("Ti2p", 460.0), ("O1s", 530.0)):
                grid = np.linspace(center - 10, center + 10, n_points)
                clean = 1.0 + 3.0 * np.exp(-((grid - center) ** 2) / (2 * 1.2**2))
                intensity.append(clean + rng.normal(0, noise_levels[region], n_points))
                energy.append(grid)
                rows.append({
                    "spectrum_index": len(rows),
                    "measurement_id": f"M{m_idx}",
                    "tool": tool,
                    "time": pd.Timestamp("2026-01-01") + pd.Timedelta(days=m_idx),
                    "die": die,
                    "region": region,
                    "split": split,
                })
    return np.array(intensity), np.array(energy), pd.DataFrame(rows)


class TestNoiseAnalysisAndDenoising(unittest.TestCase):
    """Test suite for GP noise modeling, deterministic filtering, and pipeline orchestration."""

    def test_gp_noise_estimation_and_confidence(self) -> None:
        """Verify GP fit estimates noise level, confidence bounds, and respects latent vs observation mode."""
        ary_i, ary_e, _ = _make_data({"Ti2p": 0.05, "O1s": 0.05})
        for noise_model in ("homoscedastic", "poisson_scaled"):
            with self.subTest(noise_model=noise_model):
                cfg = {**_FAST, "noise_model": noise_model}
                model = fit_regional_spectrum_by_gp(ary_e[0], ary_i[0], cfg)
                mean, half = create_mean_and_confidence_curves(model, ary_e[0], cfg)
                self.assertEqual(mean.shape, ary_e[0].shape)
                self.assertTrue((half > 0).all())

        # Latent vs observation confidence intervals
        model = fit_regional_spectrum_by_gp(ary_e[0], ary_i[0], _FAST)
        _, obs = create_mean_and_confidence_curves(model, ary_e[0], {"confidence_type": "observation"})
        _, lat = create_mean_and_confidence_curves(model, ary_e[0], {"confidence_type": "latent"})
        self.assertTrue((obs >= lat).all())
        self.assertGreater(obs.mean(), lat.mean())

    def test_deterministic_denoising_methods(self) -> None:
        """Table-driven test for Savitzky-Golay, Wavelet, and Fourier denoising routines."""
        rng = np.random.default_rng(42)
        energy = np.linspace(450.0, 470.0, 100)
        clean = 100.0 + 500.0 * np.exp(-((energy - 460.0) ** 2) / (2 * 1.5**2))
        noisy = clean + rng.normal(0.0, 20.0, energy.size)
        raw_rmse = float(np.sqrt(np.mean((noisy - clean) ** 2)))

        # SavGol
        sig_sg, conf_sg, (sigma_sg, snr_sg, _) = denoise_spectrum_savgol(
            energy, noisy, {"window_length": 11, "polynomial_order": 3}
        )
        self.assertLess(float(np.sqrt(np.mean((sig_sg - clean) ** 2))), raw_rmse)
        self.assertGreater(snr_sg, 5.0)

        # Wavelet (discrete & stationary)
        for tt in ("stationary", "discrete"):
            with self.subTest(wavelet=tt):
                sig_wv, _, (_, snr_wv, _) = denoise_spectrum_wavelet(
                    energy, noisy, {"transform_type": tt, "wavelet_name": "sym4", "decomposition_level": 2}
                )
                self.assertLess(float(np.sqrt(np.mean((sig_wv - clean) ** 2))), raw_rmse)
                self.assertGreater(snr_wv, 5.0)

        # Fourier (DCT & Butterworth)
        for ft in ("dct", "butterworth"):
            with self.subTest(fourier=ft):
                sig_ft, _, (_, snr_ft, _) = denoise_spectrum_fourier(
                    energy, noisy, {"filter_type": ft, "cutoff_ratio": 0.25}
                )
                self.assertLess(float(np.sqrt(np.mean((sig_ft - clean) ** 2))), raw_rmse)
                self.assertGreater(snr_ft, 5.0)

    def test_orchestration_and_selective_denoising(self) -> None:
        """Verify perform_spectral_noise_analysis and selective region denoise masking."""
        ary_i, ary_e, df_meta = _make_data({"Ti2p": 0.02, "O1s": 0.2})
        sig, noise, df_out = perform_spectral_noise_analysis(ary_i, ary_e, df_meta, {"method": "savgol", "show_progress": False})
        self.assertEqual(sig.shape, (8, 100))
        analysed = df_out[df_out["noise_analysis_index"].notna()]
        by_reg = analysed.groupby("region")[["gp_noise_sigma", "gp_snr"]].mean()
        self.assertGreater(by_reg.loc["O1s", "gp_noise_sigma"], by_reg.loc["Ti2p", "gp_noise_sigma"])

        # Selective regional denoise and apply_denoise toggle
        raw_backup = ary_i.copy()
        active_f, non_den, den = denoise_regional_spectra(
            ary_i, ary_e, df_meta, {"apply_denoise": False, "regions": ["Ti2p"], "method": "savgol"}
        )
        np.testing.assert_array_equal(active_f, raw_backup)
        np.testing.assert_array_equal(non_den, raw_backup)

        # Ti2p modified in den, O1s untouched
        ti_mask = (df_meta["region"] == "Ti2p").to_numpy()
        self.assertFalse(np.allclose(den[ti_mask], raw_backup[ti_mask]))
        self.assertTrue(np.allclose(den[~ti_mask], raw_backup[~ti_mask]))

    def test_defensive_error_handling(self) -> None:
        """Table-driven test for invalid noise and denoise configurations."""
        ary_i, ary_e, df_meta = _make_data({"Ti2p": 0.05, "O1s": 0.05})
        with self.assertRaises(ValueError):
            denoise_spectrum_savgol(ary_e[0], ary_i[0], {"window_length": 10})  # Even window
        with self.assertRaises(ValueError):
            denoise_spectrum_fourier(ary_e[0], ary_i[0], {"cutoff_ratio": 1.5})
        with self.assertRaises(ValueError):
            perform_spectral_noise_analysis(ary_i, ary_e, df_meta, {**_FAST, "regions": ["BadRegion"]})


class TestNoiseVisualization(unittest.TestCase):
    """Test suite for convergence tracking and noise diagnostic plots."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate test data for noise plots."""
        cls.ary_i, cls.ary_e, cls.df_meta = _make_data({"Ti2p": 0.05, "O1s": 0.1})
        cls.df_meta["material"] = "NMG"

    def tearDown(self) -> None:
        """Close open matplotlib figures."""
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_spectral_noise_analysis_and_convergence(self) -> None:
        """Verify diagnostic multi-panel noise plots and GP parameter convergence tracing."""
        h = trace_gp_hyperparameter_convergence(
            self.ary_e[0], self.ary_i[0], {"training": {"num_iterations": 10, "learning_rate": 0.1}}
        )
        self.assertIn("loss", h)
        self.assertEqual(len(h["loss"]), 10)

        with tempfile.TemporaryDirectory() as tmp:
            save_file = os.path.join(tmp, "convergence.png")
            fig = plot_gp_hyperparameter_convergence(
                (self.ary_i, self.ary_e, self.df_meta),
                {"measurement_id": "M0", "die": 0, "num_iterations": 10, "save_path": save_file},
            )
            self.assertIsInstance(fig, matplotlib.figure.Figure)
            self.assertTrue(os.path.exists(save_file))

    def test_plot_denoise_comparisons(self) -> None:
        """Verify raw vs denoised spectral comparison and atomic percentage MAE plots."""
        _, ary_raw, ary_den = denoise_regional_spectra(
            self.ary_i, self.ary_e, self.df_meta, {"regions": ["Ti2p"], "method": "savgol"}
        )
        fig_spec, axes = plot_denoise_spectrum_comparison(
            self.ary_e, (ary_raw, ary_den), self.df_meta, {"die": 0}
        )
        self.assertIsInstance(fig_spec, matplotlib.figure.Figure)
        lines = axes.ravel()[0].get_lines()
        self.assertEqual(len(lines), 2)

        df_samples, df_summary = compare_denoised_atomic_percentages(
            self.ary_e, (ary_raw, ary_den), self.df_meta
        )
        fig_box, ax_box = plot_denoise_atomic_percentage_mae(df_samples)
        self.assertIsInstance(fig_box, matplotlib.figure.Figure)
        self.assertIsNotNone(ax_box)


if __name__ == "__main__":
    unittest.main()
