"""Unit tests for denoise package and visualization.plot_noise_analysis."""

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

_FAST = {"training": {"num_iterations": 60, "learning_rate": 0.1}, "show_progress": False}


def _make_data(
    noise_levels: dict[str, float], n_points: int = 120, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build canonical arrays: 2 regions x 2 dies, 2 split measurements."""
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
                    "spectrum_index": len(rows), "measurement_id": f"M{m_idx}",
                    "tool": tool, "time": pd.Timestamp("2026-01-01") + pd.Timedelta(days=m_idx),
                    "die": die, "region": region, "split": split,
                })
    return np.array(intensity), np.array(energy), pd.DataFrame(rows)


class TestNoiseAnalysis(unittest.TestCase):
    """Tests for the GP noise analysis building blocks and orchestrator."""

    def test_fit_recovers_noise_level(self) -> None:
        ary_i, ary_e, _ = _make_data({"Ti2p": 0.05, "O1s": 0.05})
        for noise_model in ("homoscedastic", "poisson_scaled"):
            cfg = {**_FAST, "noise_model": noise_model}
            model = fit_regional_spectrum_by_gp(ary_e[0], ary_i[0], cfg)
            mean, half = create_mean_and_confidence_curves(model, ary_e[0], cfg)
            self.assertEqual(mean.shape, ary_e[0].shape)
            self.assertTrue((half > 0).all())
            self.assertLess(abs(float(model.likelihood.noise.item() ** 0.5 * model.intensity_scale) - 0.05), 0.04)

    def test_confidence_type_latent_is_narrower(self) -> None:
        ary_i, ary_e, _ = _make_data({"Ti2p": 0.1, "O1s": 0.1})
        model = fit_regional_spectrum_by_gp(ary_e[0], ary_i[0], _FAST)
        _, obs = create_mean_and_confidence_curves(model, ary_e[0], {"confidence_type": "observation"})
        _, lat = create_mean_and_confidence_curves(model, ary_e[0], {"confidence_type": "latent"})
        self.assertTrue((obs >= lat).all())
        self.assertGreater(obs.mean(), lat.mean())

    def test_orchestrator_default_excludes_unpaired(self) -> None:
        ary_i, ary_e, df_meta = _make_data({"Ti2p": 0.02, "O1s": 0.2})
        original = df_meta.copy()
        signal, noise, df_out = perform_spectral_noise_analysis(ary_i, ary_e, df_meta, _FAST)
        pd.testing.assert_frame_equal(df_meta, original)
        self.assertEqual(signal.shape, noise.shape)
        self.assertEqual(signal.shape[0], 8)
        analysed = df_out[df_out["noise_analysis_index"].notna()]
        self.assertEqual(sorted(analysed["noise_analysis_index"]), list(range(8)))
        self.assertTrue(df_out.loc[df_out["split"].isna(), "noise_analysis_index"].isna().all())
        # Noisier region has larger noise and smaller SNR.
        by_region = analysed.groupby("region")[["gp_noise_sigma", "gp_snr"]].mean()
        self.assertGreater(by_region.loc["O1s", "gp_noise_sigma"], by_region.loc["Ti2p", "gp_noise_sigma"])
        self.assertLess(by_region.loc["O1s", "gp_snr"], by_region.loc["Ti2p", "gp_snr"])
        # Position consistency: signal row matches the spectrum's own mean.
        row = analysed.iloc[0]
        np.testing.assert_allclose(
            signal[int(row["noise_analysis_index"])], ary_i[int(row["spectrum_index"])], atol=1.0
        )

    def test_selection_filters_and_unpaired_split(self) -> None:
        ary_i, ary_e, df_meta = _make_data({"Ti2p": 0.05, "O1s": 0.05})
        cfg = {**_FAST, "splits": ["train"], "dies": [1], "regions": ["O1s"], "unpaired_split": "train"}
        signal, _, df_out = perform_spectral_noise_analysis(ary_i, ary_e, df_meta, cfg)
        self.assertEqual(signal.shape[0], 2)  # train + unpaired measurement
        analysed = df_out[df_out["noise_analysis_index"].notna()]
        self.assertTrue((analysed["die"] == 1).all() and (analysed["region"] == "O1s").all())

    def test_errors_are_informative(self) -> None:
        ary_i, ary_e, df_meta = _make_data({"Ti2p": 0.05, "O1s": 0.05})
        with self.assertRaisesRegex(ValueError, "Regions"):
            perform_spectral_noise_analysis(ary_i, ary_e, df_meta, {**_FAST, "regions": ["Zz9"]})
        with self.assertRaisesRegex(ValueError, "Unknown noise analysis config key"):
            perform_spectral_noise_analysis(ary_i, ary_e, df_meta, {"bogus": 1})
        with self.assertRaisesRegex(ValueError, "No spectra selected|Dies"):
            perform_spectral_noise_analysis(ary_i, ary_e, df_meta, {**_FAST, "dies": [7]})
        bad = ary_i.copy()
        bad[0, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            perform_spectral_noise_analysis(bad, ary_e, df_meta, {**_FAST, "regions": ["Ti2p"]})
        with self.assertRaisesRegex(ValueError, "Invalid"):
            fit_regional_spectrum_by_gp(ary_e[0], ary_i[0], {"kernel": {"type": "foo"}})
        bad_split = df_meta.copy()
        bad_split.loc[0, "split"] = "foo"
        with self.assertRaisesRegex(ValueError, "Invalid split labels"):
            perform_spectral_noise_analysis(ary_i, ary_e, bad_split, _FAST)
        duplicated = df_meta.copy()
        duplicated.loc[1, "spectrum_index"] = 0
        with self.assertRaisesRegex(ValueError, "Duplicate spectrum_index"):
            perform_spectral_noise_analysis(ary_i, ary_e, duplicated, _FAST)
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            perform_spectral_noise_analysis(ary_i, ary_e, df_meta, {"regions": "Ti2p"})
        with self.assertRaisesRegex(ValueError, "duplicate energy"):
            fit_regional_spectrum_by_gp(np.ones(10), np.arange(10.0))


class TestBatchedAnalysis(unittest.TestCase):
    """Tests for the batched and multiprocess speed paths."""

    def setUp(self) -> None:
        import gc
        gc.collect()

    def tearDown(self) -> None:
        import gc
        gc.collect()

    def test_batched_matches_sequential(self) -> None:
        ary_i, ary_e, df_meta = _make_data({"Ti2p": 0.02, "O1s": 0.2})
        for noise_model in ("homoscedastic", "poisson_scaled"):
            base = {**_FAST, "noise_model": noise_model, "unpaired_split": "train"}
            seq = perform_spectral_noise_analysis(ary_i, ary_e, df_meta, {**base, "batch_size": 1})
            bat = perform_spectral_noise_analysis(ary_i, ary_e, df_meta, {**base, "batch_size": 5})
            np.testing.assert_allclose(bat[0], seq[0], rtol=1e-3, atol=1e-4)
            np.testing.assert_allclose(bat[1], seq[1], rtol=1e-3, atol=1e-4)
            for column in ("gp_noise_sigma", "gp_snr", "gp_lengthscale"):
                np.testing.assert_allclose(bat[2][column], seq[2][column], rtol=1e-3, equal_nan=True)

    def test_verify_rtol_and_workers(self) -> None:
        ary_i, ary_e, df_meta = _make_data({"Ti2p": 0.05, "O1s": 0.1})
        cfg = {**_FAST, "batch_size": 4, "batch_verify_rtol": 1e-3}
        reference = perform_spectral_noise_analysis(ary_i, ary_e, df_meta, cfg)
        parallel = perform_spectral_noise_analysis(ary_i, ary_e, df_meta, {**cfg, "num_workers": 2})
        np.testing.assert_allclose(parallel[0], reference[0], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(parallel[1], reference[1], rtol=1e-6, atol=1e-8)
        sequential_parallel = perform_spectral_noise_analysis(
            ary_i, ary_e, df_meta, {**_FAST, "batch_size": 1, "num_workers": 2}
        )
        np.testing.assert_allclose(sequential_parallel[2]["gp_noise_sigma"].dropna(),
                                   reference[2]["gp_noise_sigma"].dropna(), rtol=1e-3)

    def test_speed_config_validation_and_error_label(self) -> None:
        ary_i, ary_e, df_meta = _make_data({"Ti2p": 0.05, "O1s": 0.1})
        for bad in ({"batch_size": 0}, {"num_workers": -1}, {"batch_verify_rtol": 0}):
            with self.assertRaisesRegex(ValueError, "must be"):
                perform_spectral_noise_analysis(ary_i, ary_e, df_meta, {**_FAST, **bad})
        bad_data = ary_i.copy()
        bad_data[2, 3] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite.*|spectrum meta"):
            perform_spectral_noise_analysis(bad_data, ary_e, df_meta, {**_FAST, "batch_size": 8})


class TestNoisePlots(unittest.TestCase):
    """Tests for the selectable noise analysis plots."""

    def test_all_and_selected_plots(self) -> None:
        ary_i, ary_e, df_meta = _make_data({"Ti2p": 0.05, "O1s": 0.1})
        signal, noise, df_out = perform_spectral_noise_analysis(ary_i, ary_e, df_meta, _FAST)
        data = (ary_i, ary_e, df_out, signal, noise)
        with tempfile.TemporaryDirectory() as tmp:
            figures = plot_spectral_noise_analysis(data, {"save_dir": tmp})
            self.assertEqual(len(figures), 4)
            figures = plot_spectral_noise_analysis(data, {"plots": ["snr_by_region"]})
            self.assertEqual(list(figures), ["snr_by_region"])
        with self.assertRaisesRegex(ValueError, "Unknown plots"):
            plot_spectral_noise_analysis(data, {"plots": ["nope"]})

    def test_trace_gp_hyperparameter_convergence(self) -> None:
        """Verify trace_gp_hyperparameter_convergence records histories."""
        ary_i, ary_e, _ = _make_data({"Ti2p": 0.05, "O1s": 0.1})
        cfg = {"training": {"num_iterations": 10, "learning_rate": 0.1}}
        h = trace_gp_hyperparameter_convergence(ary_e[0], ary_i[0], cfg)
        for key in ("loss", "noise_sigma", "lengthscale", "outputscale"):
            self.assertIn(key, h)
            self.assertEqual(len(h[key]), 10)
            self.assertTrue(all(np.isfinite(val) for val in h[key]))

    def test_plot_gp_hyperparameter_convergence_success_and_save(self) -> None:
        """Verify plot_gp_hyperparameter_convergence creates figure and file."""
        ary_i, ary_e, df_meta = _make_data({"Ti2p": 0.05, "O1s": 0.1})
        with tempfile.TemporaryDirectory() as tmp:
            save_file = os.path.join(tmp, "sub", "test_convergence.png")
            cfg = {
                "measurement_id": "M0",
                "die": 0,
                "num_iterations": 10,
                "save_path": save_file,
            }
            fig = plot_gp_hyperparameter_convergence(
                (ary_i, ary_e, df_meta), cfg
            )
            self.assertIsInstance(fig, matplotlib.figure.Figure)
            self.assertTrue(os.path.exists(save_file))
            self.assertGreater(os.path.getsize(save_file), 1000)

    def test_plot_gp_hyperparameter_convergence_defensive_errors(self) -> None:
        """Verify defensive error checking in GP convergence plot."""
        ary_i, ary_e, df_meta = _make_data({"Ti2p": 0.05, "O1s": 0.1})
        with self.assertRaises(ValueError):
            plot_gp_hyperparameter_convergence("not a tuple")  # type: ignore

        with self.assertRaises(ValueError):
            plot_gp_hyperparameter_convergence(
                (ary_i, ary_e, pd.DataFrame()), {}
            )

        with self.assertRaises(ValueError):
            plot_gp_hyperparameter_convergence(
                (ary_i, ary_e, df_meta),
                {"measurement_id": "nonexistent_id", "die": 0},
            )

        with self.assertRaises(ValueError):
            plot_gp_hyperparameter_convergence(
                (ary_i, ary_e, df_meta),
                {"regions": ["nonexistent_region"]},
            )

    def test_denoise_comparison_plots(self) -> None:
        """Verify spectrum overlay and atomic percentage MAE box plots."""
        ary_i, ary_e, df_meta = _make_data({"Ti2p": 0.08, "O1s": 0.08})
        df_meta["material"] = "NMG"
        _, ary_raw, ary_den = denoise_regional_spectra(
            ary_i,
            ary_e,
            df_meta,
            {"regions": ["Ti2p"], "method": "savgol"},
        )
        fig_spec, axes = plot_denoise_spectrum_comparison(
            ary_e, (ary_raw, ary_den), df_meta, {"die": 0}
        )
        self.assertIsInstance(fig_spec, matplotlib.figure.Figure)
        lines = axes.ravel()[0].get_lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].get_color(), "black")
        self.assertEqual(lines[1].get_color(), "#C71585")
        self.assertEqual(lines[1].get_linestyle(), "--")

        df_samples, df_summary = compare_denoised_atomic_percentages(
            ary_e, (ary_raw, ary_den), df_meta
        )
        self.assertFalse(df_samples.empty)
        self.assertFalse(df_summary.empty)
        fig_box, ax_box = plot_denoise_atomic_percentage_mae(df_samples)
        self.assertIsInstance(fig_box, matplotlib.figure.Figure)
        self.assertIsNotNone(ax_box)
        matplotlib.pyplot.close("all")


class TestDenoiseTechniquesAndOrchestration(unittest.TestCase):
    """Tests for SavGol, Wavelet, Fourier, and denoise orchestration."""

    def test_savgol_wavelet_fourier_single_spectrum(self) -> None:
        """Verify all deterministic techniques reduce noise and estimate SNR."""
        rng = np.random.default_rng(42)
        energy = np.linspace(450.0, 470.0, 100)
        clean = 100.0 + 500.0 * np.exp(-((energy - 460.0) ** 2) / (2 * 1.5**2))
        noisy = clean + rng.normal(0.0, 20.0, energy.size)
        raw_rmse = float(np.sqrt(np.mean((noisy - clean) ** 2)))

        # 1. Savitzky-Golay
        sig_sg, conf_sg, (sigma_sg, snr_sg, win_ev) = denoise_spectrum_savgol(
            energy, noisy, {"window_length": 11, "polynomial_order": 3}
        )
        self.assertLess(
            float(np.sqrt(np.mean((sig_sg - clean) ** 2))), raw_rmse
        )
        self.assertEqual(conf_sg.shape, energy.shape)
        self.assertGreater(sigma_sg, 0.0)
        self.assertGreater(snr_sg, 5.0)
        self.assertGreater(win_ev, 0.0)

        # 2. Wavelet (Stationary & Discrete, Universal & Bayes)
        for transform_type in ("stationary", "discrete"):
            for threshold_rule in ("universal", "bayes"):
                sig_wv, _, (sigma_wv, snr_wv, _) = denoise_spectrum_wavelet(
                    energy,
                    noisy,
                    {
                        "transform_type": transform_type,
                        "wavelet_name": "sym4",
                        "decomposition_level": 2,
                        "threshold_rule": threshold_rule,
                    },
                )
                self.assertLess(
                    float(np.sqrt(np.mean((sig_wv - clean) ** 2))), raw_rmse
                )
                self.assertGreater(sigma_wv, 0.0)
                self.assertGreater(snr_wv, 5.0)

        # 3. Fourier (DCT & Butterworth)
        for filter_type in ("dct", "butterworth"):
            sig_ft, _, (sigma_ft, snr_ft, _) = denoise_spectrum_fourier(
                energy,
                noisy,
                {"filter_type": filter_type, "cutoff_ratio": 0.25},
            )
            self.assertLess(
                float(np.sqrt(np.mean((sig_ft - clean) ** 2))), raw_rmse
            )
            self.assertGreater(sigma_ft, 0.0)
            self.assertGreater(snr_ft, 5.0)

    def test_noise_analysis_with_all_methods(self) -> None:
        """Verify perform_spectral_noise_analysis supports all 4 methods."""
        ary_i, ary_e, df_meta = _make_data({"Ti2p": 0.02, "O1s": 0.2})
        for method in ("savgol", "wavelet", "fourier"):
            sig, noise, df_out = perform_spectral_noise_analysis(
                ary_i,
                ary_e,
                df_meta,
                {"method": method, "show_progress": False},
            )
            self.assertEqual(sig.shape, (8, 120))
            self.assertEqual(noise.shape, (8, 120))
            analysed = df_out[df_out["noise_analysis_index"].notna()]
            by_reg = analysed.groupby("region")[["gp_noise_sigma", "gp_snr"]].mean()
            self.assertGreater(
                by_reg.loc["O1s", "gp_noise_sigma"],
                by_reg.loc["Ti2p", "gp_noise_sigma"],
            )
            self.assertLess(
                by_reg.loc["O1s", "gp_snr"], by_reg.loc["Ti2p", "gp_snr"]
            )

    def test_denoise_regional_spectra_and_apply_flag(self) -> None:
        """Verify deep-copy safety, selective filtering, and apply_denoise."""
        ary_i, ary_e, df_meta = _make_data({"Ti2p": 0.1, "O1s": 0.1})
        raw_backup = ary_i.copy()

        # Case 1: apply_denoise=False -> active matches raw, den modifies Ti2p
        active_false, non_den, den = denoise_regional_spectra(
            ary_i,
            ary_e,
            df_meta,
            {
                "apply_denoise": False,
                "regions": ["Ti2p"],
                "apply_to": "source",
                "source_tool": "J4",
                "target_tool": "J5",
                "method": "savgol",
                "region_configs": {
                    "Ti2p": {
                        "method": "fourier",
                        "fourier": {"filter_type": "dct", "cutoff_ratio": 0.3},
                    }
                },
            },
        )
        np.testing.assert_array_equal(ary_i, raw_backup)
        np.testing.assert_array_equal(non_den, raw_backup)
        np.testing.assert_array_equal(active_false, raw_backup)

        # O1s rows and J5 rows must be untouched in 'den'
        untouched_mask = (df_meta["region"] == "O1s") | (df_meta["tool"] == "J5")
        untouched_idx = df_meta.loc[untouched_mask, "spectrum_index"].to_numpy()
        np.testing.assert_array_equal(den[untouched_idx], raw_backup[untouched_idx])

        # Ti2p rows on J4 (both paired M0 and unpaired M2) must be denoised
        touched_mask = (df_meta["region"] == "Ti2p") & (df_meta["tool"] == "J4")
        touched_idx = df_meta.loc[touched_mask, "spectrum_index"].to_numpy()
        self.assertFalse(
            np.allclose(den[touched_idx], raw_backup[touched_idx])
        )

        # Case 2: apply_denoise=True -> active matches denoised array
        active_true, _, den_true = denoise_regional_spectra(
            ary_i,
            ary_e,
            df_meta,
            {"apply_denoise": True, "regions": ["Ti2p"], "method": "wavelet"},
        )
        np.testing.assert_array_equal(active_true, den_true)
        np.testing.assert_array_equal(ary_i, raw_backup)

    def test_denoise_config_validation_errors(self) -> None:
        """Verify informative errors on invalid denoise configurations."""
        ary_i, ary_e, df_meta = _make_data({"Ti2p": 0.05, "O1s": 0.05})
        with self.assertRaisesRegex(ValueError, "odd integer"):
            denoise_spectrum_savgol(ary_e[0], ary_i[0], {"window_length": 10})
        with self.assertRaisesRegex(ValueError, "Invalid transform_type"):
            denoise_spectrum_wavelet(
                ary_e[0], ary_i[0], {"transform_type": "swt"}
            )
        with self.assertRaisesRegex(ValueError, "cutoff_ratio"):
            denoise_spectrum_fourier(
                ary_e[0], ary_i[0], {"cutoff_ratio": 1.5}
            )
        with self.assertRaisesRegex(ValueError, "not found in df_meta"):
            denoise_regional_spectra(
                ary_i, ary_e, df_meta, {"regions": ["Cl2p"]}
            )


if __name__ == "__main__":
    unittest.main()
