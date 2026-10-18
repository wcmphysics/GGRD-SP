"""Unit tests for models.noise_analysis and visualization.plot_noise_analysis."""

from __future__ import annotations

import tempfile
import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend for testing

from models.noise_analysis import (
    create_mean_and_confidence_curves,
    fit_regional_spectrum_by_gp,
    perform_spectral_noise_analysis,
)
from visualization.plot_noise_analysis import plot_spectral_noise_analysis

_FAST = {"training": {"num_iterations": 60, "learning_rate": 0.1}, "show_progress": False}


def _make_data(
    noise_levels: dict[str, float], n_points: int = 120, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build canonical arrays: 2 regions x 2 dies, 2 measurements with splits."""
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


if __name__ == "__main__":
    unittest.main()
