"""Unit and smoke tests for RMS spectral intensity standardization."""

import unittest
import warnings

import numpy as np
import pandas as pd

import models
import utility
from utility.bundle import SpectralDataBundle
from utility.standardize import (
    fit_intensity_standardization,
    inverse_standardize_spectral_intensity,
    standardize_spectral_intensity,
)


def _build_synthetic_spectral_fixture() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build deterministic fixture with source/target tools and all splits."""
    rng = np.random.default_rng(42)
    rows = [
        ("M_J4_0", "J4", "train"),
        ("M_J4_1", "J4", "train"),
        ("M_J4_2", "J4", "val"),
        ("M_J4_3", "J4", "test"),
        ("M_J4_4", "J4", "none"),
        ("M_J5_0", "J5", "train"),
        ("M_J5_1", "J5", "train"),
        ("M_J5_2", "J5", "val"),
        ("M_J5_3", "J5", "test"),
        ("M_J5_4", "J5", "none"),
    ]
    n_rows = len(rows)
    n_points = 32
    ary_intensity = (rng.uniform(0.05, 0.5, size=(n_rows, n_points))).astype(
        np.float32
    )
    ary_energy = np.tile(
        np.linspace(450.0, 470.0, n_points, dtype=np.float32), (n_rows, 1)
    )
    records = [
        {
            "spectrum_index": idx,
            "measurement_id": mid,
            "tool": tool,
            "die": 0,
            "region": "Ti2p",
            "split": split,
            "n_points": n_points,
        }
        for idx, (mid, tool, split) in enumerate(rows)
    ]
    return ary_intensity, ary_energy, pd.DataFrame(records)


class TestIntensityStandardization(unittest.TestCase):
    """Test suite for source-tool train-only RMS intensity standardization."""

    def test_strict_no_data_leakage_across_splits_and_target_tool(self) -> None:
        """Perturbing val/test/none and target_tool rows keeps RMS exact."""
        ary_int, ary_eng, df_meta = _build_synthetic_spectral_fixture()
        cfg = {"enabled": True, "eps": 1e-6, "source_tool": "J4"}

        rms_base = fit_intensity_standardization(ary_int, df_meta, config=cfg)
        ary_std_base, _, _ = standardize_spectral_intensity(
            ary_int, ary_eng, df_meta, config=cfg
        )

        # Perturb val, test, none, and ALL target_tool rows (including train).
        ary_perturbed = ary_int.copy()
        leak_mask = ~((df_meta["tool"] == "J4") & (df_meta["split"] == "train"))
        leak_idx = df_meta.loc[leak_mask, "spectrum_index"].to_numpy(dtype=int)
        ary_perturbed[leak_idx] += np.float32(1e6)

        rms_perturbed = fit_intensity_standardization(
            ary_perturbed, df_meta, config=cfg
        )
        ary_std_perturbed, _, _ = standardize_spectral_intensity(
            ary_perturbed, ary_eng, df_meta, config=cfg
        )

        self.assertEqual(rms_base, rms_perturbed)
        src_train_idx = df_meta.loc[
            (df_meta["tool"] == "J4") & (df_meta["split"] == "train"),
            "spectrum_index",
        ].to_numpy(dtype=int)
        np.testing.assert_array_equal(
            ary_std_base[src_train_idx], ary_std_perturbed[src_train_idx]
        )

    def test_exact_rms_formula_and_split_application(self) -> None:
        """Verify RMS formula, active-split scaling, and untouched none rows."""
        ary_int, ary_eng, df_meta = _build_synthetic_spectral_fixture()
        cfg = {"enabled": True, "eps": 1e-6, "source_tool": "J4"}

        src_train_idx = df_meta.loc[
            (df_meta["tool"] == "J4") & (df_meta["split"] == "train"),
            "spectrum_index",
        ].to_numpy(dtype=int)
        expected_rms = float(
            np.sqrt(np.mean(np.square(ary_int[src_train_idx].astype(np.float64))))
        )

        rms = fit_intensity_standardization(ary_int, df_meta, config=cfg)
        self.assertAlmostEqual(rms, expected_rms, places=12)

        ary_std, ary_eng_out, df_out = standardize_spectral_intensity(
            ary_int, ary_eng, df_meta, config=cfg
        )
        self.assertIs(ary_eng_out, ary_eng)
        self.assertTrue(df_out.attrs["intensity_standardization_enabled"])

        active_mask = df_out["split"].isin(("train", "val", "test"))
        active_idx = df_out.loc[active_mask, "spectrum_index"].to_numpy(dtype=int)
        none_mask = df_out["split"] == "none"
        none_idx = df_out.loc[none_mask, "spectrum_index"].to_numpy(dtype=int)

        np.testing.assert_allclose(
            ary_std[active_idx],
            ary_int[active_idx] / np.float32(expected_rms),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_array_equal(ary_std[none_idx], ary_int[none_idx])
        np.testing.assert_allclose(
            df_out.loc[active_mask, "intensity_standardization_rms"].to_numpy(),
            expected_rms,
        )
        np.testing.assert_array_equal(
            df_out.loc[none_mask, "intensity_standardization_rms"].to_numpy(),
            np.ones(len(none_idx), dtype=np.float64),
        )

    def test_near_zero_rms_clamps_to_eps_with_warning(self) -> None:
        """Zero/near-zero source training spectra emit UserWarning and clamp."""
        ary_int, ary_eng, df_meta = _build_synthetic_spectral_fixture()
        src_train_idx = df_meta.loc[
            (df_meta["tool"] == "J4") & (df_meta["split"] == "train"),
            "spectrum_index",
        ].to_numpy(dtype=int)
        ary_int[src_train_idx] = 0.0
        eps = 1e-5
        cfg = {"enabled": True, "eps": eps, "source_tool": "J4"}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rms = fit_intensity_standardization(ary_int, df_meta, config=cfg)
        self.assertEqual(rms, eps)
        self.assertTrue(
            any(issubclass(w.category, UserWarning) for w in caught)
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ary_std, _, df_out = standardize_spectral_intensity(
                ary_int, ary_eng, df_meta, config=cfg
            )
        self.assertTrue(np.all(np.isfinite(ary_std)))
        self.assertAlmostEqual(
            float(df_out.loc[df_out["split"] == "train", "intensity_standardization_rms"].iloc[0]),
            eps,
        )

    def test_disabled_flag_is_noop(self) -> None:
        """enabled=False leaves intensities unchanged and records factor 1.0."""
        ary_int, ary_eng, df_meta = _build_synthetic_spectral_fixture()
        ary_std, ary_eng_out, df_out = standardize_spectral_intensity(
            ary_int, ary_eng, df_meta, config={"enabled": False, "source_tool": "J4"}
        )
        self.assertIs(ary_eng_out, ary_eng)
        self.assertFalse(df_out.attrs["intensity_standardization_enabled"])
        np.testing.assert_array_equal(ary_std, ary_int)
        np.testing.assert_array_equal(
            df_out["intensity_standardization_rms"].to_numpy(),
            np.ones(len(df_out), dtype=np.float64),
        )

    def test_metadata_recovery_of_normalized_and_raw_intensity(self) -> None:
        """Recorded normalization_scale and RMS recover I_norm and I_raw."""
        ary_raw, ary_eng, df_meta = _build_synthetic_spectral_fixture()
        ary_raw = ary_raw * np.float32(5000.0)
        ary_norm, ary_eng, df_norm = utility.normalize_spectral_data(
            ary_raw,
            ary_eng,
            df_meta,
            config={"normalization_mode": "by_total_flux_single_die", "eps": 1e-6},
        )
        ary_std, ary_eng, df_std = standardize_spectral_intensity(
            ary_norm,
            ary_eng,
            df_norm,
            config={"enabled": True, "eps": 1e-6, "source_tool": "J4"},
        )

        idx = df_std["spectrum_index"].to_numpy(dtype=int)
        rms_col = df_std["intensity_standardization_rms"].to_numpy(dtype=np.float64)[:, None]
        norm_col = df_std["normalization_scale"].to_numpy(dtype=np.float64)[:, None]

        recovered_norm = ary_std[idx].astype(np.float64) * rms_col
        recovered_raw = recovered_norm * norm_col
        np.testing.assert_allclose(recovered_norm, ary_norm[idx], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(recovered_raw, ary_raw[idx], rtol=1e-5, atol=1e-3)

    def test_inverse_standardize_on_stage10_predictions_and_full_df(self) -> None:
        """Inverse standardization restores normalized scale on pred & full."""
        ary_norm, ary_eng, df_meta = _build_synthetic_spectral_fixture()
        cfg = {"enabled": True, "eps": 1e-6, "source_tool": "J4"}
        ary_std, _, df_std = standardize_spectral_intensity(
            ary_norm, ary_eng, df_meta, config=cfg
        )

        # Direct inversion on full df_std preserves 'none' rows untouched.
        ary_inv_full, df_inv_full = inverse_standardize_spectral_intensity(
            ary_std, df_std
        )
        np.testing.assert_allclose(ary_inv_full, ary_norm, rtol=1e-5, atol=1e-6)
        self.assertEqual(
            float(df_inv_full.loc[df_inv_full["split"] == "none", "intensity_standardization_rms"].iloc[0]),
            1.0,
        )

        # Stage 10-like df_meta_pred without RMS column uses reference_df_meta.
        active_src = df_std[
            (df_std["tool"] == "J4") & (df_std["split"].isin(("train", "val", "test")))
        ].reset_index(drop=True)
        ary_pred_std = ary_std[active_src["spectrum_index"].to_numpy(dtype=int)]
        df_meta_pred = pd.DataFrame(
            {
                "spectrum_index": np.arange(len(active_src), dtype=int),
                "tool": "J5",
                "split": active_src["split"].to_numpy(),
            }
        )
        ary_pred_norm, df_pred_out = inverse_standardize_spectral_intensity(
            ary_pred_std, df_meta_pred, reference_df_meta=df_std
        )
        expected_norm = ary_norm[active_src["spectrum_index"].to_numpy(dtype=int)]
        np.testing.assert_allclose(ary_pred_norm, expected_norm, rtol=1e-5, atol=1e-6)
        self.assertIn("intensity_standardization_rms", df_pred_out.columns)
        expected_rms = float(
            df_std.loc[df_std["split"] == "train", "intensity_standardization_rms"].iloc[0]
        )
        np.testing.assert_allclose(
            df_pred_out["intensity_standardization_rms"].to_numpy(), expected_rms
        )

    def test_inverse_standardize_unassigned_prediction_splits(self) -> None:
        """Inverse scaling still applies when pred split is unassigned."""
        ary_norm, ary_eng, df_meta = _build_synthetic_spectral_fixture()
        cfg = {"enabled": True, "eps": 1e-6, "source_tool": "J4"}
        ary_std, _, df_std = standardize_spectral_intensity(
            ary_norm, ary_eng, df_meta, config=cfg
        )
        df_pred = pd.DataFrame(
            {
                "spectrum_index": [0, 1],
                "tool": ["J5", "J5"],
                "split": ["unassigned", "unpaired"],
                "is_predicted": [True, True],
            }
        )
        ary_pred_inv, df_pred_out = inverse_standardize_spectral_intensity(
            ary_std[:2], df_pred, reference_df_meta=df_std
        )
        np.testing.assert_allclose(ary_pred_inv, ary_norm[:2], rtol=1e-5, atol=1e-6)
        expected_rms = float(
            df_std.loc[df_std["split"] == "train", "intensity_standardization_rms"].iloc[0]
        )
        np.testing.assert_allclose(
            df_pred_out["intensity_standardization_rms"].to_numpy(), expected_rms
        )

    def test_fit_error_cases(self) -> None:
        """Invalid config, missing columns, NaN, or duplicates raise error."""
        ary_int, ary_eng, df_meta = _build_synthetic_spectral_fixture()

        with self.assertRaises(ValueError):
            fit_intensity_standardization(ary_int, df_meta, config={})
        with self.assertRaises(ValueError):
            fit_intensity_standardization(
                ary_int, df_meta, config={"source_tool": "J4", "eps": 0.0}
            )
        with self.assertRaises(ValueError):
            fit_intensity_standardization(
                ary_int[0], df_meta, config={"source_tool": "J4"}
            )
        with self.assertRaises(ValueError):
            standardize_spectral_intensity(
                ary_int[0], ary_eng, df_meta, config={"enabled": False}
            )
        for col in ("split", "tool", "spectrum_index"):
            with self.subTest(missing_col=col):
                with self.assertRaises(ValueError):
                    fit_intensity_standardization(
                        ary_int,
                        df_meta.drop(columns=[col]),
                        config={"source_tool": "J4"},
                    )
        with self.assertRaises(ValueError):
            fit_intensity_standardization(
                ary_int, df_meta, config={"source_tool": "NON_EXISTENT_TOOL"}
            )

    def test_inverse_and_data_integrity_error_cases(self) -> None:
        """Bad RMS, 1D arrays, NaN intensities, and duplicates raise error."""
        ary_int, ary_eng, df_meta = _build_synthetic_spectral_fixture()

        with self.assertRaises(ValueError):
            inverse_standardize_spectral_intensity(ary_int, df_meta)
        with self.assertRaises(ValueError):
            inverse_standardize_spectral_intensity(ary_int[0], df_meta.iloc[:1])

        df_bad_rms = df_meta.copy()
        df_bad_rms["intensity_standardization_rms"] = np.nan
        with self.assertRaises(ValueError):
            inverse_standardize_spectral_intensity(ary_int, df_bad_rms)

        df_inconsistent = df_meta.copy()
        df_inconsistent["intensity_standardization_rms"] = np.linspace(
            1.0, 2.0, len(df_inconsistent)
        )
        with self.assertRaises(ValueError):
            inverse_standardize_spectral_intensity(ary_int, df_inconsistent)

        ary_nan = ary_int.copy()
        ary_nan[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_intensity_standardization(
                ary_nan, df_meta, config={"source_tool": "J4"}
            )

        df_dup = df_meta.copy()
        df_dup.loc[df_dup.index[1], "spectrum_index"] = 0
        with self.assertRaises(ValueError):
            standardize_spectral_intensity(
                ary_int, ary_eng, df_dup, config={"enabled": True, "source_tool": "J4"}
            )
        df_dup_rms = df_dup.copy()
        df_dup_rms["intensity_standardization_rms"] = 0.5
        with self.assertRaises(ValueError):
            inverse_standardize_spectral_intensity(ary_int, df_dup_rms)

    def _run_smoke_pipeline(self, use_sliding_window: bool) -> None:
        """Run Stages 1-12 smoke test with standardization enabled."""
        cfg = utility.load_config("config/config.yaml")
        cfg["data_generation"]["measurements_per_tool"] = {"J4": 6, "J5": 6}
        cfg["data_generation"]["measurements_per_t7_code"] = 6
        cfg["data_generation"]["regions"] = ["Ti2p", "O1s"]
        cfg["data_generation"]["seed"] = 123
        cfg["splitting"]["seed"] = 123
        cfg["standardization"]["enabled"] = True
        cfg["dataset"]["regions"] = ["Ti2p", "O1s"]
        cfg["dataset"]["use_sliding_window"] = use_sliding_window
        cfg["training"]["train_config"]["epochs"] = 2
        cfg["training"]["train_config"]["verbose"] = False
        cfg["bayesian_opt"]["use_bayesian_opt"] = False

        bundle = SpectralDataBundle(
            *utility.generate_pseudo_measurements(cfg["data_generation"])
        ).validate()
        df_meta = utility.pair_source_target_spectra(bundle.df_meta, cfg["pairing"])
        df_meta, _, session_splits = utility.split_measurement_sessions(
            df_meta, cfg["splitting"]
        )
        prep_res = utility.preprocess(
            bundle.ary_intensity, bundle.ary_energy, df_meta, config=cfg
        )
        df_meta = prep_res[0]
        ary_norm, ary_energy, df_meta = utility.normalize_spectral_data(
            bundle.ary_intensity, bundle.ary_energy, df_meta, cfg["normalization"]
        )
        ary_model, ary_energy, df_meta = standardize_spectral_intensity(
            ary_norm, ary_energy, df_meta, cfg["standardization"]
        )
        bundle_norm = SpectralDataBundle(ary_norm, ary_energy, df_meta).validate()
        bundle_model = SpectralDataBundle(ary_model, ary_energy, df_meta).validate()

        regional_datasets = models.create_regional_datasets(
            bundle_model.df_meta,
            bundle_model.ary_intensity,
            bundle_model.ary_energy,
            config=cfg["dataset"],
        )
        criterion = models.SpectralLoss(config=cfg["cost"])
        training_results = models.perform_model_training_and_bayesian_optimization(
            regional_datasets,
            criterion,
            config=cfg["training"],
            bayesian_opt_config=cfg["bayesian_opt"],
        )
        pred_cfg = {**cfg["prediction"], "session_splits": session_splits}
        pred_int, pred_eng, df_meta_pred = models.predict_spectra(
            models=training_results["models"],
            data=bundle_model,
            config=pred_cfg,
        )
        pred_int, df_meta_pred = inverse_standardize_spectral_intensity(
            pred_int, df_meta_pred, reference_df_meta=bundle_model.df_meta
        )
        bundle_pred = SpectralDataBundle(
            pred_int, pred_eng, df_meta_pred
        ).validate()

        df_metrics, _ = utility.calculate_prediction_metrics(
            bundle_norm, bundle_pred, config=cfg["metrics_report"]
        )
        self.assertFalse(df_metrics.empty)
        self.assertTrue(np.all(np.isfinite(bundle_pred.ary_intensity)))
        self.assertIn("intensity_standardization_rms", bundle_pred.df_meta.columns)

        _, df_atomic_summary = utility.calculate_atomic_percentage_split_statistics(
            bundle_norm, bundle_pred, config=cfg["metrics_report"]
        )
        self.assertFalse(df_atomic_summary.empty)

    def test_end_to_end_smoke_full_spectrum(self) -> None:
        """Smoke test full-spectrum pipeline with standardization enabled."""
        self._run_smoke_pipeline(use_sliding_window=False)

    def test_end_to_end_smoke_sliding_window(self) -> None:
        """Smoke test sliding-window pipeline with standardization enabled."""
        self._run_smoke_pipeline(use_sliding_window=True)


if __name__ == "__main__":
    unittest.main()
