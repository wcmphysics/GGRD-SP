"""Unit tests for models.gp_flux and visualization.plot_gp_die_flux_timeline."""

from __future__ import annotations

import unittest

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")  # Non-interactive backend for testing

from models.gp_flux import (
    DieFluxExactGPModel,
    SigmoidStepKernel,
    build_composite_kernel,
    build_gp_flux_dataset,
    evaluate_gp_die_flux_models,
    get_default_gp_flux_config,
    predict_gp_die_flux,
    run_gp_flux_pipeline,
    train_gp_die_flux_model,
)
from utility.flux_and_intensity import calculate_flux_and_intensity
from utility.pairing import pair_source_target_spectra
from utility.pseudo_measurement import generate_pseudo_measurements
from utility.splitting import split_measurement_sessions
from visualization.plot_timeline import plot_gp_die_flux_timeline


class TestSigmoidStepKernel(unittest.TestCase):
    """Test suite for SigmoidStepKernel covariance mathematics."""

    def test_kernel_forward_shape_and_symmetry(self) -> None:
        """Verify step kernel evaluation shape, symmetry, and bounds."""
        kern = SigmoidStepKernel(step_time=5.0, steepness=2.0)
        x = torch.tensor([[1.0], [5.0], [9.0]], dtype=torch.float32)
        k_mat = kern(x, x).to_dense()

        self.assertEqual(k_mat.shape, (3, 3))
        # Symmetry check
        np.testing.assert_allclose(
            k_mat.detach().numpy(),
            k_mat.detach().numpy().T,
            atol=1e-6,
        )
        # Bounded between 0 and 1
        self.assertTrue((k_mat >= 0.0).all())
        self.assertTrue((k_mat <= 1.0).all())

    def test_kernel_diag(self) -> None:
        """Verify diag=True matches diagonal of full covariance evaluation."""
        kern = SigmoidStepKernel(step_time=4.0, steepness=1.5)
        x = torch.tensor([[2.0], [4.0], [6.0]], dtype=torch.float32)
        full_diag = torch.diagonal(kern(x, x).to_dense())
        fast_diag = kern(x, x, diag=True)
        np.testing.assert_allclose(
            fast_diag.detach().numpy(),
            full_diag.detach().numpy(),
            atol=1e-6,
        )

    def test_learnable_step_time_toggle(self) -> None:
        """Verify learn_step_time flag toggles Parameter vs Buffer."""
        kern_learn = SigmoidStepKernel(learn_step_time=True)
        self.assertIn("raw_step_time", dict(kern_learn.named_parameters()))

        kern_fixed = SigmoidStepKernel(learn_step_time=False)
        self.assertNotIn("raw_step_time", dict(kern_fixed.named_parameters()))
        self.assertIn("raw_step_time", dict(kern_fixed.named_buffers()))


class TestBuildCompositeKernel(unittest.TestCase):
    """Test suite for configurable kernel composition factory."""

    def test_kernel_factory_options(self) -> None:
        """Verify building kernels with various continuous types."""
        for k_type in ["rbf", "matern52", "matern32", "matern12"]:
            cfg = {
                "time_continuous": {"type": k_type, "initial_lengthscale": 3.0},
                "time_jump": {"enabled": True, "initial_step_time_days": 8.0},
                "die": {"type": "index", "num_dies": 9, "rank": 2},
                "composition": "product_with_jump",
            }
            kernel = build_composite_kernel(cfg)
            x = torch.tensor([[1.0, 0], [5.0, 1]], dtype=torch.float32)
            cov = kernel(x, x).to_dense()
            self.assertEqual(cov.shape, (2, 2))

    def test_kernel_without_jump(self) -> None:
        """Verify building kernel with time jump disabled."""
        cfg = {
            "time_continuous": {"type": "rbf"},
            "time_jump": {"enabled": False},
            "die": {"type": "rbf"},
            "composition": "product",
        }
        kernel = build_composite_kernel(cfg)
        x = torch.tensor([[1.0, 0.0], [3.0, 1.0]], dtype=torch.float32)
        cov = kernel(x, x).to_dense()
        self.assertEqual(cov.shape, (2, 2))


class TestGPFluxPipeline(unittest.TestCase):
    """Integration test suite for dataset building, training, and evaluation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create deterministic multi-tool dataset through Stage 4 splitting."""
        gen_cfg = {
            "material": "NMG",
            "regions": ["Al2p", "Ti2p"],
            "n_points": 50,
            "measurements_per_tool": {"J4": 6, "J5": 6},
            "measurements_per_t7_code": 3,
            "interval_hours_range": (2.0, 6.0),
            "die_variation_std": 0.02,
            "noise_relative_std": 0.01,
            "seed": 42,
        }
        i_arr, e_arr, meta = generate_pseudo_measurements(gen_cfg)
        meta = calculate_flux_and_intensity(i_arr, e_arr, meta)
        meta = pair_source_target_spectra(
            meta,
            {
                "source_tool": "J4",
                "target_tool": "J5",
                "time_threshold_hours": 12.0,
                "material": "all",
            },
        )
        meta, _, _ = split_measurement_sessions(
            meta,
            {
                "source_tool": "J4",
                "target_tool": "J5",
                "train_ratio": 0.5,
                "val_ratio": 0.25,
                "test_ratio": 0.25,
            },
        )
        cls.df_meta = meta

    def test_build_gp_flux_dataset(self) -> None:
        """Verify dataset extraction, absolute elapsed days, and splitting."""
        cfg = get_default_gp_flux_config()
        cfg["reference_date"] = "2026-01-01"
        ds = build_gp_flux_dataset(self.df_meta, "J4", cfg)

        self.assertEqual(ds["tool"], "J4")
        self.assertGreater(len(ds["train_x"]), 0)
        self.assertEqual(ds["train_x"].shape[1], 2)
        # Check continuous absolute elapsed days are positive
        self.assertTrue((ds["train_x"][:, 0] >= 0.0).all())
        # Target normalization check: train mean ~ 0, std ~ 1
        self.assertAlmostEqual(float(ds["train_y"].mean()), 0.0, places=4)
        self.assertAlmostEqual(float(ds["train_y"].std(unbiased=False)), 1.0, places=4)

    def test_unpaired_session_routing(self) -> None:
        """Verify unpaired sessions (split='none') route to unpaired_split."""
        meta_with_none = self.df_meta.copy()
        # Set one measurement to 'none'
        first_m = meta_with_none["measurement_id"].iloc[0]
        meta_with_none.loc[meta_with_none["measurement_id"] == first_m, "split"] = "none"

        cfg = {"unpaired_split": "train"}
        ds = build_gp_flux_dataset(meta_with_none, "J4", cfg)
        unpaired_rows = ds["die_df"][ds["die_df"]["measurement_id"] == first_m]
        self.assertTrue((unpaired_rows["split"] == "train").all())

    def test_train_and_predict(self) -> None:
        """Verify model training convergence and posterior prediction bounds."""
        cfg = {
            "training": {"num_iterations": 20, "learning_rate": 0.05},
            "kernel": {"time_jump": {"enabled": True}},
        }
        ds = build_gp_flux_dataset(self.df_meta, "J4", cfg)
        model, likelihood = train_gp_die_flux_model(ds["train_x"], ds["train_y"], cfg)

        self.assertIsInstance(model, DieFluxExactGPModel)
        pred_mean, pred_lower, pred_upper = predict_gp_die_flux(
            model, likelihood, ds["train_x"][:5], ds["scaler"]
        )

        self.assertEqual(len(pred_mean), 5)
        # Check 2-sigma bounds: lower <= mean <= upper
        self.assertTrue((pred_lower <= pred_mean + 1e-4).all())
        self.assertTrue((pred_mean <= pred_upper + 1e-4).all())
        # Check physical scale flux is positive
        self.assertTrue((pred_mean > 0.0).all())

    def test_run_gp_flux_pipeline_and_evaluate(self) -> None:
        """Verify pipeline orchestration and evaluation table output."""
        cfg = {
            "training": {"num_iterations": 15, "learning_rate": 0.05},
        }
        bundle, df_metrics = run_gp_flux_pipeline(self.df_meta, cfg)

        self.assertIn("models", bundle)
        self.assertIn("datasets", bundle)
        self.assertIn("J4", bundle["models"])
        self.assertIn("J5", bundle["models"])

        # Check metrics DataFrame columns
        expected_cols = [
            "tool",
            "die",
            "mae_train",
            "mape_train_percentage",
            "mae_val",
            "mape_val_percentage",
            "mae_test",
            "mape_test_percentage",
        ]
        self.assertEqual(list(df_metrics.columns), expected_cols)
        # Check presence of tool mean rows
        self.assertTrue((df_metrics["die"] == "All").any())

    def test_plot_gp_die_flux_timeline(self) -> None:
        """Verify two-panel timeline visualization generates cleanly."""
        cfg = {
            "training": {"num_iterations": 10, "learning_rate": 0.05},
            "plot": {"plot_die": 0},
        }
        bundle, _ = run_gp_flux_pipeline(self.df_meta, cfg)
        fig, (ax_s, ax_t) = plot_gp_die_flux_timeline(bundle, cfg)

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax_s)
        self.assertIsNotNone(ax_t)
        self.assertIn("Die 0", ax_s.get_title())
        self.assertIn("Die 0", ax_t.get_title())

        # Verify custom split_colors support
        cfg_custom = {**cfg, "split_colors": {"test": "limegreen"}}
        fig_cust, _ = plot_gp_die_flux_timeline(bundle, cfg_custom)
        self.assertIsNotNone(fig_cust)

    def test_tool_ordering_robustness(self) -> None:
        """Verify pipeline correctly identifies tools when rows are inverted."""
        inverted_meta = self.df_meta.sort_values(by="tool", ascending=False)
        cfg = {"training": {"num_iterations": 5, "learning_rate": 0.05}}
        bundle, _ = run_gp_flux_pipeline(inverted_meta, cfg)
        self.assertIn("J4", bundle["models"])
        self.assertIn("J5", bundle["models"])


if __name__ == "__main__":
    unittest.main()
