"""Unit tests for the unified 8-step spectral transformation orchestration pipeline."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
import torch

from models.cost import NormalizedMSELoss, format_loss_log10
from models.orchestration import (
    _resolve_region_config,
    instantiate_model,
    run_residual_unet_pipeline,
    run_resnet_pipeline,
    run_spectral_pipeline,
    run_unet_pipeline,
)
from models.resnet import ResNet1D
from models.unet import ConventionalUNet1D, ResidualUNet1D, UNet1D
from utility.pairing import pair_source_target_spectra
from utility.pseudo_measurement import generate_pseudo_measurements


class TestOrchestrationPipeline(unittest.TestCase):
    """Test suite for the unified 8-step pipeline across models and sliding window toggles."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate a lightweight synthetic dataset for pipeline execution."""
        pseudo_cfg = {
            "material": "NMG",
            "regions": ["Al2p", "Ti2p"],
            "n_points": 40,
            "measurements_per_tool": {"J4": 8, "H1": 8},
            "interval_hours_range": (1.0, 4.0),
            "tool_offsets": {
                "J4": {"shift_ev": 0.0, "scale": 1000.0},
                "H1": {"shift_ev": 0.5, "scale": 1200.0},
            },
            "seed": 123,
        }
        ary_intensity, ary_energy, meta_df = generate_pseudo_measurements(pseudo_cfg)
        meta_df = pair_source_target_spectra(
            meta_df,
            config={"source_tool": "J4", "target_tool": "H1", "time_threshold_hours": 12.0},
        )
        cls.ary_intensity = ary_intensity
        cls.ary_energy = ary_energy
        cls.meta_df = meta_df

    def test_instantiate_model(self) -> None:
        """Verify model instantiation across supported types and error handling."""
        m_res = instantiate_model("resnet", seq_len=30, config={"hidden_channels": 8})
        self.assertIsInstance(m_res, ResNet1D)

        m_res_u = instantiate_model("residual_unet", seq_len=32, config={"base_channels": 4, "depth": 2})
        self.assertIsInstance(m_res_u, ResidualUNet1D)
        self.assertTrue(m_res_u.residual)

        m_conv_u = instantiate_model("unet", seq_len=32, config={"base_channels": 4, "depth": 2})
        self.assertIsInstance(m_conv_u, ConventionalUNet1D)
        self.assertFalse(m_conv_u.residual)

        with self.assertRaises(ValueError):
            instantiate_model("unknown_arch", seq_len=30)

    def test_format_loss_log10(self) -> None:
        """Verify log10 loss string formatting."""
        s1 = format_loss_log10(0.001)
        self.assertIn("log10: -3.0000", s1)

        s2 = format_loss_log10(0.0)
        self.assertIn("log10: N/A", s2)

        s3 = format_loss_log10(-0.5)
        self.assertIn("log10: N/A", s3)

    def test_run_resnet_full_spectrum(self) -> None:
        """Verify 8-step pipeline execution for ResNet in full-spectrum mode."""
        cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p"],
            "train_config": {"epochs": 2, "batch_size": 4, "verbose": True},
            "predict_source": True,
        }
        res = run_spectral_pipeline(
            data=(self.ary_intensity, self.ary_energy, self.meta_df),
            model_type="resnet",
            use_sliding_window=False,
            config=cfg,
        )
        self.assertIn("Al2p", res["models"])
        self.assertIsInstance(res["models"]["Al2p"], ResNet1D)
        self.assertIn("Al2p", res["evaluation"])
        self.assertIn("Al2p", res["test_evaluation"])
        self.assertIsNotNone(res["predictions"])

        pred_i, pred_e, pred_m = res["predictions"]
        self.assertEqual(pred_i.shape[1], 40)
        self.assertTrue(np.all(pred_i >= 0.0))  # zero clamping verified

    def test_run_resnet_sliding_window(self) -> None:
        """Verify 8-step pipeline execution for ResNet with sliding window patch augmentation."""
        cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p"],
            "window_size": 15,
            "stride": 7,
            "train_config": {"epochs": 2, "batch_size": 8, "verbose": True},
            "predict_source": True,
        }
        res = run_spectral_pipeline(
            data=(self.ary_intensity, self.ary_energy, self.meta_df),
            model_type="resnet",
            use_sliding_window=True,
            config=cfg,
        )
        self.assertIn("Al2p", res["models"])
        model_entry = res["models"]["Al2p"]
        self.assertIsInstance(model_entry, tuple)
        self.assertIsInstance(model_entry[0], ResNet1D)
        self.assertEqual(model_entry[1], 15)  # window_size
        self.assertEqual(model_entry[2], 7)   # stride

        pred_i, _, _ = res["predictions"]
        self.assertEqual(pred_i.shape[1], 40)
        self.assertTrue(np.all(pred_i >= 0.0))

    def test_run_residual_unet_pipelines(self) -> None:
        """Verify Residual U-Net pipeline execution in both full and sliding window modes."""
        # Full spectrum
        cfg_full = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Ti2p"],
            "train_config": {"epochs": 2, "batch_size": 4, "base_channels": 4, "depth": 2},
            "predict_source": True,
        }
        res_full = run_residual_unet_pipeline(
            meta_df=self.meta_df,
            ary_intensity=self.ary_intensity,
            ary_energy=self.ary_energy,
            config=cfg_full,
        )
        self.assertIsInstance(res_full["models"]["Ti2p"], ResidualUNet1D)

        # Sliding window
        cfg_sw = dict(cfg_full)
        cfg_sw["use_sliding_window"] = True
        cfg_sw["window_size"] = 16
        cfg_sw["stride"] = 8
        res_sw = run_residual_unet_pipeline(
            meta_df=self.meta_df,
            ary_intensity=self.ary_intensity,
            ary_energy=self.ary_energy,
            config=cfg_sw,
        )
        self.assertIsInstance(res_sw["models"]["Ti2p"][0], ResidualUNet1D)

    def test_run_conventional_unet_pipeline(self) -> None:
        """Verify Conventional U-Net (no shortcut) pipeline execution."""
        cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p"],
            "train_config": {"epochs": 2, "batch_size": 4, "base_channels": 4, "depth": 2},
            "predict_source": True,
        }
        res = run_unet_pipeline(
            meta_df=self.meta_df,
            ary_intensity=self.ary_intensity,
            ary_energy=self.ary_energy,
            config=cfg,
        )
        self.assertIsInstance(res["models"]["Al2p"], ConventionalUNet1D)
        self.assertFalse(res["models"]["Al2p"].residual)

    def test_run_spectral_pipeline_with_bayesian_opt(self) -> None:
        """Verify unified pipeline execution with Step 6 Ax Bayesian optimization enabled."""
        cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p"],
            "use_bayesian_opt": True,
            "bayesian_opt_config": {
                "num_trials": 2,
                "epochs_per_trial": 2,
                "window_size_choices": [15, 21],
                "kernel_sizes": [3],
                "hidden_channels": [8],
            },
            "train_config": {"epochs": 2, "batch_size": 4},
            "predict_source": True,
        }
        res = run_spectral_pipeline(
            data=(self.ary_intensity, self.ary_energy, self.meta_df),
            model_type="resnet",
            use_sliding_window=True,
            config=cfg,
        )
        self.assertIn("Al2p", res["bayesian_opt_results"])
        self.assertIn("best_parameters", res["bayesian_opt_results"]["Al2p"])
        self.assertIn("Al2p", res["models"])
        model_entry = res["models"]["Al2p"]
        self.assertIsInstance(model_entry, tuple)
        # Verify window_size is one of the searched choices
        self.assertIn(model_entry[1], [15, 21])

    def test_resolve_region_config(self) -> None:
        """Verify hierarchical configuration resolution and per-region overrides."""
        global_cfg = {
            "window_size_ev": 4.0,
            "sliding_stride_ev": 2.0,
            "train_config": {"batch_size": 32, "learning_rate": 1e-4, "depth": 3},
            "bayesian_opt_config": {"num_trials": 3, "batch_sizes": [16, 32]},
            "region_configs": {
                "Al2p": {
                    "window_size_ev": 2.5,
                    "sliding_stride_ev": 0.8,
                    "batch_size": 16,
                    "learning_rate": 2e-3,
                },
                "Ti2p": {
                    "depth": 2,
                    "bayesian_opt_config": {"batch_sizes": [8, 16, 32]},
                },
            },
        }

        # Region with explicit overrides
        sw_al, tr_al, bo_al = _resolve_region_config(global_cfg, "Al2p")
        self.assertEqual(sw_al["window_size_ev"], 2.5)
        self.assertEqual(sw_al["sliding_stride_ev"], 0.8)
        self.assertEqual(tr_al["batch_size"], 16)
        self.assertEqual(tr_al["learning_rate"], 2e-3)
        self.assertEqual(tr_al["depth"], 3)  # inherited from global train_config
        self.assertEqual(bo_al["batch_sizes"], [16, 32])  # inherited from global bo_cfg

        # Region with nested and arch overrides
        sw_ti, tr_ti, bo_ti = _resolve_region_config(global_cfg, "Ti2p")
        self.assertEqual(sw_ti["window_size_ev"], 4.0)  # default
        self.assertEqual(tr_ti["batch_size"], 32)  # default
        self.assertEqual(tr_ti["depth"], 2)  # overridden
        self.assertEqual(bo_ti["batch_sizes"], [8, 16, 32])  # overridden

        # Region without explicit overrides falls back entirely to globals
        sw_c, tr_c, bo_c = _resolve_region_config(global_cfg, "C1s")
        self.assertEqual(sw_c["window_size_ev"], 4.0)
        self.assertEqual(tr_c["batch_size"], 32)
        self.assertEqual(tr_c["learning_rate"], 1e-4)
        self.assertEqual(bo_c["num_trials"], 3)

    def test_run_spectral_pipeline_with_region_configs(self) -> None:
        """Verify pipeline execution with heterogeneous per-region window sizes and batch sizes."""
        cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p", "Ti2p"],
            "window_size": 20,
            "stride": 10,
            "train_config": {"epochs": 2, "batch_size": 8, "base_channels": 4, "depth": 2},
            "region_configs": {
                "Al2p": {
                    "window_size": 15,
                    "stride": 7,
                    "batch_size": 4,
                },
                "Ti2p": {
                    "window_size": 21,
                    "stride": 9,
                    "batch_size": 8,
                },
            },
            "predict_source": True,
        }
        res = run_spectral_pipeline(
            data=(self.ary_intensity, self.ary_energy, self.meta_df),
            model_type="residual_unet",
            use_sliding_window=True,
            config=cfg,
        )
        self.assertIn("Al2p", res["models"])
        self.assertIn("Ti2p", res["models"])

        # Check that Al2p used its custom window size 15
        m_al, w_al, s_al = res["models"]["Al2p"]
        self.assertEqual(w_al, 15)
        self.assertEqual(s_al, 7)

        # Check that Ti2p used its custom window size 21
        m_ti, w_ti, s_ti = res["models"]["Ti2p"]
        self.assertEqual(w_ti, 21)
        self.assertEqual(s_ti, 9)

        # Ensure prediction completed and clamped for all regions
        pred_i, _, _ = res["predictions"]
        self.assertEqual(pred_i.shape[1], 40)
        self.assertTrue(np.all(pred_i >= 0.0))

    def test_bayesian_opt_with_batch_sizes(self) -> None:
        """Verify that batch_sizes parameter search works in Bayesian optimization."""
        cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p"],
            "use_bayesian_opt": True,
            "bayesian_opt_config": {
                "num_trials": 2,
                "epochs_per_trial": 2,
                "batch_sizes": [4, 8],
                "kernel_sizes": [3],
                "hidden_channels": [8],
            },
            "train_config": {"epochs": 2},
            "predict_source": False,
        }
        res = run_spectral_pipeline(
            data=(self.ary_intensity, self.ary_energy, self.meta_df),
            model_type="resnet",
            use_sliding_window=False,
            config=cfg,
        )
        bo_res = res["bayesian_opt_results"]["Al2p"]
        self.assertIn("best_parameters", bo_res)
        best_p = bo_res["best_parameters"]
        self.assertIn("batch_size", best_p)
        self.assertIn(best_p["batch_size"], [4, 8])

    def test_bayesian_opt_invalid_batch_sizes(self) -> None:
        """Verify ValueError is raised if batch_sizes contains non-positive values."""
        from models.bayesian_opt import optimize_model_hyperparameters

        cfg = {"batch_sizes": [0, 16]}
        with self.assertRaises(ValueError):
            optimize_model_hyperparameters(
                train_dataset=None,  # type: ignore
                val_dataset=None,  # type: ignore
                config=cfg,
            )

    def test_resolve_region_config_singular_and_aliases(self) -> None:
        """Verify that 'region_config' (singular) and alias keys like 'epoch', 'lr', 'l2', 'batch' are resolved."""
        global_cfg = {
            "train_config": {"epochs": 100, "learning_rate": 1e-4, "batch_size": 16, "l2_weight": 1e-6},
            "region_config": {
                "Ti2p": {
                    "epoch": 200,
                    "lr": 5e-3,
                    "batch": 32,
                    "l2": 1e-4,
                    "patience": 7,
                    "use_film": True,
                    "model_type": "residual_unet",
                },
                "Al2p": {
                    "train_config": {
                        "epoch": 150,
                    },
                },
            },
        }

        # Ti2p with flat aliases and singular 'region_config'
        sw_ti, tr_ti, bo_ti = _resolve_region_config(global_cfg, "Ti2p")
        self.assertEqual(tr_ti["epochs"], 200)
        self.assertEqual(tr_ti["epoch"], 200)
        self.assertEqual(tr_ti["learning_rate"], 5e-3)
        self.assertEqual(tr_ti["lr"], 5e-3)
        self.assertEqual(tr_ti["batch_size"], 32)
        self.assertEqual(tr_ti["batch"], 32)
        self.assertEqual(tr_ti["l2_weight"], 1e-4)
        self.assertEqual(tr_ti["early_stopping_patience"], 7)
        self.assertTrue(tr_ti["use_film"])
        self.assertEqual(tr_ti["model_type"], "residual_unet")

        # Al2p with nested train_config using 'epoch'
        sw_al, tr_al, bo_al = _resolve_region_config(global_cfg, "Al2p")
        self.assertEqual(tr_al["epochs"], 150)
        self.assertEqual(tr_al["epoch"], 150)
        self.assertEqual(tr_al["learning_rate"], 1e-4)  # inherited from global train_config

    def test_run_spectral_pipeline_with_regional_epoch_and_film(self) -> None:
        """Verify pipeline respects per-region epoch, model_type, and use_film overrides."""
        cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p", "Ti2p"],
            "model_type": "resnet",
            "use_film": False,
            "train_config": {"epochs": 2, "batch_size": 8, "hidden_channels": 8},
            "region_config": {
                "Ti2p": {
                    "epoch": 3,
                    "use_film": True,
                },
            },
            "predict_source": False,
        }
        res = run_spectral_pipeline(
            data=(self.ary_intensity, self.ary_energy, self.meta_df),
            config=cfg,
        )
        self.assertIn("Al2p", res["models"])
        self.assertIn("Ti2p", res["models"])

        # Al2p model does not have FiLM
        m_al = res["models"]["Al2p"]
        self.assertFalse(getattr(m_al, "use_film", False))

        # Ti2p model has FiLM active
        m_ti = res["models"]["Ti2p"]
        self.assertTrue(getattr(m_ti, "use_film", False))
        self.assertIsNotNone(getattr(m_ti, "film_gen", None))

        # Histories verify epochs
        hist_al = res["histories"]["Al2p"]
        hist_ti = res["histories"]["Ti2p"]
        self.assertEqual(len(hist_al["train_loss"]), 2)
        self.assertEqual(len(hist_ti["train_loss"]), 3)

    def test_resolve_region_config_late_aliases_and_bo_aliases(self) -> None:
        """Verify late aliases in alias groups and nested BO config aliases synchronize properly."""
        global_cfg = {
            "use_bayesian_opt": True,
            "train_config": {"epochs": 50, "patience": 5, "l2": 1e-4},
            "bayesian_opt_config": {"use_bo": True, "num_trials": 3},
            "region_config": {
                "Ti2p": {
                    "max_epochs": 120,
                    "early_stop_patience": 12,
                    "weight_decay_l2": 2e-5,
                    "bayesian_opt_config": {"use_bo": False},
                },
            },
        }
        _, tr_ti, bo_ti = _resolve_region_config(global_cfg, "Ti2p")
        # Verify late aliases synchronize to all group members
        self.assertEqual(tr_ti["epochs"], 120)
        self.assertEqual(tr_ti["epoch"], 120)
        self.assertEqual(tr_ti["max_epochs"], 120)
        self.assertEqual(tr_ti["early_stopping_patience"], 12)
        self.assertEqual(tr_ti["patience"], 12)
        self.assertEqual(tr_ti["early_stop_patience"], 12)
        self.assertEqual(tr_ti["l2_weight"], 2e-5)
        self.assertEqual(tr_ti["l2"], 2e-5)
        self.assertEqual(tr_ti["weight_decay_l2"], 2e-5)
        # Verify BO config normalization
        self.assertFalse(bo_ti["use_bo"])
        self.assertFalse(bo_ti["use_bayesian_opt"])

    def test_empty_region_configs_fallback(self) -> None:
        """Verify that an empty 'region_configs: {}' cleanly falls back to 'region_config'."""
        global_cfg = {
            "region_configs": {},
            "region_config": {
                "Ti2p": {"epoch": 75},
            },
        }
        _, tr_ti, _ = _resolve_region_config(global_cfg, "Ti2p")
        self.assertEqual(tr_ti["epochs"], 75)

    def test_pipeline_mixed_sliding_window_inference(self) -> None:
        """Verify prediction runs smoothly when one region uses sliding window and another uses full spectrum."""
        cfg = {
            "source_tool": "J4",
            "target_tool": "H1",
            "regions": ["Al2p", "Ti2p"],
            "use_sliding_window": True,  # global default True
            "train_config": {"epochs": 2, "batch_size": 8, "hidden_channels": 8},
            "region_config": {
                "Al2p": {
                    "window_size": 15,
                    "stride": 7,
                },
                "Ti2p": {
                    "use_sliding_window": False,  # Ti2p overrides to full spectrum
                },
            },
            "predict_source": True,
        }
        res = run_spectral_pipeline(
            data=(self.ary_intensity, self.ary_energy, self.meta_df),
            config=cfg,
        )
        # Al2p model is saved as a tuple (model, w_size, stride)
        self.assertIsInstance(res["models"]["Al2p"], tuple)
        # Ti2p model is saved as a standalone nn.Module
        self.assertNotIsInstance(res["models"]["Ti2p"], tuple)

        # Predictions generated without dimension mismatch
        pred_i, pred_e, pred_meta = res["predictions"]
        self.assertEqual(pred_i.shape[1], 40)
        self.assertTrue(len(pred_meta) > 0)


if __name__ == "__main__":
    unittest.main()


