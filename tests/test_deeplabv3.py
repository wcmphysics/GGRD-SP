"""Unit tests for the 1D DeepLabV3 neural network architecture and pipeline."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from models.bayesian_opt import optimize_model_hyperparameters
from models.cost import NormalizedMSELoss
from models.dataset import SpectrumPairDataset
from models.deeplabv3 import ASPP1D, DeepLabV3, DeepLabV3_1D, ResNet1DBackbone
from models.orchestration import instantiate_model, run_deeplabv3_pipeline
from models.trainer import train_model_region
from utility.pairing import pair_source_target_spectra
from utility.pseudo_measurement import generate_pseudo_measurements


class TestDeepLabV3Architecture(unittest.TestCase):
    """Test suite for DeepLabV3 model modules and forward transformations."""

    def test_output_shapes_2d_and_3d(self) -> None:
        """Verify 1D DeepLabV3 preserves exact sequence length for 2D and 3D inputs."""
        model = DeepLabV3(
            n_points=100,
            config={
                "backbone_channels": 16,
                "aspp_channels": 16,
                "kernel_size": 3,
                "multi_grid": (1, 2, 4),
                "aspp_rates": (2, 4, 6),
            },
        )

        # 2D input (batch, n_points)
        x_2d = torch.randn(4, 100)
        y_2d = model(x_2d)
        self.assertEqual(y_2d.shape, (4, 100))

        # 3D input (batch, 1, n_points)
        x_3d = torch.randn(4, 1, 100)
        y_3d = model(x_3d)
        self.assertEqual(y_3d.shape, (4, 1, 100))

    def test_various_sequence_lengths(self) -> None:
        """Verify model handles various sequence lengths including small patches and odd lengths."""
        lengths = [15, 25, 50, 77, 101]
        for seq_len in lengths:
            model = DeepLabV3(
                n_points=seq_len,
                config={
                    "backbone_channels": 8,
                    "aspp_channels": 8,
                    "kernel_size": 3,
                    "multi_grid": (1, 2, 4),
                    "aspp_rates": (2, 4, 6),
                },
            )
            x = torch.randn(2, seq_len)
            y = model(x)
            self.assertEqual(y.shape, (2, seq_len), f"Mismatch for sequence length {seq_len}")

    def test_multi_grid_adaptation(self) -> None:
        """Verify backbone dynamically adapts number of residual blocks to multi_grid tuple."""
        # 3 blocks
        backbone_3 = ResNet1DBackbone(
            in_channels=1,
            out_channels=16,
            multi_grid=(1, 2, 4),
            kernel_size=3,
            n_points=50,
        )
        self.assertEqual(len(backbone_3.blocks), 3)

        # 4 blocks
        backbone_4 = ResNet1DBackbone(
            in_channels=1,
            out_channels=16,
            multi_grid=(1, 2, 2, 4),
            kernel_size=3,
            n_points=50,
        )
        self.assertEqual(len(backbone_4.blocks), 4)

        # Test forward pass with custom multi_grid in full DeepLabV3
        model_custom = DeepLabV3(
            n_points=50,
            config={"multi_grid": (1, 2, 2, 4), "backbone_channels": 16, "aspp_channels": 16},
        )
        x = torch.randn(2, 50)
        y = model_custom(x)
        self.assertEqual(y.shape, (2, 50))

    def test_aspp_branches_and_rates(self) -> None:
        """Verify ASPP module produces correct output shape and respects custom dilation rates."""
        aspp = ASPP1D(
            in_channels=16,
            out_channels=16,
            aspp_rates=(3, 6, 9),
            kernel_size=3,
            n_points=60,
        )
        self.assertEqual(len(aspp.atrous_branches), 3)

        feat_in = torch.randn(2, 16, 60)
        feat_out = aspp(feat_in)
        self.assertEqual(feat_out.shape, (2, 16, 60))

    def test_global_pooling_safeguard(self) -> None:
        """Verify global average pooling branch works even when patch is small."""
        # When sequence length is small, dilation might approach window size
        model = DeepLabV3(
            n_points=12,
            config={
                "backbone_channels": 8,
                "aspp_channels": 8,
                "kernel_size": 3,
                "multi_grid": (1, 2),
                "aspp_rates": (2, 4, 6),
            },
        )
        x = torch.randn(2, 12)
        y = model(x)
        self.assertEqual(y.shape, (2, 12))
        self.assertFalse(torch.isnan(y).any())

    def test_residual_mode_toggle(self) -> None:
        """Verify residual mode and direct prediction mode both produce valid outputs."""
        model_res = DeepLabV3(n_points=40, config={"residual": True, "backbone_channels": 8, "aspp_channels": 8})
        model_dir = DeepLabV3(n_points=40, config={"residual": False, "backbone_channels": 8, "aspp_channels": 8})

        x = torch.randn(2, 40)
        y_res = model_res(x)
        y_dir = model_dir(x)

        self.assertEqual(y_res.shape, (2, 40))
        self.assertEqual(y_dir.shape, (2, 40))

    def test_l2_regularization(self) -> None:
        """Verify L2 regularization calculation only penalizes conv weight matrices."""
        model = DeepLabV3(n_points=30, config={"backbone_channels": 8, "aspp_channels": 8})
        l2_reg = model.get_l2_regularization()

        self.assertTrue(torch.is_tensor(l2_reg))
        self.assertGreater(l2_reg.item(), 0.0)

        # Verify exact sum of Conv1d weights and total exclusion of BatchNorm1d
        expected_conv_l2 = sum(
            torch.sum(m.weight ** 2)
            for m in model.modules()
            if isinstance(m, torch.nn.Conv1d) and m.weight.requires_grad
        )
        self.assertAlmostEqual(l2_reg.item(), expected_conv_l2.item(), places=5)

        # Verify no BatchNorm parameter contributes
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm1d) and m.weight is not None:
                # If BatchNorm were included, diff would be >= sum(m.weight**2)
                self.assertNotAlmostEqual(
                    l2_reg.item(),
                    (expected_conv_l2 + torch.sum(m.weight ** 2)).item(),
                )

    def test_invalid_tensor_dimensions_raise(self) -> None:
        """Verify model raises ValueError on invalid dimensions or channel counts."""
        model = DeepLabV3(n_points=30, config={"backbone_channels": 8, "aspp_channels": 8})

        # 1D tensor
        with self.assertRaises(ValueError):
            model(torch.randn(30))

        # 4D tensor
        with self.assertRaises(ValueError):
            model(torch.randn(2, 1, 1, 30))

        # 3D tensor with 2 channels (expects 1 channel)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 2, 30))

    def test_extremely_short_sequence_padding_safety(self) -> None:
        """Verify safe padding handles sequence lengths smaller than dilation pad without crashing."""
        # For kernel_size=3 and dilation=6, pad=6. A sequence of length 3 would crash standard ReflectionPad1d.
        model = DeepLabV3(
            n_points=3,
            config={
                "backbone_channels": 8,
                "aspp_channels": 8,
                "aspp_rates": (2, 4, 6),
                "multi_grid": (1, 2),
            },
        )
        x = torch.randn(2, 3)
        y = model(x)
        self.assertEqual(y.shape, (2, 3))
        self.assertFalse(torch.isnan(y).any())

    def test_extreme_multi_grid_configurations(self) -> None:
        """Verify single block and multi-block multi_grid tuples execute properly."""
        # Single block
        model_single = DeepLabV3(
            n_points=30,
            config={"multi_grid": (1,), "backbone_channels": 8, "aspp_channels": 8},
        )
        self.assertEqual(len(model_single.backbone.blocks), 1)
        self.assertEqual(model_single(torch.randn(2, 30)).shape, (2, 30))

        # 5 blocks
        model_extended = DeepLabV3(
            n_points=30,
            config={"multi_grid": (1, 2, 2, 4, 8), "backbone_channels": 8, "aspp_channels": 8},
        )
        self.assertEqual(len(model_extended.backbone.blocks), 5)
        self.assertEqual(model_extended(torch.randn(2, 30)).shape, (2, 30))

    def test_backward_gradient_flow(self) -> None:
        """Verify backward gradients propagate through all components of DeepLabV3."""
        model = DeepLabV3(n_points=50, config={"backbone_channels": 16, "aspp_channels": 16})
        x = torch.randn(2, 50, requires_grad=True)
        y_pred = model(x)
        loss = torch.mean((y_pred - torch.ones_like(y_pred)) ** 2) + 1e-4 * model.get_l2_regularization()
        loss.backward()

        self.assertIsNotNone(x.grad)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for parameter {name}")

    def test_instantiate_model_factory(self) -> None:
        """Verify instantiate_model factory correctly resolves 'deeplabv3'."""
        model = instantiate_model("deeplabv3", seq_len=80, config={"backbone_channels": 16})
        self.assertIsInstance(model, DeepLabV3)
        self.assertEqual(model.n_points, 80)

        # Alias test
        alias_model = instantiate_model("deeplab", seq_len=40)
        self.assertIsInstance(alias_model, DeepLabV3)


class TestDeepLabV3PipelineIntegration(unittest.TestCase):
    """End-to-end integration tests for DeepLabV3 training and prediction."""

    @classmethod
    def setUpClass(cls) -> None:
        pseudo_cfg = {
            "material": "NMG",
            "regions": ["Al2p", "Ti2p"],
            "n_points": 40,
            "measurements_per_tool": {"J4": 4, "H1": 4},
            "measurements_per_t7_code": 2,
            "interval_hours_range": (1.0, 5.0),
            "tool_offsets": {
                "J4": {"shift_ev": 0.0, "scale": 10000.0},
                "H1": {"shift_ev": 0.5, "scale": 11000.0},
            },
            "die_variation_std": 0.01,
            "noise_relative_std": 0.01,
            "seed": 42,
        }
        cls.ary_intensity, cls.ary_energy, cls.meta_df = generate_pseudo_measurements(pseudo_cfg)
        cls.paired_meta = pair_source_target_spectra(
            cls.meta_df,
            config={
                "source_tool": "J4",
                "target_tool": "H1",
                "time_threshold_hours": 12.0,
            },
        )

    def test_training_one_epoch_normalized_loss(self) -> None:
        """Verify DeepLabV3 completes a training step with NormalizedMSELoss."""
        x_synth = np.random.uniform(10.0, 100.0, size=(10, 40))
        y_synth = x_synth + np.random.normal(0.0, 1.0, size=(10, 40))
        dataset = SpectrumPairDataset(x_synth, y_synth)

        train_res = train_model_region(
            train_dataset=dataset,
            val_dataset=dataset,
            config={
                "model_type": "deeplabv3",
                "epochs": 2,
                "batch_size": 4,
                "learning_rate": 1e-3,
                "backbone_channels": 8,
                "aspp_channels": 8,
                "kernel_size": 3,
                "verbose": False,
            },
        )
        self.assertIn("best_val_loss", train_res)
        self.assertLess(train_res["best_val_loss"], float("inf"))
        self.assertIsInstance(train_res["model"], DeepLabV3)

    def test_run_deeplabv3_pipeline_full_spectrum(self) -> None:
        """Verify run_deeplabv3_pipeline runs successfully without sliding window."""
        pipeline_cfg = {
            "model_type": "deeplabv3",
            "use_sliding_window": False,
            "regions": ["Al2p"],
            "source_tool": "J4",
            "target_tool": "H1",
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "seed": 42,
            "train_config": {
                "epochs": 2,
                "batch_size": 4,
                "learning_rate": 1e-3,
                "backbone_channels": 8,
                "aspp_channels": 8,
                "kernel_size": 3,
                "verbose": False,
            },
            "predict_source": True,
        }

        results = run_deeplabv3_pipeline(
            meta_df=self.paired_meta,
            ary_intensity=self.ary_intensity,
            ary_energy=self.ary_energy,
            config=pipeline_cfg,
        )

        self.assertIn("models", results)
        self.assertIn("Al2p", results["models"])
        self.assertIsInstance(results["models"]["Al2p"], DeepLabV3)
        self.assertIn("predictions", results)
        pred_int, pred_ene, pred_meta = results["predictions"]
        self.assertEqual(pred_int.shape[1], 40)
        self.assertEqual(pred_ene.shape[1], 40)
        self.assertTrue(len(pred_meta) > 0)

    def test_run_deeplabv3_pipeline_sliding_window(self) -> None:
        """Verify run_deeplabv3_pipeline runs successfully with sliding window patch mode."""
        pipeline_cfg = {
            "model_type": "deeplabv3",
            "use_sliding_window": True,
            "regions": ["Al2p"],
            "source_tool": "J4",
            "target_tool": "H1",
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "seed": 42,
            "window_size_points": 15,
            "sliding_stride_points": 5,
            "train_config": {
                "epochs": 2,
                "batch_size": 4,
                "learning_rate": 1e-3,
                "backbone_channels": 8,
                "aspp_channels": 8,
                "kernel_size": 3,
                "verbose": False,
            },
            "predict_source": True,
        }

        results = run_deeplabv3_pipeline(
            meta_df=self.paired_meta,
            ary_intensity=self.ary_intensity,
            ary_energy=self.ary_energy,
            config=pipeline_cfg,
        )

        self.assertIn("models", results)
        self.assertIn("Al2p", results["models"])
        model_tuple = results["models"]["Al2p"]
        self.assertIsInstance(model_tuple, tuple)
        self.assertIsInstance(model_tuple[0], DeepLabV3)
        self.assertIn("predictions", results)
        pred_int, pred_ene, pred_meta = results["predictions"]
        self.assertEqual(pred_int.shape[1], 40)
        self.assertTrue(len(pred_meta) > 0)

    def test_bayesian_optimization_deeplabv3(self) -> None:
        """Verify Ax Bayesian optimization searches DeepLabV3 hyperparameters."""
        x_synth = np.random.uniform(10.0, 100.0, size=(10, 40))
        y_synth = x_synth + np.random.normal(0.0, 1.0, size=(10, 40))
        dataset = SpectrumPairDataset(x_synth, y_synth)

        ax_res = optimize_model_hyperparameters(
            dataset,
            dataset,
            config={
                "model_type": "deeplabv3",
                "num_trials": 2,
                "epochs_per_trial": 2,
                "kernel_sizes": [3, 5],
                "backbone_channels": [8, 16],
                "aspp_channels": [8, 16],
                "batch_sizes": [4],
                "seed": 42,
            },
        )

        self.assertIn("best_parameters", ax_res)
        self.assertIn("best_val_loss", ax_res)
        self.assertEqual(len(ax_res["trials_data"]), 2)
        self.assertIn(ax_res["best_parameters"]["kernel_size"], [3, 5])
        self.assertIn(ax_res["best_parameters"]["backbone_channels"], [8, 16])
        self.assertIn(ax_res["best_parameters"]["aspp_channels"], [8, 16])


if __name__ == "__main__":
    unittest.main()

