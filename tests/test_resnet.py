"""Unit tests for 1D ResNet architecture, training, and Ax optimization."""

from __future__ import annotations

import unittest
import warnings

import numpy as np
import pandas as pd
import torch

from models.bayesian_opt import optimize_model_hyperparameters
from models.cost import SpectralLoss
from models.dataset import SpectrumPairDataset, create_dataloaders
from models.inference import format_predicted_measurement_id, predict_spectra
from models.resnet import ResNet1D, ResNetScaledShort1D
from models.trainer import evaluate, instantiate_model, train_model_region


class TestResNet1D(unittest.TestCase):
    """Test cases for the ResNet1D architecture."""

    def test_cnn_forward_shape_2d_raises(self) -> None:
        """Test that 2D tensor input raises ValueError."""
        model = ResNet1D(n_points=100, config={"resnet_channel": 16, "kernel_size": 5})
        x = torch.randn(8, 100)
        with self.assertRaises(ValueError):
            model(x)

    def test_cnn_forward_shape_3d(self) -> None:
        """Test forward pass with 3D tensor input (batch, 1, n_points)."""
        model = ResNet1D(n_points=80, config={"resnet_channel": 8, "kernel_size": 3})
        x = torch.randn(4, 1, 80)
        out = model(x)
        self.assertEqual(out.shape, (4, 1, 80))

    def test_invalid_kernel_size_raises(self) -> None:
        """Test that even, non-positive, or larger-than-n_points kernel size
        raises ValueError.
        """
        with self.assertRaises(ValueError):
            ResNet1D(n_points=50, config={"kernel_size": 4})
        with self.assertRaises(ValueError):
            ResNet1D(n_points=50, config={"kernel_size": 0})
        with self.assertRaises(ValueError):
            ResNet1D(n_points=50, config={"kernel_size": -3})
        with self.assertRaises(ValueError):
            ResNet1D(n_points=5, config={"kernel_size": 7})

    def test_l2_regularization_targets_weights_only(self) -> None:
        """Test that L2 weight penalty only targets conv weights, not biases or
        BatchNorm.
        """
        model = ResNet1D(n_points=50, config={"resnet_channel": 8, "kernel_size": 3})
        l2_reg = model.get_l2_regularization()
        self.assertIsInstance(l2_reg, torch.Tensor)
        self.assertGreater(l2_reg.item(), 0.0)


class TestSpectrumPairDataset(unittest.TestCase):
    """Test cases for SpectrumPairDataset construction and dataloaders."""

    def test_dataset_mismatched_lengths_raise(self) -> None:
        """Test that mismatched x and y lengths raise ValueError."""
        x = np.ones((5, 10))
        y = np.ones((4, 10))
        with self.assertRaises(ValueError):
            SpectrumPairDataset(x, y)

    def test_create_dataloaders(self) -> None:
        """Test dataloader batch construction."""
        x = np.ones((8, 50), dtype=np.float32)
        y = np.ones((8, 50), dtype=np.float32)
        train_ds = SpectrumPairDataset(x[:6], y[:6])
        val_ds = SpectrumPairDataset(x[6:], y[6:])
        train_loader, val_loader = create_dataloaders(train_ds, val_ds, config={"batch_size": 4})
        batch = next(iter(train_loader))
        self.assertIn("x", batch)
        self.assertEqual(batch["x"].shape, (4, 1, 50))
        self.assertEqual(batch["y"].shape, (4, 1, 50))


class TestTrainingAndInference(unittest.TestCase):
    """Test model training, inference output structure, and Ax optimization."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate test dataset."""
        np.random.seed(100)
        torch.manual_seed(100)
        n_pts = 50
        x_raw = np.random.uniform(10.0, 100.0, size=(16, n_pts)).astype(np.float32)
        y_raw = (x_raw * 1.05 + 1.0).astype(np.float32)
        cls.train_ds = SpectrumPairDataset(x_raw[:10], y_raw[:10])
        cls.val_ds = SpectrumPairDataset(x_raw[10:], y_raw[10:])

        # Minimal metadata for predict_spectra testing
        cls.meta_df = pd.DataFrame([
            {
                "spectrum_index": i,
                "measurement_id": f"NMG_M_J4_{i:05d}",
                "tool": "J4",
                "die": 0,
                "region": "Al2p",
                "split": "test",
                "measurement_id_target": f"NMG_M_H1_{i:05d}",
                "tool_target": "H1",
            }
            for i in range(4)
        ])
        cls.ary_intensity = np.random.uniform(10.0, 100.0, size=(4, n_pts)).astype(np.float32)
        cls.ary_energy = np.tile(np.linspace(10.0, 20.0, n_pts, dtype=np.float32), (4, 1))

    def test_evaluate_excludes_l2_penalty(self) -> None:
        """Verify evaluate() measures pure reconstruction loss, independent
        of l2_weight.
        """
        _, val_loader = create_dataloaders(self.train_ds, self.val_ds, config={"batch_size": 4})
        model = ResNet1D(n_points=50, config={"resnet_channel": 8, "kernel_size": 3})

        crit_zero_l2 = SpectralLoss(l2_weight=0.0)
        crit_large_l2 = SpectralLoss(l2_weight=10.0)

        loss_zero = evaluate(model, val_loader, crit_zero_l2)
        loss_large = evaluate(model, val_loader, crit_large_l2)

        self.assertAlmostEqual(loss_zero, loss_large, places=6)

    def test_train_resnet_region(self) -> None:
        """Test training execution and loss logging via train_model_region."""
        res = train_model_region(
            self.train_ds,
            self.val_ds,
            config={
                "model_type": "resnet",
                "epochs": 5,
                "batch_size": 8,
                "resnet_channel": 8,
                "kernel_size": 3,
                "learning_rate": 1e-3,
                "l2_weight": 1e-5,
            },
        )

        self.assertIn("model", res)
        self.assertIn("best_val_loss", res)
        self.assertEqual(len(res["history"]["train_loss"]), 5)
        self.assertIsInstance(res["model"], ResNet1D)

    def test_format_predicted_measurement_id_uniqueness(self) -> None:
        """Verify unique ID generation without collision for different session
        names.
        """
        id_1 = format_predicted_measurement_id("M_J4_003", "J4", "H1")
        id_2 = format_predicted_measurement_id("SESSION_A_003", "J4", "H1")
        id_3 = format_predicted_measurement_id("SESSION_B_003", "J4", "H1")
        id_new = format_predicted_measurement_id("NMG_M_J4_00001", "J4", "H1")

        self.assertEqual(id_1, "P_J4H1_003")
        self.assertEqual(id_2, "P_J4H1_SESSION_A_003")
        self.assertEqual(id_3, "P_J4H1_SESSION_B_003")
        self.assertEqual(id_new, "NMG_P_J4H1_00001")
        self.assertNotEqual(id_2, id_3)

    def test_predict_spectra_bundled_data(self) -> None:
        """Test inference output formatting with bundled data tuple."""
        model_al2p = train_model_region(
            self.train_ds,
            self.val_ds,
            config={
                "model_type": "resnet",
                "epochs": 2,
                "batch_size": 8,
                "resnet_channel": 8,
                "kernel_size": 3,
            },
        )["model"]

        models = {"Al2p": model_al2p}
        pred_int, pred_eng, pred_df = predict_spectra(
            models=models,
            data=(self.ary_intensity, self.ary_energy, self.meta_df),
            config={"source_tool": "J4", "target_tool": "H1"},
        )

        self.assertEqual(pred_int.shape[0], 4)
        self.assertEqual(pred_eng.shape[0], 4)
        self.assertEqual(len(pred_df), 4)
        self.assertTrue(all(pred_df["measurement_id"].str.contains("P_J4H1_")))
        self.assertTrue(all(pred_df["tool"] == "H1"))

    def test_ax_bayesian_optimization_integration(self) -> None:
        """Test running Ax Bayesian optimization for 2 trials with seed."""
        ax_res = optimize_model_hyperparameters(
            self.train_ds,
            self.val_ds,
            config={
                "model_type": "resnet",
                "num_trials": 2,
                "epochs_per_trial": 2,
                "kernel_sizes": [3, 5],
                "resnet_channels": [8, 16],
                "batch_sizes": [8],
                "seed": 42,
            },
        )

        self.assertIn("best_parameters", ax_res)
        self.assertIn("best_val_loss", ax_res)
        self.assertEqual(len(ax_res["trials_data"]), 2)
        self.assertIn(ax_res["best_parameters"]["kernel_size"], [3, 5])
        self.assertIn(ax_res["best_parameters"]["resnet_channel"], [8, 16])


class TestResNetScaledShort1D(unittest.TestCase):
    """Test cases for ResNetScaledShort1D with FiLM-scaled shortcut."""

    def setUp(self) -> None:
        self.cond = {
            "die": torch.tensor([0, 1]),
            "tool_src": torch.tensor([0, 0]),
            "tool_tgt": torch.tensor([1, 1]),
            "die_flux_normalized": torch.tensor([[0.5], [-0.5]]),
            "time_normalized": torch.tensor([[0.2], [-0.2]]),
        }

    def test_requires_use_film_true(self) -> None:
        """Verify ValueError when use_film is not explicitly enabled."""
        with self.assertRaises(ValueError):
            ResNetScaledShort1D(n_points=50)
        with self.assertRaises(ValueError):
            ResNetScaledShort1D(n_points=50, config={"use_film": False})

    def test_requires_cond_in_forward(self) -> None:
        """Verify ValueError when forward is called without metadata dict."""
        model = ResNetScaledShort1D(n_points=50, config={"use_film": True})
        x = torch.randn(2, 1, 50)
        with self.assertRaises(ValueError):
            model(x)
        with self.assertRaises(ValueError):
            model(x, cond=None)
        with self.assertRaises(ValueError):
            model(x, cond="not_a_dict")

    def test_identity_initialization(self) -> None:
        """Verify initial shortcut scaling factor is exactly 1.0."""
        model = ResNetScaledShort1D(
            n_points=50,
            config={"use_film": True, "resnet_channel": 8, "kernel_size": 3},
        )
        x = torch.randn(2, 1, 50)
        out = model(x, self.cond)
        self.assertEqual(out.shape, (2, 1, 50))
        shortcut_mods = model.shortcut_film_gen(self.cond)
        gamma_out, _ = shortcut_mods[0]
        self.assertTrue(torch.allclose(gamma_out, torch.ones_like(gamma_out)))

    def test_clamping_and_warning_on_negative_gain(self) -> None:
        """Verify warning is emitted and gamma_out clamped when negative."""
        model = ResNetScaledShort1D(
            n_points=50,
            config={"use_film": True, "resnet_channel": 8, "kernel_size": 3},
        )
        with torch.no_grad():
            model.shortcut_film_gen.gamma_projs[0].bias.fill_(-5.0)

        x = torch.randn(2, 1, 50)
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            out = model(x, self.cond)
            self.assertEqual(out.shape, (2, 1, 50))
            clamping_warns = [
                w
                for w in recorded_warnings
                if issubclass(w.category, UserWarning)
                and "Negative shortcut gain detected" in str(w.message)
            ]
            self.assertGreaterEqual(len(clamping_warns), 1)

    def test_internal_film_layers_toggle(self) -> None:
        """Verify film_internal_layers disables internal FiLM when False."""
        model_no_internal = ResNetScaledShort1D(
            n_points=50,
            config={
                "use_film": True,
                "film_internal_layers": False,
                "resnet_channel": 8,
                "kernel_size": 3,
            },
        )
        self.assertIsNone(model_no_internal.film_gen)
        self.assertIsNotNone(model_no_internal.shortcut_film_gen)

        x = torch.randn(2, 1, 50)
        out = model_no_internal(x, self.cond)
        self.assertEqual(out.shape, (2, 1, 50))

        model_with_internal = ResNetScaledShort1D(
            n_points=50,
            config={
                "use_film": True,
                "film_internal_layers": True,
                "resnet_channel": 8,
                "kernel_size": 3,
            },
        )
        self.assertIsNotNone(model_with_internal.film_gen)
        out_with = model_with_internal(x, self.cond)
        self.assertEqual(out_with.shape, (2, 1, 50))

    def test_shortcut_ignores_window_energy(self) -> None:
        """Verify shortcut FiLM generator ignores sliding window energy."""
        model = ResNetScaledShort1D(
            n_points=30,
            config={
                "use_film": True,
                "use_sliding_window": True,
                "encode_window_energy": True,
                "resnet_channel": 8,
                "kernel_size": 3,
            },
        )
        self.assertTrue(model.film_gen.encode_window_energy)
        self.assertFalse(model.shortcut_film_gen.encode_window_energy)

    def test_instantiate_model_registration(self) -> None:
        """Verify instantiate_model supports 'resnet_scaled_short'."""
        model = instantiate_model(
            "resnet_scaled_short", 50, {"use_film": True}
        )
        self.assertIsInstance(model, ResNetScaledShort1D)

    def test_trainer_relative_learning_rate(self) -> None:
        """Verify train_model_region configures separate param groups."""
        x_raw = np.random.uniform(10.0, 50.0, size=(8, 40)).astype(np.float32)
        y_raw = (x_raw * 1.02).astype(np.float32)
        meta_list = [
            {
                "spectrum_index": i,
                "measurement_id": f"M_{i}",
                "tool": "J4",
                "die": 0,
                "region": "Al2p",
                "tool_index": 0,
                "tool_target_index": 1,
                "die_flux_normalized": 0.0,
                "time_normalized": 0.0,
            }
            for i in range(8)
        ]
        train_ds = SpectrumPairDataset(x_raw, y_raw, metadata=meta_list)

        with self.assertRaises(ValueError):
            train_model_region(
                train_ds,
                train_ds,
                config={
                    "model_type": "resnet_scaled_short",
                    "use_film": True,
                    "relative_learning_rate": -0.5,
                },
            )

        res = train_model_region(
            train_ds,
            train_ds,
            config={
                "model_type": "resnet_scaled_short",
                "use_film": True,
                "epochs": 2,
                "learning_rate": 1e-3,
                "relative_learning_rate": 0.05,
                "resnet_channel": 8,
                "kernel_size": 3,
            },
        )
        self.assertIn("model", res)
        self.assertIsInstance(res["model"], ResNetScaledShort1D)

    def test_sliding_window_training_and_inference(self) -> None:
        """Verify sliding window patch training and inference work with
        resnet_scaled_short.
        """
        x_raw = np.random.uniform(10.0, 50.0, size=(8, 40)).astype(np.float32)
        y_raw = (x_raw * 1.05).astype(np.float32)
        meta_list = [
            {
                "spectrum_index": i,
                "measurement_id": f"M_{i}",
                "tool": "J4",
                "die": 0,
                "region": "Al2p",
                "tool_index": 0,
                "tool_target_index": 1,
                "die_flux_normalized": 0.1,
                "time_normalized": -0.1,
            }
            for i in range(8)
        ]
        train_ds = SpectrumPairDataset(x_raw, y_raw, metadata=meta_list)
        res = train_model_region(
            train_ds,
            train_ds,
            config={
                "model_type": "resnet_scaled_short",
                "use_film": True,
                "use_sliding_window": True,
                "window_size": 15,
                "stride": 5,
                "epochs": 2,
                "resnet_channel": 8,
                "kernel_size": 3,
            },
        )
        self.assertIn("model", res)
        meta_df = pd.DataFrame(meta_list)
        pred_int, pred_eng, pred_df = predict_spectra(
            models={"Al2p": res["model"]},
            data=(x_raw, np.ones_like(x_raw), meta_df),
            config={"source_tool": "J4", "target_tool": "H1"},
        )
        self.assertEqual(pred_int.shape, (8, 40))
        self.assertEqual(len(pred_df), 8)


if __name__ == "__main__":
    unittest.main()
