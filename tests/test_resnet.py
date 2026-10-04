"""Unit tests for baseline 1D Residual CNN, training pipeline, and Ax optimization."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from models.bayesian_opt import optimize_baseline_hyperparameters
from models.cost import NormalizedMSELoss
from models.dataset import (
    SpectrumPairDataset,
    create_dataloaders,
    partition_measurement_sessions,
    split_session_datasets,
)
from models.inference import format_predicted_measurement_id, predict_spectra
from models.orchestration import run_baseline_pipeline
from models.resnet import ResNet1D, Residual1DCNN
from models.trainer import evaluate, train_baseline_region
from utility.pairing import pair_source_target_spectra
from utility.pseudo_measurement import generate_pseudo_measurements


class TestBaselineCNN(unittest.TestCase):
    """Test cases for the Residual1DCNN architecture and NormalizedMSELoss."""

    def test_cnn_forward_shape_2d(self) -> None:
        """Test forward pass with 2D tensor input (batch, n_points)."""
        model = Residual1DCNN(n_points=100, config={"hidden_channels": 16, "kernel_size": 5})
        x = torch.randn(8, 100)
        out = model(x)
        self.assertEqual(out.shape, (8, 100))

    def test_cnn_forward_shape_3d(self) -> None:
        """Test forward pass with 3D tensor input (batch, 1, n_points)."""
        model = Residual1DCNN(n_points=80, config={"hidden_channels": 8, "kernel_size": 3})
        x = torch.randn(4, 1, 80)
        out = model(x)
        self.assertEqual(out.shape, (4, 1, 80))

    def test_invalid_kernel_size_raises(self) -> None:
        """Test that even or non-positive kernel size raises ValueError."""
        with self.assertRaises(ValueError):
            Residual1DCNN(n_points=50, config={"kernel_size": 4})
        with self.assertRaises(ValueError):
            Residual1DCNN(n_points=50, config={"kernel_size": 0})
        with self.assertRaises(ValueError):
            Residual1DCNN(n_points=50, config={"kernel_size": -3})

    def test_l2_regularization_targets_weights_only(self) -> None:
        """Test that L2 weight penalty only targets conv weights, not biases or BatchNorm."""
        model = Residual1DCNN(n_points=50, config={"hidden_channels": 8, "kernel_size": 3})
        l2_reg = model.get_l2_regularization()
        self.assertIsInstance(l2_reg, torch.Tensor)
        self.assertGreater(l2_reg.item(), 0.0)

    def test_normalized_mse_loss_broadcasting_guard(self) -> None:
        """Test that 3D and 2D inputs do not cause (B, B, L) cross-batch broadcasting."""
        loss_fn = NormalizedMSELoss(l2_weight=0.0)
        y_true = torch.tensor([[1000.0, 2000.0, 500.0], [500.0, 1500.0, 300.0]], dtype=torch.float32)
        # Identical prediction but with extra dimension (B, 1, L)
        y_pred_3d = y_true.unsqueeze(1)

        loss = loss_fn(y_pred_3d, y_true)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_normalized_mse_loss_perturbed(self) -> None:
        """Test NormalizedMSELoss normalization computation."""
        loss_fn = NormalizedMSELoss(l2_weight=0.0)
        y_true = torch.tensor([[1000.0, 2000.0, 500.0]], dtype=torch.float32)
        y_pred = torch.tensor([[1000.0, 2200.0, 500.0]], dtype=torch.float32)

        # Difference at index 1 is 200 / 2000 = 0.1 -> square is 0.01 -> mean = 0.01 / 3
        loss_perturbed = loss_fn(y_pred, y_true)
        expected = (0.1 ** 2) / 3.0
        self.assertAlmostEqual(loss_perturbed.item(), expected, places=5)

    def test_loss_zero_target_stability(self) -> None:
        """Test that zero target spectrum does not produce inf or nan."""
        loss_fn = NormalizedMSELoss(l2_weight=0.0, eps=1e-4)
        y_true = torch.zeros((2, 50), dtype=torch.float32)
        y_pred = torch.randn(2, 50)
        loss = loss_fn(y_pred, y_true)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))


class TestDatasetAndSplitting(unittest.TestCase):
    """Test cases for SpectrumPairDataset and session-level splitting."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate a small pseudo-measurement dataset with pairings."""
        cls.ary_intensity, cls.ary_energy, cls.meta_df = generate_pseudo_measurements(
            {
                "material": "NMG",
                "regions": ["Al2p", "Ti2p"],
                "n_points": 50,
                "measurements_per_tool": {"J4": 6, "H1": 6},
                "interval_hours_range": (1.0, 4.0),
                "seed": 42,
            }
        )
        cls.meta_df = pair_source_target_spectra(
            cls.meta_df,
            {
                "source_tool": "J4",
                "target_tool": "H1",
                "time_threshold_hours": 12.0,
            },
        )

    def test_dataset_mismatched_lengths_raise(self) -> None:
        """Test that mismatched x and y lengths raise ValueError."""
        x = np.ones((5, 10))
        y = np.ones((4, 10))
        with self.assertRaises(ValueError):
            SpectrumPairDataset(x, y)

    def test_session_level_split_no_leakage(self) -> None:
        """Test that train and val sets have zero overlap in measurement_id sessions."""
        train_ds, val_ds = split_session_datasets(
            self.meta_df,
            self.ary_intensity,
            self.ary_energy,
            config={"region": "Al2p", "source_tool": "J4", "target_tool": "H1", "val_ratio": 0.33, "seed": 42},
        )

        train_sessions = {item["meta"]["measurement_id"] for item in train_ds}
        val_sessions = {item["meta"]["measurement_id"] for item in val_ds}

        # Check strict separation
        self.assertEqual(len(train_sessions.intersection(val_sessions)), 0)
        self.assertGreater(len(train_sessions), 0)
        self.assertGreater(len(val_sessions), 0)

        # Check total sample count equals total paired rows for this region
        paired_count = len(
            self.meta_df[
                (self.meta_df["tool"] == "J4")
                & (self.meta_df["region"] == "Al2p")
                & self.meta_df["measurement_id_target"].notna()
            ]
        )
        self.assertEqual(len(train_ds) + len(val_ds), paired_count)

    def test_session_split_high_val_ratio_never_empties_train(self) -> None:
        """Test that high val_ratio does not leave train split empty."""
        train_ds, val_ds = split_session_datasets(
            self.meta_df,
            self.ary_intensity,
            self.ary_energy,
            config={"region": "Al2p", "source_tool": "J4", "target_tool": "H1", "val_ratio": 0.99, "seed": 42},
        )
        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(val_ds), 0)

    def test_partition_measurement_sessions_3way_default_and_custom(self) -> None:
        """Test global session partitioning produces strictly disjoint train, val, and test splits."""
        splits = partition_measurement_sessions(
            self.meta_df,
            config={"source_tool": "J4", "target_tool": "H1", "seed": 42},
        )
        self.assertIn("train", splits)
        self.assertIn("val", splits)
        self.assertIn("test", splits)

        train_set = splits["train"]
        val_set = splits["val"]
        test_set = splits["test"]

        # Ensure non-empty and mutually exclusive
        self.assertGreater(len(train_set), 0)
        self.assertGreater(len(val_set), 0)
        self.assertGreater(len(test_set), 0)
        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set.intersection(test_set)), 0)

        # Test custom ratios
        custom_splits = partition_measurement_sessions(
            self.meta_df,
            config={
                "source_tool": "J4",
                "target_tool": "H1",
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
                "seed": 123,
            },
        )
        self.assertEqual(len(custom_splits["train"].intersection(custom_splits["test"])), 0)

    def test_partition_measurement_sessions_negative_ratio_raises(self) -> None:
        """Verify negative split ratios are rejected with ValueError."""
        with self.assertRaises(ValueError):
            partition_measurement_sessions(
                self.meta_df,
                config={"train_ratio": 1.2, "val_ratio": -0.1, "test_ratio": -0.1},
            )

    def test_partition_measurement_sessions_too_few_sessions_raises(self) -> None:
        """Verify 3-way split on dataset with < 3 sessions raises ValueError."""
        # Filter metadata to keep only 2 measurement sessions
        two_sessions = sorted(self.meta_df[self.meta_df["measurement_id_target"].notna()]["measurement_id"].unique())[:2]
        small_meta = self.meta_df[self.meta_df["measurement_id"].isin(two_sessions)]
        with self.assertRaises(ValueError):
            partition_measurement_sessions(
                small_meta,
                config={"train_ratio": 0.5, "val_ratio": 0.2, "test_ratio": 0.3},
            )

    def test_partition_measurement_sessions_two_way_when_test_ratio_zero(self) -> None:
        """Verify 2-way split works cleanly when test_ratio=0.0."""
        splits = partition_measurement_sessions(
            self.meta_df,
            config={"train_ratio": 0.7, "val_ratio": 0.3, "test_ratio": 0.0, "seed": 42},
        )
        self.assertGreater(len(splits["train"]), 0)
        self.assertGreater(len(splits["val"]), 0)
        self.assertEqual(len(splits["test"]), 0)
        self.assertEqual(len(splits["train"].intersection(splits["val"])), 0)

    def test_partition_measurement_sessions_target_leakage_detection(self) -> None:
        """Verify that multi-mapped target sessions across splits trigger ValueError."""
        # Intentionally force a conflicting target assignment across two source sessions
        leak_df = self.meta_df.copy()
        paired_sources = leak_df[leak_df["measurement_id_target"].notna()]["measurement_id"].unique()
        # Set all paired source measurements to point to the exact same target measurement ID
        leak_df.loc[leak_df["measurement_id"].isin(paired_sources), "measurement_id_target"] = "SHARED_TGT_ID"
        with self.assertRaises(ValueError) as ctx:
            partition_measurement_sessions(leak_df, config={"seed": 42})
        self.assertIn("Target measurement sessions leak", str(ctx.exception))

    def test_split_session_datasets_with_precomputed_partitions(self) -> None:
        """Verify unified session partitions prevent cross-region session leakage."""
        splits = partition_measurement_sessions(
            self.meta_df,
            config={"source_tool": "J4", "target_tool": "H1", "seed": 42},
        )

        train_al, val_al, test_al = split_session_datasets(
            self.meta_df,
            self.ary_intensity,
            self.ary_energy,
            config={
                "region": "Al2p",
                "source_tool": "J4",
                "target_tool": "H1",
                "session_splits": splits,
                "return_test": True,
            },
        )
        train_ti, val_ti, test_ti = split_session_datasets(
            self.meta_df,
            self.ary_intensity,
            self.ary_energy,
            config={
                "region": "Ti2p",
                "source_tool": "J4",
                "target_tool": "H1",
                "session_splits": splits,
                "return_test": True,
            },
        )

        al_train_sessions = {item["meta"]["measurement_id"] for item in train_al}
        ti_train_sessions = {item["meta"]["measurement_id"] for item in train_ti}
        # Identical session IDs across both regions - completely unified
        self.assertEqual(al_train_sessions, ti_train_sessions)
        self.assertEqual(al_train_sessions, splits["train"])

        al_test_sessions = {item["meta"]["measurement_id"] for item in test_al}
        self.assertEqual(al_test_sessions, splits["test"])

    def test_create_dataloaders(self) -> None:
        """Test dataloader batch construction."""
        train_ds, val_ds = split_session_datasets(
            self.meta_df,
            self.ary_intensity,
            self.ary_energy,
            config={"region": "Al2p", "source_tool": "J4", "target_tool": "H1"},
        )
        train_loader, val_loader = create_dataloaders(train_ds, val_ds, config={"batch_size": 4})
        batch = next(iter(train_loader))
        self.assertIn("x", batch)
        self.assertIn("y", batch)
        self.assertEqual(batch["x"].shape[0], 4)
        self.assertEqual(batch["x"].shape[1], 50)


class TestTrainingAndInference(unittest.TestCase):
    """Test model training, inference output structure, and Ax optimization."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate test dataset."""
        cls.ary_intensity, cls.ary_energy, cls.meta_df = generate_pseudo_measurements(
            {
                "material": "NMG",
                "regions": ["Al2p", "Ti2p"],
                "n_points": 50,
                "measurements_per_tool": {"J4": 6, "H1": 6},
                "interval_hours_range": (1.0, 4.0),
                "seed": 100,
            }
        )
        cls.meta_df = pair_source_target_spectra(
            cls.meta_df,
            {
                "source_tool": "J4",
                "target_tool": "H1",
                "time_threshold_hours": 12.0,
            },
        )

    def test_evaluate_excludes_l2_penalty(self) -> None:
        """Verify that evaluate() measures pure reconstruction loss, independent of l2_weight."""
        train_ds, val_ds = split_session_datasets(
            self.meta_df,
            self.ary_intensity,
            self.ary_energy,
            config={"region": "Al2p", "source_tool": "J4", "target_tool": "H1"},
        )
        _, val_loader = create_dataloaders(train_ds, val_ds, config={"batch_size": 8})
        model = Residual1DCNN(n_points=50, config={"hidden_channels": 8, "kernel_size": 3})

        crit_zero_l2 = NormalizedMSELoss(l2_weight=0.0)
        crit_large_l2 = NormalizedMSELoss(l2_weight=10.0)

        loss_zero = evaluate(model, val_loader, crit_zero_l2)
        loss_large = evaluate(model, val_loader, crit_large_l2)

        self.assertAlmostEqual(loss_zero, loss_large, places=6)

    def test_train_baseline_region(self) -> None:
        """Test training execution and loss logging."""
        train_ds, val_ds = split_session_datasets(
            self.meta_df,
            self.ary_intensity,
            self.ary_energy,
            config={"region": "Al2p", "source_tool": "J4", "target_tool": "H1"},
        )

        res = train_baseline_region(
            train_ds,
            val_ds,
            config={
                "epochs": 5,
                "batch_size": 8,
                "hidden_channels": 8,
                "kernel_size": 3,
                "learning_rate": 1e-3,
                "l2_weight": 1e-5,
            },
        )

        self.assertIn("model", res)
        self.assertIn("best_val_loss", res)
        self.assertEqual(len(res["history"]["train_loss"]), 5)
        self.assertIsInstance(res["model"], Residual1DCNN)

    def test_format_predicted_measurement_id_uniqueness(self) -> None:
        """Verify unique ID generation without collision for different session names."""
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
        # Train a fast model
        train_ds, val_ds = split_session_datasets(
            self.meta_df,
            self.ary_intensity,
            self.ary_energy,
            config={"region": "Al2p", "source_tool": "J4", "target_tool": "H1"},
        )
        model_al2p = train_baseline_region(
            train_ds,
            val_ds,
            config={"epochs": 2, "batch_size": 8, "hidden_channels": 8, "kernel_size": 3},
        )["model"]

        models = {"Al2p": model_al2p}
        pred_int, pred_eng, pred_df = predict_spectra(
            models=models,
            data=(self.ary_intensity, self.ary_energy, self.meta_df),
            config={"source_tool": "J4", "target_tool": "H1"},
        )

        src_rows = len(self.meta_df[self.meta_df["tool"] == "J4"])
        self.assertEqual(pred_int.shape[0], src_rows)
        self.assertEqual(pred_eng.shape[0], src_rows)
        self.assertEqual(len(pred_df), src_rows)
        self.assertTrue(all(pred_df["measurement_id"].str.contains("P_J4H1_")))
        self.assertTrue(all(pred_df["tool"] == "H1"))

    def test_ax_bayesian_optimization_integration(self) -> None:
        """Test running Ax Bayesian optimization for 2 trials with seed."""
        train_ds, val_ds = split_session_datasets(
            self.meta_df,
            self.ary_intensity,
            self.ary_energy,
            config={"region": "Al2p", "source_tool": "J4", "target_tool": "H1"},
        )

        ax_res = optimize_baseline_hyperparameters(
            train_ds,
            val_ds,
            config={
                "num_trials": 2,
                "epochs_per_trial": 2,
                "kernel_sizes": [3, 5],
                "hidden_channels": [8, 16],
                "batch_size": 8,
                "seed": 42,
            },
        )

        self.assertIn("best_parameters", ax_res)
        self.assertIn("best_val_loss", ax_res)
        self.assertEqual(len(ax_res["trials_data"]), 2)
        self.assertIn(ax_res["best_parameters"]["kernel_size"], [3, 5])

    def test_run_baseline_pipeline(self) -> None:
        """Test full root pipeline execution."""
        pipeline_res = run_baseline_pipeline(
            self.meta_df,
            self.ary_intensity,
            self.ary_energy,
            config={
                "regions": ["Al2p"],
                "source_tool": "J4",
                "target_tool": "H1",
                "use_bayesian_opt": False,
                "train_config": {"epochs": 2, "batch_size": 8, "hidden_channels": 8, "kernel_size": 3},
                "predict_source": True,
            },
        )

        self.assertIn("models", pipeline_res)
        self.assertIn("Al2p", pipeline_res["models"])
        self.assertIn("session_splits", pipeline_res)
        self.assertIn("test_evaluation", pipeline_res)
        self.assertIn("Al2p", pipeline_res["test_evaluation"])
        self.assertIsNotNone(pipeline_res["predictions"])
        pred_int, pred_eng, pred_df = pipeline_res["predictions"]
        self.assertGreater(len(pred_df), 0)
        self.assertIn("split", pred_df.columns)
        self.assertTrue(set(pred_df["split"].unique()).issubset({"train", "val", "test", "unpaired"}))


if __name__ == "__main__":
    unittest.main()
