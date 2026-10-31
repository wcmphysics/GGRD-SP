"""Unit tests for the sliding window model and pipeline in models/."""

from __future__ import annotations

import unittest
import numpy as np
import torch

import pandas as pd
from torch.utils.data import Dataset, Subset

from models.bayesian_opt import optimize_model_hyperparameters
from models.cost import SpectralLoss
from models.dataset import SpectrumPairDataset, SpectrumPatchDataset
from models.inference import (
    _predict_sliding_window_region,
    predict_sliding_window_spectra,
    predict_sliding_window_spectrum,
    predict_sliding_window_tensor,
)
from models.resnet import ResNet1D
from models.trainer import evaluate_sliding_window, train_model_region

_DEFAULT_RESNET_FIXED_CONFIG = {
    "dropout": 0.0,
    "use_batch_norm": True,
    "padding_mode": "reflect",
    "use_film": False,
}

_DEFAULT_RESNET_TEST_CONFIG = {
    "resnet_channel": 8,
    "kernel_size": 3,
    **_DEFAULT_RESNET_FIXED_CONFIG,
}


class TestSlidingWindowPipeline(unittest.TestCase):
    """Test suite for sliding window dataset, training, inference, and
    BoTorch.
    """

    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(42)
        torch.manual_seed(42)
        n_points = 60
        x_raw = np.random.uniform(10.0, 50.0, size=(12, n_points)).astype(np.float32)
        y_raw = (x_raw * 1.05 + 1.0).astype(np.float32)

        cls.train_ds = SpectrumPairDataset(x_raw[:8], y_raw[:8])
        cls.val_ds = SpectrumPairDataset(x_raw[8:], y_raw[8:])

    def test_spectrum_patch_dataset_and_reconstruction(self) -> None:
        """Verify patch dataset slicing, window prediction, and reconstruction.
        """
        patch_ds = SpectrumPatchDataset(self.train_ds, window_size=15, stride=5)
        self.assertGreater(len(patch_ds), len(self.train_ds))
        self.assertEqual(patch_ds[0]["x"].shape, (1, 15))

        model = ResNet1D(15, config=_DEFAULT_RESNET_TEST_CONFIG)
        spectrum = self.train_ds[0]["x"].cpu().numpy().squeeze()
        pred = predict_sliding_window_spectrum(
            model, spectrum, config={"window_size": 15, "stride": 5, "batch_size": 16}
        )
        self.assertEqual(len(pred), len(spectrum))

        loss = evaluate_sliding_window(
            model=model,
            full_val_dataset=self.val_ds,
            criterion=SpectralLoss({"l2_weight": 0.0}),
            config={"window_size": 15, "stride": 5},
        )
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)

    def test_train_sliding_window_region(self) -> None:
        """Verify train_model_region training execution with sliding window."""
        train_res = train_model_region(
            self.train_ds,
            self.val_ds,
            config={
                **_DEFAULT_RESNET_TEST_CONFIG,
                "model_type": "resnet",
                "use_sliding_window": True,
                "window_size": 15,
                "stride": 5,
                "epochs": 2,
                "batch_size": 8,
                "verbose": False,
            },
        )
        self.assertIn("model", train_res)
        self.assertIn("best_val_loss", train_res)
        self.assertEqual(len(train_res["history"]["train_loss"]), 2)

    def test_optimize_sliding_window_hyperparameters(self) -> None:
        """Verify Ax Bayesian optimization for sliding window model."""
        ax_res = optimize_model_hyperparameters(
            self.train_ds,
            self.val_ds,
            config={
                **_DEFAULT_RESNET_FIXED_CONFIG,
                "model_type": "resnet",
                "use_sliding_window": True,
                "num_trials": 2,
                "epochs_per_trial": 2,
                "window_size_choices": [11, 15],
                "kernel_sizes": [3],
                "resnet_channels": [8],
                "seed": 42,
            },
        )
        self.assertIn("best_parameters", ax_res)
        self.assertEqual(len(ax_res["trials_data"]), 2)

    def test_sliding_window_parameter_validations(self) -> None:
        """Verify validation errors for invalid sliding window parameters."""
        # Table-driven training parameter validation
        invalid_train_cfgs = [
            {"batch_size": [16]},                          # list instead of int
            {"window_size": [15]},                         # list instead of int
            {"window_size_choice": "15"},                  # invalid type
            {"window_size": 5, "kernel_size": 7},          # kernel > window
        ]
        for cfg in invalid_train_cfgs:
            with self.subTest(train_cfg=cfg):
                with self.assertRaises(ValueError):
                    train_model_region(
                        self.train_ds,
                        self.val_ds,
                        config={"model_type": "resnet", **cfg},
                    )

        # Table-driven optimization parameter validation
        invalid_bo_cfgs = [
            {"kernel_sizes": [4]},                         # even kernel
            {"learning_rate_bounds": (0.01, 0.001)},       # inverted bounds
            {"l2_bounds": (-1e-4, 1e-2)},                  # negative lower bound
            {"window_size_choices": 15},                   # scalar instead of list
            {"window_size_choices": None},                 # None choices
            {"window_size_choices": [5], "kernel_sizes": [7]}, # kernel > window
        ]
        for cfg in invalid_bo_cfgs:
            with self.subTest(bo_cfg=cfg):
                expected_err = (
                    TypeError if cfg.get("window_size_choices") == 15 else ValueError
                )
                with self.assertRaises(expected_err):
                    optimize_model_hyperparameters(
                        self.train_ds,
                        self.val_ds,
                        config={"model_type": "resnet", "use_sliding_window": True, **cfg},
                    )

    def test_concatenate_heterogeneous_spectrum_lengths(self) -> None:
        """Verify concatenate merges patch datasets of different full lengths."""
        ds1 = SpectrumPairDataset(torch.randn(4, 80), torch.randn(4, 80))
        p1 = SpectrumPatchDataset(ds1, window_size=12, stride=4)
        ds2 = SpectrumPairDataset(torch.randn(6, 120), torch.randn(6, 120))
        p2 = SpectrumPatchDataset(ds2, window_size=12, stride=4)

        merged = SpectrumPatchDataset.concatenate([p1, p2])
        self.assertEqual(len(merged), len(p1) + len(p2))
        self.assertEqual(merged[0]["x"].shape, (1, 12))
        self.assertEqual(merged[len(p1)]["x"].shape, (1, 12))
        self.assertEqual(merged.x.shape, (len(merged), 1, 12))

    def test_evaluate_sliding_window_vectorized_and_fallbacks(self) -> None:
        """Verify vectorized evaluate_sliding_window numerical parity
        and fallbacks.
        """
        model = ResNet1D(15, config=_DEFAULT_RESNET_TEST_CONFIG)
        crit = SpectralLoss({"w_mse": 1.0, "w_shape": 0.5, "w_gradient": 0.5, "l2_weight": 0.0})

        # 1. Numerical parity against sample-by-sample
        eval_cfg = {"window_size": 15, "stride": 5, "batch_size": 4}
        loss_vec = evaluate_sliding_window(
            model=model, full_val_dataset=self.val_ds, criterion=crit, config=eval_cfg
        )

        # Compute sample-by-sample manually
        total_sample = 0.0
        for i in range(len(self.val_ds)):
            item = self.val_ds[i]
            rec = predict_sliding_window_spectrum(
                model,
                item["x"].cpu().numpy().squeeze(),
                config={"window_size": 15, "stride": 5, "clamp_non_negative": True},
            )
            yp = torch.from_numpy(rec).float().view(1, 1, -1)
            total_sample += crit(yp, item["y"].unsqueeze(0), model=None).item()
        expected_loss = total_sample / len(self.val_ds)
        self.assertAlmostEqual(loss_vec, expected_loss, places=6)

        # 2. Empty dataset error
        empty_ds = SpectrumPairDataset(np.zeros((0, 60)), np.zeros((0, 60)))
        with self.assertRaises(ValueError):
            evaluate_sliding_window(model, empty_ds, crit, eval_cfg)

        # 3. Invalid stride > window_size
        with self.assertRaises(ValueError):
            evaluate_sliding_window(model, self.val_ds, crit, {"window_size": 10, "stride": 12})

        # 4. Window size > sequence length
        with self.assertRaises(ValueError):
            evaluate_sliding_window(model, self.val_ds, crit, {"window_size": 100, "stride": 5})

        # 5. Heterogeneous dataset triggers safe fallback
        class HeterogeneousDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 2

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                length = 60 if idx == 0 else 80
                return {
                    "x": torch.ones(1, length),
                    "y": torch.ones(1, length),
                }

        het_ds = HeterogeneousDataset()
        loss_het = evaluate_sliding_window(model, het_ds, crit, {"window_size": 15, "stride": 5})
        self.assertIsInstance(loss_het, float)
        self.assertGreater(loss_het, 0.0)

    def test_evaluate_sliding_window_robust_inputs(self) -> None:
        """Verify evaluate_sliding_window gracefully handles 1D, float64,
        NumPy, and Subset datasets.
        """
        model = ResNet1D(15, config=_DEFAULT_RESNET_TEST_CONFIG)
        crit = SpectralLoss()
        eval_cfg = {"window_size": 15, "stride": 5}

        # 1. Float64 inputs
        class F64Dataset(Dataset):
            def __len__(self) -> int:
                return 2

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                return {
                    "x": torch.ones(1, 60, dtype=torch.float64),
                    "y": torch.ones(1, 60, dtype=torch.float64),
                }

        loss_f64 = evaluate_sliding_window(model, F64Dataset(), crit, eval_cfg)
        self.assertIsInstance(loss_f64, float)

        # 2. NumPy ndarray inputs
        class NumpyDataset(Dataset):
            def __len__(self) -> int:
                return 2

            def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
                return {
                    "x": np.ones((1, 60), dtype=np.float32),
                    "y": np.ones((1, 60), dtype=np.float32),
                }

        loss_np = evaluate_sliding_window(model, NumpyDataset(), crit, eval_cfg)
        self.assertAlmostEqual(loss_f64, loss_np, places=5)

        # 3. 1D tensor inputs
        class OneDDataset(Dataset):
            def __len__(self) -> int:
                return 2

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                return {"x": torch.ones(60), "y": torch.ones(60)}

        loss_1d = evaluate_sliding_window(model, OneDDataset(), crit, eval_cfg)
        self.assertAlmostEqual(loss_np, loss_1d, places=5)

        # 4. PyTorch Subset wrapper
        sub = Subset(self.val_ds, [0, 1])
        loss_sub = evaluate_sliding_window(model, sub, crit, eval_cfg)
        self.assertIsInstance(loss_sub, float)

    def test_predict_sliding_window_spectra_and_region(self) -> None:
        """Verify predict_sliding_window_spectra and
        _predict_sliding_window_region numerical parity with
        sample-by-sample predict_sliding_window_spectrum.
        """
        model = ResNet1D(15, config=_DEFAULT_RESNET_TEST_CONFIG)
        rx = np.random.RandomState(42).randn(4, 60).astype(np.float32)
        cfg = {"window_size": 15, "stride": 5, "clamp_non_negative": True}

        # 1. Vectorized spectra prediction parity
        vec_pred = predict_sliding_window_spectra(model, rx, config=cfg)
        seq_pred = np.vstack(
            [predict_sliding_window_spectrum(model, x, config=cfg) for x in rx]
        )
        np.testing.assert_allclose(vec_pred, seq_pred, rtol=1e-5, atol=1e-6)

        # 2. Region-level sliding window prediction parity
        group = pd.DataFrame(
            {"region": ["Ti2p"] * len(rx), "region_index": [0] * len(rx)}
        )
        orig_indices = np.arange(len(rx))
        reg_pred = _predict_sliding_window_region(
            model_entry=model,
            region_x=rx,
            orig_indices=orig_indices,
            group=group,
            device=torch.device("cpu"),
            cfg=cfg,
        )
        np.testing.assert_allclose(reg_pred, seq_pred, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

