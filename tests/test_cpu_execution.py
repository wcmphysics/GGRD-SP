"""Regression tests for CPU-only training and prediction."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.dataset import SpectrumPairDataset
from models.inference import predict_spectra
from models.trainer import train_model_region


class TestCPUExecution(unittest.TestCase):
    """Ensure device selection never automatically enables GPU execution."""

    def setUp(self) -> None:
        """Prepare a small model and one source spectrum."""
        self.model = nn.Conv1d(1, 1, 1)
        self.intensity = np.ones((1, 5), dtype=np.float32)
        self.energy = np.arange(5, dtype=np.float32).reshape(1, -1)
        self.dataset = SpectrumPairDataset(self.intensity, self.intensity)
        self.metadata = pd.DataFrame([{
            "spectrum_index": 0, "measurement_id": "source_01",
            "tool": "J4", "die": 0, "region": "Al2p", "split": "test",
        }])

    def test_defaults_remain_cpu_when_cuda_is_available(self) -> None:
        """Train and predict on CPU even when CUDA availability is true."""
        with patch("torch.cuda.is_available", return_value=True):
            result = train_model_region(
                self.dataset, self.dataset,
                config={"model": self.model, "epochs": 1, "verbose": False},
            )
            predicted, _, _ = predict_spectra(
                {"Al2p": result["model"]},
                (self.intensity, self.energy, self.metadata),
            )
        self.assertEqual(next(result["model"].parameters()).device.type, "cpu")
        np.testing.assert_equal(predicted.shape, self.intensity.shape)

    def test_explicit_non_cpu_devices_rejected(self) -> None:
        """Reject GPU devices before moving models or executing tensors."""
        for device in ("cuda", "mps"):
            with self.subTest(device=device):
                with self.assertRaisesRegex(ValueError, "expected a CPU device"):
                    train_model_region(
                        self.dataset, self.dataset,
                        config={"model": self.model, "device": device},
                    )
                with self.assertRaisesRegex(ValueError, "expected a CPU device"):
                    predict_spectra(
                        {"Al2p": self.model},
                        (self.intensity, self.energy, self.metadata),
                        config={"device": device},
                    )
