"""Regression tests for regional weights with sliding-window spectra."""

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.cost import SpectralLoss
from models.dataset import SpectrumPairDataset, SpectrumPatchDataset, default_spectrum_collate
from models.trainer import evaluate, evaluate_sliding_window


class ZeroPrediction(nn.Module):
    """Return zero for both vectorized and per-sample prediction paths."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Preserve spectral shape while returning zero intensity."""
        return x * 0.0


class TestRegionalLossSlidingWindow(unittest.TestCase):
    """Exercise weighting through patches and both reconstruction paths."""

    def setUp(self) -> None:
        """Prepare two equal-length regional spectra with different weights."""
        x = torch.zeros(2, 1, 9)
        y = torch.tensor([1.0, 2.0]).view(2, 1, 1).expand_as(x).clone()
        self.dataset = SpectrumPairDataset(
            x, y, metadata=[{"region": "Al2p"}, {"region": "Ti2p"}]
        )
        self.criterion = SpectralLoss({"region_weights": {"Al2p": 2.0, "Ti2p": 3.0}})

    def test_patch_loss_keeps_full_target_maximum(self) -> None:
        """Forward both regional context and target scale for patch batches."""
        self.dataset.y[:, :, -1] *= 10.0
        patches = SpectrumPatchDataset(self.dataset, window_size=5, stride=2)
        loader = DataLoader(patches, batch_size=4, collate_fn=default_spectrum_collate)
        # Full-max-normalized patch errors are 0.01, 0.01, and 0.208.
        self.assertAlmostEqual(evaluate(ZeroPrediction(), loader, self.criterion), 0.19, places=6)

    def test_reconstructed_validation_uses_regional_weights(self) -> None:
        """Apply weights in vectorized and per-sample reconstruction paths."""
        for sample_by_sample in (False, True):
            with self.subTest(sample_by_sample=sample_by_sample):
                model = ZeroPrediction()
                model.use_film = sample_by_sample
                loss = evaluate_sliding_window(
                    model, self.dataset, self.criterion,
                    {"window_size": 5, "stride": 2},
                )
                self.assertAlmostEqual(loss, 2.5)
