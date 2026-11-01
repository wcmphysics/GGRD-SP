"""Regression tests for regional weights in training and validation."""

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.cost import SpectralLoss
from models.dataset import SpectrumPairDataset, default_spectrum_collate
from models.trainer import _calculate_batch_loss, evaluate, train_one_epoch


class ConstantPrediction(nn.Module):
    """Expose one trainable offset for checking weighted loss gradients."""

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the same trainable intensity at every point."""
        return x * 0.0 + self.offset


class TestRegionalLossTraining(unittest.TestCase):
    """Verify configured regional factors reach the actual loss and updates."""

    def setUp(self) -> None:
        """Prepare a mixed-region batch with analytically known errors."""
        self.x = torch.zeros(4, 1, 5)
        self.y = torch.arange(1.0, 5.0).view(4, 1, 1).expand_as(self.x)
        self.metadata = [{"region": region} for region in ("Al2p", "Ti2p", "Al2p", "Ti2p")]
        self.dataset = SpectrumPairDataset(self.x, self.y, metadata=self.metadata)
        self.criterion = SpectralLoss({
            "normalized_by_target_max": False,
            "region_weights": {"Al2p": 2.0, "Ti2p": 3.0},
        })

    def test_weighted_validation_is_batch_size_invariant(self) -> None:
        """Apply each sample's regional factor across partial batches."""
        for batch_size in (1, 3, 4):
            with self.subTest(batch_size=batch_size):
                loader = DataLoader(
                    self.dataset, batch_size=batch_size,
                    collate_fn=default_spectrum_collate,
                )
                self.assertAlmostEqual(
                    evaluate(ConstantPrediction(), loader, self.criterion), 20.0, places=5
                )

    def test_regional_weights_change_optimizer_update(self) -> None:
        """Ensure weighted gradients reach the optimizer on mixed regions."""
        model = ConstantPrediction()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = DataLoader(self.dataset, batch_size=4, collate_fn=default_spectrum_collate)
        self.assertAlmostEqual(train_one_epoch(model, loader, optimizer, self.criterion), 20.0)
        self.assertAlmostEqual(model.offset.item(), 0.13, places=6)

    def test_regularization_is_added_once(self) -> None:
        """Keep parameter regularization independent of regional factors."""
        model = nn.Linear(1, 1, bias=False)
        nn.init.constant_(model.weight, 2.0)
        self.criterion.l2_weight = 0.1
        loss = _calculate_batch_loss(
            self.criterion, self.x, self.y,
            {"model": model, "meta": self.metadata},
        )
        self.assertAlmostEqual(loss.item(), 20.4, places=5)

    def test_missing_region_metadata_rejected(self) -> None:
        """Reject weighted training that cannot identify sample regions."""
        for metadata in (
            None, [{}] * 4, [{"region": float("nan")}] * 4,
            [{"region": ""}] * 4, [{"region": "  "}] * 4,
        ):
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(ValueError, "Regional loss weighting"):
                    _calculate_batch_loss(self.criterion, self.x, self.y, {"meta": metadata})
