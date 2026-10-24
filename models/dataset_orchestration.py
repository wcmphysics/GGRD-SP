"""Dataset orchestration module for GGRD-SP.

Re-exports dataset orchestration routines consolidated into models.dataset for
backward compatibility.
"""

from __future__ import annotations

from models.dataset import (
    create_regional_datasets,
    create_unified_dataset,
)

__all__ = [
    "create_regional_datasets",
    "create_unified_dataset",
]
