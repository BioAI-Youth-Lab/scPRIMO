#!/usr/bin/env python3
"""
Recovery wrapper for the old-flow cINN module used by SciPlex alignment work.

The recovered NIPS old-flow implementation is dataset-agnostic: once the
covariate layout, perturbation key, and cache artifacts are prepared
appropriately, the same module can be reused for SciPlex training/export.
This wrapper gives SciPlex recovery scripts a stable import target without
touching `perturbnet/cinn/flow_update.py`.
"""

from __future__ import annotations

from primo.models.grca_flow import (  # noqa: E402,F401
    ConcatDatasetWithIndices,
    ConditionalFlatCouplingFlow,
    Net2NetFlow_TFVAE_Covariate_Flow,
)

__all__ = [
    "ConcatDatasetWithIndices",
    "ConditionalFlatCouplingFlow",
    "Net2NetFlow_TFVAE_Covariate_Flow",
]
