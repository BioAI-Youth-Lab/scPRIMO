from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Any, Optional

from primo.paths import DATA_DIR


@dataclass
class KFoldSplitConfig:
    """Configuration for loading a k-fold split."""

    fold: int
    k: int = 5
    split_dir: str = str(DATA_DIR / "splits")
    test_size: Optional[float] = None


def _build_split_filename(cfg: KFoldSplitConfig) -> str:
    """Construct split filename compatible with existing pkl files."""
    if cfg.test_size is None:
        return f"kfold_{cfg.k}_split_{cfg.fold}.pkl"
    return f"kfold_{cfg.k}_split_{cfg.fold}_testsize_{cfg.test_size}.pkl"


def load_kfold_split(cfg: KFoldSplitConfig) -> Dict[str, Any]:
    """
    Load a k-fold split from disk.

    This is a thin wrapper around the original utility functions, but
    makes the interface configuration-driven and dataset-agnostic.
    """
    filename = _build_split_filename(cfg)
    path = os.path.join(cfg.split_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path, "rb") as f:
        split = pickle.load(f)
    if not isinstance(split, dict):
        raise ValueError(f"Unexpected split object type: {type(split)}")
    return split



