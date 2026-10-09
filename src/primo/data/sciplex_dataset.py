from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional

import numpy as np
import anndata as ad

from .splits import KFoldSplitConfig, load_kfold_split
from primo.paths import CHECKPOINT_DIR, DATA_DIR


@dataclass
class SciplexDatasetConfig:
    """Configuration for building the sciplex dataset bundle."""

    h5ad_path: str = str(DATA_DIR / "sciplex" / "sciplex_processed.h5ad")
    split_dir: str = str(DATA_DIR / "sciplex" / "splits")
    fold: int = 3
    k: int = 5
    test_size: Optional[float] = None

    good_smile_key: str = "good_smile_chemvae"
    good_smile_value: int = 1

    cell_type_col: str = "cell_type"
    dose_col: str = "dose"
    cell_type_categories: Sequence[str] = ("A549", "K562", "MCF7")
    sciplex_perturbation_col: str = "product_name_clean"
    perturbation_key: str = "perturb_string_processed"
    trt_key: str = "ordered_all_smile_chemvae"
    embed_key: str = "ordered_all_onehot_chemvae"
    trt_chemvae_key: str = "ordered_all_trt_chemvae"

    chem_onehot_path: str = str(CHECKPOINT_DIR / "chemicalVAE" / "onehot_zinc.npy")

    control_col: str = "product_name_clean"
    control_label: str = "Vehicle"


@dataclass
class SciplexDatasetBundle:
    """Container for train / test splits and associated features."""

    adata_train: ad.AnnData
    adata_test: ad.AnnData
    covariates_embedding: np.ndarray
    chem_onehot: np.ndarray
    adata_control_ref: Optional[ad.AnnData] = None


def _build_covariates_embedding(
    adata_obj: ad.AnnData,
    cell_type_col: str,
    dose_col: str,
    categories: Sequence[str],
) -> np.ndarray:
    """Build covariate matrix [cell_type one-hot, dose]."""
    import pandas as pd

    cell_type = pd.get_dummies(adata_obj.obs[cell_type_col])[list(categories)].to_numpy()
    dose_time = adata_obj.obs[[dose_col]].to_numpy()
    return np.concatenate([cell_type, dose_time], axis=1)


def build_sciplex_dataset(cfg: SciplexDatasetConfig) -> SciplexDatasetBundle:
    """
    Load sciplex AnnData, apply filtering, and construct train/test splits.
    """
    adata_raw = ad.read_h5ad(cfg.h5ad_path)
    adata_control_ref: Optional[ad.AnnData] = None
    if cfg.control_col in adata_raw.obs.columns:
        control_mask = adata_raw.obs[cfg.control_col].astype(str) == str(cfg.control_label)
        if np.asarray(control_mask).any():
            adata_control_ref = adata_raw[control_mask, :].copy()

    adata = adata_raw

    if cfg.good_smile_key in adata.obs.columns:
        mask = np.asarray(adata.obs[cfg.good_smile_key] == cfg.good_smile_value)
        adata = adata[mask, :].copy()

    split_cfg = KFoldSplitConfig(
        fold=cfg.fold, k=cfg.k, split_dir=cfg.split_dir, test_size=cfg.test_size
    )

    split = load_kfold_split(split_cfg)

    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    adata_train = adata[train_idx, :].copy()
    adata_test = adata[test_idx, :].copy()

    chem_onehot = np.load(cfg.chem_onehot_path)

    covariates_embedding = _build_covariates_embedding(
        adata, cfg.cell_type_col, cfg.dose_col, cfg.cell_type_categories
    )

    return SciplexDatasetBundle(
        adata_train=adata_train,
        adata_test=adata_test,
        covariates_embedding=covariates_embedding,
        chem_onehot=chem_onehot,
        adata_control_ref=adata_control_ref,
    )

