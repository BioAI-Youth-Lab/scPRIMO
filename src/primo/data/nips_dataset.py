from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional

import numpy as np
import anndata as ad
import pandas as pd

from .splits import KFoldSplitConfig, load_kfold_split
from primo.paths import CHECKPOINT_DIR, DATA_DIR


@dataclass
class NipsDatasetConfig:
    h5ad_path: str = str(DATA_DIR / "nips" / "nips_processed.h5ad")
    split_dir: str = str(DATA_DIR / "nips" / "splits")
    fold: int = 3
    k: int = 3
    test_size: Optional[float] = 0.4

    cell_type_col: str = "cell_type"
    dose_col: str = "dose_uM"
    cell_type_categories: Sequence[str] = (
        "T cells CD4+",
        "T regulatory cells",
        "T cells CD8+",
        "NK cells",
        "B cells",
        "Myeloid cells",
    )

    chem_onehot_path: str = str(CHECKPOINT_DIR / "chemicalVAE" / "onehot_zinc.npy")


@dataclass
class NipsDatasetBundle:
    adata_train: ad.AnnData
    adata_test: ad.AnnData
    covariates_embedding: np.ndarray
    chem_onehot: np.ndarray


def _build_covariates_embedding(
    adata_obj: ad.AnnData,
    cell_type_col: str,
    dose_col: str,
    categories: Sequence[str],
) -> np.ndarray:
    cell_type = pd.get_dummies(adata_obj.obs[cell_type_col])[list(categories)].to_numpy()
    dose_time = adata_obj.obs[[dose_col]].to_numpy()
    return np.concatenate([cell_type, dose_time], axis=1)


def build_nips_dataset(cfg: NipsDatasetConfig) -> NipsDatasetBundle:
    adata = ad.read_h5ad(cfg.h5ad_path)
    adata.obs["indices"] = list(range(adata.X.shape[0]))

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

    return NipsDatasetBundle(
        adata_train=adata_train,
        adata_test=adata_test,
        covariates_embedding=covariates_embedding,
        chem_onehot=chem_onehot,
    )

