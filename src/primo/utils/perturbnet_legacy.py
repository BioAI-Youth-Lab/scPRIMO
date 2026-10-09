"""Compatibility re-exports for utilities migrated from PerturbNet.

The release code imports the split modules directly.  This module keeps the old
``primo.utils.perturbnet_legacy`` import path available for external scripts that
were copied from the original experimental workspace.
"""

from primo.conditioning.enhanced_covariates import (
    build_train_enhanced_cov,
    build_unseen_enhanced_cov_with_similars,
    fit_train_stats_pca,
)
from primo.evaluation.legacy_metrics import (
    NormalizedRevisionRSquare,
    compute_centroid_metrics,
    compute_fid_like,
    compute_mean_z_r2,
    compute_mmd_rbf,
    fidscore,
    fidscore_scgen_extend,
    fidscore_scvi_extend,
    fidscore_vae_extend,
    knn_mixing_rate,
)
from primo.llm.similar_utils import (
    extract_cell_drug_combinations,
    extract_similars_candidates,
    load_similars_map,
    similars_entry_has_candidates,
)
from primo.utils.legacy_plots import (
    boxplot_metrics,
    contourplot_space_mapping,
    umapPlot_latent_check,
)
from primo.utils.perturbation_embeddings import (
    ConcatDatasetWithIndices,
    Seq_to_Embed_ESM,
    create_train_test_splits_by_key,
    pad_smile,
    prepare_embeddings_cinn,
    smiles_to_hot,
)
from primo.utils.splits import (
    load_data_testsize,
    load_kfold_split,
    load_kfold_split_testsize,
)
from primo.utils.standardize import Standardize, StandardizeLoad


__all__ = [
    "ConcatDatasetWithIndices",
    "NormalizedRevisionRSquare",
    "Seq_to_Embed_ESM",
    "Standardize",
    "StandardizeLoad",
    "boxplot_metrics",
    "build_train_enhanced_cov",
    "build_unseen_enhanced_cov_with_similars",
    "compute_centroid_metrics",
    "compute_fid_like",
    "compute_mean_z_r2",
    "compute_mmd_rbf",
    "contourplot_space_mapping",
    "create_train_test_splits_by_key",
    "extract_cell_drug_combinations",
    "extract_similars_candidates",
    "fidscore",
    "fidscore_scgen_extend",
    "fidscore_scvi_extend",
    "fidscore_vae_extend",
    "fit_train_stats_pca",
    "knn_mixing_rate",
    "load_data_testsize",
    "load_kfold_split",
    "load_kfold_split_testsize",
    "load_similars_map",
    "pad_smile",
    "prepare_embeddings_cinn",
    "similars_entry_has_candidates",
    "smiles_to_hot",
    "umapPlot_latent_check",
]
