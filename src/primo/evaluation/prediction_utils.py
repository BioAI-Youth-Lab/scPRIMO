"""
Helper functions for generating predictions from trained models.
"""

from __future__ import annotations

import os
import pickle

from tqdm import tqdm 
import numpy as np
import pandas as pd
import torch
import anndata as ad
from anndata import AnnData
from typing import Dict, List, Optional, Tuple, Literal, Any
from scipy import sparse

from primo.conditioning.enhanced_covariates import build_unseen_enhanced_cov_with_similars
from primo.utils.perturbation_embeddings import prepare_embeddings_cinn
from primo.utils.standardize import StandardizeLoad
from primo.predict.generator import SCVIZ_CheckNet2Net
from primo.vendor.perturbnet.pytorch_scvi.scvi_generate_z import scvi_predictive_z
from primo.evaluation.deg_utils import (
    get_deg_indices_from_precomputed,
    get_trt_from_perturbation_sciplex,
)


def load_std_model(std_param_dir: str, device: torch.device) -> StandardizeLoad:
    mu_std = np.load(os.path.join(std_param_dir, "mu.npy"))
    std_std = np.load(os.path.join(std_param_dir, "std.npy"))
    return StandardizeLoad(mu_std, std_std, device)


def _compute_control_means(
    adata_train: AnnData,
    cell_type_col: str,
    control_label: str,
    perturbation_col: str
) -> Dict[str, np.ndarray]:
    """Compute mean expression of control cells for each cell type."""
    control_means = {}
    unique_cell_types = adata_train.obs[cell_type_col].unique()
    
    for cell_type in unique_cell_types:
        ctrl_mask = (
            (adata_train.obs[perturbation_col] == control_label) &
            (adata_train.obs[cell_type_col] == cell_type)
        )
        if ctrl_mask.sum() > 0:
            y_ctrl = adata_train[ctrl_mask].layers.get("counts", adata_train[ctrl_mask].X)
            if sparse.issparse(y_ctrl):
                y_ctrl = y_ctrl.toarray()
            else:
                y_ctrl = np.array(y_ctrl)
            control_means[cell_type] = y_ctrl.mean(axis=0)
    
    return control_means


def _compute_deg_indices(
    adata_test: AnnData,
    perturbations: List[str],
    cell_types: List[str],
    n_deg_genes: int,
    dataset_mode: str,
    perturbation_col: str
) -> Dict[str, np.ndarray]:
    """Pre-compute DEG indices for all relevant conditions."""
    deg_dict = {}
    
    # We use a composite key for storage: "{cell_type}_{perturbation}"
    # For Sciplex, the key is just perturbation name, but we can store it redundantly 
    # or handle logic later. To be consistent, let's store per condition.
    
    for pert in perturbations:
        for ct in cell_types:
            key = f"{ct}_{pert}"
            try:
                if dataset_mode == "sciplex":
                    trt = get_trt_from_perturbation_sciplex(adata_test, pert, perturbation_col)
                    if trt:
                        indices = get_deg_indices_from_precomputed(
                            adata_test, trt, n_deg_genes, "sciplex"
                        )
                        deg_dict[key] = indices
                else:
                    indices = get_deg_indices_from_precomputed(
                        adata_test, pert, n_deg_genes, "nips", ct
                    )
                    deg_dict[key] = indices
            except (ValueError, KeyError):
                pass
                
    return deg_dict


def _build_name_to_smiles(
    adata_test: AnnData,
    perturbation_col: str,
    perturbation_key: str,
    adata_train: Optional[AnnData] = None,
) -> Dict[str, str]:
    """Build a mapping from drug name to SMILES/perturbation key."""
    name_to_smiles = {}
    
    def _update_map(adata):
        if perturbation_col in adata.obs and perturbation_key in adata.obs:
             # Extract unique pairs
             pairs = adata.obs[[perturbation_col, perturbation_key]].drop_duplicates()
             for _, row in pairs.iterrows():
                 name_to_smiles[str(row[perturbation_col])] = str(row[perturbation_key])

    _update_map(adata_test)
    if adata_train is not None:
        _update_map(adata_train)
        
    return name_to_smiles


def generate_predictions_sciplex(
    model_cinn,
    adata_test: AnnData,
    perturbation_col: str,
    chemvae_model,
    std_model,
    scvi_model,
    covariates: np.ndarray,
    adata_train: Optional[AnnData] = None,
    device: str = "cuda:0",
    use_enhanced_cov: bool = False,
    train_fit: Optional[Dict] = None,
    similars_map: Optional[Dict] = None,
    perturbation_key: str = "perturb_string_processed",
    trt_key: str = "ordered_all_smile_chemvae",
    embed_key: str = "ordered_all_onehot_chemvae",
    trt_chemvae_key: str = "ordered_all_trt_chemvae",
    cell_type_col: str = "cell_type",
    control_label: str = "control",
) -> AnnData:
    
    model_cinn.eval()
    model_g = model_cinn.model_con
    model_g.eval()
    
    scvi_model_de = scvi_predictive_z(scvi_model)
    perturbnet_model = SCVIZ_CheckNet2Net(model_cinn, device, scvi_model_de)
    
    # Build name_to_smiles mapping for enhanced covariates
    name_to_smiles = None
    if use_enhanced_cov and similars_map:
        name_to_smiles = _build_name_to_smiles(
            adata_test, perturbation_col, perturbation_key, adata_train
        )

    perturb_with_onehot_removed = np.array(adata_test.obs[perturbation_key])
    trt_unseen_list = np.unique(perturb_with_onehot_removed)
    
    trt_pert_list = []
    for trt in trt_unseen_list:
        pert = adata_test.obs[adata_test.obs[perturbation_key] == trt][perturbation_col].unique()[0]
        trt_pert_list.append(pert)
    
    trt_treatment_list = []
    for i in range(len(trt_unseen_list)):
        trt = trt_unseen_list[i]
        idx = np.where(adata_test.uns[trt_key] == trt)[0][0]
        trt_treatment_list.append(adata_test.uns[trt_chemvae_key][idx])
    
    unseen_onehot_idx = []
    for trt in trt_unseen_list:
        onehot_idx = np.where(adata_test.uns[trt_key] == trt)[0][0]
        unseen_onehot_idx.append(onehot_idx)
    unseen_onehot = adata_test.uns[embed_key][unseen_onehot_idx]
    
    result_adatas = []
    np.random.seed(2025)
    
    for indice_trt in tqdm(range(len(trt_unseen_list)), desc="Predicting"):
        trt_type = trt_unseen_list[indice_trt]
        pert = trt_pert_list[indice_trt]
        treatment = trt_treatment_list[indice_trt]
        
        idx_trt_type = np.where(perturb_with_onehot_removed == trt_type)[0]
        if len(idx_trt_type) == 0:
            continue
        
        onehot_indice_trt = np.tile(unseen_onehot[[indice_trt]], (len(idx_trt_type), 1, 1))
        
        with torch.no_grad():
            _, _, _, embdata_torch = model_g(torch.tensor(onehot_indice_trt).float().to(device))
            embdata_np = std_model.standardize_z(embdata_torch.cpu().detach().numpy())
        
        Lsample_obs = scvi_model.get_latent_library_size(adata=adata_test, give_mean=False)
        Lsample_idx = np.random.choice(range(Lsample_obs.shape[0]), len(idx_trt_type), replace=True)
        library_trt_latent = Lsample_obs[Lsample_idx]
        
        if use_enhanced_cov and train_fit is not None:
            trt_len = len(idx_trt_type)
            unseen_trt_expanded = [pert] * trt_len
            
            dose_col = "dose"
            enhanced_cov_unseen = build_unseen_enhanced_cov_with_similars(
                adata_meta_like=adata_test[idx_trt_type],
                unseen_names=unseen_trt_expanded,
                similars_map=similars_map if similars_map else {},
                train_fit=train_fit,
                cell_type_col=cell_type_col,
                dose_col=dose_col,
                name_to_smiles=name_to_smiles,
            )
            cov_perturb = enhanced_cov_unseen
        else:
            # IMPORTANT: Ensure idx_trt_type correctly indexes into covariates
            # covariates must be aligned with adata_test
            # idx_trt_type contains indices relative to adata_test (0 to N_test-1)
            # if covariates has shape (N_test, dim), then this is correct.
            cov_perturb = covariates[idx_trt_type]

        with torch.no_grad():
            _, fake_data_all = perturbnet_model.sample_data(
                z_drug_std=embdata_np,
                covariates=cov_perturb,
                library_data=library_trt_latent,
            )
        
        # Create result subset
        subset = adata_test[idx_trt_type].copy()
        subset.layers["prediction"] = fake_data_all
        result_adatas.append(subset)
    
    # Concatenate results
    if not result_adatas:
        return AnnData()
    
    final_adata = ad.concat(result_adatas)
    
    # Add metadata
    if adata_train is not None:
        control_means = _compute_control_means(adata_train, cell_type_col, control_label, perturbation_col)
        final_adata.uns["control_means"] = control_means
    
    # Pre-compute DEGs
    # Sciplex uses perturbation name for DEGs
    unique_perts = final_adata.obs[perturbation_col].unique()
    unique_cts = final_adata.obs[cell_type_col].unique()
    deg_dict = _compute_deg_indices(adata_test, unique_perts, unique_cts, 50, "sciplex", perturbation_col)
    final_adata.uns["deg_indices"] = deg_dict
    
    return final_adata


def generate_predictions_nips(
    model_cinn,
    adata_test: AnnData,
    perturbation_col: str,
    cell_type_col: str,
    chemvae_model,
    std_model,
    scvi_model,
    covariates: np.ndarray,
    adata_train: Optional[AnnData] = None,
    device: str = "cuda:0",
    use_enhanced_cov: bool = False,
    train_fit: Optional[Dict] = None,
    similars_map: Optional[Dict] = None,
    perturbation_key: str = "SMILES",
    trt_key: str = "ordered_all_smile_chemvae",
    embed_key: str = "ordered_all_onehot_chemvae",
    dose_col: str = "dose_uM",
    control_label: str = "control",
) -> AnnData:
    
    model_cinn.eval()
    model_g = model_cinn.model_con
    model_g.eval()
    
    scvi_model_de = scvi_predictive_z(scvi_model)
    perturbnet_model = SCVIZ_CheckNet2Net(model_cinn, device, scvi_model_de)
    
    # Build name_to_smiles mapping for enhanced covariates
    name_to_smiles = None
    if use_enhanced_cov and similars_map:
        name_to_smiles = _build_name_to_smiles(
            adata_test, perturbation_col, perturbation_key, adata_train
        )

    perturb_with_onehot_removed = np.array(adata_test.obs[perturbation_key])
    trt_unseen_list = np.unique(perturb_with_onehot_removed)
    
    trt_pert_list = []
    for trt in trt_unseen_list:
        pert = adata_test.obs[adata_test.obs[perturbation_key] == trt][perturbation_col].unique()[0]
        trt_pert_list.append(pert)
    
    unseen_onehot_idx = []
    for trt in trt_unseen_list:
        onehot_idx = np.where(adata_test.uns[trt_key] == trt)[0][0]
        unseen_onehot_idx.append(onehot_idx)
    unseen_onehot = adata_test.uns[embed_key][unseen_onehot_idx]
    
    unique_cell_types = adata_test.obs[cell_type_col].unique()
    
    result_adatas = []
    np.random.seed(2025)
    
    for indice_trt in tqdm(range(len(trt_unseen_list)), desc="Predicting"):
        trt_type = trt_unseen_list[indice_trt]
        pert = trt_pert_list[indice_trt]
        
        if pert == 'DMSO':
            continue
        
        idx_trt_all = np.where(perturb_with_onehot_removed == trt_type)[0]
        
        onehot_indice_trt_single = unseen_onehot[[indice_trt]]
        with torch.no_grad():
            _, _, _, embdata_torch_single = model_g(torch.tensor(onehot_indice_trt_single).float().to(device))
            embdata_np_single = std_model.standardize_z(embdata_torch_single.cpu().detach().numpy())
        
        for c_type in unique_cell_types:
            indices_of_c_type = np.where(adata_test.obs[cell_type_col].values == c_type)[0]
            idx_trt_type = np.intersect1d(idx_trt_all, indices_of_c_type)
            
            if idx_trt_type.shape[0] > 2000:
                idx_trt_type = np.random.choice(idx_trt_type, 2000, replace=False)
            elif idx_trt_type.shape[0] <= 5:
                continue
            
            n_samples = len(idx_trt_type)
            embdata_np = np.tile(embdata_np_single, (n_samples, 1))
            
            Lsample_obs = scvi_model.get_latent_library_size(adata=adata_test, give_mean=False)
            Lsample_idx = np.random.choice(range(Lsample_obs.shape[0]), n_samples, replace=True)
            library_trt_latent = Lsample_obs[Lsample_idx]
            
            if use_enhanced_cov and train_fit is not None:
                trt_len = n_samples
                unseen_trt_expanded = [pert] * trt_len
                
                enhanced_cov_unseen = build_unseen_enhanced_cov_with_similars(
                    adata_meta_like=adata_test[idx_trt_type],
                    unseen_names=unseen_trt_expanded,
                    similars_map=similars_map if similars_map else {},
                    train_fit=train_fit,
                    cell_type_col=cell_type_col,
                    dose_col=dose_col,
                    name_to_smiles=name_to_smiles,
                )
                cov_perturb = enhanced_cov_unseen
            else:
                cov_perturb = covariates[idx_trt_type]
            
            with torch.no_grad():
                _, fake_data_all = perturbnet_model.sample_data(
                    z_drug_std=embdata_np,
                    covariates=cov_perturb,
                    library_data=library_trt_latent,
                )
            
            # Create result subset
            subset = adata_test[idx_trt_type].copy()
            subset.layers["prediction"] = fake_data_all
            result_adatas.append(subset)
    
    if not result_adatas:
        return AnnData()
        
    final_adata = ad.concat(result_adatas)
    
    # Add metadata
    if adata_train is not None:
        control_means = _compute_control_means(adata_train, cell_type_col, control_label, perturbation_col)
        final_adata.uns["control_means"] = control_means
    
    # Pre-compute DEGs
    unique_perts = final_adata.obs[perturbation_col].unique()
    unique_cts = final_adata.obs[cell_type_col].unique()
    deg_dict = _compute_deg_indices(adata_test, unique_perts, unique_cts, 50, "nips", perturbation_col)
    final_adata.uns["deg_indices"] = deg_dict

    return final_adata


def load_perturbation_model(
    model: Any,
    model_path: str,
    device: str | torch.device,
) -> Any:

    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    model_file = os.path.join(model_path, "model_params.pt")
    if not os.path.exists(model_file):
        raise ValueError(
            f"Model file not found: {model_file}. "
            "Model should be saved using model.save() which creates model_params.pt"
        )
    
    model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
    model.to(device)
    model.eval()
    
    return model
